"""Persistent tracker for 劇情 (story) completion.

Design notes
------------

The game's story hub hosts four sub-sections:

    * 主線劇情 (main): grid-based episode push, many chapters each
    * 短篇劇情 (short): independent one-shot cutscenes, "New"/"完成" badged
    * 支線劇情 (side):  independent one-shot cutscenes, "New"/"完成" badged
    * 重播 (replay): nothing to earn, ignored

For *short* and *side* the game already maintains visible state ("New"
badge = unplayed-and-has-reward; "完成" label = fully done).  We could
rely entirely on that — but OCR can mis-read the badges on a noisy
frame, and a run that aborts mid-cutscene should NOT get re-credited
with a story it hasn't actually finished.  So we keep our own ledger:

    data/story_progress.json:
    {
      "short": { "completed": ["<hash_of_title>", ...] },
      "side":  { "completed": [...] },
      "main":  { "episodes": { "1": ["1-1", "1-2"], ... } }
    }

For short/side we hash the Chinese title (case-sensitive, whitespace-
normalised) rather than using its position — the list reorders as new
content ships, but a story's title is stable.  ``mark_done(category,
title)`` is idempotent and writes atomically.

Main-story progress is stubbed for now (we store an episodes dict but
the pusher skill hasn't been ported yet) — leaving the shape in place
means the future grid-push work doesn't have to migrate the file.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


_STORY_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "story_progress.json"


# ── Helpers ────────────────────────────────────────────────────────────

def story_key(title: str) -> str:
    """Stable ID for a short/side story title.

    We hash rather than storing the raw title because OCR can produce
    slightly different unicode normalisations and the file would bloat
    with near-duplicates.  The 12-hex-char prefix is unique enough for
    hundreds of titles — Serenade Promenade currently ships ~20 short
    stories total, so collision risk is negligible.
    """
    norm = "".join(title.split())  # strip all whitespace, keep CJK glyphs as-is
    if not norm:
        return ""
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


# ── Data types ────────────────────────────────────────────────────────

@dataclass
class CategoryProgress:
    completed: Set[str] = field(default_factory=set)

    def is_done(self, title: str) -> bool:
        key = story_key(title)
        return bool(key) and key in self.completed

    def mark(self, title: str) -> bool:
        key = story_key(title)
        if not key or key in self.completed:
            return False
        self.completed.add(key)
        return True


@dataclass
class MainProgress:
    """Placeholder for main-story progress (episode→stage list).

    Not wired to a pusher yet; kept so future commits don't migrate the
    file.  Field name mirrors reference's ``main_story_available_episodes``.
    """
    episodes: Dict[str, List[str]] = field(default_factory=dict)


# ── Store ─────────────────────────────────────────────────────────────

class StoryProgressStore:
    """Single-file JSON ledger for all four story sub-sections.

    Process-safety mirrors :class:`event_progress.EventProgressStore`:
    re-entrant lock + tmp-then-replace writes.  Unknown categories read
    as empty — callers can be added without a migration step.
    """

    SHORT = "short"
    SIDE = "side"
    MAIN = "main"
    REPLAY = "replay"  # present for completeness; never written

    def __init__(self, progress_path: Path = _STORY_FILE):
        self._path = progress_path
        self._lock = threading.RLock()
        self._short = CategoryProgress()
        self._side = CategoryProgress()
        self._main = MainProgress()
        self._load()

    # -- io -----------------------------------------------------------

    def _load(self) -> None:
        with self._lock:
            if not self._path.exists():
                return
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                return
            short = raw.get(self.SHORT) or {}
            side = raw.get(self.SIDE) or {}
            main = raw.get(self.MAIN) or {}
            self._short.completed = set(short.get("completed") or [])
            self._side.completed = set(side.get("completed") or [])
            self._main.episodes = {
                str(k): list(v or []) for k, v in (main.get("episodes") or {}).items()
            }

    def save(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".json.tmp")
            data = {
                self.SHORT: {"completed": sorted(self._short.completed)},
                self.SIDE: {"completed": sorted(self._side.completed)},
                self.MAIN: {"episodes": {k: list(v) for k, v in self._main.episodes.items()}},
            }
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._path)

    # -- short/side ---------------------------------------------------

    def _cat(self, category: str) -> CategoryProgress:
        if category == self.SHORT:
            return self._short
        if category == self.SIDE:
            return self._side
        raise ValueError(f"unknown story category: {category!r}")

    def is_done(self, category: str, title: str) -> bool:
        with self._lock:
            return self._cat(category).is_done(title)

    def mark_done(self, category: str, title: str) -> bool:
        """Record a story as completed; returns True if newly added.

        Persists to disk immediately so a crash in the post-cutscene
        dismiss loop can't cost us the credit.
        """
        with self._lock:
            changed = self._cat(category).mark(title)
            if changed:
                self.save()
            return changed

    def list_completed(self, category: str) -> List[str]:
        with self._lock:
            return sorted(self._cat(category).completed)

    # -- main ---------------------------------------------------------

    def main_episode_cleared_stages(self, episode: str) -> List[str]:
        with self._lock:
            return list(self._main.episodes.get(str(episode), []))

    def mark_main_stage_cleared(self, episode: str, stage: str) -> bool:
        with self._lock:
            key = str(episode)
            lst = self._main.episodes.setdefault(key, [])
            if stage in lst:
                return False
            lst.append(stage)
            self.save()
            return True


# ── Singleton ─────────────────────────────────────────────────────────

_singleton: Optional[StoryProgressStore] = None


def get_store() -> StoryProgressStore:
    global _singleton
    if _singleton is None:
        # Read _STORY_FILE at call-time so tests can redirect the
        # path via `sp._STORY_FILE = tmp` + reset_singleton().  Using
        # the default arg would capture the original module path at
        # `def` time and ignore the override.
        _singleton = StoryProgressStore(progress_path=_STORY_FILE)
    return _singleton


def reset_singleton() -> None:
    """For tests.  Forces the next :func:`get_store` to re-read disk."""
    global _singleton
    _singleton = None
