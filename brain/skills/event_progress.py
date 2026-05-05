"""Per-event progress tracking for the EventActivity skill.

Blue Archive events expose three independent phases — Story, Quest
(mission) and Challenge — each of which is a list of numbered nodes
(e.g. story 1-9, quest 1-12, challenge 1-4).  The user's expectation
is "finish the whole story (1..N) before moving on to quest".

reference (pur1fying/blue_archive_auto_script) re-discovers progress every
run by opening each node's info panel and reading the in-game SSS
badge.  That is robust but slow, and if the bot gets interrupted mid-
node there is no persisted knowledge of "which ones we already beat".
We want a self-developed variant that:

1. Keeps a **persistent, per-event** record of completed nodes, so a
   restart picks up exactly where it left off.
2. Stores **per-event metadata** (total_story, total_mission, …) so
   the skill knows when a phase is done without extra probing.
3. Exposes a tiny, testable API (`mark_done`, `next_node`,
   `phase_done`) that the skill can call at the points where it
   advances its internal counters — no big refactor of the existing
   1800-line skill required.

File layout:

    data/event_metadata.json   — ships with the repo, listing phase
                                 sizes + phase order per known event
                                 (human-curated from reference's JSON +
                                 verified against real screenshots).

    data/event_progress.json   — written by the skill each time a
                                 node clears.  Keyed by event id
                                 (e.g. "serenade_promenade").

The progress file uses a sorted list of integers per phase so it is
diff-friendly and trivially inspectable.
"""
from __future__ import annotations

import datetime
import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_METADATA_FILE = _DATA_DIR / "event_metadata.json"
_PROGRESS_FILE = _DATA_DIR / "event_progress.json"


# ── Phase names ────────────────────────────────────────────────────────

PHASES = ("story", "mission", "challenge")


# ── Metadata ───────────────────────────────────────────────────────────


@dataclass
class EventMetadata:
    """Static per-event configuration.

    ``phase_order`` defaults to ``("story", "mission", "challenge")`` —
    matching reference's hard-coded order — but is kept configurable in case
    a specific event wants a different flow (e.g. story-only).
    """

    event_id: str
    display_name: str = ""
    total_story: int = 0
    total_mission: int = 0
    total_challenge: int = 0
    phase_order: List[str] = field(default_factory=lambda: list(PHASES))

    def total_for(self, phase: str) -> int:
        return {
            "story": self.total_story,
            "mission": self.total_mission,
            "challenge": self.total_challenge,
        }.get(phase, 0)


# ── Per-event progress ─────────────────────────────────────────────────


@dataclass
class PhaseProgress:
    """Which numbered nodes (1-indexed) of a phase are done."""

    completed: List[int] = field(default_factory=list)

    def add(self, node: int) -> None:
        if node < 1:
            return
        if node in self.completed:
            return
        self.completed.append(node)
        self.completed.sort()

    def done(self, total: int) -> bool:
        if total <= 0:
            return True
        return len(self.completed) >= total and all(
            i in self.completed for i in range(1, total + 1)
        )

    def next_node(self, total: int) -> Optional[int]:
        """First unfinished node in [1, total], or None if all done."""
        if total <= 0:
            return None
        for i in range(1, total + 1):
            if i not in self.completed:
                return i
        return None


@dataclass
class EventProgress:
    event_id: str
    story: PhaseProgress = field(default_factory=PhaseProgress)
    mission: PhaseProgress = field(default_factory=PhaseProgress)
    challenge: PhaseProgress = field(default_factory=PhaseProgress)
    # Free-form string: one of 'story', 'mission', 'challenge' or ''.
    # We persist the last active phase so a restart can jump back in.
    last_phase: str = ""
    last_updated: str = ""

    def phase(self, name: str) -> PhaseProgress:
        return {
            "story": self.story,
            "mission": self.mission,
            "challenge": self.challenge,
        }[name]


# ── Store ──────────────────────────────────────────────────────────────


class EventProgressStore:
    """Single-file JSON store keyed by ``event_id``.

    Designed to be process-safe for our use case: the skill runs in a
    single thread, and backend restarts serialise file I/O anyway.  A
    re-entrant lock makes it safe to call ``mark_done`` and ``save``
    from nested helpers without deadlocking.

    Also persists a top-level ``current_event_id`` memo (the event id
    last set via :meth:`set_current_event`), which the skill uses as a
    fallback when it starts already on the event page and so never got
    a chance to classify the lobby banner.
    """

    def __init__(
        self,
        metadata_path: Path = _METADATA_FILE,
        progress_path: Path = _PROGRESS_FILE,
    ):
        self._metadata_path = metadata_path
        self._progress_path = progress_path
        self._lock = threading.RLock()
        self._metadata: Dict[str, EventMetadata] = {}
        self._progress: Dict[str, EventProgress] = {}
        self._current_event_id: str = ""
        self._load()

    # -- io ---------------------------------------------------------------

    def _load(self) -> None:
        with self._lock:
            self._metadata = self._load_metadata()
            self._progress = self._load_progress()
            self._current_event_id = self._load_current_event_id()

    def _load_current_event_id(self) -> str:
        if not self._progress_path.exists():
            return ""
        try:
            raw = json.loads(self._progress_path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return str(raw.get("current_event_id", "") or "")

    def _load_metadata(self) -> Dict[str, EventMetadata]:
        if not self._metadata_path.exists():
            return {}
        try:
            raw = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        out: Dict[str, EventMetadata] = {}
        for eid, payload in (raw.get("events") or {}).items():
            out[eid] = EventMetadata(
                event_id=eid,
                display_name=payload.get("display_name", ""),
                total_story=int(payload.get("total_story", 0) or 0),
                total_mission=int(payload.get("total_mission", 0) or 0),
                total_challenge=int(payload.get("total_challenge", 0) or 0),
                phase_order=list(payload.get("phase_order", PHASES)),
            )
        return out

    def _load_progress(self) -> Dict[str, EventProgress]:
        if not self._progress_path.exists():
            return {}
        try:
            raw = json.loads(self._progress_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        out: Dict[str, EventProgress] = {}
        for eid, payload in (raw.get("events") or {}).items():
            out[eid] = EventProgress(
                event_id=eid,
                story=PhaseProgress(
                    completed=sorted({
                        int(x) for x in payload.get("story", {}).get("completed", [])
                        if isinstance(x, (int, float))
                    })
                ),
                mission=PhaseProgress(
                    completed=sorted({
                        int(x) for x in payload.get("mission", {}).get("completed", [])
                        if isinstance(x, (int, float))
                    })
                ),
                challenge=PhaseProgress(
                    completed=sorted({
                        int(x) for x in payload.get("challenge", {}).get("completed", [])
                        if isinstance(x, (int, float))
                    })
                ),
                last_phase=payload.get("last_phase", "") or "",
                last_updated=payload.get("last_updated", "") or "",
            )
        return out

    def save(self) -> None:
        with self._lock:
            self._progress_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._progress_path.with_suffix(".json.tmp")
            data = {
                "current_event_id": self._current_event_id,
                "events": {
                    eid: {
                        "story": {"completed": list(p.story.completed)},
                        "mission": {"completed": list(p.mission.completed)},
                        "challenge": {"completed": list(p.challenge.completed)},
                        "last_phase": p.last_phase,
                        "last_updated": p.last_updated,
                    } for eid, p in self._progress.items()
                },
            }
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self._progress_path)

    # -- current-event memo ---------------------------------------------
    #
    # The skill sets this the moment the lobby banner matcher identifies
    # the running event.  On a subsequent run where the skill starts
    # directly on the event page (e.g. chained from another skill that
    # opened the event, or user-triggered from the dashboard with the
    # event page already open), the memo provides a reliable fallback
    # so the progress lookup still scopes to the right event id.  The
    # memo survives the banner-template-matcher failing because we
    # deliberately persist only ids that came from a successful match.

    def get_current_event_id(self) -> str:
        with self._lock:
            return self._current_event_id

    def set_current_event_id(self, event_id: str) -> None:
        if not event_id:
            return
        with self._lock:
            if event_id == self._current_event_id:
                return
            self._current_event_id = event_id
            self.save()

    # -- getters ---------------------------------------------------------

    def metadata(self, event_id: str) -> EventMetadata:
        """Return metadata for ``event_id`` or a sensible default.

        For events auto-detected via period text (event_id starts with
        ``auto_``), use BA-typical totals (story=12, mission=12,
        challenge=4) so star-resync can declare a phase done once all
        nodes are marked, instead of looping forever waiting for an
        unreachable threshold.  Other unknown events still get zeros
        (vacuous done → graceful exit).
        """
        with self._lock:
            if event_id in self._metadata:
                return self._metadata[event_id]
            if event_id.startswith("auto_"):
                return EventMetadata(
                    event_id=event_id,
                    display_name=f"Auto-detected ({event_id})",
                    total_story=12,
                    total_mission=12,
                    total_challenge=4,
                )
            return EventMetadata(event_id=event_id)

    def progress(self, event_id: str) -> EventProgress:
        """Return (and lazily-create) the progress entry for ``event_id``."""
        with self._lock:
            if event_id not in self._progress:
                self._progress[event_id] = EventProgress(event_id=event_id)
            return self._progress[event_id]

    # -- writes ----------------------------------------------------------

    def mark_done(self, event_id: str, phase: str, node: int) -> None:
        """Record that ``node`` of ``phase`` finished for ``event_id``.

        Idempotent — marking the same node twice is a no-op.  Writes
        to disk each time so a crash between nodes still keeps state.
        """
        if phase not in PHASES:
            return
        with self._lock:
            prog = self.progress(event_id)
            prog.phase(phase).add(node)
            prog.last_phase = phase
            prog.last_updated = datetime.datetime.utcnow().isoformat(
                timespec="seconds"
            )
            self.save()

    def reset(self, event_id: str) -> None:
        """Wipe all progress for an event (e.g. event rotated out)."""
        with self._lock:
            self._progress.pop(event_id, None)
            if self._current_event_id == event_id:
                self._current_event_id = ""
            self.save()

    # -- convenience (keeps the skill terse) -----------------------------

    def next_node(self, event_id: str, phase: str) -> Optional[int]:
        meta = self.metadata(event_id)
        prog = self.progress(event_id)
        return prog.phase(phase).next_node(meta.total_for(phase))

    def phase_done(self, event_id: str, phase: str) -> bool:
        meta = self.metadata(event_id)
        prog = self.progress(event_id)
        return prog.phase(phase).done(meta.total_for(phase))

    def phase_order(self, event_id: str) -> List[str]:
        return list(self.metadata(event_id).phase_order)

    def summary(self, event_id: str) -> Dict[str, str]:
        """Human-readable one-liner per phase for logs."""
        meta = self.metadata(event_id)
        prog = self.progress(event_id)
        return {
            phase: f"{len(prog.phase(phase).completed)}/{meta.total_for(phase)}"
            for phase in PHASES
        }


# ── Module-level singleton -------------------------------------------------

_singleton: Optional[EventProgressStore] = None
_singleton_lock = threading.Lock()


def get_store() -> EventProgressStore:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = EventProgressStore()
        return _singleton


def reset_singleton() -> None:
    """Testing hook — force a re-read of the JSON files on next access."""
    global _singleton
    with _singleton_lock:
        _singleton = None
