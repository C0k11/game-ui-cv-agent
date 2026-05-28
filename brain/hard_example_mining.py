"""Auto-collect hard examples from production pipeline ticks for active learning.

Definition of "hard":
  1. Borderline confidence detections (LOW_CONF_MIN ≤ conf ≤ LOW_CONF_MAX)
     — the model is uncertain. Likely either a missed positive or noisy
     false positive. Human review tells us which.
  2. Frame had a skill failure ("expected button not found", "stuck X ticks")
     while the skill was in a state that expected UI to be detectable. Same
     human review process.

Output structure:
    data/hard_examples/<YYYYMMDD>/<run_id>/<tick>.jpg     screenshot copy
    data/hard_examples/<YYYYMMDD>/<run_id>/<tick>.txt     YOLO label format,
                                                          one line per
                                                          low-conf detection
                                                          (conf in 6th col so
                                                          reviewer sees it)
    data/hard_examples/<YYYYMMDD>/<run_id>/_meta.json     run metadata
                                                          (skill, sub_state,
                                                          reason)

The dashboard can later mount this folder as a labeling dataset — the auto
labels show the model's borderline guesses pre-loaded so reviewer just has
to confirm / correct / add missing boxes.

API:
    maybe_save_hard_example(...)  -- call from pipeline tick after detect
                                     (no-op unless conf in borderline range)
"""
from __future__ import annotations
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
HARD_DIR = REPO_ROOT / "data" / "hard_examples"

# Tuning knobs — adjust based on noise tolerance.
LOW_CONF_MIN = 0.30   # below this = ignore as noise
LOW_CONF_MAX = 0.60   # above this = trust the model
MAX_BORDERLINE_PER_FRAME = 8     # if too many low-conf hits, frame is probably
                                  # just confusing — cap to avoid dumping every
                                  # tick
MAX_PER_RUN = 200     # safety cap so a stuck pipeline can't fill the disk


_run_counts: dict[str, int] = {}  # run_id → frames saved this run


def _today() -> str:
    return datetime.now().strftime("%Y%m%d")


def _run_dir(run_id: str) -> Path:
    d = HARD_DIR / _today() / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def maybe_save_hard_example(
    screenshot_path: Optional[str],
    yolo_boxes: list,
    *,
    run_id: str,
    tick: int,
    skill_name: str = "",
    sub_state: str = "",
    reason: str = "",
) -> Optional[Path]:
    """If the current frame contains borderline-conf detections, persist
    a copy + pseudo-labels for human review. Returns saved label path or None.

    Args:
        screenshot_path: absolute path to the original screenshot jpg
        yolo_boxes:      list of YoloBox (or compatible) with .confidence /
                         .cls_id / .x1y1x2y2
        run_id:          pipeline run id (used to scope the hard-example dir)
        tick:            tick counter, used as filename stem
        skill_name / sub_state / reason: contextual metadata for review
    """
    if not screenshot_path or not yolo_boxes:
        return None

    # Per-run cap to prevent runaway dumps
    if _run_counts.get(run_id, 0) >= MAX_PER_RUN:
        return None

    borderline = [
        b for b in yolo_boxes
        if LOW_CONF_MIN <= float(getattr(b, "confidence", 0.0)) <= LOW_CONF_MAX
    ]
    if not borderline:
        return None
    if len(borderline) > MAX_BORDERLINE_PER_FRAME:
        # Frame is just noisy — skip rather than dump tons of garbage
        return None

    src = Path(screenshot_path)
    if not src.is_file():
        return None

    out_dir = _run_dir(run_id)
    out_stem = f"tick_{tick:05d}"
    out_jpg = out_dir / f"{out_stem}.jpg"
    out_txt = out_dir / f"{out_stem}.txt"

    # Skip if already saved this tick (multiple calls in same tick)
    if out_jpg.exists():
        return None

    try:
        shutil.copy2(src, out_jpg)
    except OSError as e:
        print(f"[HardExample] copy fail {src} -> {out_jpg}: {e}")
        return None

    # Emit pseudo-labels in YOLO format (cls cx cy w h conf shape).
    # Reviewer opens this in dashboard Annotate page and sees pre-filled boxes.
    lines = []
    for b in borderline:
        x1 = float(getattr(b, "x1", 0))
        y1 = float(getattr(b, "y1", 0))
        x2 = float(getattr(b, "x2", 0))
        y2 = float(getattr(b, "y2", 0))
        cls_id = int(getattr(b, "cls_id", 0))
        conf = float(getattr(b, "confidence", 0.0))
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        # YOLO base format: cls cx cy w h. Extend with conf as 6th column —
        # ultralytics ignores extra columns, dashboard parses them.
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}")

    out_txt.write_text("\n".join(lines), encoding="utf-8")

    # Meta append (one frame per JSON line, easier to grep / append)
    meta_path = out_dir / "_meta.jsonl"
    meta_line = json.dumps({
        "tick": tick,
        "skill": skill_name,
        "sub_state": sub_state,
        "reason": reason,
        "n_borderline": len(borderline),
        "min_conf": min((b.confidence for b in borderline), default=0.0),
        "max_conf": max((b.confidence for b in borderline), default=0.0),
        "source_jpg": str(src),
    }, ensure_ascii=False)
    with meta_path.open("a", encoding="utf-8") as f:
        f.write(meta_line + "\n")

    _run_counts[run_id] = _run_counts.get(run_id, 0) + 1
    return out_txt


def get_run_count(run_id: str) -> int:
    return _run_counts.get(run_id, 0)


def reset_run(run_id: str) -> None:
    """Clear in-memory counter (useful between dry-runs / tests)."""
    _run_counts.pop(run_id, None)
