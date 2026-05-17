"""Wholesale audit: run the YOLO26n-cls avatar classifier against every
schedule-roster frame in every trajectory.

Why this exists (2026-05-17):
  We just shipped vision/avatar_classifier.py.  Before declaring victory,
  sanity-check it on historical data — every tick where the BA schedule
  roster was open.  Output a coverage report:
    * how many cells got classified vs went to __unknown__
    * confidence distribution
    * per-class hit counts (which characters does the model "see" in the wild)
    * resolution mix (strip coords only line up with one resolution)

This is SINGLE-FRAME eval — no temporal voting.  Vote requires 3 frames
of the same (room_idx, cell_idx) which only happens during a live roster
scan; replaying historical frames out-of-order doesn't simulate that
correctly.  Single-frame top1 + margin + conf is the right metric here.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
TRAJ_ROOT = REPO / "data" / "trajectories"
STRIPS_JSON = REPO / "data" / "schedule_avatar_regions.json"


def load_strips() -> Tuple[List[Dict[str, float]], int]:
    if not STRIPS_JSON.exists():
        raise FileNotFoundError(STRIPS_JSON)
    cfg = json.loads(STRIPS_JSON.read_text(encoding="utf-8"))
    return cfg.get("strips", []), int(cfg.get("cells_per_room", 3))


_ROSTER_ROOM_NAMES = (
    "視聽室", "體育館", "圖書館", "教室", "實驗室", "射擊場", "載具庫",
)


def is_schedule_roster_tick(d: Dict[str, Any]) -> bool:
    """A trajectory tick where the schedule roster overlay is visible.

    Detection: Schedule skill is active AND OCR shows BOTH:
      - 全體課程表 header text, AND
      - at least one room name (視聽室/體育館/...)

    The header-text-only filter is ambiguous because 全體課程表 also
    appears as a BUTTON LABEL on the location-detail screen (you click
    it to OPEN the roster).  68.6% of frames matching only header text
    are location-detail not roster-open — at those frames the avatar
    strips crop random UI chrome (TIPS button, page numbers, headers).
    Room names appear ONLY in the actual roster overlay so they're a
    reliable disambiguator.
    """
    if d.get("skill") != "Schedule":
        return False
    has_header = False
    has_room = False
    for b in d.get("ocr_boxes", []) or []:
        text = b.get("text") or ""
        if "全體課程表" in text or "全体课程表" in text:
            has_header = True
        if any(rn in text for rn in _ROSTER_ROOM_NAMES):
            has_room = True
        if has_header and has_room:
            return True
    return False


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Cap total roster frames evaluated (debug).")
    ap.add_argument("--fast-conf", type=float, default=0.95)
    ap.add_argument("--fast-margin", type=float, default=0.30)
    ap.add_argument("--unknown-conf", type=float, default=0.50)
    ap.add_argument("--filter-size", default=None,
                    help="Filter frames by exact 'WxH' (strips were drawn at "
                         "one resolution; BA renders fixed-pixel UI so other "
                         "resolutions misalign).  Example: 3840x2160")
    args = ap.parse_args()
    size_filter: Optional[Tuple[int, int]] = None
    if args.filter_size:
        w, h = (int(x) for x in args.filter_size.lower().split("x"))
        size_filter = (w, h)

    strips, cells_per_room = load_strips()
    if not strips:
        print(f"[err] no strips in {STRIPS_JSON}")
        return 1
    print(f"[strips] {len(strips)} strips x {cells_per_room} cells/room")

    from vision.avatar_classifier import AvatarClassifier
    clf = AvatarClassifier()
    if not clf.available:
        print(f"[err] classifier model missing at {clf.model_path}")
        return 1
    clf._ensure_loaded()
    cls_names = clf.get_class_names()
    print(f"[clf] {len(cls_names)} classes loaded")

    # ── walk trajectories ──
    roster_frames: List[Tuple[Path, Dict[str, Any]]] = []
    for run in sorted(TRAJ_ROOT.iterdir()):
        if not run.is_dir() or not run.name.startswith("run_"):
            continue
        for tj in sorted(run.glob("tick_*.json")):
            try:
                d = json.loads(tj.read_text(encoding="utf-8"))
            except Exception:
                continue
            if is_schedule_roster_tick(d):
                roster_frames.append((tj, d))
                if args.max_frames and len(roster_frames) >= args.max_frames:
                    break
        if args.max_frames and len(roster_frames) >= args.max_frames:
            break

    print(f"[scan] found {len(roster_frames)} roster-open ticks across "
          f"{len({p.parent.name for p, _ in roster_frames})} runs")

    if not roster_frames:
        return 0

    # ── classify every cell ──
    n_cells_total = 0
    n_cells_skipped = 0  # too small / empty crop
    source_counts = Counter()
    class_top1_counts = Counter()      # how often each class wins top1
    class_fast_counts = Counter()      # how often each class wins layer-1 fast-path
    conf_all: List[float] = []
    conf_fast: List[float] = []
    conf_unknown: List[float] = []     # top1 conf of cells that went to __unknown__
    resolution_mix: Counter = Counter()
    per_run_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"frames": 0, "cells": 0, "fast": 0, "unknown": 0, "pending": 0}
    )

    for i, (tj, d) in enumerate(roster_frames):
        jpg = tj.with_suffix(".jpg")
        if not jpg.exists():
            continue
        try:
            img = cv2.imdecode(np.fromfile(str(jpg), dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            continue
        if img is None:
            continue
        h, w = img.shape[:2]
        if size_filter is not None and (w, h) != size_filter:
            continue
        resolution_mix[(w, h)] += 1
        run_name = tj.parent.name
        per_run_stats[run_name]["frames"] += 1

        # crop all cells (room×slot)
        crops: List[Tuple[Any, int, int]] = []
        for room_idx, s in enumerate(strips):
            px1 = max(0, int(s["x1"] * w))
            py1 = max(0, int(s["y1"] * h))
            px2 = min(w, int(s["x2"] * w))
            py2 = min(h, int(s["y2"] * h))
            strip = img[py1:py2, px1:px2]
            if strip.size == 0:
                continue
            strip_h, strip_w = strip.shape[:2]
            cell_w = max(1, strip_w // cells_per_room)
            cell_size = min(cell_w, strip_h)
            if cell_size < 16:
                n_cells_skipped += cells_per_room
                continue
            sy = max(0, (strip_h - cell_size) // 2)
            for slot in range(cells_per_room):
                sx = slot * cell_w + max(0, (cell_w - cell_size) // 2)
                if sx + cell_size > strip_w:
                    break
                cell = strip[sy:sy + cell_size, sx:sx + cell_size]
                if cell.size == 0:
                    continue
                crops.append((cell, room_idx, slot))
        if not crops:
            continue

        # disable buffer: we want pure single-frame results (each tick is
        # independent — buffering across runs makes no sense)
        clf.reset_buffer()

        # FRESH per-frame classification — fast path + unknown only,
        # NO vote (vote would buffer with no other frames coming).
        results = clf._model.predict(
            [c[0] for c in crops], verbose=False, imgsz=clf.imgsz, device=clf.device,
        )
        for (cell, room_idx, slot), r in zip(crops, results):
            probs = r.probs.data.cpu().numpy()
            top5 = clf._top5(probs)
            top1_name, top1_conf = top5[0]
            top2_conf = top5[1][1] if len(top5) > 1 else 0.0
            margin = top1_conf - top2_conf

            conf_all.append(top1_conf)
            class_top1_counts[top1_name] += 1
            n_cells_total += 1

            if top1_conf >= args.fast_conf and margin >= args.fast_margin:
                source_counts["fast"] += 1
                class_fast_counts[top1_name] += 1
                conf_fast.append(top1_conf)
                per_run_stats[run_name]["fast"] += 1
            elif top1_conf < args.unknown_conf:
                source_counts["unknown"] += 1
                conf_unknown.append(top1_conf)
                per_run_stats[run_name]["unknown"] += 1
            else:
                source_counts["pending"] += 1  # would buffer for vote in live mode
                per_run_stats[run_name]["pending"] += 1
            per_run_stats[run_name]["cells"] += 1

        if (i + 1) % 100 == 0:
            print(f"  scanned {i+1}/{len(roster_frames)} frames | "
                  f"cells {n_cells_total} | fast {source_counts['fast']} "
                  f"unknown {source_counts['unknown']}")

    # ── REPORT ──
    print()
    print("=" * 72)
    print(f"TOTAL: {n_cells_total} cells across {len(roster_frames)} roster frames")
    print(f"  skipped: {n_cells_skipped} (too tiny)")
    print()

    print("─── Resolution mix ───")
    for (w, h), n in resolution_mix.most_common():
        print(f"  {w:>5d}x{h:<5d}  {n:>5d} frames")
    print()

    print("─── Verdict source (single-frame eval) ───")
    for src in ("fast", "pending", "unknown"):
        n = source_counts[src]
        pct = 100 * n / n_cells_total if n_cells_total else 0
        print(f"  {src:<8s}  {n:>5d}  ({pct:5.1f}%)")
    print()

    if conf_all:
        confs = np.array(conf_all)
        print("─── Top1 confidence distribution (all cells) ───")
        print(f"  min:    {confs.min():.3f}")
        print(f"  median: {np.median(confs):.3f}")
        print(f"  mean:   {confs.mean():.3f}")
        print(f"  max:    {confs.max():.3f}")
        for thr in (0.50, 0.70, 0.85, 0.95):
            n = int((confs >= thr).sum())
            print(f"  >={thr:.2f}: {n:>5d} ({100*n/len(confs):5.1f}%)")
    print()

    print("─── Top-15 fast-path classes (what the model confidently sees) ───")
    for cls, n in class_fast_counts.most_common(15):
        try:
            print(f"  {n:>5d}  {cls}")
        except UnicodeEncodeError:
            print(f"  {n:>5d}  <cjk>")
    print()

    print("─── Top-15 overall top1 classes (incl. low-conf guesses) ───")
    for cls, n in class_top1_counts.most_common(15):
        try:
            print(f"  {n:>5d}  {cls}")
        except UnicodeEncodeError:
            print(f"  {n:>5d}  <cjk>")
    print()

    # Per-run snapshot (last 8 runs)
    print("─── Per-run stats (most recent 8) ───")
    print(f"  {'run':<30s}{'frames':>8s}{'cells':>7s}{'fast%':>8s}{'unk%':>7s}")
    runs_sorted = sorted(per_run_stats.keys(), reverse=True)[:8]
    for r in runs_sorted:
        st = per_run_stats[r]
        if st["cells"] == 0:
            continue
        fast_pct = 100 * st["fast"] / st["cells"]
        unk_pct = 100 * st["unknown"] / st["cells"]
        print(f"  {r:<30s}{st['frames']:>8d}{st['cells']:>7d}"
              f"{fast_pct:>7.1f}%{unk_pct:>6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
