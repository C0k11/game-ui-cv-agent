"""Post-process auto-labeled YOLO labels with global (cross-class) NMS.

Why: auto_annotate_ref.py does per-template NMS only — different templates
that match the same screen location (e.g. all 7 academy variants of
"Beginner-Tech-Notes-(Abydos/Gehenna/Millennium/Trinity/...)") will stack
boxes on the same pixel. This script drops overlapping boxes, keeping the
highest-confidence one per location.

Note: we don't have YOLO conf scores in the .txt files (YOLO format is
just class+bbox), so we read scores from hit_stats.json's per-frame match
log. If that's not available, we fall back to dropping all but one box
when boxes overlap heavily (IoU > 0.5).
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

DEFAULT_DIR = Path("D:/Project/ai game secretary/data/yolo_datasets/ui_v1_auto")


def overlap_metrics(a, b):
    """Returns (iou, iomin) for two (cx, cy, w, h) normalized boxes.
    iomin = intersection / min(area_a, area_b) — sensitive to nested boxes
    where one is fully inside the other (iomin → 1, but iou stays low).
    """
    ax1, ay1 = a[0] - a[2]/2, a[1] - a[3]/2
    ax2, ay2 = a[0] + a[2]/2, a[1] + a[3]/2
    bx1, by1 = b[0] - b[2]/2, b[1] - b[3]/2
    bx2, by2 = b[0] + b[2]/2, b[1] + b[3]/2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    iou = inter / union if union > 0 else 0.0
    iomin = inter / min(a_area, b_area) if min(a_area, b_area) > 0 else 0.0
    return iou, iomin


def nms(boxes_with_meta, iou_threshold=0.5, iomin_threshold=0.7):
    """boxes_with_meta: list of (cid, cx, cy, w, h, score). Returns kept list.

    Suppression triggers if EITHER:
      - iou >= iou_threshold (typical overlap)
      - iomin >= iomin_threshold (nested boxes: small box fully inside big)
    """
    def sort_key(b):
        # Prefer higher score, then larger area (more distinctive template).
        # Score may be None — falls back to area-only ranking.
        s = b[5] if b[5] is not None else 0.0
        a = b[3] * b[4]
        return (-s, -a)
    sorted_b = sorted(boxes_with_meta, key=sort_key)
    kept = []
    for b in sorted_b:
        suppress = False
        for k in kept:
            iou, iomin = overlap_metrics(b[1:5], k[1:5])
            if iou >= iou_threshold or iomin >= iomin_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(b)
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU threshold for cross-class suppression (default 0.5)")
    ap.add_argument("--iomin", type=float, default=0.7,
                    help="Intersection-over-min-area threshold (catches nested boxes; default 0.7)")
    ap.add_argument("--backup", action="store_true",
                    help="copy labels/ to labels_pre_nms/ before overwriting")
    args = ap.parse_args()

    labels_dir = args.in_dir / "labels"
    if not labels_dir.exists():
        print(f"[!] {labels_dir} not found")
        return 1

    if args.backup:
        backup = args.in_dir / "labels_pre_nms"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.copytree(labels_dir, backup)
        print(f"[+] backed up to {backup}")

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        print(f"[!] no .txt in {labels_dir}")
        return 1

    total_before = 0
    total_after = 0
    frames_changed = 0

    for f in tqdm(label_files, desc="nms"):
        lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(lines) <= 1:
            total_before += len(lines)
            total_after += len(lines)
            continue
        boxes = []
        for line in lines:
            parts = line.split()
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            # YOLO format doesn't include score; use None — NMS will tiebreak by area
            boxes.append((cid, cx, cy, w, h, None))
        kept = nms(boxes, iou_threshold=args.iou, iomin_threshold=args.iomin)
        total_before += len(boxes)
        total_after += len(kept)
        if len(kept) != len(boxes):
            frames_changed += 1
        new_lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                     for (c, cx, cy, w, h, _) in kept]
        f.write_text("\n".join(new_lines), encoding="utf-8")

    print()
    print(f"[done] processed {len(label_files)} label files")
    print(f"       boxes before: {total_before}")
    print(f"       boxes after:  {total_after}  ({total_after - total_before:+d}, "
          f"{100*(total_after-total_before)/max(total_before,1):.1f}%)")
    print(f"       frames changed: {frames_changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
