# -*- coding: utf-8 -*-
"""Batch-fix 红点/黄点 label confusion by PIXEL COLOR arbitration.

The prefill teacher (ui v7) confuses the two dot classes (they differ only by
color; live recalls 红0.85/黄0.94 and the same fixed lobby badge repeats the
same mistake across hundreds of frames). The dashboard has no batch edit —
this script crops every 红点/黄点 box, measures the dominant hue of saturated
pixels, and flips labels that disagree with the measured color.

Usage:
  py -X utf8 scripts/fix_dot_colors.py --dataset run_20260610_v8queue          # dry-run report
  py -X utf8 scripts/fix_dot_colors.py --dataset run_20260610_v8queue --apply  # backup + write
"""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_images"

RED_IDX, YELLOW_IDX = 5, 6   # master: 红点=5 黄点=6

# OpenCV hue (0-180): red wraps at both ends; BA badge yellow is orange-gold.
def _classify_hue(h: float) -> str:
    if h < 14 or h > 165:
        return "red"
    if 16 <= h <= 45:
        return "yellow"
    return "ambiguous"


def _dot_color(img, xc, yc, w, h) -> str:
    H, W = img.shape[:2]
    # center 70% of the box — skip the anti-aliased rim
    bw, bh = w * W * 0.7, h * H * 0.7
    x1, y1 = int(max(0, xc * W - bw / 2)), int(max(0, yc * H - bh / 2))
    x2, y2 = int(min(W, xc * W + bw / 2)), int(min(H, yc * H + bh / 2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return "ambiguous"
    crop = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] > 110) & (hsv[..., 2] > 110)   # saturated, bright
    if mask.sum() < 8:
        return "ambiguous"
    return _classify_hue(float(np.median(hsv[..., 0][mask])))


def _process(txt_path_str: str):
    txt = Path(txt_path_str)
    jpg = txt.with_suffix(".jpg")
    if not jpg.exists():
        return None
    lines = txt.read_text(encoding="utf-8").splitlines()
    img = None
    changed, flips = False, []
    out = []
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5 or int(parts[0]) not in (RED_IDX, YELLOW_IDX):
            out.append(ln)
            continue
        if img is None:
            img = cv2.imread(str(jpg))
        idx = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        color = _dot_color(img, xc, yc, w, h)
        want = RED_IDX if color == "red" else YELLOW_IDX if color == "yellow" else idx
        if want != idx:
            flips.append((idx, want, xc, yc))
            parts[0] = str(want)
            changed = True
        out.append(" ".join(parts))
    return (str(txt), out, flips) if changed else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    ds = RAW / args.dataset
    txts = sorted(p for p in ds.glob("frame_*.txt"))
    print(f"scanning {len(txts)} label files in {ds.name} ...")

    results = []
    with ProcessPoolExecutor(max_workers=24) as ex:
        for r in ex.map(_process, [str(p) for p in txts], chunksize=32):
            if r:
                results.append(r)

    r2y = sum(1 for _, _, fl in results for a, b, *_ in fl if a == RED_IDX)
    y2r = sum(1 for _, _, fl in results for a, b, *_ in fl if a == YELLOW_IDX)
    print(f"frames needing fix: {len(results)}  |  红点→黄点: {r2y}  黄点→红点: {y2r}")
    for path, _, fl in results[:8]:
        det = ", ".join(f"{'红→黄' if a == RED_IDX else '黄→红'}@({x:.2f},{y:.2f})" for a, b, x, y in fl)
        print(f"  {Path(path).name}: {det}")
    if len(results) > 8:
        print(f"  ... and {len(results) - 8} more frames")

    if not args.apply:
        print("dry-run only — rerun with --apply to write (backs up first)")
        return
    bak = ds / "_labels_bak_dotfix"
    bak.mkdir(exist_ok=True)
    for path, out, _ in results:
        p = Path(path)
        shutil.copy2(p, bak / p.name)
        p.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"applied. backups in {bak}")


if __name__ == "__main__":
    main()
