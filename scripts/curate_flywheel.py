# -*- coding: utf-8 -*-
"""Curate clean flywheel frames into a single dashboard label queue.

Takes the raw clean-frame run dirs (ADB screencap, overlay-free), drops
near-duplicate frames (idle/cooldown screens repeat for minutes at 1f/2.5s),
merges everything into ONE data/raw_images/<out> dataset (PNG→JPG), and
pre-labels each kept frame with the current ui model (5-column YOLO txt with
MASTER class indices — never a 6th column, see flywheel_label_import memory).

Usage:
  py -X utf8 scripts/curate_flywheel.py --out run_20260610_v8queue ^
      --include "run_20260609_18*" "run_20260609_19*" ... --dedup 3.0
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_images"
TRAJ = ROOT / "data" / "trajectories"
MASTER = RAW / "_classes.txt"
# v9 active (2026-06-12) — prefill today's clean frames with the live model.
UI_WEIGHTS = Path(r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v9\weights\best.pt")

_THUMB = (48, 27)   # grayscale thumb for near-dupe metric


def _dedup_dir(args):
    """Sequential near-dupe filter inside one source dir. Returns kept paths."""
    src_dir, thresh = args
    frames = sorted([p for p in Path(src_dir).iterdir()
                     if p.suffix.lower() in (".jpg", ".png") and not p.name.startswith("_")])
    kept = []
    last_thumb = None
    for p in frames:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        thumb = cv2.resize(img, _THUMB).astype(np.float32)
        if last_thumb is not None and float(np.abs(thumb - last_thumb).mean()) < thresh:
            continue   # near-identical to the last KEPT frame → drop
        kept.append(str(p))
        last_thumb = thumb
    return src_dir, len(frames), kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dataset name under data/raw_images/")
    ap.add_argument("--include", nargs="+", required=True, help="source dir glob(s)")
    ap.add_argument("--from-traj", action="store_true",
                    help="search data/trajectories/ instead of raw_images/")
    ap.add_argument("--dedup", type=float, default=3.0, help="mean-abs-diff threshold on 48x27 gray thumbs")
    ap.add_argument("--conf", type=float, default=0.20, help="pre-label conf floor (model floor)")
    args = ap.parse_args()

    out_dir = RAW / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    search_root = TRAJ if args.from_traj else RAW
    srcs = sorted({d for pat in args.include for d in search_root.iterdir()
                   if d.is_dir() and fnmatch.fnmatch(d.name, pat) and d.name != args.out})
    if not srcs:
        print("no source dirs matched"); sys.exit(1)
    print(f"sources: {len(srcs)} dirs")

    # 1) parallel near-dupe filter (one worker per dir, fan out across cores)
    total_in, keep_list = 0, []
    with ProcessPoolExecutor(max_workers=24) as ex:
        for src, n_in, kept in ex.map(_dedup_dir, [(str(d), args.dedup) for d in srcs]):
            total_in += n_in
            keep_list.extend(kept)
            print(f"  {Path(src).name}: {n_in} -> {len(kept)}")
    print(f"dedup: {total_in} -> {len(keep_list)} frames")

    # 2) merge/copy into the queue dir (PNG -> JPG), manifest maps provenance
    manifest = []
    out_paths = []
    for i, src in enumerate(sorted(keep_list)):
        dst = out_dir / f"frame_{i:06d}.jpg"
        sp = Path(src)
        if sp.suffix.lower() == ".png":
            img = cv2.imread(src)
            cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            shutil.copy2(src, dst)
        try:
            prov = str(sp.relative_to(search_root))
        except ValueError:
            prov = str(sp)
        manifest.append((dst.name, prov))
        out_paths.append(str(dst))
    with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("frame", "source")] + manifest)

    # 3) master classes (append-only file is the single source of truth)
    master = [c.strip() for c in MASTER.read_text(encoding="utf-8").splitlines() if c.strip()]
    name2idx = {n: i for i, n in enumerate(master)}
    (out_dir / "classes.txt").write_text("\n".join(master) + "\n", encoding="utf-8")

    # 4) GPU batch pre-label with the ui model. 5 columns ONLY (cls xc yc w h,
    #    normalized) — a 6th column would be parsed as an OBB angle.
    from ultralytics import YOLO
    model = YOLO(str(UI_WEIGHTS))
    boxes_written, skipped_names = 0, set()
    BATCH = 16
    for s in range(0, len(out_paths), BATCH):
        chunk = out_paths[s:s + BATCH]
        # imgsz=960: the training/serving size — tiny cls (dots/badges) lose
        # their few-pixel color signal at the 640 default (live 2026-06-10).
        for path, res in zip(chunk, model.predict(chunk, conf=args.conf, imgsz=960, verbose=False)):
            h, w = res.orig_shape
            lines = []
            for b in res.boxes:
                name = model.names[int(b.cls[0])]
                idx = name2idx.get(name)
                if idx is None:
                    skipped_names.add(name)
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            Path(path).with_suffix(".txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            boxes_written += len(lines)
    print(f"pre-labeled {len(out_paths)} frames, {boxes_written} boxes"
          + (f", skipped cls not in master: {skipped_names}" if skipped_names else ""))
    print(f"queue ready: data/raw_images/{args.out}  (dashboard datasets list)")


if __name__ == "__main__":
    main()
