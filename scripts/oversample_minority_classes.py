"""Symlink-based oversampling for minority classes in a YOLO label folder.

Why: classes with only 1-3 frames train poorly because YOLO sees them
rarely per epoch. For static UI detection (vs fine-grained fused_avatar),
overfitting is a non-issue — train ≈ test distribution. Simply replicating
minority-class frames N times is the cheapest fix.

Strategy:
  1. Walk every *.txt label file, count which classes each frame contains
  2. For each class, compute frame_count
  3. For classes with frame_count < --target (default 8), symlink the
     containing frames (both .jpg and .txt) K times where K = ceil(target / current)
  4. Symlink names use _copyK suffix to coexist with originals.

Effect:
  - Original frames untouched
  - Minority classes appear `target` times per epoch instead of 1-3
  - Zero extra disk (symlinks)
  - val/_ui_val_pool untouched (this script only writes into --dir)

Usage:
    py scripts/oversample_minority_classes.py
    py scripts/oversample_minority_classes.py --target 10 --dry-run
"""
from __future__ import annotations
import argparse
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path,
                    default=Path("D:/Project/ai game secretary/data/raw_images/run_20260521_103956_distinct"),
                    help="folder containing frame_*.jpg + frame_*.txt label pairs")
    ap.add_argument("--target", type=int, default=8,
                    help="minimum frame count per class after oversampling (default 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan only, don't create symlinks")
    ap.add_argument("--clean", action="store_true",
                    help="remove all existing *_copy*.jpg/txt symlinks before running")
    args = ap.parse_args()

    if not args.dir.is_dir():
        print(f"[!] {args.dir} not found")
        return 1

    # Optional cleanup of previous copies
    if args.clean:
        rm = 0
        for f in list(args.dir.glob("*_copy*.*")):
            try:
                f.unlink(); rm += 1
            except Exception:
                pass
        print(f"[clean] removed {rm} previous copy files")

    classes_file = args.dir / "classes.txt"
    if not classes_file.exists():
        print(f"[!] missing {classes_file}")
        return 1
    classes = classes_file.read_text(encoding="utf-8").splitlines()
    print(f"[load] {len(classes)} classes defined")

    # Pass 1: scan labels (excluding any *_copy* from previous runs)
    label_files = [
        f for f in args.dir.glob("*.txt")
        if not f.name.startswith("_")  # skip _kept.txt etc
        and "_copy" not in f.name      # skip prior copies
        and f.stem != "classes"
    ]
    print(f"[scan] {len(label_files)} original label files")

    frame_classes = {}     # frame_stem -> set of cls_ids
    class_to_frames = defaultdict(set)  # cls_id -> set of frame_stems
    for txt in label_files:
        lines = [l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
        cids = set()
        for line in lines:
            try:
                cids.add(int(line.split()[0]))
            except Exception:
                pass
        if not cids:
            continue
        frame_classes[txt.stem] = cids
        for c in cids:
            class_to_frames[c].add(txt.stem)

    print(f"[stat] {len(frame_classes)} non-empty frames, {len(class_to_frames)} classes in use")

    # Pass 2: find minority classes
    minority = {
        c: frames for c, frames in class_to_frames.items()
        if len(frames) < args.target
    }
    print(f"[stat] {len(minority)} minority classes (frame_count < {args.target})")
    if not minority:
        print("[done] no minority classes — nothing to oversample")
        return 0

    # Pass 3: plan copies. For each minority class c, every frame containing c
    # should be replicated K-1 times, where K = ceil(target / current_count).
    # Frames may host multiple minority classes — take max K across them.
    frame_copy_count = {}  # frame_stem -> K (final replica multiplier)
    for c, frames in minority.items():
        cur = len(frames)
        K = math.ceil(args.target / cur)
        for f in frames:
            frame_copy_count[f] = max(frame_copy_count.get(f, 1), K)

    print(f"\n[plan] minority class breakdown:")
    for c in sorted(minority, key=lambda c: len(minority[c])):
        cname = classes[c] if c < len(classes) else f"cls_{c}"
        cur = len(minority[c])
        K = math.ceil(args.target / cur)
        print(f"  class {c:3d} {cname:30s}: {cur} → {cur*K} frames (K={K})")

    print(f"\n[plan] {len(frame_copy_count)} unique frames to replicate (K-1 copies each)")
    total_new = sum(K - 1 for K in frame_copy_count.values())
    print(f"[plan] total new symlinks to create: ~{total_new * 2} (jpg + txt pairs)")

    if args.dry_run:
        print("[dry-run] no files created")
        return 0

    # Pass 4: create symlinks
    created = 0
    failed = 0
    for stem, K in frame_copy_count.items():
        src_jpg = args.dir / f"{stem}.jpg"
        src_txt = args.dir / f"{stem}.txt"
        # If src is itself a symlink to a real file, follow it for the symlink target
        if not src_jpg.exists() or not src_txt.exists():
            continue
        target_jpg = src_jpg.resolve()
        target_txt = src_txt.resolve()
        for k in range(1, K):
            new_jpg = args.dir / f"{stem}_copy{k}.jpg"
            new_txt = args.dir / f"{stem}_copy{k}.txt"
            if new_jpg.exists() and new_txt.exists():
                continue
            try:
                if not new_jpg.exists():
                    try:
                        new_jpg.symlink_to(target_jpg)
                    except OSError:
                        shutil.copy2(target_jpg, new_jpg)
                if not new_txt.exists():
                    try:
                        new_txt.symlink_to(target_txt)
                    except OSError:
                        shutil.copy2(target_txt, new_txt)
                created += 2
            except Exception as e:
                failed += 1
                print(f"[!] failed for {stem}_copy{k}: {e}")

    print(f"\n[done] created {created} symlinks/copies, {failed} failures")

    # Re-stat for confirmation
    print("\n[verify] class frame counts after oversampling:")
    after_class_to_frames = defaultdict(set)
    for txt in args.dir.glob("*.txt"):
        if txt.name.startswith("_") or txt.stem == "classes":
            continue
        try:
            lines = [l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
            for line in lines:
                cid = int(line.split()[0])
                after_class_to_frames[cid].add(txt.stem)
        except Exception:
            pass
    still_low = [c for c, f in after_class_to_frames.items() if len(f) < args.target]
    if still_low:
        print(f"  [!] {len(still_low)} classes still under target (likely co-occur with other minorities)")
    else:
        print(f"  [OK] all {len(after_class_to_frames)} in-use classes >= {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
