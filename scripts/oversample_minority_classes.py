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
    ap.add_argument("--dir", type=Path, action="append", default=None,
                    help="folder containing frame_*.jpg + frame_*.txt pairs. "
                         "Can be passed multiple times for multi-dir oversampling "
                         "(class counts are pooled across all dirs).")
    ap.add_argument("--target", type=int, default=8,
                    help="minimum frame count per class after oversampling (default 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan only, don't create symlinks")
    ap.add_argument("--clean", action="store_true",
                    help="remove all existing *_copy*.jpg/txt symlinks before running")
    args = ap.parse_args()

    # Default = the 4 UI train dirs (after sync). User can override with --dir.
    if args.dir is None:
        ROOT = Path("D:/Project/ai game secretary/data/raw_images")
        args.dir = [
            ROOT / "run_20260521_103956_distinct",
            ROOT / "run_20260527_094158",
            ROOT / "run_20260527_101545",
            ROOT / "run_20260518_002646",
        ]

    dirs = [d for d in args.dir if d.is_dir()]
    missing = [d for d in args.dir if not d.is_dir()]
    if missing:
        for d in missing:
            print(f"[!] dir missing, skipping: {d}")
    if not dirs:
        print("[!] no valid dirs to process")
        return 1
    print(f"[load] processing {len(dirs)} dirs:")
    for d in dirs:
        print(f"        {d.name}")

    # Optional cleanup across all dirs
    if args.clean:
        rm = 0
        for d in dirs:
            for f in list(d.glob("*_copy*.*")):
                try:
                    f.unlink(); rm += 1
                except Exception:
                    pass
        print(f"[clean] removed {rm} previous copy files across {len(dirs)} dirs")

    # Schema (from first dir's classes.txt)
    classes_file = dirs[0] / "classes.txt"
    if not classes_file.exists():
        print(f"[!] missing schema: {classes_file}")
        return 1
    classes = classes_file.read_text(encoding="utf-8").splitlines()
    print(f"[load] {len(classes)} classes defined (schema from {dirs[0].name})")

    # Pass 1: scan ALL dirs, pool class frame counts globally.
    # Each frame is keyed by (dir_index, stem) so we know where to write copies.
    frame_key_to_classes = {}        # (dir_idx, stem) -> set(cls)
    class_to_frame_keys = defaultdict(set)  # cls -> set of (dir_idx, stem)
    for di, d in enumerate(dirs):
        label_files = [
            f for f in d.glob("*.txt")
            if not f.name.startswith("_") and "_copy" not in f.name and f.stem != "classes"
        ]
        for txt in label_files:
            try:
                lines = [l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
            except Exception:
                continue
            cids = set()
            for line in lines:
                try:
                    cids.add(int(line.split()[0]))
                except Exception:
                    pass
            if not cids:
                continue
            key = (di, txt.stem)
            frame_key_to_classes[key] = cids
            for c in cids:
                class_to_frame_keys[c].add(key)

    print(f"[stat] {len(frame_key_to_classes)} non-empty frames pooled across dirs, "
          f"{len(class_to_frame_keys)} classes in use globally")

    # Pass 2: find global minority classes (pooled across all dirs)
    minority = {
        c: keys for c, keys in class_to_frame_keys.items()
        if len(keys) < args.target
    }
    print(f"[stat] {len(minority)} minority classes (pooled frame_count < {args.target})")
    if not minority:
        print("[done] no minority classes — nothing to oversample")
        return 0

    # Pass 3: plan copies per frame_key
    frame_copy_count = {}  # key -> K
    for c, keys in minority.items():
        cur = len(keys)
        K = math.ceil(args.target / cur)
        for k in keys:
            frame_copy_count[k] = max(frame_copy_count.get(k, 1), K)

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

    # Pass 4: create symlinks in each frame's home dir
    created = 0
    failed = 0
    for (di, stem), K in frame_copy_count.items():
        d = dirs[di]
        src_jpg = d / f"{stem}.jpg"
        src_txt = d / f"{stem}.txt"
        if not src_jpg.exists() or not src_txt.exists():
            continue
        target_jpg = src_jpg.resolve()
        target_txt = src_txt.resolve()
        for k in range(1, K):
            new_jpg = d / f"{stem}_copy{k}.jpg"
            new_txt = d / f"{stem}_copy{k}.txt"
            if new_jpg.exists() and new_txt.exists():
                continue
            try:
                if not new_jpg.exists():
                    try: new_jpg.symlink_to(target_jpg)
                    except OSError: shutil.copy2(target_jpg, new_jpg)
                if not new_txt.exists():
                    try: new_txt.symlink_to(target_txt)
                    except OSError: shutil.copy2(target_txt, new_txt)
                created += 2
            except Exception as e:
                failed += 1
                print(f"[!] failed {d.name}/{stem}_copy{k}: {e}")

    print(f"\n[done] created {created} symlinks/copies, {failed} failures")

    # Verify post-oversample (also includes copies this time)
    print("\n[verify] class frame counts after oversampling (pooled):")
    after = defaultdict(set)
    for di, d in enumerate(dirs):
        for txt in d.glob("*.txt"):
            if txt.name.startswith("_") or txt.stem == "classes":
                continue
            try:
                for line in txt.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        after[int(line.split()[0])].add((di, txt.stem))
            except Exception:
                pass
    still_low = sorted([(c, len(f)) for c, f in after.items() if len(f) < args.target],
                       key=lambda x: x[1])
    if still_low:
        print(f"  [!] {len(still_low)} classes still under target:")
        for c, n in still_low[:15]:
            cname = classes[c] if c < len(classes) else f"cls_{c}"
            print(f"      cls {c:3d} {cname:30s}: {n} frames")
    else:
        print(f"  [OK] all {len(after)} in-use classes >= {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
