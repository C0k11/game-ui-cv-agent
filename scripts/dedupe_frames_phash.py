"""Deduplicate trajectory frames via perceptual hash (dHash).

Algorithm:
  1. Compute 8x8 dHash (64-bit) for each frame — measures gradient pattern
  2. Sort frames in chronological order
  3. Keep a frame if its dHash differs from the last kept frame by
     >= --threshold hamming bits (default 5 — empirically distinct)
  4. Symlink (or copy) kept frames into --out

Outputs:
    <out>/frame_NNNNN.jpg     symlinks (or copies)
    <out>/_kept.txt           original paths of kept frames
    <out>/_dropped.txt        original paths of dropped frames (with reason)

Usage:
    py scripts/dedupe_frames_phash.py \
        --src "D:/Project/ai game secretary/data/raw_images/run_20260521_103956" \
        --out "D:/Project/ai game secretary/data/raw_images/run_20260521_103956_distinct"
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="source dir of frames")
    ap.add_argument("--out", type=Path, required=True, help="output dir for distinct frames")
    ap.add_argument("--threshold", type=int, default=5,
                    help="minimum hamming distance to keep next frame (default 5)")
    ap.add_argument("--copy", action="store_true",
                    help="copy files instead of symlinking (slower, more disk)")
    args = ap.parse_args()

    src = args.src
    out = args.out
    if not src.is_dir():
        print(f"[!] {src} not found")
        return 1
    out.mkdir(parents=True, exist_ok=True)

    frames = sorted(src.glob("*.jpg"))
    print(f"[scan] {len(frames)} source frames in {src}")
    if not frames:
        return 1

    kept = []
    dropped = []
    last_hash = None

    for f in tqdm(frames, desc="dedupe"):
        try:
            h = imagehash.dhash(Image.open(f))
        except Exception as e:
            dropped.append((f, f"hash error: {e}"))
            continue
        if last_hash is None:
            kept.append((f, h, "first"))
            last_hash = h
            continue
        dist = h - last_hash
        if dist >= args.threshold:
            kept.append((f, h, f"dist={dist}"))
            last_hash = h
        else:
            dropped.append((f, f"dist={dist} < {args.threshold}"))

    # Materialize kept frames
    for f, h, reason in kept:
        dst = out / f.name
        if dst.exists():
            dst.unlink()
        if args.copy:
            shutil.copy2(f, dst)
        else:
            try:
                dst.symlink_to(f)
            except OSError:
                shutil.copy2(f, dst)

    (out / "_kept.txt").write_text(
        "\n".join(f"{str(f)}\t{h}\t{r}" for f, h, r in kept),
        encoding="utf-8",
    )
    (out / "_dropped.txt").write_text(
        "\n".join(f"{str(f)}\t{r}" for f, r in dropped),
        encoding="utf-8",
    )

    print()
    print(f"[done] kept    {len(kept)}/{len(frames)} frames ({100*len(kept)/len(frames):.1f}%)")
    print(f"       dropped {len(dropped)} (near-duplicates within threshold {args.threshold})")
    print(f"[out]  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
