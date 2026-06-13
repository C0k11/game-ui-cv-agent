# -*- coding: utf-8 -*-
"""Move EMPTY-label frames (0 boxes) out of prefill pools (user 2026-06-13:
挂机立绘模式没有任何UI的frame可以删). A frame whose v9 prefill found nothing is
either the idle-showcase 立绘 screen or a capture glitch — zero training value
for the UI classes. Moved (not hard-deleted) to data/_empty_frames_backup/<pool>/
so it's recoverable.

Usage: py scripts/delete_empty_frames.py <pool> [pool2 ...]   (or a name pattern)
"""
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
BACKUP = Path(r"D:\Project\ai game secretary\data\_empty_frames_backup")


def main():
    args = sys.argv[1:]
    if not args:
        print("usage: delete_empty_frames.py <pool|pattern> ...")
        return
    pools = []
    for a in args:
        if (RAW / a).is_dir():
            pools.append(a)
        else:  # treat as pattern
            pools += [p.name for p in RAW.iterdir()
                      if p.is_dir() and a in p.name and p.name.endswith("_clean")]
    pools = sorted(set(pools))
    total = 0
    for pool in pools:
        pdir = RAW / pool
        moved = 0
        for txt in sorted(pdir.glob("frame_*.txt")):
            body = txt.read_text(encoding="utf-8").strip()
            if body:   # has ≥1 label line → keep
                continue
            jpg = txt.with_suffix(".jpg")
            bdir = BACKUP / pool
            bdir.mkdir(parents=True, exist_ok=True)
            if jpg.exists():
                shutil.move(str(jpg), str(bdir / jpg.name))
            shutil.move(str(txt), str(bdir / txt.name))
            moved += 1
        if moved:
            print(f"{pool}: moved {moved} empty frames")
        total += moved
    print(f"\n[done] moved {total} empty (no-UI/立绘) frames → {BACKUP}")


if __name__ == "__main__":
    main()
