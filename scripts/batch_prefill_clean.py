# -*- coding: utf-8 -*-
"""Batch v9-prefill the bot's auto-recorded flywheel _clean pools IN PLACE.

The pipeline saves overlay-free ADB clean frames to data/raw_images/run_*_clean
during every run. They show in the dashboard 数据飞轮 list but have NO labels
until prefilled. This loads v9 ONCE and batch-predicts every frame across all
matched pools (GPU saturated), writing 5-col master-index labels (never a 6th
col — that's read as an OBB angle by the dashboard).

Usage:
  py scripts/batch_prefill_clean.py 2026061           # all _clean pools matching
  py scripts/batch_prefill_clean.py 2026061 --force    # re-label even if txt exist
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
WEIGHTS = r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v9\weights\best.pt"
MASTER = [l.strip() for l in open(RAW / "_classes.txt", encoding="utf-8") if l.strip()]
NAME2IDX = {n: i for i, n in enumerate(MASTER)}


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "2026061"
    force = "--force" in sys.argv

    pools = sorted(p for p in RAW.iterdir()
                   if p.is_dir() and pat in p.name and p.name.endswith("_clean"))
    # collect frames needing labels
    jobs = []
    for pool in pools:
        jpgs = sorted(pool.glob("*.jpg"))
        if not jpgs:
            continue
        n_txt = sum(1 for t in pool.glob("*.txt") if t.name != "classes.txt")
        if n_txt >= len(jpgs) and not force:
            print(f"  skip {pool.name} (already {n_txt}/{len(jpgs)} labelled)")
            continue
        # write master classes.txt into the pool for the dashboard
        (pool / "classes.txt").write_text("\n".join(MASTER) + "\n", encoding="utf-8")
        jobs += [str(j) for j in jpgs]
        print(f"  queue {pool.name}: {len(jpgs)} frames")
    if not jobs:
        print("nothing to prefill")
        return
    print(f"\nprefilling {len(jobs)} frames with v9 …", flush=True)

    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    n2m = {i: NAME2IDX[n] for i, n in model.names.items() if n in NAME2IDX}
    boxes_written = 0
    B = 24
    for s in range(0, len(jobs), B):
        chunk = jobs[s:s + B]
        results = model.predict(chunk, conf=0.20, imgsz=960, verbose=False)
        for path, res in zip(chunk, results):
            h, w = res.orig_shape
            lines = []
            for b in res.boxes:
                mi = n2m.get(int(b.cls[0]))
                if mi is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{mi} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            Path(path).with_suffix(".txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            boxes_written += len(lines)
        if (s // B) % 10 == 0:
            print(f"  {s + len(chunk)}/{len(jobs)} frames, {boxes_written} boxes",
                  flush=True)
    print(f"\n[done] {len(jobs)} frames, {boxes_written} boxes prefilled")
    print("pools:", ", ".join(p.name for p in pools))


if __name__ == "__main__":
    main()
