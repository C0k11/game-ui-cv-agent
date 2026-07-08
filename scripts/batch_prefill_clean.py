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
sys.path.insert(0, r"D:\Project\ai game secretary")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
WEIGHTS = r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v12\weights\best.pt"
MASTER = [l.strip() for l in open(RAW / "_classes.txt", encoding="utf-8") if l.strip()]
NAME2IDX = {n: i for i, n in enumerate(MASTER)}


_DOT_CLS = {"红点", "黄点"}


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "2026061"
    force = "--force" in sys.argv
    # --any-suffix: also prefill pools NOT ending in _clean (e.g. a manual
    # capture/start recording like the 战术大赛商店 material run_20260614_205540,
    # which IS clean ADB but lacks the auto _clean suffix). Only flip this on
    # for pools you have VISUALLY CONFIRMED are overlay-free — DXcam/trajectory
    # runs are burned and must never be prefilled into train.
    any_suffix = "--any-suffix" in sys.argv
    # 红点/黄点 fire a POSITION PRIOR at low conf on grey/blue entry placeholders
    # (user 2026-06-13: pale-blue 社交 badge labelled 红点). v9 conf separates:
    # false position-prior firings ~0.2-0.45, real dots ~0.75+. So stamp dots
    # only at a higher floor; other cls keep the low floor.
    dot_conf = 0.55
    for a in sys.argv:
        if a.startswith("--dot-conf="):
            dot_conf = float(a.split("=", 1)[1])

    pools = sorted(p for p in RAW.iterdir()
                   if p.is_dir() and pat in p.name
                   and (any_suffix or p.name.endswith("_clean")))
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
    print(f"\nprefilling {len(jobs)} frames with v12 …", flush=True)

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
                local = int(b.cls[0])
                mi = n2m.get(local)
                if mi is None:
                    continue
                # dot classes need a higher conf floor (drop position-prior FPs)
                if model.names[local] in _DOT_CLS and float(b.conf[0]) < dot_conf:
                    continue
                # dot HSV posterior (2026-07-08): conf 已不可分(假点0.72爬进真点
                # 区间0.85-0.92), 颜色占比完美分离(真0.67+/假0.00)。非红非黄→丢;
                # 颜色与模型 cls 相反(红黄混淆)→ 按颜色改写 cls。
                if model.names[local] in _DOT_CLS:
                    bx1, by1, bx2, by2 = [float(v) for v in b.xyxy[0]]
                    from brain.skills.base import classify_dot_color
                    seen = classify_dot_color(
                        res.orig_img, bx1 / w, by1 / h, bx2 / w, by2 / h)
                    if seen is None:
                        continue
                    if seen != model.names[local]:
                        mi = NAME2IDX[seen]
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                # 397 获得奖励 语义混淆闸 (2026-07-08): 戰鬥結果弹窗的 WIN! 横幅
                # (同为黄色斜体艺术字)被误标 397。尺寸干净分离: 真标题 h≤0.094
                # (p90, n=277) / WIN 假框 h≥0.115 (n=18) → h>0.105 拒。
                if model.names[local] == "获得奖励" and bh > 0.105:
                    continue
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
