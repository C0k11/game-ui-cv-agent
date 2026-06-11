# -*- coding: utf-8 -*-
"""Multi-teacher ADD-ONLY weak-class assist for a label queue.

Each teacher contributes ONLY the classes it is strongest at (user scheme
2026-06-10): v7 did the base prefill; this pass adds what v7 misses —
  v6c (unified)  → 双倍或三倍活动进行中(452) + 制造入口(14)
  ui v5          → 任务大厅入口(17)
  emoticon v26n  → Emoticon_Action(451), ONLY on cafe-interior frames
                   (gated by existing 咖啡厅收益25/邀请卷24 labels — v26n
                   false-fires on yellow strips elsewhere; human reviews).

ADD-ONLY: never rewrites or removes existing boxes (human corrections are
sacred). A detection is added only if no existing same-class box IoU>0.5 and
no any-class box IoU>0.6 at that spot. All inference at imgsz=960.
Also writes _cafe_frames.txt (frames with cafe-interior labels) for the
human emoticon eyeball pass.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_images"
YOLO_RUNS = Path(r"D:\Project\ml_cache\models\yolo\runs")

CAFE_GATE = {"24", "25"}          # 咖啡厅邀请卷 / 咖啡厅收益 → cafe interior
PASSES = [
    # (weights, {master_idx: (cls_name, conf)}, cafe_only)
    (YOLO_RUNS / "unified_yolo26x_v6c" / "weights" / "best.pt",
     {452: ("双倍或三倍活动进行中", 0.15), 14: ("制造入口", 0.30)}, False),
    (YOLO_RUNS / "ui_yolo26m_v5" / "weights" / "best_real.pt",
     {17: ("任务大厅入口", 0.30)}, False),
    (YOLO_RUNS / "emoticon_yolo26n" / "weights" / "best.pt",
     {451: ("Emoticon_Action", 0.50)}, True),
]


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / (a[3] * a[4] + b[3] * b[4] - inter)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    ds = RAW / args.dataset

    frames = sorted(ds.glob("frame_*.jpg"))
    labels = {}
    cafe_frames = []
    for jpg in frames:
        txt = jpg.with_suffix(".txt")
        lines = txt.read_text(encoding="utf-8").splitlines() if txt.exists() else []
        boxes = []
        for ln in lines:
            p = ln.split()
            if len(p) == 5:
                boxes.append([int(p[0])] + [float(v) for v in p[1:]])
        labels[jpg] = (lines, boxes)
        if any(ln.split()[0] in CAFE_GATE for ln in lines if ln.strip()):
            cafe_frames.append(jpg.name)

    (ds / "_cafe_frames.txt").write_text("\n".join(cafe_frames) + "\n", encoding="utf-8")
    print(f"cafe-interior frames: {len(cafe_frames)} → _cafe_frames.txt")

    from ultralytics import YOLO
    added = {}          # jpg -> list of new lines
    stats = {}
    for weights, targets, cafe_only in PASSES:
        model = YOLO(str(weights))
        name2master = {n: i for i, (n, _c) in ((i, t) for i, t in targets.items())} if False else {
            n: i for i, (n, _c) in targets.items()}
        min_conf = min(c for _n, c in targets.values())
        todo = [j for j in frames if (not cafe_only or j.name in set(cafe_frames))]
        print(f"[{weights.parent.parent.name}] scanning {len(todo)} frames for {list(name2master)} ...")
        BATCH = 8
        for s in range(0, len(todo), BATCH):
            chunk = todo[s:s + BATCH]
            for jpg, res in zip(chunk, model.predict([str(p) for p in chunk],
                                                     conf=min_conf, imgsz=960, verbose=False)):
                h, w = res.orig_shape
                _lines, boxes = labels[jpg]
                for b in res.boxes:
                    name = model.names[int(b.cls[0])]
                    if name not in name2master:
                        continue
                    midx = name2master[name]
                    if float(b.conf[0]) < targets[midx][1]:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                    cand = [midx, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                            (x2 - x1) / w, (y2 - y1) / h]
                    dup = any((_iou(cand, ex) > 0.6) or (ex[0] == midx and _iou(cand, ex) > 0.5)
                              for ex in boxes)
                    if dup:
                        continue
                    boxes.append(cand)
                    added.setdefault(jpg, []).append(
                        f"{midx} {cand[1]:.6f} {cand[2]:.6f} {cand[3]:.6f} {cand[4]:.6f}")
                    stats[targets[midx][0]] = stats.get(targets[midx][0], 0) + 1

    print("added boxes:", stats or "none", f"| frames touched: {len(added)}")
    if not args.apply:
        print("dry-run — rerun with --apply to write")
        return
    bak = ds / "_labels_bak_assist"
    bak.mkdir(exist_ok=True)
    for jpg, new_lines in added.items():
        txt = jpg.with_suffix(".txt")
        if txt.exists():
            shutil.copy2(txt, bak / txt.name)
        old = txt.read_text(encoding="utf-8").rstrip("\n") if txt.exists() else ""
        txt.write_text((old + "\n" if old else "") + "\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"applied. backups in {bak}")


if __name__ == "__main__":
    main()
