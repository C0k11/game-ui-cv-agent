# -*- coding: utf-8 -*-
"""Old-trainset label audit — prediction/label disagreement mining.

用户 2026-06-11: "你以后训练也要看看老的模型的训练集... 老的训练集会有瑕疵
就像红黄点这些经典问题, 你也要修复下老的训练集。"

Runs the CURRENT best ui model over every frame in the ui_v2 TRAIN split and
mines three disagreement classes against the on-disk labels:

  MISSING_LABEL : model predicts a class at high conf, no GT box overlaps
                  (IoU<0.3 with any same-class GT) → likely an unlabeled
                  instance (teacher missed it back then).
  PHANTOM_LABEL : GT box has NO prediction of its class at conf>=0.10
                  anywhere near (IoU<0.3) → either the model's blind spot OR
                  a wrong/stale label. Ranked by how strong the model is on
                  that class elsewhere.
  CLASS_SWAP    : GT box overlaps (IoU>=0.5) a high-conf prediction of a
                  DIFFERENT class → the red/yellow-dot disease pattern.

Output: data/_audit_train_labels.csv (one row per finding, ranked) +
console per-class summary. Human review via the dashboard queue.

GPU: batched predict @960. CPU: label parsing in the main loop (cheap).
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, r"D:\Project\ai game secretary")
sys.stdout.reconfigure(encoding="utf-8")

DS = Path(r"D:\Project\ml_cache\models\yolo\dataset\ui_v2")
RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"
OUT = Path(rf"D:\Project\ai game secretary\data\_audit_{SPLIT}_labels.csv")
WEIGHTS = sys.argv[2] if len(sys.argv) > 2 else \
    r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v8b\weights\best_real.pt"

MASTER = [l.strip() for l in open(RAW / "_classes.txt", encoding="utf-8") if l.strip()]
NAME2M = {n: i for i, n in enumerate(MASTER)}

HI_CONF = 0.65      # MISSING_LABEL threshold
SWAP_CONF = 0.55    # CLASS_SWAP threshold


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def main():
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    n2m = {i: NAME2M[n] for i, n in model.names.items() if n in NAME2M}

    img_dir, lbl_dir = DS / "images" / SPLIT, DS / "labels" / SPLIT
    jpgs = sorted(img_dir.glob("*.jpg"))
    print(f"auditing {len(jpgs)} {SPLIT} frames with {Path(WEIGHTS).parent.parent.name}")

    findings = []
    counts = Counter()
    B = 24
    for s in range(0, len(jpgs), B):
        chunk = jpgs[s:s + B]
        results = model.predict([str(p) for p in chunk], conf=0.10, imgsz=960,
                                verbose=False)
        for jpg, res in zip(chunk, results):
            lbl = lbl_dir / (jpg.stem + ".txt")
            gt = []
            if lbl.exists():
                for ln in lbl.read_text(encoding="utf-8").splitlines():
                    p = ln.split()
                    if len(p) != 5:
                        continue
                    c = int(p[0])
                    xc, yc, w, h = map(float, p[1:])
                    gt.append((c, (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)))
            H, W = res.orig_shape
            preds = []
            for b in res.boxes:
                mi = n2m.get(int(b.cls[0]))
                if mi is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                preds.append((mi, float(b.conf[0]),
                              (x1 / W, y1 / H, x2 / W, y2 / H)))

            # MISSING_LABEL: high-conf pred with no same-class GT overlap
            for mi, conf, pb in preds:
                if conf < HI_CONF:
                    continue
                if not any(c == mi and iou(pb, gb) >= 0.3 for c, gb in gt):
                    findings.append(("MISSING_LABEL", jpg.name, MASTER[mi],
                                     round(conf, 2), round((pb[0]+pb[2])/2, 3),
                                     round((pb[1]+pb[3])/2, 3)))
                    counts[("MISSING", MASTER[mi])] += 1

            # PHANTOM_LABEL + CLASS_SWAP
            for c, gb in gt:
                same = [p for p in preds if p[0] == c and iou(p[2], gb) >= 0.3]
                if same:
                    continue
                swap = [p for p in preds
                        if p[0] != c and p[1] >= SWAP_CONF and iou(p[2], gb) >= 0.5]
                if swap:
                    other = max(swap, key=lambda p: p[1])
                    findings.append(("CLASS_SWAP", jpg.name, MASTER[c],
                                     round(other[1], 2),
                                     round((gb[0]+gb[2])/2, 3),
                                     round((gb[1]+gb[3])/2, 3),
                                     MASTER[other[0]]))
                    counts[("SWAP", f"{MASTER[c]}->{MASTER[other[0]]}")] += 1
                else:
                    findings.append(("PHANTOM_LABEL", jpg.name, MASTER[c], 0.0,
                                     round((gb[0]+gb[2])/2, 3),
                                     round((gb[1]+gb[3])/2, 3)))
                    counts[("PHANTOM", MASTER[c])] += 1
        if (s // B) % 20 == 0:
            print(f"  {s + len(chunk)}/{len(jpgs)} frames, {len(findings)} findings",
                  flush=True)

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "frame", "gt_or_pred_cls", "conf", "cx", "cy", "swap_to"])
        for row in findings:
            w.writerow(list(row) + [""] * (7 - len(row)))
    print(f"\n[done] {len(findings)} findings → {OUT}")

    print("\n== top disagreement classes ==")
    for (kind, cls), n in counts.most_common(40):
        print(f"  {kind:8s} {cls}: {n}")


if __name__ == "__main__":
    main()
