# -*- coding: utf-8 -*-
"""Review freshly-labeled v9 pools (2026-06-11) — sanity + v8 disagreement.

Per pool: label format check (5-col, cls id in range), class histogram.
Then v8 disagreement mining (same thresholds as audit_train_labels.py):
  MISSING : v8 pred conf>=0.65, no same-class GT IoU>=0.3
  SWAP    : GT overlaps (IoU>=0.5) a different-class v8 pred conf>=0.55
PHANTOM is skipped for cls>=469 (new arena-shop classes v8 can't know) and
reported only as a count elsewhere — fresh hand labels, blind-spot expected.

Output: data/_review_v9_pools.csv + console summary.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
OUT = Path(r"D:\Project\ai game secretary\data\_review_v9_pools.csv")
WEIGHTS = r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v8b\weights\best_real.pt"

MASTER = [l.strip() for l in open(RAW / "_classes.txt", encoding="utf-8") if l.strip()]
POOLS = sorted(p for p in RAW.iterdir()
               if p.is_dir() and p.name.startswith("run_20260611"))

HI_CONF, SWAP_CONF = 0.65, 0.55
NEW_CLS_LO = 469


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
    n2m = {i: MASTER.index(n) for i, n in model.names.items() if n in MASTER}

    hist = Counter()
    fmt_bad = []
    findings = []
    phantom_n = 0

    for pool in POOLS:
        jpgs = sorted(pool.glob("*.jpg"))
        print(f"{pool.name}: {len(jpgs)} frames", flush=True)
        B = 24
        for s in range(0, len(jpgs), B):
            chunk = jpgs[s:s + B]
            results = model.predict([str(p) for p in chunk], conf=0.10,
                                    imgsz=960, verbose=False)
            for jpg, res in zip(chunk, results):
                txt = jpg.with_suffix(".txt")
                gt = []
                if txt.exists():
                    for ln in txt.read_text(encoding="utf-8").splitlines():
                        p = ln.split()
                        if not p:
                            continue
                        if len(p) != 5:
                            fmt_bad.append((pool.name, jpg.name, ln[:40]))
                            continue
                        c = int(p[0])
                        if c >= len(MASTER):
                            fmt_bad.append((pool.name, jpg.name, f"cls {c} OOR"))
                            continue
                        xc, yc, w, h = map(float, p[1:])
                        gt.append((c, (xc - w/2, yc - h/2, xc + w/2, yc + h/2)))
                        hist[c] += 1
                H, W = res.orig_shape
                preds = []
                for b in res.boxes:
                    mi = n2m.get(int(b.cls[0]))
                    if mi is None:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                    preds.append((mi, float(b.conf[0]),
                                  (x1/W, y1/H, x2/W, y2/H)))
                for mi, conf, pb in preds:
                    if conf < HI_CONF:
                        continue
                    if not any(c == mi and iou(pb, gb) >= 0.3 for c, gb in gt):
                        findings.append(("MISSING", pool.name, jpg.name,
                                         MASTER[mi], round(conf, 2),
                                         round((pb[0]+pb[2])/2, 3),
                                         round((pb[1]+pb[3])/2, 3), ""))
                for c, gb in gt:
                    same = [p for p in preds if p[0] == c and iou(p[2], gb) >= 0.3]
                    if same:
                        continue
                    swap = [p for p in preds
                            if p[0] != c and p[1] >= SWAP_CONF and iou(p[2], gb) >= 0.5]
                    if swap:
                        other = max(swap, key=lambda p: p[1])
                        findings.append(("SWAP", pool.name, jpg.name, MASTER[c],
                                         round(other[1], 2),
                                         round((gb[0]+gb[2])/2, 3),
                                         round((gb[1]+gb[3])/2, 3),
                                         MASTER[other[0]]))
                    elif c < NEW_CLS_LO:
                        phantom_n += 1

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "pool", "frame", "cls", "conf", "cx", "cy", "swap_to"])
        w.writerows(findings)

    print(f"\n[done] {len(findings)} findings → {OUT}  (old-cls phantom: {phantom_n})")
    if fmt_bad:
        print(f"\n!! format problems: {len(fmt_bad)}")
        for row in fmt_bad[:10]:
            print("  ", row)
    else:
        print("\nformat: all labels 5-col, cls ids in range ✓")

    print(f"\n== histogram: new/key classes ==")
    KEY = [469, 470, 471, 472, 473, 455, 456, 450, 402, 451, 452, 55, 109]
    for c in KEY:
        print(f"  {c:3d} {MASTER[c]}: {hist.get(c, 0)}")
    print(f"\n== top 25 labeled classes overall ==")
    for c, n in hist.most_common(25):
        print(f"  {c:3d} {MASTER[c]}: {n}")
    kinds = Counter(k for k, *_ in findings)
    swaps = Counter(f"{r[3]}->{r[7]}" for r in findings if r[0] == "SWAP")
    print(f"\nkinds: {dict(kinds)}")
    print("top swaps:", swaps.most_common(10))
    miss = Counter(r[3] for r in findings if r[0] == "MISSING")
    print("top missing:", miss.most_common(15))


if __name__ == "__main__":
    main()
