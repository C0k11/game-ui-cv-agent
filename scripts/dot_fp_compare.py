# -*- coding: utf-8 -*-
"""Empirical position-prior test: run v9 vs v10 on the flywheel _clean frames
(lobby/menu screens = the grey/blue placeholder habitat where v9 fired false
红点/黄点). The val set can't measure this (FPs land on un-labelled grey badges).

For every frame we collect each model's 红点/黄点 detections (conf>=0.20, the
prefill floor) and bucket by conf:
  prior-zone   0.20-0.55  (v9's false position-prior firings live here)
  strong-zone  >=0.55     (real dots, but high-conf pale-blue FPs hide here too)
Then we match v9->v10 dots by IoU>0.3 and count:
  suppressed   v9 fired, v10 silent at that spot   (FP v10 fixed — GOOD)
  persistent   both fired                          (kept — real OR shared FP)
  new          v10 fired where v9 silent           (v10-only)
A montage of suppressed (top) and persistent-in-prior-zone (bottom) is saved
for eyeball — numbers alone can't tell a fixed-FP from a lost-real-dot.

Run: py scripts/dot_fp_compare.py
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
V9 = r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v9\weights\best.pt"
V10 = r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v10\weights\best.pt"
OUT = Path(r"D:\Project\ai game secretary\data\_dot_fp_audit")
DOT_NAMES = {"红点", "黄点"}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def dots(model, paths, names):
    """path -> list of (x1,y1,x2,y2,conf) for 红点/黄点."""
    out = {}
    B = 24
    dotset = {i for i, n in names.items() if n in DOT_NAMES}
    for s in range(0, len(paths), B):
        chunk = paths[s:s + B]
        res = model.predict(chunk, conf=0.20, imgsz=960, verbose=False)
        for p, r in zip(chunk, res):
            ds = []
            for bx in r.boxes:
                if int(bx.cls[0]) in dotset:
                    x1, y1, x2, y2 = [float(v) for v in bx.xyxy[0]]
                    ds.append((x1, y1, x2, y2, float(bx.conf[0])))
            out[p] = ds
    return out


def main():
    pools = sorted(p for p in RAW.iterdir()
                   if p.is_dir() and p.name.endswith("_clean")
                   and ("20260612" in p.name or "20260613" in p.name))
    frames = []
    for pool in pools:
        frames += [str(j) for j in sorted(pool.glob("*.jpg"))]
    print(f"{len(frames)} flywheel _clean frames across {len(pools)} pools",
          flush=True)
    if not frames:
        return

    from ultralytics import YOLO
    m9, m10 = YOLO(V9), YOLO(V10)
    print("running v9 …", flush=True)
    d9 = dots(m9, frames, m9.names)
    print("running v10 …", flush=True)
    d10 = dots(m10, frames, m10.names)

    def band(ds, lo, hi):
        return sum(1 for d in ds if lo <= d[4] < hi)

    tot9 = sum(len(v) for v in d9.values())
    tot10 = sum(len(v) for v in d10.values())
    p9 = sum(band(v, 0.20, 0.55) for v in d9.values())
    p10 = sum(band(v, 0.20, 0.55) for v in d10.values())
    s9 = sum(band(v, 0.55, 1.01) for v in d9.values())
    s10 = sum(band(v, 0.55, 1.01) for v in d10.values())

    suppressed = persistent = newd = 0
    supp_samples, pers_prior_samples = [], []
    for f in frames:
        a, b = d9.get(f, []), d10.get(f, [])
        bm = [False] * len(b)
        for da in a:
            hit = -1
            for j, db in enumerate(b):
                if not bm[j] and iou(da[:4], db[:4]) > 0.3:
                    hit = j
                    break
            if hit >= 0:
                bm[hit] = True
                persistent += 1
                if da[4] < 0.55 and len(pers_prior_samples) < 40:
                    pers_prior_samples.append((f, da))
            else:
                suppressed += 1
                if len(supp_samples) < 40:
                    supp_samples.append((f, da))
        newd += bm.count(False)

    print("\n================ DOT FIRINGS on flywheel frames ================")
    print(f"{'':<16}{'v9':>10}{'v10':>10}")
    print(f"{'total dots':<16}{tot9:>10}{tot10:>10}")
    print(f"{'prior 0.20-0.55':<16}{p9:>10}{p10:>10}   <- position-prior zone")
    print(f"{'strong >=0.55':<16}{s9:>10}{s10:>10}")
    print(f"\nmatched v9->v10 (IoU>0.3):")
    print(f"  suppressed (v9 fired, v10 silent): {suppressed}  <- FPs v10 dropped")
    print(f"  persistent (both fired)          : {persistent}")
    print(f"     of which prior-zone (<0.55)   : {len(pers_prior_samples)}+ "
          f"(capped sample)")
    print(f"  v10-only new                     : {newd}")

    # montage for eyeball
    import cv2
    import numpy as np
    OUT.mkdir(parents=True, exist_ok=True)

    def crop(f, d, pad=40):
        img = cv2.imread(f)
        if img is None:
            return None
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in d[:4]]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        c = img[y1:y2, x1:x2]
        if c.size == 0:
            return None
        c = cv2.resize(c, (160, 160))
        cv2.putText(c, f"{d[4]:.2f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
        return c

    def grid(samples, name):
        tiles = [t for t in (crop(f, d) for f, d in samples) if t is not None]
        if not tiles:
            print(f"  ({name}: no tiles)")
            return
        cols = 8
        rows = (len(tiles) + cols - 1) // cols
        canvas = np.zeros((rows * 160, cols * 160, 3), np.uint8)
        for i, t in enumerate(tiles):
            r, c = divmod(i, cols)
            canvas[r * 160:(r + 1) * 160, c * 160:(c + 1) * 160] = t
        cv2.imwrite(str(OUT / name), canvas)
        print(f"  wrote {OUT / name} ({len(tiles)} tiles)")

    print("\nmontages:")
    grid(supp_samples, "suppressed_by_v10.jpg")
    grid(pers_prior_samples, "persistent_prior_zone.jpg")


if __name__ == "__main__":
    main()
