# -*- coding: utf-8 -*-
"""两段式状态对扩散器: 模板定位 + 像素特征判亮/灰, 一次同时补两个 cls。

为什么不能直接用 spread_template_label.py 扩散灰态(2026-08-02 实测定罪):
`TM_CCOEFF_NORMED` 归一化时**减去均值**, 正好把亮度差消掉只匹配形状 —
而「購買」二字亮态灰态形状完全一样。实测样板帧 frame_000366: 亮态 103
得分 0.9591, 比一半灰态(0.8623-0.8704)还高  单靠模板分**必然**把亮态
污染成灰态。所以状态对必须两段式:
   模板匹配只负责**定位**(召回该按钮所有实例, 不分状态)
   每个命中框量 (V, B-R) 双特征判状态, 落中间带的一律 unsure 不标

判据在 806 个人审定稿框上标定: 灰 V≤143 / 亮 V≥167 (中间 24 点空隙),
V<158 & B-R<83 二分 806/806 = 100%。**这组数值只对「購買」有效**,
换按钮必须重新标定 — 「全部選擇」的灰比亮更亮(白底钮去饱和), 方向是反的。

Usage:
  py -X utf8 scripts/spread_state_pair.py --bright-cls 103 --dark-cls 489 \
     --tpl-frame flywheel_20260801/frame_000362.jpg \
     --tpl-box 0.545 0.482 0.0305 0.0284 \
     --loc-thr 0.85 --dark-v 158 --dark-br 83 --bright-v 167 [--apply]
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_images"
MANIFEST = ROOT / "data" / "_flywheel_prelabel_manifest.csv"
MASK_CLS = {118, 397, 142}
DIM_V = 100.0

_TPL = None
_CFG = None


def _init(tpl_bytes, shape, cfg):
    global _TPL, _CFG
    _TPL = np.frombuffer(tpl_bytes, dtype=np.uint8).reshape(shape)
    _CFG = cfg


def _scan(rel):
    c = _CFG
    txt = (RAW / rel).with_suffix(".txt")
    lines = [l.split() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    clss = {int(p[0]) for p in lines}
    if MASK_CLS & clss:
        return rel, "mask_skip", []
    img = cv2.imread(str(RAW / rel))
    if img is None:
        return rel, "no_image", []
    if float(cv2.cvtColor(cv2.resize(img, (480, 270)), cv2.COLOR_BGR2HSV)[:, :, 2].mean()) < DIM_V:
        return rel, "dim_skip", []
    H, W = img.shape[:2]
    th, tw = _TPL.shape[:2]
    res = cv2.matchTemplate(img, _TPL, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= c["loc_thr"])
    if len(xs) == 0:
        return rel, "no_hit", []
    order = np.argsort(-res[ys, xs])
    kept = []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if all(abs(x - kx) > tw // 2 or abs(y - ky) > th // 2 for kx, ky, _ in kept):
            kept.append((x, y, float(res[y, x])))
    have = [(float(p[1]), float(p[2])) for p in lines
            if int(p[0]) in (c["bright"], c["dark"])]
    out = []
    for (x, y, s) in kept:
        cx, cy = (x + tw / 2) / W, (y + th / 2) / H
        if any(abs(cx - hx) < c["dedup_r"] and abs(cy - hy) < c["dedup_r"] for hx, hy in have):
            continue
        crop = img[y:y + th, x:x + tw]
        v = float(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2].mean())
        br = float(crop[:, :, 0].mean() - crop[:, :, 2].mean())
        if v < c["dark_v"] and br < c["dark_br"]:
            state = c["dark"]
        elif c["bright_v"] <= v <= c["bright_v_max"] and br >= c["bright_br"]:
            state = c["bright"]
        else:
            # 中间带 + **转场淡出**帧  unsure, 不标。淡出时按钮渐隐混白:
            # V 反而升高(203-208 > 正常亮态 197)而 B-R 掉下来(32-83 < 85),
            # 只设 V 下限会把它们当亮态收进来 (2026-08-02 sheet 抓到 3 例)。
            state = 0
        out.append((cx, cy, s, v, br, state))
    return rel, "hit", out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bright-cls", type=int, required=True)
    ap.add_argument("--dark-cls", type=int, required=True)
    ap.add_argument("--tpl-frame", required=True)
    ap.add_argument("--tpl-box", nargs=4, type=float, required=True)
    ap.add_argument("--loc-thr", type=float, default=0.85)
    ap.add_argument("--dark-v", type=float, default=158)
    ap.add_argument("--dark-br", type=float, default=83)
    ap.add_argument("--bright-v", type=float, default=165)
    ap.add_argument("--bright-v-max", type=float, default=200)
    ap.add_argument("--bright-br", type=float, default=85)
    ap.add_argument("--dedup-r", type=float, default=0.010)
    ap.add_argument("--tag", default="statepair")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    names = [c.strip() for c in (RAW / "_classes.txt").read_text(encoding="utf-8").splitlines() if c.strip()]
    timg = cv2.imread(str(RAW / args.tpl_frame))
    TH, TW = timg.shape[:2]
    cx, cy, bw, bh = args.tpl_box
    tpl = np.ascontiguousarray(timg[int((cy - bh / 2) * TH):int((cy + bh / 2) * TH),
                                    int((cx - bw / 2) * TW):int((cx + bw / 2) * TW)])
    print(f"亮 {args.bright_cls}「{names[args.bright_cls]}」 / 灰 {args.dark_cls}「{names[args.dark_cls]}」"
          f"  tpl {tpl.shape}")

    sys.path.insert(0, str(ROOT))
    from scripts.prelabel_flywheel_inplace import BATTLE_MARKERS
    frames = [r["frame"] for r in csv.DictReader(open(MANIFEST, encoding="utf-8"))
              if not any(m in Path(r["frame"]).parts[0].lower() for m in BATTLE_MARKERS)]
    print(f"pool {len(frames)}  loc_thr={args.loc_thr}  "
          f"dark:V<{args.dark_v}&BR<{args.dark_br}  "
          f"bright:V∈[{args.bright_v},{args.bright_v_max}]&BR>={args.bright_br}")

    cfg = dict(loc_thr=args.loc_thr, dark_v=args.dark_v, dark_br=args.dark_br,
               bright_v=args.bright_v, bright_v_max=args.bright_v_max,
               bright_br=args.bright_br, dedup_r=args.dedup_r,
               bright=args.bright_cls, dark=args.dark_cls)
    results = []
    with ProcessPoolExecutor(max_workers=24, initializer=_init,
                             initargs=(tpl.tobytes(), tpl.shape, cfg)) as ex:
        for i, out in enumerate(ex.map(_scan, frames, chunksize=16)):
            results.append(out)
            if i % 1500 == 0:
                print(f"  {i}/{len(frames)}", flush=True)

    verd = collections.Counter(v for _, v, _ in results)
    hits = [(f, h) for f, v, h in results if v == "hit" and h]
    cnt = collections.Counter(st for _, hs in hits for *_, st in hs)
    print(f"verdicts: {dict(verd)}")
    print(f"命中帧 {len(hits)}    亮{args.bright_cls}: {cnt[args.bright_cls]}  "
          f"灰{args.dark_cls}: {cnt[args.dark_cls]}  unsure(不标): {cnt[0]}")

    with open(ROOT / "data" / f"_spread_{args.tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("frame", "cx", "cy", "score", "V", "BR", "cls"))
        for fr, hs in hits:
            for (hx, hy, s, v, br, st) in hs:
                w.writerow((fr, f"{hx:.6f}", f"{hy:.6f}", f"{s:.4f}", f"{v:.1f}", f"{br:.1f}", st))

    # sheet: 灰/亮/unsure 各自采样, 边框颜色区分
    COLOR = {args.dark_cls: (0, 0, 255), args.bright_cls: (0, 200, 0), 0: (0, 220, 255)}
    CELL, COLS, PER = 460, 7, 28
    buckets = collections.defaultdict(list)
    for fr, hs in hits:
        for h in hs:
            buckets[h[5]].append((fr, h))
    pages = []
    for st in (args.dark_cls, args.bright_cls, 0):
        items = buckets.get(st, [])
        if not items:
            continue
        step = max(1, len(items) // (PER * 2))
        picks = items[::step][:PER * 2]
        tiles = []
        for fr, (hx, hy, s, v, br, _) in picks:
            img = cv2.imread(str(RAW / fr))
            H, W = img.shape[:2]
            x1, y1 = int((hx - bw * 1.5) * W), int((hy - bh * 2.2) * H)
            x2, y2 = int((hx + bw * 1.5) * W), int((hy + bh * 2.2) * H)
            crop = img[max(0, y1):y2, max(0, x1):x2].copy()
            if crop.size == 0:
                continue
            rx = int(hx * W - bw * W / 2) - max(0, x1)
            ry = int(hy * H - bh * H / 2) - max(0, y1)
            cv2.rectangle(crop, (rx, ry), (rx + int(bw * W), ry + int(bh * H)), COLOR[st], 3)
            cv2.putText(crop, f"V{v:.0f} B{br:.0f}", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR[st], 2)
            tiles.append(cv2.resize(crop, (CELL, int(crop.shape[0] * CELL / crop.shape[1]))))
        if not tiles:
            continue
        rh = max(t.shape[0] for t in tiles)
        nr = (len(tiles) + COLS - 1) // COLS
        page = np.zeros((nr * rh, COLS * CELL, 3), np.uint8)
        for j, t in enumerate(tiles):
            r, c = divmod(j, COLS)
            page[r * rh:r * rh + t.shape[0], c * CELL:(c + 1) * CELL] = t
        lbl = {args.dark_cls: "dark", args.bright_cls: "bright", 0: "unsure"}[st]
        fn = ROOT / "data" / f"_spread_{args.tag}_{lbl}.jpg"
        cv2.imwrite(str(fn), page, [cv2.IMWRITE_JPEG_QUALITY, 86])
        pages.append(f"{fn.name}({len(tiles)}/{len(items)})")
    print("sheets:", pages)

    if args.apply:
        n = collections.Counter()
        for fr, hs in hits:
            txt = (RAW / fr).with_suffix(".txt")
            lines = [l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
            keys = {(p[0],) + tuple(round(float(x), 5) for x in p[1:5])
                    for p in (l.split() for l in lines)}
            for (hx, hy, s, v, br, st) in hs:
                if st == 0:
                    continue
                k = (str(st), round(hx, 5), round(hy, 5), round(bw, 5), round(bh, 5))
                if k in keys:
                    continue
                lines.append(f"{st} {hx:.6f} {hy:.6f} {bw:.6f} {bh:.6f}")
                keys.add(k)
                n[st] += 1
            txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"APPLIED {dict(n)}")
    else:
        print("(dry — 审 sheet 后加 --apply)")


if __name__ == "__main__":
    main()
