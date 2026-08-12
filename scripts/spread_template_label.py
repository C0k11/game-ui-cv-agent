# -*- coding: utf-8 -*-
"""通用模板扩散器: 把一个已定稿的框模板扫遍飞轮池, 产出候选标注 + 自检 sheet.

用途: 某个 cls 判据已在样板帧上人审定稿(几何+假阳排除), 要把它批量铺到
成千上万张零标注帧上。模板匹配本身就是检测器 — 不需要先用 cls 指纹捞场景
(指纹选宽会捞错画面, 选窄会漏, 2026-08-02 实测教训)。

纪律(硬编码在流程里, 别绕过):
  - 遮罩帧不标: 帧内出现 118取消键/397获得奖励/142点击继续 任一  整帧跳过
    (dim 对话框 / 獲得獎勵 overlay / tooltip 压暗, 三形态都不是该 cls 的真态)
  - 已有同 cls 标注附近的命中不重复写 (中心距 < dedup_r)
  - 默认 --dry: 只出 CSV + contact sheet 给人审, --apply 才落 txt
  - 落盘一律**数值键**去重比对, 绝不用字符串比对 (dataset build 会截行尾零,
    字符串比对会把同一个框判成缺失  灌重复行, 2026-08-02 自伤过一次)

Usage:
  # Bonus(110): 模板取自 flywheel_20260801/frame_000245 最終行那个框
  py -X utf8 scripts/spread_template_label.py --cls 110 \
      --tpl-frame flywheel_20260801/frame_000245.jpg \
      --tpl-box 0.369 0.6338 0.044 0.0267 --thr 0.78
  py -X utf8 scripts/spread_template_label.py ... --apply
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw_images"
MANIFEST = ROOT / "data" / "_flywheel_prelabel_manifest.csv"
MASK_CLS = {118, 397, 142}          # 取消键 / 获得奖励 / 点击继续 = 遮罩帧信号
DIM_V = 100.0                        # 全帧 V 低于此 = 被压暗遮罩层盖住, 整帧跳过

_TPL = None                          # per-worker 模板缓存


def _init(tpl_bytes: bytes, shape: tuple) -> None:
    global _TPL
    _TPL = np.frombuffer(tpl_bytes, dtype=np.uint8).reshape(shape)


def _scan(task) -> tuple:
    """单帧扫描: 返回 (frame, verdict, hits)。hits=[(cx,cy,score)]。"""
    rel, cls_id, thr, dedup_r = task
    txt = (RAW / rel).with_suffix(".txt")
    # 全池模式下大量帧没有预标 txt —— 无 txt  无 cls 可查, 遮罩闸退化为
    # 只靠亮度判据(下方 DIM_V), 这是可接受的降级(宁可多出几张给人审)。
    lines = ([l.split() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
             if txt.exists() else [])
    clss = {int(p[0]) for p in lines}
    if MASK_CLS & clss:
        return rel, f"mask_skip{sorted(MASK_CLS & clss)}", []
    img = cv2.imread(str(RAW / rel))
    if img is None:
        return rel, "no_image", []
    # 压暗遮罩层闸: 道具说明 tooltip / 引导层会把整帧压暗, 而它们**没有**
    # 118/397/142 任何一个 cls 可查 (2026-08-02 Bonus 扩散实测漏网 5 帧,
    # 正是 task#32「tooltip 压暗致感知失明」的本体)。全帧 V 双峰断层干净:
    # 压暗帧 57-62 / 正常帧 ≥138, 阈值 100 落在断层中间。
    if float(cv2.cvtColor(cv2.resize(img, (480, 270)), cv2.COLOR_BGR2HSV)[:, :, 2].mean()) < DIM_V:
        return rel, "dim_overlay_skip", []
    H, W = img.shape[:2]
    th, tw = _TPL.shape[:2]
    if H < th or W < tw:
        return rel, "frame_too_small", []
    res = cv2.matchTemplate(img, _TPL, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= thr)
    if len(xs) == 0:
        return rel, "no_hit", []
    # 分数降序 + 简易 NMS (半个模板尺寸内算同一个)
    order = np.argsort(-res[ys, xs])
    kept = []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if all(abs(x - kx) > tw // 2 or abs(y - ky) > th // 2 for kx, ky, _ in kept):
            kept.append((x, y, float(res[y, x])))
    have = [(float(p[1]), float(p[2])) for p in lines if int(p[0]) == cls_id]
    hits = []
    for (x, y, s) in kept:
        cx, cy = (x + tw / 2) / W, (y + th / 2) / H
        if any(abs(cx - hx) < dedup_r and abs(cy - hy) < dedup_r for hx, hy in have):
            continue                 # 已标过, 不重复
        hits.append((cx, cy, s))
    return rel, ("hit" if hits else "dup_or_none"), hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", type=int, required=True)
    ap.add_argument("--tpl-frame", required=True, help="模板帧, 相对 data/raw_images/")
    ap.add_argument("--tpl-box", nargs=4, type=float, required=True, metavar=("CX", "CY", "W", "H"))
    ap.add_argument("--thr", type=float, default=0.78)
    ap.add_argument("--dedup-r", type=float, default=0.010)
    ap.add_argument("--tag", default="", help="输出文件名后缀")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--scan-all", action="store_true",
                    help="扫全飞轮池(27K帧)而不是仅预标过的; 稀有场景必须用这个")
    ap.add_argument("--extra-dir", action="append", default=[],
                    help="额外扫描的 raw_images 子目录(可多次)")
    args = ap.parse_args()

    names = [c.strip() for c in (RAW / "_classes.txt").read_text(encoding="utf-8").splitlines() if c.strip()]
    cls_name = names[args.cls]

    timg = cv2.imread(str(RAW / args.tpl_frame))
    TH, TW = timg.shape[:2]
    cx, cy, bw, bh = args.tpl_box
    tpl = np.ascontiguousarray(timg[int((cy - bh / 2) * TH):int((cy + bh / 2) * TH),
                                    int((cx - bw / 2) * TW):int((cx + bw / 2) * TW)])
    print(f"cls {args.cls} 「{cls_name}」  template {tpl.shape} from {args.tpl_frame}")

    # 战斗域帧不碰 (UI cls 模板扫战斗帧 = 假阳污染 battle 侧素材)
    sys.path.insert(0, str(ROOT))
    from scripts.prelabel_flywheel_inplace import BATTLE_MARKERS, _flywheel_dirs
    if args.scan_all:
        # 全池模式: 模板匹配**不依赖预标**, 不必局限在 dedup 后那 7,057 帧。
        # 稀有场景(craft「材料不足」这种一天只出现几秒的)恰恰是被 dedup 当
        # "近重复"砍掉、或压根没预标的那两万帧里才有量 —— 2026-08-02 实测:
        # 预标池里含 craft 開始製造键的只有 11 帧, 全池才是真正的存量。
        all_frames = []
        for d in _flywheel_dirs():
            all_frames += [str(p.relative_to(RAW)) for p in d.glob("*.jpg")]
        for extra in (args.extra_dir or []):
            all_frames += [str(p.relative_to(RAW)) for p in (RAW / extra).glob("*.jpg")]
    else:
        all_frames = [r["frame"] for r in csv.DictReader(open(MANIFEST, encoding="utf-8"))]
    frames = [f for f in all_frames
              if not any(m in Path(f).parts[0].lower() for m in BATTLE_MARKERS)]
    if len(frames) != len(all_frames):
        print(f"battle-domain frames excluded: {len(all_frames) - len(frames)}")
    print(f"pool: {len(frames)} frames  thr={args.thr}")

    tasks = [(f, args.cls, args.thr, args.dedup_r) for f in frames]
    results = []
    with ProcessPoolExecutor(max_workers=24, initializer=_init,
                             initargs=(tpl.tobytes(), tpl.shape)) as ex:
        for i, out in enumerate(ex.map(_scan, tasks, chunksize=16)):
            results.append(out)
            if i % 1500 == 0:
                print(f"  {i}/{len(frames)}", flush=True)

    import collections
    verdicts = collections.Counter(v for _, v, _ in results)
    hits = [(f, h) for f, v, h in results if v == "hit"]
    n_box = sum(len(h) for _, h in hits)
    print(f"verdicts: {dict(verdicts)}")
    print(f"HIT frames: {len(hits)}  boxes: {n_box}")

    tag = args.tag or f"cls{args.cls}"
    out_csv = ROOT / "data" / f"_spread_{tag}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("frame", "cx", "cy", "score"))
        for fr, hs in hits:
            for (hx, hy, s) in hs:
                w.writerow((fr, f"{hx:.6f}", f"{hy:.6f}", f"{s:.4f}"))
    print(f"csv: {out_csv.relative_to(ROOT)}")

    # 分数分布 (定阈值的依据)
    all_s = sorted((s for _, hs in hits for *_, s in hs), reverse=True)
    if all_s:
        a = np.array(all_s)
        print(f"score: max={a.max():.3f} p90={np.percentile(a,90):.3f} "
              f"p50={np.percentile(a,50):.3f} p10={np.percentile(a,10):.3f} min={a.min():.3f}")

    # contact sheet: 命中区域裁切, 每页 24 格
    CELL, COLS, PER = 620, 6, 24
    tiles = []
    for fr, hs in hits:
        img = cv2.imread(str(RAW / fr))
        H, W = img.shape[:2]
        for (hx, hy, s) in hs:
            x1, y1 = int((hx - bw * 2.2) * W), int((hy - bh * 2.2) * H)
            x2, y2 = int((hx + bw * 4.5) * W), int((hy + bh * 2.2) * H)
            crop = img[max(0, y1):y2, max(0, x1):x2].copy()
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            # 命中框在 crop 内的位置
            rx, ry = int((hx * W - max(0, x1)) - bw * W / 2), int((hy * H - max(0, y1)) - bh * H / 2)
            cv2.rectangle(crop, (rx, ry), (rx + int(bw * W), ry + int(bh * H)), (0, 0, 255), 3)
            cv2.putText(crop, f"{s:.2f} {Path(fr).parts[0][-11:]}", (6, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            tiles.append(cv2.resize(crop, (CELL, int(ch * CELL / cw))))
    pages = []
    if tiles:
        rh = max(t.shape[0] for t in tiles)
        for p0 in range(0, len(tiles), PER):
            chunk = tiles[p0:p0 + PER]
            nr = (len(chunk) + COLS - 1) // COLS
            page = np.zeros((nr * rh, COLS * CELL, 3), np.uint8)
            for j, t in enumerate(chunk):
                r, c = divmod(j, COLS)
                page[r * rh:r * rh + t.shape[0], c * CELL:(c + 1) * CELL] = t
            fn = ROOT / "data" / f"_spread_{tag}_sheet{len(pages)}.jpg"
            cv2.imwrite(str(fn), page, [cv2.IMWRITE_JPEG_QUALITY, 86])
            pages.append(fn.name)
    print(f"sheets ({len(pages)}): {pages}")

    if args.apply:
        n = 0
        for fr, hs in hits:
            txt = (RAW / fr).with_suffix(".txt")
            lines = [l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
            keys = {(p[0],) + tuple(round(float(x), 5) for x in p[1:5])
                    for p in (l.split() for l in lines)}
            for (hx, hy, s) in hs:
                new = f"{args.cls} {hx:.6f} {hy:.6f} {bw:.6f} {bh:.6f}"
                k = (str(args.cls),) + tuple(round(float(x), 5) for x in new.split()[1:5])
                if k in keys:
                    continue
                lines.append(new)
                keys.add(k)
                n += 1
            txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"APPLIED {n} boxes to {len(hits)} frames")
    else:
        print("(dry run — 人审 sheet 后加 --apply 落盘)")


if __name__ == "__main__":
    main()
