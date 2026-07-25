# -*- coding: utf-8 -*-
"""检测帧里有没有**烧录的 YOLO overlay 框** —— 素材进标注池前的硬闸。

## 背景
`server/app.py:2330` 记着一条 2026-06-05 的纪律: trajectory 帧因 overlay 烧录
风险**已退役, 不再进 label 队列**(2026-05-28 删过 11 个烧录 run)。
但那条纪律是 **DXcam 当主力帧源**时定的。现在主 tick 帧源已改成
**scrcpy → ADB**(server/app.py:1255-1275), 两者都是 **Android 内部取流**,
Windows 桌面层的 overlay **物理上进不去**。只有 fallback 链的
`hf`(可能是 DXcam) 和 `bitblt`(窗口抓取) 仍会烧。
⇒ 前提变了, 但**保证不了 100%** —— 所以不能靠"目检两张就放行", 要逐帧检测。

## 判据(来自 overlay 自己的绘制代码, 不是猜)
`scripts/yolo_overlay.py:138-149` 的调色板是
    colorsys.hsv_to_rgb(h/360, **S=0.9, V=1.0**)
`:369` 用 `CreatePen(PS_SOLID, 3, color)` 画 **3px 实线**矩形, `:407` 同色写标签。
⇒ 特征 = **高饱和(S≈0.9)高亮度(V≈1.0)的轴对齐细直线**。
游戏 UI 的边框普遍带抗锯齿/渐变/低饱和, 极少出现成片的这种线。

⛔**已实测的局限(2026-07-25, 必读)**: 这个判据在**战斗帧上误报严重**。
实跑 176 帧 val 候选, score 最高的 147 分那张目检**完全干净** —— 战斗画面里
本来就全是高饱和细线: 角色血条、buff 图标条、COST 分段条、AUTO 黄按钮、
冰面条纹。⇒ **本工具只能当粗筛/告警, 不能当准入闸。**
真正可靠的帧源闸是**按图像尺寸认帧源**(见 build_val_gap_queue.SAFE_FRAME_SIZES):
  3840x2160=ADB / 2560x1440=scrcpy → Android 内部取流, overlay 物理进不去
  3612x2033 / 2364x1331 → 窗口抓取, 有风险
尺寸是硬证据, 像素统计是启发式 —— 优先用前者。

跑:
    py scripts/detect_overlay_burn.py <dir_or_glob> [--thr 2] [--jobs 14]
    py scripts/detect_overlay_burn.py --selftest      # 合成正样本自检
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

# overlay 画笔: S=0.9 V=1.0。留容差吃 JPG 压缩带来的漂移。
_S_LO, _S_HI = 0.72, 1.00
_V_LO = 0.80
_LINE_MIN = 45      # 轴对齐线段最短像素(短于此的当噪声)
_THICK_MAX = 6      # overlay 笔宽 3px, 留余量


def burn_score(img) -> tuple:
    """返回 (score, 水平线段数, 垂直线段数)。score = 线段总数。"""
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    mask = ((s >= _S_LO) & (s <= _S_HI) & (v >= _V_LO)).astype(np.uint8)
    if mask.sum() == 0:
        return 0, 0, 0
    # 只保留"细"的结构: 原 mask 减去粗块(大核开运算的结果) → 去掉大色块
    thick = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT,
                                  (_THICK_MAX * 3, _THICK_MAX * 3)))
    thin = cv2.subtract(mask, thick)
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (_LINE_MIN, 1))
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, _LINE_MIN))
    hor = cv2.morphologyEx(thin, cv2.MORPH_OPEN, h_k)
    ver = cv2.morphologyEx(thin, cv2.MORPH_OPEN, v_k)
    n_h = cv2.connectedComponents(hor)[0] - 1
    n_v = cv2.connectedComponents(ver)[0] - 1
    return n_h + n_v, n_h, n_v


def scan_one(path: str):
    import cv2
    img = cv2.imread(path)
    if img is None:
        return (path, -1, 0, 0)
    sc, h, v = burn_score(img)
    return (path, sc, h, v)


def selftest() -> int:
    """拿一张真实干净帧, 合成 overlay 画上去, 验证检测器**真能逮到**。
    没有正样本的检测器等于没验证过(2026-05-28 那批烧录 run 已被删,
    手上没有天然正样本)。"""
    import colorsys
    import cv2
    import numpy as np
    cands = sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data/raw_images/_val_v15_gap/*.jpg")))
    if not cands:
        print("selftest: 找不到样本帧")
        return 1
    src = cands[len(cands) // 2]
    img = cv2.imread(src)
    if img is None:
        print("selftest: 读图失败")
        return 1
    base, _, _ = burn_score(img)
    fake = img.copy()
    h, w = fake.shape[:2]
    rng = np.random.default_rng(7)
    for k in range(8):                      # 画 8 个 overlay 风格的框
        hh = (k * 47) % 360
        r, g, b = colorsys.hsv_to_rgb(hh / 360.0, 0.9, 1.0)
        col = (int(b * 255), int(g * 255), int(r * 255))
        x1 = int(rng.integers(20, w - 320))
        y1 = int(rng.integers(20, h - 220))
        cv2.rectangle(fake, (x1, y1), (x1 + 300, y1 + 200), col, 3)
    burned, bh, bv = burn_score(fake)
    out = os.path.join(os.environ.get("TEMP", "."), "_overlay_selftest.jpg")
    cv2.imwrite(out, fake)
    print(f"selftest 源帧: {os.path.basename(src)}")
    print(f"  干净帧 score = {base}")
    print(f"  合成 8 个 overlay 框后 score = {burned} (H {bh} / V {bv})")
    ok = burned >= base + 8
    print(f"  → 检测器{'✅有效' if ok else '⛔失灵'}"
          f" (合成正样本已存 {out}, 可目检)")
    return 0 if ok else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="")
    ap.add_argument("--thr", type=int, default=2,
                    help="score ≥ 此值判为疑似烧录(默认 2)")
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.target:
        ap.error("需要 <dir_or_glob> 或 --selftest")

    files = (sorted(glob.glob(os.path.join(a.target, "**", "*.jpg"),
                              recursive=True))
             if os.path.isdir(a.target) else sorted(glob.glob(a.target)))
    if not files:
        print("没有 jpg")
        return 1
    print(f"扫描 {len(files)} 帧 ...")
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(scan_one, files, chunksize=8):
            rows.append(r)
    bad = [r for r in rows if r[1] >= a.thr]
    unread = [r for r in rows if r[1] < 0]
    scores = sorted(r[1] for r in rows if r[1] >= 0)
    print(f"  读不出 {len(unread)}")
    if scores:
        print(f"  score: min={scores[0]} 中位={scores[len(scores)//2]} "
              f"p95={scores[int(len(scores)*0.95)]} max={scores[-1]}")
    print(f"  ⛔疑似烧录(score ≥ {a.thr}): **{len(bad)}** / {len(rows)}")
    for p, sc, h, v in sorted(bad, key=lambda r: -r[1])[:15]:
        print(f"     {os.path.basename(p):48} score {sc} (H{h}/V{v})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
