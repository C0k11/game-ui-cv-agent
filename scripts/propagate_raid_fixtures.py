# -*- coding: utf-8 -*-
"""总力战固定物传播标注 (2026-07-10, 用户 spec: "两个柱子和塞特都是固定的").

制約解除決戰(賽特)镜头固定 → 三个固定目标用固定框批量标注:
  478 塞特的愤怒 (用户示范 7 帧的中位框)
  479 柱子 ×2 (新类, 带 HP 条的机制柱, 手工目检定框)

只适用总力战池(110430/110759)。104427/104718 是綜合戰術考試(体育场), 无此三物。

⚠ 2026-07-10 用户改判: 柱子不建新类, 就用 476「我方」标(友方保护目标),
坐标抄用户 frame_000009 示范框; master「柱子」行已删(479 行)。本脚本的
PILLAR_IDX/框已同步 — 重跑前注意 pass0 的幂等清理按坐标匹配柱框。

存在性判据(不标暂停画面/转场/开场/死亡, 全部实测校准 2026-07-10):
  ① cls 闸: 帧含 弹窗叉叉19/确认键20/加载中22/取消键118/暂停菜单131-133/
     胜利136/获得奖励397 → 三固定物全不标(暂停/结算/加载画面)
  ② 时序内插(v2): 严 HSV 血条闸只当"确定活"信号(boss 顶栏青条/柱子黄绿条,
     严阈值零误报); 目标死了不复活 → 首个活帧~最后活帧区间内非 skip 帧全标,
     区间外(开场天空/死亡后结算)全不标。EX 演出暗滤镜帧(血条被压暗/遮挡,
     颜色闸漏检 — frame_000040 实锤)夹在活帧之间, 由内插覆盖。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
POOLS = ["run_20260710_110430", "run_20260710_110759"]
ALL_TODAY = POOLS + ["run_20260710_104427", "run_20260710_104718"]

SKIP_CLS = {19, 20, 22, 118, 131, 132, 133, 136, 397}
SETH_IDX, PILLAR_IDX = 478, 476                       # 柱子=我方(用户改判)
SETH_BOX = (0.6463, 0.3607, 0.3515, 0.4267)          # 用户 7 帧中位
PILLARS = [(0.1532, 0.5056, 0.0631, 0.2718),          # A 左中(用户 frame_000009)
           (0.4779, 0.7845, 0.0673, 0.2958)]          # B 中下(用户 frame_000009)
BOSS_HP_R = (0.33, 0.028, 0.60, 0.048)
CYAN = (np.array((80, 80, 120)), np.array((105, 255, 255)))
PA_HP_R = (0.120, 0.372, 0.156, 0.393)
PB_HP_R = (0.455, 0.637, 0.501, 0.651)
GREEN = (np.array((30, 80, 120)), np.array((90, 255, 255)))


def color_frac(img, region, lo_hi):
    h, w = img.shape[:2]
    x1, y1 = int(region[0] * w), int(region[1] * h)
    x2, y2 = int(region[2] * w), int(region[3] * h)
    crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    return cv2.inRange(crop, lo_hi[0], lo_hi[1]).mean() / 255.0


def main():
    master = [l.strip() for l in
              open(RAW / "_classes.txt", encoding="utf-8") if l.strip()]
    assert master[SETH_IDX] == "塞特的愤怒" and master[PILLAR_IDX] == "我方"

    for name in ALL_TODAY:   # classes.txt 统一刷成最新 master
        (RAW / name / "classes.txt").write_text(
            "\n".join(master) + "\n", encoding="utf-8")

    USER_SETH_FRAMES = {f"frame_{i:06d}" for i in range(8, 15)}  # 用户示范, 保留

    def matches_pillar(l, boxes):
        p = l.split()
        if int(p[0]) != PILLAR_IDX:
            return False
        cx, cy = float(p[1]), float(p[2])
        return any(abs(cx - b[0]) < 0.02 and abs(cy - b[1]) < 0.02
                   for b in boxes)

    def is_pillar_box(l):
        return matches_pillar(l, PILLARS)

    for name in POOLS:
        pool = RAW / name
        txts = sorted(pool.glob("frame_*.txt"))
        # pass 0: 清掉本脚本旧输出(478 除用户示范帧 / 柱框按坐标匹配), 幂等重入
        for txt in txts:
            lines = [l for l in txt.read_text(encoding="utf-8").splitlines()
                     if l.strip()]
            keep = []
            for l in lines:
                c = int(l.split()[0])
                if is_pillar_box(l):        # 柱子=476, 只按坐标删, 不伤小人
                    continue
                if c == SETH_IDX and txt.stem not in USER_SETH_FRAMES:
                    continue
                if c == 477:            # 总力战无普通敌, 残留敌方=塞特误检
                    continue
                keep.append(l)
            txt.write_text("\n".join(keep) + ("\n" if keep else ""),
                           encoding="utf-8")

        # pass 1: 严 HSV 找"确定活"帧 (零误报, EX 暗帧漏检没关系)
        alive = {"seth": [], "pA": [], "pB": []}
        skip_frames = set()
        for i, txt in enumerate(txts):
            cls_set = {int(l.split()[0]) for l in
                       txt.read_text(encoding="utf-8").splitlines() if l.strip()}
            if cls_set & SKIP_CLS:
                skip_frames.add(i)
                continue
            img = cv2.imread(str(txt.with_suffix(".jpg")))
            if img is None:
                skip_frames.add(i)
                continue
            # 上限闸: 开场天空帧整个 crop 都是天蓝色 → cyan=1.0 (frame_000010
            # 实锤); 真血条在 crop 里最多 ~0.25 → 区间判据排除整片同色。
            if 0.04 < color_frac(img, BOSS_HP_R, CYAN) < 0.5:
                alive["seth"].append(i)
            if 0.05 < color_frac(img, PA_HP_R, GREEN) < 0.8:
                alive["pA"].append(i)
            if 0.08 < color_frac(img, PB_HP_R, GREEN) < 0.8:
                alive["pB"].append(i)

        # pass 2: [first_alive, last_alive] 区间内非 skip 帧全标
        spans = {k: (min(v), max(v)) if v else None for k, v in alive.items()}
        n_seth = n_pillar = 0
        for i, txt in enumerate(txts):
            if i in skip_frames:
                continue
            lines = [l for l in txt.read_text(encoding="utf-8").splitlines()
                     if l.strip()]
            cls_set = {int(l.split()[0]) for l in lines}
            if spans["seth"] and spans["seth"][0] <= i <= spans["seth"][1] \
                    and SETH_IDX not in cls_set:
                x, y, w, h = SETH_BOX
                lines.append(f"{SETH_IDX} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                n_seth += 1
            for key, box in (("pA", PILLARS[0]), ("pB", PILLARS[1])):
                if spans[key] and spans[key][0] <= i <= spans[key][1] \
                        and not any(matches_pillar(l, [box]) for l in lines):
                    x, y, w, h = box
                    lines.append(
                        f"{PILLAR_IDX} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    n_pillar += 1
            txt.write_text("\n".join(lines) + ("\n" if lines else ""),
                           encoding="utf-8")
        print(f"{name}: spans={ {k: v for k, v in spans.items()} }  "
              f"+seth {n_seth}  +pillar {n_pillar}  skip {len(skip_frames)}")


if __name__ == "__main__":
    main()
