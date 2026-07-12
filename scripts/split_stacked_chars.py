# -*- coding: utf-8 -*-
"""我方并框拆分器 (2026-07-12, 用户: 一框包两人怎么办/能不能直接上划分).

原理 = 用户洞察「血条渲染在顶层永不重叠」的规则实现:
  对每个「我方」框, 在框内做 HSV 亮绿血条检测(宽扁连通域) → 检出 N≥2 条
  血条 = 框里装了 N 个学生 → 按每条血条为锚拆成 N 个子框(血条x范围定宽,
  血条y向下延伸定高), 替换原并框。
用途: ①预标后处理(prefill 后跑一遍, 人审免手拆) ②combat 2.0 运行时同款
  逻辑做个体锚定。

用法: py scripts/split_stacked_chars.py <pool_substr> [--apply]
  默认分析模式: 找出所有可拆框, 存目检图到 scratchpad, 不写盘。
"""
import glob
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\Project\ai game secretary")
from vision.io_utils import imread_any  # noqa: E402

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
MY = 476
_BAK = None
SCRATCH = Path(r"C:\Users\shien\AppData\Local\Temp\claude"
               r"\D--Project-ai-game-secretary--claude-worktrees-magical-tharp-fa5d91"
               r"\a4e15e41-e17a-4cd8-8e96-4b51142a5c5a\scratchpad")


def find_hp_bars(img, x1, y1, x2, y2):
    """框内亮绿血条检测 → [(bx1,by1,bx2,by2)] 像素坐标(按 y 排序)。
    血条=高饱和草绿宽扁条(我方专属色; 敌方红/黄不命中)。"""
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return []
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (35, 110, 130), (60, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    bw = x2 - x1
    bars = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h == 0:
            continue
        # 宽扁 + 至少占框宽 1/3 (排除技能特效绿光点)
        if w / max(h, 1) >= 3.0 and w >= bw * 0.33 and h <= 28:
            bars.append((x1 + x, y1 + y, x1 + x + w, y1 + y + h))
    bars.sort(key=lambda b: b[1])
    # 同一条血条被分割成两段(受击闪烁)→ y 相近的合并
    merged = []
    for b in bars:
        if merged and abs(b[1] - merged[-1][1]) < 14:
            m0 = merged[-1]
            merged[-1] = (min(m0[0], b[0]), min(m0[1], b[1]),
                          max(m0[2], b[2]), max(m0[3], b[3]))
        else:
            merged.append(b)
    return merged


def split_box(img, x1, y1, x2, y2, bars):
    """按 N 条血条拆 N 个子框: 血条 x 范围外扩定宽; y=血条顶 → 下一条血条顶
    (末条到原框底)。"""
    H, W = img.shape[:2]
    out = []
    min_w = int((x2 - x1) * 0.62)     # 子框宽下限: 血条比身位窄, 纯血条宽切不全
    for i, (bx1, by1, bx2, by2) in enumerate(bars):
        w = max(int((bx2 - bx1) * 1.5), min_w)
        cx = (bx1 + bx2) // 2
        sx1 = max(0, cx - w // 2)
        sx2 = min(W, cx + w // 2)
        sy1 = max(0, by1 - int(0.008 * H))
        sy2 = bars[i + 1][1] - 2 if i + 1 < len(bars) else y2
        if sy2 - sy1 > 10 and sx2 - sx1 > 10:
            out.append((sx1, sy1, sx2, sy2))
    return out


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "axis_"
    apply = "--apply" in sys.argv
    pools = [Path(p) for p in glob.glob(str(RAW / f"*{pat}*")) if Path(p).is_dir()]
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tiles, n_split, n_boxes, changed_files = [], 0, 0, 0
    for pool in pools:
        for lbl in sorted(pool.glob("*.txt")):
            if lbl.name == "classes.txt":
                continue
            lines = [l.split() for l in lbl.read_text(encoding="utf-8").splitlines()
                     if l.strip()]
            my = [(i, l) for i, l in enumerate(lines) if int(l[0]) == MY]
            if not my:
                continue
            img = None
            new_lines, replaced = list(lines), False
            for i, l in my:
                n_boxes += 1
                if img is None:
                    img = imread_any(str(lbl.with_suffix(".jpg")))
                    if img is None:
                        break
                    H, W = img.shape[:2]
                cx, cy, w, h = (float(v) for v in l[1:5])
                x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
                x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
                bars = find_hp_bars(img, x1, y1, x2, y2)
                if len(bars) >= 2:
                    subs = split_box(img, x1, y1, x2, y2, bars)
                    if len(subs) >= 2:
                        n_split += 1
                        replaced = True
                        new_lines[i] = None
                        for (sx1, sy1, sx2, sy2) in subs:
                            new_lines.append([str(MY),
                                              f"{(sx1+sx2)/2/W:.6f}",
                                              f"{(sy1+sy2)/2/H:.6f}",
                                              f"{(sx2-sx1)/W:.6f}",
                                              f"{(sy2-sy1)/H:.6f}"])
                        if len(tiles) < 12:      # 目检图: 原框红 / 子框绿
                            vis = img.copy()
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            for (sx1, sy1, sx2, sy2) in subs:
                                cv2.rectangle(vis, (sx1, sy1), (sx2, sy2),
                                              (0, 255, 0), 2)
                            cx1, cy1 = max(0, x1 - 60), max(0, y1 - 40)
                            tiles.append(cv2.resize(
                                vis[cy1:min(H, y2 + 40), cx1:min(W, x2 + 220)],
                                (300, 340)))
            if replaced and apply:
                import shutil
                import time
                global _BAK
                if _BAK is None:
                    _BAK = RAW / "_backups" / (
                        time.strftime("%Y%m%d_%H%M%S") + "_split")
                    _BAK.mkdir(parents=True, exist_ok=True)
                shutil.copy2(lbl, _BAK / f"{pool.name[:32]}__{lbl.name}")
                keep = [" ".join(l) for l in new_lines if l is not None]
                lbl.write_text("\n".join(keep) + "\n", encoding="utf-8")
                changed_files += 1
    print(f"我方框 {n_boxes} 个, 可拆并框 {n_split} 个"
          + (f", 已改写 {changed_files} 文件" if apply else " (分析模式未写盘)"))
    if tiles:
        rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
        wmax = max(r.shape[1] for r in rows)
        rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1],
                                   cv2.BORDER_CONSTANT) for r in rows]
        cv2.imencode(".jpg", np.vstack(rows))[1].tofile(
            str(SCRATCH / "split_preview.jpg"))
        print("目检图: split_preview.jpg (红=原并框, 绿=拆分子框)")


if __name__ == "__main__":
    main()
