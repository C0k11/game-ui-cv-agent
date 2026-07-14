# -*- coding: utf-8 -*-
"""倍速类标注全库审计 (2026-07-14, 用户实锤: 单▶被模型判「战斗三倍速」,
疑似从早代就类语义污染).

词表: 412=战斗1倍速(▶) / 135=战斗2倍速(▶▶) / 129=战斗三倍速(▶▶▶)。
方法: 全 battle 训练池 + 大蛇池, 裁出全部 {412,135,129} GT 框, 数框内深色
三角连通域个数 = 实际倍速, 输出 标注 × 实际 混淆矩阵 + 抽样拼图人工复核。

用法:
  py scripts/audit_speed_labels.py           # 矩阵 + 拼图导出
  py scripts/audit_speed_labels.py --fix     # (确认矩阵后) 按箭头数重写 cls
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\Project\ai game secretary")
from vision.io_utils import imread_any  # noqa: E402

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
OUT = Path(r"D:\Project\ai game secretary\data\_speed_audit")
POOLS = [
    "run_battle_material_20260708",
    "run_20260710_110430", "run_20260710_104718",
    "run_20260710_110759", "run_20260710_104427",
    "axis_碧蓝档案_大决战_33_耶罗尼姆斯_作业考古合集_p02_2_重甲_水局4010w_BV1KNNc64EEf_p2",
    "axis_碧蓝档案_大决战_28_赫赛德_作业考古合集_p08_8_弹甲_4003w_BV19XFNzHEup_p8",
    "axis_碧蓝档案_大决战_32_白_黑_作业考古合集_p02_2_特甲_妹爱黑子3984w_BV1PtLn6zEF4_p2",
    "axis_碧蓝档案_大决战_27_薇娜_作业考古合集_p05_5_弹甲_国家队3949w_BV1giiYBeELr_p5",
]
SPEED = {412: "1倍速", 135: "2倍速", 129: "三倍速"}
N2CLS = {1: 412, 2: 135, 3: 129}


def count_arrows(crop: np.ndarray) -> int:
    """倍速按钮内深色▶个数。
    v2: 黄色高亮态(按下/发光)按钮上三角互相粘连成一个连通域 — v1 直接数
    连通域把 3▶ 数成 1(赫赛德 27 框假警报实锤)。改为: 每个显著连通域按
    宽高比 w/(h*0.72) 推内含三角数(单▶ w/h≈0.72, 粘连 n 个 ≈ n×0.72)。"""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = (g < 110).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    H, W = g.shape
    area = H * W
    cnt = 0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if a < area * 0.02 or h < H * 0.25:
            continue
        # 实心度过滤: 标注框边缘阴影是细长条, 三角/三角串是实心块
        if a / max(w * h, 1) < 0.30:
            continue
        cnt += max(1, min(3, round(w / (h * 0.72))))
    return min(cnt, 4)


def main():
    fix = "--fix" in sys.argv
    OUT.mkdir(exist_ok=True)
    mat = defaultdict(Counter)          # (pool_short, 标注cls) -> Counter(实际n)
    crops_by_key = defaultdict(list)
    fixes = []                          # (lbl_path, line_idx, old_cls, new_cls)

    for pool_name in POOLS:
        pool = RAW / pool_name
        if not pool.is_dir():
            print(f"⚠ 池不存在: {pool_name}")
            continue
        short = pool_name[:24]
        for lbl in sorted(pool.glob("*.txt")):
            if lbl.name == "classes.txt":
                continue
            lines = lbl.read_text(encoding="utf-8").splitlines()
            hits = [(i, l) for i, l in enumerate(lines)
                    if l.strip() and int(l.split()[0]) in SPEED]
            if not hits:
                continue
            img = imread_any(str(lbl.with_suffix(".jpg")))
            if img is None:
                continue
            H, W = img.shape[:2]
            for li, l in hits:
                c, xc, yc, w, h = l.split()[:5]
                c = int(c)
                xc, yc, w, h = map(float, (xc, yc, w, h))
                x1 = int((xc - w / 2) * W)
                y1 = int((yc - h / 2) * H)
                x2 = int((xc + w / 2) * W)
                y2 = int((yc + h / 2) * H)
                crop = img[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue
                n = count_arrows(crop)
                mat[(short, c)][n] += 1
                if len(crops_by_key[(short, c, n)]) < 24:
                    crops_by_key[(short, c, n)].append(
                        cv2.resize(crop, (96, 64)))
                if fix and n in N2CLS and N2CLS[n] != c:
                    fixes.append((lbl, li, c, N2CLS[n]))

    print(f"\n{'池':<26} {'标注':<8} {'实际1▶':>6} {'实际2▶':>6} "
          f"{'实际3▶':>6} {'其他':>5}")
    for (short, c), cnt in sorted(mat.items()):
        other = sum(v for k, v in cnt.items() if k not in (1, 2, 3))
        print(f"{short:<26} {SPEED[c]:<8} {cnt.get(1, 0):>6} "
              f"{cnt.get(2, 0):>6} {cnt.get(3, 0):>6} {other:>5}")

    # 抽样拼图(人工复核计数器本身)
    for (short, c, n), crops in crops_by_key.items():
        rows = [np.hstack(crops[i:i + 8] +
                          [np.zeros((64, 96, 3), np.uint8)] * (8 - len(crops[i:i + 8])))
                for i in range(0, len(crops), 8)]
        grid = np.vstack(rows)
        ok, buf = cv2.imencode(".jpg", grid)
        if ok:
            buf.tofile(str(OUT / f"{short}_标注{SPEED[c]}_实际{n}箭头.jpg"))
    print(f"\n抽样拼图 → {OUT}")

    if fix and fixes:
        import shutil
        import time
        stamp = time.strftime("%Y%m%d_%H%M%S")
        bak = RAW / "_backups" / f"{stamp}_speedfix"
        bak.mkdir(parents=True, exist_ok=True)
        by_file = defaultdict(list)
        for lbl, li, old, new in fixes:
            by_file[lbl].append((li, old, new))
        for lbl, edits in by_file.items():
            shutil.copy2(lbl, bak / f"{lbl.parent.name[:30]}__{lbl.name}")
            lines = lbl.read_text(encoding="utf-8").splitlines()
            for li, old, new in edits:
                parts = lines[li].split()
                assert int(parts[0]) == old
                parts[0] = str(new)
                lines[li] = " ".join(parts)
            lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[fix] 重写 {len(fixes)} 框 / {len(by_file)} 文件, 备份 → {bak}")


if __name__ == "__main__":
    main()
