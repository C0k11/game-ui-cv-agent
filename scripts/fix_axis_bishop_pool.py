# -*- coding: utf-8 -*-
"""凹轴耶罗尼姆斯池 标注修复 (2026-07-12, 用户人审后四毒点).

用户已审: 我方/敌方/主教口径(主教=44帧, boss基本静止)。遗留:
  1. 主教只标 44/444 战斗帧 → 400 帧假负样本 (训练毒药)
  2. HUD 漏标 123 帧 (暗帧 v4 漏检: 战斗暂停/倍速键)
  3. 倍速键疑似混标 (412 1倍速 278框 vs 129 三倍速 45框, 用户截图 UI=3x)
  4. 478 塞特的愤怒 残留 1 框

方案 = 传播 + NCC 图像校验 (绝不盲补):
  主教: 未标战斗帧 ← 时间最近已标框, crop 与已标模板均值 NCC ≥ 阈值才写
  HUD : 位置固定, 同法; 倍速键额外做 1x/3x 模板判别改写
  全程先备份池 label 到 data/raw_images/_backups/

用法: py scripts/fix_axis_bishop_pool.py [--apply]   (默认分析模式只报告)
"""
import glob
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\Project\ai game secretary")
from vision.io_utils import imread_any  # noqa: E402  (cv2.imread 不吃中文路径)
RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
POOL = Path(glob.glob(str(RAW / "axis_*耶罗尼姆斯*"))[0])
BISHOP, SETH = 480, 478
PAUSE, X3, X1, AUTO_OFF = 128, 129, 412, 134
APPLY = "--apply" in sys.argv


def load(t: Path):
    out = []
    for ln in t.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) >= 5:
            out.append([int(p[0])] + [float(v) for v in p[1:5]])
    return out


def save(t: Path, boxes):
    lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
             for b in boxes]
    t.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def crop(img, b, pad=0.0):
    h, w = img.shape[:2]
    cx, cy, bw, bh = b[1], b[2], b[3] * (1 + pad), b[4] * (1 + pad)
    x1, y1 = max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h))
    x2, y2 = min(w, int((cx + bw / 2) * w)), min(h, int((cy + bh / 2) * h))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def ncc(a, b):
    if a is None or b is None:
        return -1.0
    b = cv2.resize(b, (a.shape[1], a.shape[0]))
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ga -= ga.mean(); gb -= gb.mean()
    d = np.sqrt((ga * ga).sum() * (gb * gb).sum())
    return float((ga * gb).sum() / d) if d > 1e-6 else -1.0


def main():
    frames, imgs = {}, {}
    for t in sorted(POOL.glob("*.txt")):
        if t.name == "classes.txt":
            continue
        n = int(re.search(r"(\d+)", t.stem).group(1))
        frames[n] = load(t)
        imgs[n] = t.with_suffix(".jpg")

    def read(n):
        return imread_any(str(imgs[n]))

    combat = sorted(n for n, bs in frames.items()
                    if any(b[0] in (476, 477, BISHOP) for b in bs))
    combat_set = set(combat)
    report = defaultdict(list)
    changed = set()

    # ── 1. 478 残留 → 480 (在主教位置邻域才改) ──
    for n, bs in frames.items():
        for b in bs:
            if b[0] == SETH:
                if 0.6 < b[1] < 1.0 and b[2] < 0.5:
                    b[0] = BISHOP
                    report["seth_fixed"].append(n)
                else:
                    report["seth_manual"].append((n, round(b[1], 3), round(b[2], 3)))
                changed.add(n)

    # ── 2. 主教: 全帧模板搜索 (boss 会动/相机移动, 最近邻传播 p50 仅 0.21
    #    已证伪 → matchTemplate 半分辨率全图搜, 多模板取最大响应) ──
    labeled_b = sorted(n for n in combat if any(b[0] == BISHOP for b in frames[n]))
    tpl_b = [(n, next(b for b in frames[n] if b[0] == BISHOP)) for n in labeled_b]
    SCALE = 0.5
    # 按连续已标段抽模板(段首/段中/段尾各一) — 均匀抽样会漏掉用户补的
    # 小段新形态(boss 转阶段后模板对不上 = 上轮 p50 掉到 0.60 的根因)。
    segs = []
    for n, b in tpl_b:
        if segs and n - segs[-1][-1][0] <= 3:
            segs[-1].append((n, b))
        else:
            segs.append([(n, b)])
    picks = []
    for seg in segs:
        idxs = {0, len(seg) // 2, len(seg) - 1}
        picks += [seg[i] for i in sorted(idxs)]
    tpls = []          # (gray_tpl, w_norm, h_norm)
    for n, b in picks:
        c = crop(read(n), b)
        if c is None:
            continue
        g = cv2.cvtColor(cv2.resize(c, None, fx=SCALE, fy=SCALE),
                         cv2.COLOR_BGR2GRAY)
        tpls.append((g, b[3], b[4]))
    scores_b = []      # (frame, score, cx, cy, w, h)
    for n in combat:
        if n in labeled_b:
            continue
        img = read(n)
        gs = cv2.cvtColor(cv2.resize(img, None, fx=SCALE, fy=SCALE),
                          cv2.COLOR_BGR2GRAY)
        best = (-1.0, 0, 0, 0, 0)
        for g, bw, bh in tpls:
            if g.shape[0] >= gs.shape[0] or g.shape[1] >= gs.shape[1]:
                continue
            r = cv2.matchTemplate(gs, g, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx > best[0]:
                cx = (loc[0] + g.shape[1] / 2) / gs.shape[1]
                cy = (loc[1] + g.shape[0] / 2) / gs.shape[0]
                best = (mx, cx, cy, bw, bh)
        scores_b.append((n, round(best[0], 3), best[1], best[2], best[3], best[4]))
    scores_b.sort(key=lambda x: x[1])
    report["bishop_scores"] = [(n, s) for n, s, *_ in scores_b]
    if APPLY:
        TH_B = float(next((a.split("=")[1] for a in sys.argv
                           if a.startswith("--bishop-th=")), 0.55))
        for n, s, cx, cy, bw, bh in scores_b:
            if s >= TH_B:
                frames[n].append([BISHOP, cx, cy, bw, bh])
                changed.add(n)
                report["bishop_filled"].append(n)
            else:
                report["bishop_skip"].append((n, s))

    # ── 3. HUD 补标 (位置固定, 中位框 + NCC) ──
    for cls, name in ((PAUSE, "暂停"), (AUTO_OFF, "AUTO关")):
        boxes = [b for bs in frames.values() for b in bs if b[0] == cls]
        if not boxes:
            continue
        med = [cls] + list(np.median(np.array([b[1:] for b in boxes]), axis=0))
        srcs = [n for n in combat if any(b[0] == cls for b in frames[n])]
        tpls = [crop(read(n), med) for n in srcs[:: max(1, len(srcs) // 12)]]
        sc = []
        for n in combat:
            if any(b[0] == cls for b in frames[n]):
                continue
            c = crop(read(n), med)
            s = max((ncc(t, c) for t in tpls if t is not None), default=-1)
            sc.append((n, round(s, 3)))
        sc.sort(key=lambda x: x[1])
        report[f"hud_{name}_scores"] = sc
        if APPLY:
            TH = 0.70
            for n, s in sc:
                if s >= TH:
                    frames[n].append(list(med))
                    changed.add(n)
                    report[f"hud_{name}_filled"].append(n)

    # ── 3.5 倍速键补标: 缺倍速的战斗帧, 固定位 crop 与 1x/3x 两组模板
    #    判别写入 (状态类不能盲贴中位框; within 0.93/0.99 cross 0.53 可分) ──
    sp_all = [b for bs in frames.values() for b in bs if b[0] in (X1, X3)]
    if sp_all:
        med_sp = [0] + list(np.median(np.array([b[1:] for b in sp_all]), axis=0))
        t_by = {}
        for cls in (X1, X3):
            src = [n for n in combat if any(b[0] == cls for b in frames[n])]
            t_by[cls] = [c for c in (crop(read(n), med_sp)
                         for n in src[:: max(1, len(src) // 8)]) if c is not None]
        sc_sp = []
        for n in combat:
            if any(b[0] in (X1, X3) for b in frames[n]):
                continue
            c = crop(read(n), med_sp)
            s1 = max((ncc(t, c) for t in t_by[X1]), default=-1)
            s3 = max((ncc(t, c) for t in t_by[X3]), default=-1)
            cls, s = (X1, s1) if s1 >= s3 else (X3, s3)
            sc_sp.append((n, cls, round(s, 3)))
        sc_sp.sort(key=lambda x: x[2])
        report["speed_fill_scores"] = sc_sp
        if APPLY:
            TH_SP = 0.60
            for n, cls, s in sc_sp:
                if s >= TH_SP:
                    frames[n].append([cls] + list(med_sp[1:]))
                    changed.add(n)
                    report["speed_filled"].append((n, cls))
                else:
                    report["speed_skip"].append((n, s))

    # ── 4. 倍速键 1x/3x 判别: 全部倍速框 crop 与两类模板比 ──
    sp1 = [(n, b) for n, bs in frames.items() for b in bs if b[0] == X1]
    sp3 = [(n, b) for n, bs in frames.items() for b in bs if b[0] == X3]
    all_sp = [b for _, b in sp1 + sp3]
    if sp1 and sp3 and all_sp:
        med_sp = [0] + list(np.median(np.array([b[1:] for b in all_sp]), axis=0))
        # 模板从"用户确认过的三倍速帧"没法自动知道 — 用两类各自 crop 的
        # 聚类中心互相打分: 若两类模板本身高度相似 = 模型在乱分, 报告出来。
        t1 = [crop(read(n), med_sp) for n, _ in sp1[:: max(1, len(sp1) // 8)]]
        t3 = [crop(read(n), med_sp) for n, _ in sp3[:: max(1, len(sp3) // 8)]]
        t1 = [t for t in t1 if t is not None]; t3 = [t for t in t3 if t is not None]
        cross = np.mean([ncc(a, b) for a in t1[:5] for b in t3[:5]])
        within1 = np.mean([ncc(a, b) for i, a in enumerate(t1[:5])
                           for b in t1[i + 1:5]]) if len(t1) > 1 else 1
        within3 = np.mean([ncc(a, b) for i, a in enumerate(t3[:5])
                           for b in t3[i + 1:5]]) if len(t3) > 1 else 1
        report["speed_sim"] = {"cross_1x_vs_3x": round(float(cross), 3),
                               "within_1x": round(float(within1), 3),
                               "within_3x": round(float(within3), 3),
                               "n_1x": len(sp1), "n_3x": len(sp3)}

    # ── 落盘 ──
    if APPLY and changed:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        bak = RAW / "_backups" / f"{stamp}_axisfix"
        bak.mkdir(parents=True, exist_ok=True)
        for n in changed:
            src = imgs[n].with_suffix(".txt")
            shutil.copy2(src, bak / src.name)
        for n in changed:
            save(imgs[n].with_suffix(".txt"), frames[n])
        print(f"[apply] {len(changed)} 帧已改, 备份 → {bak}")

    # ── 报告 ──
    print(f"pool={POOL.name} 战斗帧={len(combat)}")
    print(f"478残留改480: {report['seth_fixed']} | 位置存疑需手动: {report['seth_manual']}")
    sb = report["bishop_scores"]
    if sb:
        arr = np.array([s for _, s in sb])
        print(f"\n主教未标 {len(sb)} 帧 NCC 分布: min={arr.min():.2f} "
              f"p10={np.percentile(arr,10):.2f} p50={np.percentile(arr,50):.2f} "
              f"p90={np.percentile(arr,90):.2f} max={arr.max():.2f}")
        print("  最低10帧:", sb[:10])
    for k in ("hud_暂停_scores", "hud_AUTO关_scores"):
        s = report[k]
        if s:
            arr = np.array([x for _, x in s])
            print(f"\n{k}: {len(s)}帧 min={arr.min():.2f} p50={np.percentile(arr,50):.2f} max={arr.max():.2f}")
            print("  最低6帧:", s[:6])
    if "speed_sim" in report:
        print("\n倍速键判别:", report["speed_sim"])
    if report["speed_fill_scores"]:
        s = report["speed_fill_scores"]
        arr = np.array([x for _, _, x in s])
        print(f"倍速补标候选 {len(s)}帧: min={arr.min():.2f} p50={np.percentile(arr,50):.2f} 最低6: {s[:6]}")
    if APPLY and report["speed_filled"]:
        from collections import Counter as _C
        print(f"倍速已补 {len(report['speed_filled'])} (类分布 {dict(_C(c for _, c in report['speed_filled']))}) 跳过 {len(report['speed_skip'])}: {report['speed_skip'][:6]}")
    if APPLY:
        print(f"\n主教补 {len(report['bishop_filled'])} 跳过 {len(report['bishop_skip'])}: {report['bishop_skip'][:10]}")
        for k in ("hud_暂停_filled", "hud_AUTO关_filled"):
            print(f"{k}: {len(report[k])}")


if __name__ == "__main__":
    main()
