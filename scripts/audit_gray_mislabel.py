# -*- coding: utf-8 -*-
"""亮/灰状态对**存量错标**审计与修复(先审后改, 默认只出报告)。

2026-08-02 定罪: v14「conf 0.99 检出灰键」不是模型缺陷 — 训练集里
灰键本身就被标成亮态。已训练池随机抽样实测: 444 开始制造 35% / 55 全部
选择 23% / 456 批量扫荡 9% 的框其实是灰态。补标之前必须先修存量, 否则
新标的灰和旧标的「灰当亮」在同一训练集里互相打架。

判据 = 框内 HSV **S(饱和度)** 双峰。彩色按钮(蓝/黄/绿底)灰化 = 去饱和,
S 从 ~185 掉到 ~3, 中间带零样本, 比 V 干净得多(V 会被 dim/淡出干扰)。
⚠白底钮(全部选择 55)天生低饱和, S 只有 11 vs 灰 3.4 — **间隔全族最薄**,
一律 100% 人审, 不许自动改。

Usage:
  py -X utf8 scripts/audit_gray_mislabel.py                 # 全族报告
  py -X utf8 scripts/audit_gray_mislabel.py --pair 444 --apply
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
sys.path.insert(0, str(ROOT))

# (亮 cls, 灰 cls, S 阈值, 是否允许自动改, 标签)
# ⛔⛔阈值**逐族实测标定**, 绝不跨族套用 — 2026-08-02 我拿购买族的 S=60
# 套给全族, 判出「加号 30,516/30,540 应改灰」这种荒谬结果: stepper 的 +/−
# 是**低饱和浅色钮**, 亮态 S 才 17-26, 60 把整族亮态全判成灰。
# 标定法: 阈值 =(灰态 p99 + 亮态 p50)/2, 再看**谷底样本数**(阈值±20% 区间内
# 的框数) — 谷底越少双峰越干净; 谷底 ≳ 亮态总数 = 该族 S 根本不可分。
PAIRS = [
    # ── 双峰干净, 可自动改 ──
    (444, 485, 94, True,  "开始制造"),          # 谷底 30 / 应改 221 (41%)
    (456, 457, 95, True,  "批量扫荡开始"),      # 谷底 1  / 应改 12
    (114, 112, 11, True,  "MIN"),               # 谷底 0  / 应改 39
    (115, 113, 12, True,  "减号"),              # 谷底 0  / 应改 41
    (89,  90,  97, True,  "领取奖励"),          # 谷底 9  / 应改 6
    (107, 413, 98, True,  "全部领取"),          # 应改 1
    (106, 396, 95, True,  "领取"),              # 应改 0 (已干净)
    (417, 416, 95, True,  "一次领取"),          # 应改 0 (已干净)
    # ── 阈值改用「真灰态 p99 + margin」后候选归零 = 本来就没有错标 ──
    # ⛔⛔阈值公式 (灰p99 + 亮p50)/2 有**系统性缺陷**: 亮态一侧若含多种颜色
    # 主题, p50 被拉高 → 阈值飘到两个主题之间, 把「另一主题的正常态」误判成灰。
    # 2026-08-02 对照 sheet 当场逮到三例:
    #   確認键 真灰 S=1-4(纯白钮) vs 候选 S=73-91 = **蓝色次要按钮**
    #          (掃蕩完成弹窗那个蓝確認一直点得动) → 阈值 93 会把蓝確認全标成灰
    #   MAX    真灰 S=0-1 vs 候选 S=23 = 灰蓝色 MAX 字(正常态)
    #   加号    真灰 S=0   vs 候选 S<14
    # ⇒ 阈值必须从**真灰态样本**反推(p99+margin), 不能从亮态 p50 反推。
    (111, 117, 5,  False, "MAX(真灰S≤1 → 候选归零)"),
    (26,  116, 5,  False, "加号(真灰S=0 → 候选归零)"),
    (20,  23,  10, False, "確認键(真灰S≤4 → 候选归零)"),
    # 入场键的「灰」是**降饱和但仍有色**(S=65-66 vs 亮 122-144), 不是去饱和;
    # 阈值取两峰中点 94, 49 个候选待人审。
    (79,  82,  94, False, "入场键(降饱和态, 49候选待审)"),
    # ── S 判据不成立, 不做 ──
    # 103/489 购买: 训练集里**没有**灰态样本(489 是今天新建的), 亮态 S 137-144
    #   分布极窄无双峰 → 训练集零错标, 别用启发式硬切。
    # 141/432 跳过故事: 灰 p99=37 > 亮 p50=33, **方向是反的**, S 不可分。
    # 55/404 全部选择: 白底钮, 灰比亮**更亮更饱和**(唯一灰样本 S=68 vs 亮 p50=11),
    #   方向反 → 必须改用 V 判据(实测 亮 V=173 / 灰 V=204), 见 template_label_assets。
]


def _scan(task):
    """量一个目录里某 cls 全部框的 S/V/B-R。"""
    d, bright, dark = task
    out = []
    for txt in Path(d).glob("*.txt"):
        if txt.stem == "classes":
            continue
        lines = [l.split() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
        tg = [(i, p) for i, p in enumerate(lines) if int(p[0]) in (bright, dark)]
        if not tg:
            continue
        img = cv2.imread(str(txt.with_suffix(".jpg")))
        if img is None:
            continue
        H, W = img.shape[:2]
        for i, p in tg:
            cx, cy, bw, bh = (float(x) for x in p[1:5])
            crop = img[int((cy - bh / 2) * H):int((cy + bh / 2) * H),
                       int((cx - bw / 2) * W):int((cx + bw / 2) * W)]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            out.append((str(txt), i, int(p[0]), float(hsv[:, :, 1].mean()),
                        float(hsv[:, :, 2].mean()), cx, cy, bw, bh))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, help="只跑某个亮 cls 的对")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--sheet-frac", type=float, default=0.2, help="抽检比例")
    args = ap.parse_args()

    from scripts.build_ui_v2 import REAL_SOURCES, SYNTH_SOURCES, VAL_SOURCES
    names = [c.strip() for c in (RAW / "_classes.txt").read_text(encoding="utf-8").splitlines() if c.strip()]
    dirs = [RAW / s for s in list(REAL_SOURCES) + list(SYNTH_SOURCES) + list(VAL_SOURCES)
            if (RAW / s).is_dir()]

    pairs = [p for p in PAIRS if args.pair is None or p[0] == args.pair]
    summary = []
    for (bright, dark, s_thr, auto, label) in pairs:
        recs = []
        with ProcessPoolExecutor(max_workers=24) as ex:
            for out in ex.map(_scan, [(str(d), bright, dark) for d in dirs], chunksize=4):
                recs.extend(out)
        if not recs:
            summary.append((label, bright, dark, 0, 0, 0, "无样本"))
            continue
        S = np.array([r[3] for r in recs])
        cur_bright = [r for r in recs if r[2] == bright]
        cur_dark = [r for r in recs if r[2] == dark]
        # 错标 = 现标亮态但 S < 阈值
        flip = [r for r in cur_bright if r[3] < s_thr]
        # ⛔离群闸: 候选自身的 S 应聚成一个窄主峰(真灰态去饱和后极稳定,
        # 实测 444 的 219/221 全是 S=3 V=177)。落在 median±6·MAD 外的是
        # **点击特效帧**(白光圈盖住按钮 → S 假性下降, 实测 S=74/84 V=213/224)
        # 或别的污染 → 踢进人审, 不自动改。
        outlier = []
        if len(flip) >= 8:
            fs = np.array([r[3] for r in flip])
            med = float(np.median(fs))
            mad = float(np.median(np.abs(fs - med))) or 1.0
            keep, outlier = [], []
            for r in flip:
                (outlier if abs(r[3] - med) > 6 * mad else keep).append(r)
            if outlier:
                print(f"  ⚠离群踢出人审: {len(outlier)} (S={[round(o[3]) for o in outlier][:6]})")
            flip = keep
        # 反向审计 = 现标灰态但 S >= 阈值
        rev = [r for r in cur_dark if r[3] >= s_thr]
        mid = int(((S >= s_thr * 0.4) & (S < s_thr)).sum()) if s_thr > 10 else \
              int(((S >= s_thr) & (S < s_thr * 2.5)).sum())
        summary.append((label, bright, dark, len(cur_bright), len(cur_dark), len(flip),
                        f"rev={len(rev)} auto={auto}"))
        print(f"\n=== {label}  亮{bright}({len(cur_bright)}框) / 灰{dark}({len(cur_dark)}框) ===")
        if len(cur_bright):
            b = np.array([r[3] for r in cur_bright])
            print(f"  现标亮态 S: p5={np.percentile(b,5):.1f} p50={np.percentile(b,50):.1f} p95={np.percentile(b,95):.1f}")
        print(f"  ⛔应改灰(S<{s_thr}): {len(flip)}   反向可疑(现灰但S>={s_thr}): {len(rev)}")

        tag = f"gray_{bright}_{dark}"
        with open(ROOT / "data" / f"_grayfix_{tag}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(("txt", "line", "cur_cls", "S", "V", "cx", "cy", "bw", "bh", "action"))
            for r in flip:
                w.writerow(list(r) + [f"->{dark}"])
            for r in rev:
                w.writerow(list(r) + [f"REVIEW(cur={dark})"])

        # contact sheet
        picks = flip if auto else flip            # 人审对同样出 flip
        step = max(1, int(1 / max(args.sheet_frac, 1e-6))) if auto else 1
        picks = picks[::step]
        if picks:
            CELL, COLS = 420, 8
            tiles = []
            for (t, i, c, s, v, cx, cy, bw, bh) in picks[:160]:
                img = cv2.imread(str(Path(t).with_suffix(".jpg")))
                if img is None:
                    continue
                H, W = img.shape[:2]
                x1, y1 = int((cx - bw * 1.1) * W), int((cy - bh * 1.6) * H)
                x2, y2 = int((cx + bw * 1.1) * W), int((cy + bh * 1.6) * H)
                crop = img[max(0, y1):y2, max(0, x1):x2].copy()
                if crop.size == 0:
                    continue
                rx = int(cx * W - bw * W / 2) - max(0, x1)
                ry = int(cy * H - bh * H / 2) - max(0, y1)
                cv2.rectangle(crop, (rx, ry), (rx + int(bw * W), ry + int(bh * H)), (0, 0, 255), 3)
                cv2.putText(crop, f"S{s:.0f}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                tiles.append(cv2.resize(crop, (CELL, int(crop.shape[0] * CELL / crop.shape[1]))))
            if tiles:
                rh = max(t.shape[0] for t in tiles)
                nr = (len(tiles) + COLS - 1) // COLS
                page = np.zeros((nr * rh, COLS * CELL, 3), np.uint8)
                for j, t in enumerate(tiles):
                    r, c = divmod(j, COLS)
                    page[r * rh:r * rh + t.shape[0], c * CELL:(c + 1) * CELL] = t
                cv2.imwrite(str(ROOT / "data" / f"_grayfix_{tag}.jpg"), page,
                            [cv2.IMWRITE_JPEG_QUALITY, 86])
                print(f"  sheet: _grayfix_{tag}.jpg ({len(tiles)}/{len(flip)})")

        if args.apply:
            if not auto:
                print("  ⛔该对 auto=False, 拒绝自动改 — 人审 sheet 后用 --force-pair 手动执行")
                continue
            by_txt = collections.defaultdict(list)
            for (t, i, c, s, v, *_rest) in flip:
                by_txt[t].append(i)
            n = 0
            for t, idxs in by_txt.items():
                p = Path(t)
                lines = p.read_text(encoding="utf-8").splitlines()
                for i in idxs:
                    parts = lines[i].split()
                    if int(parts[0]) == bright:
                        parts[0] = str(dark)
                        lines[i] = " ".join(parts)
                        n += 1
                p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"  APPLIED {n} 框 {bright}->{dark}")

    print("\n===== 汇总 =====")
    print(f"{'族':22s} {'亮':>4} {'灰':>4} {'现亮框':>7} {'现灰框':>7} {'应改灰':>7}  备注")
    for (label, b, d, nb, nd, nf, note) in summary:
        print(f"{label:22s} {b:>4} {d:>4} {nb:>7} {nd:>7} {nf:>7}  {note}")


if __name__ == "__main__":
    main()
