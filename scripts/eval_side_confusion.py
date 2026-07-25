# -*- coding: utf-8 -*-
"""敌我混淆专项评测 —— 对比任意两个 battle 权重在**人审 GT** 上的表现。

回答一个问题: 黄机甲还会不会被认成「我方」。

⛔**分母必须拆两套报, 只报一套会骗人**(2026-07-25 实测撞到):
  v10 在 holdout 上"敌→友 5/68=7.4%"看着比全池的 22.5% 好得多, 真相是
  **召回只有 43.8%** —— 大部分敌人压根没检出, 没检出就无从误分类。
  ⇒ 一个什么都不检的模型能拿 0% 误分类。所以:
    敌→友(全)  = GT敌方被判我方 / **全部 GT 敌方框**   ← 端到端损害
    敌→友(匹配)= GT敌方被判我方 / **匹配上的 GT 敌方框** ← 纯分类错误率
  两个都要看: 前者降可能是"检得多了但也错得多", 后者降才是分类真的变好。
指标:
  召回      = 匹配上的 GT 框 / GT 框总数
  Boss 召回 = Boss GT 被检出(任意类) / Boss GT 数

用法:
  py scripts/eval_side_confusion.py                       # v10 vs v10s, 全 GT 池
  py scripts/eval_side_confusion.py --holdout             # 只评种子 holdout(帧号>=48)
  py scripts/eval_side_confusion.py --weights A.pt B.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from vision.io_utils import imread_any                      # noqa: E402

RAW = _ROOT / "data" / "raw_images"
RUNS = Path(r"D:\Project\ml_cache\models\yolo\runs")
GT_POOL = "run_20260715_025638_botplay_clean"   # 唯一人审过的 botplay 池
IDENT = {476, 477, 478, 479, 480, 481, 482, 483}
ALLY, ENEMY, BOSS = 476, 477, 479
MATCH_IOU = 0.45
SEED_SPLIT = 48        # 与 build_battle_v10s.py 一致: >=48 是 holdout


def iou_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    it = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return it / (aa[:, None] + bb[None, :] - it + 1e-9)


def load_gt(holdout_only: bool):
    master = [l.strip() for l in
              open(RAW / "_classes.txt", encoding="utf-8") if l.strip()]
    frames = []
    for jp in sorted((RAW / GT_POOL).glob("*.jpg"),
                     key=lambda p: int(re.search(r"(\d+)", p.stem).group(1))):
        idx = int(re.search(r"(\d+)", jp.stem).group(1))
        if holdout_only and idx < SEED_SPLIT:
            continue
        boxes = []
        t = jp.with_suffix(".txt")
        if t.exists():
            for ln in t.read_text(encoding="utf-8").splitlines():
                p = ln.split()
                if len(p) >= 5 and int(p[0]) in IDENT:
                    c = int(p[0]); xc, yc, w, h = map(float, p[1:5])
                    boxes.append((c, np.array([xc - w / 2, yc - h / 2,
                                               xc + w / 2, yc + h / 2])))
        if boxes:
            frames.append((jp, boxes))
    return master, frames


def evaluate(weights: str, master, frames, conf: float):
    from ultralytics import YOLO
    n2i = {n: i for i, n in enumerate(master)}
    m = YOLO(weights)
    m2m = {i: n2i[n] for i, n in m.names.items() if n in n2i}
    stat = {"gt": 0, "match": 0, "e2f": 0, "f2e": 0,
            "gt_e": 0, "gt_a": 0, "gt_boss": 0, "boss_hit": 0, "pred": 0,
            "m_e": 0, "m_a": 0}   # 匹配上的敌方/我方 GT 数(第二套分母)
    for jp, gt in frames:
        img = imread_any(str(jp))
        H, W = img.shape[:2]
        r = m.predict(img, conf=conf, iou=0.5, imgsz=960, verbose=False)[0]
        pr = []
        if r.boxes is not None:
            for b in r.boxes:
                mi = m2m.get(int(b.cls[0]))
                if mi in IDENT:
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                    pr.append((mi, np.array([x1 / W, y1 / H, x2 / W, y2 / H])))
        stat["pred"] += len(pr)
        gb = np.array([b for _, b in gt])
        pb = np.array([b for _, b in pr]) if pr else np.zeros((0, 4))
        M = iou_mat(gb, pb)
        used = set()
        for gi, (gc, _) in enumerate(gt):
            stat["gt"] += 1
            stat["gt_e"] += int(gc == ENEMY)
            stat["gt_a"] += int(gc == ALLY)
            stat["gt_boss"] += int(gc == BOSS)
            if not len(pr):
                continue
            pj = int(np.argmax(M[gi]))
            if M[gi][pj] >= MATCH_IOU and pj not in used:
                used.add(pj)
                stat["match"] += 1
                pc = pr[pj][0]
                stat["boss_hit"] += int(gc == BOSS)
                stat["m_e"] += int(gc == ENEMY)
                stat["m_a"] += int(gc == ALLY)
                if gc == ENEMY and pc == ALLY:
                    stat["e2f"] += 1
                if gc == ALLY and pc == ENEMY:
                    stat["f2e"] += 1
    return stat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="*", default=[])
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--holdout", action="store_true",
                    help="只评种子 holdout(帧号>=48) —— v10s 没训过的那部分")
    a = ap.parse_args()

    ws = a.weights or [
        str(RUNS / "battle_yolo26s_v10" / "weights" / "best.pt"),
        str(RUNS / "battle_yolo26s_v10s" / "weights" / "best.pt"),
    ]
    master, frames = load_gt(a.holdout)
    n_e = sum(1 for _, g in frames for c, _ in g if c == ENEMY)
    n_a = sum(1 for _, g in frames for c, _ in g if c == ALLY)
    scope = "种子 holdout(v10s 没训过)" if a.holdout else "全人审池"
    print(f"GT: {scope} · {len(frames)} 帧 · 我方 {n_a} / 敌方 {n_e}  "
          f"(conf={a.conf}, IoU={MATCH_IOU})\n")
    print(f"{'权重':<24}{'总召回':>8}{'敌方召回':>10}"
          f"{'敌→友(全)':>13}{'敌→友(匹配)':>14}{'友→敌(匹配)':>14}"
          f"{'Boss':>8}")
    for w in ws:
        if not os.path.exists(w):
            print(f"  {os.path.basename(os.path.dirname(os.path.dirname(w))):<22}"
                  f"  (权重不存在, 跳过)")
            continue
        s = evaluate(w, master, frames, a.conf)
        nm = os.path.basename(os.path.dirname(os.path.dirname(w)))
        print(f"  {nm:<22}{100*s['match']/max(s['gt'],1):>7.1f}%"
              f"{100*s['m_e']/max(s['gt_e'],1):>9.1f}%"
              f"{s['e2f']:>6}/{s['gt_e']:<3}"
              f"({100*s['e2f']/max(s['gt_e'],1):>4.1f}%)"
              f"{s['e2f']:>5}/{s['m_e']:<3}"
              f"({100*s['e2f']/max(s['m_e'],1):>5.1f}%)"
              f"{s['f2e']:>5}/{s['m_a']:<3}"
              f"({100*s['f2e']/max(s['m_a'],1):>5.1f}%)"
              f"{s['boss_hit']:>4}/{s['gt_boss']:<3}")
    print("\n⚠读法: 「敌→友(全)」降了可能只是检得更少; "
          "**「敌方召回」升 + 「敌→友(匹配)」降** 才是真的变好。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
