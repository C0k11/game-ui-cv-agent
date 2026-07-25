# -*- coding: utf-8 -*-
"""⛔敌我混淆专项审计 —— 用户 2026-07-25: "经常敌方标记成友方"。

在**人审过的**池上量 battle 模型的身份类混淆矩阵, 并回答三个决策问题:
  ① 单帧混淆率到底多少(不是感觉, 是数字)
  ② track 级类别投票能纠回多少(这才是修翻转的正解, ReID 只是让 track 更长)
  ③ 位置先验(我方偏左/敌方偏右)有多强, 能不能当兜底判据

⚠分母口径: 只统计**匹配上 GT 的预测框**(IoU≥0.45)。漏检/误检是另一回事,
单独报, 绝不混进"混淆率"里稀释或夸大。

跑: py scripts/audit_side_confusion.py <pool_substr> [--track]
"""
from __future__ import annotations

import glob
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from vision.io_utils import imread_any  # noqa: E402

RAW = Path(_ROOT) / "data" / "raw_images"
IDENTITY = {476, 477, 478, 479, 480, 481, 482, 483}
MATCH_IOU = 0.45
CONF = 0.35


def iou_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def load_gt(txt: Path) -> list:
    """→ [(master_cls, xyxy_norm)]  仅身份类。"""
    out = []
    if not txt.exists():
        return out
    for ln in txt.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        if c not in IDENTITY:
            continue
        xc, yc, w, h = map(float, p[1:5])
        out.append((c, np.array([xc - w / 2, yc - h / 2,
                                 xc + w / 2, yc + h / 2])))
    return out


def main() -> int:
    pat = sys.argv[1] if len(sys.argv) > 1 else "025638"
    use_track = "--track" in sys.argv
    pools = [Path(p) for p in sorted(glob.glob(str(RAW / "*")))
             if os.path.isdir(p) and pat in os.path.basename(p)]
    if not pools:
        print(f"没有匹配 '{pat}' 的池")
        return 1

    from track_prefill import _latest_battle_weights
    from ultralytics import YOLO
    master = [l.strip() for l in
              open(RAW / "_classes.txt", encoding="utf-8") if l.strip()]
    n2i = {n: i for i, n in enumerate(master)}
    model = YOLO(_latest_battle_weights())
    m2master = {i: n2i[n] for i, n in model.names.items() if n in n2i}

    conf_mat = Counter()          # (gt_cls, pred_cls) -> n
    miss = Counter()              # gt 无匹配预测
    ghost = Counter()             # 预测无匹配 gt
    pos_gt = defaultdict(list)    # gt_cls -> [cx]
    track_obs = defaultdict(list)  # (pool,tid) -> [(fi, pred_cls, gt_cls)]

    for pool in pools:
        jpgs = sorted(pool.glob("*.jpg"),
                      key=lambda p: int(re.search(r"(\d+)", p.stem).group(1))
                      if re.search(r"(\d+)", p.stem) else 0)
        print(f"\npool={pool.name}  {len(jpgs)} 帧"
              f"{' [track模式]' if use_track else ''}")
        for fi, jp in enumerate(jpgs):
            gt = load_gt(jp.with_suffix(".txt"))
            for c, b in gt:
                pos_gt[c].append((b[0] + b[2]) / 2)
            img = imread_any(str(jp))
            if img is None:
                continue
            H, W = img.shape[:2]
            if use_track:
                r = model.track(
                    img, persist=True, conf=0.10, iou=0.5, imgsz=960,
                    tracker=str(Path(_ROOT) / "configs" / "trackers" /
                                "bytetrack_axis3fps.yaml"), verbose=False)[0]
            else:
                r = model.predict(img, conf=CONF, iou=0.5, imgsz=960,
                                  verbose=False)[0]
            pred = []
            tids = []
            if r.boxes is not None:
                for b in r.boxes:
                    mi = m2master.get(int(b.cls[0]))
                    if mi not in IDENTITY:
                        continue
                    if float(b.conf[0]) < CONF and not use_track:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                    pred.append((mi, np.array([x1 / W, y1 / H, x2 / W, y2 / H])))
                    tids.append(int(b.id[0]) if (use_track and b.id is not None)
                                else -1)
            gb = np.array([b for _, b in gt]) if gt else np.zeros((0, 4))
            pb = np.array([b for _, b in pred]) if pred else np.zeros((0, 4))
            M = iou_mat(gb, pb)
            used_p = set()
            for gi in range(len(gt)):
                if not len(pred):
                    miss[gt[gi][0]] += 1
                    continue
                pj = int(np.argmax(M[gi]))
                if M[gi][pj] >= MATCH_IOU and pj not in used_p:
                    used_p.add(pj)
                    conf_mat[(gt[gi][0], pred[pj][0])] += 1
                    if use_track and tids[pj] >= 0:
                        track_obs[(pool.name, tids[pj])].append(
                            (fi, pred[pj][0], gt[gi][0]))
                else:
                    miss[gt[gi][0]] += 1
            for pj in range(len(pred)):
                if pj not in used_p:
                    ghost[pred[pj][0]] += 1

    nm = {i: master[i] for i in IDENTITY}
    tot = sum(conf_mat.values())
    if miss:
        print(f"\n漏检按类: " + "  ".join(
            f"{nm[c]}={n}" for c, n in sorted(miss.items(), key=lambda kv: -kv[1])))
    if ghost:
        print(f"多余框按类: " + "  ".join(
            f"{nm[c]}={n}" for c, n in sorted(ghost.items(), key=lambda kv: -kv[1])))
    print(f"\n{'='*70}\n匹配上的框(IoU≥{MATCH_IOU}): {tot}   "
          f"漏检 {sum(miss.values())}   多余框 {sum(ghost.values())}")
    print("=" * 70)
    print("混淆矩阵 (行=人审GT, 列=模型预测)")
    gts = sorted({g for g, _ in conf_mat})
    prs = sorted({p for _, p in conf_mat})
    print(f"{'GT\\pred':<10}" + "".join(f"{nm[p][:6]:>8}" for p in prs)
          + f"{'错率':>8}")
    for g in gts:
        row = sum(conf_mat[(g, p)] for p in prs)
        wrong = row - conf_mat[(g, g)]
        print(f"{nm[g][:9]:<10}"
              + "".join(f"{conf_mat[(g,p)]:>8}" for p in prs)
              + f"{100*wrong/max(row,1):>7.1f}%")

    e2f = conf_mat[(477, 476)]
    f2e = conf_mat[(476, 477)]
    n_e = sum(conf_mat[(477, p)] for p in prs)
    n_f = sum(conf_mat[(476, p)] for p in prs)
    print(f"\n⭐用户主诉「敌方被标成友方」: {e2f}/{n_e} = "
          f"{100*e2f/max(n_e,1):.2f}%   (反向 友方→敌方 {f2e}/{n_f} = "
          f"{100*f2e/max(n_f,1):.2f}%)")

    print(f"\n{'='*70}\n位置先验(人审 GT 的 cx 分布) — 我方偏左/敌方偏右?")
    print("=" * 70)
    for c in sorted(pos_gt):
        v = np.array(pos_gt[c])
        print(f"  {nm[c][:8]:<9} n={len(v):<5} cx 均值 {v.mean():.3f}  "
              f"p10 {np.percentile(v,10):.3f}  中位 {np.median(v):.3f}  "
              f"p90 {np.percentile(v,90):.3f}")
    if 476 in pos_gt and 477 in pos_gt:
        a, b = np.array(pos_gt[476]), np.array(pos_gt[477])
        best, bestacc = 0.0, 0.0
        for t in np.arange(0.05, 0.96, 0.01):
            acc = ((a < t).sum() + (b >= t).sum()) / (len(a) + len(b))
            if acc > bestacc:
                best, bestacc = t, acc
        print(f"  → 单纯用 cx 阈值 {best:.2f} 分敌我: 准确率 "
              f"{100*bestacc:.1f}%  (够不够当兜底判据看这个数)")

    if use_track and track_obs:
        print(f"\n{'='*70}\ntrack 级类别投票能纠回多少")
        print("=" * 70)
        fixed = broke = kept_bad = 0
        for key, obs in track_obs.items():
            if len(obs) < 3:
                continue
            vote = Counter(p for _, p, _ in obs).most_common(1)[0][0]
            for _fi, p, g in obs:
                if p != g and vote == g:
                    fixed += 1
                elif p == g and vote != g:
                    broke += 1
                elif p != g and vote != g:
                    kept_bad += 1
        print(f"  轨迹数(≥3帧) {sum(1 for o in track_obs.values() if len(o)>=3)}")
        print(f"  ✅投票纠回错框: {fixed}")
        print(f"  ⛔投票**弄坏**本来对的框: {broke}   ← 这个必须接近 0 才敢用")
        print(f"  ⚠仍然错(整条轨迹都错): {kept_bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
