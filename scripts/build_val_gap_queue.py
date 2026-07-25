# -*- coding: utf-8 -*-
"""为「缺 val 的类」定向挑帧 → 生成人审队列（v14 预标只当起点，不当 GT）。

## 为什么要有它
`ui_v2` 里 184 个已学会的类中有 **55 类在 val 里一个实例都没有**
（`入场键没解锁` 2141 训练框 / `关卡得星_0` 1282 / `战斗暂停` 1158 …）。
**测不出回归的能力等于没有保障**。这个脚本把 trajectory 里的干净帧定向挑出来，
补齐这些类的量尺。

## ⛔三条纪律（写死在流程里，别绕过）
1. **预标 ≠ 标注**。模型漏检的类，预标里就没有那个框；拿预标直接当 val
   等于自己给自己出卷子，永远满分。这里 v14 预标**只用来挑帧 + 给人审当起点**，
   `--prefill` 产出的 label 必须过 dashboard 人审才能进 val。
2. **防训练集泄漏**。飞轮管线会把 trajectory `run_X` 改名成 `run_X_clean`
   导进 raw_images 再进 train，所以按**时间戳前缀**（不是全名）排除所有已被
   `build_ui_v2.py` 引用或已在 raw_images 里的 run。
3. **同 run 内间隔采样**。相邻 tick 画面几乎一样，连着挑等于把同一张图放进
   val 十次，会把覆盖率虚高成"测过了"。

跑:
    py scripts/build_val_gap_queue.py              # 只出计划(不落盘)
    py scripts/build_val_gap_queue.py --write      # 落盘到 raw_images/_val_v15_gap
    py scripts/build_val_gap_queue.py --write --prefill   # 附带 v14 预标 label
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DS = r"D:\Project\ml_cache\models\yolo\dataset\ui_v2"
_TRAJ = os.path.join(_ROOT, "data", "trajectories")
_OUT = os.path.join(_ROOT, "data", "raw_images", "_val_v15_gap")

QUOTA_RICH = 12      # 出现 ≥20 帧的类: 每类挑这么多
MIN_GAP = 10        # 同 run 内两帧最小 tick 间隔(防近重复)
MIN_GAP_TIGHT = 2   # 第 2 轮补稀有类时放宽到这个间隔
CONF = 0.30


def _stamp(name: str):
    m = re.search(r"(\d{8})_(\d{6})", name)
    return f"{m.group(1)}_{m.group(2)}" if m else None


def tainted_stamps() -> set:
    """所有已进过训练/标注池的 run 时间戳 —— 一律不许当 val。"""
    out = set()
    build = open(os.path.join(_ROOT, "scripts", "build_ui_v2.py"),
                 encoding="utf-8").read()
    for n in re.findall(r'"([A-Za-z_0-9]+)"', build):
        s = _stamp(n)
        if s:
            out.add(s)
    for d in glob.glob(os.path.join(_ROOT, "data/raw_images/*")):
        if os.path.isdir(d):
            s = _stamp(os.path.basename(d))
            if s:
                out.add(s)
    return out


def label_counts(split: str) -> Counter:
    c = Counter()
    for lf in glob.glob(os.path.join(_DS, "labels", split, "*.txt")):
        try:
            for line in open(lf, encoding="utf-8"):
                t = line.split()
                if t:
                    c[int(t[0])] += 1
        except Exception:
            pass
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--prefill", action="store_true",
                    help="对选中帧跑 v14 生成预标 label(仍需人审)")
    ap.add_argument("--months", default="202606,202607")
    a = ap.parse_args()

    import yaml
    names = yaml.safe_load(open(os.path.join(_DS, "data.yaml"),
                                encoding="utf-8"))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    idx_of = {n: i for i, n in enumerate(names)}

    tr, va = label_counts("train"), label_counts("val")
    gap = {names[i] for i in tr if tr[i] > 0 and va.get(i, 0) == 0}
    print(f"缺 val 的类: {len(gap)}")

    bad = tainted_stamps()
    months = set(a.months.split(","))
    runs = []
    for d in sorted(glob.glob(os.path.join(_TRAJ, "run_2026*"))):
        if not os.path.isdir(d):
            continue
        n = os.path.basename(d)
        s = _stamp(n)
        if not s or s[:6] not in months or s in bad:
            continue
        runs.append(n)
    print(f"干净候选 run: {len(runs)} (已按时间戳排除 {len(bad)} 个受污染源)")

    # 扫每帧含哪些 gap 类
    frame_cls = {}
    for r in runs:
        for p in sorted(glob.glob(os.path.join(_TRAJ, r, "tick_*.json"))):
            jpg = p[:-5] + ".jpg"
            if not os.path.exists(jpg):
                continue
            try:
                j = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            hit = {b["cls"] for b in (j.get("yolo_boxes") or [])
                   if b.get("conf", 0) >= CONF and b["cls"] in gap}
            if hit:
                frame_cls[(r, p)] = hit

    avail = Counter()
    for h in frame_cls.values():
        for c in h:
            avail[c] += 1
    quota = {c: (QUOTA_RICH if avail[c] >= 20 else avail[c]) for c in avail}
    print(f"候选帧(含至少一个 gap 类): {len(frame_cls)}")
    print(f"可覆盖的 gap 类: {len(avail)}/{len(gap)}   "
          f"(零出现 {len(gap) - len(avail)} 类, 需定向补录)")

    # 覆盖贪心 + 同 run 间隔。
    # ⚠两轮: 稀有类往往只在少数**连续** tick 出现, 严格间隔会把它们整类滤掉
    # (实测 MIN_GAP=10 一轮时 `跳过战斗未选` 拿到 0/5)。所以第 1 轮用严间隔
    # 保多样性, 第 2 轮对仍未满额的类放宽到 MIN_GAP_TIGHT 补齐 —— 近重复帧
    # 虽然信息量低, 但**有总比没有强**: 一个类在 val 里 0 实例 = 永远测不到。
    need = dict(quota)
    picked_set, picked = set(), []
    last_tick = defaultdict(lambda: -10 ** 9)
    pool = sorted(frame_cls.items(), key=lambda kv: -len(kv[1]))

    def _greedy(min_gap: int):
        while True:
            best, best_gain = None, 0
            for (r, p), hits in pool:
                if (r, p) in picked_set:
                    continue
                t = int(re.search(r"tick_(\d+)", p).group(1))
                if t - last_tick[r] < min_gap:
                    continue
                gain = sum(1 for c in hits if need.get(c, 0) > 0)
                if gain > best_gain:
                    best, best_gain = (r, p, t, hits), gain
            if not best or best_gain <= 0:
                return
            r, p, t, hits = best
            picked_set.add((r, p))
            picked.append((r, p))
            last_tick[r] = t
            for c in hits:
                if need.get(c, 0) > 0:
                    need[c] -= 1

    _greedy(MIN_GAP)
    n1 = len(picked)
    _greedy(MIN_GAP_TIGHT)
    n2 = len(picked)
    # 第 3 轮: 对**全语料出现 <10 帧**的极稀有类, 无条件全收(忽略间隔)。
    # 贪心 + 间隔会持续卡死这类: 它们的几帧往往挤在同一个 run 的连续 tick 里,
    # 而该 run 的 last_tick 已被别的选择占住 → 实测 `跳过战斗未选` 两轮后仍 0/5。
    # val 里 0 实例 = 这个类**永远测不到**, 这个代价比"帧近重复"大得多。
    for (r, p), hits in pool:
        if (r, p) in picked_set:
            continue
        if any(need.get(c, 0) > 0 and avail[c] < 10 for c in hits):
            picked_set.add((r, p))
            picked.append((r, p))
            for c in hits:
                if need.get(c, 0) > 0:
                    need[c] -= 1
    print(f"贪心: 严间隔({MIN_GAP}) {n1} 帧 + 宽间隔({MIN_GAP_TIGHT}) "
          f"{n2 - n1} 帧 + 极稀有类全收 {len(picked) - n2} 帧")

    got = {c: quota[c] - need.get(c, 0) for c in quota}
    full = sum(1 for c in quota if need.get(c, 0) <= 0)
    print(f"\n选中 {len(picked)} 帧 → 满额 {full}/{len(quota)} 类")
    short = {c: (got[c], quota[c]) for c in quota if need.get(c, 0) > 0}
    if short:
        print(f"未满额 {len(short)} 类: "
              + ", ".join(f"{c}({v[0]}/{v[1]})" for c, v in
                          sorted(short.items(), key=lambda kv: kv[1][0])[:15]))
    print(f"\nval 覆盖预期: {len(va)} → {len(va) + full} 类 "
          f"({100*(len(va)+full)/len(tr):.1f}% of {len(tr)} 已学会的类)")

    if not a.write:
        print("\n(未落盘。加 --write 生成人审队列)")
        return 0

    os.makedirs(_OUT, exist_ok=True)
    shutil.copy(os.path.join(_ROOT, "data/raw_images/_classes.txt"),
                os.path.join(_OUT, "classes.txt"))
    for r, p in picked:
        base = f"{r}__{os.path.basename(p)[:-5]}"
        shutil.copy(p[:-5] + ".jpg", os.path.join(_OUT, base + ".jpg"))
    print(f"→ 已复制 {len(picked)} 帧到 {_OUT}")

    if a.prefill:
        sys.path.insert(0, _ROOT)
        from brain.pipeline import _run_yolo_on_image
        import cv2
        n_box = 0
        for r, p in picked:
            base = f"{r}__{os.path.basename(p)[:-5]}"
            img = cv2.imread(os.path.join(_OUT, base + ".jpg"))
            if img is None:
                continue
            h, w = img.shape[:2]
            lines = []
            for b in _run_yolo_on_image(img, w, h, context="ui"):
                i = idx_of.get(b.cls_name)
                if i is None or b.confidence < CONF:
                    continue
                cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
                lines.append(f"{i} {cx:.6f} {cy:.6f} "
                             f"{b.x2-b.x1:.6f} {b.y2-b.y1:.6f}")
            open(os.path.join(_OUT, base + ".txt"), "w",
                 encoding="utf-8").write("\n".join(lines) + "\n")
            n_box += len(lines)
        print(f"→ v14 预标 {n_box} 框 (⛔仅供人审起点, 未审不得进 val)")
    print("\n下一步: dashboard → Annotate → 打开 _val_v15_gap 人审, "
          "审完并入 ui_v3 的 val split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
