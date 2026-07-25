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

# ⛔帧源白名单 —— 只收**Android 内部取流**的尺寸(overlay 物理进不去)。
# 背景: server/app.py:2330 有条 2026-06-05 的纪律"trajectory 因 overlay 烧录
# 风险退役, 不再进 label 队列"(2026-05-28 删过 11 个烧录 run)。那条纪律是
# **DXcam/窗口抓取当主力**时定的; 现在主 tick 帧源已是 scrcpy → ADB
# (server/app.py:1255-1275), 但 fallback 链里的 `hf`(可能 DXcam)/`bitblt`
# 仍会烧, 而 **trajectory json 不记 frame_src**, 无法逐帧回溯来源。
# ⇒ 用**尺寸**当帧源指纹(实测 val 候选里的分布, 与帧源升级时间线严丝合缝):
#     3840x2160 = ADB screencap 4K        ✅ Android 内部
#     2560x1440 = scrcpy max_size          ✅ Android 内部
#     3612x2033 / 2364x1331 = 窗口抓取的非标准尺寸(随显示缩放漂移),
#                             且集中在 06-01~06-10 —— **正是烧录事故期** ⛔
# 代价: 少 51 帧候选。收益: 素材源纯度有**硬保证**, 不靠目检也不靠
# detect_overlay_burn.py(那个在战斗帧上误报严重, 见该文件说明)。
SAFE_FRAME_SIZES = {(3840, 2160), (2560, 1440)}

QUOTA_RICH = 12      # 出现 ≥20 帧的类: 每类挑这么多
MIN_GAP = 10        # 同 run 内两帧最小 tick 间隔(防近重复)
MIN_GAP_TIGHT = 2   # 第 2 轮补稀有类时放宽到这个间隔
CONF = 0.30


def _stamp(name: str):
    m = re.search(r"(\d{8})_(\d{6})", name)
    return f"{m.group(1)}_{m.group(2)}" if m else None


def tainted_stamps() -> tuple:
    """已进过训练/标注池的 run 时间戳 + **整天拉黑**的日期, 一律不许当 val。

    ⛔2026-07-25 workflow 审计实锤(v1 的防泄漏承诺没兑现):
    merge_flywheel_pools 把 run_<date>_<HHMMSS>_clean 合成
    run_<date>_merged_clean 后 rmtree 源目录 — 目录名时间戳被抹掉, 文件只剩
    143655_/0717a_ 这类前缀, _stamp() 全部 None → 这些 train 池的源 run 一个
    都进不了污染集。已落盘的 _val_v15_gap 实锤混进 ≥12 帧与 ui_v2 train 同
    session 的帧(20260711 的 143655/144712/162423 等)。
    逐前缀恢复不可靠(±1s 改名秒漂 + 0709m_/v8queue 前缀根本不带 HHMMSS)
    ⇒ **fail-closed: 凡名字带日期但没有完整时间戳的池, 整天拉黑**。
    代价是这几天的干净 run 也进不了 val — 溯源已被 merge 销毁, 宁缺勿泄。
    """
    stamps, days = set(), set()

    def _eat(name: str) -> None:
        s = _stamp(name)
        if s:
            stamps.add(s)
            return
        m = re.search(r"(\d{8})", name)
        if m:
            days.add(m.group(1))

    build = open(os.path.join(_ROOT, "scripts", "build_ui_v2.py"),
                 encoding="utf-8").read()
    for n in re.findall(r'"([A-Za-z_0-9]+)"', build):
        _eat(n)
    for d in glob.glob(os.path.join(_ROOT, "data/raw_images/*")):
        if not os.path.isdir(d):
            continue
        base = os.path.basename(d)
        _eat(base)
        if _stamp(base) is None and re.search(r"\d{8}", base):
            # ⚠跨天合并池(实测 run_20260717_merged_ui_clean 内含 0708p/0709m/
            # 0714a/0715a-c/0716a-b/0717a-g 批次) — 池名只透出一天, 其余天要从
            # **文件批次前缀** MMDD[a-z]_ 恢复, 每个批次天整天拉黑。
            my = re.search(r"(\d{4})\d{4}", base)
            year = my.group(1) if my else "2026"
            for f in os.listdir(d):
                m = re.match(r"(\d{4})[a-z]_", f)
                if m:
                    days.add(year + m.group(1))
    return stamps, days


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

    bad, bad_days = tainted_stamps()
    months = set(a.months.split(","))
    runs = []
    for d in sorted(glob.glob(os.path.join(_TRAJ, "run_2026*"))):
        if not os.path.isdir(d):
            continue
        n = os.path.basename(d)
        s = _stamp(n)
        if (not s or s[:6] not in months or s in bad
                or s[:8] in bad_days):
            continue
        runs.append(n)
    print(f"干净候选 run: {len(runs)} (排除 {len(bad)} 个受污染时间戳 + "
          f"整天拉黑 {len(bad_days)} 天: {sorted(bad_days)})")

    # 扫每帧含哪些 gap 类
    frame_cls = {}
    n_unsafe = 0
    for r in runs:
        for p in sorted(glob.glob(os.path.join(_TRAJ, r, "tick_*.json"))):
            jpg = p[:-5] + ".jpg"
            if not os.path.exists(jpg):
                continue
            try:
                j = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            # 帧源闸: 尺寸不在白名单 = 窗口抓取源, 有 overlay 烧录风险, 直接丢
            if (j.get("image_w"), j.get("image_h")) not in SAFE_FRAME_SIZES:
                n_unsafe += 1
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
    print(f"⛔帧源闸丢弃(尺寸非 Android 内部取流, 有 overlay 烧录风险): {n_unsafe} 帧")
    print(f"候选帧(含至少一个 gap 类): {len(frame_cls)}")
    print(f"可覆盖的 gap 类: {len(avail)}/{len(gap)}   "
          f"(零出现 {len(gap) - len(avail)} 类, 需定向补录)")

    # ⭐族感知(2026-07-25 用户点破 "有的是某些 cls 的变种/未点击前的状态"):
    # 同一物件的不同状态(领取_黄/_灰、关卡得星_3/_0、全部选择/全部选择灰…)
    # **必须成对进 val**, 否则测不出**状态混淆** —— 而那是最贵的一类 bug:
    # 把不可点当可点=空点卡流程, 把可点当不可点=漏活。
    # 实测最刺眼: `关卡得星_3` val 794 实例, `关卡得星_0` val **0**, 而
    # Challenge 假阳性事故的根因正是这两类混淆 → 那个回归现在根本测不到。
    # ⇒ ①同族里已有 val 的成员, 其**缺 val 的兄弟**配额翻倍(优先补齐)
    #   ②贪心打分时, 一帧若**同时含同族多态**加权(一帧顶两个, 且天然是最能
    #     暴露混淆的样本)
    try:
        # 直接按文件路径加载, 不引入 scripts/__init__.py(避免把 scripts
        # 变成 package 影响其它脚本的相对导入)
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "_cls_fam", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "audit_cls_families.py"))
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        build_families = _mod.build_families
        _n, _tr, _va, fams = build_families(only_ui=True)
        root_of, sibling_has_val = {}, {}
        for root, mem in fams.items():
            any_val = any(m["val"] > 0 for m in mem)
            for m in mem:
                root_of[m["name"]] = root
                sibling_has_val[m["name"]] = any_val
        boosted = []
        for c in list(quota):
            if sibling_has_val.get(c) and c in gap:
                quota[c] = min(avail[c], quota[c] * 2)
                boosted.append(c)
        if boosted:
            print(f"⭐族感知: {len(boosted)} 个类的配额翻倍(同族兄弟已有 val, "
                  f"这一态必须补上才测得出状态混淆): "
                  + ", ".join(boosted[:8]) + ("…" if len(boosted) > 8 else ""))
    except Exception as e:                                    # noqa: BLE001
        root_of = {}
        print(f"⚠族感知不可用({type(e).__name__}), 退回单类配额")

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
                # 族加权: 同一帧里出现**同族多个状态**的, 是最能暴露状态混淆
                # 的样本(模型必须在同一张图上把两态分开), 优先选。
                if gain > 0 and root_of:
                    roots = Counter(root_of[c] for c in hits if c in root_of)
                    gain += sum(v - 1 for v in roots.values() if v > 1)
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
    print("\n下一步: dashboard 侧栏 **L(标注中心)** → 打开 _val_v15_gap 人审, "
          "审完并入 ui_v3 的 val split")
    return 0


if __name__ == "__main__":
    sys.exit(main())
