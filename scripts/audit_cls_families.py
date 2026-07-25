# -*- coding: utf-8 -*-
"""cls **族关系**审计 —— 485 类不是 485 个独立目标, 是「物件 × 状态」。

⛔用户 2026-07-25 点破: "你得思考 cls 之间的联系, 有的都是某些 cls 的变种
或者是没被点击之前的状态之类的"。

为什么这件事值钱 —— 状态混淆是**最贵的一类 bug**:
  把「不可点」当「可点」→ 空点、卡流程;  把「可点」当「不可点」→ 漏活、资源作废。
而只要 val 里**只有族里的一态**, 这类 bug 就**永远测不出来**。
实测到的最刺眼一例: `关卡得星_3` val 794 实例, 而 `关卡得星_0` val **0** —— 可
memory[[ui-cls-semantics]] 早就记着"Challenge 行灰星=独立类 _0, 任何得星族
startswith/子串闸都会在 Challenge tab 误通过", 那起 Challenge 假阳性事故的回归,
**用现有 val 根本测不到**。

三条用途:
 ① val 必须**成对**: 同族成员都要有实例, 否则测不出状态混淆
 ② 训练配额按族均衡: 同族一个 1425 框一个 1 框, 模型必然偏向多的那个
 ③ 代码判据按族审: 用了族里某一态, 就要问"另一态出现时会怎样"

跑: py scripts/audit_cls_families.py [--all-domains] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DS = r"D:\Project\ml_cache\models\yolo\dataset\ui_v2"

# 状态后缀模式 —— 剥掉后剩下的是"物件词根"。
# ⚠纯字符串模式, 不猜语义: 宁可漏归族(少报), 也不要把两个不相干的类硬凑一族。
STATE_PATTERNS = [
    (r"_黄$", "可用/高亮"),
    (r"_灰$|_灰色$", "不可用"),
    (r"灰色$", "不可用"),
    (r"_可点击$", "可点击"),
    (r"_已选择$|_已选中$", "已选中"),
    (r"已选中$|已选择$", "已选中"),
    (r"_未选择$|未选择$", "未选中"),
    (r"高亮$", "高亮"),
    (r"_已完成$|已完成$", "已完成"),
    (r"_未完成$|未完成$", "未完成"),
    (r"未解锁$", "未解锁"),
    (r"不可用$", "不可用"),
    (r"_已打$|已打$", "已完成"),
    (r"_已看$|已看$", "已完成"),
    (r"_0$", "空/未达成"),
    (r"_3$", "满/已达成"),
    (r"选中$", "已选中"),
    (r"灰$", "不可用"),
    (r"开启$", "开"),
    (r"关闭$", "关"),
]

_AVATAR_LO, _AVATAR_HI, _BATTLE_LO = 143, 394, 476


def is_ui(i: int) -> bool:
    return i < _AVATAR_LO or _BATTLE_LO > i >= 395


def strip_state(name: str):
    for pat, st in STATE_PATTERNS:
        if re.search(pat, name):
            root = re.sub(pat, "", name).rstrip("_")
            if root:
                return root, st
    return name, None


def counts(split: str) -> Counter:
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


def build_families(only_ui: bool = True):
    import yaml
    names = yaml.safe_load(open(os.path.join(_DS, "data.yaml"),
                                encoding="utf-8"))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    tr, va = counts("train"), counts("val")
    fam = defaultdict(list)
    for i, n in enumerate(names):
        if tr.get(i, 0) <= 0:
            continue
        if only_ui and not is_ui(i):
            continue
        root, st = strip_state(n)
        fam[root].append({"idx": i, "name": n, "state": st,
                          "train": tr.get(i, 0), "val": va.get(i, 0)})
    return names, tr, va, {k: v for k, v in fam.items() if len(v) > 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-domains", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    names, tr, va, multi = build_families(only_ui=not a.all_domains)
    n_mem = sum(len(v) for v in multi.values())
    print(f"多态族: {len(multi)} 族 / {n_mem} 类"
          f"{'' if a.all_domains else ' (仅 UI 域)'}\n")

    print("=" * 76)
    print("⛔ val 只覆盖族内一部分 → **状态混淆测不出来**")
    print("=" * 76)
    partial = []
    for root, mem in sorted(multi.items()):
        hasv = [m for m in mem if m["val"] > 0]
        if 0 < len(hasv) < len(mem):
            partial.append((root, mem))
    for root, mem in partial:
        print(f"\n【{root}】")
        for m in sorted(mem, key=lambda x: -x["train"]):
            flag = "✅val" if m["val"] > 0 else "⛔无val"
            print(f"    {m['name']:22} {str(m['state'] or '-'):8} "
                  f"train {m['train']:6d}  val {m['val']:5d}  {flag}")
    print(f"\n小计: {len(partial)} 个族只测得出一部分状态")

    print("\n" + "=" * 76)
    print("⚠️ 族内训练样本失衡 (>20x) → 模型偏向样本多的那一态")
    print("=" * 76)
    imbal = []
    for root, mem in sorted(multi.items()):
        ts = [m["train"] for m in mem if m["train"] > 0]
        if len(ts) > 1 and max(ts) / max(min(ts), 1) > 20:
            imbal.append((root, mem, max(ts) / max(min(ts), 1)))
    for root, mem, r in sorted(imbal, key=lambda x: -x[2]):
        print(f"\n【{root}】 {r:.0f}x")
        for m in sorted(mem, key=lambda x: -x["train"]):
            print(f"    {m['name']:22} train {m['train']:6d}  val {m['val']:5d}")

    print("\n" + "=" * 76)
    print("全部多态族")
    print("=" * 76)
    print("  " + " · ".join(f"{k}({len(v)})" for k, v in
                            sorted(multi.items(), key=lambda kv: -len(kv[1]))))

    if a.json:
        json.dump({"families": multi,
                   "val_partial": [p[0] for p in partial],
                   "imbalanced": [i[0] for i in imbal]},
                  open(a.json, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=1)
        print(f"\n→ {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
