# -*- coding: utf-8 -*-
"""生成 ui cls 语料库 -- 事实层自动生成, 语义层保留人写的。

事实层(自动, 保证不过期): idx / 类名 / v16 train,val 框数 / 在不在 vocab /
   哪些 flow 引用 / 是不是废案 / 同族的兄弟状态。
语义层(人写): 老 `data/ui_cls_semantics.md` 里还成立的段落照搬, 其余标 TODO。

* 分族规则: cls 是「物件 x 状态」不是独立目标（memory ui_cls_semantics）。
   按类名去掉状态后缀归族 —— **同族的 val 覆盖情况必须一起看**,
   一个族只有一个状态有 val, 状态混淆就永远测不出来。
"""
from __future__ import annotations

import io
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"D:/Project/ai game secretary")
SCRATCH = Path(__file__).resolve().parent
CLASSES = ROOT / "data" / "raw_images" / "_classes.txt"
OUT = ROOT / "data" / "ui_cls_semantics.md"

# 状态后缀 -> 归族时剥掉。顺序有意义(长的先剥)。
STATE_SUF = [
    "_已选中", "_未选择", "_已选择", "_不可用", "_可点击", "_未完成", "_已完成",
    "_灰色", "_已占", "_迷雾", "_selected", "_灰", "_黄", "_红", "_蓝", "_绿", "_白",
]

names = [l.strip() for l in io.open(CLASSES, encoding="utf-8")]
DS = Path("D:/Project/ml_cache/models/yolo/dataset/ui_v2")


def _count(split):
    from collections import Counter
    c, frames = Counter(), 0
    for f in (DS / "labels" / split).glob("*.txt"):
        frames += 1
        for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = ln.split()
            if p:
                c[int(p[0])] += 1
    return c, frames


TR, NTR = _count("train")
VA, NVA = _count("val")

# vocab 常量 -> 类名
vsrc = io.open(ROOT / "routing_v2" / "state" / "vocab.py", encoding="utf-8").read()
QUOTED = r'"([^"\n]+)"'
vocab_names = set(re.findall(QUOTED, vsrc))
const_of = defaultdict(list)
for m in re.finditer(r"^([A-Z][A-Z0-9_]*)\s*=\s*(.+)$", vsrc, re.M):
    const, val = m.group(1), m.group(2)
    for nm in re.findall(QUOTED, val):
        const_of[nm].append(const)

# 哪个 flow 引用了哪个常量
uses = defaultdict(set)
for p in sorted((ROOT / "routing_v2" / "flow").glob("*.py")) + \
         sorted((ROOT / "routing_v2" / "act").glob("*.py")) + \
         sorted((ROOT / "routing_v2" / "state").glob("pages.py")):
    src = io.open(p, encoding="utf-8").read()
    for nm, consts in const_of.items():
        for c in consts:
            if re.search(rf"\bV\.{c}\b|\b{c}\b", src):
                uses[nm].add(p.stem)
                break


def fam(n: str) -> str:
    s = n
    for suf in STATE_SUF:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s


def owner(i: int) -> str:
    """这个 idx 归哪个模型。唯一权威 = scripts/cls_domains.py。

    禁 `_classes.txt` 是**三个模型共用的 master 表**（memory val_set_crisis:
       「485 是三模型共用 master 表别拿它当 UI 分母」）。不分段就会把 252 个
       学生名当成「UI 模型 train=0 的缺口」—— ui_v2 数据集里本来就不该有它们。
    2026-08-13 修: 旧本地写法 `476..483 -> battle` 漏了 484 战斗失败
       (build_battle_v11 REMAP 在训它), UI 分母虚高 1。
    """
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from cls_domains import domain
    return domain(i)


families = defaultdict(list)
for i, n in enumerate(names):
    if n.startswith("_废弃") or owner(i) != "ui":
        continue
    families[fam(n)].append(i)

dead = [(i, n) for i, n in enumerate(names) if n.startswith("_废弃")]
live = [(i, n) for i, n in enumerate(names) if not n.startswith("_废弃")]
ui_live = [(i, n) for i, n in live if owner(i) == "ui"]
t0 = sum(1 for i, _ in ui_live if TR.get(i, 0) == 0)
v0 = sum(1 for i, _ in ui_live if VA.get(i, 0) == 0)
nov = sum(1 for i, n in ui_live if n not in vocab_names)
av_live = [(i, n) for i, n in live if owner(i) == "avatar"]
av_tr = sum(1 for i, _ in av_live if TR.get(i, 0) > 0)

L = []
A = L.append
A(f"# ui cls 语料库 -- v16 / nc={len(names)}")
A("")
A("> **写任何检测/导航逻辑前先查这里**。判断一律用 cls, OCR 只读数字。")
A("> 事实层(idx / 框数 / vocab / 引用) 由 `scripts/gen_cls_corpus.py` 从")
A("> `_classes.txt` + v16 数据集 + 源码**自动生成**, 不会过期; 语义列要人写。")
A("")
A("## 一眼看全")
A("")
A("| | |")
A("|---|---|")
A(f"| master 表总行数 | {len(names)}（现役 {len(live)} / 废案 {len(dead)}） |")
A(f"| 其中 **UI 模型自己的类** | **{len(ui_live)}**（这才是 UI 的分母） |")
A(f"| 其中 学生名（avatar 模型, idx 143-394） | {len(av_live)}，在 ui_v2 里有框的 {av_tr} |")
A("| 其中 战场实体（battle 模型, idx 476-483） | 8（用户点名不进 UI） |")
A(f"| v16 ui_v2 train | {sum(TR.values()):,} 框 / {NTR:,} 帧 |")
A(f"| v16 ui_v2 val | {sum(VA.values()):,} 框 / {NVA:,} 帧 |")
A(f"| UI 类 **train=0**（模型没学过） | **{t0}** |")
A(f"| UI 类 **val=0**（测不出来） | **{v0}** |")
A(f"| UI 类 **不在 vocab**（代码用不上） | **{nov}** |")
A("")
A("禁 **`_classes.txt` 是三个模型共用的 master 表** —— 拿 528 当 UI 分母会")
A("   把 252 个学生名算成「UI 缺口」（[[val_set_crisis]] 那条教训）。上表已分段。")
A("禁 **train=0 不一定是漏训, 也可能是废案没标名**（[[live_daily_2026_08_07]]）。")
A("禁 **val=0 的族状态混淆永远测不出来** —— 同族必须成对补 val。")
A("")
A("## 按族看（cls 是「物件 x 状态」，不是独立目标）")
A("")
A("状态后缀相同词根的归为一族。**同族要一起审**: 一个族只有一个状态有 val,")
A("模型把亮态认成灰态这种错在 val 上完全看不出来（v15 购买/购买灰色就是这么漏的）。")
A("")
A("| 族 | idx | 类名 | train | val | vocab | 被谁用 |")
A("|---|---|---|---|---|---|---|")
for f in sorted(families, key=lambda k: (-len(families[k]), k)):
    idxs = families[f]
    for j, i in enumerate(idxs):
        n = names[i]
        tr, va = TR.get(i, 0), VA.get(i, 0)
        vk = "o" if n in vocab_names else "**x**"
        us = ",".join(sorted(uses.get(n, []))) or "-"
        head = f"**{f}**" if j == 0 and len(idxs) > 1 else ("" if j else f)
        warn = ""
        if tr == 0:
            warn = " 禁train=0"
        elif va == 0:
            warn = " 注意val=0"
        A(f"| {head} | {i} | `{n}`{warn} | {tr:,} | {va:,} | {vk} | {us} |")
A("")
A("## 现役但代码里用不上（不在 vocab）")
A("")
A("这些类模型认得出来, 但 `routing_v2/state/vocab.py` 里没有常量 -> **flow 写不了判据**。")
A("要用先补常量。")
A("")
A("| idx | 类名 | train | val |")
A("|---|---|---|---|")
for i, n in ui_live:
    if n not in vocab_names:
        A(f"| {i} | `{n}` | {TR.get(i,0):,} | {VA.get(i,0):,} |")
A("")
A("## 废案（禁 只能改名不能删行 —— `_classes.txt` 按行号索引）")
A("")
A("| idx | 类名 |")
A("|---|---|")
for i, n in dead:
    A(f"| {i} | `{n}` |")
A("")

io.open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"写出 {OUT}  {len(L)} 行")
print(f"现役 {len(live)} / 废案 {len(dead)} / train=0 {t0} / val=0 {v0} / 不在vocab {nov}")
