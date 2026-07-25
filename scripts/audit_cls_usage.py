# -*- coding: utf-8 -*-
"""cls 四象限审计 —— 谁是谁的模型 / 谁是死判据 / 谁是废案。

⛔为什么要有这个脚本(2026-07-25 建, 起因见下):
全量审计一次就逮到 **shop.py 的 4 道金钱守卫全是死码** —— 它们检的
`青辉石商店_已选择` 训练 0 框、92k tick 实战 0 检出, 模型从来不认识它,
守卫从加上那天(注释还标着 deep-dive C8 / r2 C4)起一次都没生效过, 却让人
以为有四层保护。**零点防线伪装成四道, 比单点防线更危险。**
这类 bug 完全静默: 代码能跑、日志无异常、测试也过 —— 只有把"代码引用的 cls"
和"训练集真有的 cls"对撞才看得见。

跑: py scripts/audit_cls_usage.py [--fail-on-dead]
    --fail-on-dead: 发现"代码在用但训练零框"的类就 exit 1(可挂 CI/回归)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MASTER = os.path.join(_ROOT, "data", "raw_images", "_classes.txt")

# master 索引域划分(与训练路由约定一致)
_AVATAR_LO, _AVATAR_HI = 143, 394
_BATTLE_LO = 476


def domain(i: int) -> str:
    if _AVATAR_LO <= i <= _AVATAR_HI:
        return "avatar"
    if i >= _BATTLE_LO:
        return "battle"
    return "ui"


def load_master() -> list:
    with open(_MASTER, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def count_label_boxes() -> Counter:
    """raw_images 标注池里每个 master cls_id 的真实框数。"""
    cnt = Counter()
    for lf in glob.glob(os.path.join(_ROOT, "data/raw_images", "**", "*.txt"),
                        recursive=True):
        if os.path.basename(lf).startswith("_"):
            continue
        try:
            with open(lf, encoding="utf-8") as f:
                for line in f:
                    t = line.split()
                    if t:
                        cnt[int(t[0])] += 1
        except Exception:
            continue
    return cnt


def _strip_comments(text: str) -> str:
    """剥掉 # 注释和三引号 docstring。

    ⛔为什么必须剥(2026-07-25 我自己踩的坑): 第一版直接全文匹配, 把
    **注释里提到的类名**当成"代码在用" —— `大决战` 报 17 处引用、看着像
    承重判据, 实际全出现在我刚写的一句注释里; `升序` 更离谱, 命中的是
    中文短语"按 cy 升序"。差点据此给出一份错误的死判据清单。
    """
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    text = re.sub(r"'''[\s\S]*?'''", "", text)
    return "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())


def scan_code() -> str:
    src = []
    for sub in ("brain", "server", "scripts", "tests"):
        for f in glob.glob(os.path.join(_ROOT, sub, "**", "*.py"),
                           recursive=True):
            if os.path.basename(f) == os.path.basename(__file__):
                continue
            try:
                src.append(_strip_comments(open(f, encoding="utf-8").read()))
            except Exception:
                pass
    return "\n".join(src)


def scan_detections() -> Counter:
    """trajectory 实战检出(可能很慢, 只在 --with-det 时跑)。"""
    cnt = Counter()
    for run in glob.glob(os.path.join(_ROOT, "data/trajectories/run_2026*")):
        if not os.path.isdir(run):
            continue
        for p in glob.glob(os.path.join(run, "tick_*.json")):
            try:
                j = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            for b in (j.get("yolo_boxes") or []):
                cnt[b.get("cls")] += 1
    return cnt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-on-dead", action="store_true",
                    help="代码引用了训练零框的 cls → exit 1")
    ap.add_argument("--with-det", action="store_true",
                    help="同时统计 trajectory 实战检出(慢)")
    a = ap.parse_args()

    master = load_master()
    boxes = count_label_boxes()
    code = scan_code()
    det = scan_detections() if a.with_det else Counter()

    # ui_classes 常量名 → 值, 用来判断"是不是真被 skill 逻辑引用"
    sys.path.insert(0, _ROOT)
    from brain.skills import ui_classes as UC
    const = {k: v for k, v in vars(UC).items()
             if isinstance(v, str) and k.isupper()}

    def referenced(name: str) -> int:
        """该 cls 被**代码**引用的次数。

        常量类走 `UC.XXX`; 无常量的走**带引号的完整字面量** —— 绝不用裸子串
        匹配: `房间区域` 是 `房间区域未解锁` 的子串, 裸匹配会把后者算成前者
        的引用(第一版就这么误报了 11 处)。
        """
        n = 0
        has_const = False
        for k, v in const.items():
            if v == name:
                has_const = True
                n += len(re.findall(r"UC\." + k + r"\b", code))
        if not has_const:
            q = re.escape(name)
            n += len(re.findall(r'"' + q + r'"', code))
            n += len(re.findall(r"'" + q + r"'", code))
        return n

    print(f"master {len(master)} 类  |  标注池 {sum(boxes.values())} 框")
    agg = {}
    for i, n in enumerate(master):
        d = agg.setdefault(domain(i), [0, 0])
        d[0] += 1
        if boxes.get(i, 0) > 0:
            d[1] += 1
    for k, v in agg.items():
        print(f"  {k:8} {v[0]:4d} 类, 其中有标注 {v[1]:4d}")

    dead = []
    for i, n in enumerate(master):
        if boxes.get(i, 0) == 0 and referenced(n) > 0:
            dead.append((n, i, referenced(n), det.get(n, 0)))
    dead.sort(key=lambda t: -t[2])

    print(f"\n⛔死判据(代码在用 + 训练零框) {len(dead)} 条:")
    for n, i, r, dv in dead:
        mark = "  ← 承重!" if r >= 2 else ""
        print(f"   idx{i:4d} {n:24} 代码引用 {r:2d} 处"
              + (f" 实战检出 {dv}" if a.with_det else "") + mark)

    zombie = [(i, n) for i, n in enumerate(master)
              if boxes.get(i, 0) == 0 and referenced(n) == 0]
    print(f"\n💤废案(零标注 + 零引用) {len(zombie)} 条:")
    for i, n in zombie:
        print(f"   idx{i:4d} {n:24} [{domain(i)}]")

    weak = [(n, i, boxes.get(i, 0)) for i, n in enumerate(master)
            if 0 < boxes.get(i, 0) < 30 and referenced(n) > 0]
    weak.sort(key=lambda t: t[2])
    print(f"\n⚠️低样本(<30 框且代码在用) {len(weak)} 条:")
    for n, i, b in weak:
        print(f"   idx{i:4d} {n:24} 训练框 {b}")

    if a.fail_on_dead and dead:
        print(f"\n❌ {len(dead)} 条死判据 —— 代码依赖模型不认识的 cls")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
