# -*- coding: utf-8 -*-
"""cls 四象限审计 —— 谁是谁的模型 / 谁是死判据 / 谁是废案。

为什么要有这个脚本(2026-07-25 建, 起因见下):
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

# master 索引域划分: 唯一权威在 scripts/cls_domains.py(2026-08-13 修:
#    旧写法 `i >= 476 -> battle` 把 484-527 共 44 槽/41 个活 UI 类错分成
#    battle -- 常设审计自己在撒谎; 旧 load_master 还会剔空行造成 idx 错位)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cls_domains import domain, load_master as _load_master_indexed  # noqa: E402


def load_master() -> list:
    return _load_master_indexed(_MASTER)


# val 池必须排除(2026-07-25 第二次"审计工具自身有 bug"):
# 旧版把 raw_images 下**所有** txt 都当训练标注统计, 包括 `_val_v8flywheel`
# `_val_v12flywheel_0616` `_val_v15_gap` 这些**held-out val 池**。后果不只是
# 框数虚高 —— 一个类若**只在 val 里有框、train 里没有**, 就会被误判成"有标注",
# 于是**该报的死判据不报了**(漏报, 比多报危险)。
_VAL_DIR_MARKERS = ("_val", "val_pool")


def _val_source_names() -> set:
    """build_ui_v2.VAL_SOURCES 的目录名(文本抠取, 不 exec 避免顶层副作用)。

    2026-07-25 第三修: run_20260606_flywheel 这类 **v7 主 val 源**名字不带
    `_val` 字样, 光靠 _VAL_DIR_MARKERS 认不出  同款"val 当 train"漏报向量
    没堵死; 另有 4 个 `_labels_bak` 备份目录被双份计数(虚高 ~17%)。
    别再用目录名子串猜 val — 直接对齐构建脚本的真相。"""
    try:
        src = open(os.path.join(_ROOT, "scripts", "build_ui_v2.py"),
                   encoding="utf-8").read()
        m = re.search(r"VAL_SOURCES\s*=\s*\[([\s\S]*?)\]", src)
        return set(re.findall(r'"([^"]+)"', m.group(1))) if m else set()
    except Exception:
        return set()


_VAL_SRC_NAMES = _val_source_names()


def _is_val_pool(path: str) -> bool:
    parts = os.path.normpath(path).split(os.sep)
    if any(any(m in p for m in _VAL_DIR_MARKERS) for p in parts):
        return True
    if any("_labels_bak" in p or p == "_backups" for p in parts):
        return True
    return any(p in _VAL_SRC_NAMES for p in parts)


def count_label_boxes(include_val: bool = False) -> Counter:
    """raw_images 标注池里每个 master cls_id 的真实框数（默认**只算 train**）。"""
    cnt = Counter()
    for lf in glob.glob(os.path.join(_ROOT, "data/raw_images", "**", "*.txt"),
                        recursive=True):
        if os.path.basename(lf).startswith("_"):
            continue
        if not include_val and _is_val_pool(lf):
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

    为什么必须剥(2026-07-25 我自己踩的坑): 第一版直接全文匹配, 把
    **注释里提到的类名**当成"代码在用" —— `大决战` 报 17 处引用、看着像
    承重判据, 实际全出现在我刚写的一句注释里; `升序` 更离谱, 命中的是
    中文短语"按 cy 升序"。差点据此给出一份错误的死判据清单。
    """
    text = re.sub(r'"""[\s\S]*?"""', "", text)
    text = re.sub(r"'''[\s\S]*?'''", "", text)
    return "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())


#: bot 运行时决策链 —— 只有这里出现的 cls 才叫「承重判据」。
DECISION_SUBS = ("brain",)
#: 离线工具/一次性脚本/测试 —— 同名字符串出现在这里**不代表** bot 靠它决策。
AUX_SUBS = ("server", "scripts", "tests")


def scan_code(exclude_basenames: tuple = (), subs: tuple = None) -> str:
    src = []
    for sub in (subs if subs is not None else DECISION_SUBS + AUX_SUBS):
        for f in glob.glob(os.path.join(_ROOT, sub, "**", "*.py"),
                           recursive=True):
            if os.path.basename(f) == os.path.basename(__file__):
                continue
            if os.path.basename(f) in exclude_basenames:
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
                    help="代码引用了训练零框的 cls  exit 1")
    ap.add_argument("--with-det", action="store_true",
                    help="同时统计 trajectory 实战检出(慢)")
    a = ap.parse_args()

    master = load_master()
    boxes = count_label_boxes()          # 只算 train, 不含 val 池
    # 2026-08-02 第四修(实测 10 条死判据里 5 条是误报): 原版把 scripts/ 与
    # 一起扫, 于是 `scripts/ocr_training/ba_vocab.py` 的 **OCR 中文词表**
    # ("总力战"/"大决战"/"好友")、`build_fused_avatar_dataset.py` 的区域名、
    # `live_capture_verify.py` 这种一次性脚本, 全被算成"bot 靠它做决策"。
    # 承重与否只看 brain/ —— 旁支单独报, 不进死判据清单。
    def corpora(subs):
        code = scan_code(subs=subs)
        # 剔除 ui_classes.py 本体(定义行会让每个常量自带 1 次假引用)
        ex_uc = scan_code(exclude_basenames=("ui_classes.py",), subs=subs)
        # 再去掉 import 行(import 本身不是使用)
        bare = "\n".join(ln for ln in ex_uc.splitlines()
                         if not ln.lstrip().startswith(("from ", "import ")))
        return code, ex_uc, bare

    CORP = {"brain": corpora(DECISION_SUBS), "aux": corpora(AUX_SUBS)}
    det = scan_detections() if a.with_det else Counter()

    # ui_classes 常量名  值, 用来判断"是不是真被 skill 逻辑引用"
    sys.path.insert(0, _ROOT)
    from brain.skills import ui_classes as UC
    const = {k: v for k, v in vars(UC).items()
             if isinstance(v, str) and k.isupper()}

    def referenced(name: str, scope: str = "brain") -> int:
        """该 cls 被**代码**引用的次数。scope="brain" 只数决策链, "aux" 数旁支。

        常量类走 `UC.XXX`; 字面量走**带引号的完整匹配** —— 绝不用裸子串
        匹配: `房间区域` 是 `房间区域未解锁` 的子串, 裸匹配会把后者算成前者
        的引用(第一版就这么误报了 11 处)。
        2026-07-25 第三修(workflow 审计): 原版两类引用全盲 —
         `from ui_classes import X` 后**裸用 X**(不带 UC. 前缀);
         **有常量但代码写字面量**(原版 has_const 就不数字面量了)。
        两类漏数都把真死判据归进"废案", --fail-on-dead 假绿(漏报比多报危险)。
        裸名与字面量都在剔除 ui_classes.py 的语料上数(定义行不算引用)。
        """
        code, ex_uc, bare = CORP[scope]
        n = 0
        for k, v in const.items():
            if v == name:
                n += len(re.findall(r"UC\." + k + r"\b", code))
                n += len(re.findall(r"(?<![\w.])" + k + r"\b", bare))
        q = re.escape(name)
        n += len(re.findall(r'"' + q + r'"', ex_uc))
        n += len(re.findall(r"'" + q + r"'", ex_uc))
        return n

    _const_by_val = {}
    for k, v in const.items():
        _const_by_val.setdefault(v, []).append(k)

    def ref_sites(name: str):
        """定位 brain/ 里每个引用点  (相对路径, 行号, 同处并列的其它 cls 常量)。

        并列判定按**逻辑行**做: 判据经常跨行写成
            find_cls(screen, [UC.A,
                              UC.B], conf=...)
        所以从命中行往上/下各粘 2 行再找同伴, 只看 `UC.X` / 裸 `X` 且 X 是
        ui_classes 常量名。找不到同伴 = 独苗 = 真·唯一信号。
        """
        ks = _const_by_val.get(name, [])
        out = []
        for f in glob.glob(os.path.join(_ROOT, "brain", "**", "*.py"),
                           recursive=True):
            if os.path.basename(f) == "ui_classes.py":
                continue
            try:
                lines = _strip_comments(open(f, encoding="utf-8").read()).splitlines()
            except Exception:
                continue
            for ln, line in enumerate(lines, 1):
                if not any(re.search(r"(?<![\w.])(?:UC\.)?" + k + r"\b", line) for k in ks):
                    continue
                ctx = " ".join(lines[max(0, ln - 3):ln + 2])
                partners = {m for m in re.findall(r"(?<![\w.])(?:UC\.)?([A-Z][A-Z0-9_]{2,})\b", ctx)
                            if m in const and m not in ks}
                out.append((os.path.relpath(f, _ROOT), ln, sorted(partners)))
        return out

    val_boxes = count_label_boxes(include_val=True)
    print(f"master {len(master)} 类  |  **train** 标注 {sum(boxes.values())} 框"
          f"  (含 val 池共 {sum(val_boxes.values())})")
    agg = {}
    for i, n in enumerate(master):
        d = agg.setdefault(domain(i), [0, 0])
        d[0] += 1
        if boxes.get(i, 0) > 0:
            d[1] += 1
    for k, v in agg.items():
        print(f"  {k:8} {v[0]:4d} 类, 其中有标注 {v[1]:4d}")

    dead, aux_only = [], []
    for i, n in enumerate(master):
        if boxes.get(i, 0):
            continue
        rb = referenced(n, "brain")
        if rb > 0:
            dead.append((n, i, rb, det.get(n, 0)))
        elif referenced(n, "aux") > 0:
            aux_only.append((i, n, referenced(n, "aux")))
    dead.sort(key=lambda t: -t[2])

    print(f"\n死判据(**brain/ 决策链**在用 + 训练零框) {len(dead)} 条:")
    for n, i, r, dv in dead:
        sites = ref_sites(n)
        # 死判据分两性(memory cls_ownership_audit): 「唯一信号型」= 该 cls 是
        # 某个判断的**独苗**, 死了  判断永不成立  真事故; 「或列表死成员」=
        # 和别的 cls 并列在 [..] / any_of 里, 搭档活着就没事, 只是误导读代码的人。
        # 旧版只按引用次数打「承重!」, 把后者也标成承重 —— 2026-08-02 实测 5 条
        # 全是或列表成员, 一条事故都造不成。
        solo = [s for s in sites if not s[2]]
        mark = "   唯一信号!" if solo else "  (全是「或」列表成员  搭档活着就无害)"
        print(f"   idx{i:4d} {n:24} brain 引用 {r:2d} 处"
              + (f" 实战检出 {dv}" if a.with_det else "") + mark)
        for path, ln, partners in sites:
            tag = ("或[" + ",".join(partners[:3]) + "]") if partners else "独苗"
            print(f"        {path}:{ln}  {tag}")

    if aux_only:
        print(f"\n 仅旁支引用(scripts/tests/server, **不是**死判据) {len(aux_only)} 条:")
        for i, n, r in aux_only:
            print(f"   idx{i:4d} {n:24} 旁支 {r} 处")

    zombie = [(i, n) for i, n in enumerate(master)
              if boxes.get(i, 0) == 0 and referenced(n, "brain") == 0
              and referenced(n, "aux") == 0]
    print(f"\n废案(零标注 + 零引用) {len(zombie)} 条:")
    for i, n in zombie:
        print(f"   idx{i:4d} {n:24} [{domain(i)}]")

    weak = [(n, i, boxes.get(i, 0)) for i, n in enumerate(master)
            if 0 < boxes.get(i, 0) < 30 and referenced(n, "brain") > 0]
    weak.sort(key=lambda t: t[2])
    print(f"\n️低样本(<30 框且代码在用) {len(weak)} 条:")
    for n, i, b in weak:
        print(f"   idx{i:4d} {n:24} 训练框 {b}")

    if a.fail_on_dead and dead:
        print(f"\n {len(dead)} 条死判据 —— 代码依赖模型不认识的 cls")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
