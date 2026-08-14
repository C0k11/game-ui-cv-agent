# -*- coding: utf-8 -*-
"""L3 v1 离线半场 — 能力自检报告(cls 检出健康度 + skill 依赖 + 页面可达性 proxy).

L3 全场定义(l1_l3_roadmap): 每天遍历所有页面, 只到达只识别不做动作, 报告
"哪些页面到不了、哪些类检出掉了"。全场依赖 L2 导航接管(未够格), 这是
**不依赖导航的离线半场**: 用 data/trajectories 的实战检出流回答同两个问题
的可回答部分:
   cls 检出健康度 — 基线期常见的类最近静默了(DROPPED/FADING)
   skill 依赖健康 — 每个 skill 代码里引用的 cls, 近窗口零检出的列出来
   页面可达性 proxy — PageGraph 已知页面近窗口没观测到的列出来(identify
     只观测, 不导航)
诚实口径: 离线半场分不开「页面没去」和「检出坏了」— 只报事实+最后见到
日期, 分辨要靠全场(带导航)或人工。

用法: python scripts/l3_health_report.py [--recent-days 3] [--jobs 24]
输出: data/l3/report_YYYYMMDD.md + .json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TRAJ_DIR = REPO / "data" / "trajectories"
OUT_DIR = REPO / "data" / "l3"
_RUN_RE = re.compile(r"^run_(\d{8})_\d{6}$")


def scan_run(run_dir_str: str):
    """单 run 扫描(子进程): 返回 (date, n_ticks, n_err, cls_counter, page_counter,
    skills_seen)。trajectory 的 box 字段是 cls/conf(不是 cls_name/confidence)
    — pg_eval 第一版写错名拿到空结论的坑, 这里映射成 shim 再喂 identify。"""
    sys.path.insert(0, run_dir_str.split("data")[0].rstrip("\\/"))
    from brain.nav.page_graph import identify  # noqa: E402 (子进程各自 import)
    run_dir = Path(run_dir_str)
    m = _RUN_RE.match(run_dir.name)
    day = m.group(1) if m else "unknown"
    cls_counter: Counter = Counter()
    page_counter: Counter = Counter()
    skills = set()
    n_ticks = 0
    n_err = 0
    for f in sorted(run_dir.glob("tick_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            n_err += 1
            continue
        n_ticks += 1
        if d.get("skill"):
            skills.add(d["skill"])
        boxes = d.get("yolo_boxes") or []
        names = set()
        shim_boxes = []
        for b in boxes:
            c = b.get("cls")
            conf = b.get("conf", 0.0)
            if not c:
                continue
            names.add(c)
            shim_boxes.append(SimpleNamespace(cls_name=c, confidence=conf))
        for c in names:                      # 按帧去重计数(一帧算一次)
            cls_counter[c] += 1
        if shim_boxes:
            try:
                page, _ = identify(SimpleNamespace(yolo_boxes=shim_boxes))
                if page:
                    page_counter[page] += 1
            except Exception:
                pass
    return day, n_ticks, n_err, dict(cls_counter), dict(page_counter), sorted(skills)


def load_domains() -> dict:
    """master 表  cls 名  域(ui/avatar/battle)。

    唯一权威 = scripts/cls_domains.py(2026-08-13 修: 旧写法把 476 起全划
    battle, 484-527 的 41 个活 UI 类被错分, 健康报告的域统计全歪)。
    """
    import sys as _sys
    _sys.path.insert(0, str(REPO / "scripts"))
    from cls_domains import domain, load_master
    out = {}
    for i, name in enumerate(load_master()):
        if name:
            out[name] = domain(i)
    return out


# skill 文件名  trajectory `skill` 字段的运行时名(去下划线小写对齐;
# ticket_sweep 是 Bounty/JFD 两个实例的基类, 特判)
_STEM_SPECIAL = {"ticket_sweep": {"Bounty", "JFD"}}


def skill_cls_deps() -> dict:
    """静态提取 skill  引用的 cls 名。不只 UC.CONST — batch_sweep 用的
    SWEEP_BATCH_* 在 brain.screens 里, 第一版漏掉后 `批量扫荡` 报成
    '依赖方:-'(审计工具也要被审计)。"""
    from brain.skills import ui_classes as UC
    const2name = {k: v for k, v in vars(UC).items()
                  if isinstance(v, str) and not k.startswith("_")}
    try:
        from brain import screens as SC
        sc_consts = {k: v for k, v in vars(SC).items()
                     if isinstance(v, str) and k.isupper() and not k.startswith("_")}
    except Exception:
        sc_consts = {}
    deps = {}
    for f in sorted((REPO / "brain" / "skills").glob("*.py")):
        if f.name in ("__init__.py", "base.py", "ui_classes.py"):
            continue
        src = f.read_text(encoding="utf-8", errors="replace")
        refs = set(re.findall(r"UC\.([A-Z][A-Z_0-9]*)", src))
        names = {const2name[r] for r in refs if r in const2name}
        bare = set(re.findall(r"\b([A-Z][A-Z_0-9]{2,})\b", src))
        names |= {sc_consts[r] for r in bare if r in sc_consts}
        if names:
            deps[f.stem] = sorted(names)
    return deps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent-days", type=int, default=3,
                    help="最近 N 个「有 run 的日历日」算 recent 窗口")
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--min-baseline-days", type=int, default=3,
                    help="基线期至少见过这么多天才有资格判 DROPPED")
    args = ap.parse_args()

    runs = sorted(p for p in TRAJ_DIR.iterdir()
                  if p.is_dir() and _RUN_RE.match(p.name))
    if not runs:
        print(" 没扫到任何 run — 分母为 0, 拒绝出报告")
        return 1
    print(f"scanning {len(runs)} runs with {args.jobs} workers ...")

    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for r in ex.map(scan_run, [str(p) for p in runs], chunksize=8):
            results.append(r)

    total_ticks = sum(r[1] for r in results)
    total_err = sum(r[2] for r in results)
    if total_ticks == 0:
        print(" 0 tick 解析成功 — 分母为 0, 拒绝出报告(查字段名/路径)")
        return 1

    all_days = sorted({r[0] for r in results})
    recent_days = set(all_days[-args.recent_days:])
    base_days = set(all_days) - recent_days

    # per-cls: 哪些天见过 / 帧计数; per-page 同
    cls_days: dict = defaultdict(set)
    cls_frames: Counter = Counter()
    cls_last_day: dict = {}
    page_days: dict = defaultdict(set)
    skills_by_day: dict = defaultdict(set)
    for day, _n, _e, cls_c, page_c, skills in results:
        for c, n in cls_c.items():
            cls_days[c].add(day)
            cls_frames[c] += n
            cls_last_day[c] = max(cls_last_day.get(c, ""), day)
        for p in page_c:
            page_days[p].add(day)
        skills_by_day[day].update(skills)

    deps = skill_cls_deps()
    name2skills: dict = defaultdict(list)
    for sk, names in deps.items():
        for n in names:
            name2skills[n].append(sk)
    domains = load_domains()

    # recent 窗口里真跑过的 skill(runtime 名) — 用来分「skill 跑了 cls 却没
    # 出现 = 疑似检出退化」vs「页面压根没去 = 不算病」
    recent_skills = set()
    for d in recent_days:
        recent_skills.update(skills_by_day.get(d, set()))

    def _owner_ran(c: str) -> bool:
        for stem in name2skills.get(c, []):
            runtime = _STEM_SPECIAL.get(stem) or {stem.replace("_", "").lower()}
            for rs in recent_skills:
                if rs.replace("_", "").lower() in runtime or \
                        rs.replace("_", "").lower() == stem.replace("_", "").lower():
                    return True
        return False

    dropped_hot, dropped_cold, dropped_other, fading, new_cls = [], [], [], [], []
    for c, days in sorted(cls_days.items()):
        b_seen = len(days & base_days)
        r_seen = len(days & recent_days)
        dom = domains.get(c, "ui")
        if b_seen >= args.min_baseline_days and r_seen == 0:
            item = (c, b_seen, cls_last_day[c])
            if dom != "ui":
                dropped_other.append(item + (dom,))
            elif _owner_ran(c):
                dropped_hot.append(item)     # A级: 依赖方跑过, cls 却静默
            else:
                dropped_cold.append(item)    # B级: 近窗未访问该页面(低优先)
        elif base_days and b_seen / max(1, len(base_days)) >= 0.5 \
                and r_seen / max(1, len(recent_days)) <= 0.3 * (b_seen / len(base_days)):
            fading.append((c, b_seen, r_seen, cls_last_day[c]))
        elif b_seen == 0 and r_seen > 0:
            new_cls.append((c, r_seen))

    # skill 依赖: 近窗口静默的依赖 cls(整库从没见过的单列 — 那是录不到/死判据)
    dep_silent: dict = {}
    dep_never: dict = {}
    for sk, names in deps.items():
        silent = [(n, cls_last_day.get(n, "")) for n in names
                  if n in cls_days and not (cls_days[n] & recent_days)]
        never = [n for n in names if n not in cls_days]
        if silent:
            dep_silent[sk] = sorted(silent, key=lambda t: t[1])
        if never:
            dep_never[sk] = never

    from brain.nav.page_graph import PAGES
    pages_known = set(PAGES)
    pages_recent = {p for p, ds in page_days.items() if ds & recent_days}
    pages_ever = set(page_days)
    pages_unseen_recent = sorted(pages_ever - pages_recent)
    pages_never = sorted(pages_known - pages_ever)

    today = date.today().strftime("%Y%m%d")
    OUT_DIR.mkdir(exist_ok=True)
    md = OUT_DIR / f"report_{today}.md"
    js = OUT_DIR / f"report_{today}.json"

    lines = [
        f"# L3 v1 离线健康报告 · {today}",
        "",
        f"- 语料: **{len(runs)} runs / {total_ticks} ticks**"
        f" (解析失败 {total_err}), 日历日 {len(all_days)} 天",
        f"- 窗口: recent = {sorted(recent_days)} / baseline = {len(base_days)} 天",
        f"- cls 宇宙(实战出现过): **{len(cls_days)}** 类; "
        f"skill 依赖表: {len(deps)} 个 skill",
        "",
        "离线半场口径: 「近窗口 0 检出」分不开『页面没去』vs『检出坏了』,",
        "本报告只给事实 + 最后见到日期; 分辨靠带导航的全场或人工。",
        "",
        f"## A DROPPED·疑似检出退化 — **依赖方 skill 在 recent 窗口跑过**"
        f"而 cls 零检出 ({len(dropped_hot)})",
    ]
    for c, b, last in sorted(dropped_hot, key=lambda t: -t[1]):
        owners = ",".join(name2skills.get(c, [])) or "-"
        lines.append(f"- `{c}` 基线 {b} 天见过, 最后 {last}; 依赖方: {owners}")
    lines += [
        "",
        f"## B DROPPED·近窗未访问(UI 域, 低优先) ({len(dropped_cold)})",
        "  (skill 没跑到那些页面, 分不出检出好坏 — 全场遍历才能定性)",
    ]
    for c, b, last in sorted(dropped_cold, key=lambda t: -t[1])[:25]:
        owners = ",".join(name2skills.get(c, [])) or "-"
        lines.append(f"- `{c}` 基线 {b} 天, 最后 {last}; 依赖方: {owners}")
    if len(dropped_cold) > 25:
        lines.append(f"- …共 {len(dropped_cold)} 条, 全量见 json")
    _n_av = sum(1 for t in dropped_other if t[3] == "avatar")
    _n_bt = sum(1 for t in dropped_other if t[3] == "battle")
    lines += [
        "",
        f"## C DROPPED·头像/战斗域(随访问面波动, 只计数) — "
        f"avatar {_n_av} / battle {_n_bt}",
    ]
    lines += ["", f"##  FADING — day-rate 掉到基线 30% 以下 ({len(fading)})"]
    for c, b, r, last in fading:
        lines.append(f"- `{c}` 基线 {b} 天  recent {r} 天, 最后 {last}")
    lines += ["", f"##  skill 依赖静默 — 依赖 cls 在 recent 窗口零检出"]
    for sk, items in sorted(dep_silent.items()):
        lines.append(f"- **{sk}**: " + ", ".join(
            f"`{n}`(最后 {d})" for n, d in items))
    lines += ["", f"##  skill 依赖·整库从未见过 — 死判据/待录场景候选"]
    for sk, names in sorted(dep_never.items()):
        lines.append(f"- **{sk}**: " + ", ".join(f"`{n}`" for n in names))
    lines += [
        "",
        f"##  页面可达性 proxy (PageGraph {len(pages_known)} 页)",
        f"- 历史观测到 {len(pages_ever)} 页; recent 窗口观测到 {len(pages_recent)} 页",
        f"- **recent 没到过**(历史到过): {', '.join(pages_unseen_recent) or '无'}",
        f"- **图上有但整库从未观测到**: {', '.join(pages_never) or '无'}",
        "",
        f"##  NEW — 基线没有, recent 才出现 ({len(new_cls)})",
    ]
    for c, r in new_cls:
        lines.append(f"- `{c}` recent {r} 天")
    md.write_text("\n".join(lines), encoding="utf-8")

    js.write_text(json.dumps({
        "date": today, "runs": len(runs), "ticks": total_ticks,
        "parse_errors": total_err, "days": all_days,
        "recent_days": sorted(recent_days),
        "dropped_hot": [{"cls": c, "baseline_days": b, "last": d}
                        for c, b, d in dropped_hot],
        "dropped_cold": [{"cls": c, "baseline_days": b, "last": d}
                         for c, b, d in dropped_cold],
        "dropped_other": [{"cls": c, "baseline_days": b, "last": d, "domain": dm}
                          for c, b, d, dm in dropped_other],
        "fading": [{"cls": c, "baseline_days": b, "recent_days": r, "last": d}
                   for c, b, r, d in fading],
        "dep_silent": {k: [{"cls": n, "last": d} for n, d in v]
                       for k, v in dep_silent.items()},
        "dep_never": dep_never,
        "pages_unseen_recent": pages_unseen_recent,
        "pages_never_seen": pages_never,
        "new_cls": [{"cls": c, "recent_days": r} for c, r in new_cls],
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"OK  {md}")
    print(f"     dropped: hot={len(dropped_hot)} cold={len(dropped_cold)} "
          f"other={len(dropped_other)}; fading={len(fading)} "
          f"dep_silent_skills={len(dep_silent)} dep_never_skills={len(dep_never)} "
          f"pages_unseen_recent={len(pages_unseen_recent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
