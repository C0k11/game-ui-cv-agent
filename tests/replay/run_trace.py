# -*- coding: utf-8 -*-
"""Diff replay 跑批器 — 出一份决策 trace, 供改动前后互比。

用法:
    # 改动前
    py tests/replay/run_trace.py --skill EventQuest --since 20260720 --out before.json
    # 改动后
    py tests/replay/run_trace.py --skill EventQuest --since 20260720 --out after.json
    # 比
    py tests/replay/run_trace.py --diff before.json after.json

diff 只列**决策指纹变了**的 tick(动作类型/坐标/时长), reason 文案变化不算。
每处变化都要人去判断是修好了还是弄坏了 —— 录制的 action 只作旁注, 不是断言
目标(它含 bot 当时犯的错)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from tests.replay.harness import (find_runs, iter_run,  # noqa: E402
                                  replay_skill)

# skill 名(录像里的 skill 字段) → 构造器
_SKILLS = {
    "EventQuest": ("brain.skills.event_quest", "EventQuestSkill"),
    "TicketSweep": ("brain.skills.ticket_sweep", "TicketSweepSkill"),
    "Schedule": ("brain.skills.schedule", "ScheduleSkill"),
    "Cafe": ("brain.skills.cafe", "CafeSkill"),
    "Shop": ("brain.skills.shop", "ShopSkill"),
    "Arena": ("brain.skills.arena", "ArenaSkill"),
    "ArenaShop": ("brain.skills.arena_shop", "ArenaShopSkill"),
    "Mail": ("brain.skills.mail", "MailSkill"),
    "Bounty": ("brain.skills.bounty", "BountySkill"),
    "JointFiringDrill": ("brain.skills.jfd", "JointFiringDrillSkill"),
    "BatchSweep": ("brain.skills.batch_sweep", "BatchSweepSkill"),
    "SpecialSweep": ("brain.skills.special_sweep", "SpecialSweepSkill"),
    "DailyMission": ("brain.skills.daily_mission", "DailyMissionSkill"),
    "Club": ("brain.skills.club", "ClubSkill"),
    "Craft": ("brain.skills.craft", "CraftSkill"),
    "MomoTalk": ("brain.skills.momo_talk", "MomoTalkSkill"),
    "StoryMining": ("brain.skills.story_mining", "StoryMiningSkill"),
    "BuyPyroxene": ("brain.skills.buy_pyroxene", "BuyPyroxeneSkill"),
}


def _factory(skill_name: str):
    mod, cls = _SKILLS[skill_name]
    import importlib
    m = importlib.import_module(mod)
    k = getattr(m, cls)

    def make():
        return k()
    return make


def cmd_trace(a) -> int:
    if a.skill not in _SKILLS:
        print(f"未知 skill {a.skill!r}; 可选: {', '.join(sorted(_SKILLS))}")
        return 2
    runs = find_runs(since=a.since, skill=a.skill, limit=a.limit)
    print(f"命中 {len(runs)} 个 run 含 {a.skill}")
    fac = _factory(a.skill)
    all_rows = []
    for r in runs:
        ticks = list(iter_run(r, skill=a.skill))
        if not ticks:
            continue
        rows = replay_skill(fac, ticks, mode=a.mode, load_frame=a.load_frame)
        all_rows.extend(rows)
        print(f"  {os.path.basename(r):<26} {len(rows):>5} tick")
    errs = [r for r in all_rows if r["error"]]
    print(f"\n共 {len(all_rows)} tick, 异常 {len(errs)}")
    for e in errs[:10]:
        print(f"  ⚠ {e['run']}#{e['tick']} {e['error']}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"skill": a.skill, "mode": a.mode, "rows": all_rows}, f,
                  ensure_ascii=False)
    print(f"→ {a.out}")
    return 0


def cmd_diff(before: str, after: str) -> int:
    b = json.load(open(before, encoding="utf-8"))
    a = json.load(open(after, encoding="utf-8"))
    bi = {(r["run"], r["tick"]): r for r in b["rows"]}
    ai = {(r["run"], r["tick"]): r for r in a["rows"]}
    keys = sorted(set(bi) | set(ai))
    changed = []
    for k in keys:
        rb, ra = bi.get(k), ai.get(k)
        if rb is None or ra is None:
            changed.append((k, rb, ra))
            continue
        if rb["sig"] != ra["sig"] or rb["out_sub_state"] != ra["out_sub_state"]:
            changed.append((k, rb, ra))
    print(f"tick 总数 before={len(bi)} after={len(ai)} | 决策变化 {len(changed)}")
    if not changed:
        print("✅ 零变化 —— 这次改动没有改变任何历史 tick 的决策")
        return 0
    print()
    for (run, tick), rb, ra in changed[:80]:
        print(f"── {run}#{tick}  in_state={((rb or ra) or {}).get('in_sub_state')!r}")
        if rb:
            print(f"   before: {rb['sig']} → {rb['out_sub_state']!r}  {rb['reason']}")
        if ra:
            print(f"   after : {ra['sig']} → {ra['out_sub_state']!r}  {ra['reason']}")
        rec = (ra or rb).get("recorded_reason")
        if rec:
            print(f"   (当时实际: {rec})")
    if len(changed) > 80:
        print(f"\n... 另有 {len(changed) - 80} 处未显示")
    return 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skill")
    p.add_argument("--since", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--mode", default="stateless",
                   choices=["stateless", "sequential"])
    p.add_argument("--load-frame", action="store_true",
                   help="把 tick jpg 读成 BGR 挂到 ScreenState.frame(数字 OCR 类逻辑需要)")
    p.add_argument("--out", default="trace.json")
    p.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"))
    a = p.parse_args()
    if a.diff:
        return cmd_diff(*a.diff)
    if not a.skill:
        p.error("--skill 或 --diff 二选一")
    return cmd_trace(a)


if __name__ == "__main__":
    sys.exit(main())
