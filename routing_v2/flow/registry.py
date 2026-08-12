# -*- coding: utf-8 -*-
"""Flow 注册表 —— 名字 → 类。前端的模块开关直接对这张表。"""
from __future__ import annotations

from typing import Dict, List, Type

from routing_v2.flow.arena import ArenaFlow
from routing_v2.flow.base import Ctx, Flow
from routing_v2.flow.bounty import BountyFlow
from routing_v2.flow.cafe import CafeFlow
from routing_v2.flow.event import EventFlow
from routing_v2.flow.event_shop import EventShopFlow
from routing_v2.flow.facilities import (ClubFlow, CraftFlow, DailyMissionFlow,
                                        MailFlow, ShopFlow)
from routing_v2.flow.jfd import JfdFlow
from routing_v2.flow.mining import StoryMiningFlow
from routing_v2.flow.momotalk import MomoTalkFlow
from routing_v2.flow.schedule import ScheduleFlow

ALL: Dict[str, Type[Flow]] = {
    "club": ClubFlow,
    "craft": CraftFlow,
    "shop": ShopFlow,
    "cafe": CafeFlow,
    "schedule": ScheduleFlow,
    "bounty": BountyFlow,
    "jfd": JfdFlow,
    "event": EventFlow,
    "event_shop": EventShopFlow,
    "arena": ArenaFlow,
    "mail": MailFlow,
    "daily_mission": DailyMissionFlow,
    "story_mining": StoryMiningFlow,
    "momotalk": MomoTalkFlow,
}

# `daily_routine` 这个模块开关管着三条小链
COMPOSITE = {"daily_routine": ["club", "craft", "shop"]}

# ⛔⛔**开关有、flow 没有** —— 前端能打开，跑起来什么也不发生。
#    `modules` 里有 batch_sweep / special_sweep，`_ap_replan` 的默认优先级
#    列表里也写着它俩，但 `ALL` 里**从来没有过**。用户打开 → build() 静默
#    跳过 → 日志里看不出任何异常，只是"这个 flow 没跑"。
#    这就是 memory `mainline_ap_routing` 那条「全仓没有『无活动→自动切回
#    推图/扫荡』分支」的具体形态。
#    ⇒ 在这里显式登记，`build()` 遇到就**出声**（见下），别再静默。
PLANNED: Dict[str, str] = {
    "batch_sweep": "任务里玩家预设的批量扫荡方案（cls 1-7 待训，见 _pending_slots）",
    "special_sweep": "特殊任务刷经验书 / 信用点（双三倍的主要去处）",
}


def build(cfg: dict, ctx: Ctx, log=None) -> List[Flow]:
    """按 cfg["order"] 和 cfg["modules"] 装配这一轮要跑的 flow 队列。"""
    mods = cfg.get("modules", {})
    out: List[Flow] = []
    seen = set()
    say = log or (lambda _m: None)
    for name, why in PLANNED.items():
        if mods.get(name):
            say(f"[build] ⛔`{name}` 开着，但这条 flow **还没实现** → 这一轮不会跑。"
                f"（{why}）")
    for name in cfg.get("order", []):
        # 开关可能挂在**模块**上而不是 flow 名上（event_shop 归 event 模块，
        # club/craft/shop 归 daily_routine）—— 两边都认，别让"点名要跑某个
        # flow"因为 modules 里没有同名键而被静默跳过（08-08 实测 0 tick）。
        on = mods.get(name)
        if on is None and name in ALL:
            on = mods.get(ALL[name].module, False)
        if not on:
            continue
        for real in COMPOSITE.get(name, [name]):
            cls = ALL.get(real)
            if cls is None:
                if real not in PLANNED:      # PLANNED 的上面已经报过了
                    say(f"[build] ⛔order 里有 `{real}`，但注册表里没有这个 flow")
                continue
            if real in seen:
                continue
            seen.add(real)
            out.append(cls(ctx))
    return out
