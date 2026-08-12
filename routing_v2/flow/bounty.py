# -*- coding: utf-8 -*-
"""悬赏通缉 —— 票券扫荡，**不花 AP**（用户 2026-06-11 二次纠正确认）。

票是三个分支**共用**的（1/6 总数），MAX 一次就能排干，所以默认只打配置里的
第一个分支；用户在前端多选就按顺序依次打到票光。
"""
from __future__ import annotations

from routing_v2.flow.sweep import TicketSweepFlow
from routing_v2.state import vocab as V


class BountyFlow(TicketSweepFlow):
    name = "bounty"
    module = "bounty"
    ticket_cls = V.TICKET_BOUNTY
    hub_tile = V.HUB_BOUNTY
    branch_cls = tuple(V.BOUNTY_BRANCHES)      # 高架公路 / 沙漠铁道 / 教室
    branch_page = "bounty_branch"
    stage_page = "bounty_stage"
    costs_ap = False
