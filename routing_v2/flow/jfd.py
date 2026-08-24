# -*- coding: utf-8 -*-
"""学院交流会（JFD）—— 票券扫荡。

**三个学院现在有 cls 了**：三一 / 千年 / 格黑娜，各 84 train / 5 val
   （ui_v2 实测 2026-08-08）。

老 `brain/skills/jfd.py` 的注释写着「三个學院 tile 没有 YOLO cls（v6 gap，
   task #22）」并用 `_ACADEMY_POS` 三个写死坐标去点。那句话在 v6 时代是对的，
   但模型早就训了 —— **写死坐标一旦落下去，就没人回头看它补上了没**（§A8）。
   这就是为什么新架构里 `tap_at()` 强制要求 `justify`。

AP：账号带大小月卡时 JFD 扫荡**不花 AP**（用户 2026-06-11 纠正；曾因 AP 闸
    误退，白白剩下 15 张票没花）。无月卡时也安全 —— 付不起时游戏会把确认键
    变灰，MAX 也会被钳到付得起的次数。
"""
from __future__ import annotations

from routing_v2.flow.sweep import TicketSweepFlow
from routing_v2.state import vocab as V


class JfdFlow(TicketSweepFlow):
    name = "jfd"
    module = "jfd"
    ticket_cls = V.TICKET_JFD
    hub_tile = V.HUB_JFD
    # 學園交流會 卡在 08-20 `0000322_new_task_hall` 上**没有红/黄点**,
    #    所以不给 hub_dot_region(硬套会点到隔壁卡)。等 v19 训回 tile 真名。
    hub_dot_region = ()
    branch_cls = tuple(V.JFD_ACADEMIES)        # 三一 / 千年 / 格黑娜
    branch_page = "jfd_academy"
    stage_page = "jfd_stage"
    costs_ap = False
    ap_per_sweep = 15
