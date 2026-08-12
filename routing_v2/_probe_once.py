# -*- coding: utf-8 -*-
"""验证：连续 tick 停在关卡列表页时，进关 once 保护是否还活着（一次性工具）。

期望：tick0 点入场键（once 落下）；tick1-3 只 wait，**不许再点** ——
否则就是 08-09 审查抓到的「每 tick 清 once → 保护变死码 → 列表滚了点到隔壁关」。
"""
from pathlib import Path

import routing_v2.percept.read as RD
from routing_v2.flow.registry import ALL
from routing_v2.state import vocab as V
from routing_v2.state.machine import Machine
from routing_v2.tests.test_offline import B, Ctx, O, cfg

RD.read_topbar = lambda o, c: 500          # 离线没真帧，AP 打桩

ev = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
ev.state["phase"] = "bonus_clear"
ev.ctx.bag["event_topped"] = {}
ev.ctx.bag["event_farm_plan"] = [{"from_bottom": 0, "why": "fixture"}]

lst = O(B(V.EVENT_SHOP, cx=.1, cy=.5), B(V.EVENT_QUEST_SEL, cx=.6, cy=.15),
        B(V.STAGE_ENTER, cx=.9, cy=.70), B(V.STAR_3, cx=.6, cy=.70),
        B(V.AP, conf=.9, cx=.4, cy=.033))

mac = Machine(1)
rows = []
taps = 0
for i in range(4):
    st = mac.update(lst)
    act = ev.decide(lst, st)
    kind = "TAP" if (act is not None and act.is_tap) else (act.kind if act else "None")
    rows.append("tick%d changed=%-5s -> %-4s %s"
                % (i, st.changed, kind, (act.reason[:30] if act else "")))
    if act is not None and act.is_tap:
        taps += 1
        if act.once_key:                    # 模拟 runner：tap 落地才落 once
            ev.state["once:" + act.once_key] = True

rows.append("")
rows.append("入场键点击次数 = %d （期望 1；>1 说明 once 保护是死码）" % taps)
Path("routing_v2/_diag2.txt").write_text("\n".join(rows), encoding="utf-8")
print("\n".join(rows))
