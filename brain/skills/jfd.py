"""JointFiringDrillSkill (JFD) — 学院交流会 daily ticket sweep (TicketSweepSkill).

Verified flow: data/_missions_probe_log.md (Step 13-22). Structurally identical
to bounty (same TicketSweepSkill base), with two differences:
- Sweeps cost BOTH tickets AND AP (~15 AP/sweep) → _COSTS_AP=True. The base AP
  gate refuses to sweep when AP is insufficient (never buys AP with pyroxene).
- The 3 academy branch tiles (三一 / 格黑娜 / 千年) have NO YOLO cls (v6 gap,
  task #22), so we select by normalized position in the right panel (probe).

should_run: dot on the 任务大厅入口 hub tile (campaign_nav). JFD itself is a
user-toggled battle skill in skill_order.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.skills.base import ScreenState, action_click
from brain.skills.ticket_sweep import TicketSweepSkill
from brain.skills import ui_classes as UC


# Academy → right-panel position (no cls — probe-measured; documented v6 gap).
_ACADEMY_POS = {
    "三一": (0.92, 0.253),
    "格黑娜": (0.915, 0.401),
    "千年": (0.928, 0.549),
}
_DEFAULT_ORDER = ["千年", "三一", "格黑娜"]


class JointFiringDrillSkill(TicketSweepSkill):
    _TICKET_CLS = UC.TICKET_SCHOOL_EXCHANGE
    _HUB_TILE = UC.HUB_SCHOOL_EXCHANGE
    _PAGE_NAME = ""              # JFD has no PAGE_SIGNATURE → rely on ticket cls
    _CONFIG_KEY = "jfd_academy"
    _COSTS_AP = True
    _AP_PER_SWEEP = 15

    def __init__(self):
        super().__init__("JointFiringDrill")

    def should_run(self, screen: ScreenState) -> bool:
        return self.dot_on_entry(screen, [UC.NAV_TASKS])

    def _click_branch(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Select the academy by position (tiles have no cls — v6 gap). Only
        fires while the JFD ticket cls is on screen (= Academy Select page)."""
        if self.find_cls(screen, self._TICKET_CLS, conf=0.30) is None:
            return None  # not confirmed on the JFD academy-select page yet
        targets = self._branches or _DEFAULT_ORDER
        for name in targets:
            pos = _ACADEMY_POS.get(name)
            if pos is not None:
                self.log(f"select JFD academy '{name}' by position {pos} (no cls, v6 gap)")
                return action_click(pos[0], pos[1], f"select JFD academy {name}")
        return None
