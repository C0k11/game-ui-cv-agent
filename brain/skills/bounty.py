"""BountySkill — 悬赏通缉 daily ticket sweep (pure-YOLO, TicketSweepSkill).

Verified flow: data/_missions_probe_log.md (Step 3-12). bounty is the canonical
"票券扫荡型" skill; the full flow lives in TicketSweepSkill. Here we only:
- anchor on 悬赏通缉票 (TICKET_BOUNTY) for the page + ticket digit-OCR,
- enter via the 悬赏通缉 (HUB_BOUNTY) hub tile,
- select the dashboard-configured branch (高架公路 / 沙漠铁道 / 教室) by its cls.

bounty tickets are SHARED across branches (1/6 total), and a MAX sweep drains
them all, so we pick ONE branch (the first enabled) — no iteration. bounty
sweeps cost ONLY tickets (no AP).

should_run: dot on the 任务大厅入口 hub tile (campaign_nav badge).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.skills.base import ScreenState, action_click_box
from brain.skills.ticket_sweep import TicketSweepSkill, _CLS_CONF, _STAGE_PANEL
from brain.skills import ui_classes as UC


# Branch display name (incl. traditional aliases) → its right-panel cls.
_BRANCH_CLS = {
    "高架公路": UC.STAGE_HIGHWAY,
    "沙漠铁道": UC.STAGE_DESERT_RAIL, "沙漠鐵道": UC.STAGE_DESERT_RAIL,
    "教室": UC.STAGE_CLASSROOM,
}
_DEFAULT_ORDER = ["教室", "高架公路", "沙漠铁道"]


class BountySkill(TicketSweepSkill):
    _TICKET_CLS = UC.TICKET_BOUNTY
    _HUB_TILE = UC.HUB_BOUNTY
    _PAGE_NAME = "Bounty"
    _CONFIG_KEY = "bounty_branches"
    # User-corrected 2026-06-11 (2nd revision): bounty NEVER costs AP — with
    # or without monthly pass, tickets only. (My earlier "_COSTS_AP=True"
    # was a wrong inference: the missing 197 AP that day was the USER's own
    # manual 大装备 batch sweep, not bounty. 指控前先问人.)
    _COSTS_AP = False

    def __init__(self):
        super().__init__("Bounty")

    def should_run(self, screen: ScreenState) -> bool:
        # Always enter (user iron rule 2026-06-11): the LOBBY entry dot only
        # means "something in the hall has work" — it must NOT gate this skill.
        # The real signal is the 悬赏通缉 tile's own dot, checked by the hall
        # scan inside _enter (no dot there → graceful "no work" exit).
        return True

    def _click_branch(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Click the first enabled branch tile by its cls (right panel)."""
        targets = self._branches or _DEFAULT_ORDER
        for name in targets:
            cls = _BRANCH_CLS.get(name)
            if cls is None:
                continue
            box = self.find_cls(screen, cls, conf=_CLS_CONF, region=(0.58, 0.12, 1.0, 0.85))
            if box is not None:
                self.log(f"select bounty branch '{name}' (cls {cls})")
                return action_click_box(box, f"select bounty branch {name}")
        return None  # branch cls not seen this frame → base waits / times out
