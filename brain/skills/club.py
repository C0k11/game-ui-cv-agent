"""ClubSkill: Claim daily AP from club (社團/社交).

Flow:
1. ENTER: From lobby, click 社交
2. CLAIM: Click claim AP button
3. EXIT: Back to lobby
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_wait, action_back, action_done,
)


class ClubSkill(BaseSkill):
    def __init__(self):
        super().__init__("Club")
        self.max_ticks = 30

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("club timeout")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "club loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "claim":
            return self._claim(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "club unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        # Check if inside club
        club_header = screen.find_any_text(
            ["社團", "社团", "Club"],
            region=(0.0, 0.0, 0.3, 0.08), min_conf=0.6
        )
        if club_header:
            self.log("inside club")
            self.sub_state = "claim"
            return action_wait(300, "entered club")

        if screen.is_lobby():
            nav = self._nav_to(screen, ["社交"])
            if nav:
                return nav

        return action_wait(500, "entering club")

    def _claim(self, screen: ScreenState) -> Dict[str, Any]:
        # Look for AP claim button
        claim = screen.find_any_text(
            ["領取", "领取", "收取", "AP"],
            min_conf=0.6
        )
        if claim:
            self.log("claiming club AP")
            self.sub_state = "exit"
            return action_click_box(claim, "claim club AP")

        # If nothing to claim, exit
        self.sub_state = "exit"
        return action_wait(300, "no club AP to claim")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log("done")
            return action_done("club complete")
        return action_back("club exit: back to lobby")
