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
        current = self.detect_current_screen(screen)

        if current == "Club":
            self.log("inside club")
            self.sub_state = "claim"
            return action_wait(500, "entered club")

        if current == "Lobby":
            # After clicking 社交 in nav bar, a popup overlay appears with
            # 社團 / 好友 / 帮手 buttons at y~0.43-0.49. Click 社團 directly.
            club_btn = screen.find_any_text(
                ["社團", "社团"],
                region=(0.0, 0.30, 0.50, 0.60), min_conf=0.7
            )
            if club_btn:
                self.log(f"clicking '{club_btn.text}' in social overlay")
                return action_click_box(club_btn, f"click '{club_btn.text}'")

            # If overlay not showing yet, open it via nav bar
            nav = self._nav_to(screen, ["社交"])
            if nav:
                return nav
            return action_wait(300, "waiting for social overlay")

        if current and current != "Club":
            self.log(f"wrong screen '{current}', backing out")
            return action_back(f"back from {current}")

        return action_wait(500, "entering club")

    def _claim(self, screen: ScreenState) -> Dict[str, Any]:
        if not self._entered_club_check(screen):
             # Re-check header to ensure we are still in club
             return action_wait(500, "waiting for club UI")

        # Look for AP claim button
        # Usually it says "領取" or shows "AP" icon
        claim = screen.find_any_text(
            ["領取", "领取", "收取"],
            min_conf=0.6,
            region=(0.5, 0.2, 1.0, 0.8) # Right side usually
        )
        if claim:
            self.log("claiming club AP")
            self.sub_state = "exit"
            return action_click_box(claim, "claim club AP")
        
        # Check for "已領取" or similar to confirm done
        claimed = screen.find_any_text(["已領取", "已领取"], min_conf=0.6)
        if claimed:
             self.log("already claimed club AP")
             self.sub_state = "exit"
             return action_wait(200, "already claimed")

        # If nothing to claim, exit
        # Wait a bit to be sure
        if self.ticks % 5 == 0:
            self.sub_state = "exit"
            return action_wait(300, "no club AP found, exiting")
            
        return action_wait(500, "scanning for club AP")

    def _entered_club_check(self, screen: ScreenState) -> bool:
        return (screen.has_text("社團", region=(0.0, 0.0, 0.3, 0.12), min_conf=0.5) or
                screen.has_text("社团", region=(0.0, 0.0, 0.3, 0.12), min_conf=0.5) or
                screen.has_text("Club", region=(0.0, 0.0, 0.3, 0.12), min_conf=0.5))

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log("done")
            return action_done("club complete")
        return action_back("club exit: back to lobby")
