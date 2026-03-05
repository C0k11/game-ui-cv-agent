"""CraftSkill: Quick-craft items and claim finished crafts.

Flow:
1. ENTER: From lobby, click 製造 in bottom nav bar
2. CLAIM: If any finished items, click 一次領取 to claim them all
3. QUICK_CRAFT: Click 快速製造 → 開始製造 → confirm popup
4. CLAIM_AGAIN: Claim newly finished items after quick craft
5. EXIT: Back to lobby

The craft screen header reads "製造" / "制造".
Bottom of craft screen has: 快速製造 button, 一次領取 button.
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_click_yolo,
    action_wait, action_back, action_done,
)


class CraftSkill(BaseSkill):
    def __init__(self):
        super().__init__("Craft")
        self.max_ticks = 40
        self._claim_count: int = 0
        self._craft_started: bool = False
        self._craft_ticks: int = 0

    def reset(self) -> None:
        super().reset()
        self._claim_count = 0
        self._craft_started = False
        self._craft_ticks = 0

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("craft timeout")

        # Reward result popup — tap to dismiss
        reward = screen.find_any_text(
            ["獲得道具", "获得道具", "獲得獎勵", "获得奖励"],
            region=screen.CENTER, min_conf=0.6
        )
        if reward:
            return action_click(0.5, 0.9, "dismiss reward popup")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "craft loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "claim":
            return self._claim(screen)
        if self.sub_state == "quick_craft":
            return self._quick_craft(screen)
        if self.sub_state == "claim_after":
            return self._claim_after(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "craft unknown state")

    def _is_craft_screen(self, screen: ScreenState) -> bool:
        """Detect craft screen: header '製造' or '制造'."""
        return self.detect_current_screen(screen) == "Craft"

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        current = self.detect_current_screen(screen)

        if current == "Craft":
            self.log("inside craft")
            self.sub_state = "claim"
            return action_wait(500, "entered craft")

        if current == "Lobby":
            # 製造 is in the bottom nav bar
            nav = self._nav_to(screen, ["製造", "制造"])
            if nav:
                return nav
            return action_wait(300, "waiting for craft button")

        if current and current != "Craft":
            return action_back(f"back from {current}")

        return action_wait(500, "entering craft")

    def _claim(self, screen: ScreenState) -> Dict[str, Any]:
        """Claim all finished craft items via 一次領取."""
        self._craft_ticks += 1

        if self._craft_ticks > 8:
            self.log("claim phase done, moving to quick craft")
            self.sub_state = "quick_craft"
            self._craft_ticks = 0
            return action_wait(300, "claim done")

        # Look for 一次領取 / 一次领取 / 全部領取
        claim = screen.find_any_text(
            ["一次領取", "一次领取", "一鍵領取", "一键领取",
             "全部領取", "全部领取"],
            min_conf=0.6
        )
        if claim:
            self._claim_count += 1
            self.log(f"claiming finished crafts (attempt {self._claim_count})")
            return action_click_box(claim, "claim finished crafts")

        # No claim button — nothing to claim, proceed to quick craft
        self.log("no finished crafts to claim")
        self.sub_state = "quick_craft"
        self._craft_ticks = 0
        return action_wait(300, "no crafts to claim, moving to quick craft")

    def _quick_craft(self, screen: ScreenState) -> Dict[str, Any]:
        """Click 快速製造 → 開始製造 → confirm."""
        self._craft_ticks += 1

        if self._craft_ticks > 12:
            self.log("quick craft phase timeout")
            self.sub_state = "exit"
            return action_wait(300, "quick craft timeout")

        if self._craft_started:
            # After clicking 開始製造, wait for confirm popup
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确"],
                region=screen.CENTER, min_conf=0.7
            )
            if confirm:
                self.log("confirming craft")
                self.sub_state = "claim_after"
                self._craft_ticks = 0
                return action_click_box(confirm, "confirm craft")

            return action_wait(500, "waiting for craft confirm popup")

        # Look for 開始製造 / 开始制造 button (inside quick craft panel)
        start = screen.find_any_text(
            ["開始製造", "开始制造", "開始制造", "开始製造"],
            min_conf=0.6
        )
        if start:
            self.log("clicking 開始製造")
            self._craft_started = True
            return action_click_box(start, "start craft")

        # Look for 快速製造 / 快速制造 button
        quick = screen.find_any_text(
            ["快速製造", "快速制造"],
            min_conf=0.6
        )
        if quick:
            self.log("clicking 快速製造")
            return action_click_box(quick, "click quick craft")

        # No quick craft button — maybe all slots in use or not available
        self.log("no quick craft button found")
        self.sub_state = "exit"
        return action_wait(300, "no quick craft available")

    def _claim_after(self, screen: ScreenState) -> Dict[str, Any]:
        """Claim items after quick craft completes."""
        self._craft_ticks += 1

        if self._craft_ticks > 8:
            self.log("post-craft claim done")
            self.sub_state = "exit"
            return action_wait(300, "post-craft claim done")

        # Look for 一次領取 / claim all
        claim = screen.find_any_text(
            ["一次領取", "一次领取", "一鍵領取", "一键领取",
             "全部領取", "全部领取"],
            min_conf=0.6
        )
        if claim:
            self._claim_count += 1
            self.log(f"claiming post-craft items (attempt {self._claim_count})")
            return action_click_box(claim, "claim post-craft items")

        # No claim button — done
        self.log(f"craft complete ({self._claim_count} claims)")
        self.sub_state = "exit"
        return action_wait(300, "post-craft claim complete")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log(f"done ({self._claim_count} claims)")
            return action_done("craft complete")
        return action_back("craft exit: back to lobby")
