"""MomoTalkSkill: Auto-complete all unread MomoTalk conversations.

Adapted from reference momo_talk.py:
1. Navigate to MomoTalk from lobby (click MomoTalk notification icon)
2. Sort by unread, descending order
3. Find unread message indicators (red dots / notification badges)
4. Click each unread conversation → auto-complete dialogue
5. Re-check after completing (new messages may appear)
6. Exit when no more unread messages

Key OCR patterns:
- Header: "MomoTalk"
- Sort: "未讀" / "未读" (unread), "最新" (newest)
- Dialogue buttons: "回覆" / "回复" (reply), "下一步" / "Next"
- Story buttons: "確認" / "确认", "SKIP", "跳過"
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box,
    action_wait, action_back, action_done, action_swipe,
)


class MomoTalkSkill(BaseSkill):
    def __init__(self):
        super().__init__("MomoTalk")
        self.max_ticks = 80
        self._conversations_completed: int = 0
        self._scan_ticks: int = 0
        self._dialogue_ticks: int = 0
        self._in_dialogue: bool = False
        self._retries: int = 0

    def reset(self) -> None:
        super().reset()
        self._conversations_completed = 0
        self._scan_ticks = 0
        self._dialogue_ticks = 0
        self._in_dialogue = False
        self._retries = 0

    def _is_momotalk(self, screen: ScreenState) -> bool:
        return (
            screen.has_text("MomoTalk", region=(0.0, 0.0, 0.50, 0.15), min_conf=0.5)
            or screen.has_text("Momo", region=(0.0, 0.0, 0.50, 0.15), min_conf=0.5)
        )

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log(f"timeout ({self._conversations_completed} conversations completed)")
            return action_done("momotalk timeout")

        # Reward popup
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
            return action_wait(800, "momotalk loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "scan":
            return self._scan(screen)
        if self.sub_state == "dialogue":
            return self._dialogue(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "momotalk unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._is_momotalk(screen):
            self.log("inside MomoTalk")
            self.sub_state = "scan"
            self._scan_ticks = 0
            return action_wait(500, "entered MomoTalk")

        current = self.detect_current_screen(screen)
        if current == "Lobby":
            # Click MomoTalk icon (top-left area, usually has notification badge)
            momo = screen.find_any_text(
                ["MomoTalk", "Momo"],
                region=(0.05, 0.10, 0.20, 0.25), min_conf=0.5
            )
            if momo:
                return action_click_box(momo, "click MomoTalk icon")
            # Hardcoded position for MomoTalk icon
            return action_click(0.11, 0.16, "click MomoTalk icon (hardcoded)")

        if current and current != "MomoTalk":
            return action_back(f"back from {current}")

        return action_wait(500, "entering MomoTalk")

    def _scan(self, screen: ScreenState) -> Dict[str, Any]:
        """Scan for unread conversations and click the first one found."""
        self._scan_ticks += 1

        if not self._is_momotalk(screen):
            if self._in_dialogue:
                self.sub_state = "dialogue"
                return action_wait(300, "in dialogue")
            return action_wait(500, "waiting for MomoTalk UI")

        # Look for unread indicators — red notification dots or "NEW" text
        # Unread conversations show a red circle with number on the right side
        # Click the first conversation in the list (sorted by unread)
        # The conversation list is on the left side of the screen
        reply_btn = screen.find_any_text(
            ["回覆", "回复", "Reply"],
            region=(0.20, 0.20, 0.80, 0.90), min_conf=0.5
        )
        if reply_btn:
            self.log("found reply button, clicking")
            self._in_dialogue = True
            self.sub_state = "dialogue"
            return action_click_box(reply_btn, "click reply button")

        # Look for conversation entries with notification badges
        # Red dots appear at y≈0.25-0.85, x≈0.35-0.40
        # Try clicking the first conversation entry
        if self._scan_ticks <= 2:
            # Click first conversation in list
            return action_click(0.20, 0.30, "click first conversation")

        if self._scan_ticks <= 4:
            return action_click(0.20, 0.45, "click second conversation")

        # No unread found after scanning
        if self._retries < 1:
            self._retries += 1
            self._scan_ticks = 0
            # Scroll down to check more conversations
            return action_swipe(0.20, 0.70, 0.20, 0.30, 400, "scroll conversation list")

        self.log(f"no more unread conversations ({self._conversations_completed} completed)")
        self.sub_state = "exit"
        return action_wait(300, "scan complete")

    def _dialogue(self, screen: ScreenState) -> Dict[str, Any]:
        """Auto-complete a MomoTalk dialogue/story."""
        self._dialogue_ticks += 1

        if self._dialogue_ticks > 30:
            self.log("dialogue timeout, going back to scan")
            self._dialogue_ticks = 0
            self._in_dialogue = False
            self.sub_state = "scan"
            self._scan_ticks = 0
            return action_back("dialogue timeout")

        # Reply button — click to advance dialogue
        reply = screen.find_any_text(
            ["回覆", "回复", "Reply"],
            region=(0.20, 0.20, 0.80, 0.90), min_conf=0.5
        )
        if reply:
            return action_click_box(reply, "click reply")

        # Story/cutscene — skip or advance
        skip = screen.find_any_text(
            ["SKIP", "Skip", "跳過", "跳过"],
            min_conf=0.7
        )
        if skip:
            return action_click_box(skip, "skip story")

        # Confirm/next button
        confirm = screen.find_any_text(
            ["確認", "确认", "確定", "确定", "下一步", "Next"],
            region=(0.30, 0.60, 0.70, 0.95), min_conf=0.6
        )
        if confirm:
            return action_click_box(confirm, "confirm/next in dialogue")

        # If back to MomoTalk list, dialogue is complete
        if self._is_momotalk(screen):
            self._conversations_completed += 1
            self._dialogue_ticks = 0
            self._in_dialogue = False
            self.log(f"conversation #{self._conversations_completed} completed")
            self.sub_state = "scan"
            self._scan_ticks = 0
            return action_wait(300, "dialogue complete, scanning for more")

        # Tap center to advance dialogue text
        return action_click(0.50, 0.50, "tap to advance dialogue")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log(f"done ({self._conversations_completed} conversations)")
            return action_done("momotalk complete")
        return action_back("momotalk exit: back to lobby")
