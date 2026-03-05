"""MailSkill: Claim all mail rewards.

Runs AFTER all daily skills (cafe/schedule/bounty/arena/farming) to collect
reward mail (AP, credits, items) that accumulated during the run.

Flow:
1. ENTER: From lobby, click mail icon (YOLO or OCR) or hardcoded position
2. CLAIM_MAIL: Loop clicking 一鍵領取 / Claim All until no more
3. DISMISS: Handle "获得道具" result popups
4. EXIT: Back to lobby
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_click_yolo,
    action_wait, action_back, action_done,
)


class MailSkill(BaseSkill):
    def __init__(self):
        super().__init__("Mail")
        self.max_ticks = 40
        self._claim_attempts: int = 0
        self._claimed_count: int = 0

    def reset(self) -> None:
        super().reset()
        self._claim_attempts = 0
        self._claimed_count = 0

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("mail timeout")

        # Reward result popup — "获得道具" / item list — tap to dismiss
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
            return action_wait(800, "mail loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "claim_mail":
            return self._claim_mail(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "mail unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        current = self.detect_current_screen(screen)

        if current == "Mail":
            self.log("inside mail")
            self.sub_state = "claim_mail"
            return action_wait(500, "entered mail")

        if current == "Lobby":
            # Try YOLO mail icon first
            mail_yolo = screen.find_yolo_one("邮箱", min_conf=0.3)
            if mail_yolo:
                return action_click_yolo(mail_yolo, "open mail via YOLO")

            # Try OCR text
            mail_btn = screen.find_any_text(
                ["郵件", "邮件", "郵箱", "邮箱", "信箱", "Mail"],
                min_conf=0.6
            )
            if mail_btn:
                return action_click_box(mail_btn, "open mail from lobby")

            # Hardcoded fallback: mail icon near top right
            return action_click(0.89, 0.05, "click mail icon area")

        if current and current != "Mail":
            self.log(f"wrong screen '{current}', backing out")
            return action_back(f"back from {current}")

        return action_wait(500, "entering mail")

    def _claim_mail(self, screen: ScreenState) -> Dict[str, Any]:
        self._claim_attempts += 1

        # Safety: if we've tried many times, just exit
        if self._claim_attempts > 12:
            self.log(f"claim loop done ({self._claimed_count} claimed)")
            self.sub_state = "exit"
            return action_wait(300, "claim attempts exhausted")

        # Check if mailbox is empty — "没有郵件" / "沒有郵件" visible
        # The "一次领取" button stays visible (greyed out) even when empty.
        if screen.find_any_text(
            ["没有郵件", "沒有郵件", "没有邮件", "沒有邮件", "No Mail"],
            min_conf=0.6
        ):
            self.log(f"mailbox empty ({self._claimed_count} claimed)")
            self.sub_state = "exit"
            return action_wait(300, "no mail remaining")

        # Look for "一鍵領取" / "全部領取" / "Claim All" / "領取" buttons
        claim = screen.find_any_text(
            ["一鍵領取", "一键领取", "全部領取", "全部领取",
             "一次領取", "一次领取", "Claim All"],
            min_conf=0.6
        )
        if claim:
            self.log(f"claiming all mail (attempt {self._claim_attempts})")
            self._claimed_count += 1
            return action_click_box(claim, "claim all mail")

        # Individual "領取" buttons (when claim-all is gone but individual items remain)
        single_claim = screen.find_any_text(
            ["領取", "领取"],
            min_conf=0.7,
            region=(0.6, 0.1, 1.0, 0.9)
        )
        if single_claim:
            self.log("claiming individual mail item")
            self._claimed_count += 1
            return action_click_box(single_claim, "claim single mail")

        # No more claim buttons — done
        self.log(f"no more mail to claim ({self._claimed_count} claimed)")
        self.sub_state = "exit"
        return action_wait(300, "mail claiming complete")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log(f"done ({self._claimed_count} claimed)")
            return action_done("mail complete")
        return action_back("mail exit: back to lobby")
