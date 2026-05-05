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
    action_click, action_click_box,
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
            mail_btn = screen.find_any_text(
                ["郵件", "邮件", "郵箱", "邮箱", "信箱", "Mail"],
                min_conf=0.6
            )
            if mail_btn:
                return action_click_box(mail_btn, "open mail from lobby")

            # Top-right corner has 3 icons.  Verified pixel-position via
            # brightness-profile analysis on run_20260504_233357 t1
            # (3840×2160 BA window):
            #   contacts (envelope-less): center ~(0.86, 0.04)
            #   MAIL (envelope):           center ~(0.91, 0.04)
            #   grid (apps):               center ~(0.98, 0.04)
            # Old (0.89) hit contacts; my second guess (0.93) hit the
            # divider between mail and grid (empty pixels).  Correct
            # mail icon is at 0.91.
            for cx in (0.91, 0.905, 0.915):
                if screen.has_red_badge(nx=cx, ny=0.04):
                    return action_click(cx, 0.04, f"click mail icon (red-dot @ x={cx})")
            return action_click(0.91, 0.04, "click mail icon area")

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

        # Empty mailbox check (claim button stays visible but greyed when empty)
        if self.is_empty_reward_list(screen):
            self.log(f"mailbox empty ({self._claimed_count} claimed)")
            self.sub_state = "exit"
            return action_wait(300, "no mail remaining")

        # 一鍵領取 / Claim All — base-class helper normalizes variants.
        claim = self.find_claim_all_button(screen)
        if claim:
            self.log(f"claiming all mail (attempt {self._claim_attempts})")
            self._claimed_count += 1
            return action_click_box(claim, "claim all mail")

        # Fallback: single 領取 button on the right side.
        single_claim = self.find_single_claim_button(screen)
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
