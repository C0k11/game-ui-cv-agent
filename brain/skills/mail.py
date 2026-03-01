"""MailSkill: Claim mail and daily tasks.

Flow:
1. ENTER: From lobby, click 邮件箱 or navigate to mail
2. CLAIM_MAIL: Click 一键领取 / Claim All
3. CLAIM_TASKS: Switch to tasks tab, claim all
4. EXIT: Back to lobby
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_wait, action_back, action_done,
)


class MailSkill(BaseSkill):
    def __init__(self):
        super().__init__("Mail")
        self.max_ticks = 30
        self._mail_claimed: bool = False
        self._tasks_claimed: bool = False

    def reset(self) -> None:
        super().reset()
        self._mail_claimed = False
        self._tasks_claimed = False

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("mail timeout")

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
        if self.sub_state == "claim_tasks":
            return self._claim_tasks(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "mail unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        # Check if inside mail (header)
        mail_header = screen.find_any_text(
            ["郵件", "邮件", "郵箱", "邮箱", "Mail"],
            region=(0.0, 0.0, 0.3, 0.08), min_conf=0.6
        )
        if mail_header:
            self.log("inside mail")
            self.sub_state = "claim_mail"
            return action_wait(500, "entered mail")

        if screen.is_lobby():
            # Try mail icon (usually top right or sidebar)
            # Or hidden in menu. Usually on main screen top right.
            # Use '郵件' text if visible.
            mail_btn = screen.find_any_text(
                ["郵件", "邮件", "郵箱", "邮箱"],
                min_conf=0.6
            )
            if mail_btn:
                return action_click_box(mail_btn, "open mail from lobby")
            
            # Fallback: specific click for mail icon (usually near top right)
            # Normalized coordinates for mail icon: approx (0.85, 0.05) or (0.9, 0.05)?
            # Actually, standard lobby layout has mail near top right.
            # Let's try clicking the icon area if text fails.
            return action_click(0.82, 0.04, "click mail icon area")

        return action_wait(500, "entering mail")

    def _claim_mail(self, screen: ScreenState) -> Dict[str, Any]:
        if self._mail_claimed:
            self.sub_state = "claim_tasks"
            return action_wait(200, "mail claimed, moving to tasks")

        # Claim All button usually at bottom right
        claim = screen.find_any_text(
            ["一鍵領取", "一键领取", "全部領取", "全部领取", "Claim All"],
            min_conf=0.6,
            region=(0.5, 0.5, 1.0, 1.0)
        )
        if claim:
            self.log("claiming all mail")
            self._mail_claimed = True
            return action_click_box(claim, "claim all mail")

        # No claim button - maybe empty
        self._mail_claimed = True
        self.sub_state = "claim_tasks"
        return action_wait(300, "no mail to claim")

    def _claim_tasks(self, screen: ScreenState) -> Dict[str, Any]:
        if self._tasks_claimed:
            self.sub_state = "exit"
            return action_wait(200, "tasks claimed, exiting")

        # Look for tasks tab
        tasks_tab = screen.find_any_text(
            ["任務", "任务", "Tasks"],
            min_conf=0.7
        )
        if tasks_tab:
            # Click tasks tab first
            claim = screen.find_any_text(
                ["一鍵領取", "一键领取", "全部領取", "全部领取", "Claim All"],
                min_conf=0.6
            )
            if claim:
                self.log("claiming all tasks")
                self._tasks_claimed = True
                return action_click_box(claim, "claim all tasks")
            return action_click_box(tasks_tab, "switch to tasks tab")

        self._tasks_claimed = True
        self.sub_state = "exit"
        return action_wait(300, "no tasks to claim")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log("done")
            return action_done("mail complete")
        return action_back("mail exit: back to lobby")
