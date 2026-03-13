"""ApPlanningSkill: collect free AP and optionally buy AP by policy.

Purpose:
- Claim daily free AP from the AP purchase panel.
- Optionally buy extra AP if policy allows it.
- Keep premium-currency safety as default (forbid by default).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill,
    ScreenState,
    OcrBox,
    action_back,
    action_click,
    action_click_box,
    action_done,
    action_wait,
)


class ApPlanningSkill(BaseSkill):
    def __init__(self, *, forbid_premium_currency: bool = True, paid_purchase_limit: int = 0):
        super().__init__("ApPlanning")
        self.max_ticks = 90
        self._forbid_premium_currency = bool(forbid_premium_currency)
        self._paid_purchase_limit = max(0, int(paid_purchase_limit))

        self._enter_attempts: int = 0
        self._dialog_attempts: int = 0
        self._free_collected: bool = False
        self._paid_done: int = 0
        self._pending_free_confirm: bool = False
        self._pending_paid_confirm: bool = False

    def reset(self) -> None:
        super().reset()
        self._enter_attempts = 0
        self._dialog_attempts = 0
        self._free_collected = False
        self._paid_done = 0
        self._pending_free_confirm = False
        self._pending_paid_confirm = False

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            return action_done("ap planning timeout")

        reward_popup = screen.find_any_text(
            ["獲得道具", "获得道具", "獲得獎勵", "获得奖励", "購買完成", "购买完成"],
            min_conf=0.6,
        )
        if reward_popup:
            confirm = screen.find_any_text(["確認", "确认", "確定", "确定", "確", "确", "OK"], min_conf=0.55)
            if confirm:
                return action_click_box(confirm, "dismiss AP reward popup")
            return action_click(0.5, 0.9, "dismiss AP reward popup fallback")

        if self.sub_state != "plan":
            popup = self._handle_common_popups(screen)
            if popup:
                return popup

        if screen.is_loading():
            return action_wait(700, "ap planning loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "open_purchase":
            return self._open_purchase(screen)
        if self.sub_state == "plan":
            return self._plan(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "ap planning unknown state")

    def _is_ap_purchase_dialog(self, screen: ScreenState) -> bool:
        title = screen.find_any_text(
            ["購買AP", "购买AP", "體力恢復", "体力恢复", "AP恢復", "AP恢复"],
            region=(0.12, 0.05, 0.90, 0.30),
            min_conf=0.5,
        )
        controls = screen.find_any_text(
            ["取消", "確認", "确认", "購買", "购买", "每日免費", "每日免费", "青輝石", "青辉石"],
            region=(0.08, 0.18, 0.98, 0.95),
            min_conf=0.5,
        )
        return bool(title and controls)

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._is_ap_purchase_dialog(screen):
            self.sub_state = "plan"
            return action_wait(250, "AP purchase panel ready")

        current = self.detect_current_screen(screen)
        if current == "Lobby":
            self.sub_state = "open_purchase"
            return action_wait(200, "on lobby, opening AP purchase")

        if current and current != "Lobby":
            return action_back(f"ap planning: back from {current}")

        self._enter_attempts += 1
        if self._enter_attempts > 8:
            return action_done("ap planning skipped: lobby not reached")
        return action_wait(400, "ap planning waiting for lobby")

    def _open_purchase(self, screen: ScreenState) -> Dict[str, Any]:
        if self._is_ap_purchase_dialog(screen):
            self.sub_state = "plan"
            return action_wait(250, "entered AP purchase panel")

        if not screen.is_lobby():
            self.sub_state = "enter"
            return action_wait(300, "lost lobby while opening AP panel")

        self._enter_attempts += 1
        if self._enter_attempts > 12:
            self.sub_state = "exit"
            return action_wait(200, "AP purchase panel unavailable, exiting")

        ap_box = screen.find_text_one(r"\d+\s*[/|]\s*\d+", region=screen.TOP_BAR, min_conf=0.4)
        if ap_box:
            x = min(0.96, ap_box.x2 + 0.04)
            y = min(0.12, max(0.02, ap_box.cy))
            return action_click(x, y, "open AP purchase from AP bar")

        plus = screen.find_any_text(["+", "＋", "AP"], region=(0.45, 0.0, 0.90, 0.12), min_conf=0.35)
        if plus:
            return action_click_box(plus, "open AP purchase via plus button")

        return action_click(0.70, 0.05, "open AP purchase fallback")

    def _confirm_dialog_buttons(self, screen: ScreenState) -> Optional[tuple[OcrBox, OcrBox]]:
        cancel = screen.find_any_text(["取消"], region=(0.18, 0.55, 0.60, 0.90), min_conf=0.55)
        confirm = screen.find_any_text(
            ["確認", "确认", "確定", "确定", "購買", "购买", "確", "确"],
            region=(0.42, 0.55, 0.90, 0.90),
            min_conf=0.55,
        )
        if cancel and confirm:
            return cancel, confirm
        return None

    def _plan(self, screen: ScreenState) -> Dict[str, Any]:
        if not self._is_ap_purchase_dialog(screen):
            if screen.is_lobby():
                self.sub_state = "exit"
                return action_wait(250, "AP plan complete on lobby")
            self.sub_state = "open_purchase"
            return action_wait(300, "AP panel closed, reopening")

        self._dialog_attempts += 1

        pair = self._confirm_dialog_buttons(screen)
        if pair:
            cancel_btn, confirm_btn = pair
            should_confirm_free = self._pending_free_confirm and (not self._free_collected)
            if should_confirm_free:
                self._pending_free_confirm = False
                self._free_collected = True
                return action_click_box(confirm_btn, "confirm daily free AP claim")

            should_confirm_paid = (
                self._pending_paid_confirm
                and (not self._forbid_premium_currency)
                and self._paid_done < self._paid_purchase_limit
            )
            if should_confirm_paid:
                self._pending_paid_confirm = False
                self._paid_done += 1
                self.log(f"paid AP purchase confirmed {self._paid_done}/{self._paid_purchase_limit}")
                return action_click_box(confirm_btn, "confirm paid AP purchase")

            self._pending_free_confirm = False
            self._pending_paid_confirm = False
            return action_click_box(cancel_btn, "cancel AP purchase dialog")

        blocked = screen.find_any_text(
            ["不足", "不夠", "不够", "無法購買", "无法购买", "已達上限", "已达上限"],
            region=screen.CENTER,
            min_conf=0.6,
        )
        if blocked:
            ok = screen.find_any_text(["確認", "确认", "確定", "确定", "OK"], region=screen.CENTER, min_conf=0.55)
            if ok:
                self.sub_state = "exit"
                return action_click_box(ok, "ack AP purchase notice")
            self.sub_state = "exit"
            return action_back("close AP purchase notice")

        free_tag = screen.find_any_text(
            ["每日免費", "每日免费", "免費", "免费"],
            region=(0.08, 0.20, 0.80, 0.95),
            min_conf=0.55,
        )
        if free_tag and not self._free_collected:
            used = screen.find_any_text(
                ["已購買", "已购买", "已領取", "已领取"],
                region=(max(0.0, free_tag.x1), max(0.0, free_tag.y1 - 0.08), 1.0, min(1.0, free_tag.y2 + 0.10)),
                min_conf=0.55,
            )
            if used:
                self._free_collected = True
            else:
                btn = screen.find_any_text(
                    ["購買", "购买", "領取", "领取", "確認", "确认"],
                    region=(0.55, max(0.0, free_tag.y1 - 0.10), 1.0, min(1.0, free_tag.y2 + 0.12)),
                    min_conf=0.5,
                )
                self._pending_free_confirm = True
                self._pending_paid_confirm = False
                if btn:
                    return action_click_box(btn, "claim daily free AP")
                return action_click(0.88, free_tag.cy, "claim daily free AP (row tap)")

        if (not self._forbid_premium_currency) and self._paid_done < self._paid_purchase_limit:
            buy_buttons = screen.find_text(r"購買|购买", region=(0.55, 0.30, 1.0, 0.95), min_conf=0.55)
            if buy_buttons:
                buy_buttons = sorted(buy_buttons, key=lambda b: (b.cy, b.cx))
                target = buy_buttons[0]
                if self._free_collected and len(buy_buttons) > 1:
                    target = buy_buttons[1]
                self._pending_paid_confirm = True
                return action_click_box(
                    target,
                    f"attempt paid AP purchase {self._paid_done + 1}/{self._paid_purchase_limit}",
                )

            self._pending_paid_confirm = True
            return action_click(0.88, 0.56, "attempt paid AP purchase fallback")

        self.sub_state = "exit"
        return action_wait(250, "AP planning complete")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            return action_done(
                f"ap planning complete (free={int(self._free_collected)}, paid={self._paid_done})"
            )

        return action_back("ap planning exit to lobby")
