"""ScheduleSkill: Handle daily schedule (課程表).

Flow:
1. ENTER: From lobby, click 課程表
2. EXECUTE: Pick best room, click 開始日程, skip animations
3. LOOP: Repeat until tickets exhausted (日程券不足)
4. EXIT: Back to lobby
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_click_yolo,
    action_wait, action_back, action_done,
)


class ScheduleSkill(BaseSkill):
    def __init__(self):
        super().__init__("Schedule")
        self.max_ticks = 60
        self._tickets_used: int = 0

    def reset(self) -> None:
        super().reset()
        self._tickets_used = 0

    def _is_schedule(self, screen: ScreenState) -> bool:
        """Detect schedule UI: header shows 課程表."""
        return screen.has_text("課程表", region=(0.0, 0.0, 0.3, 0.08), min_conf=0.5)

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("schedule timeout")

        # ── Popup handling ──
        
        # Ticket exhausted popup
        no_ticket = screen.find_any_text(
            ["日程券不足", "日程券已用完", "不足", "購買"],
            min_conf=0.6
        )
        if no_ticket:
            # Confirm or Close
            confirm = screen.find_any_text(["確認", "确认"], region=screen.CENTER, min_conf=0.7)
            if confirm:
                self.log(f"tickets exhausted ({self._tickets_used} used), confirming popup")
                self.sub_state = "exit"
                return action_click_box(confirm, "confirm no tickets")
            
            x_btn = screen.find_yolo_one("叉叉", min_conf=0.3)
            if x_btn:
                self.log(f"tickets exhausted, closing popup via X")
                self.sub_state = "exit"
                return action_click_yolo(x_btn, "close ticket popup")
                
            self.sub_state = "exit"
            return action_click(0.5, 0.8, "dismiss ticket popup")

        # Generic popups
        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "schedule loading")

        # ── State machine ──

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "execute":
            return self._execute(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "schedule unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._is_schedule(screen):
            self.log("inside schedule")
            self.sub_state = "execute"
            return action_wait(500, "entered schedule")

        if screen.is_lobby():
            nav = self._nav_to(screen, ["課程表", "课程表"])
            if nav:
                return nav
            return action_wait(300, "waiting for schedule button")

        return action_wait(500, "entering schedule")

    def _execute(self, screen: ScreenState) -> Dict[str, Any]:
        """Select rooms and start schedule."""
        if not self._is_schedule(screen):
            return action_wait(500, "waiting for schedule UI")

        # 1. If 'Start Schedule' button is visible, click it
        start = screen.find_any_text(
            ["開始日程", "开始日程", "開始"],
            min_conf=0.6,
            region=(0.5, 0.6, 1.0, 1.0) # Usually bottom right
        )
        if start:
            self.log(f"starting schedule #{self._tickets_used + 1}")
            self._tickets_used += 1
            return action_click_box(start, "start schedule")

        # 2. Select a room
        # Try to find 'Rank' or 'LV' text indicating a room
        # Prioritize higher ranks if possible, but for now just pick the first valid one
        room_indicator = screen.find_any_text(["Rank", "LV", "Lv", "RANK"], min_conf=0.5)
        if room_indicator:
            self.log(f"selecting room at ({room_indicator.cx:.2f}, {room_indicator.cy:.2f})")
            return action_click_box(room_indicator, "select room via Rank text")
        
        # Fallback: click predefined room slots (middle of screen)
        # 3-4 rooms usually stacked vertically
        if self.ticks % 3 == 0:
            return action_click(0.6, 0.5, "select middle room (fallback)")
        
        return action_wait(500, "scanning for rooms")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log("back in lobby, done")
            return action_done("schedule complete")
        return action_back("schedule exit: back to lobby")
