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
    action_click, action_click_box, action_wait, action_back, action_done,
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
        """Detect schedule UI: header or 全体課程表."""
        return (screen.has_text("課程表", region=(0.0, 0.0, 0.3, 0.08), min_conf=0.6) or
                screen.has_text("课程表", region=(0.0, 0.0, 0.3, 0.08), min_conf=0.6) or
                screen.has_text("全体", min_conf=0.6))

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("schedule timeout")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "schedule loading")

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
            return action_wait(300, "entered schedule")

        if screen.is_lobby():
            nav = self._nav_to(screen, ["課程表", "课程表"])
            if nav:
                return nav

        return action_wait(500, "entering schedule")

    def _execute(self, screen: ScreenState) -> Dict[str, Any]:
        # Check for ticket exhausted popup
        no_ticket = screen.find_any_text(
            ["日程券不足", "日程券已用完", "不足"],
            min_conf=0.6
        )
        if no_ticket:
            self.log(f"tickets exhausted after {self._tickets_used} uses")
            self.sub_state = "exit"
            # Close the popup first
            confirm = screen.find_any_text(["確認", "确认"], region=screen.CENTER, min_conf=0.7)
            if confirm:
                return action_click_box(confirm, "confirm no tickets")
            return action_back("close ticket popup")

        # Look for start button
        start = screen.find_any_text(
            ["開始日程", "开始日程", "開始"],
            min_conf=0.6
        )
        if start:
            self.log(f"starting schedule #{self._tickets_used + 1}")
            self._tickets_used += 1
            return action_click_box(start, "start schedule")

        # Look for room selection - click any available room
        # Rooms show as clickable areas; for now just wait for UI
        if self._is_schedule(screen):
            # Click center to interact with schedule UI
            return action_click(0.5, 0.5, "interact with schedule room")

        if screen.is_lobby():
            self.log("back in lobby, schedule done")
            return action_done("schedule complete")

        return action_wait(500, "schedule executing")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log("back in lobby, done")
            return action_done("schedule complete")
        return action_back("schedule exit: back to lobby")
