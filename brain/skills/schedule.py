"""ScheduleSkill: Handle daily schedule (課程表).

Flow:
1. ENTER: From lobby, click 課程表
2. OPEN_FULL: Click 全體課程表 tab to see all locations
3. SELECT_LOCATION: Pick lowest-level location (prioritise low Lv.)
4. EXECUTE: Click 開始日程, skip animations
5. LOOP: Repeat until tickets exhausted (日程券不足)
6. EXIT: Back to lobby
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from brain.skills.base import (
    BaseSkill, ScreenState, OcrBox,
    action_click, action_click_box, action_click_yolo,
    action_wait, action_back, action_done,
)


class ScheduleSkill(BaseSkill):
    def __init__(self):
        super().__init__("Schedule")
        self.max_ticks = 80
        self._tickets_used: int = 0
        self._full_tab_clicked: bool = False
        self._location_selected: bool = False
        self._animation_ticks: int = 0

    def reset(self) -> None:
        super().reset()
        self._tickets_used = 0
        self._full_tab_clicked = False
        self._location_selected = False
        self._animation_ticks = 0

    def _is_schedule(self, screen: ScreenState) -> bool:
        """Detect schedule UI: header shows 課程表."""
        return screen.has_text("課程", region=(0.0, 0.0, 0.3, 0.10), min_conf=0.5)

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("schedule timeout")

        # ── Popup handling ──

        # Ticket exhausted popup
        # Keep this strict and center-scoped to avoid false positives from lobby text
        # like "購買青輝石" on side banners.
        no_ticket = screen.find_any_text(
            ["日程券不足", "日程券已用完", "沒有日程券", "没有日程券"],
            region=screen.CENTER,
            min_conf=0.6,
        )
        if no_ticket:
            confirm = screen.find_any_text(
                ["確認", "确认", "確", "确"],
                region=screen.CENTER, min_conf=0.7
            )
            if confirm:
                self.log(f"tickets exhausted ({self._tickets_used} used), confirming popup")
                self.sub_state = "exit"
                return action_click_box(confirm, "confirm no tickets")

            x_btn = screen.find_yolo_one("叉叉", min_conf=0.3)
            if x_btn:
                self.log("tickets exhausted, closing popup via X")
                self.sub_state = "exit"
                return action_click_yolo(x_btn, "close ticket popup")

            self.sub_state = "exit"
            return action_click(0.5, 0.8, "dismiss ticket popup")

        # Schedule result / reward popup — tap to dismiss
        result = screen.find_any_text(
            ["日程結果", "日程结果", "獲得獎勵", "获得奖励"],
            region=screen.CENTER, min_conf=0.6
        )
        if result:
            self.log("schedule result popup, tapping to dismiss")
            return action_click(0.5, 0.9, "dismiss schedule result")

        # Generic popups (inventory full, etc.)
        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        # Skip animation — click anywhere during schedule animation
        if screen.find_any_text(["SKIP", "Skip", "跳過", "跳过"], min_conf=0.7):
            skip = screen.find_any_text(["SKIP", "Skip", "跳過", "跳过"], min_conf=0.7)
            if skip:
                return action_click_box(skip, "skip schedule animation")

        if screen.is_loading():
            return action_wait(800, "schedule loading")

        # ── State machine ──

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "open_full":
            return self._open_full(screen)
        if self.sub_state == "select_location":
            return self._select_location(screen)
        if self.sub_state == "execute":
            return self._execute(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "schedule unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        current = self.detect_current_screen(screen)

        if current == "Schedule":
            self.log("inside schedule")
            self.sub_state = "open_full"
            return action_wait(500, "entered schedule")

        if current == "Lobby":
            nav = self._nav_to(screen, ["課程表", "课程表"])
            if nav:
                return nav
            return action_wait(300, "waiting for schedule button")

        if current and current != "Schedule":
            self.log(f"wrong screen '{current}', backing out")
            return action_back(f"back from {current}")

        return action_wait(500, "entering schedule")

    def _open_full(self, screen: ScreenState) -> Dict[str, Any]:
        """Click '全體課程表' tab to see all locations and students."""
        if not self._is_schedule(screen):
            return action_wait(500, "waiting for schedule UI")

        if self._full_tab_clicked:
            self.sub_state = "select_location"
            return action_wait(400, "full tab opened, selecting location")

        # Look for the 全體課程表 / 全体课程表 tab
        full_tab = screen.find_any_text(
            ["全體課程表", "全体课程表", "全體課程", "全体课程"],
            min_conf=0.5
        )
        if full_tab:
            self.log("clicking '全體課程表' tab")
            self._full_tab_clicked = True
            return action_click_box(full_tab, "open full schedule tab")

        # If we see location nodes already (Lv. indicators), skip full-tab step
        lv_hits = screen.find_text(r"Lv\.?\d+", min_conf=0.5)
        if lv_hits:
            self.log("location nodes already visible, skipping full-tab")
            self._full_tab_clicked = True
            self.sub_state = "select_location"
            return action_wait(300, "locations visible")

        # Might already be on the full tab — check for 開始日程
        start = screen.find_any_text(
            ["開始日程", "开始日程"],
            min_conf=0.6,
            region=(0.5, 0.6, 1.0, 1.0)
        )
        if start:
            self._full_tab_clicked = True
            self.sub_state = "execute"
            return action_wait(200, "start button visible, jump to execute")

        return action_wait(500, "looking for full schedule tab")

    def _select_location(self, screen: ScreenState) -> Dict[str, Any]:
        """Select the lowest-level location to prioritize leveling up weak areas."""
        if not self._is_schedule(screen):
            return action_wait(500, "waiting for schedule UI")

        # If start button is already visible, a location is selected
        start = screen.find_any_text(
            ["開始日程", "开始日程"],
            min_conf=0.6,
            region=(0.5, 0.6, 1.0, 1.0)
        )
        if start:
            self.sub_state = "execute"
            return action_wait(200, "location ready, moving to execute")

        # Find all Lv. indicators on screen — these are location nodes
        lv_hits = screen.find_text(r"Lv\.?\d+", min_conf=0.5)
        if lv_hits:
            # Parse level numbers and sort ascending (lowest first)
            def extract_level(box: OcrBox) -> int:
                m = re.search(r"(\d+)", box.text)
                return int(m.group(1)) if m else 999

            sorted_locs = sorted(lv_hits, key=extract_level)
            target = sorted_locs[0]
            lv_num = extract_level(target)
            self.log(f"selecting lowest location: Lv.{lv_num} at ({target.cx:.2f},{target.cy:.2f})")
            self._location_selected = True
            return action_click_box(target, f"select location Lv.{lv_num}")

        # No Lv. visible — try clicking a location node area
        # Schedule locations are usually in the center area
        return action_click(0.5, 0.45, "click center area to find locations")

    def _execute(self, screen: ScreenState) -> Dict[str, Any]:
        """Click start button to run schedule."""
        if not self._is_schedule(screen):
            # Might be in animation or result screen
            self._animation_ticks += 1
            if self._animation_ticks > 8:
                # Tap to skip/dismiss animation
                return action_click(0.5, 0.5, "tap to skip animation")
            return action_wait(500, "waiting for schedule UI")

        self._animation_ticks = 0

        # If 'Start Schedule' button is visible, click it
        start = screen.find_any_text(
            ["開始日程", "开始日程"],
            min_conf=0.6,
            region=(0.5, 0.6, 1.0, 1.0)
        )
        if start:
            self.log(f"starting schedule #{self._tickets_used + 1}")
            self._tickets_used += 1
            # After executing, go back to select next location
            self._location_selected = False
            self.sub_state = "select_location"
            return action_click_box(start, "start schedule")

        # No start button — need to select a location first
        self.sub_state = "select_location"
        return action_wait(300, "no start button, selecting location")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log("back in lobby, done")
            return action_done("schedule complete")
        return action_back("schedule exit: back to lobby")
