"""ScheduleSkill: Handle daily schedule (課程表).

Game UI flow (observed from trajectory tick_0043):
  - Schedule screen = "Location Select" showing locations like
    夏莱公室 RANK12, 夏莱居住區 RANK12, etc.
  - Click a location → enters the location detail view
  - Inside location: click "全體課程表" to see student roster
  - Check for target characters (from app_config.json) or YOLO '锁' (lock)
  - If targets not found: close 全體課程表 via X → click '左切換' to switch location
  - If targets found or all checked: click "開始日程"
  - Repeat until tickets exhausted (日程券不足)

YOLO classes used:
  16: 左切換 (left switch arrow)
  17: 右切換 (right switch arrow)
  18: 锁 (lock icon — location locked)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from brain.skills.base import (
    BaseSkill, ScreenState, OcrBox,
    action_click, action_click_box, action_click_yolo,
    action_wait, action_back, action_done,
)

# Known location names (Traditional Chinese) for the schedule screen
_LOCATION_NAMES = [
    "夏莱公室", "夏莱辦公室", "夏莱办公室",
    "夏莱居住區", "夏莱居住区",
    "格黑娜學園", "格黑娜学园",
    "阿拜多斯", "千年研究",
    "百鬼夜行", "紅冬", "红冬",
    "山海經", "山海经", "瓦爾基里", "瓦尔基里",
    "SRT", "聯邦學園", "联邦学园",
]


def _load_target_favorites() -> List[str]:
    """Load target character names from app_config.json."""
    try:
        cfg_path = Path(__file__).resolve().parents[2] / "data" / "app_config.json"
        if cfg_path.exists():
            data = json.loads(cfg_path.read_text("utf-8"))
            return data.get("target_favorites", [])
    except Exception:
        pass
    return []


class ScheduleSkill(BaseSkill):
    def __init__(self):
        super().__init__("Schedule")
        self.max_ticks = 120
        self._tickets_used: int = 0
        self._locations_checked: int = 0
        self._max_location_checks: int = 6
        self._animation_ticks: int = 0
        self._inside_location: bool = False
        self._roster_open: bool = False
        self._target_found: bool = False
        self._target_favorites: List[str] = []
        self._wait_ticks: int = 0

    def reset(self) -> None:
        super().reset()
        self._tickets_used = 0
        self._locations_checked = 0
        self._animation_ticks = 0
        self._inside_location = False
        self._roster_open = False
        self._target_found = False
        self._target_favorites = _load_target_favorites()
        self._wait_ticks = 0
        if self._target_favorites:
            self.log(f"target favorites loaded: {len(self._target_favorites)} characters")

    def _is_schedule(self, screen: ScreenState) -> bool:
        """Detect schedule UI: header shows 課程表."""
        return screen.has_text("課程", region=(0.0, 0.0, 0.3, 0.10), min_conf=0.5)

    def _is_location_select(self, screen: ScreenState) -> bool:
        """Detect Location Select screen: has 'LocationSelect' or RANK + location names."""
        if screen.has_text("LocationSelect", min_conf=0.7):
            return True
        if screen.has_text("Location", min_conf=0.7):
            return True
        # Fallback: RANK text + schedule header
        if self._is_schedule(screen) and screen.has_text("RANK", min_conf=0.7):
            return True
        return False

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("schedule timeout")

        # ── Popup handling ──

        # Ticket exhausted popup
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
                self.log(f"tickets exhausted ({self._tickets_used} used)")
                self.sub_state = "exit"
                return action_click_box(confirm, "confirm no tickets")
            x_btn = screen.find_yolo_one("叉叉", min_conf=0.3)
            if x_btn:
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
            # After result, go back to select next location
            self._inside_location = False
            self._roster_open = False
            return action_click(0.5, 0.9, "dismiss schedule result")

        # Generic popups
        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        # Skip animation
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
        if self.sub_state == "select_location":
            return self._select_location(screen)
        if self.sub_state == "check_roster":
            return self._check_roster(screen)
        if self.sub_state == "execute":
            return self._execute(screen)
        if self.sub_state == "switch_location":
            return self._switch_location(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "schedule unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        current = self.detect_current_screen(screen)

        if current == "Schedule":
            self.log("inside schedule")
            self.sub_state = "select_location"
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

    def _select_location(self, screen: ScreenState) -> Dict[str, Any]:
        """On Location Select screen, click the first visible location."""
        if not self._is_schedule(screen):
            return action_wait(500, "waiting for schedule UI")

        # If we already checked all locations, just pick the first one
        if self._locations_checked >= self._max_location_checks:
            self.log(f"checked {self._locations_checked} locations, using current")
            self.sub_state = "execute"
            return action_wait(300, "max locations checked")

        # Find location names on screen and click the first one
        for loc_name in _LOCATION_NAMES:
            loc = screen.find_text_one(loc_name, min_conf=0.6)
            if loc:
                self.log(f"clicking location '{loc.text}'")
                self._inside_location = True
                self.sub_state = "check_roster"
                self._roster_open = False
                self._wait_ticks = 0
                return action_click_box(loc, f"enter location '{loc.text}'")

        # Fallback: find RANK indicators and click the first one
        rank_hits = screen.find_text("RANK", min_conf=0.7)
        if rank_hits:
            # Sort by vertical position, pick first
            rank_hits.sort(key=lambda b: b.cy)
            target = rank_hits[0]
            self.log(f"clicking RANK location at ({target.cx:.2f},{target.cy:.2f})")
            self._inside_location = True
            self.sub_state = "check_roster"
            self._roster_open = False
            self._wait_ticks = 0
            return action_click_box(target, "enter location via RANK")

        return action_wait(500, "looking for locations")

    def _check_roster(self, screen: ScreenState) -> Dict[str, Any]:
        """Inside a location: click 全體課程表, check for targets/locks."""
        if not self._is_schedule(screen):
            self._wait_ticks += 1
            if self._wait_ticks > 5:
                self.sub_state = "select_location"
                self._inside_location = False
            return action_wait(500, "waiting for location view")

        self._wait_ticks = 0

        # If we haven't opened the roster yet, click 全體課程表
        if not self._roster_open:
            full_tab = screen.find_any_text(
                ["全體課程表", "全体课程表", "全體課程", "全体课程", "選課程表"],
                min_conf=0.5
            )
            if full_tab:
                self.log(f"clicking '{full_tab.text}' to view roster")
                self._roster_open = True
                return action_click_box(full_tab, "open full schedule roster")

            # If 開始日程 is visible, we can also check from here
            start = screen.find_any_text(
                ["開始日程", "开始日程"],
                min_conf=0.6
            )
            if start:
                self.log("start button visible, executing")
                self.sub_state = "execute"
                return action_wait(200, "start visible in location")

            return action_wait(400, "looking for 全體課程表")

        # Roster is open — check for locks (YOLO class 18)
        locks = screen.find_yolo("锁", min_conf=0.3)
        if locks:
            self.log(f"found {len(locks)} locked slots in this location")

        # Check for target characters (compare OCR text with favorites)
        # The roster shows student names as OCR text
        if self._target_favorites:
            all_text = " ".join(b.text for b in screen.ocr_boxes if b.confidence >= 0.5)
            for fav in self._target_favorites:
                # Extract character name from filename like "Ako_%28Dress%29.png"
                name = fav.replace(".png", "").replace("%28", "(").replace("%29", ")")
                name = name.replace("_", " ").split("(")[0].strip()
                if len(name) >= 2 and name.lower() in all_text.lower():
                    self.log(f"target character '{name}' found in roster!")
                    self._target_found = True
                    break

        self._locations_checked += 1

        if self._target_found:
            # Close roster and start schedule
            self.log("target found, closing roster to start")
            self._roster_open = False
            # Close the roster popup via X
            x_btn = screen.find_yolo_one("叉叉", min_conf=0.3)
            if x_btn:
                self.sub_state = "execute"
                return action_click_yolo(x_btn, "close roster (target found)")
            self.sub_state = "execute"
            return action_back("close roster (target found, fallback)")

        # Target not found — close roster and switch location
        self.log(f"no target in location #{self._locations_checked}, switching")
        self._roster_open = False
        self._target_found = False
        # Close roster via X button
        x_btn = screen.find_yolo_one("叉叉", min_conf=0.3)
        if x_btn:
            self.sub_state = "switch_location"
            return action_click_yolo(x_btn, "close roster (switching)")
        self.sub_state = "switch_location"
        return action_back("close roster (switching, fallback)")

    def _switch_location(self, screen: ScreenState) -> Dict[str, Any]:
        """Click 左切換 (left arrow) to switch to the previous/next location."""
        # Use YOLO to find left-switch arrow
        left = screen.find_yolo_one("左切换", min_conf=0.3)
        if left:
            self.log("clicking 左切換 to switch location")
            self.sub_state = "check_roster"
            self._roster_open = False
            self._wait_ticks = 0
            return action_click_yolo(left, "left switch location")

        # Fallback: try right switch
        right = screen.find_yolo_one("右切换", min_conf=0.3)
        if right:
            self.log("clicking 右切換 to switch location")
            self.sub_state = "check_roster"
            self._roster_open = False
            self._wait_ticks = 0
            return action_click_yolo(right, "right switch location")

        # If no arrows visible, maybe we're back at Location Select
        if self._is_location_select(screen):
            self.sub_state = "select_location"
            return action_wait(300, "back at location select")

        # Give up switching, just execute current
        if self._locations_checked >= self._max_location_checks:
            self.sub_state = "execute"
            return action_wait(300, "no more switches, executing current")

        return action_wait(400, "looking for switch arrows")

    def _execute(self, screen: ScreenState) -> Dict[str, Any]:
        """Click start button to run schedule."""
        if not self._is_schedule(screen):
            self._animation_ticks += 1
            if self._animation_ticks > 8:
                return action_click(0.5, 0.5, "tap to skip animation")
            return action_wait(500, "waiting for schedule UI")

        self._animation_ticks = 0

        # If 'Start Schedule' button is visible, click it
        start = screen.find_any_text(
            ["開始日程", "开始日程"],
            min_conf=0.6,
        )
        if start:
            self.log(f"starting schedule #{self._tickets_used + 1}")
            self._tickets_used += 1
            self._inside_location = False
            self._roster_open = False
            return action_click_box(start, "start schedule")

        # If at Location Select, need to enter a location first
        if self._is_location_select(screen):
            self.sub_state = "select_location"
            return action_wait(300, "back at location select, re-selecting")

        # No start button visible — wait
        return action_wait(400, "waiting for start button")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log("back in lobby, done")
            return action_done("schedule complete")
        return action_back("schedule exit: back to lobby")
