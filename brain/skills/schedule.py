"""ScheduleSkill: Handle daily schedule (課程表).

Game UI flow (observed from trajectory):
  - Schedule screen = "Location Select" showing locations like
    夏莱辦公室 RANK12, 夏莱居住區 RANK12, etc.
  - Click a location → enters the location detail view (isometric building)
  - Inside location: right ">" arrow to switch to next location
  - Click "全體課程表" button → opens roster overlay (rooms + students)
  - Check roster for target characters via avatar template matching
  - Close roster via X → click ">" to switch location if no target
  - To start: click center building area or "開始日程" button
  - Repeat until tickets exhausted (持有票券 = 0)

YOLO classes used (full.pt):
  - 角色头像 (character portrait) — used for avatar matching
  - 左切換 / 右切換 (switch arrows)
  - 锁 (lock icon)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from brain.skills.base import (
    BaseSkill, ScreenState, OcrBox,
    action_click, action_click_box,
    action_wait, action_back, action_done,
)

# Known location names (Traditional + Simplified Chinese)
_LOCATION_NAMES = [
    "夏莱公室", "夏莱辦公室", "夏莱办公室",
    "夏莱居住區", "夏莱居住区",
    "格黑娜學園", "格黑娜学园",
    "阿拜多斯", "千年研究",
    "百鬼夜行", "紅冬", "红冬",
    "山海經", "山海经", "瓦爾基里", "瓦尔基里",
    "SRT", "聯邦學園", "联邦学园",
]

# Hardcoded UI positions (normalized) observed from screenshots
_RIGHT_ARROW_POS = (0.99, 0.50)   # Right edge ">" arrow for location switch
_ROSTER_X_POS = (0.89, 0.14)      # X close button on roster overlay
_BUILD_CENTER_POS = (0.50, 0.50)  # Center hexagon in location detail
_ROSTER_TAB_POS = (0.84, 0.91)


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


# Minimum match score for avatar template matching.
# Archive used 0.30 but roster thumbnails are small/compressed, causing false
# positives at 0.30-0.48 (e.g. Kisaki matching many unrelated characters).
# Raised to 0.55 to eliminate false positives on roster overlay.
_AVATAR_MATCH_THRESHOLD = 0.55
# Max locations to cycle through before executing at current (full circle).
# Blue Archive has ~10 schedule locations; 12 gives safety margin.
_MAX_LOCATIONS_FULL_CIRCLE = 12


class ScheduleSkill(BaseSkill):
    def __init__(self):
        super().__init__("Schedule")
        self.max_ticks = 200
        self._tickets_used: int = 0
        self._tickets_remaining: int = -1  # -1 = unknown
        self._locations_checked: int = 0
        self._animation_ticks: int = 0
        self._roster_open: bool = False
        self._target_found: bool = False
        self._target_favorites: List[str] = []
        self._wait_ticks: int = 0
        self._visited_locations: Set[str] = set()
        self._starting_location: str = ""
        self._switch_ticks: int = 0
        self._execute_ticks: int = 0
        self._start_clicked: bool = False
        self._roster_scan_ticks: int = 0
        self._avatar_matcher = None
        self._best_match_score: float = -1.0
        self._stale_ticks: int = 0
        self._florence_matcher = None

    def reset(self) -> None:
        super().reset()
        self._tickets_used = 0
        self._tickets_remaining = -1
        self._locations_checked = 0
        self._animation_ticks = 0
        self._roster_open = False
        self._target_found = False
        self._start_clicked = False
        self._target_favorites = _load_target_favorites()
        self._wait_ticks = 0
        self._visited_locations = set()
        self._starting_location = ""
        self._switch_ticks = 0
        self._execute_ticks = 0
        self._best_match_score = -1.0
        self._stale_ticks = 0
        self._roster_scan_ticks = 0
        self._florence_matcher = None
        if self._target_favorites:
            self.log(f"target favorites loaded: {len(self._target_favorites)} characters")
        # Lazy-init avatar matcher
        if self._avatar_matcher is None and self._target_favorites:
            try:
                from vision.avatar_matcher import AvatarMatcher
                avatar_dir = Path(__file__).resolve().parents[2] / "data" / "captures" / "角色头像"
                self._avatar_matcher = AvatarMatcher(str(avatar_dir))
                self.log(f"avatar matcher loaded from {avatar_dir}")
            except Exception as e:
                self.log(f"avatar matcher init failed: {e}")
                self._avatar_matcher = None

    # ── Ticket parsing ──

    def _parse_tickets(self, screen: ScreenState) -> None:
        """Parse remaining ticket count from '持有票券 X/Y' OCR text."""
        for box in screen.ocr_boxes:
            if box.confidence < 0.6:
                continue
            m = re.search(r'(\d+)\s*/\s*(\d+)', box.text)
            if m and '票券' in box.text:
                remaining = int(m.group(1))
                total = int(m.group(2))
                if self._tickets_remaining != remaining:
                    self.log(f"tickets: {remaining}/{total}")
                self._tickets_remaining = remaining
                return

    # ── Avatar matching ──

    def _check_roster_avatars(self, screen: ScreenState) -> bool:
        """Check roster overlay for target characters via YOLO + template matching.

        First pass: AvatarMatcher (fast template + histogram).
        Second pass fallback: Florence pairwise comparison against local favorite
        reference portraits when template matching is inconclusive.
        """
        if not self._target_favorites:
            return False

        avatars = screen.find_yolo("角色头像", min_conf=0.4)
        if not avatars:
            return False

        try:
            import cv2
            import numpy as np
            img = cv2.imdecode(
                np.fromfile(screen.screenshot_path, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if img is None:
                return False
            h, w = img.shape[:2]
        except Exception:
            return False

        best_overall_name = None
        best_overall_score = -1.0
        best_overall_pos = (0.0, 0.0)
        for av in avatars:
            bx1 = max(0, int(av.x1 * w))
            by1 = max(0, int(av.y1 * h))
            bx2 = min(w, int(av.x2 * w))
            by2 = min(h, int(av.y2 * h))
            roi = img[by1:by2, bx1:bx2]
            if roi.size == 0:
                continue
            if self._avatar_matcher is not None:
                matched_name, score = self._avatar_matcher.match_avatar(
                    roi, self._target_favorites
                )
                if matched_name and score > best_overall_score:
                    best_overall_score = score
                    best_overall_name = matched_name
                    best_overall_pos = (av.cx, av.cy)
                if matched_name and score > _AVATAR_MATCH_THRESHOLD:
                    self._target_found = True
                    self.log(
                        f"AVATAR MATCH: '{matched_name}' score={score:.2f} "
                        f"at ({av.cx:.2f},{av.cy:.2f})"
                    )
                    return True
        # Log best non-matching score for debugging
        if best_overall_name:
            self.log(
                f"avatar best={best_overall_name} score={best_overall_score:.2f} "
                f"at ({best_overall_pos[0]:.2f},{best_overall_pos[1]:.2f}) "
                f"(threshold={_AVATAR_MATCH_THRESHOLD})"
            )

        # Florence fallback: use local favorite portraits as pairwise references.
        try:
            if self._florence_matcher is None:
                from vision.florence_vision import get_florence_reference_matcher
                avatar_dir = Path(__file__).resolve().parents[2] / "data" / "captures" / "角色头像"
                self._florence_matcher = get_florence_reference_matcher(str(avatar_dir))
                self.log("Florence reference matcher loaded for roster scan")
        except Exception as e:
            self.log(f"Florence matcher unavailable: {e}")
            self._florence_matcher = None

        if self._florence_matcher is None:
            return False

        florence_candidates = sorted(avatars, key=lambda b: b.confidence, reverse=True)[:6]
        for av in florence_candidates:
            bx1 = max(0, int(av.x1 * w))
            by1 = max(0, int(av.y1 * h))
            bx2 = min(w, int(av.x2 * w))
            by2 = min(h, int(av.y2 * h))
            roi = img[by1:by2, bx1:bx2]
            if roi.size == 0:
                continue
            matched_name, score = self._florence_matcher.match_candidate(roi, self._target_favorites)
            if matched_name and score > 0.5:
                self._target_found = True
                self.log(
                    f"FLORENCE MATCH: '{matched_name}' score={score:.2f} "
                    f"at ({av.cx:.2f},{av.cy:.2f})"
                )
                return True
        return False

    # ── Screen detection helpers ──

    def _is_schedule(self, screen: ScreenState) -> bool:
        """Detect schedule UI: header shows 課程 (traditional) or 课程 (simplified)."""
        return (screen.has_text("課程", region=(0.0, 0.0, 0.3, 0.10), min_conf=0.5) or
                screen.has_text("课程", region=(0.0, 0.0, 0.3, 0.10), min_conf=0.5))

    def _is_location_select(self, screen: ScreenState) -> bool:
        """Detect Location Select screen (list of all locations).

        Must distinguish from location detail (inside one location).
        Location Select has "LocationSelect" text and/or multiple RANK entries.
        Location Detail only has ONE RANK at the top-right.
        """
        if screen.has_text("LocationSelect", min_conf=0.6):
            return True
        # Multiple RANK entries at different y positions → list of locations
        rank_hits = screen.find_text("RANK", min_conf=0.7)
        if len(rank_hits) >= 2:
            ys = sorted(b.cy for b in rank_hits)
            if ys[-1] - ys[0] > 0.10:
                return True
        return False

    def _is_roster_overlay(self, screen: ScreenState) -> bool:
        """Detect full roster overlay (全體課程表 shown as title at top center).

        Wider region and partial matches for MuMu emulator compatibility.
        """
        # Full title match (widened region)
        title = screen.find_any_text(
            ["全體課程表", "全体课程表", "全體課程", "全体课程"],
            region=(0.20, 0.02, 0.80, 0.25),
            min_conf=0.5
        )
        if title:
            return True
        # Fallback: roster overlay shows room rows ("教室" / "房間" text in center)
        # plus the overlay has a dark background with student avatars
        room_label = screen.find_any_text(
            ["教室", "房間", "房间"],
            region=(0.05, 0.15, 0.50, 0.85),
            min_conf=0.6
        )
        if room_label and self._roster_open:
            return True
        return False

    def _is_inside_location(self, screen: ScreenState) -> bool:
        """Detect location detail view (inside one location, isometric building).

        Has "全體課程表" button at bottom-right and schedule header, but is NOT
        the Location Select list (which has multiple RANK entries).
        """
        if not self._is_schedule(screen):
            return False
        if self._is_location_select(screen):
            return False
        btn = screen.find_any_text(
            ["全體課程表", "全体课程表"],
            region=(0.75, 0.85, 1.0, 1.0),
            min_conf=0.5
        )
        if btn:
            return True
        if screen.has_text("RANK", min_conf=0.7):
            return True
        return False

    def _find_schedule_close(self, screen: ScreenState) -> Optional[OcrBox]:
        close = self._find_florence_hit(
            screen,
            ["close button icon", "close dialog x button", "x close icon"],
            region=(0.72, 0.02, 0.94, 0.28),
        )
        if close:
            return close
        return screen.find_text_one(r"^[Xx×]$", region=(0.72, 0.02, 0.94, 0.28), min_conf=0.7)

    def _find_schedule_switch_arrow(self, screen: ScreenState) -> Optional[OcrBox]:
        arrow = self._find_florence_hit(
            screen,
            ["right arrow button", "next location arrow", "switch location arrow"],
            region=(0.86, 0.28, 1.0, 0.72),
        )
        if arrow:
            return arrow
        return screen.find_text_one(">", region=(0.90, 0.35, 1.0, 0.65), min_conf=0.7)

    def _close_roster_action(self, screen: ScreenState, next_state: str, reason: str) -> Dict[str, Any]:
        """Helper to close the roster overlay and transition to next_state.

        Uses YOLO X detection first, then hardcoded position fallback.
        (OCR X detection skipped per project rule: all X buttons via YOLO only.)
        """
        self._roster_open = False
        close_btn = self._find_schedule_close(screen)
        if close_btn:
            self.sub_state = next_state
            return action_click_box(close_btn, f"close roster X ({reason})")
        self.sub_state = next_state
        return action_click(*_ROSTER_X_POS, f"close roster X fallback ({reason})")

    # ── State handlers ──

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
            close_btn = self._find_schedule_close(screen)
            if close_btn:
                self.sub_state = "exit"
                return action_click_box(close_btn, "close ticket popup")
            self.sub_state = "exit"
            return action_click(0.5, 0.8, "dismiss ticket popup")

        # Schedule result / reward popup — tap to dismiss
        result = screen.find_any_text(
            ["日程結果", "日程结果", "獲得獎勵", "获得奖励"],
            region=screen.CENTER, min_conf=0.6
        )
        if result:
            self.log("schedule result popup, tapping to dismiss")
            self._roster_open = False
            return action_click(0.5, 0.9, "dismiss schedule result")

        # Rank-up / bond level up popup (好感度升級 / 羈絆升級)
        # Schedule also raises bond, so these popups can appear here too.
        if screen.find_any_text(["好感度", "Rank Up"], min_conf=0.6):
            self.log("rank-up popup, tapping to dismiss")
            return action_click(0.5, 0.5, "dismiss rankup popup")

        # Bond level up screen (羈絆升級！) — full-screen animation
        bond_stat = screen.find_any_text(
            ["治愈力", "治癒力", "最大體力", "最大体力"],
            min_conf=0.6
        )
        if bond_stat:
            self.log("bond level up screen (stat text), tapping to dismiss")
            return action_click(0.5, 0.5, "dismiss bond level up")

        # Pre-level-up / bond notification screen
        if not self._is_schedule(screen):
            bond_notif = screen.find_any_text(
                ["羈絆升級", "鲜升級", "羈絆點數", "羈絆"],
                min_conf=0.6
            )
            if bond_notif:
                self.log("bond notification screen, tapping to dismiss")
                return action_click(0.5, 0.5, "dismiss bond notification")

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
        """On Location Select screen, click an unvisited location."""
        if not self._is_schedule(screen):
            self._stale_ticks += 1
            if self._stale_ticks > 10:
                self.log("schedule UI lost for too long, trying back")
                self._stale_ticks = 0
                return action_back("back (schedule UI lost)")
            return action_wait(500, "waiting for schedule UI")
        self._stale_ticks = 0

        # If we're actually inside a location (not Location Select), go check it
        if self._is_inside_location(screen):
            self.log("already inside a location, checking roster")
            self.sub_state = "check_roster"
            self._roster_open = False
            self._wait_ticks = 0
            return action_wait(300, "inside location, switching to check_roster")

        # If tickets are known and 0, exit
        if self._tickets_remaining == 0:
            self.log("no tickets remaining, exiting")
            self.sub_state = "exit"
            return action_wait(300, "no tickets")

        # Full-circle safety: if we've cycled through too many locations, execute
        if self._locations_checked >= _MAX_LOCATIONS_FULL_CIRCLE:
            self.log(f"full circle done ({self._locations_checked} locations), entering for execute")
            rank_hits = screen.find_text("RANK", min_conf=0.7)
            if rank_hits:
                rank_hits.sort(key=lambda b: b.cy)
                self.sub_state = "execute"
                self._execute_ticks = 0
                return action_click_box(rank_hits[0], "enter location for fallback execute")
            self.log("no locations found, exiting")
            self.sub_state = "exit"
            return action_wait(300, "no locations found")

        # First entry: always start at 夏莱办公室 (the first location)
        if not self._starting_location:
            for loc_name in ["夏莱公室", "夏莱辦公室", "夏莱办公室"]:
                loc = screen.find_text_one(loc_name, min_conf=0.6)
                if loc:
                    self._starting_location = loc.text
                    self._visited_locations.add(loc.text)
                    self.log(f"starting at '{loc.text}' (will full-circle back)")
                    self.sub_state = "check_roster"
                    self._roster_open = False
                    self._wait_ticks = 0
                    return action_click_box(loc, f"enter location '{loc.text}'")
            # 夏莱 not visible — click first available location
            for loc_name in _LOCATION_NAMES:
                loc = screen.find_text_one(loc_name, min_conf=0.6)
                if loc:
                    self._starting_location = loc.text
                    self._visited_locations.add(loc.text)
                    self.log(f"starting at '{loc.text}' (夏莱 not visible)")
                    self.sub_state = "check_roster"
                    self._roster_open = False
                    self._wait_ticks = 0
                    return action_click_box(loc, f"enter location '{loc.text}'")
            # Fallback: click first RANK entry
            rank_hits = screen.find_text("RANK", min_conf=0.7)
            if rank_hits:
                rank_hits.sort(key=lambda b: b.cy)
                self._starting_location = "RANK"
                self.sub_state = "check_roster"
                self._roster_open = False
                self._wait_ticks = 0
                return action_click_box(rank_hits[0], "enter first location via RANK")

        # Subsequent entries from Location Select (after back from switch_location)
        for loc_name in _LOCATION_NAMES:
            loc = screen.find_text_one(loc_name, min_conf=0.6)
            if loc and loc.text not in self._visited_locations:
                self._visited_locations.add(loc.text)
                self.log(f"clicking location '{loc.text}' (new)")
                self.sub_state = "check_roster"
                self._roster_open = False
                self._wait_ticks = 0
                return action_click_box(loc, f"enter location '{loc.text}'")

        # All known names visited — click any RANK entry
        rank_hits = screen.find_text("RANK", min_conf=0.7)
        if rank_hits:
            rank_hits.sort(key=lambda b: b.cy)
            target = rank_hits[0]
            self.log(f"all locations visited, clicking RANK at ({target.cx:.2f},{target.cy:.2f})")
            self.sub_state = "check_roster"
            self._roster_open = False
            self._wait_ticks = 0
            return action_click_box(target, "enter location via RANK (all visited)")

        return action_wait(500, "looking for locations")

    def _detect_current_location(self, screen: ScreenState) -> str:
        """Detect the current location name from the building detail view header."""
        for loc_name in _LOCATION_NAMES:
            hit = screen.find_text_one(loc_name, region=(0.60, 0.10, 1.0, 0.25), min_conf=0.6)
            if hit:
                return hit.text
        return ""

    def _check_roster(self, screen: ScreenState) -> Dict[str, Any]:
        """Inside a location: open roster, check for targets/locks, decide."""
        if not self._is_schedule(screen):
            self._wait_ticks += 1
            if self._wait_ticks > 5:
                self.sub_state = "select_location"
            return action_wait(500, "waiting for location view")

        self._wait_ticks = 0

        # Detect current location name (visible in building detail header)
        cur_loc = self._detect_current_location(screen)

        # If roster overlay is showing, analyze over multiple ticks before closing.
        # Tick 1: parse tickets & check locks
        # Tick 2: avatar matching (YOLO + template compare)
        # Tick 3: close roster and proceed to execute
        if self._is_roster_overlay(screen):
            self._roster_open = True
            self._roster_scan_ticks += 1

            if self._roster_scan_ticks == 1:
                self._target_found = False
                self._parse_tickets(screen)
                return action_wait(300, "scanning roster (1/3)")

            if self._roster_scan_ticks == 2:
                # Second scan tick: avatar matching
                if self._target_favorites:
                    self._check_roster_avatars(screen)
                return action_wait(300, "scanning roster avatars (2/3)")

            # Third tick: done scanning — execute only if favorite found,
            # otherwise switch to the next location.
            self._roster_scan_ticks = 0
            self._locations_checked += 1
            if cur_loc:
                self._visited_locations.add(cur_loc)

            self._switch_ticks = 0
            if self._target_found:
                self.log(f"favorite found in '{cur_loc}', closing roster to execute")
                return self._close_roster_action(screen, "execute", "favorite found")
            self.log(f"no favorite found in '{cur_loc}' (#{self._locations_checked}), switching location")
            return self._close_roster_action(screen, "switch_location", "no favorite found")

        # Not in roster overlay — open it if we haven't yet
        if not self._roster_open:
            full_tab = screen.find_any_text(
                ["全體課程表", "全体课程表"],
                region=(0.60, 0.80, 1.0, 1.0),
                min_conf=0.5
            )
            if not full_tab:
                full_tab = self._find_florence_hit(
                    screen,
                    ["full schedule roster button", "全體課程表 button", "全体课程表 button"],
                    region=(0.60, 0.78, 1.0, 1.0),
                )
            if full_tab:
                self.log(f"clicking '{full_tab.text}' to view roster")
                self._roster_open = True
                self._roster_scan_ticks = 0
                return action_click_box(full_tab, "open full schedule roster")

            if self._wait_ticks < 2:
                self._wait_ticks += 1
                return action_wait(300, "looking for 全體課程表")

            self.log("full roster button not found, clicking roster fallback area")
            self._roster_open = True
            self._roster_scan_ticks = 0
            self._wait_ticks = 0
            return action_click(*_ROSTER_TAB_POS, "open full schedule roster fallback")

        # Roster was opened but overlay detection failed (YOLO/OCR miss).
        # Give it a few ticks grace period before giving up.
        self._roster_scan_ticks += 1
        if self._roster_scan_ticks <= 3:
            return action_wait(400, f"waiting for roster overlay to appear ({self._roster_scan_ticks}/3)")

        self.log("roster overlay not detected after opening, switching location")
        self._roster_open = False
        self._roster_scan_ticks = 0
        self._locations_checked += 1
        self.sub_state = "switch_location"
        return action_wait(300, "roster detection failed, switching")

    def _switch_location(self, screen: ScreenState) -> Dict[str, Any]:
        """Switch to next location using right arrow or back to Location Select."""
        self._switch_ticks += 1

        # If roster overlay is still showing, close it first
        if self._is_roster_overlay(screen):
            return self._close_roster_action(screen, "switch_location", "before switch")

        # If we somehow landed at Location Select, pick a different location
        if self._is_location_select(screen):
            self.sub_state = "select_location"
            return action_wait(300, "back at location select")

        right = self._find_schedule_switch_arrow(screen)
        if right:
            self.log("clicking right switch")
            self.sub_state = "check_roster"
            self._roster_open = False
            self._wait_ticks = 0
            self._roster_scan_ticks = 0
            return action_click_box(right, "right switch")

        # Hardcoded fallback: click right arrow position (visible as ">" on right edge)
        if self._switch_ticks <= 3:
            self.log("clicking right arrow at hardcoded position")
            self.sub_state = "check_roster"
            self._roster_open = False
            self._wait_ticks = 0
            self._roster_scan_ticks = 0
            return action_click(*_RIGHT_ARROW_POS, "right arrow (hardcoded)")

        # If still stuck after many attempts, go back to Location Select
        if self._switch_ticks > 6:
            self.log("switch stuck, going back to location select")
            self.sub_state = "select_location"
            return action_back("back to location select (switch stuck)")

        return action_wait(400, "looking for switch arrows")

    def _execute(self, screen: ScreenState) -> Dict[str, Any]:
        """Click start button to run schedule.

        Flow: click building center → room info popup opens → click 课程表開始 → animation
        The "课程表開始" (Start Schedule) button appears in the room info popup,
        NOT as a standalone button on the building detail view.
        """
        self._execute_ticks += 1

        if not self._is_schedule(screen):
            self._animation_ticks += 1
            if self._animation_ticks > 8:
                return action_click(0.5, 0.5, "tap to skip animation")
            if self._animation_ticks > 15:
                # Animation done — switch to next location and execute there
                self._animation_ticks = 0
                self._start_clicked = False
                self.sub_state = "switch_location"
                self._target_found = False
                self._roster_open = False
                self._switch_ticks = 0
                return action_wait(500, "animation done, switching to next location")
            return action_wait(500, "waiting for schedule UI")

        self._animation_ticks = 0

        # Parse tickets whenever schedule UI is visible
        self._parse_tickets(screen)
        if self._tickets_remaining == 0:
            self.log("no tickets remaining after execute, exiting")
            self.sub_state = "exit"
            return action_wait(300, "tickets exhausted")

        # If roster is still showing, close it first
        if self._is_roster_overlay(screen):
            return self._close_roster_action(screen, "execute", "before execute")

        # If at Location Select, need to enter a location first
        if self._is_location_select(screen):
            self.sub_state = "select_location"
            return action_wait(300, "back at location select, re-selecting")

        # After clicking start, wait a few ticks for animation to begin
        if getattr(self, '_start_clicked', False):
            if self._execute_ticks > 4:
                # Start didn't trigger — reset and try again
                self.log("start click didn't trigger, retrying")
                self._start_clicked = False
                self._execute_ticks = 0
            return action_wait(500, "waiting for schedule to start")

        # Look for start button in room info popup
        # The button text is "课程表開始" or "課程表開始", sometimes OCR reads partial
        start = screen.find_any_text(
            ["課程表開始", "课程表開始", "课程表开始",
             "開始日程", "开始日程", "START"],
            min_conf=0.5,
        )
        if not start:
            # Partial match "開始"/"开始" in the popup area (bottom-center)
            start = screen.find_any_text(
                ["開始", "开始"],
                region=(0.30, 0.65, 0.70, 0.85),
                min_conf=0.6,
            )
        if start:
            self.log(f"starting schedule #{self._tickets_used + 1}: '{start.text}'")
            self._tickets_used += 1
            self._roster_open = False
            self._start_clicked = True
            self._execute_ticks = 0
            return action_click_box(start, "start schedule")

        # Click a room hexagon to open room info popup.
        # Room hexagons have small reward numbers (e.g. "20", "23", "30") on them.
        # Use OCR to find these numbers in the building area and click them.
        room_nums = [
            box for box in screen.ocr_boxes
            if re.match(r'^\d{1,2}$', box.text.strip())
            and 0.10 < box.cx < 0.70
            and 0.15 < box.cy < 0.70
            and box.confidence > 0.95
        ]
        if room_nums:
            idx = (self._execute_ticks - 1) % len(room_nums)
            target = room_nums[idx]
            self.log(f"clicking room hexagon (reward={target.text}) at ({target.cx:.2f},{target.cy:.2f})")
            return action_click_box(target, f"click room (reward={target.text})")

        # Fallback: try generic building center positions if no numbers found
        _BUILDING_CLICK_POS = [
            (0.35, 0.40, "building left"),
            (0.50, 0.45, "building center"),
            (0.40, 0.55, "building lower-left"),
        ]
        fb_idx = min(self._execute_ticks - 1, len(_BUILDING_CLICK_POS) - 1)
        if self._execute_ticks <= len(_BUILDING_CLICK_POS) + 3:
            x, y, desc = _BUILDING_CLICK_POS[min(fb_idx, len(_BUILDING_CLICK_POS) - 1)]
            self.log(f"no room numbers found, trying {desc}")
            return action_click(x, y, f"open room: {desc}")

        # If still no luck, switch to next location via arrow
        self.log("can't find start button, switching to next location")
        self.sub_state = "switch_location"
        self._execute_ticks = 0
        self._switch_ticks = 0
        self._target_found = False
        return action_wait(300, "can't start, switching location")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log("back in lobby, done")
            return action_done("schedule complete")
        return action_back("schedule exit: back to lobby")
