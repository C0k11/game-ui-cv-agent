"""EventFarmingSkill: Event-aware AP farming with overflow protection.

Solves two critical edge cases:
1. AP overflow (999/240): Must farm BEFORE cafe, or cafe earnings popup blocks.
2. Event bonuses (夏莱总结算 etc): These events give double drops on normal/special
   missions — must detect "活動進行中" and prioritize those.

Flow:
1. CHECK_AP: Read AP from top bar. If ap_threshold set and AP < threshold → done.
2. ENTER_CAMPAIGN: Navigate to campaign/mission hub (業務區域).
3. SCAN_BONUSES: Detect "活動進行中" badges on 任務/特殊任務.
4. ENTER_TARGET: Click the mission type with event bonus.
   Priority: 特殊任務(信用) > 特殊任務(經驗) > 任務 > Hard fallback.
5. SELECT_CREDIT: In special missions sub-menu, pick 信用 (credit) category.
6. SCROLL_BOTTOM: Scroll down to ensure we're at the last/highest stage.
7. SELECT_STAGE: Click the bottom-most visible stage.
8. SWEEP: Multi-step sweep (掃蕩 → Max → 確認).
9. EXIT: Back to lobby.

Key OCR patterns:
- Event active: "活動進行中" / "活动进行中"
- Event timer: "距離結束還剩" / "距离结束还剩"
- AP display: "999/240" in top bar
- Credit category: "信用" / "貨幣" / "回收"
- Sweep: "掃蕩" / "扫荡" / "最大" / "Max"
- AP exhausted: "AP不足" / "體力不足"
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from brain.skills.base import (
    BaseSkill, ScreenState, OcrBox,
    action_click, action_click_box, action_click_yolo,
    action_wait, action_back, action_done, action_swipe, action_scroll,
)


class EventFarmingSkill(BaseSkill):
    """Event-aware farming skill with optional AP threshold gate."""

    def __init__(self, ap_threshold: int = 0):
        super().__init__("EventFarming")
        self.max_ticks = 120
        self._ap_threshold = ap_threshold  # 0 = always farm, >0 = only if AP >= threshold
        self._current_ap: int = -1
        self._ap_before: int = -1
        self._ap_after: int = -1
        self._has_event: bool = False
        self._event_on_special: bool = False
        self._event_on_normal: bool = False
        self._target: str = ""  # "special", "normal", or "hard_fallback"
        self._in_credit_tab: bool = False
        self._scroll_count: int = 0
        self._sweep_stage: int = 0
        self._sweep_count: int = 0
        self._ap_empty: bool = False
        self._enter_attempts: int = 0
        self._scan_done: bool = False
        self._checked_normal: bool = False  # True after entering 任務 to check for 活動進行中
        self._checked_special: bool = False  # True after entering 特殊任務 to check
        self._checked_hard_tab: bool = False  # True after switching to Hard tab in 任務
        self._hard_stage_list_seen: int = 0

    def reset(self) -> None:
        super().reset()
        self._current_ap = -1
        self._ap_before = -1
        self._ap_after = -1
        self._has_event = False
        self._event_on_special = False
        self._event_on_normal = False
        self._target = ""
        self._in_credit_tab = False
        self._scroll_count = 0
        self._sweep_stage = 0
        self._sweep_count = 0
        self._ap_empty = False
        self._enter_attempts = 0
        self._scan_done = False
        self._checked_normal = False
        self._checked_special = False
        self._checked_hard_tab = False
        self._hard_stage_list_seen = 0

    # ── AP reading ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_ap(screen: ScreenState) -> int:
        """Extract current AP from top-bar OCR text like '999/240' or '120/240'."""
        for box in screen.find_text(r"\d+\s*/\s*\d+", region=screen.TOP_BAR, min_conf=0.4):
            m = re.search(r"(\d+)\s*[/|]\s*(\d+)", box.text)
            if m:
                current = int(m.group(1))
                cap = int(m.group(2))
                # Sanity: cap should be >100 (typical BA AP cap is 200-240+)
                if cap >= 100:
                    return current
        return -1

    # ── Main tick ──────────────────────────────────────────────────────

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("event_farming timeout")

        # AP exhausted popup → done
        no_ap = screen.find_any_text(
            ["AP不足", "體力不足", "体力不足", "購買AP", "购买AP"],
            min_conf=0.6
        )
        if no_ap:
            self.log("AP exhausted, done")
            self._ap_empty = True
            self.sub_state = "exit"
            cancel = screen.find_any_text(["取消"], min_conf=0.7)
            if cancel:
                return action_click_box(cancel, "cancel AP purchase")
            return action_back("dismiss AP popup")

        # Sweep result popup → dismiss
        result = screen.find_any_text(
            ["獲得獎勵", "获得奖励", "戰鬥結果", "战斗结果",
             "獲得道具", "获得道具", "掃蕩完成", "扫荡完成"],
            min_conf=0.6
        )
        if result and self.sub_state == "sweep":
            self._sweep_stage = 3  # ensure stage advances
            self._sweep_count += 1
            self.log(f"sweep #{self._sweep_count} result, dismissing")
            ok = screen.find_any_text(
                ["確認", "确认", "確", "确", "OK"],
                min_conf=0.6
            )
            if ok:
                self.sub_state = "exit"
                return action_click_box(ok, "dismiss sweep result (confirm btn)")
            return action_click(0.5, 0.9, "dismiss sweep result")

        # Skip common popup handler during sweep — sweep has its own dialog FSM
        if self.sub_state != "sweep":
            popup = self._handle_common_popups(screen)
            if popup:
                return popup

        if screen.is_loading():
            return action_wait(800, "event_farming loading")

        if self.sub_state == "":
            self.sub_state = "check_ap"

        dispatch = {
            "check_ap": self._check_ap,
            "enter_campaign": self._enter_campaign,
            "enter_event": self._enter_event,
            "check_normal": self._check_normal,
            "check_special": self._check_special,
            "scan_bonuses": self._scan_bonuses,
            "enter_target": self._enter_target,
            "select_credit": self._select_credit,
            "select_hard_tab": self._select_hard_tab,
            "scroll_bottom": self._scroll_bottom,
            "select_stage": self._select_stage,
            "sweep": self._sweep,
            "exit": self._exit,
        }
        handler = dispatch.get(self.sub_state)
        if handler:
            return handler(screen)
        return action_wait(300, "event_farming unknown state")

    # ── States ─────────────────────────────────────────────────────────

    def _check_ap(self, screen: ScreenState) -> Dict[str, Any]:
        """Gate: skip farming if AP below threshold."""
        if self._ap_threshold <= 0:
            # No threshold — always proceed
            if screen.is_lobby():
                ap = self._parse_ap(screen)
                if ap >= 0:
                    self._ap_before = ap
            self.sub_state = "enter_campaign"
            return action_wait(200, "no AP threshold, proceeding")

        # Need to be on lobby to read AP bar
        if not screen.is_lobby():
            return action_wait(500, "waiting for lobby to read AP")

        ap = self._parse_ap(screen)
        self._current_ap = ap
        if ap >= 0:
            self._ap_before = ap
        if ap < 0:
            # Can't read AP — try once more, then proceed anyway
            self.log("could not read AP, proceeding anyway")
            self.sub_state = "enter_campaign"
            return action_wait(300, "AP unreadable, proceeding")

        self.log(f"current AP = {ap}, threshold = {self._ap_threshold}")
        if ap < self._ap_threshold:
            self.log(f"AP {ap} < {self._ap_threshold}, skipping farming")
            return action_done(f"AP {ap} below threshold {self._ap_threshold}")

        self.sub_state = "enter_campaign"
        return action_wait(200, f"AP {ap} >= {self._ap_threshold}, starting farm")

    def _enter_campaign(self, screen: ScreenState) -> Dict[str, Any]:
        """Navigate to the campaign/mission hub.

        IMPORTANT: The lobby has a rotating banner on the right side that
        alternates between:
        - Previous event reward claim ("距離獎勵領取結束" / "距離獎勵結束") → SKIP
        - Current event entry ("距離結束還剩X天") → CLICK THIS to enter campaign

        The lobby left sidebar 任務 button opens DAILY TASKS, not campaign!
        """
        self._enter_attempts += 1
        current = self.detect_current_screen(screen)

        # If we see the mission hub header (confirmed not Daily Tasks)
        if current == "Mission":
            # Priority: click the top-left event banner ("距離結束還剩X天")
            # to enter the event directly, bypassing scan_bonuses flow.
            # This enters the event's own mission page with its stage list.
            # OCR frequently misreads 還 as 遗/道/違 — include all variants.
            event_banner = screen.find_any_text(
                ["距離結束還剩", "距离结束还剩", "結束還剩", "结束还剩",
                 "距離结束遗剩", "距離结束道剩", "结束遗剩", "結束遗剩"],
                region=(0.0, 0.05, 0.30, 0.25), min_conf=0.3
            )
            if event_banner:
                self.log(f"campaign hub: clicking event banner '{event_banner.text}'")
                self._has_event = True
                self._target = "event_banner"
                self.sub_state = "enter_event"
                # Click the event banner image area (below/on the timer text)
                return action_click(event_banner.cx, event_banner.cy + 0.08,
                                    "click event banner on campaign hub")

            # No event banner → fall back to scan_bonuses flow
            self.log("inside campaign menu (no event banner)")
            self.sub_state = "scan_bonuses"
            return action_wait(500, "entered campaign")

        # Accidentally entered Daily Tasks? Back out.
        if current == "DailyTasks":
            self.log("wrong screen: Daily Tasks, backing out")
            return action_back("back from Daily Tasks")

        # Check for special mission sub-headers (already deep inside)
        special_header = screen.find_any_text(
            ["特殊任務", "特殊任务", "Special"],
            region=(0.0, 0.0, 0.4, 0.15), min_conf=0.6
        )
        if special_header:
            self.log("already inside special missions")
            self._target = "special"
            self.sub_state = "select_credit"
            return action_wait(300, "already in special missions")

        if current == "Lobby":
            # The right-side banner rotates like a PPT slideshow.
            # We ONLY want to click when "距離結束還剩" is visible (current event).
            # If "距離獎勵" is visible, that's the OLD event reward claim — wait.

            # Check for old event reward banner → wait for rotation
            # Also check for "距離獎勵領取結束" which OCR may split into partial matches
            old_reward = screen.find_any_text(
                ["距離獎勵", "距离奖励", "獎勵領取", "奖励领取", "獎勵結束", "奖励结束"],
                region=(0.55, 0.0, 1.0, 0.30), min_conf=0.4
            )
            if old_reward:
                self.log(f"old event reward banner visible ('{old_reward.text}'), waiting for rotation")
                return action_wait(1500, "waiting for event banner to rotate")

            # Check for current event banner → click it
            # Must contain "還剩" or "还剩" (time remaining) to distinguish from reward banners
            current_event = screen.find_any_text(
                ["距離結束還剩", "距离结束还剩",
                 "距離结束遗剩", "距離结束道剩", "距離结束違剩",
                 "結束還剩", "结束还剩", "结束遗剩"],
                region=(0.55, 0.0, 1.0, 0.30), min_conf=0.3
            )
            if not current_event:
                # Looser match: any text containing "剩" + number pattern near event banner
                maybe = screen.find_any_text(
                    ["結束還剩", "结束还剩", "结束遗剩", "結束遗剩"],
                    region=(0.55, 0.0, 1.0, 0.30), min_conf=0.3
                )
                if maybe and "獎" not in maybe.text and "奖" not in maybe.text:
                    current_event = maybe
            if current_event:
                self.log(f"current event banner: '{current_event.text}', clicking")
                # Click the event banner area (below the timer text)
                return action_click(current_event.cx, current_event.cy + 0.10,
                                    "click current event banner")

            # Fallback: click right-side campaign 任務 button
            campaign_btn = screen.find_any_text(
                ["任務", "任务"],
                region=(0.80, 0.70, 1.0, 0.90), min_conf=0.6
            )
            if campaign_btn:
                return action_click_box(campaign_btn, "click campaign entry (right side)")

            # Last resort: click the right-side mission area directly
            return action_click(0.95, 0.83, "click campaign area (hardcoded)")

        # Event page detected (活動 header) — proceed to enter_event
        if current == "Event":
            self.log("on event page")
            self._has_event = True
            self.sub_state = "enter_event"
            return action_wait(300, "detected event page")

        if current and current not in ("Mission", "Lobby", "DailyTasks", "Event"):
            return action_back(f"back from {current}")

        if self._enter_attempts > 15:
            self.log("can't reach campaign menu, falling back to exit")
            self.sub_state = "exit"
            return action_wait(300, "give up entering campaign")

        return action_wait(500, "entering campaign")

    def _enter_event(self, screen: ScreenState) -> Dict[str, Any]:
        """Handle the event page after clicking event banner on campaign hub.

        夏莱-type events show an 活動 page with:
        - Header: "活動"
        - Right panel: "道具獲得方法" with:
            - 任務 → 入場 button
            - 特殊任務 → 入場 button
        - Bottom: event info (道具持有量, 進行狀況)

        Flow:
        1. Detect 活動 page (header "活動" + 道具獲得方法)
        2. Find 入場 buttons paired with 任務/特殊任務 (same y-row)
        3. Click 特殊任務's 入場 first (priority: credits > EXP)
        4. If not found, try 任務's 入場
        5. After entering, next state handles the landing page

        Non-夏莱 events have different layouts (story + quest).
        Those are handled separately (TODO: future).
        """
        current = self.detect_current_screen(screen)

        # Still on campaign hub → banner click may not have registered
        if current == "Mission":
            event_banner = screen.find_any_text(
                ["距離結束還剩", "距离结束还剩", "結束還剩", "结束还剩",
                 "距離结束遗剩", "距離结束道剩", "结束遗剩", "結束遗剩"],
                region=(0.0, 0.05, 0.30, 0.25), min_conf=0.3
            )
            if event_banner:
                return action_click(event_banner.cx, event_banner.cy + 0.08,
                                    "retry click event banner")
            # No banner visible — fall back to scan_bonuses on campaign hub
            self.sub_state = "scan_bonuses"
            return action_wait(500, "event banner gone, falling back to scan")

        # ── Detect 活動 page (夏莱-type event) ──
        # Header "活動" is at top-left, or "道具獲得方法" is visible
        event_header = screen.find_any_text(
            ["活動", "活动"],
            region=(0.0, 0.0, 0.20, 0.08), min_conf=0.7
        )
        item_method = screen.find_any_text(
            ["道具獲得方法", "道具获得方法", "道具獲得"],
            min_conf=0.5
        )

        # ── Check for EXPIRED event ──
        # If "活動期間已結束" or "已結束" visible, this event is over — back out
        expired = screen.find_any_text(
            ["已結束", "已结束", "活動期間已結束", "活动期间已结束"],
            min_conf=0.5
        )
        if expired:
            self._expired_count = getattr(self, '_expired_count', 0) + 1
            if self._expired_count >= 3:
                self.log(f"event EXPIRED ({self._expired_count}x), marking done")
                return action_done("event expired, skipping")
            self.log(f"event EXPIRED: '{expired.text}', backing out ({self._expired_count})")
            return action_back("back from expired event")

        # ── Non-夏莱 event type: Story/Quest/Challenge tabs ──
        # These events show tabs like "Story", "Quest", "Challenge" at top-right
        # and have 劇情/商店/任務/抽卡 buttons at bottom.
        # IMPORTANT: click the "Quest" TAB (top, combat quests with Normal/Hard),
        # NOT the bottom nav "任務" (daily event missions like card draws).
        quest_tab = screen.find_any_text(
            ["Quest"],
            region=(0.60, 0.08, 0.85, 0.22), min_conf=0.6
        )
        story_tab = screen.find_any_text(
            ["Story", "Challenge"],
            region=(0.50, 0.08, 1.0, 0.22), min_conf=0.6
        )
        if quest_tab and (story_tab or event_header):
            self.log("non-夏莱 event type, clicking Quest tab (top)")
            self.sub_state = "check_normal"
            self._checked_normal = True
            return action_click_box(quest_tab, "click Quest tab on event page")

        if event_header or item_method:
            self.log("on 活動 page (夏莱 event type)")

            # Find all 入場 buttons on the page
            enter_buttons: List[OcrBox] = []
            enter_buttons += screen.find_text("入場", min_conf=0.5)
            enter_buttons += screen.find_text("入场", min_conf=0.5)

            if not enter_buttons:
                return action_wait(500, "活動 page but no 入場 buttons found")

            # Helper: find 入場 button paired with a label (same y-row)
            def _find_paired_btn(label: Optional[OcrBox]) -> Optional[OcrBox]:
                if not label:
                    return None
                paired = [b for b in enter_buttons
                          if abs(b.cy - label.cy) < 0.08
                          and b.cx > label.cx]
                return paired[0] if paired else None

            special_label = screen.find_any_text(
                ["特殊任務", "特殊任务"], min_conf=0.6
            )
            normal_label: Optional[OcrBox] = None
            for nl in screen.find_text("任務", min_conf=0.6):
                if "特殊" not in nl.text and "懸賞" not in nl.text:
                    normal_label = nl
                    break

            # Flow: enter 任務 first to check for 活動進行中,
            # then 特殊任務. This ensures we farm the one with the badge.
            if not self._checked_normal and normal_label:
                btn = _find_paired_btn(normal_label)
                if btn:
                    self.log(f"entering 任務 to check for 活動進行中")
                    self._checked_normal = True
                    self.sub_state = "check_normal"
                    return action_click_box(btn, "click 任務 入場 on 活動 page")

            if not self._checked_special and special_label:
                btn = _find_paired_btn(special_label)
                if btn:
                    self.log(f"entering 特殊任務 to check for 活動進行中")
                    self._checked_special = True
                    self.sub_state = "check_special"
                    return action_click_box(btn, "click 特殊任務 入場 on 活動 page")

            # Both checked but neither had 活動進行中 — enter 特殊任務 anyway
            if special_label:
                btn = _find_paired_btn(special_label)
                if btn:
                    self.log("fallback: entering 特殊任務 (no 活動進行中 found)")
                    self._target = "special"
                    self.sub_state = "select_credit"
                    return action_click_box(btn, "click 特殊任務 入場 fallback")

            # Last resort: click first 入場
            self.log("clicking first available 入場 on 活動 page")
            self._target = "special"
            self.sub_state = "select_credit"
            return action_click_box(enter_buttons[0], "click first 入場 on 活動 page")

        # ── If we landed inside special missions already ──
        special_header = screen.find_any_text(
            ["特殊任務", "特殊任务"],
            region=(0.0, 0.0, 0.4, 0.15), min_conf=0.6
        )
        if special_header:
            self.log("event banner → landed in special missions")
            self._target = "special"
            self.sub_state = "select_credit"
            return action_wait(300, "entered special missions via event")

        # If 關卡目錄 or stage list visible
        stage_list = screen.find_any_text(
            ["關卡目錄", "关卡目录"],
            min_conf=0.5
        )
        if stage_list:
            self.log("event banner → landed in stage list")
            self.sub_state = "scroll_bottom"
            return action_wait(300, "entered event stage list")

        # Request Select page (special missions sub-menu)
        request_select = screen.find_any_text(
            ["Request Select", "選擇委託", "选择委托"],
            min_conf=0.5
        )
        if request_select:
            self.log("event banner → landed in request select")
            self._target = "special"
            self.sub_state = "select_credit"
            return action_wait(300, "entered request select via event")

        return action_wait(800, "waiting for event page to load")

    def _check_normal(self, screen: ScreenState) -> Dict[str, Any]:
        """Check if 任務 (normal missions) has 活動進行中 badge.

        After clicking Quest tab or 任務's 入場 on the 活動 page, we land
        on the quest page (Normal/Hard tabs visible).

        Flow:
        1. Default lands on Normal tab → check for 活動進行中
        2. If not found → click Hard tab → check again
        3. If still not found → back to 活動 page
        """
        self._check_normal_ticks = getattr(self, '_check_normal_ticks', 0) + 1

        # Detect wrong page: "活動任務" header = event daily missions (card draws)
        # This means we clicked the bottom nav "任務" instead of Quest tab.
        wrong_page = screen.find_any_text(
            ["活動任務", "活动任务"],
            region=(0.0, 0.0, 0.25, 0.08), min_conf=0.7
        )
        if wrong_page:
            self.log("on wrong page (活動任務 daily missions), backing out")
            self._check_normal_ticks = 0
            self.sub_state = "enter_event"
            return action_back("back from 活動任務 (wrong page)")

        # Verify we're on the quest page: Normal/Hard tabs visible, or
        # header "任務" at top-left. Accept tabs alone (Quest tab view).
        mission_header = screen.find_any_text(
            ["任務", "任务"],
            region=(0.0, 0.0, 0.20, 0.08), min_conf=0.7
        )
        normal_tab = screen.find_any_text(["Normal"], min_conf=0.6)
        hard_tab = screen.find_any_text(["Hard"], min_conf=0.6)
        on_mission_page = (normal_tab or hard_tab) and (mission_header or normal_tab or hard_tab)

        if not on_mission_page:
            if self._check_normal_ticks > 15:
                self.log("stuck waiting for quest page, backing out")
                self._check_normal_ticks = 0
                self.sub_state = "enter_event"
                return action_back("timeout waiting for quest page")
            return action_wait(500, "waiting for quest page to load")

        # We're confirmed on the 任務 page. Check for 活動進行中.
        # The badge appears in the left panel area (~x < 0.5, y 0.1-0.5).
        badge = screen.find_any_text(
            ["活動進行中", "活动进行中"],
            region=(0.0, 0.05, 0.5, 0.5), min_conf=0.5
        )
        if badge:
            self.log(f"任務 has 活動進行中! Farming here.")
            self._event_on_normal = True
            self._target = "normal"
            self.sub_state = "scroll_bottom"
            return action_wait(300, "任務 has event bonus, farming")

        # No badge on current tab. If we haven't tried Hard yet → click Hard tab.
        if not self._checked_hard_tab and hard_tab:
            self.log("no 活動進行中 on Normal, switching to Hard tab")
            self._checked_hard_tab = True
            return action_click_box(hard_tab, "click Hard tab to check for event")

        # Checked both Normal and Hard — no 活動進行中. Back to 活動 page.
        self.log("任務 has NO 活動進行中 on Normal or Hard, backing out")
        self.sub_state = "enter_event"
        return action_back("back from 任務 (no event bonus)")

    def _check_special(self, screen: ScreenState) -> Dict[str, Any]:
        """Check if 特殊任務 has 活動進行中 badge.

        After clicking 特殊任務's 入場 on the 活動 page, we land on the
        special missions page (header: 特殊任務, with Request Select).
        Check for 活動進行中. If found → farm here. If not → back to 活動 page.
        """
        # First: verify we're on the 特殊任務 page.
        # Must see header "特殊任務" at top-left OR "Request Select".
        special_header = screen.find_any_text(
            ["特殊任務", "特殊任务"],
            region=(0.0, 0.0, 0.4, 0.15), min_conf=0.6
        )
        request_select = screen.find_any_text(
            ["Request Select", "選擇委託", "选择委托"],
            min_conf=0.5
        )
        on_special_page = special_header or request_select

        if not on_special_page:
            # Not on 特殊任務 page yet — could be loading or transitioning
            return action_wait(500, "waiting for 特殊任務 page to load")

        # We're confirmed on the 特殊任務 page. Check for 活動進行中.
        # The badge appears in the left panel area (~x < 0.5).
        badge = screen.find_any_text(
            ["活動進行中", "活动进行中"],
            region=(0.0, 0.05, 0.5, 0.5), min_conf=0.5
        )
        if badge:
            self.log("特殊任務 has 活動進行中! Farming here.")
            self._event_on_special = True
            self._target = "special"
            self.sub_state = "select_credit"
            return action_wait(300, "特殊任務 has event bonus, farming")

        # No 活動進行中 on 特殊任務 page either → back to 活動 page
        self.log("特殊任務 has NO 活動進行中, backing out")
        self.sub_state = "enter_event"
        return action_back("back from 特殊任務 (no event bonus)")

    def _scan_bonuses(self, screen: ScreenState) -> Dict[str, Any]:
        """Scan campaign hub for event badges and decide target.

        Looks for:
        - "活動進行中" near 任務 or 特殊任務 → that section has event bonus
        - "距離結束還剩" → global event active
        """
        if self._scan_done:
            self.sub_state = "enter_target"
            return action_wait(200, "scan already done")

        self._scan_done = True

        # Detect event timer
        # OCR frequently misreads: 離→雕/離, drops 還, etc.
        event_timer = screen.find_any_text(
            ["距離結束還剩", "距离结束还剩", "結束還剩", "结束还剩",
             "結束剩", "结束剩"],
            min_conf=0.5
        )
        if not event_timer:
            event_timer = screen.find_text_one(r"距.{0,2}[结結]束.{0,2}剩", min_conf=0.5)
        if event_timer:
            self._has_event = True
            self.log(f"global event detected: '{event_timer.text}'")

        # Find all "活動進行中" / "活动进行中" badges
        # OCR misreads 動→勤 frequently
        event_badges = screen.find_text("活動進行", min_conf=0.5)
        event_badges += screen.find_text("活动进行", min_conf=0.5)
        event_badges += screen.find_text("活勤進行", min_conf=0.5)
        event_badges += screen.find_text("活勤进行", min_conf=0.5)

        # Find the campaign section labels and check which ones have badges near them
        # OCR often truncates 特殊任務→特殊任 or misreads 務→努
        special_label = screen.find_any_text(
            ["特殊任務", "特殊任务", "特殊任"],
            min_conf=0.5
        )
        normal_label = screen.find_any_text(
            ["任務", "任务", "任努"],
            min_conf=0.5
        )
        # Filter out the one that's just the header
        if normal_label and special_label:
            # If normal_label is actually part of "特殊任務", skip it
            if abs(normal_label.cx - special_label.cx) < 0.05:
                normal_label = None

        # Check proximity: is any event badge near (above) the section label?
        for badge in event_badges:
            if special_label and self._is_badge_near(badge, special_label):
                self._event_on_special = True
                self.log("event bonus on 特殊任務!")
            if normal_label and self._is_badge_near(badge, normal_label):
                self._event_on_normal = True
                self.log("event bonus on 任務!")

        # If badges found but couldn't match to any label, assume event on special
        if event_badges and not self._event_on_special and not self._event_on_normal:
            self._event_on_special = True
            self.log(f"event badge found ({len(event_badges)}) but no label matched, assuming special")

        # If no badges found but event timer exists, both could have bonus
        if self._has_event and not self._event_on_special and not self._event_on_normal:
            self._event_on_special = True
            self._event_on_normal = True
            self.log("event timer found but no specific badge, assuming both have bonus")

        # Decision priority: 特殊任務(信用) > 任務 > Hard fallback
        if self._event_on_special:
            self._target = "special"
            self.log("target: 特殊任務 (event bonus + credit priority)")
        elif self._event_on_normal:
            self._target = "normal"
            self.log("target: 任務 (event bonus)")
        else:
            self._target = "hard_fallback"
            self.log("no event bonus detected, falling back to Hard mode")

        self.sub_state = "enter_target"
        return action_wait(300, f"scan done, target={self._target}")

    @staticmethod
    def _is_badge_near(badge: OcrBox, label: OcrBox) -> bool:
        """Check if an event badge is spatially near (above or overlapping) a label."""
        # Badge should be within ~0.15 horizontal and ~0.12 vertical of label
        dx = abs(badge.cx - label.cx)
        dy = label.cy - badge.cy  # badge usually above label
        return dx < 0.25 and -0.05 < dy < 0.22

    # Campaign hub button positions (normalized, measured from MuMu 3840x2160).
    # OCR is unreliable on this screen — use hardcoded positions as fallback.
    _HUB_NORMAL_MISSIONS = (0.62, 0.28)    # 任務 (Area 29) button
    _HUB_SPECIAL_MISSIONS = (0.55, 0.67)   # 特殊任務 button
    _HUB_BOUNTY = (0.56, 0.54)             # 懸賞通緝 button

    def _enter_target(self, screen: ScreenState) -> Dict[str, Any]:
        """Click the chosen target in campaign menu."""
        self._enter_attempts += 1

        if self._target == "special":
            btn = screen.find_any_text(
                ["特殊任務", "特殊任务", "特殊任", "Special"],
                min_conf=0.5
            )
            if btn:
                self.log("entering 特殊任務")
                self.sub_state = "select_credit"
                return action_click_box(btn, "click special missions")
            # Hardcoded fallback
            if self._enter_attempts > 2:
                self.log("OCR miss, clicking 特殊任務 at hardcoded position")
                self.sub_state = "select_credit"
                return action_click(*self._HUB_SPECIAL_MISSIONS, "click special missions (hardcoded)")

        elif self._target == "normal":
            for box in screen.find_text("任務", min_conf=0.5):
                if "特殊" not in box.text and "懸賞" not in box.text and "悬" not in box.text:
                    self.log("entering 任務")
                    self.sub_state = "scroll_bottom"
                    return action_click_box(box, "click normal missions")
            # Also try "Area" text which is near the normal missions button
            area_hit = screen.find_text_one("Area", min_conf=0.6)
            if area_hit and area_hit.cy < 0.40:
                self.log("entering normal missions via Area text")
                self.sub_state = "scroll_bottom"
                return action_click_box(area_hit, "click normal missions via Area")
            if self._enter_attempts > 2:
                self.log("OCR miss, clicking 任務 at hardcoded position")
                self.sub_state = "scroll_bottom"
                return action_click(*self._HUB_NORMAL_MISSIONS, "click normal missions (hardcoded)")

        elif self._target == "hard_fallback":
            # Same as normal but go to hard tab after
            for box in screen.find_text("任務", min_conf=0.5):
                if "特殊" not in box.text and "懸賞" not in box.text and "悬" not in box.text:
                    self.log("entering normal missions for hard fallback")
                    self.sub_state = "select_hard_tab"
                    return action_click_box(box, "click normal missions")
            area_hit = screen.find_text_one("Area", min_conf=0.6)
            if area_hit and area_hit.cy < 0.40:
                self.log("entering normal missions via Area text (hard fallback)")
                self.sub_state = "select_hard_tab"
                return action_click_box(area_hit, "click normal missions via Area")
            if self._enter_attempts > 2:
                self.log("OCR miss, clicking 任務 at hardcoded position (hard fallback)")
                self.sub_state = "select_hard_tab"
                return action_click(*self._HUB_NORMAL_MISSIONS, "click normal missions (hardcoded)")

        return action_wait(500, f"looking for {self._target}")

    def _select_credit(self, screen: ScreenState) -> Dict[str, Any]:
        """In special missions sub-menu, pick credit (信用) category.

        Special missions typically have:
        - 據點防衛 (Base Defense → EXP books)
        - 信用貨幣回收 (Credit Recovery → credits) ← we want this
        """
        if self._in_credit_tab:
            self.sub_state = "scroll_bottom"
            return action_wait(300, "already in credit tab")

        # Look for credit category
        credit = screen.find_any_text(
            ["信用", "貨幣回收", "货币回收", "Credit"],
            min_conf=0.5
        )
        if credit:
            self.log(f"selecting credit category: '{credit.text}'")
            self._in_credit_tab = True
            self.sub_state = "scroll_bottom"
            return action_click_box(credit, "select credit category")

        # If we see the special mission categories but no credit, check others
        # Maybe we see 據點防衛 (base defense) — click right side for credit
        base_def = screen.find_any_text(
            ["據點防衛", "据点防卫", "Base"],
            min_conf=0.5
        )
        if base_def:
            # Credit is usually the other option — click to the right or below
            self.log("base defense visible, looking for credit tab to the right")
            credit_area = screen.find_any_text(
                ["信用", "回收"],
                min_conf=0.4
            )
            if credit_area:
                self._in_credit_tab = True
                self.sub_state = "scroll_bottom"
                return action_click_box(credit_area, "select credit from tabs")
            # Fallback: credit tab might be to the right of base defense
            return action_click(
                min(base_def.cx + 0.25, 0.9), base_def.cy,
                "click right of base defense for credit"
            )

        # If we're already in a stage list (maybe game skipped category select)
        stage = screen.find_any_text(
            ["掃蕩", "扫荡", "Sweep"],
            min_conf=0.6
        )
        if stage:
            self.log("already in stage list, skipping credit selection")
            self._in_credit_tab = True
            self.sub_state = "scroll_bottom"
            return action_wait(200, "already in stage list")

        # If we're on the 活動 page (stuck here instead of inside special missions),
        # detect it and go back to enter_event to click the 入場 button properly
        event_header = screen.find_any_text(
            ["活動", "活动"],
            region=(0.0, 0.0, 0.20, 0.08), min_conf=0.7
        )
        item_method = screen.find_any_text(
            ["道具獲得方法", "道具获得方法"],
            min_conf=0.5
        )
        if event_header or item_method:
            self.log("still on 活動 page, routing back to enter_event")
            self.sub_state = "enter_event"
            return action_wait(300, "back to enter_event from stuck select_credit")

        return action_wait(500, "looking for credit category")

    def _select_hard_tab(self, screen: ScreenState) -> Dict[str, Any]:
        """Click Hard tab inside normal missions (hard_fallback path)."""
        if screen.is_lobby():
            self.log("hard fallback drifted to lobby, re-entering campaign")
            self._hard_tab_attempts = 0
            self._hard_stage_list_seen = 0
            self._enter_attempts = 0
            self.sub_state = "enter_campaign"
            return action_wait(300, "re-enter campaign from lobby")

        touch_to_start = screen.find_text_one("TOUCH\\s*TO\\s*START", min_conf=0.6)
        if touch_to_start:
            self.log("hard fallback reached title screen, tapping to return")
            return action_click(0.5, 0.86, "tap to start from title")

        # Detect if we're on the campaign HUB (懸賞通緝/總力戰 visible) instead of
        # inside a mission page. Campaign hub has no Hard tab → bail out.
        hub_indicators = screen.find_any_text(
            ["懸賞通緝", "悬赏通缉", "總力戰", "总力战", "大決戰", "大决战",
             "學園交流會", "学园交流会", "戰術大賽", "战术大赛"],
            min_conf=0.5
        )
        if hub_indicators:
            self._hard_tab_attempts = getattr(self, '_hard_tab_attempts', 0) + 1
            self._hard_stage_list_seen = 0
            if self._hard_tab_attempts > 3:
                self.log("on campaign hub (no Hard tab here), exiting")
                self.sub_state = "exit"
                return action_wait(200, "campaign hub has no Hard tab, exiting")
            # Try clicking 任務 entry to enter normal missions
            mission_entry = screen.find_any_text(
                ["任務", "任务"],
                region=(0.35, 0.20, 0.65, 0.40), min_conf=0.6
            )
            if mission_entry:
                return action_click_box(mission_entry, "click normal missions from hub")
            return action_wait(500, "on campaign hub, looking for mission entry")

        # Look for Hard tab
        hard = screen.find_any_text(
            ["困難", "困难", "Hard", "HARD"],
            min_conf=0.6
        )
        if hard:
            self.log("clicking Hard tab")
            self._hard_stage_list_seen = 0
            self.sub_state = "scroll_bottom"
            return action_click_box(hard, "select Hard tab")

        # If we see Normal tab, Hard is usually next to it
        normal = screen.find_any_text(
            ["普通", "Normal", "NORMAL"],
            min_conf=0.6
        )
        if normal:
            self.log("Normal tab visible, clicking right for Hard")
            self._hard_stage_list_seen = 0
            return action_click(min(normal.cx + 0.15, 0.9), normal.cy, "click Hard tab area")

        # Maybe we're already in a stage list
        sweep = screen.find_any_text(["掃蕩", "扫荡", "Sweep"], min_conf=0.6)
        if sweep:
            self._hard_stage_list_seen = 0
            self.sub_state = "scroll_bottom"
            return action_wait(200, "already in stage list")

        stage_id = screen.find_text_one(r"\d+\-\d+", min_conf=0.5)
        stage_entry = screen.find_any_text(
            ["入場", "入场"],
            region=(0.78, 0.22, 0.98, 0.80),
            min_conf=0.5,
        )
        if stage_id and stage_entry:
            self._hard_stage_list_seen = getattr(self, '_hard_stage_list_seen', 0) + 1
            if self._hard_stage_list_seen <= 2:
                self.log("stage list visible but Hard OCR missed, clicking Hard tab fallback")
                return action_click(0.84, 0.22, "click Hard tab area (fallback)")
            self.log("stage list persists after Hard fallback, assuming Hard selected")
            self.sub_state = "scroll_bottom"
            return action_wait(200, "assume Hard stage list ready")

        # Timeout after too many attempts
        self._hard_tab_attempts = getattr(self, '_hard_tab_attempts', 0) + 1
        if self._hard_tab_attempts > 20:
            self.log("Hard tab not found after 20 attempts, exiting")
            self.sub_state = "exit"
            return action_wait(200, "Hard tab timeout")

        return action_wait(500, "looking for Hard tab")

    def _scroll_bottom(self, screen: ScreenState) -> Dict[str, Any]:
        """Scroll down inside the 關卡目錄 panel using mouse wheel.

        CRITICAL: Must use mouse wheel, NOT drag/swipe.
        The 關卡目錄 panel is on the RIGHT side (~x=0.55-0.95).
        Scroll must target inside that panel for it to work.

        Stage lists are typically ~14 stages with ~5 visible at once,
        so 3 large scrolls are sufficient to reach the bottom.
        """
        self._scroll_count += 1

        # After enough scrolls, assume we're at bottom
        if self._scroll_count > 3:
            self.log("scroll done, selecting bottom stage")
            self.sub_state = "select_stage"
            return action_wait(500, "finished scrolling")

        # Mouse wheel scroll DOWN inside the 關卡目錄 panel
        # Panel center is approximately at (0.75, 0.45)
        # clicks=-10 = scroll down 10 notches per tick (aggressive)
        self.log(f"wheel scroll down (attempt {self._scroll_count}/3)")
        return action_scroll(0.75, 0.45, clicks=-10, reason="wheel scroll stage list down")

    def _select_stage(self, screen: ScreenState) -> Dict[str, Any]:
        """Click the bottom-most visible 入場 button in the 關卡目錄 panel.

        After scrolling to bottom, we want the LAST (bottom-most) stage.
        Click its 入場 button directly — do NOT click the stage number/name,
        as that doesn't reliably open the stage info popup.

        The 入場 buttons are on the right side of the 關卡目錄 panel (~x=0.85-0.95).
        """
        # Find all 入場 buttons in the right-side panel
        enter_buttons: List[OcrBox] = []
        for box in screen.find_text("入場", min_conf=0.5):
            # Must be in the right-side 關卡目錄 panel area (x > 0.5)
            if box.cx > 0.5:
                enter_buttons.append(box)
        for box in screen.find_text("入场", min_conf=0.5):
            if box.cx > 0.5:
                enter_buttons.append(box)

        if enter_buttons:
            remaining_boxes = screen.find_text(r"[剩余餘].*[次數数].*(\d+)\s*[/|]\s*(\d+)", min_conf=0.5)
            available_buttons: List[OcrBox] = []
            for btn in enter_buttons:
                remaining = None
                for box in remaining_boxes:
                    if abs(box.cy - btn.cy) < 0.08 and box.cx < btn.cx:
                        m = re.search(r"(\d+)\s*[/|]\s*(\d+)", box.text)
                        if m:
                            remaining = int(m.group(1))
                            break
                if remaining is None or remaining > 0:
                    available_buttons.append(btn)

            if not available_buttons:
                self.log("all visible hard stages are out of attempts, exiting safely")
                self.sub_state = "exit"
                return action_wait(200, "no hard stages with attempts left")

            bottom_btn = max(available_buttons, key=lambda b: b.cy)
            self.log(f"clicking bottom available 入場 at y={bottom_btn.cy:.2f}")
            self.sub_state = "sweep"
            self._sweep_stage = 0
            return action_click_box(bottom_btn, f"click bottom available 入場 button")

        # No 入場 found — maybe 關卡目錄 not visible, try waiting
        self.log("no 入場 buttons found")
        return action_wait(500, "looking for 入場 buttons")

    def _sweep(self, screen: ScreenState) -> Dict[str, Any]:
        """Multi-step sweep inside the 任務資訊 popup.

        Flow after clicking 入場:
        1. 任務資訊 popup opens → shows 掃蕩 section with MIN/MAX and 掃蕩開始
        2. Click MAX to set sweep count to maximum
        3. Click 掃蕩開始 to start sweep
        4. Dismiss result popup

        IMPORTANT: OCR sometimes reads 任務資訊 as 任務資讯 (簡繁混合).
        So instead of matching the title text, detect the popup by looking
        for MAX/MIN buttons which are reliable OCR targets (English text).
        """
        # Stage 0: Wait for 任務資訊 popup, then click MAX
        if self._sweep_stage == 0:
            # Detect popup by looking for MAX button (most reliable indicator).
            # OCR reliably reads "MAX" (English, conf ~0.98).
            max_btn = screen.find_any_text(
                ["MAX"],
                min_conf=0.7
            )
            if max_btn:
                self.log("popup open (MAX visible), clicking MAX")
                self._sweep_stage = 1
                return action_click_box(max_btn, "click MAX for sweep count")

            # Secondary check: MIN button or 任務資訊/資讯 title
            popup_indicators = screen.find_any_text(
                ["MIN", "任務資訊", "任務資讯", "任务资讯"],
                min_conf=0.6
            )
            if popup_indicators:
                self.log("popup open but MAX not found yet")
                return action_wait(300, "popup open, looking for MAX")

            # Popup not open yet — click bottom 入場 again
            enter_buttons: List[OcrBox] = []
            for box in screen.find_text("入場", min_conf=0.5):
                if box.cx > 0.5:
                    enter_buttons.append(box)
            for box in screen.find_text("入场", min_conf=0.5):
                if box.cx > 0.5:
                    enter_buttons.append(box)
            if enter_buttons:
                bottom_btn = max(enter_buttons, key=lambda b: b.cy)
                return action_click_box(bottom_btn, "re-click bottom 入場")
            return action_wait(400, "waiting for stage popup")

        # Stage 1: Click 掃蕩開始
        if self._sweep_stage == 1:
            notify = screen.find_any_text(
                ["通知"],
                region=(0.3, 0.1, 0.7, 0.3), min_conf=0.6
            )
            if notify:
                confirm = screen.find_any_text(
                    ["確認", "确认", "確定", "确定", "確", "确", "Confirm"],
                    region=(0.3, 0.3, 0.8, 0.92), min_conf=0.5
                )
                self._sweep_stage = 2
                if confirm:
                    return action_click_box(confirm, "confirm sweep (stage1 popup)")
                return action_click(0.60, 0.74, "confirm sweep (stage1 hardcoded)")

            sweep_start = screen.find_any_text(
                ["掃蕩開始", "扫荡开始"],
                min_conf=0.5
            )
            if sweep_start:
                self.log("clicking 掃蕩開始")
                self._sweep_stage = 2
                return action_click_box(sweep_start, "click 掃蕩開始")

            # OCR often reads 掃蕩開始 as just "開始" (conf ~0.86).
            # The 掃蕩開始 button is at ~y=0.54-0.59 in the sweep panel.
            # The 任務開始 button is at ~y=0.72-0.77 (we do NOT want this).
            # So match "開始" only in the upper region (y < 0.65).
            for box in screen.find_text("開始", min_conf=0.5):
                if 0.4 < box.cy < 0.65 and box.cx > 0.5:
                    self.log(f"clicking 掃蕩開始 (OCR: '{box.text}' at y={box.cy:.2f})")
                    self._sweep_stage = 2
                    return action_click_box(box, "click 掃蕩開始 (partial OCR)")
            for box in screen.find_text("开始", min_conf=0.5):
                if 0.4 < box.cy < 0.65 and box.cx > 0.5:
                    self._sweep_stage = 2
                    return action_click_box(box, "click 掃蕩開始 (partial OCR)")

            for box in screen.find_text("蕩", min_conf=0.45):
                if 0.45 < box.cy < 0.66 and box.cx > 0.55:
                    self._sweep_stage = 2
                    return action_click_box(box, "click 掃蕩開始 (single-char OCR)")
            for box in screen.find_text("荡", min_conf=0.45):
                if 0.45 < box.cy < 0.66 and box.cx > 0.55:
                    self._sweep_stage = 2
                    return action_click_box(box, "click 掃蕩開始 (single-char OCR)")

            # Fallback: look for 掃蕩 text in the panel
            for box in screen.find_text("掃蕩", min_conf=0.5):
                if box.cy > 0.25 and box.cx > 0.5:
                    self._sweep_stage = 2
                    return action_click_box(box, "click sweep start")
            for box in screen.find_text("扫荡", min_conf=0.5):
                if box.cy > 0.25 and box.cx > 0.5:
                    self._sweep_stage = 2
                    return action_click_box(box, "click sweep start")
            # Final fallback: the upper blue sweep button is stable in this popup.
            if screen.find_any_text(["MAX", "MIN"], region=(0.55, 0.35, 0.92, 0.52), min_conf=0.6):
                self._sweep_stage = 2
                return action_click(0.73, 0.58, "click 掃蕩開始 (hardcoded upper button)")
            return action_wait(400, "looking for 掃蕩開始")

        # Stage 2: Confirm sweep dialog ("通知: 要使用XAP掃蕩Y次嗎？")
        # This dialog has 取消(Esc) and 確認(Space) buttons.
        # The global interceptor may also handle this, but we handle it here too.
        if self._sweep_stage == 2:
            restore_prompt = screen.find_any_text(
                ["恢復挑", "恢复挑", "青輝石", "青辉石", "可購買0次", "可购买0次"],
                region=(0.30, 0.28, 0.72, 0.62), min_conf=0.55
            )
            if restore_prompt:
                cancel_restore = screen.find_any_text(
                    ["取消"],
                    region=(0.28, 0.60, 0.52, 0.82), min_conf=0.55
                )
                self.log("challenge restore dialog detected, cancelling to avoid spending gems")
                self.sub_state = "exit"
                if cancel_restore:
                    return action_click_box(cancel_restore, "cancel challenge restore dialog")
                return action_click(0.41, 0.70, "cancel challenge restore dialog (fallback)")

            # Look for 確認 button in center area (the confirm dialog)
            # OCR sometimes reads 確認 as single char 確, so match both.
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "Confirm"],
                region=(0.3, 0.3, 0.8, 0.95), min_conf=0.6
            )
            if confirm:
                self._sweep_stage = 3
                return action_click_box(confirm, "confirm sweep")

            # Check if the 通知 dialog is visible (sweep confirm)
            notify = screen.find_any_text(
                ["通知"],
                region=(0.3, 0.1, 0.7, 0.3), min_conf=0.55
            )
            cancel = screen.find_any_text(
                ["取消"],
                region=(0.28, 0.58, 0.52, 0.82), min_conf=0.55
            )
            ap_prompt = screen.find_any_text(
                ["要使用", "AP"],
                region=(0.32, 0.38, 0.68, 0.56), min_conf=0.55
            )
            if notify or cancel or ap_prompt:
                # Dialog visible but OCR missed the confirm button — click fixed confirm spot.
                return action_click(0.60, 0.74, "confirm sweep (hardcoded)")

            # Check if sweep result already appeared (interceptor handled confirm)
            sweep_done = screen.find_any_text(
                ["掃蕩完成", "扫荡完成"],
                min_conf=0.5
            )
            if sweep_done:
                self.log("sweep result detected in stage 2, advancing to stage 3")
                self._sweep_stage = 3
                return action_wait(200, "sweep result appeared")

            skip = screen.find_any_text(["跳過", "跳过", "Skip"], min_conf=0.7)
            if skip:
                return action_click_box(skip, "skip animation")

            # Check if popup is still open (MAX button in sweep panel's right side)
            # Region filter avoids matching exp bar "MAX" on sweep result screen
            max_btn = screen.find_any_text(
                ["MAX"], min_conf=0.7,
                region=(0.7, 0.3, 0.95, 0.5)
            )
            if max_btn:
                # Still on popup, confirm dialog hasn't appeared yet — wait
                return action_wait(400, "waiting for confirm dialog")

            # Maybe sweep already started (no confirm needed)
            self._sweep_stage = 3
            return action_wait(500, "waiting for sweep result")

        # Stage 3+: Dismiss sweep result and exit
        if self._sweep_stage >= 3:
            ok = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "OK"],
                min_conf=0.6
            )
            if ok:
                self.log(f"sweep done ({self._sweep_count} total)")
                self.sub_state = "exit"
                return action_click_box(ok, "dismiss result")
            # Click anywhere to dismiss result screen
            return action_click(0.5, 0.9, "dismiss sweep result")

        return action_wait(400, "sweep processing")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        touch_to_start = screen.find_text_one("TOUCH\\s*TO\\s*START", min_conf=0.6)
        if touch_to_start:
            return action_click(0.5, 0.86, "tap to start during event_farming exit")

        if screen.is_lobby():
            ap_now = self._parse_ap(screen)
            if ap_now >= 0:
                self._ap_after = ap_now
            ap_span = f"{self._ap_before}->{self._ap_after}" if self._ap_before >= 0 and self._ap_after >= 0 else "unknown"
            ap_spent = 0
            if self._ap_before >= 0 and self._ap_after >= 0:
                ap_spent = max(0, self._ap_before - self._ap_after)

            if not self._ap_empty and self._sweep_count <= 0 and ap_spent <= 0:
                reason = f"event_farming no_sweep (target={self._target}, AP {ap_span})"
                self.log(reason)
                return action_done(reason)

            reason = (
                f"event_farming complete (target={self._target}, sweeps={self._sweep_count}, "
                f"ap_empty={self._ap_empty}, ap_spent={ap_spent}, AP {ap_span})"
            )
            self.log(reason)
            return action_done(reason)
        return action_back("event_farming exit: back to lobby")
