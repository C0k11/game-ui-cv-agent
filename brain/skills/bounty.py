"""BountySkill: Handle 悬赏通缉 (Bounties) sweep.

Flow:
1. ENTER: From lobby, click 任務 → navigate to bounties tab
2. CHECK_TICKETS: OCR parse "剩余 X/Y" — if 0, exit immediately
3. SELECT_STAGE: Click the highest/last stage in list
4. SWEEP: Click 掃蕩 → Max → 確認 — full sweep flow
5. EXIT: Back to lobby

Key detection patterns:
- Ticket count: "剩余 X/Y", "X/Y" near ticket area
- AP exhausted: "AP不足", "體力不足"
- Sweep buttons: "掃蕩", "最大", "確認", "扫荡", "Max"
"""
from __future__ import annotations

import re
from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_click_yolo,
    action_wait, action_back, action_done, action_scroll,
)


class BountySkill(BaseSkill):
    def __init__(self):
        super().__init__("Bounty")
        self.max_ticks = 60
        self._tickets_remaining: int = -1  # -1 = unknown
        self._sweep_stage: int = 0  # 0=select, 1=click sweep, 2=click max, 3=confirm, 4=done
        self._sweep_attempts: int = 0
        self._enter_attempts: int = 0

    def reset(self) -> None:
        super().reset()
        self._tickets_remaining = -1
        self._sweep_stage = 0
        self._sweep_attempts = 0
        self._stage_ticks = 0
        self._loc_ticks = 0
        self._enter_attempts = 0

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("bounty timeout")

        # AP exhausted popup — can appear at any time during sweep
        no_ap = screen.find_any_text(
            ["AP不足", "體力不足", "体力不足", "購買AP", "购买AP"],
            min_conf=0.6
        )
        if no_ap:
            self.log("AP exhausted")
            self.sub_state = "exit"
            cancel = screen.find_any_text(["取消"], min_conf=0.7)
            if cancel:
                return action_click_box(cancel, "cancel AP purchase")
            close_btn = self._find_florence_hit(
                screen,
                ["close button icon", "close dialog x button", "x close icon"],
                region=(0.0, 0.0, 0.93, 0.30),
            )
            if close_btn:
                return action_click_box(close_btn, "close AP popup")
            return action_back("dismiss AP popup")

        # Sweep result popup — dismiss and continue
        result = screen.find_any_text(
            ["獲得獎勵", "获得奖励", "戰鬥結果", "战斗结果"],
            region=screen.CENTER, min_conf=0.6
        )
        if result:
            self.log("sweep result popup, dismissing")
            return action_click(0.5, 0.9, "dismiss sweep result")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "bounty loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "check_tickets":
            return self._check_tickets(screen)
        if self.sub_state == "select_location":
            return self._select_location(screen)
        if self.sub_state == "select_stage":
            return self._select_stage(screen)
        if self.sub_state == "sweep":
            return self._sweep(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "bounty unknown state")

    def _is_bounty_screen(self, screen: ScreenState) -> bool:
        """Detect bounty screen via unique markers (header OCR is unreliable).

        Bounty screen has: "票券" (ticket count), "關卡目錄" (stage catalog),
        "入場" buttons on the right side.
        """
        # Check header region for any bounty-like text (including partial OCR)
        header = screen.find_any_text(
            ["懸賞通緝", "悬赏通缉", "懸賞", "悬赏", "通緝", "通缉", "悬通"],
            region=(0.0, 0.0, 0.3, 0.08), min_conf=0.5
        )
        if header:
            return True
        # Fallback: look for bounty-specific UI elements
        tickets = screen.find_any_text(
            ["通緝票券", "通缉票券", "通票券"],
            region=(0.0, 0.15, 0.35, 0.35), min_conf=0.7
        )
        if tickets:
            return True
        catalog = screen.find_any_text(
            ["關卡目錄", "关卡目录"],
            region=(0.55, 0.10, 0.85, 0.20), min_conf=0.7
        )
        if catalog:
            return True
        return False

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_attempts += 1
        current = self.detect_current_screen(screen)

        # Direct bounty screen detection (header OCR often fails)
        if current == "Bounty" or self._is_bounty_screen(screen):
            self.log("inside bounty")
            self.sub_state = "check_tickets"
            return action_wait(500, "entered bounty")

        # Accidentally entered Daily Tasks? Back out.
        if current == "DailyTasks":
            self.log("wrong screen: Daily Tasks, backing out")
            return action_back("back from Daily Tasks")

        if current == "Lobby":
            # "任務" is NOT in the bottom nav bar. It's on the RIGHT SIDE
            # of the lobby (~0.93, 0.83) as a campaign entry button.
            # Same approach as EventFarmingSkill._enter_campaign.
            campaign_btn = screen.find_any_text(
                ["任務", "任务"],
                region=(0.80, 0.70, 1.0, 0.90), min_conf=0.6
            )
            if campaign_btn:
                self.log(f"clicking campaign entry '{campaign_btn.text}'")
                return action_click_box(campaign_btn, "click campaign entry (right side)")
            # Hardcoded fallback — campaign button position on lobby
            return action_click(0.95, 0.83, "click campaign area (hardcoded)")

        if current and current not in ("Bounty", "Mission"):
            self.log(f"wrong screen '{current}', backing out")
            return action_back(f"back from {current}")

        # In Mission/Campaign hub — find 懸賞通緝 menu item
        # OCR frequently misreads: "懸賞通緝" → "通" / "悬通" / "懸賞" / partial
        # Full search includes partial OCR variants
        if current == "Mission":
            bounty_tab = screen.find_any_text(
                ["懸賞通緝", "悬赏通缉", "懸賞", "悬赏", "通緝", "通缉",
                 "悬通", "Bounty"],
                min_conf=0.5
            )
            if bounty_tab:
                self.log(f"clicking bounty '{bounty_tab.text}'")
                return action_click_box(bounty_tab, "click bounty in campaign")

            # OCR may read just "通" — single char at the bounty icon position
            # Campaign grid: bounty is at ~(0.56, 0.55) from trajectory data
            single = screen.find_text_one("通", region=(0.45, 0.45, 0.70, 0.60),
                                          min_conf=0.8)
            if single and len(single.text) <= 2:
                self.log(f"clicking bounty (OCR partial '{single.text}')")
                return action_click_box(single, "click bounty (partial OCR)")

            # Hardcoded fallback: bounty position in campaign grid
            if self._enter_attempts > 5:
                self.log("clicking bounty at hardcoded position")
                return action_click(0.56, 0.55, "click bounty (hardcoded)")

        if self._enter_attempts > 20:
            self.log("can't reach bounty, giving up")
            self.sub_state = "exit"
            return action_wait(300, "bounty enter timeout")

        return action_wait(500, "entering bounty")

    def _check_tickets(self, screen: ScreenState) -> Dict[str, Any]:
        """Check remaining bounty tickets via OCR (票券 X/Y or similar).

        OCR reads "懸賞通緝票券 6/6" as "通票券6/6".
        Must filter to ticket-related text to avoid matching AP "57/240".
        """
        # Check if "持有票券" label exists (confirms we're on bounty screen)
        has_ticket_label = screen.find_any_text(
            ["持有票券", "票券"],
            region=(0.0, 0.08, 0.25, 0.20), min_conf=0.7
        )

        for box in screen.ocr_boxes:
            if box.confidence < 0.5:
                continue
            m = re.search(r'(\d+)\s*/\s*(\d+)', box.text)
            if not m:
                continue
            # Prioritize ticket-specific text (票券/剩余/次數)
            # Exclude top-bar AP/currency which also has X/Y format
            is_ticket = any(k in box.text for k in ["票券", "剩余", "剩餘", "次數", "次数"])
            # Also match if "持有票券" label nearby and box is in ticket area
            if not is_ticket and has_ticket_label and box.cy > 0.10 and box.cy < 0.20:
                is_ticket = True
            if is_ticket or (box.cy > 0.10 and "/" in box.text and int(m.group(2)) <= 20):
                remaining = int(m.group(1))
                total = int(m.group(2))
                self._tickets_remaining = remaining
                self.log(f"bounty tickets: {remaining}/{total}")
                if remaining == 0:
                    self.log("no tickets remaining, exiting")
                    self.sub_state = "exit"
                    return action_wait(300, "no bounty tickets")
                self.sub_state = "select_location"
                return action_wait(300, f"have {remaining} tickets, selecting location")

        # If we can't parse tickets after a few ticks, proceed anyway
        if self.ticks > 10:
            self.log("couldn't parse ticket count, proceeding to select location")
            self.sub_state = "select_location"
            return action_wait(300, "ticket parse timeout")

        return action_wait(500, "looking for ticket count")

    def _select_location(self, screen: ScreenState) -> Dict[str, Any]:
        """Select a bounty location from LocationSelect screen.

        The bounty LocationSelect shows 3 location cards:
        - 高架公路 (~y 0.22-0.27)
        - 沙漠鐵道 (~y 0.38-0.43)
        - 教室 (~y 0.54-0.59)

        Click the first available location to enter its stage list.
        If already past LocationSelect (stage list visible with 入場), skip ahead.
        """
        self._loc_ticks = getattr(self, '_loc_ticks', 0) + 1

        # Already in stage list? (入場 buttons visible → skip to select_stage)
        enter_btns = [
            b for b in screen.ocr_boxes
            if b.confidence >= 0.6 and b.cx > 0.70
            and ("入場" in b.text or "入场" in b.text)
        ]
        if enter_btns:
            self.log("already in stage list, skipping to select_stage")
            self.sub_state = "select_stage"
            self._stage_ticks = 0
            return action_wait(300, "stage list visible")

        # Look for location names and click them
        locations = screen.find_any_text(
            ["高架公路", "沙漠鐵道", "沙漠铁道", "教室"],
            region=(0.70, 0.15, 1.0, 0.70), min_conf=0.7
        )
        if locations:
            self.log(f"clicking location '{locations.text}'")
            return action_click_box(locations, f"select bounty location '{locations.text}'")

        # Fallback: click first location card area
        if self._loc_ticks > 3:
            self.log("clicking first location (fallback)")
            return action_click(0.90, 0.25, "click first bounty location (fallback)")

        if self._loc_ticks > 6:
            self.log("location select timeout, trying select_stage anyway")
            self.sub_state = "select_stage"
            self._stage_ticks = 0
            return action_wait(300, "location select timeout")

        return action_wait(500, "waiting for location select")

    def _select_stage(self, screen: ScreenState) -> Dict[str, Any]:
        """Select the last (highest) stage from the stage list (關卡目錄).

        The bounty screen shows:
        - Left panel: branch info (懸賞通緝：教室, ticket count, school buffs)
        - Right panel: stage list (關卡目錄) with numbered stages + 入場 buttons

        We scroll the stage list down to ensure the last stage is visible,
        then click the bottom-most 入場 button.
        """
        self._stage_ticks = getattr(self, '_stage_ticks', 0) + 1

        # Look for 入場 buttons on the RIGHT side (stage list, x > 0.70)
        enter_btns = [
            b for b in screen.ocr_boxes
            if b.confidence >= 0.6 and b.cx > 0.70
            and ("入場" in b.text or "入场" in b.text)
        ]

        if enter_btns:
            # First 2 ticks: scroll stage list down to reveal the last stage
            if self._stage_ticks <= 2:
                self.log("scrolling stage list down")
                return action_scroll(0.75, 0.60, clicks=-5,
                                     reason="scroll stage list to bottom")

            # Click the bottom-most 入場 button (= last/hardest stage)
            last = max(enter_btns, key=lambda b: b.cy)
            self.log(f"clicking last stage 入場 at ({last.cx:.2f},{last.cy:.2f})")
            self.sub_state = "sweep"
            self._sweep_stage = 0
            return action_click_box(last, "enter last stage")

        # No 入場 buttons visible — might need to scroll or stage list not loaded
        if self._stage_ticks <= 4:
            return action_scroll(0.75, 0.60, clicks=-5,
                                 reason="scroll to find stages")

        # Fallback: click bottom-right area where last 入場 typically is
        self.log("clicking last stage (hardcoded fallback)")
        self.sub_state = "sweep"
        self._sweep_stage = 0
        return action_click(0.87, 0.88, "enter last stage (fallback)")

    def _sweep(self, screen: ScreenState) -> Dict[str, Any]:
        """Multi-step sweep inside the 任務資訊 popup.

        Flow after clicking 入場:
        1. 任務資訊 popup opens → shows MAX/MIN and 掃蕩開始
        2. Click MAX to set sweep count
        3. Click 掃蕩開始 to start sweep
        4. Confirm dialog (通知: 要使用XAP掃蕩Y次嗎？) → 確認
        5. Dismiss result popup
        """
        self._sweep_attempts += 1

        if self._sweep_attempts > 25:
            self.log("sweep stuck, exiting")
            self.sub_state = "exit"
            return action_wait(300, "sweep timeout")

        # Stage 0: Wait for 任務資訊 popup → click MAX
        if self._sweep_stage == 0:
            max_btn = screen.find_any_text(["MAX"], min_conf=0.7)
            if max_btn:
                self.log("popup open, clicking MAX")
                self._sweep_stage = 1
                return action_click_box(max_btn, "click MAX for sweep count")

            # Popup indicators
            popup = screen.find_any_text(
                ["MIN", "任務資訊", "任務資讯", "任务资讯"],
                min_conf=0.6
            )
            if popup:
                return action_wait(300, "popup open, looking for MAX")

            # Popup not open yet — re-click last 入場
            enter_btns = [
                b for b in screen.ocr_boxes
                if b.confidence >= 0.6 and b.cx > 0.70
                and ("入場" in b.text or "入场" in b.text)
            ]
            if enter_btns:
                last = max(enter_btns, key=lambda b: b.cy)
                return action_click_box(last, "re-click last 入場")
            return action_wait(400, "waiting for stage popup")

        # Stage 1: Click 掃蕩開始
        if self._sweep_stage == 1:
            sweep_start = screen.find_any_text(
                ["掃蕩開始", "扫荡开始"], min_conf=0.5
            )
            if sweep_start:
                self.log("clicking 掃蕩開始")
                self._sweep_stage = 2
                return action_click_box(sweep_start, "click 掃蕩開始")

            # Partial OCR: 掃蕩開始 may read as "開始" in upper region
            for box in screen.find_text("開始", min_conf=0.5):
                if 0.4 < box.cy < 0.65 and box.cx > 0.5:
                    self._sweep_stage = 2
                    return action_click_box(box, "click 掃蕩開始 (partial OCR)")
            for box in screen.find_text("开始", min_conf=0.5):
                if 0.4 < box.cy < 0.65 and box.cx > 0.5:
                    self._sweep_stage = 2
                    return action_click_box(box, "click 掃蕩開始 (partial OCR)")

            # Fallback: click 掃蕩 text in panel
            for box in screen.find_text("掃蕩", min_conf=0.5):
                if box.cy > 0.25 and box.cx > 0.5:
                    self._sweep_stage = 2
                    return action_click_box(box, "click sweep start")
            return action_wait(400, "looking for 掃蕩開始")

        # Stage 2: Confirm sweep dialog (通知: 要使用XAP掃蕩Y次嗎？)
        if self._sweep_stage == 2:
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "Confirm"],
                region=(0.3, 0.3, 0.8, 0.95), min_conf=0.6
            )
            if confirm:
                self._sweep_stage = 3
                return action_click_box(confirm, "confirm sweep")

            # Check if sweep result already appeared (interceptor handled confirm)
            sweep_done = screen.find_any_text(
                ["掃蕩完成", "扫荡完成"], min_conf=0.5
            )
            if sweep_done:
                self._sweep_stage = 3
                return action_wait(200, "sweep result appeared")

            skip = screen.find_any_text(["跳過", "跳过", "Skip"], min_conf=0.7)
            if skip:
                return action_click_box(skip, "skip animation")

            # If the 任務資訊 popup returned (MAX visible in sweep panel area),
            # the interceptor already handled confirm + result dismissal.
            # Close the popup and exit.
            if self._sweep_attempts > 8:
                popup_max = screen.find_any_text(
                    ["MAX"], region=(0.5, 0.3, 0.95, 0.5), min_conf=0.7
                )
                if popup_max:
                    self.log("sweep completed (popup returned)")
                    self.sub_state = "exit"
                    return action_back("close popup after sweep")

            return action_wait(400, "waiting for confirm dialog")

        # Stage 3+: Dismiss sweep result and exit
        if self._sweep_stage >= 3:
            ok = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "OK"],
                min_conf=0.6
            )
            if ok:
                self.log("sweep done, dismissing result")
                self.sub_state = "exit"
                return action_click_box(ok, "dismiss sweep result")
            # Click anywhere to dismiss
            return action_click(0.5, 0.9, "dismiss sweep result")

        return action_wait(400, "sweep processing")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log("done")
            return action_done("bounty complete")
        return action_back("bounty exit: back to lobby")
