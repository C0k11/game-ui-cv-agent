"""FarmingSkill: Handle Hard/Normal mode AP farming (刷体力).

Flow:
1. ENTER: From lobby, click 任務 → Hard tab
2. SCROLL_FIND: Scroll through stage list to find target stage (regex match)
3. SWEEP: Click 掃蕩 → Max → 確認 — repeat until AP exhausted or stage limit
4. EXIT: Back to lobby

This skill runs AFTER bounty/arena, consuming remaining AP on Hard stages
for character shards or equipment.

Key patterns:
- Hard tab: "困難" / "困难" / "Hard"
- Stage format: "14-3", "12-1" etc
- Sweep: "掃蕩" / "扫荡" / "最大" / "Max"
- AP exhausted: "AP不足" / "體力不足"
- Stage limit: greyed out or "次數已達上限"
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_click_yolo,
    action_wait, action_back, action_done, action_swipe,
)


class FarmingSkill(BaseSkill):
    def __init__(self):
        super().__init__("Farming")
        self.max_ticks = 100
        self._in_hard_tab: bool = False
        self._stage_found: bool = False
        self._sweep_stage: int = 0
        self._sweep_count: int = 0
        self._scroll_attempts: int = 0
        self._ap_empty: bool = False

    def reset(self) -> None:
        super().reset()
        self._in_hard_tab = False
        self._stage_found = False
        self._sweep_stage = 0
        self._sweep_count = 0
        self._scroll_attempts = 0
        self._ap_empty = False

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("farming timeout")

        # AP exhausted popup
        no_ap = screen.find_any_text(
            ["AP不足", "體力不足", "体力不足", "購買AP", "购买AP"],
            min_conf=0.6
        )
        if no_ap:
            self.log("AP exhausted, done farming")
            self._ap_empty = True
            self.sub_state = "exit"
            cancel = screen.find_any_text(["取消"], min_conf=0.7)
            if cancel:
                return action_click_box(cancel, "cancel AP purchase")
            x_btn = screen.find_yolo_one("叉叉", min_conf=0.3,
                                          region=(0.0, 0.0, 0.93, 1.0))
            if x_btn:
                return action_click_yolo(x_btn, "close AP popup")
            return action_back("dismiss AP popup")

        # Stage limit reached
        limit = screen.find_any_text(
            ["次數已達上限", "次数已达上限", "挑戰次數不足", "挑战次数不足"],
            min_conf=0.6
        )
        if limit:
            self.log("stage limit reached")
            self.sub_state = "exit"
            confirm = screen.find_any_text(["確認", "确认", "確", "确"], min_conf=0.6)
            if confirm:
                return action_click_box(confirm, "confirm stage limit")
            return action_back("dismiss stage limit")

        # Sweep result popup
        result = screen.find_any_text(
            ["獲得獎勵", "获得奖励", "戰鬥結果", "战斗结果"],
            region=screen.CENTER, min_conf=0.6
        )
        if result:
            self._sweep_count += 1
            self.log(f"sweep #{self._sweep_count} result, dismissing")
            return action_click(0.5, 0.9, "dismiss sweep result")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "farming loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "select_hard":
            return self._select_hard(screen)
        if self.sub_state == "scroll_find":
            return self._scroll_find(screen)
        if self.sub_state == "sweep":
            return self._sweep(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "farming unknown state")

    def _is_campaign_hub(self, screen: ScreenState) -> bool:
        """Detect the campaign hub (grid of modes) vs actual stage list.

        Campaign hub has: 懸賞通緝, 總力戰, 劇情, Area XX, 特殊任務, 學園交流會.
        Stage list has: Normal/Hard tabs, stage numbers, 關卡目錄.
        """
        hub_markers = screen.find_any_text(
            ["劇情", "总力", "總力", "綜合術", "綜合術", "學園交流",
             "制約解除", "大決", "特殊任務"],
            min_conf=0.5
        )
        if hub_markers:
            return True
        # Also check for "Area XX" text which is unique to campaign hub
        area = screen.find_text_one(r"Area\s*\d+", min_conf=0.5)
        if area:
            return True
        return False

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        current = self.detect_current_screen(screen)

        if current == "Mission":
            # Distinguish campaign hub (grid) from stage list (Normal/Hard tabs)
            if self._is_campaign_hub(screen):
                self.log("on campaign hub, need to click 任務 card to enter stage list")
                # Click the "任務" card in the campaign hub grid
                # It's in the top-center area with "Area XX" text below it
                mission_card = screen.find_any_text(
                    ["任務", "任务"],
                    region=(0.40, 0.15, 0.70, 0.35), min_conf=0.7
                )
                if mission_card:
                    return action_click_box(mission_card, "click 任務 card on campaign hub")
                # Fallback: click the known position of 任務 card
                return action_click(0.56, 0.24, "click 任務 card (hardcoded)")
            self.log("inside mission stage list")
            self.sub_state = "select_hard"
            return action_wait(300, "entered mission")

        # DailyTasks has same header "任務" — back out immediately
        if current == "DailyTasks":
            return action_back("back from DailyTasks")

        if current == "Lobby":
            # "任務" on the LEFT sidebar opens DailyTasks, NOT campaign.
            # Use the RIGHT-SIDE campaign entry button (~0.93, 0.83).
            campaign_btn = screen.find_any_text(
                ["任務", "任务"],
                region=(0.80, 0.70, 1.0, 0.90), min_conf=0.6
            )
            if campaign_btn:
                return action_click_box(campaign_btn, "click campaign entry (right side)")
            return action_click(0.95, 0.83, "click campaign area (hardcoded)")

        if current and current not in ("Mission", "DailyTasks"):
            return action_back(f"back from {current}")

        return action_wait(500, "entering mission")

    def _select_hard(self, screen: ScreenState) -> Dict[str, Any]:
        """Click Hard tab in mission menu."""
        if self._in_hard_tab:
            self.sub_state = "scroll_find"
            return action_wait(300, "already in hard tab")

        hard = screen.find_any_text(
            ["困難", "困难", "Hard", "HARD"],
            min_conf=0.6
        )
        if hard:
            self.log("clicking Hard tab")
            self._in_hard_tab = True
            self.sub_state = "scroll_find"
            return action_click_box(hard, "select Hard tab")

        # If we see Normal tab, look for Hard nearby
        normal = screen.find_any_text(
            ["普通", "Normal", "NORMAL"],
            min_conf=0.6
        )
        if normal:
            # Hard tab is usually next to Normal — click slightly to the right
            self.log("Normal tab visible, looking for Hard tab")
            return action_click(min(normal.cx + 0.15, 0.9), normal.cy, "click Hard tab area")

        return action_wait(500, "looking for Hard tab")

    def _scroll_find(self, screen: ScreenState) -> Dict[str, Any]:
        """Scroll through stage list to find available stages."""
        self._scroll_attempts += 1

        if self._scroll_attempts > 15:
            self.log("scroll exhausted, picking any available stage")
            # Click whatever stage is visible
            sweep = screen.find_any_text(
                ["掃蕩", "扫荡", "Sweep"],
                min_conf=0.6
            )
            if sweep:
                self.sub_state = "sweep"
                self._sweep_stage = 0
                return action_click_box(sweep, "sweep any visible stage")
            self.sub_state = "exit"
            return action_wait(300, "no sweepable stages found")

        # Look for sweepable stages — stages with a "掃蕩" button
        sweep = screen.find_any_text(
            ["掃蕩", "扫荡", "Sweep"],
            min_conf=0.6
        )
        if sweep:
            self.log(f"found sweepable stage, clicking sweep")
            self.sub_state = "sweep"
            self._sweep_stage = 1  # skip to Max since sweep already clicked
            return action_click_box(sweep, "click sweep on found stage")

        # Look for stage entries with numbers (e.g. "14-3", "Quest 3")
        # Click the first available one
        for box in screen.ocr_boxes:
            if box.confidence < 0.5:
                continue
            if re.search(r'\d+-\d+', box.text):
                self.log(f"found stage '{box.text}', clicking")
                self._stage_found = True
                return action_click_box(box, f"click stage {box.text}")

        # Scroll down to find more stages
        self.log(f"scrolling down to find stages (attempt {self._scroll_attempts})")
        return action_swipe(0.5, 0.7, 0.5, 0.3, 400, "scroll stage list down")

    def _sweep(self, screen: ScreenState) -> Dict[str, Any]:
        """Multi-step sweep: 掃蕩 → Max → 確認."""
        # Stage 0: Click sweep button
        if self._sweep_stage == 0:
            sweep = screen.find_any_text(
                ["掃蕩", "扫荡", "Sweep"],
                min_conf=0.6
            )
            if sweep:
                self._sweep_stage = 1
                return action_click_box(sweep, "click sweep")
            # No sweep button — maybe need to select stage first
            return action_wait(400, "looking for sweep button")

        # Stage 1: Click Max
        if self._sweep_stage == 1:
            max_btn = screen.find_any_text(
                ["最大", "Max", "MAX"],
                min_conf=0.6
            )
            if max_btn:
                self._sweep_stage = 2
                return action_click_box(max_btn, "click max")
            # No Max — proceed to confirm
            self._sweep_stage = 2
            return action_wait(300, "no Max, proceed to confirm")

        # Stage 2: Click confirm
        if self._sweep_stage == 2:
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "Confirm"],
                region=(0.3, 0.5, 0.7, 0.9), min_conf=0.6
            )
            if confirm:
                self._sweep_stage = 3
                return action_click_box(confirm, "confirm sweep")
            skip = screen.find_any_text(["跳過", "跳过", "Skip"], min_conf=0.7)
            if skip:
                return action_click_box(skip, "skip animation")
            return action_wait(400, "waiting for confirm")

        # Stage 3: Done — exit
        if self._sweep_stage >= 3:
            ok = screen.find_any_text(["確認", "确认", "確", "确", "OK"], min_conf=0.6)
            if ok:
                self.log(f"farming sweep done ({self._sweep_count} total)")
                self.sub_state = "exit"
                return action_click_box(ok, "dismiss result")
            self.sub_state = "exit"
            return action_wait(500, "sweep done, exiting")

        return action_wait(400, "sweep processing")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log(f"done ({self._sweep_count} sweeps, ap_empty={self._ap_empty})")
            return action_done("farming complete")
        return action_back("farming exit: back to lobby")
