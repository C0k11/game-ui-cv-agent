"""EventActivitySkill: clear event story/challenge flow before AP farming.

This skill focuses on non-sweep event content so the daily pipeline can:
1. Enter currently running event.
2. Clear story/dialogue nodes (with skip/confirm handling).
3. Clear challenge entry once (sweep or battle path).
4. Touch mission tab, then return to lobby.

It is intentionally defensive: if no event is available, it exits quickly.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill,
    ScreenState,
    action_back,
    action_click,
    action_click_box,
    action_done,
    action_wait,
)


class EventActivitySkill(BaseSkill):
    def __init__(self):
        super().__init__("EventActivity")
        self.max_ticks = 140

        self._enter_ticks: int = 0
        self._phase_ticks: int = 0

        self._story_done: bool = False
        self._story_tab_clicked: bool = False
        self._story_idle_ticks: int = 0
        self._skip_stage: int = 0  # 0=not skipping, 1=clicked MENU, 2=clicked Skip, 3=done
        self._cutscene_taps: int = 0  # how many times we tapped to advance cutscene

        self._challenge_done: bool = False
        self._challenge_tab_clicked: bool = False
        self._challenge_idle_ticks: int = 0
        self._challenge_sweep_stage: int = 0
        self._grid_auto_enabled: bool = False

        self._mission_done: bool = False
        self._mission_tab_clicked: bool = False

    def reset(self) -> None:
        super().reset()
        self._enter_ticks = 0
        self._phase_ticks = 0

        self._story_done = False
        self._story_tab_clicked = False
        self._story_idle_ticks = 0
        self._skip_stage = 0
        self._cutscene_taps = 0

        self._challenge_done = False
        self._challenge_tab_clicked = False
        self._challenge_idle_ticks = 0
        self._challenge_sweep_stage = 0
        self._grid_auto_enabled = False

        self._mission_done = False
        self._mission_tab_clicked = False

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("event activity timeout")

        # Frequent reward/result popups across story/challenge/battle.
        reward_popup = screen.find_any_text(
            [
                "獲得獎勵", "获得奖励", "獲得道具", "获得道具",
                "戰鬥結果", "战斗结果", "掃蕩完成", "扫荡完成",
                "任務完成", "任务完成", "關卡完成", "关卡完成",
            ],
            min_conf=0.6,
        )
        if reward_popup:
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "OK"],
                min_conf=0.6,
            )
            if confirm:
                return action_click_box(confirm, "dismiss event popup")
            return action_click(0.5, 0.9, "dismiss event popup fallback")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "event activity loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "story":
            return self._story(screen)
        if self.sub_state == "challenge":
            return self._challenge(screen)
        if self.sub_state == "mission":
            return self._mission(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "event activity unknown state")

    def _find_event_timer(self, screen: ScreenState, *, region) -> Optional[Any]:
        return screen.find_any_text(
            [
                "距離結束還剩", "距离结束还剩", "結束還剩", "结束还剩",
                "距離結束剩", "距离结束剩", "結束剩", "结束剩",
                "距離結束選剩", "距离结束选剩", "結束選剩", "结束选剩", "结束选剩",
                "距離结束遗剩", "距離结束道剩", "结束遗剩", "結束遗剩",
            ],
            region=region,
            min_conf=0.3,
        )

    def _find_reward_claim_timer(self, screen: ScreenState, *, region) -> Optional[Any]:
        """Detect expired event reward-claim banners (not current event)."""
        return screen.find_any_text(
            [
                "距離獎勵", "距离奖励", "獎勵領取", "奖励领取",
                "獎勵獲得結束", "奖励获得结束",
                "距離獎勵獲得結束", "距离奖励获得结束",
                "距離獎勵領取結束", "距离奖励领取结束",
                "獎勵結束", "奖励结束",
            ],
            region=region,
            min_conf=0.3,
        )

    def _is_event_page(self, screen: ScreenState) -> bool:
        event_header = screen.find_any_text(
            ["活動", "活动"],
            region=(0.0, 0.0, 0.22, 0.10),
            min_conf=0.65,
        )
        if event_header:
            return True

        tabs = screen.find_any_text(
            ["Story", "Quest", "Challenge", "劇情", "剧情", "任務", "任务"],
            region=(0.50, 0.0, 1.0, 0.24),
            min_conf=0.55,
        )
        if tabs:
            return True

        item_method = screen.find_any_text(
            ["道具獲得方法", "道具获得方法"],
            min_conf=0.55,
        )
        return item_method is not None

    def _is_event_story_screen(self, screen: ScreenState) -> bool:
        auto = screen.find_any_text(
            ["AUTO"],
            region=(0.72, 0.0, 0.92, 0.14),
            min_conf=0.65,
        )
        menu = screen.find_any_text(
            ["MENU"],
            region=(0.82, 0.0, 1.0, 0.14),
            min_conf=0.65,
        )
        skip = screen.find_any_text(["SKIP", "Skip", "跳過", "跳过"], min_conf=0.65)
        return bool((auto and menu) or skip)

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks += 1

        if self._is_event_page(screen):
            self._phase_ticks = 0
            if not self._story_done:
                self.sub_state = "story"
                self.log("event page ready -> story phase")
                return action_wait(250, "start event story phase")
            if not self._challenge_done:
                self.sub_state = "challenge"
                self.log("story done -> challenge phase")
                return action_wait(250, "start event challenge phase")
            if not self._mission_done:
                self.sub_state = "mission"
                self.log("challenge done -> mission phase")
                return action_wait(250, "start event mission phase")
            self.sub_state = "exit"
            return action_wait(200, "event phases complete")

        if self._is_event_story_screen(screen):
            if self._story_done:
                # Story marked done but still on cutscene — skip via MENU→Skip
                self.log("cutscene after story done, skipping via MENU")
                self._skip_stage = 0
            self._phase_ticks = 0
            self._story_idle_ticks = 0  # reset to prevent immediate exit
            self.sub_state = "story"
            return action_wait(250, "start event story phase from cutscene")

        current = self.detect_current_screen(screen)

        if current == "Lobby":
            # Check for OLD event reward-claim banner first — skip it
            old_reward = self._find_reward_claim_timer(screen, region=(0.55, 0.0, 1.0, 0.30))
            if old_reward:
                if self._enter_ticks <= 8:
                    return action_wait(1200, "old event reward banner, waiting for rotation")
                self.log("only old event reward banner visible, no current event")
                return action_done("reward-claim event only")

            timer = self._find_event_timer(screen, region=(0.55, 0.0, 1.0, 0.30))
            if timer:
                self.log(f"lobby event timer visible: '{timer.text}', entering event")
                return action_click(timer.cx, min(timer.cy + 0.10, 0.32), "click lobby event banner")

            if 3 <= self._enter_ticks <= 5:
                self.log("probing lobby event banner via hardcoded click")
                return action_click(0.79, 0.17, "probe lobby event banner (hardcoded)")

            campaign_btn = screen.find_any_text(
                ["任務", "任务"],
                region=(0.80, 0.70, 1.0, 0.90),
                min_conf=0.6,
            )
            if campaign_btn:
                return action_click_box(campaign_btn, "open campaign for event")
            if self._enter_ticks > 4:
                return action_click(0.95, 0.83, "open campaign area for event (hardcoded)")
            return action_wait(400, "waiting for event entry from lobby")

        if current == "Mission":
            old_reward = self._find_reward_claim_timer(screen, region=(0.0, 0.04, 0.32, 0.26))
            if old_reward:
                if self._enter_ticks <= 10:
                    return action_wait(1200, "campaign old reward banner, waiting")
                self.log("campaign: only old event reward banner")
                return action_done("campaign reward-claim event only")

            timer = self._find_event_timer(screen, region=(0.0, 0.04, 0.32, 0.26))
            if timer:
                self.log(f"campaign event timer visible: '{timer.text}', opening event")
                return action_click(timer.cx, min(timer.cy + 0.08, 0.35), "click campaign event banner")

            if self._enter_ticks > 6 and self._enter_ticks <= 10:
                self.log("probing campaign event banner via hardcoded click")
                return action_click(0.17, 0.17, "probe campaign event banner (hardcoded)")

            event_entry = screen.find_any_text(
                ["活動", "活动", "Story", "Quest", "Challenge"],
                region=(0.0, 0.04, 0.40, 0.36),
                min_conf=0.5,
            )
            if event_entry:
                return action_click_box(event_entry, "enter event from campaign")

            if self._enter_ticks > 10:
                self.log("no event entry found in campaign")
                return action_done("event not available")
            return action_wait(450, "looking for event entry in campaign")

        if current == "DailyTasks":
            return action_back("back from DailyTasks to enter event")

        if current and current not in ("Lobby", "Mission", "Event"):
            return action_back(f"back from {current} to find event")

        if self._enter_ticks > 24:
            self.log("event not found, skipping event activity")
            return action_done("event unavailable")

        return action_wait(500, "entering event")

    def _story(self, screen: ScreenState) -> Dict[str, Any]:
        self._phase_ticks += 1

        # Detect expired event — skip all phases immediately
        ended = screen.find_any_text(
            ["已结束", "已結束", "活動期已", "活动期已"],
            min_conf=0.6,
        )
        if ended:
            self.log(f"event ended: '{ended.text}', skipping story")
            self._story_done = True
            self._challenge_done = True
            self._mission_done = True
            self.sub_state = "exit"
            return action_wait(200, "event ended, skipping all phases")

        # ── Cutscene / dialog screen (AUTO + MENU visible) ──
        # BA hides Skip behind MENU. Flow: MENU → Skip → Confirm.
        # reference coordinates (1280×720): MENU(1205,34) Skip(1213,116) Confirm(766,520)
        on_cutscene = self._is_event_story_screen(screen)
        if on_cutscene:
            self._story_idle_ticks = 0  # cutscene is NOT idle

            # Direct SKIP button visible (some screens show it directly)
            skip = screen.find_any_text(
                ["SKIP", "Skip", "跳過", "跳过"],
                min_conf=0.65,
            )
            if skip:
                self._skip_stage = 2
                return action_click_box(skip, "click visible SKIP")

            # Skip confirmation dialog (center of screen)
            # OCR often truncates "確認" → "確", so match single char too.
            if self._skip_stage >= 2:
                self._cutscene_taps += 1
                confirm = screen.find_any_text(
                    ["確認", "确认", "確定", "确定", "OK", "Yes", "確", "确"],
                    region=(0.45, 0.55, 0.80, 0.85),
                    min_conf=0.55,
                )
                if confirm:
                    self._skip_stage = 3
                    self._cutscene_taps = 0
                    return action_click_box(confirm, "confirm story skip")
                # Detect "是否略過此劇情" dialog → click confirm area
                skip_prompt = screen.find_any_text(
                    ["略過", "略过", "是否略", "跳過此"],
                    region=(0.30, 0.45, 0.70, 0.70),
                    min_conf=0.55,
                )
                if skip_prompt:
                    self._skip_stage = 3
                    self._cutscene_taps = 0
                    # reference confirm pos: (766/1280, 520/720) = (0.60, 0.72)
                    return action_click(0.60, 0.72, "confirm skip (hardcoded)")
                # Fallback: after 10 ticks reset, after 5 blind-click
                if self._cutscene_taps >= 10:
                    self._cutscene_taps = 0
                    self._skip_stage = 0
                    return action_wait(200, "skip confirm timeout, retrying")
                if self._cutscene_taps >= 5:
                    return action_click(0.60, 0.72, "confirm skip (timeout fallback)")
                return action_wait(300, "waiting for skip confirm dialog")

            # Stage 0: click MENU to reveal skip option
            if self._skip_stage == 0:
                menu = screen.find_any_text(
                    ["MENU"],
                    region=(0.82, 0.0, 1.0, 0.14),
                    min_conf=0.65,
                )
                if menu:
                    self._skip_stage = 1
                    return action_click_box(menu, "click MENU to reveal skip")
                # MENU not found by OCR — try hardcoded position
                self._skip_stage = 1
                return action_click(0.94, 0.05, "click MENU (hardcoded)")

            # Stage 1: MENU was clicked, look for Skip button in dropdown
            if self._skip_stage == 1:
                skip_btn = screen.find_any_text(
                    ["SKIP", "Skip", "跳過", "跳过", "スキップ"],
                    min_conf=0.55,
                )
                if skip_btn:
                    self._skip_stage = 2
                    return action_click_box(skip_btn, "click Skip in menu")
                # Try hardcoded position (reference: 1213/1280, 116/720)
                self._skip_stage = 2
                return action_click(0.95, 0.16, "click Skip (hardcoded)")

            # Fallback: tap center to advance dialog text
            self._cutscene_taps += 1
            if self._cutscene_taps > 40:
                # Stuck on cutscene too long — try pressing back
                self._cutscene_taps = 0
                self._skip_stage = 0
                return action_back("cutscene stuck, pressing back")
            return action_click(0.5, 0.5, "advance event story text")

        # ── Not on cutscene — reset skip state ──
        self._skip_stage = 0
        self._cutscene_taps = 0

        # ── Loading / battle in progress ──
        # Very low OCR during loading screens or active battles.
        # Don't count as idle; just wait.
        if len(screen.ocr_boxes) <= 5:
            self._story_idle_ticks = 0
            return action_wait(500, "story loading/battle in progress")

        # ── Mission Info dialog (battle story node) ──
        # Screen shows "任務資訊" header with "任務開始" yellow button and
        # a disabled "掃蕩開始" (sweep). Must click "任務開始" specifically.
        # reference: "activity_task-info" → click (940/1280, 538/720) = (0.734, 0.747)
        mission_info = screen.find_any_text(
            ["任務資訊", "任务资讯"],
            min_conf=0.55,
        )
        if mission_info:
            self._story_idle_ticks = 0
            start_btn = screen.find_any_text(
                ["任務開始", "任务开始"],
                region=(0.55, 0.60, 0.90, 0.85),
                min_conf=0.55,
            )
            if start_btn:
                return action_click_box(start_btn, "click 任務開始 (story battle)")
            # reference hardcoded: (940/1280, 538/720)
            return action_click(0.734, 0.747, "click 任務開始 (hardcoded)")

        # ── Formation screen (team edit before battle) ──
        # After clicking 任務開始, lands on formation screen.
        # reference: click (1156/1280, 659/720) = (0.903, 0.915) for sortie button
        sortie = screen.find_any_text(
            ["出擊", "出击", "出撃", "出擎", "開始作戰", "开始作战",
             "戰鬥開始", "战斗开始"],
            region=(0.70, 0.75, 1.0, 0.98),
            min_conf=0.55,
        )
        if sortie:
            self._story_idle_ticks = 0
            return action_click_box(sortie, "click sortie (story battle)")

        # ── Battle result / reward confirm ──
        # After battle: fight-success-confirm, reward_acquired, etc.
        # reference: "story-fight-success-confirm" at (1117-1219, 639-687)
        next_btn = screen.find_any_text(
            ["下一步", "Next", "確認", "确认", "確定", "确定", "OK", "確", "确"],
            region=(0.25, 0.55, 0.95, 0.98),
            min_conf=0.55,
        )
        if next_btn:
            self._story_idle_ticks = 0
            return action_click_box(next_btn, "event story next/confirm")

        # Open Story tab once.
        if not self._story_tab_clicked:
            tab = screen.find_any_text(
                ["Story", "劇情", "剧情", "故事"],
                region=(0.50, 0.0, 1.0, 0.24),
                min_conf=0.5,
            )
            if tab:
                self._story_tab_clicked = True
                self._story_idle_ticks = 0
                return action_click_box(tab, "switch to event story tab")

        # Start any available story node (入場 on story list).
        # NOTE: bare "開始" removed — it wrongly matches disabled "掃蕩開始".
        start = screen.find_any_text(
            ["NEW", "前往", "進入", "进入", "入場", "入场",
             "閱讀", "阅读", "觀看", "观看"],
            region=(0.10, 0.16, 0.98, 0.96),
            min_conf=0.55,
        )
        if start:
            self._story_idle_ticks = 0
            return action_click_box(start, "start/continue event story")

        self._story_idle_ticks += 1
        if self._phase_ticks > 150 or self._story_idle_ticks > 12:
            self.log("story phase complete")
            self._story_done = True
            self._phase_ticks = 0
            self.sub_state = "enter"
            return action_wait(250, "story phase done")

        return action_wait(350, "event story scanning")

    def _challenge(self, screen: ScreenState) -> Dict[str, Any]:
        self._phase_ticks += 1

        # Grid-walk fallback controls (limited generic support).
        auto = screen.find_any_text(["AUTO", "自動", "自动"], region=(0.70, 0.0, 1.0, 0.24), min_conf=0.55)
        if auto and not self._grid_auto_enabled:
            self._grid_auto_enabled = True
            return action_click_box(auto, "enable auto on event grid")
        end_turn = screen.find_any_text(["結束回合", "结束回合", "End Turn"], region=(0.65, 0.0, 1.0, 0.28), min_conf=0.5)
        if end_turn:
            return action_click_box(end_turn, "end turn on event grid")

        # Handle cutscene/dialog during challenge (story interludes)
        skip = screen.find_any_text(["SKIP", "Skip", "跳過", "跳过"], min_conf=0.65)
        if skip:
            self._challenge_idle_ticks = 0
            return action_click_box(skip, "skip event challenge cutscene")

        if self._is_event_story_screen(screen):
            self._challenge_idle_ticks = 0
            # Click MENU to reveal Skip, then tap center as fallback
            menu = screen.find_any_text(["MENU"], region=(0.82, 0.0, 1.0, 0.14), min_conf=0.65)
            if menu:
                return action_click_box(menu, "click MENU during challenge cutscene")
            return action_click(0.94, 0.05, "click MENU (hardcoded) during challenge")

        # ── Loading / battle in progress ──
        if len(screen.ocr_boxes) <= 5:
            self._challenge_idle_ticks = 0
            return action_wait(500, "challenge loading/battle in progress")

        # ── Mission Info dialog (battle node) ──
        mission_info = screen.find_any_text(
            ["任務資訊", "任务资讯"],
            min_conf=0.55,
        )
        if mission_info:
            self._challenge_idle_ticks = 0
            start_btn = screen.find_any_text(
                ["任務開始", "任务开始"],
                region=(0.55, 0.60, 0.90, 0.85),
                min_conf=0.55,
            )
            if start_btn:
                return action_click_box(start_btn, "click 任務開始 (challenge battle)")
            return action_click(0.734, 0.747, "click 任務開始 (hardcoded)")

        # ── Formation screen ──
        sortie = screen.find_any_text(
            ["出擊", "出击", "出撃", "出擎", "開始作戰", "开始作战",
             "戰鬥開始", "战斗开始"],
            region=(0.70, 0.75, 1.0, 0.98),
            min_conf=0.55,
        )
        if sortie:
            self._challenge_idle_ticks = 0
            return action_click_box(sortie, "click sortie (challenge battle)")

        # ── Battle result confirm ──
        result_confirm = screen.find_any_text(
            ["確認", "确认", "確定", "确定", "確", "确", "OK"],
            region=(0.25, 0.50, 0.95, 0.98),
            min_conf=0.6,
        )
        if result_confirm and screen.find_any_text(["戰鬥結果", "战斗结果", "VICTORY", "DEFEAT"], min_conf=0.55):
            self._challenge_idle_ticks = 0
            return action_click_box(result_confirm, "confirm challenge battle result")

        if not self._challenge_tab_clicked:
            tab = screen.find_any_text(
                ["Challenge", "挑戰", "挑战"],
                region=(0.50, 0.0, 1.0, 0.24),
                min_conf=0.5,
            )
            if tab:
                self._challenge_tab_clicked = True
                self._challenge_idle_ticks = 0
                return action_click_box(tab, "switch to event challenge tab")

        # Sweep mini-FSM for challenge stages.
        if self._challenge_sweep_stage == 0:
            max_btn = screen.find_any_text(["MAX"], min_conf=0.7)
            if max_btn:
                self._challenge_sweep_stage = 1
                self._challenge_idle_ticks = 0
                return action_click_box(max_btn, "challenge sweep max")

            start_stage = screen.find_any_text(
                ["入場", "入场", "挑戰", "挑战"],
                region=(0.45, 0.16, 1.0, 0.95),
                min_conf=0.55,
            )
            if start_stage:
                self._challenge_idle_ticks = 0
                return action_click_box(start_stage, "enter event challenge stage")

            sweep = screen.find_any_text(["掃蕩", "扫荡"], region=(0.45, 0.35, 1.0, 0.75), min_conf=0.55)
            if sweep:
                self._challenge_idle_ticks = 0
                return action_click_box(sweep, "open event challenge sweep")

        elif self._challenge_sweep_stage == 1:
            sweep_start = screen.find_any_text(["掃蕩開始", "扫荡开始"], min_conf=0.55)
            if sweep_start:
                self._challenge_sweep_stage = 2
                self._challenge_idle_ticks = 0
                return action_click_box(sweep_start, "start challenge sweep")
            start_btn = screen.find_any_text(["開始", "开始"], region=(0.50, 0.40, 1.0, 0.70), min_conf=0.55)
            if start_btn:
                self._challenge_sweep_stage = 2
                self._challenge_idle_ticks = 0
                return action_click_box(start_btn, "start challenge sweep (fallback)")

        elif self._challenge_sweep_stage == 2:
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "OK"],
                region=(0.30, 0.30, 0.80, 0.92),
                min_conf=0.55,
            )
            if confirm:
                self._challenge_sweep_stage = 3
                self._challenge_idle_ticks = 0
                return action_click_box(confirm, "confirm challenge sweep")

        elif self._challenge_sweep_stage == 3:
            done = screen.find_any_text(
                ["掃蕩完成", "扫荡完成", "獲得獎勵", "获得奖励", "確認", "确认", "OK"],
                min_conf=0.55,
            )
            if done:
                self.log("challenge sweep complete")
                self._challenge_done = True
                self._phase_ticks = 0
                self.sub_state = "enter"
                self._challenge_sweep_stage = 0
                return action_click_box(done, "close challenge sweep result")

        self._challenge_idle_ticks += 1
        if self._phase_ticks > 70 or self._challenge_idle_ticks > 12:
            self.log("challenge phase complete")
            self._challenge_done = True
            self._phase_ticks = 0
            self.sub_state = "enter"
            self._challenge_sweep_stage = 0
            return action_wait(250, "challenge phase done")

        return action_wait(400, "event challenge scanning")

    def _mission(self, screen: ScreenState) -> Dict[str, Any]:
        self._phase_ticks += 1

        if not self._mission_tab_clicked:
            mission_tab = screen.find_any_text(
                ["Quest", "Mission", "任務", "任务"],
                region=(0.45, 0.0, 1.0, 0.24),
                min_conf=0.5,
            )
            if mission_tab:
                self._mission_tab_clicked = True
                return action_click_box(mission_tab, "switch to event mission tab")

            bottom_mission = screen.find_any_text(
                ["任務", "任务", "Quest"],
                region=(0.10, 0.78, 0.56, 0.98),
                min_conf=0.6,
            )
            if bottom_mission:
                self._mission_tab_clicked = True
                return action_click_box(bottom_mission, "open event mission from footer")

        if self._phase_ticks > 6:
            self.log("mission phase complete")
            self._mission_done = True
            self.sub_state = "enter"
            self._phase_ticks = 0
            return action_wait(200, "mission phase done")

        return action_wait(300, "event mission phase")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log(
                f"done (story={self._story_done}, challenge={self._challenge_done}, mission={self._mission_done})"
            )
            return action_done("event activity complete")
        return action_back("event activity exit: back to lobby")
