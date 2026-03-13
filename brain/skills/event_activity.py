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
            self._phase_ticks = 0
            self.sub_state = "story"
            self.log("event story/cutscene screen detected -> story phase")
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

        # Handle cutscene/dialog first.
        skip = screen.find_any_text(["SKIP", "Skip", "跳過", "跳过"], min_conf=0.7)
        if skip:
            self._story_idle_ticks = 0
            return action_click_box(skip, "skip event story cutscene")

        next_btn = screen.find_any_text(
            ["下一步", "Next", "確認", "确认", "確定", "确定", "OK"],
            region=(0.25, 0.55, 0.80, 0.95),
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

        # Start any available story node.
        start = screen.find_any_text(
            ["NEW", "開始", "开始", "前往", "進入", "进入", "入場", "入场", "閱讀", "阅读", "觀看", "观看"],
            region=(0.10, 0.16, 0.98, 0.96),
            min_conf=0.55,
        )
        if start:
            self._story_idle_ticks = 0
            return action_click_box(start, "start/continue event story")

        # Some stories include a fight entry.
        sortie = screen.find_any_text(
            ["出擊", "出击", "開始作戰", "开始作战", "戰鬥開始", "战斗开始"],
            region=(0.50, 0.55, 1.0, 0.95),
            min_conf=0.55,
        )
        if sortie:
            self._story_idle_ticks = 0
            return action_click_box(sortie, "start event story battle")

        self._story_idle_ticks += 1
        if self._phase_ticks > 60 or self._story_idle_ticks > 8:
            self.log("story phase complete")
            self._story_done = True
            self._phase_ticks = 0
            self.sub_state = "enter"
            return action_wait(250, "story phase done")

        # Keep tapping center if still inside story dialogue screen.
        if not self._is_event_page(screen):
            return action_click(0.5, 0.5, "advance event story text")

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

        skip = screen.find_any_text(["SKIP", "Skip", "跳過", "跳过"], min_conf=0.7)
        if skip:
            self._challenge_idle_ticks = 0
            return action_click_box(skip, "skip event challenge battle")

        result_confirm = screen.find_any_text(
            ["確認", "确认", "確定", "确定", "確", "确", "OK"],
            region=(0.25, 0.50, 0.80, 0.95),
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
                ["入場", "入场", "挑戰", "挑战", "開始", "开始"],
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

            fight = screen.find_any_text(
                ["出擊", "出击", "開始作戰", "开始作战", "戰鬥開始", "战斗开始"],
                region=(0.50, 0.55, 1.0, 0.95),
                min_conf=0.55,
            )
            if fight:
                self._challenge_idle_ticks = 0
                return action_click_box(fight, "start event challenge fight")

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
