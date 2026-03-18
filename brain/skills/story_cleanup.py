"""StoryCleanupSkill: one-click cleanup for main/group/mini story content.

Goal:
- Enter story hub from mission/campaign.
- Sweep available story nodes by clicking NEW/Start entries.
- Handle dialogue/cutscene skip flow (MENU → skip → confirm).
- Handle formation/battle screens.
- Cover Main Story, Group Story, and Mini Story in one pass.

reference-based architecture:
- clear_current_plot: MENU → skip-plot-button → skip-plot-notice → reward
- push_episode: check episode → click unclear node → clear → repeat
- Detect "ALL CLEAR" / no-actionable-node as section completion.
- Fast idle detection to avoid hub loops.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from brain.skills.base import (
    BaseSkill,
    ScreenState,
    action_back,
    action_click,
    action_click_box,
    action_done,
    action_swipe,
    action_wait,
)


class StoryCleanupSkill(BaseSkill):
    _SECTIONS: List[Tuple[str, List[str]]] = [
        ("main", ["主線", "主线", "Main", "主線劇情", "主线剧情"]),
        ("group", ["小組", "小组", "Group"]),
        ("mini", ["迷你", "Mini"]),
    ]

    def __init__(self):
        super().__init__("StoryCleanup")
        self.max_ticks = 160

        self._enter_ticks: int = 0
        self._section_idx: int = 0
        self._section_ticks: int = 0
        self._idle_ticks: int = 0
        self._page_swipes: int = 0
        self._section_actions: int = 0

        self._episodes_started: int = 0
        self._dialog_taps: int = 0

        self._in_dialogue: bool = False
        self._skip_armed: int = 0
        self._entered_episode_list: bool = False
        self._hub_consecutive: int = 0

    def reset(self) -> None:
        super().reset()
        self._enter_ticks = 0
        self._section_idx = 0
        self._section_ticks = 0
        self._idle_ticks = 0
        self._page_swipes = 0
        self._section_actions = 0
        self._episodes_started = 0
        self._dialog_taps = 0
        self._in_dialogue = False
        self._skip_armed = 0
        self._entered_episode_list = False
        self._hub_consecutive = 0

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("story cleanup timeout")

        # Reward/result popup — dismiss globally
        reward_popup = screen.find_any_text(
            [
                "獲得獎勵", "获得奖励", "獲得道具", "获得道具",
                "戰鬥結果", "战斗结果", "掃蕩完成", "扫荡完成",
                "任務完成", "任务完成", "關卡完成", "关卡完成",
            ],
            min_conf=0.6,
        )
        if reward_popup:
            self._in_dialogue = False
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "OK"],
                min_conf=0.6,
            )
            if confirm:
                return action_click_box(confirm, "dismiss story reward/result popup")
            return action_click(0.5, 0.9, "dismiss story reward/result popup fallback")

        # Skip-story confirmation dialog ("是否略過此劇情？")
        skip_confirm = screen.find_any_text(
            ["是否略過", "是否略过", "略過此", "略过此", "略此情"],
            region=screen.CENTER, min_conf=0.55
        )
        if skip_confirm:
            self._skip_armed = 0
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确"],
                region=(0.50, 0.55, 0.80, 0.85), min_conf=0.55
            )
            if confirm:
                return action_click_box(confirm, "confirm story skip")
            return action_click(0.61, 0.73, "confirm story skip (fallback)")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "story cleanup loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "section":
            return self._section(screen)
        if self.sub_state == "cleanup":
            return self._cleanup(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "story cleanup unknown state")

    def _is_story_hub(self, screen: ScreenState) -> bool:
        header = screen.find_any_text(
            ["劇情", "剧情", "Story"],
            region=(0.0, 0.0, 0.35, 0.16),
            min_conf=0.5,
        )
        if header:
            return True

        # Must have section tabs visible to be the hub
        section_marker = screen.find_any_text(
            ["主線", "主线", "Main", "小組", "小组", "Group", "迷你", "Mini"],
            region=(0.02, 0.08, 0.98, 0.30),
            min_conf=0.5,
        )
        if section_marker:
            return True

        return False

    def _is_story_dialogue(self, screen: ScreenState) -> bool:
        """Detect story dialogue/cutscene (SKIP/AUTO/MENU in top-right area)."""
        auto_or_menu = screen.find_any_text(
            ["AUTO", "自動", "自动", "MENU"],
            region=(0.70, 0.0, 1.0, 0.14),
            min_conf=0.55,
        )
        if auto_or_menu:
            return True
        skip = screen.find_any_text(
            ["SKIP", "Skip"],
            region=(0.60, 0.0, 1.0, 0.14),
            min_conf=0.60,
        )
        return skip is not None

    def _is_episode_list(self, screen: ScreenState) -> bool:
        """Detect an episode/chapter list within a story section."""
        return screen.find_any_text(
            ["Episode", "章", "話", "话", "Vol", "EP"],
            region=(0.0, 0.08, 1.0, 0.42),
            min_conf=0.5,
        ) is not None

    def _is_formation_screen(self, screen: ScreenState) -> bool:
        """Detect battle formation screen."""
        return screen.find_any_text(
            ["攻擊編制", "攻击编制", "出擊", "出击", "部隊編成", "编队"],
            region=(0.30, 0.40, 1.0, 0.95),
            min_conf=0.5,
        ) is not None

    def _finish_section(self, reason: str) -> Dict[str, Any]:
        if self._section_idx < len(self._SECTIONS):
            sec = self._SECTIONS[self._section_idx][0]
        else:
            sec = "unknown"
        self.log(f"section '{sec}' done ({reason}, actions={self._section_actions})")
        self._section_idx += 1
        self._section_ticks = 0
        self._idle_ticks = 0
        self._page_swipes = 0
        self._section_actions = 0
        self._in_dialogue = False
        self._skip_armed = 0
        self._entered_episode_list = False
        self._hub_consecutive = 0
        if self._section_idx >= len(self._SECTIONS):
            self.sub_state = "exit"
            return action_wait(250, "all story sections complete")
        self.sub_state = "section"
        return action_wait(250, "switching to next story section")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks += 1

        if self._is_story_hub(screen):
            self.log("story hub detected")
            self.sub_state = "section"
            self._section_idx = 0
            self._section_ticks = 0
            self._idle_ticks = 0
            self._page_swipes = 0
            self._section_actions = 0
            return action_wait(250, "entered story hub")

        current = self.detect_current_screen(screen)

        if current == "Lobby":
            campaign_btn = screen.find_any_text(
                ["任務", "任务"],
                region=(0.80, 0.70, 1.0, 0.90),
                min_conf=0.6,
            )
            if campaign_btn:
                return action_click_box(campaign_btn, "open campaign for story cleanup")
            return action_click(0.95, 0.83, "open campaign area (hardcoded)")

        if current == "Mission":
            story_entry = screen.find_any_text(
                ["劇情", "剧情", "Story"],
                min_conf=0.5,
            )
            if story_entry:
                return action_click_box(story_entry, "enter story hub")

            if self._enter_ticks > 6:
                return action_click(0.26, 0.71, "enter story hub (hardcoded)")
            return action_wait(350, "looking for story entry")

        if current == "DailyTasks":
            return action_back("back from DailyTasks to story")

        if current and current not in ("Lobby", "Mission"):
            return action_back(f"back from {current} to story")

        if self._enter_ticks > 28:
            self.log("story entry unavailable, skipping")
            return action_done("story cleanup unavailable")

        return action_wait(450, "entering story hub")

    def _section(self, screen: ScreenState) -> Dict[str, Any]:
        if self._section_idx >= len(self._SECTIONS):
            self.sub_state = "exit"
            return action_wait(200, "story sections completed")

        # If we're in a dialogue somehow, go to cleanup
        if self._is_story_dialogue(screen):
            self.sub_state = "cleanup"
            self._in_dialogue = True
            return action_wait(200, "in dialogue, switching to cleanup")

        if not self._is_story_hub(screen):
            # Could be an episode list or transition — wait briefly then reacquire
            self._idle_ticks += 1
            if self._idle_ticks > 6:
                self.sub_state = "enter"
                self._enter_ticks = 0
                self._idle_ticks = 0
            return action_wait(400, "waiting for story hub")

        sec_name, aliases = self._SECTIONS[self._section_idx]
        hit = screen.find_any_text(
            aliases,
            region=(0.02, 0.08, 0.98, 0.30),
            min_conf=0.5,
        )

        self.sub_state = "cleanup"
        self._section_ticks = 0
        self._idle_ticks = 0
        self._page_swipes = 0
        self._section_actions = 0
        self._in_dialogue = False

        if hit:
            self.log(f"switching to section '{sec_name}'")
            return action_click_box(hit, f"open story section '{sec_name}'")

        self.log(f"section tab '{sec_name}' not found, trying current page")
        return action_wait(250, f"assume section '{sec_name}' already active")

    def _cleanup(self, screen: ScreenState) -> Dict[str, Any]:
        self._section_ticks += 1

        # ── Story dialogue/cutscene handling ──
        # reference: plot_menu → skip-plot-button → skip-plot-notice → reward
        menu_btn = screen.find_any_text(
            ["MENU"],
            region=(0.88, 0.0, 1.0, 0.12), min_conf=0.6,
        )
        skip = screen.find_any_text(["SKIP", "Skip"], min_conf=0.65)

        if skip:
            self._idle_ticks = 0
            self._hub_consecutive = 0
            self._in_dialogue = True
            self._section_actions += 1
            return action_click_box(skip, "skip story dialogue")

        if self._skip_armed > 0 and menu_btn:
            # We opened MENU and expect skip prompt next
            self._skip_armed -= 1
            return action_click(0.61, 0.73, "confirm skip (armed)")

        if menu_btn:
            self._idle_ticks = 0
            self._hub_consecutive = 0
            self._in_dialogue = True
            self._skip_armed = 2
            return action_click_box(menu_btn, "open story MENU to skip")

        if self._is_story_dialogue(screen):
            self._idle_ticks = 0
            self._hub_consecutive = 0
            self._in_dialogue = True
            self._dialog_taps += 1
            # Try to find any skip-like button
            skip_plot = screen.find_any_text(
                ["跳過劇情", "跳过剧情", "Skip Story", "跳過", "跳过"],
                region=(0.80, 0.08, 1.0, 0.25), min_conf=0.5,
            )
            if skip_plot:
                return action_click_box(skip_plot, "skip plot from menu")
            # Auto-advance dialogue by tapping
            return action_click(0.5, 0.5, "advance story dialogue")

        self._skip_armed = 0
        self._in_dialogue = False

        # ── Formation/battle screen ──
        if self._is_formation_screen(screen):
            self._idle_ticks = 0
            self._hub_consecutive = 0
            self._section_actions += 1
            fight_btn = screen.find_any_text(
                ["出擊", "出击", "開始作戰", "开始作战", "戰鬥開始", "战斗开始"],
                region=(0.45, 0.55, 1.0, 0.95), min_conf=0.5,
            )
            if fight_btn:
                return action_click_box(fight_btn, "start story battle")
            return action_click(0.90, 0.91, "start battle (hardcoded)")

        # ── Hub loop detection ──
        on_hub = self._is_story_hub(screen)
        if on_hub:
            self._hub_consecutive += 1
        else:
            self._hub_consecutive = 0

        # If we keep seeing the hub without entering an episode, finish quickly
        if self._hub_consecutive >= 4 and self._section_actions == 0:
            return self._finish_section("hub loop without any actions")

        # ── Episode list detection ──
        on_episode_list = self._is_episode_list(screen)
        if on_episode_list:
            self._entered_episode_list = True
            self._hub_consecutive = 0

        # ── Episode node entry ──
        # Only look for actionable text when NOT on hub (to prevent hub text false matches)
        if not on_hub or self._entered_episode_list:
            start = screen.find_any_text(
                ["NEW", "開始", "开始", "進入", "进入", "前往", "閱讀", "阅读",
                 "觀看", "观看", "繼續", "继续", "入場", "入场"],
                region=(0.08, 0.30, 0.98, 0.95),
                min_conf=0.55,
            )
            if start:
                self._episodes_started += 1
                self._idle_ticks = 0
                self._hub_consecutive = 0
                self._section_actions += 1
                return action_click_box(start, "start/continue story node")

        # ── Confirm/next button in non-dialogue context ──
        next_btn = screen.find_any_text(
            ["下一步", "Next", "NEXT"],
            region=(0.20, 0.52, 0.86, 0.95),
            min_conf=0.55,
        )
        if next_btn:
            self._idle_ticks = 0
            self._hub_consecutive = 0
            self._section_actions += 1
            return action_click_box(next_btn, "story dialogue next")

        # ── Clear markers ── section done if all cleared
        clear_marker = screen.find_any_text(
            ["ALL CLEAR", "已完成", "已讀", "已读", "全部完成", "已通關", "已通关"],
            region=(0.08, 0.10, 0.98, 0.92),
            min_conf=0.55,
        )
        if clear_marker and self._section_ticks > 2:
            return self._finish_section(f"clear marker '{clear_marker.text}'")

        # ── Page navigation ──
        next_page = screen.find_any_text(
            ["下一頁", "下一页", ">>"],
            region=(0.88, 0.18, 1.0, 0.88),
            min_conf=0.55,
        )
        if next_page and self._page_swipes < 3:
            self._page_swipes += 1
            self._idle_ticks = 0
            return action_click_box(next_page, "story next page")

        # ── Idle tracking ──
        self._idle_ticks += 1

        # If on hub with no actionable content, finish section quickly
        if on_hub and self._idle_ticks > 3:
            return self._finish_section("no actionable content on hub")

        # If on episode list, try scrolling once
        if on_episode_list and self._idle_ticks in (2, 5) and self._page_swipes < 3:
            self._page_swipes += 1
            return action_swipe(0.88, 0.52, 0.30, 0.52, 350, "story page swipe")

        # Absolute timeout for this section
        if self._section_ticks > 30 or self._idle_ticks > 6:
            return self._finish_section("idle/timeout threshold")

        return action_wait(350, "story cleanup scanning")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log(
                f"done (episodes={self._episodes_started}, dialogue_taps={self._dialog_taps})"
            )
            return action_done("story cleanup complete")
        return action_back("story cleanup exit: back to lobby")
