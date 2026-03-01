"""BountySkill: Handle 悬赏通缉 (Bounties) sweep.

Flow:
1. ENTER: From lobby, click 任務 sidebar → navigate to bounties
2. SWEEP: Select highest difficulty, sweep max
3. EXIT: Back to lobby
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_wait, action_back, action_done,
)


class BountySkill(BaseSkill):
    def __init__(self):
        super().__init__("Bounty")
        self.max_ticks = 40
        self._swept: bool = False

    def reset(self) -> None:
        super().reset()
        self._swept = False

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("bounty timeout")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "bounty loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "sweep":
            return self._sweep(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "bounty unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        # Check if inside bounty screen
        bounty = screen.find_any_text(
            ["懸賞通緝", "悬赏通缉", "通緝", "通缉", "Bounty"],
            min_conf=0.6
        )
        if bounty:
            self.log("inside bounty")
            self.sub_state = "sweep"
            return action_wait(300, "entered bounty")

        if screen.is_lobby():
            # Click 任務 on left sidebar
            task = screen.find_any_text(
                ["任務", "任务"],
                region=screen.LEFT_SIDE, min_conf=0.7
            )
            if task:
                return action_click_box(task, "click tasks from lobby")

        # Look for bounty tab in task menu
        bounty_tab = screen.find_any_text(
            ["懸賞", "悬赏", "通緝", "通缉"],
            min_conf=0.6
        )
        if bounty_tab:
            return action_click_box(bounty_tab, "click bounty tab")

        return action_wait(500, "entering bounty")

    def _sweep(self, screen: ScreenState) -> Dict[str, Any]:
        if self._swept:
            self.sub_state = "exit"
            return action_wait(200, "sweep done")

        # Look for sweep button
        sweep = screen.find_any_text(
            ["掃蕩", "扫荡", "Sweep"],
            min_conf=0.6
        )
        if sweep:
            self.log("clicking sweep")
            self._swept = True
            return action_click_box(sweep, "sweep bounty")

        # Look for start/enter button
        start = screen.find_any_text(
            ["開始", "开始", "進入", "进入"],
            min_conf=0.6
        )
        if start:
            return action_click_box(start, "start bounty")

        # AP exhausted
        no_ap = screen.find_any_text(
            ["AP不足", "體力不足", "体力不足"],
            min_conf=0.6
        )
        if no_ap:
            self.log("AP exhausted")
            self.sub_state = "exit"
            return action_wait(200, "no AP")

        return action_wait(500, "bounty sweep waiting")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log("done")
            return action_done("bounty complete")
        return action_back("bounty exit: back to lobby")
