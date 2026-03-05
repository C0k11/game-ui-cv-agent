"""DailyTasksSkill: Claim daily quest rewards and activity chests.

Runs AFTER mail to collect daily task completion rewards (AP, gems, items)
and activity milestone chests that accumulated during the pipeline run.

Flow:
1. ENTER: From lobby, navigate to daily tasks / 工作 / 每日任務
2. CLAIM_ALL: Click 一鍵領取 to claim all completed tasks
3. CLAIM_CHESTS: Click activity milestone chests (top bar) if available
4. EXIT: Back to lobby

Key patterns:
- Navigation: "工作" / "任務" / "Tasks" / "每日" / "Daily"
- Claim all: "一鍵領取" / "全部領取" / "Claim All"
- Activity chests: YOLO or OCR detection of chest icons at top
- Reward popup: "獲得道具" / "獲得獎勵" — tap to dismiss
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_click_yolo,
    action_wait, action_back, action_done,
)


class DailyTasksSkill(BaseSkill):
    def __init__(self):
        super().__init__("DailyTasks")
        self.max_ticks = 50
        self._claim_attempts: int = 0
        self._claimed_count: int = 0
        self._chests_claimed: int = 0
        self._chest_phase: bool = False  # True = looking for chests

    def reset(self) -> None:
        super().reset()
        self._claim_attempts = 0
        self._claimed_count = 0
        self._chests_claimed = 0
        self._chest_phase = False

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("daily_tasks timeout")

        # Reward result popup — tap to dismiss
        reward = screen.find_any_text(
            ["獲得道具", "获得道具", "獲得獎勵", "获得奖励"],
            region=screen.CENTER, min_conf=0.6
        )
        if reward:
            return action_click(0.5, 0.9, "dismiss reward popup")

        # Activity chest reward popup (活躍度寶箱 / 活跃度宝箱)
        chest_reward = screen.find_any_text(
            ["活躍度", "活跃度", "Activity"],
            region=screen.CENTER, min_conf=0.6
        )
        if chest_reward:
            confirm = screen.find_any_text(
                ["確認", "确认", "確", "确", "OK"],
                region=screen.CENTER, min_conf=0.6
            )
            if confirm:
                self._chests_claimed += 1
                return action_click_box(confirm, "confirm chest reward")
            return action_click(0.5, 0.9, "dismiss chest reward")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "daily_tasks loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "claim_all":
            return self._claim_all(screen)
        if self.sub_state == "claim_chests":
            return self._claim_chests(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "daily_tasks unknown state")

    def _is_tasks_screen(self, screen: ScreenState) -> bool:
        """Detect if we're on the daily tasks / work screen."""
        # Header region check
        if screen.find_any_text(
            ["工作", "每日任務", "每日任务", "Daily", "任務總覽", "任务总览"],
            region=(0.0, 0.0, 0.40, 0.12), min_conf=0.5
        ):
            return True
        return False

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        current = self.detect_current_screen(screen)

        if self._is_tasks_screen(screen) or current == "DailyTasks":
            self.log("inside daily tasks")
            self.sub_state = "claim_all"
            return action_wait(500, "entered daily tasks")

        if current == "Lobby":
            # Try to find tasks/work button on lobby
            # Usually accessed via a sidebar icon or bottom nav
            tasks_btn = screen.find_any_text(
                ["工作", "每日", "Daily"],
                min_conf=0.6
            )
            if tasks_btn:
                return action_click_box(tasks_btn, "click tasks from lobby")

            # Try YOLO task icon
            task_yolo = screen.find_yolo_one("任务按钮", min_conf=0.3)
            if task_yolo:
                return action_click_yolo(task_yolo, "click task icon via YOLO")

            # Fallback: tasks button is often in bottom-left area or sidebar
            # Try nav bar
            nav = self._nav_to(screen, ["工作", "任務", "任务"])
            if nav:
                return nav

            # Last resort: some servers have it accessible via mission menu
            mission = screen.find_any_text(
                ["任務", "任务"],
                region=screen.LEFT_SIDE, min_conf=0.7
            )
            if mission:
                return action_click_box(mission, "click mission sidebar")

            return action_wait(300, "looking for tasks button")

        if current and current != "Lobby":
            return action_back(f"back from {current}")

        return action_wait(500, "entering daily tasks")

    def _claim_all(self, screen: ScreenState) -> Dict[str, Any]:
        """Click claim-all button repeatedly until no more rewards."""
        self._claim_attempts += 1

        if self._claim_attempts > 15:
            self.log(f"claim loop done ({self._claimed_count} claimed)")
            self.sub_state = "claim_chests"
            self._chest_phase = True
            return action_wait(300, "claim attempts exhausted, checking chests")

        # Look for claim-all button
        claim = screen.find_any_text(
            ["一鍵領取", "一键领取", "全部領取", "全部领取", "Claim All"],
            min_conf=0.6
        )
        if claim:
            self.log(f"claiming all tasks (attempt {self._claim_attempts})")
            self._claimed_count += 1
            return action_click_box(claim, "claim all tasks")

        # Individual claim buttons
        single = screen.find_any_text(
            ["領取", "领取", "Claim"],
            min_conf=0.7,
            region=(0.6, 0.1, 1.0, 0.9)
        )
        if single:
            self.log("claiming individual task reward")
            self._claimed_count += 1
            return action_click_box(single, "claim single task")

        # No more claim buttons — move to chests
        self.log(f"no more tasks to claim ({self._claimed_count} claimed)")
        self.sub_state = "claim_chests"
        self._chest_phase = True
        return action_wait(300, "tasks done, checking chests")

    def _claim_chests(self, screen: ScreenState) -> Dict[str, Any]:
        """Click activity milestone chests at the top of the tasks screen.

        These are usually small chest icons along an activity progress bar.
        They become claimable (glow/animate) when activity points reach thresholds.
        """
        # Look for chest-related YOLO detections
        chest = screen.find_yolo_one("宝箱", min_conf=0.3)
        if chest:
            self.log(f"clicking activity chest at ({chest.cx:.2f},{chest.cy:.2f})")
            self._chests_claimed += 1
            return action_click_yolo(chest, "claim activity chest")

        # OCR fallback: look for activity progress text and click chest areas
        # Activity bar is usually at top of tasks screen (y: 0.10-0.20)
        activity = screen.find_any_text(
            ["活躍度", "活跃度", "Activity"],
            region=(0.0, 0.05, 1.0, 0.25), min_conf=0.5
        )
        if activity and self._chests_claimed == 0:
            # Click along the activity bar to try to hit chests
            # Chests are typically at 25%, 50%, 75%, 100% of bar width
            positions = [(0.35, 0.15), (0.50, 0.15), (0.65, 0.15), (0.80, 0.15)]
            idx = self._chests_claimed % len(positions)
            self._chests_claimed += 1
            px, py = positions[idx]
            self.log(f"clicking activity chest area ({px},{py})")
            return action_click(px, py, f"click chest position {idx}")

        # Done with chests
        self.log(f"chests done ({self._chests_claimed} claimed)")
        self.sub_state = "exit"
        return action_wait(300, "chests complete")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log(f"done ({self._claimed_count} tasks, {self._chests_claimed} chests)")
            return action_done("daily_tasks complete")
        return action_back("daily_tasks exit: back to lobby")
