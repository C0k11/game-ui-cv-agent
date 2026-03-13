"""PassRewardSkill: Claim battle pass mission and track rewards.

Adapted from reference collect_pass_reward flow:
1. ENTER: open pass menu from lobby
2. CLAIM_MISSION: switch to mission tab and claim available rewards
3. CLAIM_REWARD: return to pass reward page and claim available rewards
4. EXIT: back to lobby

Pass can be unavailable on some servers/accounts; this skill exits quickly in that case.
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


class PassRewardSkill(BaseSkill):
    def __init__(self):
        super().__init__("PassReward")
        self.max_ticks = 60
        self._enter_ticks: int = 0
        self._phase_ticks: int = 0
        self._idle_claim_ticks: int = 0
        self._claim_clicks: int = 0
        self._phase_initialized: bool = False

    def reset(self) -> None:
        super().reset()
        self._enter_ticks = 0
        self._phase_ticks = 0
        self._idle_claim_ticks = 0
        self._claim_clicks = 0
        self._phase_initialized = False

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("pass reward timeout")

        reward_popup = screen.find_any_text(
            ["獲得道具", "获得道具", "獲得獎勵", "获得奖励", "受取完了"],
            region=screen.CENTER,
            min_conf=0.6,
        )
        if reward_popup:
            return action_click(0.5, 0.9, "dismiss pass reward popup")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(700, "pass loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "claim_mission":
            return self._claim_mission(screen)
        if self.sub_state == "claim_reward":
            return self._claim_reward(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "pass reward unknown state")

    def _is_pass_screen(self, screen: ScreenState) -> bool:
        header = screen.find_any_text(
            ["通行證", "通行证", "PASS", "Pass"],
            region=(0.0, 0.0, 0.45, 0.16),
            min_conf=0.55,
        )
        if header:
            return True

        season = screen.find_any_text(
            ["賽季結束", "赛季结束", "賽季", "赛季"],
            region=(0.25, 0.12, 0.85, 0.35),
            min_conf=0.55,
        )
        if season:
            return True

        claim_area = screen.find_any_text(
            ["一鍵領取", "一键领取", "全部領取", "全部领取", "受取", "Claim"],
            region=(0.55, 0.55, 1.0, 0.95),
            min_conf=0.55,
        )
        return claim_area is not None

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks += 1
        current = self.detect_current_screen(screen)

        if current == "Pass" or self._is_pass_screen(screen):
            self.log("inside pass")
            self.sub_state = "claim_mission"
            self._phase_ticks = 0
            self._idle_claim_ticks = 0
            self._phase_initialized = False
            return action_wait(300, "entered pass")

        if current == "Lobby":
            pass_btn = screen.find_any_text(
                ["PASS", "Pass", "通行證", "通行证"],
                min_conf=0.55,
            )
            if pass_btn:
                return action_click_box(pass_btn, "open pass menu")

            # reference JP fallback x/y: (341~451, 630~662) on 1280x720.
            if self._enter_ticks == 4:
                return action_click(0.31, 0.90, "open pass menu (hardcoded)")

            if self._enter_ticks > 8:
                self.log("pass entry not found, skipping")
                self.sub_state = "exit"
                return action_wait(200, "pass unavailable")

            return action_wait(350, "looking for pass entry")

        if current and current not in ("Pass", "Lobby"):
            return action_back(f"back from {current}")

        if self._enter_ticks > 12:
            self.sub_state = "exit"
            return action_wait(200, "pass enter timeout")

        return action_wait(400, "entering pass")

    def _claim_current_tab(self, screen: ScreenState, phase_name: str) -> Optional[Dict[str, Any]]:
        claim = screen.find_any_text(
            ["一鍵領取", "一键领取", "全部領取", "全部领取", "Claim All", "受取"],
            region=(0.55, 0.55, 1.0, 0.95),
            min_conf=0.55,
        )
        if claim:
            self._claim_clicks += 1
            self._idle_claim_ticks = 0
            self.log(f"{phase_name}: claim all ({self._claim_clicks})")
            return action_click_box(claim, f"pass {phase_name}: claim all")

        single = screen.find_any_text(
            ["領取", "领取", "Claim"],
            region=(0.55, 0.15, 1.0, 0.95),
            min_conf=0.68,
        )
        if single:
            self._claim_clicks += 1
            self._idle_claim_ticks = 0
            self.log(f"{phase_name}: claim single ({self._claim_clicks})")
            return action_click_box(single, f"pass {phase_name}: claim single")

        self._idle_claim_ticks += 1
        if self._idle_claim_ticks > 4:
            return None
        return action_wait(350, f"{phase_name}: waiting claim buttons")

    def _claim_mission(self, screen: ScreenState) -> Dict[str, Any]:
        self._phase_ticks += 1

        if not self._phase_initialized:
            self._phase_initialized = True
            self._idle_claim_ticks = 0
            mission_tab = screen.find_any_text(
                ["任務", "任务", "Mission"],
                region=(0.18, 0.80, 0.58, 0.98),
                min_conf=0.5,
            )
            if mission_tab:
                return action_click_box(mission_tab, "switch to pass mission tab")
            return action_click(0.30, 0.90, "switch to pass mission tab (hardcoded)")

        claim_action = self._claim_current_tab(screen, "mission")
        if claim_action:
            return claim_action

        if self._phase_ticks > 16:
            self.log("mission claim phase timeout, moving to reward tab")

        self.sub_state = "claim_reward"
        self._phase_ticks = 0
        self._idle_claim_ticks = 0
        self._phase_initialized = False
        return action_wait(250, "mission claims complete")

    def _claim_reward(self, screen: ScreenState) -> Dict[str, Any]:
        self._phase_ticks += 1

        if not self._phase_initialized:
            self._phase_initialized = True
            self._idle_claim_ticks = 0

            reward_tab = screen.find_any_text(
                ["獎勵", "奖励", "Reward"],
                region=(0.10, 0.80, 0.60, 0.98),
                min_conf=0.5,
            )
            if reward_tab:
                return action_click_box(reward_tab, "switch to pass reward tab")

            return action_back("return to pass reward page")

        claim_action = self._claim_current_tab(screen, "reward")
        if claim_action:
            return claim_action

        if self._phase_ticks > 16:
            self.log("reward claim phase timeout")

        self.sub_state = "exit"
        return action_wait(250, "pass reward claims complete")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log(f"done ({self._claim_clicks} claims)")
            return action_done("pass reward complete")
        return action_back("pass reward exit: back to lobby")
