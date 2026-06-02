"""DailyMissionSkill — claim 每日任务 rewards (pure-YOLO rewrite of daily_tasks).

Verified flow (interactive probe 2026-06-01, data/_daily_reward_probe_log.md).
★ CRITICAL distinction the probe corrected:
  - lobby LEFT 每日领奖 (NAV_DAILY_REWARD, ~0.045,0.358) → THIS 任務 reward screen.
  - lobby RIGHT 任务大厅入口 (NAV_TASKS) → the BATTLE hub (bounty/arena/etc).
  We enter via NAV_DAILY_REWARD, never NAV_TASKS.

The 全體 tab aggregates every task category, so claiming there covers
每天/每週/成就/挑戰 — no tab switching needed.

★ MUST RUN LAST: the other dailies (cafe/craft/bounty/arena/...) must complete
first to unlock these mission rewards. Placed at the end of DailyRoutine.

⛔ NEVER touch 立即前往 (the "go" button on UNFINISHED tasks — events/challenges/
campaign). We only ever click the exact 全部领取_黄 / 领取_黄 claim cls, so an
unfinished task's go-button is never clicked.

State machine
-------------
enter         lobby → NAV_DAILY_REWARD → 任務 screen (全體 tab).
claim_all     全部领取_黄 (CLAIM_ALL_YELLOW) → reward dismiss. Loop until it
              greys (全部领取_灰色 / 一键领取灰色).
claim_single  remaining 领取_黄 (CLAIM_YELLOW, e.g. 完成每日任務8次 meta) → claim
              each → reward dismiss. Done when no 黄 claim cls remain.
exit          BTN_HOME / BTN_BACK → lobby → done.

Detectors: base "ui" only.
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click_box, action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC

_CLS_CONF = 0.30
# 每日领奖 entry + its red dot live in the lobby's left panel.
_ENTRY_DOT_REGION = (0.00, 0.30, 0.13, 0.40)

_ENTER_MAX = 20
_CLAIM_ALL_MAX = 26
_CLAIM_SINGLE_MAX = 20
_EXIT_MAX = 14


class DailyMissionSkill(BaseSkill):
    # Page markers: any claim/done cls of the 任務 screen.
    _PAGE_CLS = [
        UC.CLAIM_ALL_YELLOW, UC.CLAIM_ALL_GREY, UC.CLAIM_ONEKEY_GREY,
        UC.CLAIM_YELLOW, UC.CLAIM_GREY, UC.NODE_DONE,
    ]

    def should_run(self, screen: ScreenState) -> bool:
        # Run when a red dot sits by the 每日领奖 entry. Entry not visible ⇒
        # defer (True). (Rewards unlock as other dailies finish — run it last.)
        entry = self.find_cls(screen, UC.NAV_DAILY_REWARD, conf=0.40)
        if entry is None:
            # Fall back to a region scan (the entry cls is weak, 16f).
            if self.dot_in_region(screen, _ENTRY_DOT_REGION, dot_classes=(UC.DOT_RED,)):
                return True
            return self.dot_on_entry(screen, [UC.NAV_DAILY_REWARD])
        region = (entry.x1 - 0.02, entry.y1 - 0.05, entry.x2 + 0.05, entry.y2 + 0.02)
        return self.dot_in_region(screen, region, dot_classes=(UC.DOT_RED,))

    def __init__(self):
        super().__init__("DailyMission")
        self.max_ticks = 80
        self._init_state()

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._entered: bool = False
        self._all_claims: int = 0
        self._single_claims: int = 0

    def reset(self) -> None:
        super().reset()
        self._init_state()

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0

    def _on_page(self, screen: ScreenState) -> bool:
        if self.detect_screen_yolo(screen) == "Lobby":
            self._entered = False
            return False
        if self.find_cls(screen, self._PAGE_CLS, conf=_CLS_CONF) is not None:
            return True
        # No claim cls (all done) but we clicked our entry and we're off-lobby.
        return self._entered

    # ── tick ────────────────────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log(f"timeout (all={self._all_claims}, single={self._single_claims})")
            return action_done("daily_mission timeout")

        # Global: reward reveal → dismiss via continue / header (NEVER center).
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
        if cont is not None:
            return action_click_box(cont, "dismiss reward (continue)")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if got is not None:
            return action_click_box(got, "dismiss reward (header)")

        if screen.is_loading():
            return action_wait(700, "daily_mission loading")

        if self.sub_state == "":
            self._goto("enter")

        handler = {
            "enter": self._enter,
            "claim_all": self._claim_all,
            "claim_single": self._claim_single,
            "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "daily_mission unknown state")
        return handler(screen)

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._on_page(screen):
            self.log("inside 每日领奖 (任務) → claim_all")
            self._goto("claim_all")
            return action_wait(400, "entered daily mission")

        if screen.is_lobby():
            act = self.click_cls(screen, UC.NAV_DAILY_REWARD, "open 每日领奖", conf=_CLS_CONF)
            if act is not None:
                self._entered = True
                return act
            self.log("on lobby but no 每日领奖 cls — YOLO gap; waiting")
            return action_wait(400, "waiting for 每日领奖 cls")

        if self._phase_ticks > _ENTER_MAX:
            return action_done("could not reach daily mission")
        if len(screen.yolo_boxes or []) < 2:
            return action_wait(700, "no UI, likely loading")
        return action_back("daily_mission: recover toward lobby")

    def _claim_all(self, screen: ScreenState) -> Dict[str, Any]:
        if not self._on_page(screen):
            if screen.is_lobby():
                self._goto("enter")
                return action_wait(300, "claim_all: back on lobby")
            if self._phase_ticks > _CLAIM_ALL_MAX:
                self._goto("exit")
                return action_wait(300, "claim_all lost page → exit")
            return action_wait(400, "waiting for 任務 UI")

        # 全部领取_黄 → one tap claims a batch (rewards come in 2-3 popups).
        claim_all = self.find_cls(screen, UC.CLAIM_ALL_YELLOW, conf=_CLS_CONF)
        if claim_all is not None:
            self._all_claims += 1
            self.log(f"全部领取_黄 (#{self._all_claims})")
            return action_click_box(claim_all, "claim all daily-mission rewards")

        # Claim-all greyed → batch rewards done; sweep remaining single claims.
        if self.find_cls(screen, [UC.CLAIM_ALL_GREY, UC.CLAIM_ONEKEY_GREY], conf=_CLS_CONF) is not None \
                or self._phase_ticks > _CLAIM_ALL_MAX:
            self.log("全部领取 greyed (or budget) → claim_single")
            self._goto("claim_single")
            return action_wait(250, "claim-all done → single")
        return action_wait(400, "waiting for 全部领取_黄")

    def _claim_single(self, screen: ScreenState) -> Dict[str, Any]:
        if not self._on_page(screen):
            if screen.is_lobby():
                self.log("back on lobby → done")
                return action_done(f"daily_mission complete (all={self._all_claims}, single={self._single_claims})")
            self._goto("exit")
            return action_wait(300, "single lost page → exit")

        # Remaining individual 领取_黄 (meta tasks like 完成每日任務8次). NEVER
        # touch 立即前往 — we only match the exact claim cls.
        single = self.find_cls(screen, UC.CLAIM_YELLOW, conf=_CLS_CONF)
        if single is not None:
            self._single_claims += 1
            self.log(f"领取_黄 single (#{self._single_claims})")
            return action_click_box(single, "claim single daily-mission reward")

        # No 黄 claim cls left → all done.
        if self._phase_ticks > _CLAIM_SINGLE_MAX or \
                self.find_cls(screen, UC.CLAIM_ALL_YELLOW, conf=_CLS_CONF) is None:
            self.log(f"no 黄 claim cls → done (all={self._all_claims}, single={self._single_claims})")
            self._goto("exit")
            return action_wait(250, "no more claims → exit")
        return action_wait(400, "waiting (claim_single)")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log(f"done (all={self._all_claims}, single={self._single_claims})")
            return action_done(f"daily_mission complete (all={self._all_claims}, single={self._single_claims})")
        if self._phase_ticks > _EXIT_MAX:
            return action_done("daily_mission exit timeout")
        home = self.find_cls(screen, UC.BTN_HOME, conf=_CLS_CONF)
        if home is not None:
            return action_click_box(home, "daily_mission exit: home")
        back = self.find_cls(screen, UC.BTN_BACK, conf=_CLS_CONF)
        if back is not None:
            return action_click_box(back, "daily_mission exit: back")
        return action_back("daily_mission exit: ESC toward lobby")
