# -*- coding: utf-8 -*-
"""BatchSweepSkill — 批量掃蕩 (刷体力): spend remaining AP on the user's saved
sweep preset, MAXed, at the 任務 stage screen.

Spec = the 11-step live walk 2026-06-11 (data/_batch_sweep_probe_log.md,
frames logs/_explore_000..011.jpg). AP 999→199 verified, zero pyroxene.

This is the FIRST skill driven end-to-end by the screen semantizer
(brain.screens.classify_screen): every state verifies the SCREEN before
clicking — UNKNOWN screen ⇒ wait/recover, never blind-click (用户铁律:
识别上了再决定, 不抢跑不乱点; 图稳不图快).

Ordering (user 2026-06-11): hall block = 悬赏通缉 → 学院交流会 → 批量掃蕩 →
战术大赛 — ticket activities first (their own AP+ticket spend is bounded),
batch sweep then eats whatever AP remains, arena (no AP) last.

Flow
----
enter        lobby → 任务大厅入口 (fixed-pos fallback) → task_hall
hall         task_hall → click 任务关卡推图 tile
stage        stage_select → click 批量掃蕩 (cls455; live-missed → fixed
             (0.417,0.815) fallback)
dialog       sweep_batch_dialog → preset = first tab (user's saved presets)
             → click MAX → count>0 (MAX greys) → click 批量掃蕩 yellow
             (cls456; fallback (0.868,0.818)). All-grey steppers + no start
             ⇒ AP exhausted ⇒ close & done (MAX-clamped-to-0 case).
confirm      掃蕩內容 dialog (確認+取消): ⛔ pyroxene-in-body gate → 確認.
running      skip键 overlay → click skip.
results      result_page (確認+X, no 取消) → 確認 (loops over result pages).
close        back at sweep_batch_dialog (steppers grey = swept) → X →
             stage_select → 回大厅 → lobby → done.

Money: AP only. The confirm gate cancels on any pyroxene in the dialog body.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC
from brain.screens import (
    classify_screen, SWEEP_BATCH, SWEEP_BATCH_START, SWEEP_BATCH_START_GREY,
)

_CLS_CONF = 0.30
# Fixed-position fallbacks from the live walk (v8 missed these cls live).
_POS_SWEEP_BTN = (0.417, 0.815)     # 批量掃蕩 on stage_select (cls455 gap)
_POS_START_BTN = (0.868, 0.818)     # yellow 批量掃蕩 in dialog (cls456 gap)
_POS_HALL_ENTRY = (0.935, 0.80)     # 任务大厅入口 fallback (event-skin gap)
_POS_CAMPAIGN = (0.599, 0.244)      # 任务关卡推图 tile (cls67 backup)

_MIN_AP = 20          # below this, a sweep can't run — skip the whole trip
_ENTER_MAX = 22
_PHASE_MAX = 14
_RESULT_MAX = 16


class BatchSweepSkill(BaseSkill):
    """Spend remaining AP via the saved 批量掃蕩 preset (MAX count)."""

    def should_run(self, screen: ScreenState) -> bool:
        # AP gate: don't even travel when AP can't fund one sweep. Clean-frame
        # read; unreadable → go look anyway (the dialog's MAX-grey case exits
        # safely — fail-open here is harmless, it costs only the walk).
        try:
            from brain.pipeline import _read_topbar_clean
            ap = _read_topbar_clean(UC.TOPBAR_AP)
            if ap is not None and ap < _MIN_AP:
                self.log(f"AP {ap} < {_MIN_AP} → nothing to sweep, skip")
                return False
        except Exception:
            pass
        return True

    def __init__(self):
        super().__init__("BatchSweep")
        self.max_ticks = 90
        self._init_state()

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._enter_ticks: int = 0
        self._maxed: bool = False
        self._started: bool = False
        self._result_confirms: int = 0
        self._swept: bool = False

    def reset(self) -> None:
        super().reset()
        self._init_state()
        self._hard_assist_done = False

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0

    # ── semantizer shorthand ────────────────────────────────────────────
    def _screen(self, screen: ScreenState):
        return classify_screen(screen.yolo_boxes)

    # ── tick ────────────────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1
        if self.ticks >= self.max_ticks:
            self.log(f"timeout (swept={self._swept})")
            return action_done("batch_sweep timeout")
        if screen.is_loading():
            return action_wait(700, "loading")
        if self.sub_state == "":
            self._goto("enter")
        handler = {
            "enter": self._enter,
            "hall": self._hall,
            "stage": self._stage,
            "dialog": self._dialog,
            "confirm": self._confirm,
            "running": self._running,
            "results": self._results,
            "close": self._close,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "unknown state")
        return handler(screen)

    # ── states ──────────────────────────────────────────────────────────
    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks += 1
        sid, _, _ = self._screen(screen)
        if sid == "task_hall":
            self._goto("hall")
            return action_wait(300, "in hall")
        if sid in ("stage_select", "sweep_batch_dialog"):
            self._goto("hall" if sid == "task_hall" else
                       ("stage" if sid == "stage_select" else "dialog"))
            return action_wait(250, f"resume at {sid}")
        if sid == "lobby":
            entry = self.find_cls(screen, UC.NAV_TASKS, conf=0.20)
            if entry is not None:
                # Pace the entry retry (popup-dismiss disease class).
                if self._phase_ticks % 3 != 1:
                    return action_wait(600, "hall entry clicked — settling")
                return action_click_box(entry, "open task hall")
            if self._enter_ticks > 5:
                if self._phase_ticks % 3 != 1:
                    return action_wait(600, "hall entry (fixed) — settling")
                return action_click(*_POS_HALL_ENTRY, "open task hall (fixed pos)")
            return action_wait(400, "lobby: hall entry not seen")
        if self._enter_ticks > _ENTER_MAX:
            self.log("can't reach hall → done")
            return action_done("batch_sweep unreachable")
        # Unknown screen — semantizer rule: don't click what we can't name.
        if len(screen.yolo_boxes or []) < 2:
            return action_wait(700, "no UI — likely loading")
        return self.nav_home(screen, "batch_sweep recover")

    def _hall(self, screen: ScreenState) -> Dict[str, Any]:
        sid, _, _ = self._screen(screen)
        if sid == "stage_select":
            self._goto("stage")
            return action_wait(300, "stage select reached")
        if sid != "task_hall":
            if self._phase_ticks > _PHASE_MAX:
                self._goto("enter")
                return action_wait(300, "hall lost → re-enter")
            return action_wait(450, f"waiting for hall ({sid})")
        tile = self.find_cls(screen, "任务关卡推图", conf=_CLS_CONF)
        if self._phase_ticks % 3 != 1:
            return action_wait(600, "campaign tile clicked — settling")
        if tile is not None:
            return action_click_box(tile, "open campaign stages")
        return action_click(*_POS_CAMPAIGN, "open campaign stages (fixed pos)")

    def _stage(self, screen: ScreenState) -> Dict[str, Any]:
        sid, _, _ = self._screen(screen)
        if sid == "sweep_batch_dialog":
            self._goto("dialog")
            return action_wait(300, "sweep dialog open")
        if sid != "stage_select":
            if self._phase_ticks > _PHASE_MAX:
                self._goto("enter")
                return action_wait(300, "stage select lost → re-enter")
            return action_wait(450, f"waiting for stage select ({sid})")
        btn = self.find_cls(screen, SWEEP_BATCH, conf=0.25)
        if self._phase_ticks % 3 != 1:
            return action_wait(600, "批量掃蕩 clicked — settling")
        if btn is not None:
            return action_click_box(btn, "open batch sweep dialog")
        # v9 context bias (user-diagnosed 2026-06-12): 455 trained almost
        # entirely on Hard-selected pages → blind when Normal selected.
        # User's mitigation: click the Hard tab once so the model can SEE
        # the button (cls-driven beats blind fixed-pos; sweep itself uses
        # the saved preset, independent of the visible tab). One attempt.
        from brain.skills.ui_classes import STAGE_HARD
        hard = self.find_cls(screen, STAGE_HARD, conf=0.4)
        if hard is not None and not getattr(self, "_hard_assist_done", False):
            self._hard_assist_done = True
            return action_click_box(hard, "select Hard tab (455 visibility assist)")
        # final fallback: fixed position from the live walk.
        return action_click(*_POS_SWEEP_BTN, "open batch sweep dialog (fixed pos)")

    def _dialog(self, screen: ScreenState) -> Dict[str, Any]:
        sid, _, dialog = self._screen(screen)
        if dialog == "confirm_dialog":
            self._goto("confirm")
            return action_wait(200, "confirm dialog up")
        if sid != "sweep_batch_dialog":
            if self._phase_ticks > _PHASE_MAX:
                self._goto("enter")
                return action_wait(300, "dialog lost → re-enter")
            return action_wait(450, f"waiting for sweep dialog ({sid})")

        # Preset: the user's saved tabs — first tab is selected by default,
        # nothing to click (profile-configurable later if needed).

        # MAX once. After MAX the steppers grey out (count at AP ceiling).
        if not self._maxed:
            max_btn = self.find_cls(screen, UC.QTY_MAX, conf=_CLS_CONF)
            if max_btn is not None:
                # 双发 latch(2026-07-21 mutate-before-ack: MAX 首发被吞时旧码已
                # _maxed=True → MAX 没点上只扫默认次数)。连发两 tick 再 latch。
                self._max_fires = getattr(self, "_max_fires", 0) + 1
                if self._max_fires >= 2:
                    self._maxed = True
                self.log(f"click MAX (fire {self._max_fires}, count → AP ceiling)")
                return action_click_box(max_btn, "sweep count MAX")
            # MAX already grey? count may already be capped — proceed.
            if self.find_cls(screen, UC.QTY_MAX_GREY, conf=_CLS_CONF) is not None:
                # All-grey steppers AND a grey start = nothing affordable.
                if (self.find_cls(screen, SWEEP_BATCH_START_GREY, conf=0.25) is not None
                        and self.find_cls(screen, SWEEP_BATCH_START, conf=0.25) is None):
                    self.log("steppers+start grey → AP exhausted, nothing to sweep")
                    self._goto("close")
                    return action_wait(250, "nothing affordable → close")
                self._maxed = True
                self.log("MAX already grey (count capped) → start")
            else:
                if self._phase_ticks > _PHASE_MAX:
                    self._goto("close")
                    return action_wait(300, "MAX never seen → close")
                return action_wait(400, "waiting for MAX")

        # Start: cls456 when seen, else the yellow button's fixed spot.
        if self._phase_ticks % 3 != 1:
            return action_wait(700, "start clicked — settling")
        start = self.find_cls(screen, SWEEP_BATCH_START, conf=0.25)
        if start is not None:
            self._started = True
            self.log("click 批量掃蕩 start (cls)")
            return action_click_box(start, "start batch sweep")
        if self.find_cls(screen, SWEEP_BATCH_START_GREY, conf=0.25) is not None:
            self.log("start greyed → nothing affordable, close")
            self._goto("close")
            return action_wait(250, "start grey → close")
        self._started = True
        self.log("click 批量掃蕩 start (fixed pos, cls456 gap)")
        return action_click(*_POS_START_BTN, "start batch sweep (fixed pos)")

    def _confirm(self, screen: ScreenState) -> Dict[str, Any]:
        # ⛔ Money gate: pyroxene in the dialog BODY (top bar cy<0.10 excluded)
        # = a buy dialog — cancel, never confirm.
        pyx = self.find_cls(screen, UC.TOPBAR_PYROXENE, conf=0.20,
                            region=(0.15, 0.12, 0.85, 0.75))
        if pyx is not None:
            self.log("⛔ pyroxene in dialog body — cancel, never buy")
            cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=0.20)
            self._goto("close")
            if cancel is not None:
                return action_click_box(cancel, "cancel (pyroxene dialog)")
            return action_back("dismiss pyroxene dialog")

        confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF)
        cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF)
        if confirm is not None and cancel is not None:
            if self._phase_ticks % 3 != 1:
                return action_wait(700, "confirm clicked — settling")
            # reason 加"確認键"(2026-07-21 mutate-before-ack: 渲染好的确认框属
            # "看到就点", 稳定门豁免立即点 → 不被吞 → goto running 前置安全)。
            self.log("掃蕩內容 confirm (AP only, verified)")
            self._goto("running")
            return action_click_box(confirm, "confirm batch sweep (AP, 確認键)")
        sid, _, _ = self._screen(screen)
        if sid == "sweep_running":
            self._goto("running")
            return action_wait(200, "sweep running")
        if self._phase_ticks > _PHASE_MAX:
            self._goto("results")
            return action_wait(300, "confirm gone → results")
        return action_wait(400, "waiting for 掃蕩內容 dialog")

    def _running(self, screen: ScreenState) -> Dict[str, Any]:
        skip = self.find_cls(screen, UC.BATTLE_SKIP, conf=_CLS_CONF)
        if skip is not None:
            self.log("skip sweep animation")
            self._goto("results")
            return action_click_box(skip, "skip sweep animation")
        sid, _, _ = self._screen(screen)
        if sid == "result_page":
            self._goto("results")
            return action_wait(200, "results up")
        if self._phase_ticks > _PHASE_MAX:
            self._goto("results")
            return action_wait(300, "no skip seen → results")
        return action_wait(500, "sweep running")

    def _results(self, screen: ScreenState) -> Dict[str, Any]:
        sid, _, dialog = self._screen(screen)
        if sid == "sweep_batch_dialog":
            self._swept = True
            self.log("back at sweep dialog — sweep complete")
            self._goto("close")
            return action_wait(300, "swept → close")
        if self._result_confirms >= _RESULT_MAX:
            self._goto("close")
            return action_wait(300, "result cap → close")
        confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF)
        if confirm is not None:
            if self._phase_ticks % 3 != 1:
                return action_wait(600, "result confirm — settling")
            self._result_confirms += 1
            self._swept = True
            self.log(f"dismiss 掃蕩完成 page (#{self._result_confirms})")
            return action_click_box(confirm, "dismiss sweep results")
        return action_wait(450, f"waiting for results ({sid})")

    def _close(self, screen: ScreenState) -> Dict[str, Any]:
        sid, _, _ = self._screen(screen)
        if sid == "lobby":
            self.log(f"done (swept={self._swept})")
            return action_done(f"batch_sweep complete (swept={self._swept})")
        if sid == "task_hall":
            # 收尾停 hub(用户 2026-07-07: hub 内技能别退大厅 — 下一个 hub skill
            # (arena / special_sweep 回马枪) 从 hub 直接起, 省一次 lobby 往返)。
            self.log(f"done on hub (swept={self._swept})")
            return action_done(f"batch_sweep complete on hub (swept={self._swept})")
        if self._phase_ticks > _ENTER_MAX:
            return action_done(f"batch_sweep exit timeout (swept={self._swept})")
        if self._phase_ticks % 3 != 1:
            return action_wait(600, "closing — settling")
        if sid == "sweep_batch_dialog":
            close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
            if close is not None:
                return action_click_box(close, "close sweep dialog")
            return action_back("close sweep dialog (ESC)")
        if sid == "stage_select":
            home = self.find_cls(screen, UC.BTN_HOME, conf=_CLS_CONF)
            if home is not None:
                return action_click_box(home, "stage select → lobby")
            return action_back("stage select → back")
        # result page leftovers etc.
        confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF)
        if confirm is not None and self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF) is None:
            return action_click_box(confirm, "close: dismiss leftover result")
        return self.nav_home(screen, "batch_sweep close")
