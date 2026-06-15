# -*- coding: utf-8 -*-
"""SpecialSweepSkill — 智能 AP 分配: 把体力优先砸在「多倍加成板块」(2x/3x bonus).

用户 2026-06-15 定的 AP 优先级: 当期活动(活动剧情/活动quest, 可扫的) > 双倍三倍加成
板块 > 正常关。周年庆例外(战斗向, 不沾AP, 无 skill)。本 skill 处理「2x/3x bonus 在
特殊任务」这一档 (2026-06-15 实况): 扫 特殊任务 → 信用货币回收 (用户优先, 信用点干啥都要).

DYNAMIC: 进 hub 后扫 `双倍或三倍活动进行中`(452, v12@0.93) — 只有当它落在 特殊任务
(70) 上才进去扫; 否则 graceful done (bonus 不在这, 留给 batch_sweep 扫正常关)。
所以编排上 special_sweep 排在 batch_sweep **之前**: 有 bonus 先吃 bonus, 没有就退、
batch_sweep 兜底扫正常。

Flow (cls-driven; 特殊任务那几屏不在 semantizer 里):
  enter      lobby → 任务大厅入口 → hub
  board      hub: 452 在 特殊任务(70) 上 → 点 特殊任务; 否则 done(bonus 不在这)
  commission Request Select → 点 信用货币回收(454)
  stage      关卡列表 → 点一个 入场键(79)
  sweep      任務資訊 popup → MAX(111) → 扫荡开始(108)。MAX/start 灰 = AP 耗尽 → close
  confirm    掃蕩內容 dialog(確認+取消): ⛔青辉石 in body → 取消; 否则 確認
  running    skip 键 → results
  results    result page 確認 dismiss → 回 任務資訊 → AP 还够再扫 / 否则 close
  close      任務資訊 X / 返回 → 关卡列表 → 回大厅 → lobby → done

Money: AP only. confirm gate 见青辉石在 body 一律取消(同 batch_sweep)。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click, action_click_box, action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC

_CLS_CONF = 0.30
_MIN_AP = 20           # below this a sweep can't run — skip the trip
# Per-run AP cost of a special-task stage (Lv.78 信用货币回收 = 40, live 2026-06-15).
# 扫前 AP < 此值 → close, 绝不点扫荡开始触发「購買體力(青辉石)」框。保守取实测值。
_SWEEP_COST = 40
_ENTER_MAX = 22
_PHASE_MAX = 14
_RESULT_MAX = 16
_SWEEP_ROUNDS_MAX = 12  # safety cap on re-sweep loops

# Fixed-pos fallbacks (from the 2026-06-15 live walk frames special1/2/3.png).
_POS_SPECIAL_TAB = (0.550, 0.677)   # 特殊任务 tile on hub
_POS_CREDIT = (0.88, 0.40)          # 信用货币回收 commission (cls454)
_POS_MAX = (0.84, 0.42)             # MAX in 任務資訊 sweep panel (right of 加号)
_POS_SWEEP_START = (0.73, 0.56)     # 扫荡开始 (cls108)
# 452 must sit within this radius of the 特殊任务 tile to count as "bonus here".
_BONUS_NEAR = 0.13


class SpecialSweepSkill(BaseSkill):
    """Sweep the 2x/3x-bonus 特殊任务 board (信用货币回收) for 3x AP value."""

    def should_run(self, screen: ScreenState) -> bool:
        # AP gate only (the 452-board check needs the hub, done in _board).
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
        super().__init__("SpecialSweep")
        self.max_ticks = 120
        self._init_state()

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._enter_ticks: int = 0
        self._maxed: bool = False
        self._started: bool = False
        self._result_confirms: int = 0
        self._sweep_rounds: int = 0
        self._swept: bool = False

    def reset(self) -> None:
        super().reset()
        self._init_state()

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0

    # ── helpers ──────────────────────────────────────────────────────────
    def _on_hub(self, screen: ScreenState) -> bool:
        return self.detect_screen_yolo(screen) == "Mission"

    def _bonus_on_special(self, screen: ScreenState) -> Optional[YoloBox]:
        """Return the 特殊任务 tile box IFF a 452 bonus badge sits on/near it."""
        tile = self.find_cls(screen, UC.HUB_SPECIAL, conf=_CLS_CONF)
        if tile is None:
            return None
        for b in self.find_all_cls(screen, UC.EVENT_DOUBLE_TRIPLE, conf=_CLS_CONF):
            if abs(b.cx - tile.cx) < _BONUS_NEAR and abs(b.cy - tile.cy) < _BONUS_NEAR:
                return tile
        return None

    # ── tick ─────────────────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1
        if self.ticks >= self.max_ticks:
            self.log(f"timeout (swept={self._swept})")
            return action_done("special_sweep timeout")
        if screen.is_loading():
            return action_wait(700, "loading")
        if self.sub_state == "":
            self._goto("enter")
        handler = {
            "enter": self._enter, "board": self._board,
            "commission": self._commission, "stage": self._stage,
            "sweep": self._sweep, "confirm": self._confirm,
            "running": self._running, "results": self._results,
            "close": self._close,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "unknown state")
        return handler(screen)

    # ── states ───────────────────────────────────────────────────────────
    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks += 1
        if self._on_hub(screen):
            self._goto("board")
            return action_wait(300, "in hub → check bonus board")
        page = self.detect_screen_yolo(screen)
        if page == "Lobby":
            entry = self.find_cls(screen, UC.NAV_TASKS, conf=0.20)
            if entry is not None:
                if self._phase_ticks % 3 != 1:
                    return action_wait(600, "hall entry clicked — settling")
                return action_click_box(entry, "open task hall")
            return action_wait(400, "lobby: hall entry not seen")
        if self._enter_ticks > _ENTER_MAX:
            self.log("can't reach hub → done")
            return action_done("special_sweep unreachable")
        if len(screen.yolo_boxes or []) < 2:
            return action_wait(700, "no UI — likely loading")
        if page is not None:
            return action_back(f"recover toward lobby ({page})")
        return action_wait(450, "entering hub")

    def _board(self, screen: ScreenState) -> Dict[str, Any]:
        # DYNAMIC routing: only sweep 特殊任务 when the 2x/3x badge is on it.
        tile = self._bonus_on_special(screen)
        if tile is not None:
            self.log("2x/3x bonus 在 特殊任务 → 进入扫荡")
            if self._phase_ticks % 3 != 1:
                return action_wait(500, "特殊任务 clicked — settling")
            self._goto("commission")
            return action_click_box(tile, "open 特殊任务 (bonus board)")
        if self._phase_ticks > _PHASE_MAX:
            # No bonus on 特殊任务 (moved elsewhere / not detected) — nothing for
            # this skill; batch_sweep handles normal stages.
            self.log("2x/3x bonus 不在 特殊任务 → done (留给 batch_sweep)")
            return action_done("special_sweep: bonus not on 特殊任务")
        return action_wait(400, "scanning hub for 2x/3x bonus board")

    def _commission(self, screen: ScreenState) -> Dict[str, Any]:
        # On the 据点防御 stage list already? (re-entry) → stage.
        if self.find_cls(screen, UC.STAGE_ENTER, conf=_CLS_CONF) is not None:
            self._goto("stage")
            return action_wait(250, "already on stage list → stage")
        credit = self.find_cls(screen, UC.SPECIAL_CREDIT, conf=_CLS_CONF)
        if credit is not None:
            if self._phase_ticks % 3 != 1:
                return action_wait(500, "信用货币回收 clicked — settling")
            self.log("select 信用货币回收 (用户优先: 信用点)")
            self._goto("stage")
            return action_click_box(credit, "select 信用货币回收 commission")
        if self._phase_ticks > _PHASE_MAX:
            # commission cls missed → fixed pos.
            self._goto("stage")
            return action_click(*_POS_CREDIT, "select 信用货币回收 (fixed pos)")
        return action_wait(400, "waiting for 委托 select (信用货币回收)")

    def _stage(self, screen: ScreenState) -> Dict[str, Any]:
        # In the sweep popup already (re-entry)? MAX/start visible → sweep.
        if self.find_cls(screen, [UC.SWEEP_START, UC.QTY_MAX, UC.QTY_MAX_GREY],
                         conf=_CLS_CONF) is not None:
            self._goto("sweep")
            return action_wait(250, "sweep panel open → sweep")
        enters = self.find_all_cls(screen, UC.STAGE_ENTER, conf=_CLS_CONF)
        if enters:
            # pick the TOP-most 入场键 (lowest cy) — highest unlocked stage tends
            # to give the most per-AP, and it's always present.
            top = min(enters, key=lambda b: b.cy)
            if self._phase_ticks % 3 != 1:
                return action_wait(500, "入场 clicked — settling")
            self.log(f"enter stage (入场键 {len(enters)} found)")
            self._goto("sweep")
            return action_click_box(top, "enter stage (入场键)")
        if self._phase_ticks > _PHASE_MAX:
            self._goto("close")
            return action_wait(300, "no 入场键 → close")
        return action_wait(400, "waiting for stage list (入场键)")

    def _sweep(self, screen: ScreenState) -> Dict[str, Any]:
        # ⛔ MONEY GATE #0 (2026-06-15 事故根治): NEVER sweep when AP can't fund one
        # run. 一次 MAX 扫荡后余 AP < 单次成本时, 点扫荡开始会弹「購買體力(青辉石)」框 →
        # 险些用青辉石补 AP。扫前读 AP, 不够就 close(从源头不触发买体力框)。读不出=
        # fail-closed 也 close(不盲扫)。
        try:
            from brain.pipeline import _read_topbar_clean
            ap = _read_topbar_clean(UC.TOPBAR_AP)
        except Exception:
            ap = None
        if ap is None or ap < _SWEEP_COST:
            self.log(f"AP={ap} < 单次成本{_SWEEP_COST}(或读不出) → close, 绝不触发买体力框")
            self._goto("close")
            return action_wait(250, "AP 不够一次扫荡 → close (money-safe)")
        # MAX once per round (sets count to the AP/limit ceiling).
        if not self._maxed:
            max_btn = self.find_cls(screen, UC.QTY_MAX, conf=_CLS_CONF)
            if max_btn is not None:
                self._maxed = True
                self.log("click MAX (count → ceiling)")
                return action_click_box(max_btn, "sweep count MAX")
            if self.find_cls(screen, UC.QTY_MAX_GREY, conf=_CLS_CONF) is not None:
                # MAX grey = count already capped (AP ceiling or stage daily limit).
                # If 扫荡开始 also unavailable ⇒ AP exhausted ⇒ close.
                if self.find_cls(screen, UC.SWEEP_START, conf=0.25) is None:
                    self.log("MAX+start grey → AP/limit exhausted → close")
                    self._goto("close")
                    return action_wait(250, "nothing affordable → close")
                self._maxed = True
                self.log("MAX already grey (count capped) → start")
            else:
                if self._phase_ticks > _PHASE_MAX:
                    self._goto("close")
                    return action_wait(300, "MAX never seen → close")
                # MAX cls missed but panel up → fixed-pos MAX.
                if self.find_cls(screen, UC.SWEEP_START, conf=0.25) is not None:
                    self._maxed = True
                    return action_click(*_POS_MAX, "sweep count MAX (fixed pos)")
                return action_wait(400, "waiting for MAX / sweep panel")

        # 扫荡开始.
        if self._phase_ticks % 3 != 1:
            return action_wait(700, "扫荡开始 clicked — settling")
        start = self.find_cls(screen, UC.SWEEP_START, conf=0.25)
        if start is not None:
            self._started = True
            self._sweep_rounds += 1
            self.log(f"click 扫荡开始 (round {self._sweep_rounds})")
            self._goto("confirm")
            return action_click_box(start, "start special sweep")
        if self._phase_ticks > _PHASE_MAX:
            self._goto("close")
            return action_wait(300, "扫荡开始 never seen → close")
        return action_click(*_POS_SWEEP_START, "start special sweep (fixed pos)")

    def _confirm(self, screen: ScreenState) -> Dict[str, Any]:
        # ⛔ Money gate (2026-06-15 事故加强): 青辉石出现在 topbar 以下任意位置(cy>0.10)
        # = 「購買體力」买AP框 → 取消, 绝不确认。原 region(0.15-0.85,0.12-0.75)漏了买AP框
        # 的青辉石位置, 当场花了30青辉石。扩大到整个对话框区(topbar cy<0.10 仍排除)。
        # conf 用模型地板 0.20 — 危险检测要最大灵敏度。
        pyx = self.find_cls(screen, UC.TOPBAR_PYROXENE, conf=0.20,
                            region=(0.08, 0.10, 0.94, 0.86))
        if pyx is not None:
            self.log("⛔ 青辉石在对话框内(买AP框) — 取消, 绝不买青辉石")
            cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=0.20)
            self._goto("close")
            if cancel is not None:
                return action_click_box(cancel, "cancel (buy-AP/pyroxene dialog)")
            return action_back("dismiss buy-AP/pyroxene dialog")

        confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF)
        cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF)
        if confirm is not None and cancel is not None:
            if self._phase_ticks % 3 != 1:
                return action_wait(700, "confirm clicked — settling")
            self.log("掃蕩內容 confirm (AP only, verified)")
            self._goto("running")
            return action_click_box(confirm, "confirm special sweep (AP)")
        # No confirm dialog (swept directly) → look for skip/result.
        if self.find_cls(screen, UC.BATTLE_SKIP, conf=_CLS_CONF) is not None:
            self._goto("running")
            return action_wait(150, "no confirm → running")
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
        if self._phase_ticks > _PHASE_MAX:
            self._goto("results")
            return action_wait(300, "no skip seen → results")
        return action_wait(500, "sweep running")

    def _results(self, screen: ScreenState) -> Dict[str, Any]:
        # Back at the 任務資訊 sweep panel (MAX/start visible) ⇒ the MAX sweep is
        # done. ⛔ DO NOT re-sweep (2026-06-15 事故根治): a MAX sweep already spends
        # all affordable AP in ONE op; re-sweeping the same stage with the leftover
        # (< 单次成本) is what popped the「購買體力」框 → 险些买青辉石。一次 MAX 扫完
        # 就 close。剩余 AP(< 一次成本)留着, 不冒险。
        if self.find_cls(screen, [UC.SWEEP_START, UC.QTY_MAX, UC.QTY_MAX_GREY],
                         conf=_CLS_CONF) is not None and \
                self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF,
                              region=(0.30, 0.55, 0.70, 0.88)) is None:
            self._swept = True
            self.log("MAX 扫荡完成 → close (不 re-sweep, 防低AP触发买体力框)")
            self._goto("close")
            return action_wait(250, "swept → close")
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
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
        if cont is not None:
            return action_click_box(cont, "dismiss result (continue)")
        return action_wait(450, "waiting for results")

    def _close(self, screen: ScreenState) -> Dict[str, Any]:
        page = self.detect_screen_yolo(screen)
        if page == "Lobby":
            self.log(f"done (swept={self._swept}, rounds={self._sweep_rounds})")
            return action_done(f"special_sweep complete (swept={self._swept})")
        if self._phase_ticks > _ENTER_MAX:
            return action_done(f"special_sweep exit timeout (swept={self._swept})")
        if self._phase_ticks % 3 != 1:
            return action_wait(600, "closing — settling")
        home = self.find_cls(screen, UC.BTN_HOME, conf=_CLS_CONF)
        if home is not None:
            return action_click_box(home, "→ lobby (home)")
        close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
        if close is not None:
            return action_click_box(close, "close popup (X)")
        return action_back("close: ESC toward lobby")
