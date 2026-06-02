"""CraftSkill — start quick-crafts + collect finished ones (pure-YOLO rewrite).

Verified flow (interactive probe 2026-06-01, data/_craft_probe_log.md). The old
skill had a "进制造就退" bug (it never started crafts) AND a SAFETY hole: it
looked for the wrong claim cls and could trigger the 立即完成 rush dialog.

★★ HARD RULES (probe-derived) ★★
- The craft-main 一次領取 (CLAIM_ONCE_YELLOW) is AMBIGUOUS:
    • crafts FINISHED → tapping it collects them FREE (direct GOT_REWARD).
    • crafts RUNNING  → tapping it pops 「是否立即完成?」 which spends 製造券
      (a consumable). We NEVER spend券 → if a confirm dialog appears after the
      tap, CANCEL it. (製造券 isn't pyroxene, but user spec: 让制造自然跑完.)
- per-slot 立即完成 (blue, no cls) is NEVER clicked (we have no cls → never
  blind-click there).
- START is safe: 快速制造 costs 信用点 only (probe: 19,000/unit). 快速制造 →
  MAX → 开始制造 → 确认键(確定製造N次) spends credits (abundant) — fine.

State machine
-------------
enter    lobby → NAV_CRAFT → Craft page (CRAFT_QUICK/CRAFT_START signature).
collect  一次领取黄色 → tap. If a confirm dialog (确认键+取消键) appears = the
         券-rush 立即完成 → CANCEL (never spend券). GOT_REWARD = free collect
         (dismissed globally). 一次领取灰色 / none = nothing to collect.
start    快速制造 → MAX_可点击 → 开始制造 → 确认键(确定製造N次, credits, safe).
         Nothing startable (slots busy) → exit.
exit     BTN_HOME / BTN_BACK → lobby → done.

Detectors: base "ui" only.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click_box, action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC

_CLS_CONF = 0.30
# Center-bottom band where craft confirm / 立即完成 dialogs put 确认键/取消键
# (probe: y≈0.70 for both the start-confirm and the券-rush dialog).
_DIALOG_BAND = (0.28, 0.60, 0.72, 0.82)

_ENTER_MAX = 20
_COLLECT_MAX = 16
_START_MAX = 18
_EXIT_MAX = 14


class CraftSkill(BaseSkill):
    def should_run(self, screen: ScreenState) -> bool:
        # Craft is NOT dot-gated — always enter to collect finished + start new
        # (user spec: 制造不需要靠黄点识别). Self-limiting: busy slots can't
        # restart, so re-runs the same day spend nothing.
        return True

    def __init__(self):
        super().__init__("Craft")
        self.max_ticks = 90
        self._init_state()

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._entered: bool = False
        self._collect_done: bool = False
        self._collect_settle: int = 0
        self._maxed_clicks: int = 0
        self._started: bool = False
        self._claims: int = 0

    def reset(self) -> None:
        super().reset()
        self._init_state()

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0

    # ── helpers ──────────────────────────────────────────────────────────
    def _is_craft(self, screen: ScreenState) -> bool:
        page = self.detect_screen_yolo(screen)
        if page == "Lobby":
            self._entered = False
            return False
        if page == "Craft":
            return True
        # All-slots-cooling: CTA still renders (快速制造 persists), but trust the
        # entry gate if a transient frame drops the signature.
        return self._entered and page is None

    def _confirm_dialog(self, screen: ScreenState) -> Optional[YoloBox]:
        """A 2-button confirm dialog (确认键 AND 取消键 in the button band)."""
        confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_DIALOG_BAND)
        cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF, region=_DIALOG_BAND)
        if confirm is not None and cancel is not None:
            return confirm
        return None

    # ── tick ────────────────────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log(f"timeout (claims={self._claims}, started={self._started})")
            return action_done("craft timeout")

        # Global: free-collect reward popup → dismiss via continue / header.
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
        if cont is not None:
            return action_click_box(cont, "dismiss reward via continue")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if got is not None:
            self._claims += 1
            return action_click_box(got, "dismiss reward (got crafts)")

        if screen.is_loading():
            return action_wait(700, "craft loading")

        if self.sub_state == "":
            self._goto("enter")

        handler = {
            "enter": self._enter,
            "collect": self._collect,
            "start": self._start,
            "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "craft unknown state")
        return handler(screen)

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._is_craft(screen):
            self.log("inside craft → collect")
            self._goto("collect")
            return action_wait(400, "entered craft")

        if screen.is_lobby():
            act = self.click_cls(screen, UC.NAV_CRAFT, "open craft", conf=_CLS_CONF)
            if act is not None:
                self._entered = True
                return act
            self.log("on lobby but no 制造入口 — YOLO gap; waiting")
            return action_wait(400, "waiting for 制造入口 cls")

        if self._phase_ticks > _ENTER_MAX:
            return action_done("could not reach craft")
        if len(screen.yolo_boxes or []) < 2:
            return action_wait(700, "no UI detected, likely loading")
        return action_back("craft: recover toward lobby")

    def _collect(self, screen: ScreenState) -> Dict[str, Any]:
        if not self._is_craft(screen) and not self._confirm_dialog(screen):
            if screen.is_lobby():
                self._goto("enter")
                return action_wait(300, "collect: back on lobby, re-enter")
            if self._phase_ticks > _COLLECT_MAX:
                self._goto("exit")
                return action_wait(300, "collect lost craft → exit")
            return action_wait(400, "waiting for craft UI (collect)")

        # ★ A confirm dialog here = the 立即完成(券) rush from tapping 一次領取
        # while crafts are RUNNING → CANCEL, never spend券.
        dlg = self._confirm_dialog(screen)
        if dlg is not None:
            self.log("⛔ 一次领取 → 立即完成(券) dialog → cancel, never spend券")
            self._collect_done = True
            cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF, region=_DIALOG_BAND)
            if cancel is not None:
                return action_click_box(cancel, "cancel 立即完成 (keep券)")
            return action_back("cancel 立即完成 (ESC)")

        if self._collect_settle > 0:
            self._collect_settle -= 1
            return action_wait(400, f"collect settle ({self._collect_settle})")

        if self._collect_done:
            self._goto("start")
            return action_wait(250, "collect done → start")

        # 一次领取黄色: finished crafts collect FREE; running ones pop the券
        # dialog (caught above). Tap it, then settle to see which happens.
        yellow = self.find_cls(screen, UC.CLAIM_ONCE_YELLOW, conf=_CLS_CONF)
        if yellow is not None:
            self.log("tapping 一次领取黄色 (collect finished crafts)")
            self._collect_settle = 2
            return action_click_box(yellow, "collect finished crafts")

        # Grey claim-all or nothing → nothing (free) to collect.
        self.log("no 一次领取黄色 → nothing to collect, start phase")
        self._goto("start")
        return action_wait(250, "no free collect → start")

    def _start(self, screen: ScreenState) -> Dict[str, Any]:
        if not self._is_craft(screen) and not self._confirm_dialog(screen):
            if screen.is_lobby():
                self._goto("exit")
                return action_wait(300, "start: on lobby → exit")
            if self._phase_ticks > _START_MAX:
                self._goto("exit")
                return action_wait(300, "start lost craft → exit")
            return action_wait(400, "waiting for craft UI (start)")

        # craft-start confirm (確定製造N次) — reached via 开始制造 → safe credits.
        dlg = self._confirm_dialog(screen)
        if dlg is not None:
            self.log("confirming craft start (確定製造N次, 耗信用点 — safe)")
            self._started = True
            self._goto("exit")
            return action_click_box(dlg, "confirm craft start (credits)")

        # In the 快速制造 dialog? MAX_可点击 / 开始制造 present.
        max_btn = self.find_cls(screen, UC.QTY_MAX, conf=_CLS_CONF)
        if max_btn is not None and self._maxed_clicks < 3:
            self._maxed_clicks += 1
            self.log("set MAX quantity (YOLO MAX_可点击)")
            return action_click_box(max_btn, "set craft quantity MAX")
        start_btn = self.find_cls(screen, UC.CRAFT_START, conf=_CLS_CONF)
        if start_btn is not None:
            self.log("clicking 开始制造")
            return action_click_box(start_btn, "start craft (开始制造)")

        # Not in dialog → open it via 快速制造.
        quick = self.find_cls(screen, UC.CRAFT_QUICK, conf=_CLS_CONF)
        if quick is not None and self._phase_ticks <= _START_MAX:
            self.log("opening 快速制造 dialog")
            return action_click_box(quick, "open quick-craft")

        # Nothing startable (slots busy / no free slot) or budget out → exit.
        self.log("nothing startable (busy slots / YOLO gap) → exit")
        self._goto("exit")
        return action_wait(300, "no startable craft → exit")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log(f"done (claims={self._claims}, started={self._started})")
            return action_done(f"craft complete (claims={self._claims}, started={self._started})")
        if self._phase_ticks > _EXIT_MAX:
            return action_done("craft exit timeout")
        # Close any leftover dialog first, then home/back.
        close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF, region=(0.55, 0.08, 0.97, 0.30))
        if close is not None:
            return action_click_box(close, "close leftover craft dialog")
        home = self.find_cls(screen, UC.BTN_HOME, conf=_CLS_CONF)
        if home is not None:
            return action_click_box(home, "craft exit: home button")
        back = self.find_cls(screen, UC.BTN_BACK, conf=_CLS_CONF)
        if back is not None:
            return action_click_box(back, "craft exit: back button")
        return action_back("craft exit: ESC toward lobby")
