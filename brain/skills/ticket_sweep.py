"""TicketSweepSkill — shared base for 悬赏通缉(bounty) + 学院交流会(JFD).

Verified flow (interactive probe 2026-06-01, data/_missions_probe_log.md).
bounty & JFD are isomorphic ("票券扫荡型"); this base captures the common flow
and subclasses fill in the ticket cls / hub tile / branch picker / AP cost.

★★ THREE-LAYER pyroxene protection (the buy-ticket money bug) ★★
  ① digit-OCR the ticket count at entry → 0 (or unreadable-and-confirm-greyed)
     ⇒ NEVER sortie/sweep (a 0-ticket sweep pops a 購買票券 青辉石 dialog).
  ② never click 立即完成 / buy buttons.
  ③ at the sweep-confirm dialog: if a 青辉石 icon sits in the dialog BODY
     (a buy dialog) OR the confirm is greyed (灰色确认 = insufficient) ⇒ CANCEL.
Tickets are SHARED across branches (probe: 1/6 total), so one MAX sweep on a
single branch drains them all — we pick ONE configured branch, no iteration.

State machine
-------------
enter        lobby → NAV_TASKS → hub → _HUB_TILE → on-page (ticket cls).
ticket_check digit-OCR ticket X/Y. 0 → exit. >0 → branch.
branch       subclass _click_branch() navigates to the configured branch's
             stage list (bounty: cls tiles; JFD: position — no cls, v6 gap).
stage        find 入场键 in the right panel, swipe to the bottom (positions
             stabilize), click the lowest (= highest difficulty) 入场键.
sortie       任務資訊 popup. ⛔ pyroxene-buy guard. If _COSTS_AP, gate on AP.
             MAX_可点击 → MAX (when affordable); else single sweep. → 扫荡开始.
confirm      sweep-confirm dialog. ⛔ pyroxene/grey guard → 确认键.
result       掃蕩完成 popup (WGC transition → poll/re-detect) → 确认键 dismiss.
             Re-read tickets: 0 → exit; >0 → sortie again.
exit         返回键 / 回大厅 → lobby (or hub) → done.

Detectors: base "ui" + "battle" (set by SKILL_YOLO_MAP).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click, action_click_box, action_wait, action_back, action_done,
    action_swipe,
)
from brain.skills import ui_classes as UC

_APP_CONFIG_FILE = Path(__file__).resolve().parents[2] / "data" / "app_config.json"
_CLS_CONF = 0.30
# Sweep-confirm 确认键/取消键 band (probe: y≈0.70).
_CONFIRM_BAND = (0.28, 0.60, 0.72, 0.82)
# 掃蕩完成 reward popup 确认键 sits LOWER (probe: ~0.5, 0.81).
_DONE_CONFIRM_BAND = (0.30, 0.74, 0.70, 0.90)
# A 青辉石 icon inside THIS band = a buy dialog (NOT the top-bar balance at cy<0.10).
_PYROXENE_BODY_REGION = (0.30, 0.16, 0.75, 0.48)
# Stage list lives in the right panel.
_STAGE_PANEL = (0.58, 0.12, 1.0, 0.98)


def _load_profile_list(key: str) -> List[str]:
    """Read an ordered string list from the active app_config profile."""
    try:
        if not _APP_CONFIG_FILE.exists():
            return []
        data = json.loads(_APP_CONFIG_FILE.read_text("utf-8"))
        active = data.get("active_profile", "default")
        profile = (data.get("profiles") or {}).get(active, {})
        raw = profile.get(key)
        if not isinstance(raw, list):
            return []
        out: List[str] = []
        for item in raw:
            s = str(item or "").strip()
            if s and s not in out:
                out.append(s)
        return out
    except Exception:
        return []


class TicketSweepSkill(BaseSkill):
    # ── subclass config (override) ──
    _TICKET_CLS: str = UC.TICKET_BOUNTY        # ticket icon (digit-OCR anchor)
    _HUB_TILE: str = UC.HUB_BOUNTY             # hub entry tile cls
    _PAGE_NAME: str = "Bounty"                 # detect_screen_yolo page (or "")
    _CONFIG_KEY: str = "bounty_branches"       # app_config profile key
    _COSTS_AP: bool = False                    # JFD sweeps cost AP
    _AP_PER_SWEEP: int = 15                    # estimated AP per sweep (JFD)
    _MAX_SWEEP_CYCLES: int = 8                 # safety cap on sweep cycles

    def __init__(self, name: str):
        super().__init__(name)
        self.max_ticks = 120
        self._branches: List[str] = []
        self._init_state()

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._enter_ticks: int = 0
        self._swipe_count: int = 0
        self._last_stage_y: float = -1.0
        self._sweep_cycles: int = 0
        self._tickets: Optional[int] = None
        self._maxed: bool = False
        self._safe_to_max: bool = False
        self._branch_clicks: int = 0
        self._branch_settle: int = 0

    def reset(self) -> None:
        super().reset()
        self._init_state()
        self._branches = _load_profile_list(self._CONFIG_KEY)
        self.log(f"{self.name} branches (config {self._CONFIG_KEY}): {self._branches or 'default'}")

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0

    # ── subclass hooks ───────────────────────────────────────────────────
    def _click_branch(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Navigate to the configured branch's stage list. Return an action to
        click the branch, or None to wait. Default: no branch select needed."""
        return None

    # ── shared helpers ────────────────────────────────────────────────────
    def _on_page(self, screen: ScreenState) -> bool:
        if self.find_cls(screen, self._TICKET_CLS, conf=_CLS_CONF) is not None:
            return True
        if self._PAGE_NAME and self.detect_screen_yolo(screen) == self._PAGE_NAME:
            return True
        return False

    def _read_tickets(self, screen: ScreenState) -> Optional[int]:
        res = self.read_count(screen, self._TICKET_CLS, side="right", span=0.10)
        if res is None:
            return None
        return res[0]

    def _read_ap(self, screen: ScreenState) -> Optional[int]:
        res = self.read_count(screen, UC.TOPBAR_AP, side="right", span=0.10)
        return res[0] if res is not None else None

    def _pyroxene_buy_dialog(self, screen: ScreenState) -> bool:
        """A 青辉石 icon in the dialog body = a buy-ticket/buy-AP dialog."""
        return self.find_cls(
            screen, UC.TOPBAR_PYROXENE, conf=_CLS_CONF, region=_PYROXENE_BODY_REGION
        ) is not None

    def _stage_enters(self, screen: ScreenState) -> List[YoloBox]:
        return self.find_all_cls(screen, UC.STAGE_ENTER, conf=_CLS_CONF, region=_STAGE_PANEL)

    # ── tick ────────────────────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done(f"{self.name} timeout")

        if screen.is_loading():
            return action_wait(700, f"{self.name} loading")

        if self.sub_state == "":
            self._goto("enter")

        handler = {
            "enter": self._enter,
            "ticket_check": self._ticket_check,
            "branch": self._branch,
            "stage": self._stage,
            "sortie": self._sortie,
            "confirm": self._confirm,
            "result": self._result,
            "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, f"{self.name} unknown state")
        return handler(screen)

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks += 1
        if self._on_page(screen):
            self.log(f"inside {self.name} → ticket_check")
            self._goto("ticket_check")
            return action_wait(400, "entered page")

        page = self.detect_screen_yolo(screen)
        if page == "Lobby":
            act = self.click_cls(screen, UC.NAV_TASKS, "open campaign hub", conf=_CLS_CONF)
            if act is not None:
                return act
            return action_wait(400, "lobby: NAV_TASKS not seen")
        if page == "Mission":
            act = self.click_cls(screen, self._HUB_TILE, f"click {self.name} tile", conf=_CLS_CONF)
            if act is not None:
                return act
            return action_wait(450, "hub: tile not seen")

        if self._enter_ticks > 22:
            self.log("can't reach page, exiting")
            self._goto("exit")
            return action_wait(300, "enter timeout")
        if page is not None:
            return action_back(f"back from {page}")
        return action_wait(450, "entering page")

    def _ticket_check(self, screen: ScreenState) -> Dict[str, Any]:
        # ⛔ Defense ①: read the ticket count; 0 ⇒ never sortie (buy-ticket trap).
        tickets = self._read_tickets(screen)
        if tickets is not None:
            self._tickets = tickets
            if tickets <= 0:
                self.log("tickets = 0 → exit (never buy tickets)")
                self._goto("exit")
                return action_wait(300, "0 tickets → exit")
            self.log(f"tickets = {tickets} → branch")
            self._goto("branch")
            return action_wait(250, "tickets ok → branch")

        # Unreadable ticket count: defenses ②③ still protect (the confirm-dialog
        # pyroxene/grey guard), so proceed but log it.
        if self._phase_ticks > 8:
            self.log("ticket count unreadable — proceeding (confirm-dialog guard backstops)")
            self._goto("branch")
            return action_wait(300, "ticket unread → branch (guarded)")
        return action_wait(350, "reading ticket count")

    def _branch(self, screen: ScreenState) -> Dict[str, Any]:
        # Already on a stage list (入场键 visible) → stage select.
        if self._stage_enters(screen):
            self._goto("stage")
            return action_wait(250, "stage list visible → stage")

        if self._branch_settle > 0:
            self._branch_settle -= 1
            return action_wait(350, f"branch settle ({self._branch_settle})")

        act = self._click_branch(screen)
        if act is not None:
            self._branch_clicks += 1
            self._branch_settle = 2
            return act

        if self._phase_ticks > 12:
            self.log("branch select timeout — trying stage anyway")
            self._goto("stage")
            return action_wait(300, "branch timeout → stage")
        return action_wait(400, "selecting branch")

    def _stage(self, screen: ScreenState) -> Dict[str, Any]:
        enters = self._stage_enters(screen)
        if not enters:
            # Only locked stages? bail.
            if self.find_cls(screen, UC.STAGE_ENTER_LOCKED, conf=_CLS_CONF, region=_STAGE_PANEL):
                self.log("only locked stages → exit")
                self._goto("exit")
                return action_wait(300, "locked stages → exit")
            if self._phase_ticks > 10:
                self.log("no 入场键 found → exit")
                self._goto("exit")
                return action_wait(300, "no stage cls → exit")
            return action_wait(400, "waiting for 入场键")

        # Swipe to the bottom: when the lowest 入场键 y stops moving, we're there.
        max_y = max(b.cy for b in enters)
        if self._swipe_count < 6 and abs(max_y - self._last_stage_y) > 0.03:
            self._last_stage_y = max_y
            self._swipe_count += 1
            return action_swipe(0.75, 0.72, 0.75, 0.32, 500, "swipe stage list to bottom")

        # Bottom reached → click the lowest (= highest difficulty) 入场键.
        last = max(enters, key=lambda b: b.cy)
        self.log(f"enter highest stage 入场键 ({last.cx:.2f},{last.cy:.2f})")
        self._goto("sortie")
        return action_click_box(last, "enter highest stage")

    def _sortie(self, screen: ScreenState) -> Dict[str, Any]:
        # ⛔ Defense ③ (early): a buy dialog can pop here too.
        if self._pyroxene_buy_dialog(screen):
            self.log("⛔ pyroxene buy dialog at sortie — cancel + exit")
            return self._cancel_and_exit(screen)

        # Confirm dialog already up → confirm state.
        if self.find_cls(screen, [UC.BTN_CONFIRM, UC.BTN_CONFIRM_GREY], conf=_CLS_CONF, region=_CONFIRM_BAND):
            self._goto("confirm")
            return action_wait(200, "confirm dialog → confirm")

        if not self.find_cls(screen, [UC.SWEEP_START, UC.QTY_MAX, UC.QTY_MAX_GREY], conf=_CLS_CONF):
            # 任務資訊 not open yet — re-enter the stage, or bail.
            if self._phase_ticks > 12:
                self.log("任務資訊 never opened → exit")
                self._goto("exit")
                return action_wait(300, "no sortie popup → exit")
            return action_wait(400, "waiting for 任務資訊 popup")

        # AP gate (JFD): can we afford? safe_to_max = can afford MAX (all tickets).
        if self._COSTS_AP and not self._maxed:
            ap = self._read_ap(screen)
            tix = self._tickets or 1
            if ap is not None:
                if ap < self._AP_PER_SWEEP:
                    self.log(f"AP {ap} < {self._AP_PER_SWEEP}/sweep → exit (no buy-AP)")
                    self._goto("exit")
                    return action_wait(300, "insufficient AP → exit")
                self._safe_to_max = ap >= tix * self._AP_PER_SWEEP
                self.log(f"AP {ap}, tickets {tix} → safe_to_max={self._safe_to_max}")
            else:
                # AP unreadable → don't MAX (sweep 1 at a time; grey-confirm guards).
                self._safe_to_max = False
        else:
            self._safe_to_max = True  # bounty: no AP cost → always MAX

        # MAX (one shot) only when affordable; else leave qty=1 (single sweep).
        if not self._maxed and self._safe_to_max:
            max_btn = self.find_cls(screen, UC.QTY_MAX, conf=_CLS_CONF)
            if max_btn is not None:
                self._maxed = True
                self.log("set sweep MAX (affordable)")
                return action_click_box(max_btn, "set sweep MAX")
            # MAX greyed (e.g. 1 ticket) → proceed to sweep start.
            self._maxed = True

        sweep = self.find_cls(screen, UC.SWEEP_START, conf=_CLS_CONF)
        if sweep is not None:
            self.log("click 扫荡开始")
            self._goto("confirm")
            return action_click_box(sweep, "click sweep start")
        return action_wait(400, "waiting for 扫荡开始")

    def _confirm(self, screen: ScreenState) -> Dict[str, Any]:
        # Sweep done already (掃蕩完成 popped) → result.
        if self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF, region=_DONE_CONFIRM_BAND) is not None \
                or self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_DONE_CONFIRM_BAND) is not None:
            # Could be the 掃蕩完成 reward popup (确认键 lower at ~0.81).
            self._goto("result")
            return action_wait(150, "sweep done popup → result")

        # ⛔ pyroxene buy dialog → cancel + exit.
        if self._pyroxene_buy_dialog(screen):
            self.log("⛔ pyroxene buy dialog at confirm — cancel + exit")
            return self._cancel_and_exit(screen)

        # ⛔ greyed confirm = insufficient (AP/ticket) → cancel + exit.
        if self.find_cls(screen, UC.BTN_CONFIRM_GREY, conf=_CLS_CONF, region=_CONFIRM_BAND) is not None:
            self.log("⛔ confirm greyed (insufficient) — cancel + exit")
            return self._cancel_and_exit(screen)

        confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_CONFIRM_BAND)
        if confirm is not None:
            self._sweep_cycles += 1
            self.log(f"confirm sweep (cycle {self._sweep_cycles}) — currency verified (票券)")
            self._goto("result")
            return action_click_box(confirm, "confirm sweep (tickets, not pyroxene)")

        if self._phase_ticks > 10:
            self._goto("result")
            return action_wait(300, "no confirm dialog → result")
        return action_wait(350, "waiting for sweep-confirm dialog")

    def _result(self, screen: ScreenState) -> Dict[str, Any]:
        # 掃蕩完成 reward popup — re-detect (WGC transition frames). Dismiss via
        # the lower 确认键, or GOT_REWARD header / 点击继续字样.
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
        if cont is not None:
            return action_click_box(cont, "dismiss sweep reward (continue)")
        done_confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_DONE_CONFIRM_BAND)
        if done_confirm is not None:
            self.log("dismiss 掃蕩完成 (确认键)")
            return action_click_box(done_confirm, "dismiss 掃蕩完成")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if got is not None:
            return action_click_box(got, "dismiss sweep reward (header)")

        # Reward dismissed → re-read tickets. MAX drains all → usually 0 now.
        if self._sweep_cycles >= self._MAX_SWEEP_CYCLES:
            self.log("sweep cycle cap → exit")
            self._goto("exit")
            return action_wait(300, "sweep cap → exit")

        tickets = self._read_tickets(screen)
        if tickets is not None and tickets <= 0:
            self.log("tickets drained (0) → exit")
            self._goto("exit")
            return action_wait(300, "tickets 0 → exit")
        if tickets is not None and tickets > 0:
            # Single-sweep path (JFD low-AP) leaves tickets → sweep again.
            self._tickets = tickets
            self._maxed = self._safe_to_max  # keep MAX state if we maxed
            self.log(f"{tickets} tickets remain → sortie again")
            self._goto("sortie")
            return action_wait(300, "more tickets → sortie")

        # Ticket unreadable after a MAX sweep → assume drained → exit.
        if self._phase_ticks > 8:
            self.log("post-sweep ticket unread (MAX likely drained) → exit")
            self._goto("exit")
            return action_wait(300, "post-sweep → exit")
        return action_wait(350, "settling after sweep")

    def _cancel_and_exit(self, screen: ScreenState) -> Dict[str, Any]:
        self._goto("exit")
        cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF)
        if cancel is not None:
            return action_click_box(cancel, "cancel (never buy pyroxene)")
        close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
        if close is not None:
            return action_click_box(close, "close buy dialog (X)")
        return action_back("dismiss buy dialog (ESC)")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        page = self.detect_screen_yolo(screen)
        if page == "Lobby":
            self.log(f"done ({self._sweep_cycles} sweeps)")
            return action_done(f"{self.name} complete")
        if page == "Mission":
            # Stay on the hub — the next campaign skill re-uses it.
            self.log(f"done on hub ({self._sweep_cycles} sweeps)")
            return action_done(f"{self.name} complete (on hub)")
        if self._phase_ticks > 16:
            return action_done(f"{self.name} exit timeout")
        # A leftover 掃蕩完成 / dialog blocks ESC — dismiss its 确认键/X first.
        done_confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_DONE_CONFIRM_BAND)
        if done_confirm is not None:
            return action_click_box(done_confirm, "exit: dismiss leftover result")
        close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
        if close is not None:
            return action_click_box(close, "exit: close dialog")
        back = self.find_cls(screen, UC.BTN_BACK, conf=_CLS_CONF)
        if back is not None:
            return action_click_box(back, "exit: back key")
        return action_back("exit: ESC toward hub/lobby")
