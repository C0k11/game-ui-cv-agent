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
# Deep-dive C5 (2026-06-09): aligned to schedule's LIVE-VERIFIED region (icon
# at cy≈0.577 > old 0.48 bound — same miss risk as arena C4).
_PYROXENE_BODY_REGION = (0.20, 0.12, 0.82, 0.64)
# Stage list lives in the right panel.
_STAGE_PANEL = (0.58, 0.12, 1.0, 0.98)
# 任務資訊 popup MAX button fixed pos (right of 加号; proven on special_sweep
# 2026-06-15). Fallback when cls111 MAX_可点击 is missed → 防只扫1票.
_POS_TICKET_MAX = (0.84, 0.42)
# Re-click the 入場键 this many times if 任務資訊 never opens (a dropped tap —
# root-fixed by AdbInput._IO_LOCK, but kept as self-healing so a single lost
# enter never costs the whole sweep). Live 2026-06-15: swept 0, manual same-pos
# tap opened it → tap was lost, not mis-aimed.
_SORTIE_MAX_RETRIES = 2


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
    _MAX_RENDER_WAIT: int = 5                  # frames to wait for solid MAX before giving up
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
        self._max_wait: int = 0
        self._branch_clicks: int = 0
        self._branch_settle: int = 0
        self._sortie_retries: int = 0

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
        """digit-OCR the 持有票券 X/Y next to the ticket icon. ★ money defense #1
        (0 tickets ⇒ never sortie → never the buy-pyroxene trap), so the read
        must be robust. read_count's generic strip (x-start = icon.x2 + 0.005)
        clipped the first digit on live frames → None (live 2026-06-02: bounty
        read None 8× then fell through to the guarded path). Anchor on the
        top-left ticket badge + OCR a tighter-left, wider strip (verified: span
        0.11 from icon.x2+0.002 → '6/6')."""
        if screen.frame is None:
            return None
        try:
            from brain.pipeline import run_digit_ocr, parse_count
        except Exception:
            return None
        icon = self.find_cls(screen, self._TICKET_CLS, conf=0.20,
                             region=(0.0, 0.04, 0.26, 0.36))
        if icon is None:
            # Ticket cls anchor FLICKERS (live 2026-06-09: bounty 票 cls
            # zero-detected on the branch page while the counter rendered
            # plainly top-left). The counter is a stable page fixture →
            # fixed-region OCR fallback on the DIGITS zone only (0708 新皮肤
            # 「持有票券 6/6」布局, 两页帧离线验证 '6/6' ✓)。
            raw = run_digit_ocr(screen.frame, (0.115, 0.121, 0.185, 0.163))
            res = parse_count(raw)
            if res is not None and res[0] is not None:
                self.log(f"tickets via fixed-region fallback: {res[0]} (raw {raw!r})")
                return res[0]
            self.log(f"[tkdbg] no icon anchor; fallback raw={raw!r}")
            return None
        bh = icon.y2 - icon.y1
        x1 = max(0.0, icon.x2 + 0.002)
        x2 = min(1.0, x1 + 0.11)
        y1s, y2s = icon.y1 - bh * 0.4, icon.y2 + bh * 0.4
        raw = run_digit_ocr(screen.frame, (x1, y1s, x2, y2s))
        res = parse_count(raw)
        if res is None or res[0] is None:
            # 0708 更新换皮:「持有票券 6/6」斜体数字在含中文标签的整条 strip 上
            # 被 det 漏检(live 实锤 raw=None ×8 → fail-closed 0 sweeps; 同帧
            # 去掉标签只留数字区就读出)。数字区 = 标签右侧 x1+0.055 起。
            raw = run_digit_ocr(screen.frame, (min(1.0, x1 + 0.055), y1s, x2, y2s))
            res = parse_count(raw)
        if res is None or res[0] is None:
            self.log(f"[tkdbg] icon@({icon.x1:.3f},{icon.y1:.3f},{icon.x2:.3f},"
                     f"{icon.y2:.3f}) conf={icon.confidence:.2f} both strips "
                     f"unread (last raw={raw!r})")
            return None
        return res[0]

    def _read_ap(self, screen: ScreenState) -> Optional[int]:
        # Calibrated clean-frame read (2026-06-11): the generic read_count span
        # left-truncated 199/240 → '1/240' live → JFD exited with 13 sweeps of
        # AP unspent. _read_topbar_clean votes over clean ADB frames with the
        # per-currency span (AP 0.06) — fail to the live read only if no clean
        # source is registered.
        try:
            from brain.pipeline import _read_topbar_clean
            ap = _read_topbar_clean(UC.TOPBAR_AP)
            if ap is not None:
                return ap
        except Exception:
            pass
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
            act = self.click_cls(screen, UC.NAV_TASKS, "open campaign hub", conf=0.20)
            if act is not None:
                return act
            # 任务大厅入口 (19f) systematically misses on event-skinned lobbies
            # (live 2026-06-09: the 任務 tile wears a 正在進行考試 banner → cls
            # never fired once all day → enter timed out, 0 sweeps). The tile
            # is a fixed right-side fixture → fixed-slot fallback after a few
            # patient ticks. 根治 = 补标 (clean-flywheel frames have it).
            if self._enter_ticks > 4:
                self.log("任务大厅入口 cls missed → fixed-pos fallback (0.935,0.80)")
                return action_click(0.935, 0.80, "open campaign hub (fixed pos)")
            return action_wait(400, "lobby: NAV_TASKS not seen")
        if page == "Mission":
            # ★ Hall scan (user iron rule 2026-06-11): the per-activity dot is
            # only visible HERE — tile with no red/yellow dot = no work today,
            # exit gracefully instead of entering blind.
            has_work = self.hall_tile_dot(screen, self._HUB_TILE)
            if has_work is False:
                self.log(f"hall scan: {self._HUB_TILE} 无红黄点 → no work today, done")
                return action_done(f"{self.name} no work (hall scan)")
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

        # Deep-dive C7 (2026-06-09): unreadable ticket count must FAIL CLOSED.
        # The old "proceed, confirm-dialog guard backstops" relied on a guard
        # whose region was mis-sized (C5) — 票数读不出 ⇒ 不出击, period
        # (money rule #3: 0/unknown tickets → never sortie).
        if self._phase_ticks > 8:
            self.log("ticket count unreadable after retries → exit (money fail-closed)")
            self._goto("exit")
            return action_wait(300, "ticket unread → exit")
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
            # 任務資訊 not open yet. The 入場 tap intermittently DROPS under adbd
            # contention (live 2026-06-15: skill tap lost → popup never showed →
            # 0 tickets swept; manual same-pos tap opened it). Root-fixed by the
            # AdbInput I/O lock; self-healing backstop here — re-click the 入場键
            # (bounded) instead of giving up with tickets unspent.
            if self._phase_ticks > 7:
                if self._sortie_retries < _SORTIE_MAX_RETRIES:
                    self._sortie_retries += 1
                    self.log(f"任務資訊 未开 (入場 tap 可能丢失) → 回 stage 重点入場键 "
                             f"retry {self._sortie_retries}/{_SORTIE_MAX_RETRIES}")
                    self._goto("stage")
                    return action_wait(300, "re-enter stage (tap-loss retry)")
                self.log("任務資訊 never opened after retries → exit")
                self._goto("exit")
                return action_wait(300, "no sortie popup → exit")
            return action_wait(400, "waiting for 任務資訊 popup")

        # AP gate (JFD): the MAX button sweeps as many times as AP+tickets allow
        # — the GAME caps it, never overspends (you can't go negative on AP).
        # So MAX is safe whenever AP ≥ one sweep; the old `ap ≥ tix×AP_PER_SWEEP`
        # gate wrongly fell back to a SINGLE sweep whenever full tickets couldn't
        # all be afforded (live 2026-06-09: 24 tickets needed 360 AP > 240 cap →
        # safe_to_max False → swept only 1, leaving AP+tickets unspent).
        if self._COSTS_AP and not self._maxed:
            ap = self._read_ap(screen)
            if ap is not None:
                if ap < self._AP_PER_SWEEP:
                    self.log(f"AP {ap} < {self._AP_PER_SWEEP}/sweep → exit (no buy-AP)")
                    self._goto("exit")
                    return action_wait(300, "insufficient AP → exit")
                self._safe_to_max = True  # game caps MAX at affordable count
                self.log(f"AP {ap} ≥ {self._AP_PER_SWEEP} → MAX (game caps to affordable)")
            else:
                # AP unreadable → don't MAX (sweep 1 at a time; grey-confirm guards).
                self._safe_to_max = False
        else:
            self._safe_to_max = True  # bounty: no AP cost → always MAX

        # MAX (one shot) only when affordable; else leave qty=1 (single sweep).
        if not self._maxed and self._safe_to_max:
            # QTY_MAX(MAX_可点击)是弱类: 在明显可点的蓝 MAX 上只 fire 到 conf≈0.26
            # (<0.30, special_sweep.py:228 实测), 用 _CLS_CONF=0.30 检不到 → 退化固定位
            # MAX(2026-06-26 实测 bounty/jfd 都走了固定位)。降到 0.20 地板对齐 special_sweep
            # 让 cls 路径优先(action_click_box 比硬编码 0.84,0.42 跨分辨率更鲁棒)。bounty/jfd
            # 纯票券: 即便 0.20 在灰 MAX 上误 fire, 游戏把 count 钳到持有票数→confirm "用N票券"
            # 不弹买票框, 且 _confirm 青辉石防线兜底 → 安全。固定位 fallback 仍保留兜底。
            max_btn = self.find_cls(screen, UC.QTY_MAX, conf=0.20)
            if max_btn is not None:
                self._maxed = True
                self.log("set sweep MAX (affordable)")
                return action_click_box(max_btn, "set sweep MAX")
            # No SOLID MAX yet. During the 任務資訊 popup open-animation the MAX
            # button renders grey-then-solid; abandoning on the first miss (live
            # 2026-06-09 JFD) set _maxed=True too early → swept ONCE with 24
            # tickets unspent. Wait a few frames for the solid MAX to settle;
            # only if it never appears (truly greyed = 1 ticket) fall back.
            if self._max_wait < self._MAX_RENDER_WAIT:
                self._max_wait += 1
                return action_wait(350, f"waiting MAX render ({self._max_wait})")
            # cls111 MAX_可点击 flaky (live 2026-06-15: bounty/jfd 只扫了1票, MAX 没
            # 检到就退化 single sweep, 浪费票). 等待后仍没检到 → 点 任務資訊 popup 的
            # 固定位 MAX(special_sweep 已验证 0.84,0.42)再扫, 不退 single。游戏把 MAX
            # 钳到可负担数, 安全; 若 MAX 是灰的(真只剩1票/到上限), 点固定位是无害空操作。
            if not getattr(self, "_max_fixed_tried", False):
                self._max_fixed_tried = True
                self._maxed = True
                self.log("MAX cls 没检到 → 固定位 MAX (防只扫1票)")
                return action_click(*_POS_TICKET_MAX, "set sweep MAX (fixed pos)")
            self._maxed = True
            self.log("MAX greyed after wait → single sweep")

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
        return self.nav_home(screen, "ticket exit")
