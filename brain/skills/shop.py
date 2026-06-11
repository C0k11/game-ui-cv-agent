"""ShopSkill — daily 一般(信用点) shop bulk-buy with dynamic budget (rewrite).

Verified flow (interactive probe 2026-06-01, data/_shop_probe_log.md). The old
skill blindly confirmed ANY purchase. The user requires DYNAMIC budget planning:
read the credit balance and only buy when we stay above a configured reserve.

★★ HARD RULES ★★
- NEVER touch the 青辉石商店 tab (SHOP_TAB_PYROXENE_SEL) — that spends pyroxene.
  We stay on the default 一般(信用点) tab; if we ever detect the pyroxene tab
  selected, abort. The 一般 tab is 100% credits, so this skill cannot spend
  pyroxene/real money by construction.
- Dynamic budget: read top-bar credit balance (reliable anchor) + best-effort
  the dialog 總購買價格. Buy only if balance − total ≥ reserve (or, when the
  total can't be read, only when balance is comfortably above reserve+ceiling).
  Reserve is configurable (app_config profile `shop_credit_reserve`).

State machine
-------------
enter    lobby → NAV_SHOP → shop page (默认 一般 tab). Pyroxene-tab guard.
select   全部选择 (SHOP_SELECT_ALL) → all items green-checked. 全部选择灰 = nothing
         to buy / already bought today → exit.
buy      选择购买 (SHOP_BUY_SELECTED) → purchase-confirm dialog.
confirm  read budget → balance−total ≥ reserve → 确认键; else 取消键 → exit.
exit     GOT_REWARD dismissed globally → lobby → done.

Detectors: base "ui" only. OCR: DIGITS ONLY (credit balance / total).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click, action_click_box, action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_APP_CONFIG_FILE = _DATA_DIR / "app_config.json"

_CLS_CONF = 0.30
# Center-bottom band where the purchase-confirm 确认键/取消键 sit (probe ~0.79).
_DIALOG_BAND = (0.28, 0.72, 0.72, 0.90)
# Default credits to keep in reserve when the dialog total can't be read.
_DEFAULT_RESERVE = 5_000_000
# Conservative ceiling for an unread 一般-tab total (observed ~3.1M; 20M is a
# safe upper bound). Only used when the precise total OCR fails.
_ASSUMED_MAX_TOTAL = 20_000_000

_ENTER_MAX = 20
_SELECT_MAX = 16
_BUY_MAX = 14
_CONFIRM_MAX = 12
_EXIT_MAX = 14


def _load_shop_reserve() -> int:
    """Read shop_credit_reserve from the active app_config profile (credits to
    keep). Mirrors cafe._load_invite_targets. Default _DEFAULT_RESERVE."""
    try:
        if not _APP_CONFIG_FILE.exists():
            return _DEFAULT_RESERVE
        data = json.loads(_APP_CONFIG_FILE.read_text("utf-8"))
        active = data.get("active_profile", "default")
        profile = (data.get("profiles") or {}).get(active, {})
        raw = profile.get("shop_credit_reserve")
        if raw is None:
            return _DEFAULT_RESERVE
        return max(0, int(raw))
    except Exception:
        return _DEFAULT_RESERVE


class ShopSkill(BaseSkill):
    # Any of these cls prove we're inside the shop.
    _SHOP_PAGE_CLS = [
        UC.SHOP_SELECT_ALL, UC.SHOP_SELECT_ALL_GREY, UC.SHOP_ALL_SELECTED,
        UC.SHOP_BUY, UC.SHOP_BUY_SELECTED, UC.CURRENCY, UC.CURRENCY_SEL,
        UC.SHOP_TAB_CREDIT, UC.SHOP_TAB_CREDIT_SEL,
    ]

    def __init__(self):
        super().__init__("Shop")
        self.max_ticks = 60
        self._reserve = _DEFAULT_RESERVE
        self._init_state()

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._select_clicks: int = 0
        self._buy_clicks: int = 0
        self._balance: Optional[int] = None   # credit balance read on the GRID
        self._purchased: bool = False
        self._budget_logged: bool = False

    def reset(self) -> None:
        super().reset()
        self._init_state()
        self._reserve = _load_shop_reserve()
        self.log(f"shop credit reserve = {self._reserve:,}")

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0

    # ── helpers ──────────────────────────────────────────────────────────
    def _on_shop(self, screen: ScreenState) -> bool:
        return self.find_cls(screen, self._SHOP_PAGE_CLS, conf=0.25) is not None

    def _confirm_dialog(self, screen: ScreenState) -> Optional[YoloBox]:
        confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_DIALOG_BAND)
        cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF, region=_DIALOG_BAND)
        if confirm is not None and cancel is not None:
            return confirm
        return None

    def _capture_balance(self, screen: ScreenState) -> None:
        """Read + cache the credit balance from the TOP-BAR — must be on the GRID
        view (top bar visible; the confirm dialog dims it + has no currency cls).

        ★ The shop screen has MANY 信用点 (TOPBAR_CREDIT) boxes — every item's
        price label is one (cy≈0.44/0.79) PLUS the top-bar balance (cy≈0.03).
        read_count anchors on the highest-conf box → a GRID PRICE, not the
        balance (live 2026-06-02: read 40000 instead of ~141M → wrongly
        cancelled). So we anchor on the TOP-BAR 信用点 only (cy<0.10) and OCR the
        digit strip to its right (verified: span 0.11 → 141,602,561)."""
        if self._balance is not None or screen.frame is None:
            return
        # Prefer a CLEAN ADB frame (2026-06-11): the in-shop top bar read on the
        # live frame returns None/truncated → budget unverifiable → cancel,
        # which blocked every purchase. The clean frame reads it correctly.
        try:
            from brain.pipeline import _read_topbar_clean
            cr = _read_topbar_clean(UC.TOPBAR_CREDIT)
            if cr is not None:
                self._balance = cr
                self.log(f"shop credit balance (clean frame) = {self._balance:,}")
                return
        except Exception:
            pass
        icon = self.find_cls(screen, UC.TOPBAR_CREDIT, conf=_CLS_CONF,
                             region=(0.0, 0.0, 1.0, 0.10))
        if icon is None:
            return
        try:
            from brain.pipeline import run_digit_ocr, parse_count
        except Exception:
            return
        x1 = min(1.0, icon.x2 + 0.003)
        x2 = min(1.0, x1 + 0.115)
        raw = run_digit_ocr(screen.frame, (x1, icon.y1 - 0.012, x2, icon.y2 + 0.012))
        res = parse_count(raw)
        if res is not None and res[0] is not None:
            self._balance = res[0]
            self.log(f"shop credit balance (top-bar) = {self._balance:,} (raw {raw!r})")

    def _balance_from_snapshot(self) -> None:
        """Fallback: the shop grid's top-bar 信用点 cls is FLAKY (live 2026-06-09:
        missed on every grid frame → balance=None → wrongly cancelled). The
        LOBBY snapshot read seconds earlier is authoritative — use it when the
        in-shop read failed and the snapshot is fresh (<10 min)."""
        if self._balance is not None:
            return
        try:
            import time as _t
            from brain.pipeline import get_resource_snapshot
            snap = get_resource_snapshot()
            cr, ts = snap.get("credits"), snap.get("ts", 0.0)
            if cr is not None and (_t.time() - ts) < 600:
                self._balance = int(cr)
                self.log(f"balance from LOBBY snapshot = {self._balance:,} (in-shop read failed)")
        except Exception:
            pass

    def _affordable(self) -> bool:
        """Buy only when the grid-read balance stays above reserve even after a
        worst-case purchase (一般-tab totals observed ~3M; 20M ceiling is safe).
        Balance unread → fail-safe cancel (never blind-buy)."""
        if self._balance is None:
            return False
        return self._balance >= (self._reserve + _ASSUMED_MAX_TOTAL)

    # ── tick ────────────────────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log(f"timeout (purchased={self._purchased})")
            return action_done(f"shop timeout (purchased={self._purchased})")

        # Global: purchase reward popup → dismiss via continue / header.
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
        if cont is not None:
            return action_click_box(cont, "dismiss reward via continue")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if got is not None:
            return action_click_box(got, "dismiss purchase reward")

        if screen.is_loading():
            return action_wait(700, "shop loading")

        if self.sub_state == "":
            self._goto("enter")

        handler = {
            "enter": self._enter,
            "select": self._select,
            "buy": self._buy,
            "confirm": self._confirm,
            "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "shop unknown state")
        return handler(screen)

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        # ⛔ pyroxene-tab guard: never operate on the 青辉石 shop.
        if self.find_cls(screen, UC.SHOP_TAB_PYROXENE_SEL, conf=_CLS_CONF) is not None:
            self.log("⛔ on 青辉石商店 tab — never buy pyroxene, exiting")
            self._goto("exit")
            return action_wait(300, "pyroxene tab → exit")

        if self._on_shop(screen):
            self.log("inside shop (一般 tab) → select")
            self._goto("select")
            return action_wait(400, "entered shop")

        if screen.is_lobby():
            act = self.click_cls(screen, UC.NAV_SHOP, "open shop", conf=_CLS_CONF)
            if act is not None:
                return act
            self.log("on lobby but no 商店入口 — YOLO gap; waiting")
            return action_wait(400, "waiting for 商店入口 cls")

        if self._phase_ticks > _ENTER_MAX:
            return action_done("could not reach shop")
        if len(screen.yolo_boxes or []) < 2:
            return action_wait(700, "no UI detected, likely loading")
        return action_back("shop: recover toward lobby")

    def _select(self, screen: ScreenState) -> Dict[str, Any]:
        if self.find_cls(screen, UC.SHOP_TAB_PYROXENE_SEL, conf=_CLS_CONF) is not None:
            self._goto("exit")
            return action_wait(300, "pyroxene tab → exit")

        if not self._on_shop(screen):
            if screen.is_lobby():
                self._goto("enter")
                return action_wait(300, "select: back on lobby")
            if self._phase_ticks > _SELECT_MAX:
                self._goto("exit")
                return action_wait(300, "select lost shop → exit")
            return action_wait(400, "waiting for shop UI (select)")

        # Read the credit balance HERE (grid view, top bar visible) — the
        # confirm dialog dims the top bar so it can't be read there.
        self._capture_balance(screen)

        # Nothing to buy / already bought today.
        if self.find_cls(screen, UC.SHOP_SELECT_ALL_GREY, conf=_CLS_CONF) is not None:
            self.log("全部选择灰 → nothing to buy / done, exiting")
            self._goto("exit")
            return action_wait(300, "shop nothing to select")

        # Already selected → go buy. v8 reality (probed 2026-06-11):
        #  · SHOP_ALL_SELECTED (cls402 已全部选择) has ZERO training samples →
        #    never fires.
        #  · SHOP_BUY_SELECTED (cls450 选择购买) flickers out live.
        #  · BUT the checked select-all box renders a GREEN_CHECK (绿勾) right
        #    at the checkbox (≈0.878,0.12), and the grid fills with 绿勾.
        # So "selected" = a 绿勾 in the select-all checkbox zone, OR the buy
        # button present (either cls). The actual spend is still gated by the
        # _confirm credit-reserve check, so advancing here is money-safe.
        sel_checked = self.find_cls(
            screen, UC.GREEN_CHECK, conf=0.35, region=(0.84, 0.06, 0.91, 0.17)
        )
        if (sel_checked is not None
                or self.find_cls(screen, UC.SHOP_BUY_SELECTED, conf=_CLS_CONF) is not None
                or self.find_cls(screen, UC.SHOP_ALL_SELECTED, conf=_CLS_CONF) is not None):
            self._goto("buy")
            return action_wait(250, "items selected (绿勾/buy cls) → buy")

        sel = self.find_cls(screen, UC.SHOP_SELECT_ALL, conf=_CLS_CONF)
        if sel is not None:
            # Pace retries (稳定规则 2026-06-11): give the toggle a beat to
            # render before judging the click dropped.
            if self._phase_ticks % 3 != 1:
                return action_wait(500, "select-all clicked — settling")
            self._select_clicks += 1
            # v8 boxes only the 全部選擇 LABEL TEXT (x1≈0.902); the real
            # hit-area is the checkbox square LEFT of it, centered ≈0.875
            # (zoom-measured 2026-06-11 after text-center & x1-0.013 both
            # missed). Offset 0.027 left of the label's x1 lands on it.
            cb_x = max(0.0, sel.x1 - 0.027)
            self.log(f"select all (checkbox @ {cb_x:.3f},{sel.cy:.3f})")
            return action_click(cb_x, sel.cy, "select all items")

        if self._phase_ticks > _SELECT_MAX:
            self.log("no 全部选择 cls — YOLO gap; exiting")
            self._goto("exit")
            return action_wait(300, "no select-all cls")
        return action_wait(400, "waiting for 全部选择 cls")

    def _buy(self, screen: ScreenState) -> Dict[str, Any]:
        # ⛔ Pyroxene-tab guard (deep-dive C8, 2026-06-09): the tab was only
        # checked in _enter/_select — if the view drifted onto the 青辉石 tab
        # by the time we're buying, a confirm here would spend pyroxene.
        if self.find_cls(screen, UC.SHOP_TAB_PYROXENE_SEL, conf=_CLS_CONF) is not None:
            self.log("⛔ pyroxene tab selected at buy stage — abort shop")
            self._goto("exit")
            return action_wait(300, "pyroxene tab at buy → exit")
        # Confirm dialog already up → budget decision.
        if self._confirm_dialog(screen) is not None:
            self._goto("confirm")
            return action_wait(200, "confirm dialog up → budget check")

        if not self._on_shop(screen):
            if screen.is_lobby():
                self._goto("enter")
                return action_wait(300, "buy: back on lobby")
            if self._phase_ticks > _BUY_MAX:
                self._goto("exit")
                return action_wait(300, "buy lost shop → exit")
            return action_wait(400, "waiting for shop UI (buy)")

        # The 選擇購買 button: cls450 (选择购买) when v8 sees it, else the same
        # yellow button gets MISCLASSIFIED as 任务开始 (probed 2026-06-11:
        # 选择购买 absent, 任务开始@0.91,0.92 c0.63 = the buy button). Accept
        # either in the bottom-right shop strip. Money-safe: the confirm dialog
        # still gates the spend on the credit-reserve check (never pyroxene).
        buy = self.find_cls(screen, UC.SHOP_BUY_SELECTED, conf=_CLS_CONF)
        if buy is None:
            buy = self.find_cls(screen, UC.TASK_START, conf=0.30,
                                region=(0.80, 0.86, 0.99, 0.98))
        if buy is not None:
            if self._buy_clicks >= 4:
                self.log("buy button clicked 4x, no dialog — exiting")
                self._goto("exit")
                return action_wait(300, "buy stuck → exit")
            # Pace (稳定规则): the confirm dialog takes a beat to render.
            if self._phase_ticks % 3 != 1 and self._buy_clicks > 0:
                return action_wait(450, "buy clicked — settling for dialog")
            self._buy_clicks += 1
            self.log(f"click 選擇購買 (#{self._buy_clicks}, cls={buy.cls_name})")
            return action_click_box(buy, "buy selected items")

        # No batch-buy button AND not selected — re-select or bail. Only loop
        # back to select if we are NOT already selected (绿勾 absent), else the
        # buy button is just flickering — wait for it.
        sel_checked = self.find_cls(screen, UC.GREEN_CHECK, conf=0.35,
                                    region=(0.84, 0.06, 0.91, 0.17))
        if sel_checked is None and self.find_cls(screen, UC.SHOP_SELECT_ALL, conf=_CLS_CONF) is not None:
            self._goto("select")
            return action_wait(250, "not selected → back to select")
        if self._phase_ticks > _BUY_MAX:
            self._goto("exit")
            return action_wait(300, "no buy cls → exit")
        return action_wait(400, "waiting for 选择购买 cls")

    def _confirm(self, screen: ScreenState) -> Dict[str, Any]:
        # ⛔ Pyroxene-tab guard (deep-dive r2 C4): _affordable() only checks the
        # CREDIT balance — on the 青辉石 tab it would happily "afford" a
        # pyroxene purchase. Never confirm while that tab is selected.
        if self.find_cls(screen, UC.SHOP_TAB_PYROXENE_SEL, conf=_CLS_CONF) is not None:
            self.log("⛔ pyroxene tab at confirm — cancelling, never buy")
            cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF, region=_DIALOG_BAND)
            self._goto("exit")
            if cancel is not None:
                return action_click_box(cancel, "cancel (pyroxene tab)")
            return action_back("cancel (pyroxene tab, ESC)")
        confirm = self._confirm_dialog(screen)
        if confirm is None:
            # Dialog gone — purchased (reward) or dismissed → exit.
            if self._phase_ticks > _CONFIRM_MAX:
                self._goto("exit")
                return action_wait(300, "confirm dialog gone → exit")
            return action_wait(300, "waiting for confirm dialog")

        # Try one more balance read in case the dialog left the top bar visible
        # (mostly it won't — the grid-view read in _select is the real source).
        self._capture_balance(screen)
        self._balance_from_snapshot()  # lobby-snapshot fallback (flaky in-shop cls)
        if not self._budget_logged:
            self.log(f"budget: balance={self._balance} reserve={self._reserve:,} "
                     f"ceiling={_ASSUMED_MAX_TOTAL:,}")
            self._budget_logged = True

        if self._affordable():
            self.log("affordable (balance ≥ reserve+ceiling) → confirm (credits)")
            self._purchased = True
            self._goto("exit")
            return action_click_box(confirm, "confirm shop purchase (within budget)")

        # Not affordable / can't verify → cancel (never overspend / blind-buy).
        self.log("⛔ not within budget / unverifiable → cancel")
        cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF, region=_DIALOG_BAND)
        self._goto("exit")
        if cancel is not None:
            return action_click_box(cancel, "cancel purchase (budget)")
        return action_back("cancel purchase (budget, ESC)")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if self.detect_screen_yolo(screen) == "Lobby":
            status = "purchased" if self._purchased else "no purchase"
            self.log(f"done ({status})")
            return action_done(f"shop complete ({status})")
        if self._phase_ticks > _EXIT_MAX:
            return action_done("shop exit timeout")
        home = self.find_cls(screen, UC.BTN_HOME, conf=_CLS_CONF)
        if home is not None:
            return action_click_box(home, "shop exit: home button")
        back = self.find_cls(screen, UC.BTN_BACK, conf=_CLS_CONF)
        if back is not None:
            return action_click_box(back, "shop exit: back button")
        return action_back("shop exit: ESC toward lobby")
