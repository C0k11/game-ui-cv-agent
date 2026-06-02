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
    action_click_box, action_wait, action_back, action_done,
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

    def _read_budget(self, screen: ScreenState) -> Tuple[Optional[int], Optional[int]]:
        """(balance, total). Balance from the reliable top-bar credit; total is
        a best-effort read of the dialog's 總購買價格 via 货币数量显示区域."""
        balance = None
        bres = self.read_count(screen, UC.TOPBAR_CREDIT, side="right", span=0.12)
        if bres is not None:
            balance = bres[0]

        total = None
        if screen.frame is not None:
            try:
                from brain.pipeline import run_digit_ocr, parse_count
                # 货币数量显示区域 boxes inside the dialog (exclude the top bar).
                areas = self.find_all_cls(
                    screen, UC.CURRENCY_QTY_AREA, conf=_CLS_CONF,
                    region=(0.30, 0.30, 0.95, 0.78),
                )
                if areas:
                    # held above, total below → the lowest box is 總購買價格.
                    tbox = max(areas, key=lambda b: b.cy)
                    raw = run_digit_ocr(
                        screen.frame, (tbox.x1, tbox.y1, tbox.x2, tbox.y2)
                    )
                    res = parse_count(raw)
                    if res is not None:
                        total = res[0]
            except Exception:
                pass
        return balance, total

    def _affordable(self, balance: Optional[int], total: Optional[int]) -> bool:
        if balance is None:
            return False  # can't verify → fail safe (never blind-buy)
        if total is not None:
            return (balance - total) >= self._reserve
        # Total unreadable → only buy when balance is comfortably above reserve.
        return balance >= (self._reserve + _ASSUMED_MAX_TOTAL)

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

        # Nothing to buy / already bought today.
        if self.find_cls(screen, UC.SHOP_SELECT_ALL_GREY, conf=_CLS_CONF) is not None:
            self.log("全部选择灰 → nothing to buy / done, exiting")
            self._goto("exit")
            return action_wait(300, "shop nothing to select")

        # Already selected → go buy.
        if (self.find_cls(screen, UC.SHOP_BUY_SELECTED, conf=_CLS_CONF) is not None
                or self.find_cls(screen, UC.SHOP_ALL_SELECTED, conf=_CLS_CONF) is not None):
            self._goto("buy")
            return action_wait(250, "items selected → buy")

        sel = self.find_cls(screen, UC.SHOP_SELECT_ALL, conf=_CLS_CONF)
        if sel is not None:
            self._select_clicks += 1
            self.log("select all (YOLO 全部选择)")
            return action_click_box(sel, "select all items")

        if self._phase_ticks > _SELECT_MAX:
            self.log("no 全部选择 cls — YOLO gap; exiting")
            self._goto("exit")
            return action_wait(300, "no select-all cls")
        return action_wait(400, "waiting for 全部选择 cls")

    def _buy(self, screen: ScreenState) -> Dict[str, Any]:
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

        buy = self.find_cls(screen, UC.SHOP_BUY_SELECTED, conf=_CLS_CONF)
        if buy is not None:
            if self._buy_clicks >= 4:
                self.log("选择购买 clicked 4x, no dialog — exiting")
                self._goto("exit")
                return action_wait(300, "buy stuck → exit")
            self._buy_clicks += 1
            self.log(f"click 选择购买 (#{self._buy_clicks})")
            return action_click_box(buy, "buy selected items")

        # No batch-buy button — re-select or bail.
        if self.find_cls(screen, UC.SHOP_SELECT_ALL, conf=_CLS_CONF) is not None:
            self._goto("select")
            return action_wait(250, "no 选择购买 → back to select")
        if self._phase_ticks > _BUY_MAX:
            self._goto("exit")
            return action_wait(300, "no buy cls → exit")
        return action_wait(400, "waiting for 选择购买 cls")

    def _confirm(self, screen: ScreenState) -> Dict[str, Any]:
        confirm = self._confirm_dialog(screen)
        if confirm is None:
            # Dialog gone — purchased (reward) or dismissed → exit.
            if self._phase_ticks > _CONFIRM_MAX:
                self._goto("exit")
                return action_wait(300, "confirm dialog gone → exit")
            return action_wait(300, "waiting for confirm dialog")

        balance, total = self._read_budget(screen)
        if not self._budget_logged:
            self.log(f"budget: balance={balance} total={total} reserve={self._reserve:,}")
            self._budget_logged = True

        if self._affordable(balance, total):
            self.log("affordable (≥reserve) → confirm purchase (credits)")
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
