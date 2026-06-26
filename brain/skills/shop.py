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
        # shop统一 (2026-06-16): True = 买完信用点留在店内(done-in-shop), 由紧随的
        # arena_shop 在同一次访问继续切战术大赛 tab。daily 编排 shop→arena_shop 永远
        # 相邻, 故默认 True。单跑/无 arena_shop 跟随时设 False 则正常退大厅。
        self.chain_in_shop = True
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

    def _read_balance_longest(self, frame, region):
        """OCR a region → the LONGEST contiguous digit run (commas stripped),
        as int, or None. Bypasses run_digit_ocr's comma-group early-return,
        which on a live OCR fragment ('28,096' + '458') returned a short
        number that got reserve-rejected (2026-06-12 shop never bought). Only
        accepts >= 1M (real credit balances are millions; shorter = truncated /
        garbage). Logs the raw on miss so the next failure is debuggable."""
        if frame is None:
            return None
        try:
            import re as _re
            import cv2
            from brain.pipeline import _get_ocr
            h, w = frame.shape[:2]
            x1, y1 = int(region[0]*w), int(region[1]*h)
            x2, y2 = int(region[2]*w), int(region[3]*h)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                return None
            if crop.shape[0] < 40:
                sc = 40.0 / crop.shape[0]
                crop = cv2.resize(crop, (int(crop.shape[1]*sc), 40),
                                  interpolation=cv2.INTER_CUBIC)
            result, _ = _get_ocr()(crop)
            if not result:
                return None
            try:
                result = sorted(result, key=lambda ln: min(p[0] for p in ln[0]))
            except Exception:
                pass
            raw = "".join(line[1] for line in result)
            digits = _re.sub(r"[^0-9]", "", raw.replace(",", "").replace("，", ""))
            if not digits:
                return None
            val = int(digits)
            if val >= 1_000_000:
                return val
            if not getattr(self, "_bal_miss_logged", False):
                self.log(f"持有數量 read too small: raw={raw!r} digits={digits!r}")
                self._bal_miss_logged = True
            return None
        except Exception:
            return None

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
        return self.nav_home(screen, "shop recover")

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
            # 今日已售罄 robustness (2026-06-16 实测): 当日商品全部买光后 ITEM 卡变灰,
            # 但「全部選擇」按钮本身不变灰 → SHOP_SELECT_ALL_GREY 检测落空 → 旧码无限
            # 点全部選擇(售罄商品选不中→无绿勾→不前进)直到 max_ticks。绿勾检测在上面,
            # 走到这=点了仍无绿勾。点≥3次仍无绿勾 = 无可选 = 售罄 → exit(钱安全, 没选没买)。
            if self._select_clicks >= 3:
                self.log("全部選擇 ×3 无绿勾 → 今日已售罄 nothing to buy → exit")
                self._goto("exit")
                return action_wait(300, "shop sold out (3x select no 绿勾) → exit")
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

        # ★ PRIMARY budget source (2026-06-12): the confirm dialog itself shows
        # 持有數量 (balance) and 總購買價格 (total) in large clear text — read
        # BOTH from fixed dialog regions and do the exact check. This replaces
        # the topbar read (vote-flaky: '25,583,379' OCR'd unstably → None →
        # false cancel, live t0163) and the over-conservative 20M ceiling
        # (balance 25.58M vs reserve5M+20M gate = razor margin).
        # The dialog's 持有數量 is white-on-light, clear, and IMMUNE to the
        # topbar's left-truncation (live 2026-06-12: topbar 28,096,458 OCR'd
        # as 96,458 → shop never bought). It IS the topbar balance. Read it
        # from the fixed region — but only AFTER the popup finishes rendering,
        # so retry on the in-screen (never-vanishing) dialog instead of bailing
        # to the truncated topbar on the first miss. Use the LONGEST-digit-run
        # parse (NOT run_digit_ocr's comma logic, which returned a short
        # fragment live → rejected → 8 retries → cancel, 2026-06-12).
        # 持有數量 region: 4K 帧上 RapidOCR det 对 7位逗号数字('7,432,788')分组易碎,
        # 旧区(0.26,0.63,0.47,0.71)会丢最左组只读出 '432788'<1M → 误 fail-closed 取消
        # (2026-06-26 实测+多agent帧取证)。垂直留白放宽到 (0.61,0.73) 后 det 一次读出整串
        # '7,432,788' conf0.895。读小=取消(fail-closed方向), 改区不引入 fail-open。
        bal = self._read_balance_longest(screen.frame, (0.28, 0.61, 0.49, 0.73))
        if bal is None:
            if self._phase_ticks <= 8:
                return action_wait(450, f"confirm dialog up, re-reading 持有數量 "
                                        f"({self._phase_ticks}/8)")
            # Exhausted retries → fail-CLOSED cancel (never blind-buy on an
            # unread balance). Don't fall through to the truncated topbar.
            self.log("⛔ 持有數量 unreadable after retries → cancel (fail-closed)")
            cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF, region=_DIALOG_BAND)
            self._goto("exit")
            if cancel is not None:
                return action_click_box(cancel, "cancel purchase (unread balance)")
            return action_back("cancel purchase (unread, ESC)")

        # 總購買價格 sits on a DARK band — det fragments the digits beyond
        # repair (offline 2026-06-12: raw/invert/otsu/threshold all chop
        # '3,121,500' into overlapping pieces). Use the price when it reads,
        # else a realistic ceiling: 一般-tab full daily stock is a FIXED ~3.1M
        # basket — 8M = 2.5x observed, far saner than the old 20M.
        from brain.pipeline import run_digit_ocr, parse_count
        dlg_tot = parse_count(run_digit_ocr(screen.frame, (0.59, 0.63, 0.83, 0.71)))
        _DIALOG_CEILING = 8_000_000
        tot = dlg_tot[0] if (dlg_tot and dlg_tot[0] and dlg_tot[0] >= 100_000) else _DIALOG_CEILING
        self.log(f"dialog budget: 持有={bal:,} 總價={tot:,} reserve={self._reserve:,}")
        if bal - tot >= self._reserve:
            self._purchased = True
            self._goto("exit")
            return action_click_box(confirm, "confirm shop purchase (within budget)")
        self.log("⛔ dialog budget: balance-total < reserve → cancel")
        cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF, region=_DIALOG_BAND)
        self._goto("exit")
        if cancel is not None:
            return action_click_box(cancel, "cancel purchase (budget)")
        return action_back("cancel purchase (budget, ESC)")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        # ── shop统一 (用户 2026-06-16: 信用点商店 + 战术大赛商店是同一个商店, 左栏
        # 下滑切 tab 即可, 没必要退大厅再进) ──────────────────────────────────
        # 信用点(一般 tab)买完后, 若仍在商店网格内, 就 done-IN-SHOP(不 nav_home),
        # 让紧随的 arena_shop 在同一次访问里继续(arena_shop._enter 已支持 _on_shop
        # → 直接 swipe 到战术大赛 tab, 无需从大厅重进)。复用 arena_shop 全部钱防线。
        # 仅在确实回到大厅(已被别处导走)或超时才正常收尾。
        if self.chain_in_shop and self._on_shop(screen):
            status = "purchased" if self._purchased else "no purchase"
            self.log(f"done ON shop grid ({status}) → 留在店内, arena_shop 接力切战术大赛 tab")
            return action_done(f"shop complete in-shop ({status}) → arena tab next")
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
        return self.nav_home(screen, "shop exit")
