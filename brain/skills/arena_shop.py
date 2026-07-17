# -*- coding: utf-8 -*-
"""ArenaShopSkill — 战术大赛商店买体力: spend 战术大赛货币 (a non-premium
battle currency) on the two daily energy drinks (下级 30AP / 一般 60AP) to
refill AP.

Spec = live walk 2026-06-12 (data/_arena_shop_probe_log.md, _explore_012..018).
Steps 1-5 (navigate → arena tab → select drinks → 選擇購買 appears) were live-
verified on v9. Step 6 (confirm dialog) was NOT explored — so the confirm path
is built fail-CLOSED and reuses the generic 通知-dialog button classes; it needs
ONE live calibration pass before this skill joins the unattended chain.

⛔ MONEY SAFETY (three layers, none optional):
  1. Tab lock: select / buy ONLY while 战术大赛商店已选择 (470) is on screen.
     Wrong tab → never click a buy button.
  2. Pyroxene guard: any 青辉石 / 购买青辉石 in the confirm-dialog body → CANCEL.
     (战术大赛货币 ≠ 青辉石; this is the hard premium-currency firewall.)
  3. Balance gate: read 战术大赛商店货币 (471) topbar balance, compare to the
     summed item price; unreadable or insufficient → DON'T buy (fail-closed —
     skipping a non-premium drink costs nothing).

State machine
-------------
enter    lobby → 商店入口 → shop_grid.
locate   shop_grid → swipe the left tab column down to reveal 战术大赛商店 (469)
         → tap → verify 战术大赛商店已选择 (470). Capped swipes.
select   arena tab → read balance → tap each energy drink (472/473) not yet
         绿勾-checked → all checked + 选择购买 present → buy.
buy      战术大赛商店已选择 still on + 选择购买 (450) → click it → confirm.
confirm  通知 dialog → pyroxene guard → 确认键. No dialog / grey confirm → exit.
exit     回大厅 / X → lobby → done.

NOT in DEFAULT_SKILLS — register + run via sub_only for the live calibration of
the confirm step; integrate into the harvest (after shop) once verified.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click, action_click_box, action_wait, action_back, action_done,
    action_swipe,
)
from brain.skills import ui_classes as UC

try:
    from brain.screens import classify_screen
except Exception:  # screens optional at import time
    classify_screen = None

_CLS_CONF = 0.30
# Daily energy-drink prices in 战术大赛货币 (fixed shop prices, probe 2026-06-12).
_PRICES = {UC.ENERGY_DRINK_LOW: 15, UC.ENERGY_DRINK_MID: 30}

# Fixed-pos fallbacks from the walk (cls can flicker on event-skin lobbies).
_POS_SHOP_ENTRY = (0.621, 0.953)     # 商店入口 on lobby
_POS_HOME = (0.965, 0.033)           # 回大厅 top-right
# 战术大赛 tab fixed pos in the left column (cls 469/470 center, measured on 238
# live frames cx0.069 cy0.395). cls 469 (UNSELECTED tab) is starved (27 train
# instances vs 470's 534) → YOLO misses it → click this fixed pos to switch tabs.
_POS_ARENA_TAB = (0.068, 0.395)
# Left tab column ONE swipe to reveal 战术大赛商店 (shop opens on the 一般/credit
# tab; 战术大赛 sits below the fold). Magnitude ~0.40 — live 2026-06-14: one swipe
# 0.75→0.35 brings 战术大赛 to cy0.396 where v11 detects cls469 @0.96. user: 滑一下
# 就行别猛滑 → single swipe, this distance lands it right.
_SWIPE_FROM = (0.05, 0.75)
_SWIPE_TO = (0.05, 0.35)

# Confirm-dialog body region (pyroxene guard scan) + button band.
_DIALOG_BODY = (0.15, 0.12, 0.85, 0.78)
_DIALOG_BAND = (0.28, 0.78, 0.72, 0.95)

_ENTER_MAX = 20
_LOCATE_MAX = 24          # includes several swipes
_SELECT_MAX = 24
_BUY_MAX = 12
_CONFIRM_MAX = 12
_EXIT_MAX = 14
# Tap-once selection (live 2026-06-13: the per-tick 绿勾 re-check + the gate's
# cap re-click TOGGLED 下级 on/off forever — never reached 一般). Tap a drink
# ONCE, then WAIT (not re-tap) for its 绿勾 to render; re-tap only if the tap was
# genuinely lost. Never tap a card already showing 绿勾 (that would deselect it).
_RENDER_WAIT = 3          # _phase_ticks to wait for a tapped drink's 绿勾
_MAX_TAP = 2              # per-drink tap cap (lost-tap recovery, never spam)
# Both drink cards may fade in over a couple frames after the tab switch — scan a
# short window before committing _want so 一般 isn't missed (live 2026-06-14: only
# 下级 bought because _want was committed on the first in-tab frame).
_DRINK_SCAN = 3           # _phase_ticks to accumulate detected drinks
_SWIPE_SETTLE = 2         # _phase_ticks to let the locate swipe's scroll settle
                          # before the fixed-pos fallback fires (audit 2026-06-14:
                          # firing it pre-settle can hit the wrong tab — scroll
                          # may still be at the pre-swipe position)


class ArenaShopSkill(BaseSkill):
    def should_run(self, screen: ScreenState) -> bool:
        # Always travels; bails inside if the arena tab can't be found or the
        # balance can't fund a drink. (Balance is unreadable from the lobby.)
        return True

    def __init__(self):
        super().__init__("ArenaShop")
        self.max_ticks = 110
        self._init_state()

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._swipes: int = 0
        self._swipe_tick: int = -10          # _phase_ticks of the locate swipe
        self._entry_clicked: bool = False    # clicked 商店入口 → wait for load, no back
        self._balance: Optional[int] = None
        self._drink_seen: set = set()        # drinks detected over the scan window
        self._want: Dict[str, int] = {}      # cls_name → price, what we'll buy
        self._selected: set = set()          # cls_names confirmed green-checked
        self._skipped: set = set()           # cls_names we gave up selecting
        self._tap_count: Dict[str, int] = {} # cls_name → times tapped
        self._last_tap_tick: Dict[str, int] = {}  # per-drink last-tap tick (render-wait)
        self._purchased: bool = False

    def reset(self) -> None:
        super().reset()
        self._init_state()

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0

    # ── helpers ──────────────────────────────────────────────────────────
    def _in_arena_tab(self, screen: ScreenState) -> bool:
        return self.find_cls(screen, UC.ARENA_SHOP_TAB_SEL, conf=_CLS_CONF) is not None

    def _on_shop(self, screen: ScreenState) -> bool:
        if self.find_cls(screen, [UC.ARENA_SHOP_TAB, UC.ARENA_SHOP_TAB_SEL,
                                  UC.SHOP_TAB_CREDIT, UC.SHOP_TAB_CREDIT_SEL],
                         conf=0.25) is not None:
            return True
        if classify_screen is not None:
            try:
                sid, _, _ = classify_screen(screen.yolo_boxes or [])
                return sid == "shop_grid"
            except Exception:
                pass
        return False

    def _read_arena_balance(self, screen: ScreenState) -> Optional[int]:
        """Topbar 战术大赛商店货币 balance (the icon with cy<0.13; price-tag
        icons sit lower and are excluded). Digits to the right of the icon."""
        icon = self.find_cls(screen, UC.ARENA_SHOP_CURRENCY, conf=0.30,
                             region=(0.30, 0.0, 0.80, 0.13))
        if icon is None or screen.frame is None:
            return None
        try:
            from brain.pipeline import run_digit_ocr, parse_count
        except Exception:
            return None
        bw, bh = icon.x2 - icon.x1, icon.y2 - icon.y1
        # span: spec calibration — start a touch left of the icon's right edge,
        # run ~6 icon-widths right; 3x upscale handled inside run_digit_ocr.
        x1 = max(0.0, icon.x2 - bw * 0.2)
        x2 = min(1.0, icon.x2 + bw * 6.0)
        y1 = max(0.0, icon.y1 - bh * 0.4)
        y2 = min(1.0, icon.y2 + bh * 0.4)
        res = parse_count(run_digit_ocr(screen.frame, (x1, y1, x2, y2)))
        if res is None or res[0] is None:
            return None
        # sanity: arena currency tops out in the low thousands; reject 6+ digit
        # OCR fusions (a price-tag bleed-through) and anything below a price.
        val = res[0]
        if val < 1 or val > 99999:
            return None
        return val

    # ── tick ─────────────────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1
        if self.ticks >= self.max_ticks:
            self.log(f"timeout (purchased={self._purchased})")
            return action_done("arena_shop timeout")
        if screen.is_loading():
            return action_wait(700, "arena_shop loading")
        if self.sub_state == "":
            self._goto("enter")
        handler = {
            "enter": self._enter, "locate": self._locate, "select": self._select,
            "buy": self._buy, "confirm": self._confirm, "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "arena_shop unknown state")
        return handler(screen)

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._on_shop(screen):
            self._goto("locate")
            return action_wait(300, "in shop → locate arena tab")
        if screen.is_lobby():
            entry = self.find_cls(screen, UC.NAV_SHOP, conf=_CLS_CONF)
            if entry is not None:
                self._entry_clicked = True
                return action_click_box(entry, "open shop (YOLO 商店入口)")
            self._entry_clicked = True
            return action_click(*_POS_SHOP_ENTRY, "open shop (fixed pos)")
        if self._phase_ticks > _ENTER_MAX:
            return action_done("could not reach shop")
        if len(screen.yolo_boxes or []) < 2:
            return action_wait(600, "no UI — likely loading")
        # Not lobby, not shop, but UI present = the shop-opening transition. If we
        # already clicked the entry, WAIT for it to load — do NOT back-bounce
        # (user 2026-06-14: 转场期别瞎退; the back-press cancelled the transition →
        # bounced to lobby → re-click). Only recover-back if we never clicked entry.
        if self._entry_clicked:
            return action_wait(500, "等商店加载(已点入口, 不瞎退)")
        return self.nav_home(screen, "arena_shop recover")

    def _locate(self, screen: ScreenState) -> Dict[str, Any]:
        # 470 (已选择) on screen → we're on the arena tab. v11 detects 470 well
        # (534 train instances) so this is reliable.
        if self._in_arena_tab(screen):
            self.log("战术大赛商店 tab active → select")
            self._goto("select")
            return action_wait(300, "arena tab active")
        # Switch to the 战术大赛 tab. The tab is at a FIXED left-column position,
        # ALWAYS visible (the old swipe-below-fold assumption was wrong — measured
        # cy0.395, mid-screen). cls 469 (UNSELECTED tab) is starved (27 instances)
        # so YOLO usually misses it → prefer YOLO when it DOES fire, else click the
        # fixed pos. Event-driven: click once → 1s settle + _dedup hold wait for 470
        # to render → loop. No %3 pacing counter, no swipe (user 2026-06-14: 不靠计数
        # 器, locate 卡是代码 bug 不是模型 — v11 认得 470/471/472/473, 只 469 弱).
        # 469 (UNSELECTED tab). v11 detects it @0.96 WHEN VISIBLE (live 2026-06-14
        # — the "469 weak" theory was wrong: it's not weak, it's just OFF-SCREEN
        # because the shop opens on the 一般/credit tab with 战术大赛 below the fold).
        # So: YOLO-click it when present; if absent → swipe ONCE to reveal, then YOLO
        # picks it up next tick.
        tab = self.find_cls(screen, UC.ARENA_SHOP_TAB, conf=0.30)
        if tab is not None:
            return action_click_box(tab, "switch to 战术大赛商店 tab (YOLO 469)")
        # 469 not on screen → below the fold → ONE swipe (~0.40) reveals it.
        if self._swipes < 1:
            self._swipes += 1
            self._swipe_tick = self._phase_ticks
            self.log("战术大赛 tab 在 fold 下 → 下滑一下露出(单次)")
            return action_swipe(*_SWIPE_FROM, *_SWIPE_TO, duration_ms=500,
                                reason="reveal 战术大赛 tab (one swipe)")
        if self._phase_ticks > _LOCATE_MAX:
            self.log("战术大赛 tab unreachable after swipe → exit")
            self._goto("exit")
            return action_wait(300, "arena tab unreachable → exit")
        # Let the swipe's scroll settle before the blind fixed-pos fallback — YOLO
        # gets a few ticks to pick up 469 @0.96 (the common case), and the fixed pos
        # is only valid once the scroll has landed (audit 2026-06-14: pre-settle it
        # can hit the wrong tab).
        if self._phase_ticks - self._swipe_tick < _SWIPE_SETTLE:
            return action_wait(450, "等下滑落定再定位(防点错tab)")
        # If a credit/other shop tab is clearly selected, the scroll didn't land on
        # 战术大赛 → don't blind-tap the fixed pos; wait (the fixed pos is only the
        # right tab at the post-swipe scroll position).
        if self.find_cls(screen, [UC.SHOP_TAB_CREDIT_SEL], conf=0.30) is not None:
            return action_wait(450, "滚动态没对(信用点tab选中) → 等, 不盲点固定位")
        # swiped + settled + 469 still not detected → fixed-pos safety net (valid at
        # the post-swipe scroll position, measured 0.068,0.395).
        return action_click(*_POS_ARENA_TAB,
                            "switch to 战术大赛 tab (fixed pos — post-swipe safety)")

    def _green_on(self, screen: ScreenState, card: YoloBox) -> bool:
        """True if a 绿勾 marks THIS card selected. The check sits at the card's
        top-left corner (live: 下级 card x1≈0.76 → 绿勾 center ≈0.74,0.625)."""
        return self.find_cls(
            screen, UC.GREEN_CHECK, conf=0.35,
            region=(card.x1 - 0.05, card.y1 - 0.06, card.cx, card.cy),
        ) is not None

    def _select(self, screen: ScreenState) -> Dict[str, Any]:
        # ⛔ tab lock: only ever select while the arena tab is confirmed active.
        if not self._in_arena_tab(screen):
            if self._phase_ticks > _SELECT_MAX:
                self._goto("exit")
                return action_wait(300, "lost arena tab → exit")
            return action_wait(400, "re-confirming arena tab")

        # Decide what to buy ONCE: which drinks are present, summed price,
        # and whether the balance covers it (fail-closed).
        if not self._want:
            # Accumulate detected drinks over a short window — both cards fade in
            # over a couple frames after the tab switch, so a one-frame snapshot
            # misses 一般 (live 2026-06-14). Commit once both are seen OR the scan
            # window passes (handles a genuinely sold-out / single-drink day).
            self._drink_seen |= {n for n in (UC.ENERGY_DRINK_LOW, UC.ENERGY_DRINK_MID)
                                 if self.find_cls(screen, n, conf=_CLS_CONF) is not None}
            if len(self._drink_seen) < 2 and self._phase_ticks <= _DRINK_SCAN:
                return action_wait(350, f"扫描饮料中 ({len(self._drink_seen)}/2)")
            drinks = set(self._drink_seen)
            if not drinks:
                if self._phase_ticks > _SELECT_MAX:
                    self.log("no energy drinks detected → exit")
                    self._goto("exit")
                    return action_wait(300, "no drinks → exit")
                return action_wait(400, "waiting for energy-drink cls")
            bal = self._read_arena_balance(screen)
            self._balance = bal
            want = {n: _PRICES[n] for n in drinks}
            total = sum(want.values())
            if bal is None:
                self.log("⛔ 战术大赛货币 balance unreadable → skip (fail-closed)")
                self._goto("exit")
                return action_wait(300, "balance unreadable → exit")
            if bal < total:
                self.log(f"⛔ balance {bal} < total {total} → skip (can't afford)")
                self._goto("exit")
                return action_wait(300, "insufficient → exit")
            self._want = want
            self.log(f"buying {list(want)} total={total} 货币, balance={bal} — ok")

        # Tap each wanted drink ONCE; confirm via its 绿勾; NEVER re-tap a card
        # that already shows 绿勾 (that would deselect it — the toggle bug).
        for name in self._want:
            if name in self._selected:
                continue
            card = self.find_cls(screen, name, conf=_CLS_CONF)
            if card is None:
                continue
            if self._green_on(screen, card):          # already selected → done
                self._selected.add(name)
                self.log(f"{name} 已选中(绿勾)")
                continue
            tapped = self._tap_count.get(name, 0)
            if tapped > 0:
                # already tapped — WAIT for the 绿勾 to render, do NOT re-tap
                # (re-tapping mid-render toggles it back off → livelock). Per-drink
                # last-tap tick (audit 2026-06-14: a shared scalar let drink A's tap
                # time pollute drink B's render-wait window).
                if self._phase_ticks - self._last_tap_tick.get(name, -10) < _RENDER_WAIT:
                    return action_wait(450, f"{name} 已点 — 等绿勾渲染")
                if tapped >= _MAX_TAP:
                    # genuinely not selecting → give up THIS drink (under-buy is
                    # free; never spam-toggle). Move on to any other drink.
                    self.log(f"⚠️ {name} 点{tapped}次仍无绿勾 → 放弃该饮料")
                    self._skipped.add(name)
                    self._selected.add(name)   # treat as handled
                    continue
            self._tap_count[name] = tapped + 1
            self._last_tap_tick[name] = self._phase_ticks
            self.log(f"select {name} (tap {tapped + 1})")
            return action_click_box(card, f"select {name}")

        # ⛔ MONEY GUARD: count 绿勾 on screen. If MORE items are checked than the
        # drinks we wanted, an accidental 全部选择 happened → ABORT, never bulk-buy
        # (live 2026-06-13: bot bought 11 items / 469 货币 via 全部选择 path).
        greens = sum(1 for b in (screen.yolo_boxes or [])
                     if b.cls_name == UC.GREEN_CHECK and b.confidence >= 0.35)
        real_want = [n for n in self._want if n not in self._skipped]
        if greens > len(real_want):
            self.log(f"⛔ {greens} 绿勾 > {len(real_want)} 想买 → 疑似全选, 中止不买")
            self._goto("exit")
            return action_wait(300, "over-selected (全选?) → abort")
        if not real_want:
            self.log("无可选饮料 → exit")
            self._goto("exit")
            return action_wait(300, "nothing selectable → exit")

        # All wanted drinks selected, count sane → go buy when 选择购买 shows.
        if self.find_cls(screen, UC.SHOP_BUY_SELECTED, conf=_CLS_CONF) is not None:
            self._goto("buy")
            return action_wait(250, "drinks selected → buy")
        if self._phase_ticks > _SELECT_MAX:
            self._goto("exit")
            return action_wait(300, "选择购买 never appeared → exit")
        return action_wait(400, "waiting for 选择购买 after select")

    def _buy(self, screen: ScreenState) -> Dict[str, Any]:
        # ⛔ tab lock still required.
        if not self._in_arena_tab(screen) and self.find_cls(
                screen, UC.SHOP_BUY_SELECTED, conf=_CLS_CONF) is None:
            # dialog may have replaced the grid — only advance if a confirm
            # dialog is up; otherwise recover.
            if self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF,
                             region=_DIALOG_BAND) is not None:
                self._goto("confirm")
                return action_wait(200, "confirm dialog up")
            if self._phase_ticks > _BUY_MAX:
                self._goto("exit")
                return action_wait(300, "buy lost context → exit")
            return action_wait(400, "waiting in buy")
        buy = self.find_cls(screen, UC.SHOP_BUY_SELECTED, conf=_CLS_CONF)
        if buy is not None:
            self.log("click 选择购买")
            self._goto("confirm")
            return action_click_box(buy, "buy selected drinks")
        if self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF,
                         region=_DIALOG_BAND) is not None:
            self._goto("confirm")
            return action_wait(200, "confirm dialog up")
        if self._phase_ticks > _BUY_MAX:
            self._goto("exit")
            return action_wait(300, "no 选择购买 → exit")
        return action_wait(400, "waiting for 选择购买")

    def _confirm(self, screen: ScreenState) -> Dict[str, Any]:
        # ⛔ HARD pyroxene firewall: any premium-currency marker in the dialog
        # body → cancel, never confirm. (战术大赛货币 ≠ 青辉石.)
        pyro = self.find_cls(screen, [UC.SHOP_BUY_PYROXENE, "青辉石"],
                             conf=_CLS_CONF, region=_DIALOG_BODY)
        if pyro is not None:
            self.log("⛔ pyroxene in confirm dialog → CANCEL (never spend青辉石)")
            cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF,
                                   region=_DIALOG_BAND)
            self._goto("exit")
            if cancel is not None:
                return action_click_box(cancel, "cancel (pyroxene present)")
            return action_back("cancel (pyroxene, ESC)")

        confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF,
                                region=_DIALOG_BAND)
        if confirm is None:
            # grey (insufficient) confirm = can't afford after all → cancel out.
            grey = self.find_cls(screen, UC.BTN_CONFIRM_GREY, conf=_CLS_CONF,
                                 region=_DIALOG_BAND)
            if grey is not None:
                self.log("⛔ grey confirm (insufficient) → cancel")
                cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF)
                self._goto("exit")
                if cancel is not None:
                    return action_click_box(cancel, "cancel (grey confirm)")
                return action_back("cancel (grey, ESC)")
            if self._phase_ticks > _CONFIRM_MAX:
                self._goto("exit")
                return action_wait(300, "confirm dialog gone → exit")
            return action_wait(300, "waiting for confirm dialog")

        # ✅ WHITELIST firewall (fail-closed, user 铁律 + 2026-06-14 audit): a
        # confirm button is present, but only click it if this is POSITIVELY the
        # energy-drink dialog — require an energy drink (472/473) in the dialog
        # body. v11 detects 472 @0.96 in-body on the real confirm frame (471货币
        # sits at the bg tab edge cx0.03, NOT a reliable in-body anchor). Absent →
        # this is NOT our drink purchase (could be a mis-routed premium/other
        # dialog) → CANCEL. The old code was blacklist-only ("no pyroxene →
        # confirm") = fail-OPEN: a mis-detected premium dialog would get confirmed.
        drink = self.find_cls(screen, [UC.ENERGY_DRINK_LOW, UC.ENERGY_DRINK_MID],
                              conf=_CLS_CONF, region=_DIALOG_BODY)
        if drink is None:
            self.log("⛔ confirm 框内无饮料正向证据(472/473) → 取消(fail-closed 白名单)")
            cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF,
                                   region=_DIALOG_BAND)
            self._goto("exit")
            if cancel is not None:
                return action_click_box(cancel, "cancel (no drink anchor, fail-closed)")
            return action_back("cancel (no drink anchor, ESC)")

        # Budget gated in _select; currency non-premium; pyroxene firewall passed;
        # drink anchor present → confirm (spends 战术大赛货币 only).
        self.log(f"confirm purchase (战术大赛货币, balance was {self._balance})")
        self._purchased = True
        self._goto("exit")
        return action_click_box(confirm, "confirm energy-drink purchase")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log(f"done (purchased={self._purchased})")
            return action_done(f"arena_shop complete (purchased={self._purchased})")
        if self._phase_ticks > _EXIT_MAX:
            return action_done("arena_shop exit timeout")
        close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF,
                              region=(0.55, 0.04, 0.97, 0.30))
        if close is not None:
            return action_click_box(close, "close dialog/popup")
        home = self.find_cls(screen, UC.BTN_HOME, conf=_CLS_CONF)
        if home is not None:
            return action_click_box(home, "arena_shop exit: home")
        return action_click(*_POS_HOME, "arena_shop exit: home (fixed)")
