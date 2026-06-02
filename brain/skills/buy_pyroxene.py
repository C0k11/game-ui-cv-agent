"""BuyPyroxeneSkill — claim the daily FREE combo pack (pure-YOLO rewrite).

Verified flow (interactive probe 2026-06-01, data/_buy_pyroxene_probe_log.md).
All clicks resolved through YOLO cls (ui_classes) — NO OCR, NO hardcoded pixel
positions. The ONLY purpose is to claim the **每日免費組合包** (gives AP×10 +
credits×10K, NOT pyroxene).

★★ HARD RULE — NEVER spend pyroxene / real money ★★
The shop has CAD$ packs sitting right next to the free one. We ONLY ever click
a 购买 (SHOP_BUY) button that has a 免费 (FREE) price-label directly above it in
the same column, and we ONLY confirm a purchase dialog that shows the 免费 cls.
Any dialog without 免费 (i.e. a CAD price) ⇒ cancel immediately.

State machine
-------------
enter      lobby → click SHOP_BUY_PYROXENE (购买青辉石). Retry on ADB drop
           (re-click while still on lobby). Wait for the shop popup.
combo_tab  shop opens on 特別販售 tab → click COMBO_PACK (组合包未选择) to switch
           to the 組合包 tab. Done when COMBO_PACK_SEL (组合包已选择) shows.
buy        on 組合包 tab: find FREE (免费) label, click the SHOP_BUY directly
           below it (same column). No FREE cls ⇒ today's pack already claimed
           ⇒ exit. NEVER click a 购买 without FREE above it.
confirm    "是否購買該商品？" dialog (BTN_CONFIRM + BTN_CANCEL). Poll for the
           FREE cls in the price area: present ⇒ BTN_CONFIRM; never appears
           ⇒ BTN_CANCEL (treat as unexpected paid item, abort).
reward     GOT_REWARD popup → dismiss via STORY_TAP_CONTINUE / GOT_REWARD header
           (NEVER tap screen center — that hits the item icons). Loop until gone.
exit       close shop via BTN_CLOSE_X (retry on drop) → lobby → done.

Detectors: base "ui" only (no avatar/battle/cafe) — not in SKILL_YOLO_MAP.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click, action_click_box, action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC

# ── tuning knobs ─────────────────────────────────────────────────────────
_CLS_CONF = 0.30              # default UI cls confidence floor

# Purchase-confirm dialog button band (确认键 / 取消键 sit center-bottom ~y0.82).
_DIALOG_BAND = (0.28, 0.74, 0.72, 0.92)
# Price area inside the confirm dialog where the 免费 label sits (~0.66, 0.61).
_DIALOG_PRICE_REGION = (0.48, 0.48, 0.88, 0.74)
# Same-column tolerance when pairing a FREE label with its 购买 button.
_FREE_COL_DX = 0.10

# Per-sub-state tick budgets — every phase is bounded, never dead-waits.
_ENTER_MAX = 22
_COMBO_MAX = 16
_BUY_MAX = 16
_CONFIRM_MAX = 10            # poll the dialog for the FREE cls before bailing
_REWARD_MAX = 18
_EXIT_MAX = 14


class BuyPyroxeneSkill(BaseSkill):
    """Claim the daily free combo pack. Never spends premium currency."""

    def should_run(self, screen: ScreenState) -> bool:
        """Run only when the lobby 购买青辉石 entry carries a red dot.

        The badge sits just above-right of the entry banner (probe: entry
        ~(0.115,0.359), red dot ~(0.149,0.345)), so dot_on_entry's strict
        inside-bbox test can miss it. We find the entry dynamically and scan a
        region expanded up/right of its bbox for a red dot. Entry not visible
        ⇒ defer (return True) — we're probably not on the lobby yet.
        """
        entry = self.find_cls(screen, UC.SHOP_BUY_PYROXENE, conf=0.40)
        if entry is None:
            return True
        region = (entry.x1 - 0.02, entry.y1 - 0.05, entry.x2 + 0.05, entry.y2 + 0.02)
        return self.dot_in_region(screen, region, dot_classes=(UC.DOT_RED,))

    def __init__(self):
        super().__init__("BuyPyroxene")
        # enter(~6)+combo(~4)+buy(~3)+confirm(~6)+reward(~6)+exit(~6) ≈ 31;
        # 80 leaves slack for ADB-drop retries.
        self.max_ticks = 80
        self._init_state()

    # ── state init / reset ────────────────────────────────────────────────

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._bought: bool = False         # set once the free pack is confirmed
        self._buy_retry: Optional[tuple] = None  # (cx,cy) to re-press on drop

    def reset(self) -> None:
        super().reset()
        self._init_state()

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0

    # ── shared cls helpers ────────────────────────────────────────────────

    def _shop_open(self, screen: ScreenState) -> bool:
        """Shop popup open = either combo-pack tab present (selected or not)."""
        return self.find_cls(
            screen, [UC.COMBO_PACK, UC.COMBO_PACK_SEL], conf=_CLS_CONF
        ) is not None

    def _on_combo_tab(self, screen: ScreenState) -> bool:
        return self.find_cls(screen, UC.COMBO_PACK_SEL, conf=_CLS_CONF) is not None

    def _confirm_dialog(self, screen: ScreenState) -> Optional[YoloBox]:
        """Purchase-confirm dialog = BOTH 确认键 and 取消键 in the button band.
        Returns the 确认键 box (the thing we click on a verified-free pack)."""
        confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_DIALOG_BAND)
        cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF, region=_DIALOG_BAND)
        if confirm is not None and cancel is not None:
            return confirm
        return None

    def _free_buy_button(self, screen: ScreenState) -> Optional[YoloBox]:
        """The 购买 button of the FREE pack — paired geometrically with FREE.

        Find the 免费 (FREE) price label, then among all 购买 (SHOP_BUY)
        buttons pick the one directly BELOW it in the same column. Returns None
        when no FREE label is on screen (pack already claimed today) OR no 购买
        sits under it. This is the SOLE purchase path — a 购买 with no FREE
        above it is a CAD pack and is never returned."""
        free = self.find_cls(screen, UC.FREE, conf=_CLS_CONF)
        if free is None:
            return None
        buys = self.find_all_cls(screen, UC.SHOP_BUY, conf=_CLS_CONF)
        cands = [b for b in buys if b.cy > free.cy and abs(b.cx - free.cx) < _FREE_COL_DX]
        if not cands:
            return None
        return min(cands, key=lambda b: abs(b.cx - free.cx))

    def _close_x(self, screen: ScreenState) -> Optional[YoloBox]:
        return self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF,
                             region=(0.55, 0.04, 0.95, 0.30))

    # ── tick: global guards + dispatch ─────────────────────────────────────

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout, exiting")
            return action_done("buy_pyroxene timeout")

        if screen.is_loading():
            return action_wait(600, "shop loading")

        if self.sub_state == "":
            self._goto("enter")

        handler = {
            "enter": self._enter,
            "combo_tab": self._combo_tab,
            "buy": self._buy,
            "confirm": self._confirm,
            "reward": self._reward,
            "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "buy_pyroxene unknown state")
        return handler(screen)

    # ── enter ───────────────────────────────────────────────────────────────

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._shop_open(screen):
            self.log("shop popup open → combo tab")
            self._goto("combo_tab")
            return action_wait(300, "entered shop")

        if screen.is_lobby():
            entry = self.find_cls(screen, UC.SHOP_BUY_PYROXENE, conf=_CLS_CONF)
            if entry is not None:
                self.log("clicking 购买青辉石 entry (retry on drop)")
                return action_click_box(entry, "open buy-pyroxene shop")
            self.log("on lobby but no 购买青辉石 cls — YOLO gap; waiting")
            return action_wait(400, "waiting for 购买青辉石 cls")

        if self._phase_ticks > _ENTER_MAX:
            self.log("enter budget exhausted, giving up")
            return action_done("could not reach buy-pyroxene shop")
        # Unknown / transition screen — wait, then nudge back toward lobby.
        if len(screen.yolo_boxes or []) < 2:
            return action_wait(600, "no UI detected, likely loading")
        return action_back("recover toward lobby before buy-pyroxene")

    # ── combo_tab ────────────────────────────────────────────────────────────

    def _combo_tab(self, screen: ScreenState) -> Dict[str, Any]:
        if not self._shop_open(screen):
            if screen.is_lobby():
                self._goto("enter")
                return action_wait(300, "combo_tab: back on lobby, re-enter")
            if self._phase_ticks > _COMBO_MAX:
                self.log("combo_tab: shop lost, exiting")
                self._goto("exit")
                return action_wait(300, "combo_tab lost shop → exit")
            return action_wait(400, "waiting for shop UI (combo_tab)")

        # Already on the 組合包 tab → go buy.
        if self._on_combo_tab(screen):
            self._goto("buy")
            return action_wait(250, "on 組合包 tab → buy")

        # On 特別販售 / 青輝石 tab → click the unselected 組合包 tab.
        tab = self.find_cls(screen, UC.COMBO_PACK, conf=_CLS_CONF)
        if tab is not None:
            self.log("switching to 組合包 tab (YOLO 组合包未选择)")
            return action_click_box(tab, "switch to combo-pack tab")

        if self._phase_ticks > _COMBO_MAX:
            self.log("combo tab cls never found, exiting")
            self._goto("exit")
            return action_wait(300, "no combo tab cls → exit")
        return action_wait(350, "waiting for 组合包 tab cls")

    # ── buy ──────────────────────────────────────────────────────────────────

    def _buy(self, screen: ScreenState) -> Dict[str, Any]:
        # A confirm dialog may already be up (e.g. retry race) → handle it.
        if self._confirm_dialog(screen) is not None:
            self._goto("confirm")
            return action_wait(200, "confirm dialog up → confirm")

        if not self._shop_open(screen):
            if screen.is_lobby():
                self._goto("enter")
                return action_wait(300, "buy: back on lobby, re-enter")
            if self._phase_ticks > _BUY_MAX:
                self._goto("exit")
                return action_wait(300, "buy lost shop → exit")
            return action_wait(400, "waiting for shop UI (buy)")

        free_buy = self._free_buy_button(screen)
        if free_buy is not None:
            self._buy_retry = (free_buy.cx, free_buy.cy)
            self.log(f"clicking FREE pack 购买 at ({free_buy.cx:.2f},{free_buy.cy:.2f})")
            self._goto("confirm")
            return action_click_box(free_buy, "buy FREE daily combo pack")

        # No FREE label → pack already claimed today (or already bought now).
        if self._phase_ticks > 4:  # let the tab settle before concluding
            why = "claimed just now" if self._bought else "already claimed today"
            self.log(f"no FREE pack 购买 visible → done ({why})")
            self._goto("exit")
            return action_wait(250, f"no free pack → exit ({why})")
        return action_wait(300, "waiting for FREE pack to render")

    # ── confirm ──────────────────────────────────────────────────────────────

    def _confirm(self, screen: ScreenState) -> Dict[str, Any]:
        # Reward popup already showing (confirm registered) → reward state.
        if (self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF) is not None
                or self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF) is not None):
            self._bought = True
            self._goto("reward")
            return action_wait(200, "reward popup up → reward")

        confirm_btn = self._confirm_dialog(screen)
        if confirm_btn is None:
            # Dialog not up yet. Re-press the buy button (ADB may have dropped),
            # then give it a few ticks. If never appears, fall back to buy.
            if self._phase_ticks > _CONFIRM_MAX:
                if self._on_combo_tab(screen) or self._shop_open(screen):
                    self.log("confirm dialog never appeared → back to buy")
                    self._goto("buy")
                    return action_wait(300, "no dialog → re-pick buy")
                self._goto("exit")
                return action_wait(300, "confirm timeout → exit")
            if self._buy_retry is not None and self._phase_ticks % 4 == 0:
                bx, by = self._buy_retry
                self.log("confirm dialog absent, re-pressing buy button (drop?)")
                return action_click(bx, by, "re-press FREE buy (no dialog)")
            return action_wait(300, "waiting for purchase-confirm dialog")

        # ★ SAFETY GATE: only confirm when the 免费 cls is in the price area.
        free_in_dialog = self.find_cls(
            screen, UC.FREE, conf=_CLS_CONF, region=_DIALOG_PRICE_REGION
        )
        if free_in_dialog is not None:
            self.log("dialog shows 免费 → confirming purchase (YOLO 确认键)")
            self._bought = True
            self._goto("reward")
            return action_click_box(confirm_btn, "confirm FREE purchase")

        # Dialog up but no FREE yet. FREE is static here, so poll a few frames
        # (rule #5). If it never shows, this is NOT a free pack → cancel.
        if self._phase_ticks > _CONFIRM_MAX:
            self.log("⛔ dialog has NO 免费 cls (paid?) — cancelling, never buy")
            cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF, region=_DIALOG_BAND)
            if cancel is not None:
                self._goto("exit")
                return action_click_box(cancel, "cancel non-free purchase")
            close = self._close_x(screen)
            if close is not None:
                self._goto("exit")
                return action_click_box(close, "close non-free dialog")
            self._goto("exit")
            return action_back("abort non-free purchase (ESC)")
        return action_wait(250, "polling dialog for 免费 cls")

    # ── reward ───────────────────────────────────────────────────────────────

    def _reward(self, screen: ScreenState) -> Dict[str, Any]:
        # Prefer the 点击继续字样 strip; else tap the 获得奖励 header box.
        # NEVER tap screen center (probe: that hits the item icons, no advance).
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
        if cont is not None:
            self.log("dismiss reward (点击继续字样)")
            return action_click_box(cont, "dismiss reward via continue")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if got is not None:
            self.log("dismiss reward (获得奖励 header)")
            return action_click_box(got, "dismiss reward via header")

        # Neither present → reward closed → back on combo tab → exit.
        if self._shop_open(screen) or self._phase_ticks > 2:
            self.log("reward dismissed → exit")
            self._goto("exit")
            return action_wait(250, "reward done → exit")
        return action_wait(300, "waiting for reward popup to settle")

    # ── exit ─────────────────────────────────────────────────────────────────

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log("back in lobby, buy_pyroxene done")
            return action_done("buy_pyroxene complete")
        if self._phase_ticks > _EXIT_MAX:
            self.log("exit budget exhausted, reporting done")
            return action_done("buy_pyroxene exit timeout")
        # Close the shop popup (ADB drop on the X is common — retry).
        close = self._close_x(screen)
        if close is not None:
            return action_click_box(close, "close shop popup (X)")
        home = self.find_cls(screen, UC.BTN_HOME, conf=_CLS_CONF)
        if home is not None:
            return action_click_box(home, "exit: home button")
        return action_back("buy_pyroxene exit: ESC toward lobby")
