"""ShopSkill: Buy daily free / low-price shop items. YOLO-only clicks.

Design (2026-05-28 full YOLO rewrite — mirrors mail.py):
- Every click target resolved via self.find_cls/click_cls (ui_classes cls).
- NO OCR. The daily free-buy loop needs no pure-digit read (no count/AP),
  so there is zero OCR call in this skill — all state is inferred from YOLO.
- Page state inferred from YOLO cls: seeing SHOP_SELECT_ALL / SHOP_BUY /
  CURRENCY / FREE / COMBO_PACK = inside shop; NAV_SHOP or detect_screen_yolo
  == "Lobby" = on lobby.
- If YOLO can't see a needed cls, log + wait (surface the gap) — never fall
  back to OCR or blind hardcoded coords.

Flow:
1. enter:      from lobby, click NAV_SHOP (商店入口); cooldown to avoid re-tap
2. select_all: click SHOP_SELECT_ALL (全部选择). Grey variant => nothing to
               buy / already bought today => exit
3. purchase:   click SHOP_BUY (购买) => BTN_CONFIRM (确认). Grey confirm =>
               unavailable => exit. GOT_REWARD popup => dismiss => exit
4. exit:       detect_screen_yolo()=="Lobby" => done; else BTN_HOME/BTN_BACK/ESC

States: enter → select_all → purchase → exit
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click_box, action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC


# cls that prove we're inside the shop (any one is enough).
_SHOP_PAGE_CLS = [
    UC.SHOP_SELECT_ALL, UC.SHOP_SELECT_ALL_GREY, UC.SHOP_BUY,
    UC.CURRENCY, UC.CURRENCY_SEL, UC.FREE,
    UC.COMBO_PACK, UC.COMBO_PACK_SEL,
]


class ShopSkill(BaseSkill):
    def __init__(self):
        super().__init__("Shop")
        self.max_ticks = 50
        self._enter_click_cooldown: int = 0
        self._select_attempts: int = 0
        self._purchase_attempts: int = 0
        self._confirm_clicks: int = 0
        self._buy_clicks: int = 0
        self._purchased: bool = False
        self._post_click_wait: int = 0

    def reset(self) -> None:
        super().reset()
        self._enter_click_cooldown = 0
        self._select_attempts = 0
        self._purchase_attempts = 0
        self._confirm_clicks = 0
        self._buy_clicks = 0
        self._purchased = False
        self._post_click_wait = 0

    # ── page inference via YOLO cls (no OCR) ──────────────────────────
    def _on_shop_page(self, screen: ScreenState) -> bool:
        """Inside shop iff any shop-specific cls is visible."""
        return self.find_cls(screen, _SHOP_PAGE_CLS, conf=0.25) is not None

    # ── state machine ─────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        if self.ticks >= self.max_ticks:
            self.log(f"timeout (purchased={self._purchased})")
            return action_done(f"shop timeout (purchased={self._purchased})")

        # Reward popup (獲得獎勵) — dismiss via YOLO cls, then exit.
        reward = self.find_cls(screen, UC.GOT_REWARD, conf=0.30)
        if reward is not None:
            if not self._purchased:
                self._purchased = True
                self.log("purchase reward detected (GOT_REWARD)")
            self.sub_state = "exit"
            self._post_click_wait = 2
            return action_click_box(reward, "dismiss reward popup (YOLO 获得奖励)")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "shop loading")

        if self._post_click_wait > 0:
            self._post_click_wait -= 1
            return action_wait(500, f"shop: settling ({self._post_click_wait})")

        if self.sub_state == "":
            self.sub_state = "enter"
        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "select_all":
            return self._select_all(screen)
        if self.sub_state == "purchase":
            return self._purchase(screen)
        if self.sub_state == "exit":
            return self._exit(screen)
        return action_wait(300, "shop unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._on_shop_page(screen):
            self.log("inside shop")
            self.sub_state = "select_all"
            self._post_click_wait = 1
            return action_wait(400, "entered shop")

        # Cooldown so we don't re-tap the nav entry while it animates in.
        if self._enter_click_cooldown > 0:
            self._enter_click_cooldown -= 1
            return action_wait(500, f"shop: enter cooldown ({self._enter_click_cooldown})")

        # YOLO 商店入口 cls.
        shop_btn = self.find_cls(screen, UC.NAV_SHOP, conf=0.30)
        if shop_btn is not None:
            self._enter_click_cooldown = 4
            return action_click_box(shop_btn, "open shop (YOLO 商店入口)")

        # Not lobby + not shop → back out toward lobby.
        if self.detect_screen_yolo(screen) not in (None, "Lobby"):
            return action_back("shop: backing toward lobby")
        self.log("no 商店入口 cls visible — YOLO gap; waiting")
        return action_wait(400, "shop: waiting for 商店入口 detection")

    def _select_all(self, screen: ScreenState) -> Dict[str, Any]:
        """Click 全部选择 (select-all). Grey variant => nothing to buy."""
        self._select_attempts += 1

        if not self._on_shop_page(screen):
            if self._select_attempts > 8:
                self.log("not on shop page after select retries — exiting")
                self.sub_state = "exit"
                return action_wait(300, "shop: lost page during select")
            return action_wait(500, "shop: waiting for shop UI")

        # Grey select-all = no daily items available / already bought.
        grey = self.find_cls(screen, UC.SHOP_SELECT_ALL_GREY, conf=0.30)
        if grey is not None:
            self.log("select-all GREY (nothing to buy / already done)")
            self.sub_state = "exit"
            return action_wait(300, "shop nothing to select")

        sel = self.find_cls(screen, UC.SHOP_SELECT_ALL, conf=0.30)
        if sel is not None:
            self.log("select all (YOLO 全部选择)")
            self.sub_state = "purchase"
            self._purchase_attempts = 0
            self._post_click_wait = 2
            return action_click_box(sel, "select all items (YOLO 全部选择)")

        # No select-all cls at all. If a buy button is already showing
        # (items pre-selected), jump straight to purchase.
        if self.find_cls(screen, UC.SHOP_BUY, conf=0.30) is not None:
            self.log("no select-all but 购买 visible → purchase")
            self.sub_state = "purchase"
            self._purchase_attempts = 0
            return action_wait(300, "shop: proceeding to purchase")

        if self._select_attempts > 8:
            self.log("no 全部选择 cls after retries — YOLO gap; exiting")
            self.sub_state = "exit"
            return action_wait(300, "shop: no select-all cls")
        return action_wait(400, "shop: waiting for 全部选择 (YOLO gap)")

    def _purchase(self, screen: ScreenState) -> Dict[str, Any]:
        """Click 购买 (SHOP_BUY) → 确认 (BTN_CONFIRM). Grey confirm/buy =>
        unavailable => exit. Reward popup handled at tick() level."""
        self._purchase_attempts += 1

        if self._purchase_attempts > 16:
            self.log("purchase loop limit — exiting")
            self.sub_state = "exit"
            return action_wait(300, "shop: purchase timeout")

        # 1. Confirm dialog (确认) — appears after clicking 购买.
        confirm = self.find_cls(
            screen, UC.BTN_CONFIRM, conf=0.30, region=(0.30, 0.50, 0.90, 0.90),
        )
        if confirm is not None:
            self.log(f"confirm purchase (#{self._confirm_clicks + 1})")
            self._purchased = True
            self._confirm_clicks += 1
            self._post_click_wait = 2
            return action_click_box(confirm, "confirm purchase (YOLO 确认键)")

        # Grey confirm = can't afford / nothing valid selected → exit.
        if self.find_cls(
            screen, UC.BTN_CONFIRM_GREY, conf=0.30, region=(0.30, 0.50, 0.90, 0.90),
        ) is not None:
            self.log("confirm GREY (insufficient) — exiting")
            self.sub_state = "exit"
            return action_wait(300, "shop: confirm unavailable")

        # 2. Grey buy button = nothing purchasable → exit.
        if self.find_cls(screen, UC.SHOP_BUY_PYROXENE, conf=0.30) is not None:
            # 购买青辉石 is a paid-currency CTA — never auto-buy. Treat as done.
            self.log("only 购买青辉石 (paid) visible — skipping, exiting")
            self.sub_state = "exit"
            return action_wait(300, "shop: paid-only, nothing free to buy")

        # 3. Click 购买 (bulk buy of selected items).
        buy = self.find_cls(screen, UC.SHOP_BUY, conf=0.30)
        if buy is not None:
            if self._buy_clicks >= 3:
                self.log("购买 clicked 3x with no confirm — giving up")
                self.sub_state = "exit"
                return action_wait(300, "shop: buy stuck")
            self.log(f"click 购买 (#{self._buy_clicks + 1})")
            self._buy_clicks += 1
            self._post_click_wait = 2
            return action_click_box(buy, "buy selected items (YOLO 购买)")

        # 4. No buy / confirm cls. If we already confirmed, wait for reward.
        if self._purchased:
            self.sub_state = "exit"
            return action_wait(300, "shop: purchase done, exiting")

        # Nothing to buy was ever found.
        if self._purchase_attempts >= 4 and self._buy_clicks == 0:
            self.log("no 购买/确认 cls — nothing purchasable (YOLO gap or empty)")
            self.sub_state = "exit"
            return action_wait(300, "shop: nothing purchasable")
        return action_wait(400, "shop: waiting for 购买 cls (YOLO gap)")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        # On lobby iff >=2 nav icons (detect_screen_yolo).
        if self.detect_screen_yolo(screen) == "Lobby":
            status = "purchased" if self._purchased else "no purchase needed"
            self.log(f"done ({status})")
            return action_done(f"shop complete ({status})")
        # Prefer YOLO home/back button over blind ESC.
        home = self.find_cls(screen, UC.BTN_HOME, conf=0.30)
        if home is not None:
            return action_click_box(home, "shop exit: home button")
        back = self.find_cls(screen, UC.BTN_BACK, conf=0.30)
        if back is not None:
            return action_click_box(back, "shop exit: back button")
        return action_back("shop exit: ESC toward lobby")
