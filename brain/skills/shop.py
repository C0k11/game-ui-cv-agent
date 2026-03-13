"""ShopSkill: Buy daily items from the shop.

Flow:
1. ENTER: From lobby, click 商店 in nav bar
2. SELECT_TAB: Click 一般 (Normal) tab
3. SELECT_ALL: Click 全部選擇 / select-all checkbox
4. PURCHASE: Click 選擇購買 → confirm dialog → dismiss reward
5. EXIT: Back to lobby

Completion detection (from reference shop_utils.get_purchase_state):
- "purchase-available": 選擇購買 button active → proceed
- "purchase-unavailable": button greyed / no items → done
- "refresh-button-appear": refresh button visible → items already bought today

Key patterns:
- Header: "商店" (Shop)
- Tabs: "一般" (Normal), "特殊" (Special), etc.
- Select all: "全部選擇" / "全部选择"
- Bulk purchase: "選擇購買" / "选择购买" (bottom bar after select-all)
- Confirm dialog: "取消" + "確認"
- Reward: "獲得道具" / "獲得獎勵"
- Refresh: "更新" / "刷新" / "Refresh" (means already bought)
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box,
    action_wait, action_back, action_done,
)


class ShopSkill(BaseSkill):
    def __init__(self):
        super().__init__("Shop")
        self.max_ticks = 50
        self._tab_clicked: bool = False
        self._selected_all: bool = False
        self._purchase_confirmed: bool = False
        self._purchase_ticks: int = 0
        self._enter_ticks: int = 0
        self._select_ticks: int = 0

    def reset(self) -> None:
        super().reset()
        self._tab_clicked = False
        self._selected_all = False
        self._purchase_confirmed = False
        self._purchase_ticks = 0
        self._enter_ticks = 0
        self._select_ticks = 0

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("shop timeout")

        # Reward popup — dismiss and mark purchase done
        reward = screen.find_any_text(
            ["獲得道具", "获得道具", "獲得獎勵", "获得奖励"],
            min_conf=0.6
        )
        if reward:
            if not self._purchase_confirmed:
                self._purchase_confirmed = True
                self.log("purchase reward detected")
            ok = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "OK"],
                min_conf=0.55
            )
            if ok:
                self.sub_state = "exit"
                return action_click_box(ok, "dismiss reward popup")
            return action_click(0.5, 0.9, "dismiss reward popup")

        # Inventory full popup
        inv_full = screen.find_any_text(
            ["背包已满", "背包已滿", "空間不足", "空间不足"],
            region=screen.CENTER, min_conf=0.6
        )
        if inv_full:
            self.log("inventory full, exiting shop")
            self.sub_state = "exit"
            confirm = screen.find_any_text(
                ["確認", "确认", "確", "确"], region=screen.CENTER, min_conf=0.6
            )
            if confirm:
                return action_click_box(confirm, "confirm inventory full")
            return action_click(0.5, 0.73, "confirm inventory full (fallback)")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "shop loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "select_tab":
            return self._select_tab(screen)
        if self.sub_state == "select_all":
            return self._select_all(screen)
        if self.sub_state == "purchase":
            return self._purchase(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "shop unknown state")

    def _is_shop(self, screen: ScreenState) -> bool:
        if (screen.has_text("商店", region=(0.0, 0.0, 0.3, 0.10), min_conf=0.5) or
                screen.has_text("Shop", region=(0.0, 0.0, 0.3, 0.10), min_conf=0.5)):
            return True
        if screen.find_any_text(["一般", "青輝石", "青辉石"], region=(0.0, 0.10, 0.20, 0.40), min_conf=0.6):
            filter_toggle = screen.find_any_text(["濾器", "滤器", "OFF"], region=(0.68, 0.08, 0.86, 0.16), min_conf=0.6)
            select_all = screen.find_any_text(["全部選擇", "全部选择", "全部擇", "全部摆"], region=(0.84, 0.08, 1.0, 0.16), min_conf=0.55)
            if filter_toggle or select_all:
                return True
        return False

    def _is_already_purchased(self, screen: ScreenState) -> bool:
        """Detect if items are already purchased today.

        reference checks for "refresh" button appearance as a signal that
        today's items have already been bought.
        """
        refresh = screen.find_any_text(
            ["更新", "刷新", "Refresh"],
            region=(0.85, 0.88, 1.0, 0.98), min_conf=0.55
        )
        return refresh is not None

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks += 1
        current = self.detect_current_screen(screen)

        if current == "Shop" or self._is_shop(screen):
            self.log("inside shop")
            self.sub_state = "select_tab"
            return action_wait(400, "entered shop")

        if current == "Lobby":
            nav = self._nav_to(screen, ["商店"])
            if nav:
                return nav
            return action_click(0.50, 0.95, "click shop nav area (hardcoded)")

        if current and current != "Shop":
            return action_back(f"back from {current}")

        if self._enter_ticks > 12:
            self.log("can't reach shop, skipping")
            return action_done("shop unavailable")

        return action_wait(500, "entering shop")

    def _select_tab(self, screen: ScreenState) -> Dict[str, Any]:
        """Click the 一般 (Normal) tab in the shop."""
        if not self._is_shop(screen):
            return action_wait(500, "waiting for shop UI")

        # Check if items already purchased (refresh button visible)
        if self._is_already_purchased(screen):
            self.log("items already purchased today (refresh button visible)")
            self.sub_state = "exit"
            return action_wait(250, "shop already done")

        tab = screen.find_any_text(
            ["一般", "Normal"],
            region=(0.0, 0.08, 0.5, 0.20), min_conf=0.6
        )
        if tab:
            self.log(f"clicking '{tab.text}' tab")
            self._tab_clicked = True
            self.sub_state = "select_all"
            return action_click_box(tab, f"click '{tab.text}' tab")

        if self._tab_clicked or self.ticks > 6:
            self.sub_state = "select_all"
            return action_wait(300, "tab already selected or timeout")

        return action_wait(400, "looking for 一般 tab")

    def _select_all(self, screen: ScreenState) -> Dict[str, Any]:
        """Click 全部選擇 (Select All) checkbox.

        OCR often misreads "全部選擇" as "全部選選" — include fuzzy variants.
        The checkbox is at top-right of the shop (x~0.90, y~0.10-0.14).
        """
        self._select_ticks += 1

        if not self._is_shop(screen):
            return action_wait(500, "waiting for shop UI")

        # Check if items already purchased
        if self._is_already_purchased(screen):
            self.log("items already purchased today")
            self.sub_state = "exit"
            return action_wait(250, "shop already done")

        sel = screen.find_any_text(
            ["全部選擇", "全部选择", "全部選選", "全部选选", "Select All"],
            region=(0.80, 0.08, 1.0, 0.16), min_conf=0.55
        )
        if sel:
            self.log(f"clicking '{sel.text}' (select all)")
            self._selected_all = True
            self.sub_state = "purchase"
            self._purchase_ticks = 0
            return action_click_box(sel, "select all items")

        if self._select_ticks > 4:
            self.log("select-all OCR miss, clicking hardcoded position")
            self._selected_all = True
            self.sub_state = "purchase"
            self._purchase_ticks = 0
            return action_click(0.93, 0.12, "select all (hardcoded)")

        return action_wait(400, "looking for select-all")

    def _purchase(self, screen: ScreenState) -> Dict[str, Any]:
        """Click 選擇購買 (bulk purchase) → confirm dialog → dismiss reward.

        After clicking 全部選擇, a bottom bar appears:
          "現在選擇的商品24個" | "取消選擇" | [選擇購買]
        The 選擇購買 button is in the bottom-right (~0.91, 0.92).

        Completion detection:
        - Confirm dialog with 取消+確認 → click 確認 (mark purchase_confirmed)
        - Reward popup → dismiss (mark exit)
        - Bottom bar disappears → already bought or nothing selected
        - Refresh button → already purchased
        """
        self._purchase_ticks += 1

        if self._purchase_ticks > 20:
            self.log("purchase timeout, exiting")
            self.sub_state = "exit"
            return action_wait(300, "purchase timeout")

        # 1. Confirm dialog (取消 + 確認) — after clicking 選擇購買
        cancel_in_dialog = screen.find_any_text(
            ["取消"],
            region=(0.25, 0.58, 0.55, 0.82), min_conf=0.7
        )
        confirm_in_dialog = screen.find_any_text(
            ["確認", "确认", "確定", "确定"],
            region=(0.50, 0.58, 0.80, 0.82), min_conf=0.6
        )
        if cancel_in_dialog and confirm_in_dialog:
            self.log("confirming bulk purchase")
            self._purchase_confirmed = True
            return action_click_box(confirm_in_dialog, "confirm purchase")

        # 2. Already purchased today (refresh button appeared)
        if self._is_already_purchased(screen):
            self.log("shop refresh visible, items purchased")
            self.sub_state = "exit"
            return action_wait(250, "purchase complete (refresh visible)")

        # 3. If we already confirmed, wait for reward popup then exit
        if self._purchase_confirmed:
            # Reward popup is handled at tick() level; if we reach here, just wait/exit
            if self._purchase_ticks > 5:
                self.sub_state = "exit"
                return action_wait(250, "purchase confirmed, exiting")
            return action_wait(500, "waiting for purchase result")

        # 4. Click 選擇購買 in the bottom bar
        bulk_buy = screen.find_any_text(
            ["選擇購買", "选择购买", "選購買", "选购买", "擇購買", "择购买"],
            region=(0.70, 0.85, 1.0, 0.99), min_conf=0.35
        )
        if bulk_buy:
            self.log(f"clicking bulk purchase '{bulk_buy.text}'")
            return action_click_box(bulk_buy, "bulk purchase (選擇購買)")

        # 5. Check if bottom selection bar is visible (items are selected)
        selection_bar = screen.find_any_text(
            ["現在選", "现在选", "取消選", "取消选", "商品"],
            region=(0.20, 0.85, 0.85, 0.99), min_conf=0.5
        )
        if selection_bar:
            # Bar visible but 選擇購買 OCR missed → hardcoded click
            self.log("selection bar visible, clicking 選擇購買 hardcoded")
            return action_click(0.91, 0.92, "bulk purchase (hardcoded)")

        # 6. No bottom bar visible — items may not be selectable or already bought
        if self._purchase_ticks >= 5 and not self._selected_all:
            self.log("no purchase bar visible, nothing to buy")
            self.sub_state = "exit"
            return action_wait(250, "nothing purchasable")

        # Try hardcoded after a few ticks
        if self._purchase_ticks >= 4:
            return action_click(0.91, 0.92, "bulk purchase fallback click")

        return action_wait(400, "looking for 選擇購買 button")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            status = "purchased" if self._purchase_confirmed else "no purchase needed"
            self.log(f"done ({status})")
            return action_done(f"shop complete ({status})")
        return action_back("shop exit: back to lobby")
