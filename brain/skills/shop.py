"""ShopSkill: Buy daily items from the shop.

Flow:
1. ENTER: From lobby, click 商店 in nav bar
2. SELECT_TAB: Click 一般 (Normal) tab
3. SELECT_ALL: Click 全部選擇 / select-all checkbox
4. PURCHASE: Click 購買 / purchase button → confirm
5. EXIT: Back to lobby

Key patterns:
- Header: "商店" (Shop)
- Tabs: "一般" (Normal), "特殊" (Special), etc.
- Select all: "全部選擇" / "全部选择"
- Purchase: "購買" / "购买"
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_click_yolo,
    action_wait, action_back, action_done,
)


class ShopSkill(BaseSkill):
    def __init__(self):
        super().__init__("Shop")
        self.max_ticks = 40
        self._tab_clicked: bool = False
        self._selected_all: bool = False
        self._purchased: bool = False
        self._purchase_ticks: int = 0

    def reset(self) -> None:
        super().reset()
        self._tab_clicked = False
        self._selected_all = False
        self._purchased = False
        self._purchase_ticks = 0

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("shop timeout")

        # Reward popup — dismiss
        reward = screen.find_any_text(
            ["獲得道具", "获得道具", "獲得獎勵", "获得奖励"],
            region=screen.CENTER, min_conf=0.6
        )
        if reward:
            return action_click(0.5, 0.9, "dismiss reward popup")

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
        return (screen.has_text("商店", region=(0.0, 0.0, 0.3, 0.10), min_conf=0.6) or
                screen.has_text("Shop", region=(0.0, 0.0, 0.3, 0.10), min_conf=0.6))

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        current = self.detect_current_screen(screen)

        if current == "Shop" or self._is_shop(screen):
            self.log("inside shop")
            self.sub_state = "select_tab"
            return action_wait(500, "entered shop")

        if current == "Lobby":
            nav = self._nav_to(screen, ["商店"])
            if nav:
                return nav
            return action_wait(300, "waiting for shop button")

        if current and current != "Shop":
            self.log(f"wrong screen '{current}', backing out")
            return action_back(f"back from {current}")

        return action_wait(500, "entering shop")

    def _select_tab(self, screen: ScreenState) -> Dict[str, Any]:
        """Click the 一般 (Normal) tab in the shop."""
        if not self._is_shop(screen):
            return action_wait(500, "waiting for shop UI")

        # Look for 一般 tab
        tab = screen.find_any_text(
            ["一般", "Normal"],
            region=(0.0, 0.08, 0.5, 0.20), min_conf=0.6
        )
        if tab:
            self.log(f"clicking '{tab.text}' tab")
            self._tab_clicked = True
            self.sub_state = "select_all"
            return action_click_box(tab, f"click '{tab.text}' tab")

        # If already on 一般 tab (highlighted), skip
        if self._tab_clicked or self.ticks > 5:
            self.sub_state = "select_all"
            return action_wait(300, "tab already selected or timeout")

        return action_wait(400, "looking for 一般 tab")

    def _select_all(self, screen: ScreenState) -> Dict[str, Any]:
        """Click 全部選擇 (Select All) checkbox.

        OCR often misreads "全部選擇" as "全部選選" — include fuzzy variants.
        The checkbox is at top-right of the shop (x~0.90, y~0.10-0.14).
        """
        if not self._is_shop(screen):
            return action_wait(500, "waiting for shop UI")

        # Look for select-all text — OCR variants include misread "選選"
        sel = screen.find_any_text(
            ["全部選擇", "全部选择", "全部選選", "全部选选", "Select All"],
            region=(0.80, 0.08, 1.0, 0.16), min_conf=0.6
        )
        if sel:
            self.log(f"clicking '{sel.text}' (select all)")
            self._selected_all = True
            self.sub_state = "purchase"
            return action_click_box(sel, "select all items")

        # Fallback: click the known hardcoded position (top-right checkbox)
        # OCR at (0.90, 0.10-0.13) per trajectory data
        if self.ticks > 8:
            self.log("select-all OCR miss, clicking hardcoded position")
            self._selected_all = True
            self.sub_state = "purchase"
            return action_click(0.93, 0.12, "select all (hardcoded)")

        return action_wait(400, "looking for select-all")

    def _purchase(self, screen: ScreenState) -> Dict[str, Any]:
        """Click 選擇購買 (bulk purchase) button in bottom-right, then confirm.

        After clicking 全部選擇, a bottom bar appears:
          "現在選擇的商品24個" | "取消選擇" | [選擇購買]
        OCR reads "選擇購買" as "選購買" (conf ~0.76) at bottom-right (~0.91, 0.92).
        Do NOT click individual per-item "購買" buttons (y~0.47-0.50).
        """
        self._purchase_ticks += 1

        if self._purchase_ticks > 15:
            self.log("purchase timeout")
            self.sub_state = "exit"
            return action_wait(300, "purchase timeout")

        # Priority 1: Handle confirm dialog (取消 + 確)
        # This appears after clicking 選擇購買 — must confirm before anything else
        cancel = screen.find_any_text(
            ["取消"],
            region=(0.25, 0.60, 0.75, 0.90), min_conf=0.8
        )
        if cancel:
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确"],
                region=(0.50, 0.60, 0.75, 0.90), min_conf=0.7
            )
            if confirm:
                self.log("confirming bulk purchase")
                self._purchased = True
                self.sub_state = "exit"
                return action_click_box(confirm, "confirm purchase")

        # Priority 2: Reward result popup — dismiss and exit
        reward = screen.find_any_text(
            ["獲得道具", "获得道具", "通知"],
            region=screen.CENTER, min_conf=0.7
        )
        if reward and self._purchased:
            self.sub_state = "exit"
            return action_click(0.5, 0.9, "dismiss purchase result")

        # Priority 3: Click "選擇購買" in the bottom bar (y > 0.88)
        # OCR reads it as "選購買" or "選擇購買" — look in bottom-right area
        if not self._purchased:
            bulk_buy = screen.find_any_text(
                ["選擇購買", "选择购买", "選購買", "选购买"],
                region=(0.80, 0.88, 1.0, 0.96), min_conf=0.5
            )
            if bulk_buy:
                self.log(f"clicking bulk purchase '{bulk_buy.text}' at ({bulk_buy.cx:.2f},{bulk_buy.cy:.2f})")
                return action_click_box(bulk_buy, "bulk purchase (選擇購買)")

            # Fallback: if "現在選" or "取消選" visible in bottom bar,
            # the 選擇購買 button should be at the far right
            selection_bar = screen.find_any_text(
                ["現在選", "取消選"],
                region=(0.40, 0.88, 0.85, 0.96), min_conf=0.6
            )
            if selection_bar:
                self.log("selection bar visible, clicking 選擇購買 at hardcoded position")
                return action_click(0.91, 0.92, "bulk purchase (hardcoded)")

            # If shop is showing normal view (individual 購買 visible, no selection bar)
            # then the purchase was already handled by the pipeline interceptor.
            # Detect: shop header visible + individual 購買 buttons + no selection bar
            if self._is_shop(screen) and self._purchase_ticks >= 3:
                indiv_buy = screen.find_any_text(
                    ["購買"], region=(0.45, 0.40, 0.95, 0.55), min_conf=0.6
                )
                if indiv_buy:
                    self.log("purchase already handled (interceptor), exiting")
                    self._purchased = True
                    self.sub_state = "exit"
                    return action_wait(300, "purchase complete (interceptor)")

        return action_wait(400, "looking for 選擇購買 button")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log("done")
            return action_done("shop complete")
        return action_back("shop exit: back to lobby")
