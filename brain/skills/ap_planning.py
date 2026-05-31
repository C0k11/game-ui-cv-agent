"""ApPlanningSkill: claim the daily FREE AP, never buy paid AP. YOLO-only clicks.

Design (2026-05-28 full YOLO rewrite — mirrors mail.py / shop.py):
- Every click target resolved via self.find_cls/click_cls (ui_classes cls).
- The ONLY OCR is _read_ap() reading the top-bar "X/240" digits — used for
  logging / the completion report, NOT for any click decision.
- Page state inferred from YOLO cls: seeing FREE / BTN_CONFIRM(_GREY) /
  BTN_CANCEL / TOPBAR_PYROXENE inside the dialog = AP purchase panel open;
  detect_screen_yolo()=="Lobby" = back on lobby.
- If YOLO can't see a needed cls, log + wait (surface the gap) — never fall
  back to OCR-for-clicking or blind hardcoded coords.

SAFETY (unchanged contract): only ever claim the FREE daily AP. Never click a
paid-currency (青辉石) purchase. forbid_premium_currency / paid_purchase_limit
are kept for API compatibility, but see the gap note in _plan(): ui_classes has
NO cls to identify the *paid* AP tiers (only UC.FREE for the free one), so the
paid path can't target a row via YOLO and is conservatively skipped.

States: enter → open_purchase → plan → exit
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill,
    ScreenState,
    action_back,
    action_click_box,
    action_done,
    action_wait,
)
from brain.skills import ui_classes as UC


# cls that prove the AP purchase dialog is open (any one is enough).
_AP_PANEL_CLS = [
    UC.FREE, UC.BTN_CONFIRM, UC.BTN_CONFIRM_GREY, UC.BTN_CANCEL,
]


class ApPlanningSkill(BaseSkill):
    def __init__(self, *, forbid_premium_currency: bool = True, paid_purchase_limit: int = 0):
        super().__init__("ApPlanning")
        self.max_ticks = 90
        self._forbid_premium_currency = bool(forbid_premium_currency)
        self._paid_purchase_limit = max(0, int(paid_purchase_limit))

        self._enter_attempts: int = 0
        self._dialog_attempts: int = 0
        self._free_collected: bool = False
        self._paid_done: int = 0
        self._pending_free_confirm: bool = False
        self._enter_click_cooldown: int = 0
        self._post_click_wait: int = 0
        self._last_ap: Optional[int] = None

    def reset(self) -> None:
        super().reset()
        self._enter_attempts = 0
        self._dialog_attempts = 0
        self._free_collected = False
        self._paid_done = 0
        self._pending_free_confirm = False
        self._enter_click_cooldown = 0
        self._post_click_wait = 0
        self._last_ap = None

    # ── OCR: digits only (the one allowed OCR use) ────────────────────
    def _read_ap(self, screen: ScreenState) -> Optional[int]:
        """Read current AP from the top-bar 'X/240' counter. ONLY OCR call.

        Used purely for logging + the completion report (e.g. to confirm AP
        rose after a free claim). Never drives a click decision.
        """
        for box in screen.find_text(r"\d+\s*/\s*\d+", region=screen.TOP_BAR, min_conf=0.4):
            m = re.search(r"(\d+)\s*[/|]\s*(\d+)", box.text or "")
            if m:
                try:
                    cur, cap = int(m.group(1)), int(m.group(2))
                except ValueError:
                    continue
                # Sanity: BA AP cap is ~200-240+; reject stray digit pairs.
                if cap >= 100:
                    return cur
        return None

    # ── page inference via YOLO cls (no OCR) ──────────────────────────
    def _on_ap_panel(self, screen: ScreenState) -> bool:
        """Inside the AP purchase dialog iff any panel cls is visible."""
        return self.find_cls(screen, _AP_PANEL_CLS, conf=0.25) is not None

    # ── state machine ─────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        if self.ticks >= self.max_ticks:
            self.log(f"timeout (free={int(self._free_collected)}, paid={self._paid_done})")
            return action_done("ap planning timeout")

        # Reward popup (獲得獎勵 / 購買完成) — dismiss via YOLO cls.
        reward = self.find_cls(screen, UC.GOT_REWARD, conf=0.30)
        if reward is not None:
            self._free_collected = True
            self._post_click_wait = 2
            return action_click_box(reward, "dismiss AP reward popup (YOLO 获得奖励)")

        # Common popups (notifications / bond level-up / etc). The AP panel
        # itself uses BTN_CANCEL/BTN_CONFIRM cls which we drive in _plan(),
        # so only run the shared OCR popup handler outside the plan state to
        # avoid it stealing the dialog's confirm button.
        if self.sub_state != "plan":
            popup = self._handle_common_popups(screen)
            if popup:
                return popup

        if screen.is_loading():
            return action_wait(700, "ap planning loading")

        if self._post_click_wait > 0:
            self._post_click_wait -= 1
            return action_wait(500, f"ap planning: settling ({self._post_click_wait})")

        if self.sub_state == "":
            self.sub_state = "enter"
        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "open_purchase":
            return self._open_purchase(screen)
        if self.sub_state == "plan":
            return self._plan(screen)
        if self.sub_state == "exit":
            return self._exit(screen)
        return action_wait(300, "ap planning unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._on_ap_panel(screen):
            self.sub_state = "plan"
            self._dialog_attempts = 0
            return action_wait(250, "AP purchase panel ready")

        scr = self.detect_screen_yolo(screen)
        if scr == "Lobby":
            self.sub_state = "open_purchase"
            self._enter_attempts = 0
            return action_wait(200, "on lobby, opening AP purchase")
        if scr not in (None, "Lobby"):
            return action_back(f"ap planning: back from {scr}")

        self._enter_attempts += 1
        if self._enter_attempts > 8:
            return action_done("ap planning skipped: lobby not reached")
        return action_wait(400, "ap planning waiting for lobby")

    def _open_purchase(self, screen: ScreenState) -> Dict[str, Any]:
        """Open the AP buy dialog by clicking the '+' (TOPBAR_PLUS) next to
        the AP/stamina widget. Grey-only plus => AP not buyable now => exit."""
        if self._on_ap_panel(screen):
            self.sub_state = "plan"
            self._dialog_attempts = 0
            return action_wait(250, "entered AP purchase panel")

        # Cooldown so we don't re-tap the plus while the dialog animates in.
        if self._enter_click_cooldown > 0:
            self._enter_click_cooldown -= 1
            return action_wait(500, f"ap planning: open cooldown ({self._enter_click_cooldown})")

        if self.detect_screen_yolo(screen) not in (None, "Lobby"):
            self.sub_state = "enter"
            return action_wait(300, "lost lobby while opening AP panel")

        self._enter_attempts += 1
        if self._enter_attempts > 12:
            self.sub_state = "exit"
            return action_wait(200, "AP purchase panel unavailable, exiting")

        # The AP '+' sits at the top-bar, left half (AP is the leftmost
        # currency). Constrain the region so we grab the AP plus, not the
        # credit/pyroxene one further right.
        ap_plus = self.find_cls(
            screen, UC.TOPBAR_PLUS, conf=0.30, region=(0.0, 0.0, 0.55, 0.12),
        )
        if ap_plus is not None:
            self._enter_click_cooldown = 4
            return action_click_box(ap_plus, "open AP purchase (YOLO 加号)")

        # Grey plus = AP not purchasable (already maxed / no eligibility).
        grey_plus = self.find_cls(
            screen, UC.TOPBAR_PLUS_GREY, conf=0.30, region=(0.0, 0.0, 0.55, 0.12),
        )
        if grey_plus is not None:
            self.log("AP plus is GREY (not buyable now) — exiting")
            self.sub_state = "exit"
            return action_wait(300, "ap planning: plus greyed")

        self.log("no 加号 cls near AP widget — YOLO gap; waiting")
        return action_wait(400, "ap planning: waiting for 加号 detection")

    def _plan(self, screen: ScreenState) -> Dict[str, Any]:
        if not self._on_ap_panel(screen):
            if self.detect_screen_yolo(screen) == "Lobby":
                self.sub_state = "exit"
                return action_wait(250, "AP plan complete on lobby")
            self.sub_state = "open_purchase"
            self._enter_attempts = 0
            return action_wait(300, "AP panel closed, reopening")

        self._dialog_attempts += 1
        if self._dialog_attempts > 20:
            self.log("plan loop limit — cancelling out")
            return self._cancel_panel(screen, "ap planning: plan timeout")

        ap_now = self._read_ap(screen)
        if ap_now is not None:
            self._last_ap = ap_now

        # 1. Confirm pending (we just clicked FREE / a buy → confirm dialog up).
        if self._pending_free_confirm:
            confirm = self.find_cls(
                screen, UC.BTN_CONFIRM, conf=0.30, region=(0.30, 0.50, 0.90, 0.92),
            )
            if confirm is not None:
                self._pending_free_confirm = False
                self._free_collected = True
                self._post_click_wait = 2
                self.log(f"confirm free AP claim (ap={ap_now})")
                return action_click_box(confirm, "confirm daily free AP claim (YOLO 确认键)")
            # Grey confirm = can't claim (already taken / ineligible) → cancel.
            if self.find_cls(
                screen, UC.BTN_CONFIRM_GREY, conf=0.30, region=(0.30, 0.50, 0.90, 0.92),
            ) is not None:
                self._pending_free_confirm = False
                self._free_collected = True  # nothing left to take
                self.log("confirm GREY (free already claimed / ineligible) — cancelling")
                return self._cancel_panel(screen, "ap planning: free unavailable")
            # Confirm dialog not up yet — wait a tick.
            return action_wait(400, "ap planning: waiting for confirm dialog")

        # 2. Claim the FREE tier (UC.FREE cls). This is the only auto-buy.
        if not self._free_collected:
            free = self.find_cls(screen, UC.FREE, conf=0.30)
            if free is not None:
                self._pending_free_confirm = True
                self._post_click_wait = 1
                self.log(f"select free AP (YOLO 免费, ap={ap_now})")
                return action_click_box(free, "claim daily free AP (YOLO 免费)")
            # No FREE cls visible — either already claimed today, or YOLO
            # gap. We cannot tell paid rows apart without OCR (no paid cls),
            # so do NOT guess — treat free as done and exit cleanly.
            self.log("no 免费 cls in AP panel — free already taken or YOLO gap")
            self._free_collected = True

        # 3. Paid AP purchase path.
        #    GAP: ui_classes has NO cls for paid AP tiers (only UC.FREE for
        #    the free one). Identifying a paid row would require OCR-driven
        #    clicking, which is forbidden. So even when policy ALLOWS paid
        #    buys we cannot safely target one via YOLO — skip & surface gap.
        if (not self._forbid_premium_currency) and self._paid_done < self._paid_purchase_limit:
            self.log(
                "paid AP requested but no paid-tier cls exists "
                "(YOLO gap) — refusing to OCR-click paid currency; skipping"
            )

        # 4. Done with the free claim — close the panel safely (never leave
        #    a paid purchase armed).
        return self._cancel_panel(screen, "ap planning complete")

    def _cancel_panel(self, screen: ScreenState, reason: str) -> Dict[str, Any]:
        """Close the AP dialog without buying — prefer 取消, then 叉叉."""
        self._pending_free_confirm = False
        cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=0.30)
        if cancel is not None:
            self.sub_state = "exit"
            return action_click_box(cancel, "cancel AP purchase dialog (YOLO 取消键)")
        close_x = self.find_cls(screen, UC.BTN_CLOSE_X, conf=0.30)
        if close_x is not None:
            self.sub_state = "exit"
            return action_click_box(close_x, "close AP dialog (YOLO 弹窗叉叉)")
        self.sub_state = "exit"
        return action_back(reason)

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log(f"done (free={int(self._free_collected)}, paid={self._paid_done}, ap={self._last_ap})")
            return action_done(
                f"ap planning complete (free={int(self._free_collected)}, paid={self._paid_done})"
            )
        # Prefer YOLO home/back over blind ESC.
        home = self.find_cls(screen, UC.BTN_HOME, conf=0.30)
        if home is not None:
            return action_click_box(home, "ap planning exit: home button")
        back = self.find_cls(screen, UC.BTN_BACK, conf=0.30)
        if back is not None:
            return action_click_box(back, "ap planning exit: back button")
        return action_back("ap planning exit to lobby")
