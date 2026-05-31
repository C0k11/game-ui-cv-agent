"""CraftSkill: Quick-craft items and claim finished crafts. YOLO-only clicks.

Design (2026-05-28 full YOLO rewrite):
- Every click target resolved via self.find_cls/click_cls (ui_classes cls).
  No OCR button fallback, no detect_current_screen (OCR). Page state is
  inferred from YOLO cls via detect_screen_yolo() == "Craft".
- If YOLO can't see a needed cls, log + wait (surface the gap) — don't
  fall back to OCR or blind hardcoded clicks.
- should_run always returns True: 制造 doesn't gate on a 黄点 — we always
  enter, quick-craft, and claim (user spec 2026-05-28).

Flow / states:
1. enter:        From lobby, click 制造入口 (NAV_CRAFT) → wait for Craft page.
2. claim:        Claim finished crafts via CLAIM_ALL_YELLOW / CLAIM_YELLOW /
                 CLAIM_REWARD_YELLOW.
3. quick_craft:  快速制造 (CRAFT_QUICK) → 开始制造 (CRAFT_START) → 确认键.
4. claim_after:  Claim newly finished crafts; loop more cycles up to _max_crafts.
5. exit:         Back to lobby (BTN_HOME / BTN_BACK).
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click_box, action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC


# Any yellow claim variant seen on the craft page = something to collect.
_CRAFT_CLAIM_ACTIVE = [
    UC.CLAIM_ALL_YELLOW, UC.CLAIM_YELLOW, UC.CLAIM_REWARD_YELLOW,
]
# Either start-craft CTA = a craft slot is ready to (re)start.
_CRAFT_CTA = [UC.CRAFT_QUICK, UC.CRAFT_START]


class CraftSkill(BaseSkill):
    def should_run(self, screen: ScreenState) -> bool:
        # Craft is NOT dot-gated — always enter to quick-craft + claim.
        return True

    def __init__(self):
        super().__init__("Craft")
        self.max_ticks = 80
        self._claim_count: int = 0
        self._craft_started: bool = False
        self._craft_ticks: int = 0
        self._craft_cycles: int = 0
        self._max_crafts: int = 3
        self._pending_cycle_started: bool = False
        self._enter_click_cooldown: int = 0
        # Entry-clicked gate: PAGE_SIGNATURES["Craft"] is just the two CTAs
        # (快速制造/开始制造). When all 3 slots are mid-cooldown neither CTA
        # renders, so detect_screen_yolo() never returns "Craft" and _enter
        # would spam NAV_CRAFT to max_ticks. Once we've clicked our own entry
        # and we're off-lobby, treat that as on-page even with no CTA visible.
        self._entered: bool = False

    def reset(self) -> None:
        super().reset()
        self._claim_count = 0
        self._craft_started = False
        self._craft_ticks = 0
        self._craft_cycles = 0
        self._pending_cycle_started = False
        self._enter_click_cooldown = 0
        self._entered = False

    # ── page inference via YOLO cls (no OCR) ──────────────────────────
    def _is_craft_screen(self, screen: ScreenState) -> bool:
        """On the craft page when EITHER the YOLO page signature matches
        (a CTA / claim cls is visible) OR we've clicked our own entry and
        we're off-lobby (covers the all-slots-cooling case where neither
        快速制造 nor 开始制造 renders → signature alone never fires)."""
        page = self.detect_screen_yolo(screen)
        if page == "Lobby":
            # Back on lobby → re-arm the entry gate.
            self._entered = False
            return False
        if page == "Craft":
            return True
        # A claim cls is a positive craft-page marker on its own.
        if self.find_cls(screen, _CRAFT_CLAIM_ACTIVE, conf=0.25) is not None:
            return True
        # No CTA signature (all slots cooling, or YOLO gap). Trust the entry
        # gate: we clicked 制造入口 and we're not on lobby → we're on craft.
        # Only once the post-click cooldown has elapsed, so we don't latch
        # "on-page" during the load transition (page momentarily None).
        if self._entered and page is None and self._enter_click_cooldown == 0:
            return True
        return False

    # ── state machine ─────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("craft timeout")

        # Reward result popup ("獲得獎勵") — tap to dismiss (YOLO cls).
        reward = self.find_cls(screen, UC.GOT_REWARD, conf=0.30)
        if reward is not None:
            return action_click_box(reward, "dismiss reward popup (YOLO 获得奖励)")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "craft loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "claim":
            return self._claim(screen)
        if self.sub_state == "quick_craft":
            return self._quick_craft(screen)
        if self.sub_state == "claim_after":
            return self._claim_after(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "craft unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._is_craft_screen(screen):
            self.log("inside craft")
            self.sub_state = "claim"
            return action_wait(500, "entered craft")

        # Cooldown so we don't spam-tap the entry icon between frames.
        if self._enter_click_cooldown > 0:
            self._enter_click_cooldown -= 1
            return action_wait(500, f"craft: enter cooldown ({self._enter_click_cooldown})")

        page = self.detect_screen_yolo(screen)
        if page == "Lobby":
            # 制造入口 in the bottom nav bar.
            act = self.click_cls(screen, UC.NAV_CRAFT, "open craft", conf=0.30)
            if act is not None:
                self._enter_click_cooldown = 3
                self._entered = True  # entry-clicked gate: now on-page is trusted
                return act
            self.log("no 制造入口 on lobby — YOLO gap; waiting")
            return action_wait(400, "craft: waiting for 制造入口 detection")

        # On some other page → back out toward lobby.
        if page is not None and page != "Craft":
            return action_back(f"craft: backing from {page}")

        self.log("unknown screen — backing toward lobby")
        return action_back("craft: backing toward lobby")

    def _claim(self, screen: ScreenState) -> Dict[str, Any]:
        """Claim all finished craft items via CLAIM_* yellow cls."""
        self._craft_ticks += 1

        if self._craft_ticks > 8:
            self.log("claim phase done, moving to quick craft")
            self.sub_state = "quick_craft"
            self._craft_ticks = 0
            return action_wait(300, "claim done")

        claim = self.find_cls(screen, _CRAFT_CLAIM_ACTIVE, conf=0.25)
        if claim is not None:
            self._claim_count += 1
            self.log(f"claiming finished crafts (#{self._claim_count}, cls={claim.cls_name})")
            return action_click_box(claim, f"claim finished crafts (YOLO {claim.cls_name})")

        # No claim cls — nothing to collect, proceed to quick craft.
        self.log("no finished crafts to claim")
        self.sub_state = "quick_craft"
        self._craft_ticks = 0
        return action_wait(300, "no crafts to claim, moving to quick craft")

    def _quick_craft(self, screen: ScreenState) -> Dict[str, Any]:
        """快速制造 (CRAFT_QUICK) → 开始制造 (CRAFT_START) → 确认键."""
        self._craft_ticks += 1

        if self._craft_ticks > 12:
            self.log("quick craft phase timeout")
            self.sub_state = "claim_after"
            self._craft_ticks = 0
            return action_wait(300, "quick craft timeout")

        if self._craft_started:
            # After 开始制造, look for the confirm popup OR a claim cls.
            confirm = self.find_cls(
                screen, UC.BTN_CONFIRM, conf=0.30, region=screen.CENTER,
            )
            if confirm is not None:
                self.log("confirming craft")
                if self._pending_cycle_started:
                    self._craft_cycles += 1
                    self._pending_cycle_started = False
                self.sub_state = "claim_after"
                self._craft_ticks = 0
                return action_click_box(confirm, "confirm craft (YOLO 确认键)")

            # Grey confirm = can't craft (insufficient mats) — bail to claim.
            grey = self.find_cls(
                screen, UC.BTN_CONFIRM_GREY, conf=0.30, region=screen.CENTER,
            )
            if grey is not None:
                self.log("confirm is grey (insufficient) — skipping craft")
                self._pending_cycle_started = False
                self.sub_state = "claim_after"
                self._craft_ticks = 0
                return action_wait(300, "craft confirm greyed out")

            # Craft may complete without a confirm popup — claim directly.
            claim = self.find_cls(screen, _CRAFT_CLAIM_ACTIVE, conf=0.25)
            if claim is not None:
                self.log("craft completed, claiming")
                if self._pending_cycle_started:
                    self._craft_cycles += 1
                    self._pending_cycle_started = False
                self.sub_state = "claim_after"
                self._craft_ticks = 0
                return action_click_box(claim, f"claim after quick craft (YOLO {claim.cls_name})")

            # Back on the main craft screen (craft started silently).
            if self._craft_ticks >= 5 and self._is_craft_screen(screen):
                self.log("craft done, back on main screen")
                if self._pending_cycle_started:
                    self._craft_cycles += 1
                    self._pending_cycle_started = False
                self.sub_state = "claim_after"
                self._craft_ticks = 0
                return action_wait(300, "craft completed, claiming")

            return action_wait(500, "waiting for craft confirm popup")

        # 开始制造 button.
        start = self.find_cls(screen, UC.CRAFT_START, conf=0.30)
        if start is not None:
            self.log("clicking 开始制造")
            self._craft_started = True
            self._pending_cycle_started = True
            return action_click_box(start, "start craft (YOLO 开始制造)")

        # 快速制造 button.
        quick = self.find_cls(screen, UC.CRAFT_QUICK, conf=0.30)
        if quick is not None:
            self.log("clicking 快速制造")
            return action_click_box(quick, "click quick craft (YOLO 快速制造)")

        # No craft CTA cls — slots in use / unavailable, or YOLO gap.
        self.log("no craft CTA cls (快速制造/开始制造) — done or YOLO gap")
        self.sub_state = "exit"
        return action_wait(300, "no quick craft available")

    def _claim_after(self, screen: ScreenState) -> Dict[str, Any]:
        """Claim items after quick craft, then loop more cycles if possible."""
        self._craft_ticks += 1

        if self._craft_ticks > 8:
            self.log("post-craft claim phase timeout")
            # Continue crafting more slots if a craft CTA is still present.
            if self._craft_cycles < self._max_crafts and \
                    self.find_cls(screen, _CRAFT_CTA, conf=0.30) is not None:
                self.log(f"continuing craft cycle {self._craft_cycles + 1}/{self._max_crafts}")
                self.sub_state = "quick_craft"
                self._craft_started = False
                self._pending_cycle_started = False
                self._craft_ticks = 0
                return action_wait(250, "continue next craft cycle")
            self.sub_state = "exit"
            return action_wait(300, "post-craft claim done")

        # Claim newly finished crafts.
        claim = self.find_cls(screen, _CRAFT_CLAIM_ACTIVE, conf=0.25)
        if claim is not None:
            self._claim_count += 1
            self.log(f"claiming post-craft items (#{self._claim_count}, cls={claim.cls_name})")
            return action_click_box(claim, f"claim post-craft items (YOLO {claim.cls_name})")

        # No claim cls — loop another craft cycle if a CTA is available.
        if self._craft_cycles < self._max_crafts and \
                self.find_cls(screen, _CRAFT_CTA, conf=0.30) is not None:
            self.log(f"starting next craft cycle {self._craft_cycles + 1}/{self._max_crafts}")
            self.sub_state = "quick_craft"
            self._craft_started = False
            self._pending_cycle_started = False
            self._craft_ticks = 0
            return action_wait(250, "continue crafting")

        self.log(f"craft complete ({self._claim_count} claims, {self._craft_cycles} cycles)")
        self.sub_state = "exit"
        return action_wait(300, "post-craft claim complete")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        # On lobby iff detect_screen_yolo sees >=2 nav icons.
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log(f"done ({self._claim_count} claims, {self._craft_cycles} cycles)")
            return action_done("craft complete")
        # Prefer YOLO home/back button over blind ESC.
        home = self.find_cls(screen, UC.BTN_HOME, conf=0.30)
        if home is not None:
            return action_click_box(home, "craft exit: home button")
        back = self.find_cls(screen, UC.BTN_BACK, conf=0.30)
        if back is not None:
            return action_click_box(back, "craft exit: back button")
        return action_back("craft exit: ESC toward lobby")
