"""ClubSkill: Claim daily AP from club (社團/社交). YOLO-only clicks.

Design (2026-05-28 full YOLO rewrite — see docs/yolo_migration_spec.md):
- Every click target resolved via self.find_cls/click_cls (ui_classes cls).
- NO OCR at all: club has no digit counter to read (unlike mail's X/200),
  and all buttons (社交入口 / 社團 / 領取_黃 / 綠勾) have YOLO cls. So this
  skill is purely cls-driven.
- No detect_current_screen (OCR). detect_screen_yolo() has no dedicated
  "Club" page, so we infer club state heuristically: after clicking
  NAV_SOCIAL we're off-lobby AND can see CLUB or a CLAIM cls.
- If YOLO can't see a needed cls, log + wait (surface the gap) — never
  OCR-fallback, never blind hardcoded clicks.
- The club daily sign-in popup (社團簽到獎勵) + bond level-up are handled
  globally by _handle_common_popups.

States: enter → claim → exit
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click_box, action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC


class ClubSkill(BaseSkill):
    _LOBBY_DOT_ENTRIES = [UC.NAV_SOCIAL]

    def should_run(self, screen):
        return self.dot_on_entry(screen, self._LOBBY_DOT_ENTRIES)

    def __init__(self):
        super().__init__("Club")
        self.max_ticks = 30
        self._claim_attempts: int = 0
        self._claimed: bool = False
        self._enter_click_cooldown: int = 0
        self._post_claim_wait: int = 0
        # Entry-clicked gate (see daily_tasks): a claim cls only counts as
        # "inside club" once we've clicked our own social/club entry. Stops
        #串页 off a leftover claim screen or a transient YOLO claim misfire.
        self._entered: bool = False

    def reset(self) -> None:
        super().reset()
        self._claim_attempts = 0
        self._claimed = False
        self._enter_click_cooldown = 0
        self._post_claim_wait = 0
        self._entered = False

    # ── page inference via YOLO cls (no OCR) ──────────────────────────
    def _on_club_page(self, screen: ScreenState) -> bool:
        """Inside club iff we've clicked our own entry (self._entered) AND
        (off-lobby) AND we can see the 社團 tile or any claim cls
        (active/grey/green-check). The entry gate keeps a leftover/other-skill
        claim screen from串页 into club's in-page判定. Pure predicate — the
        _entered re-arm on lobby happens in _enter (cooldown-guarded) so the
        social overlay (lobby icons still showing underneath) doesn't reset it
        mid-entry."""
        if not self._entered:
            return False
        if self.detect_screen_yolo(screen) == "Lobby":
            return False
        return self.find_cls(
            screen,
            [UC.CLUB] + UC.CLAIM_ACTIVE + UC.CLAIM_DONE
            + [UC.GREEN_CHECK, UC.CLAIM_REWARD_GREY],
            conf=0.20,
        ) is not None

    # ── state machine ─────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        if self.ticks >= self.max_ticks:
            self.log(f"timeout (claimed={self._claimed})")
            return action_done("club timeout")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "club loading")

        if self.sub_state == "":
            self.sub_state = "enter"
        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "claim":
            return self._claim(screen)
        if self.sub_state == "exit":
            return self._exit(screen)
        return action_wait(300, "club unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        # Re-arm the entry gate only on a CONFIRMED return to lobby, and only
        # when not mid-entry (cooldown==0). Guarding on cooldown stops the
        # social overlay — which still shows lobby nav icons underneath — from
        # clearing _entered the tick after we click 社交入口.
        if self._enter_click_cooldown == 0 and self.detect_screen_yolo(screen) == "Lobby":
            self._entered = False

        if self._on_club_page(screen):
            self.log("inside club")
            self.sub_state = "claim"
            self._post_claim_wait = 2
            return action_wait(500, "entered club")

        # Cooldown so we don't spam the social/club entry while the overlay
        # is animating in.
        if self._enter_click_cooldown > 0:
            self._enter_click_cooldown -= 1
            return action_wait(500, f"club: enter cooldown ({self._enter_click_cooldown})")

        # Step 2: social overlay open (社團 tile visible) → click it.
        club_btn = self.find_cls(screen, UC.CLUB, conf=0.30)
        if club_btn is not None:
            self._enter_click_cooldown = 3
            self._entered = True  # entry-clicked gate: now in-page is trusted
            return action_click_box(club_btn, f"open club (YOLO {UC.CLUB} {club_btn.confidence:.2f})")

        # Step 1: on lobby → click 社交入口 to open the social overlay.
        social_btn = self.find_cls(screen, UC.NAV_SOCIAL, conf=0.30)
        if social_btn is not None:
            self._enter_click_cooldown = 3
            self._entered = True  # entry-clicked gate: now in-page is trusted
            return action_click_box(social_btn, f"open social (YOLO {UC.NAV_SOCIAL} {social_btn.confidence:.2f})")

        # Not lobby + not club + no social entry → back out toward lobby.
        if self.detect_screen_yolo(screen) not in (None, "Lobby"):
            return action_back("club: backing toward lobby")
        self.log("no 社交入口/社團 cls — YOLO gap; waiting")
        return action_wait(400, "club: waiting for 社交入口/社團 detection")

    def _claim(self, screen: ScreenState) -> Dict[str, Any]:
        self._claim_attempts += 1

        if self._post_claim_wait > 0:
            self._post_claim_wait -= 1
            return action_wait(500, f"club: settling ({self._post_claim_wait})")

        # Already claimed: green check or greyed claim button → done.
        if self.find_cls(
            screen, [UC.GREEN_CHECK] + UC.CLAIM_DONE, conf=0.25,
        ) is not None:
            self.log("club AP already claimed (绿勾/灰色领取)")
            self.sub_state = "exit"
            return action_wait(300, "already claimed")

        # Result popup "獲得獎勵" — tap to dismiss, then we're done.
        got = self.find_cls(screen, UC.GOT_REWARD, conf=0.30)
        if got is not None:
            self._claimed = True
            self.sub_state = "exit"
            return action_click_box(got, f"dismiss reward popup (YOLO {UC.GOT_REWARD})")

        # Claim the AP reward. Club uses the yellow single/reward claim cls.
        claim = self.find_cls(
            screen,
            [UC.CLAIM_YELLOW, UC.CLAIM_REWARD_YELLOW,
             UC.CLAIM_ALL_YELLOW, UC.CLAIM_BLUE],
            conf=0.25,
        )
        if claim is not None:
            self._claimed = True
            self.log(f"claiming club AP (#{self._claim_attempts}, cls={claim.cls_name})")
            self._post_claim_wait = 3
            return action_click_box(claim, f"claim club AP (YOLO {claim.cls_name})")

        # No claim cls visible. Give it a few ticks (page may still load),
        # then exit — surfaces a gap if claim cls truly never appears.
        if self._claim_attempts > 8:
            self.log(f"no claim cls after {self._claim_attempts} attempts (claimed={self._claimed}) — YOLO gap")
            self.sub_state = "exit"
            return action_wait(300, "no club claim cls found")
        return action_wait(400, "club: waiting for claim cls (YOLO gap)")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        # On lobby iff we see >=2 nav icons.
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log(f"done (claimed={self._claimed})")
            return action_done(f"club complete (claimed={self._claimed})")
        # Prefer YOLO home/back button over blind ESC.
        home = self.find_cls(screen, UC.BTN_HOME, conf=0.30)
        if home is not None:
            return action_click_box(home, "club exit: home button")
        back = self.find_cls(screen, UC.BTN_BACK, conf=0.30)
        if back is not None:
            return action_click_box(back, "club exit: back button")
        return action_back("club exit: ESC toward lobby")
