"""PassRewardSkill: Claim battle-pass (战令/通行证) mission + reward rewards.

Design (2026-05-28 full YOLO rewrite — mail.py is the reference paradigm):
- Every click target resolved via self.find_cls/click_cls (ui_classes cls).
  NO OCR button fallback, NO detect_current_screen (OCR), NO hardcoded blind
  coords for buttons that have a cls.
- Page state inferred from YOLO cls: seeing any CLAIM_* cls = inside pass;
  seeing >=2 lobby nav icons (detect_screen_yolo=="Lobby") = back on lobby.
- If YOLO can't see a needed cls, log + wait (surface the gap) — don't fake
  progress or blind-click.
- Pass has NO numeric counter (unlike mail's X/200), so this skill keeps ZERO
  OCR. Progress is tracked by claim-click count + an idle-streak that flips to
  the next tab/exit once no active claim cls remains.

KNOWN cls GAP (see report): the pass mission/reward TAB-SWITCH buttons have no
dedicated ui_classes cls. We fall back to action_click on the tab's fixed
bottom-bar coords, clearly commented. Once a cls is trained, replace those.

Pass can be unavailable on some servers/accounts; this skill exits quickly via
the enter-timeout in that case.

States: enter → claim_mission → claim_reward → exit
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill,
    ScreenState,
    action_back,
    action_click,
    action_click_box,
    action_done,
    action_wait,
)
from brain.skills import ui_classes as UC


class PassRewardSkill(BaseSkill):
    # Pass rewards live behind the 招募/战令 entry; its lobby icon is 招募入口.
    _LOBBY_DOT_ENTRIES = [UC.NAV_RECRUIT]

    def should_run(self, screen: ScreenState) -> bool:
        return self.dot_on_entry(screen, self._LOBBY_DOT_ENTRIES)

    def __init__(self):
        super().__init__("PassReward")
        self.max_ticks = 60
        self._enter_ticks: int = 0
        self._enter_click_cooldown: int = 0
        self._phase_ticks: int = 0
        self._idle_claim_ticks: int = 0
        self._claim_clicks: int = 0
        self._phase_initialized: bool = False
        self._post_claim_wait: int = 0
        # Entry-clicked gate (see daily_tasks): a claim cls only counts as
        # "inside pass" once we've clicked our own 招募入口 entry. Stops串页
        # off a leftover claim screen or a transient YOLO claim misfire.
        self._entered: bool = False

    def reset(self) -> None:
        super().reset()
        self._enter_ticks = 0
        self._enter_click_cooldown = 0
        self._phase_ticks = 0
        self._idle_claim_ticks = 0
        self._claim_clicks = 0
        self._phase_initialized = False
        self._post_claim_wait = 0
        self._entered = False

    # ── page inference via YOLO cls (no OCR) ──────────────────────────
    def _on_pass_page(self, screen: ScreenState) -> bool:
        """Inside pass iff we've clicked our own entry (self._entered) AND
        we're NOT on lobby AND a claim cls (active or grey) is visible. The
        entry gate keeps a leftover/other-skill claim screen — or a transient
        YOLO claim-cls misfire on lobby — from串页 into pass's in-page判定.
        Pure predicate — the _entered re-arm on lobby happens in _enter."""
        if not self._entered:
            return False
        if self.detect_screen_yolo(screen) == "Lobby":
            return False
        return self.find_cls(
            screen, UC.CLAIM_ACTIVE + UC.CLAIM_DONE, conf=0.20,
        ) is not None

    # ── state machine ─────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        if self.ticks >= self.max_ticks:
            self.log(f"timeout ({self._claim_clicks} claims)")
            return action_done("pass reward timeout")

        # Result popup "獲得獎勵" — YOLO cls, tap to dismiss (no OCR).
        reward_popup = self.find_cls(screen, UC.GOT_REWARD, conf=0.30)
        if reward_popup is not None:
            self._post_claim_wait = 1
            return action_click(0.5, 0.9, "dismiss pass reward popup (YOLO 获得奖励)")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(700, "pass loading")

        if self.sub_state == "":
            self.sub_state = "enter"
        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "claim_mission":
            return self._claim_mission(screen)
        if self.sub_state == "claim_reward":
            return self._claim_reward(screen)
        if self.sub_state == "exit":
            return self._exit(screen)
        return action_wait(300, "pass reward unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks += 1

        # Re-arm the entry gate only on a CONFIRMED return to lobby and only
        # when not mid-entry (cooldown==0), so a still-visible lobby nav bar
        # during the page transition doesn't clear _entered prematurely.
        if self._enter_click_cooldown == 0 and self.detect_screen_yolo(screen) == "Lobby":
            self._entered = False

        if self._on_pass_page(screen):
            self.log("inside pass")
            self.sub_state = "claim_mission"
            self._phase_ticks = 0
            self._idle_claim_ticks = 0
            self._phase_initialized = False
            self._post_claim_wait = 2
            return action_wait(300, "entered pass")

        # Cooldown so we don't spam the entry icon while the page loads.
        if self._enter_click_cooldown > 0:
            self._enter_click_cooldown -= 1
            return action_wait(500, f"pass: enter cooldown ({self._enter_click_cooldown})")

        # YOLO 招募入口 cls (战令/通行证 entry).
        recruit = self.find_cls(screen, UC.NAV_RECRUIT, conf=0.30)
        if recruit is not None:
            self._enter_click_cooldown = 4
            self._entered = True  # entry-clicked gate: now in-page is trusted
            return action_click_box(recruit, "open pass (YOLO 招募入口)")

        # Not lobby + not pass → back out toward lobby.
        if self.detect_screen_yolo(screen) not in (None, "Lobby"):
            return action_back("pass: backing toward lobby")

        # Entry icon not seen. On servers without a pass this never appears,
        # so bail out instead of looping forever.
        if self._enter_ticks > 10:
            self.log("招募入口 not found — pass unavailable / YOLO gap; exiting")
            self.sub_state = "exit"
            return action_wait(200, "pass unavailable")
        return action_wait(400, "pass: waiting for 招募入口 detection")

    def _claim_current_tab(
        self, screen: ScreenState, phase_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Claim whatever is claimable on the current tab. Returns an action
        while there's work, or None once the tab is drained (idle streak)."""
        if self._post_claim_wait > 0:
            self._post_claim_wait -= 1
            return action_wait(500, f"pass {phase_name}: settling ({self._post_claim_wait})")

        # 一键领取/全部领取 (claim-all) — bottom-right CTA zone.
        claim_all = self.find_cls(
            screen, UC.CLAIM_ALL_YELLOW, conf=0.20,
            region=(0.55, 0.78, 1.00, 1.00),
        )
        if claim_all is not None:
            self._claim_clicks += 1
            self._idle_claim_ticks = 0
            self._post_claim_wait = 3
            self.log(f"{phase_name}: claim all ({self._claim_clicks}, cls={claim_all.cls_name})")
            return action_click_box(claim_all, f"pass {phase_name}: claim all (YOLO {claim_all.cls_name})")

        # Per-row single claim (领取_黄 / 领取奖励_黄).
        single = self.find_cls(
            screen, [UC.CLAIM_YELLOW, UC.CLAIM_REWARD_YELLOW], conf=0.25,
            region=(0.55, 0.15, 1.00, 0.95),
        )
        if single is not None:
            self._claim_clicks += 1
            self._idle_claim_ticks = 0
            self._post_claim_wait = 3
            self.log(f"{phase_name}: claim single ({self._claim_clicks}, cls={single.cls_name})")
            return action_click_box(single, f"pass {phase_name}: claim single (YOLO {single.cls_name})")

        # No active claim cls. If we can positively see the done/grey state,
        # the tab is finished — stop quickly instead of idling the full streak.
        if self.find_cls(screen, UC.CLAIM_DONE, conf=0.25) is not None:
            self.log(f"{phase_name}: only grey claim cls — tab done")
            return None

        # No claim cls at all: surface as a brief YOLO gap, then move on.
        self._idle_claim_ticks += 1
        if self._idle_claim_ticks > 4:
            self.log(f"{phase_name}: no claim cls (YOLO gap) — moving on")
            return None
        return action_wait(400, f"pass {phase_name}: waiting for claim cls (YOLO gap)")

    def _claim_mission(self, screen: ScreenState) -> Dict[str, Any]:
        self._phase_ticks += 1

        if not self._phase_initialized:
            self._phase_initialized = True
            self._idle_claim_ticks = 0
            self._post_claim_wait = 1
            # GAP: pass 任务 tab has no dedicated ui_classes cls. Fixed
            # bottom-bar coord (任务 tab, left-of-center) is a documented
            # blind click — replace with click_cls once a tab cls is trained.
            return action_click(0.30, 0.93, "switch to pass 任务 tab (GAP: no cls, fixed coord)")

        claim_action = self._claim_current_tab(screen, "mission")
        if claim_action:
            return claim_action

        if self._phase_ticks > 16:
            self.log("mission claim phase timeout, moving to reward tab")

        self.sub_state = "claim_reward"
        self._phase_ticks = 0
        self._idle_claim_ticks = 0
        self._phase_initialized = False
        return action_wait(250, "mission claims complete")

    def _claim_reward(self, screen: ScreenState) -> Dict[str, Any]:
        self._phase_ticks += 1

        if not self._phase_initialized:
            self._phase_initialized = True
            self._idle_claim_ticks = 0
            self._post_claim_wait = 1
            # GAP: pass 奖励 tab has no dedicated ui_classes cls. Fixed
            # bottom-bar coord (奖励 tab, right-of-center) is a documented
            # blind click — replace with click_cls once a tab cls is trained.
            return action_click(0.50, 0.93, "switch to pass 奖励 tab (GAP: no cls, fixed coord)")

        claim_action = self._claim_current_tab(screen, "reward")
        if claim_action:
            return claim_action

        if self._phase_ticks > 16:
            self.log("reward claim phase timeout")

        self.sub_state = "exit"
        return action_wait(250, "pass reward claims complete")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        # On lobby iff we see >=2 nav icons.
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log(f"done ({self._claim_clicks} claims)")
            return action_done(f"pass reward complete ({self._claim_clicks} claims)")
        # Prefer YOLO home/back button over blind ESC.
        home = self.find_cls(screen, UC.BTN_HOME, conf=0.30)
        if home is not None:
            return action_click_box(home, "pass exit: home button")
        back = self.find_cls(screen, UC.BTN_BACK, conf=0.30)
        if back is not None:
            return action_click_box(back, "pass exit: back button")
        return action_back("pass reward exit: ESC toward lobby")
