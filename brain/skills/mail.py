"""MailSkill — claim all mailbox rewards (pure-YOLO rewrite).

Verified flow (interactive probe 2026-06-01, data/_mail_probe_log.md). Mail is
the daily收口: bounty / JFD / arena / 社團 rewards all funnel into the mailbox,
so it must claim everything. NO OCR — the probe found a clean pure-YOLO done
signal: 一次領取 going grey (CLAIM_ONCE_GREY) means the queue is drained. The
old X/200 digit read was a progress crutch and is dropped (it relied on
full-frame OCR which is disabled in pure-YOLO mode anyway).

The old settings-popup-repeat / timeout bug did NOT reproduce in the probe;
the root cause was re-tapping the envelope icon (which becomes a ⚙ gear inside
the mailbox). We avoid it by only clicking NAV_MAIL while on the lobby.

State machine
-------------
enter   lobby → click NAV_MAIL (邮件箱). Envelope cls is weak (19f) so fall
        back to the red dot in the top-right mail zone. Wait for the Mail page.
claim   CLAIM_ONCE_YELLOW (一次领取黄色) → click (claims all). GOT_REWARD popup
        → dismiss via header / 点击继续字样 (handled globally). Done when only
        CLAIM_ONCE_GREY (一次领取灰色) remains = queue drained.
exit    BTN_HOME / BTN_BACK → lobby → done.

Detectors: base "ui" only.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click, action_click_box, action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC

_CLS_CONF = 0.30
_MAIL_ZONE = (0.86, 0.0, 0.97, 0.09)   # top-right envelope + its red dot

_ENTER_MAX = 20
_CLAIM_MAX = 30
_EXIT_MAX = 14


class MailSkill(BaseSkill):

    def should_run(self, screen: ScreenState) -> bool:
        # Run when a red dot sits in the top-right mail zone (reliable), else
        # defer to dot_on_entry (True when the envelope cls isn't even visible).
        if self.find_cls(screen, UC.DOT_RED, conf=0.35, region=_MAIL_ZONE) is not None:
            return True
        return self.dot_on_entry(screen, [UC.NAV_MAIL])

    def __init__(self):
        super().__init__("Mail")
        self.max_ticks = 70
        self._init_state()

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._claims: int = 0

    def reset(self) -> None:
        super().reset()
        self._init_state()

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0

    def _on_mail(self, screen: ScreenState) -> bool:
        return self.detect_screen_yolo(screen) == "Mail"

    def _dismiss_reward(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """GOT_REWARD popup → tap 点击继续字样 / 获得奖励 header (NEVER center)."""
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
        if cont is not None:
            return action_click_box(cont, "dismiss reward via continue")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if got is not None:
            return action_click_box(got, "dismiss reward via header")
        return None

    # ── state machine ──────────────────────────────────────────────────────

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log(f"timeout (claims={self._claims})")
            return action_done("mail timeout")

        # Global: reward-result popup can appear after any claim.
        reward = self._dismiss_reward(screen)
        if reward is not None:
            self.log("dismissing reward popup")
            return reward

        if screen.is_loading():
            return action_wait(700, "mail loading")

        if self.sub_state == "":
            self._goto("enter")

        handler = {
            "enter": self._enter,
            "claim": self._claim,
            "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "mail unknown state")
        return handler(screen)

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._on_mail(screen):
            self.log("inside mailbox → claim")
            self._goto("claim")
            return action_wait(400, "entered mail")

        if screen.is_lobby():
            mail_btn = self.find_cls(screen, UC.NAV_MAIL, conf=_CLS_CONF)
            if mail_btn is not None:
                self.log("opening mail (YOLO 邮件箱)")
                return action_click_box(mail_btn, "open mailbox")
            # Envelope cls missed but its red dot is in the mail zone — click
            # the dot's anchor (its own center) to open the mailbox.
            dot = self.find_cls(screen, UC.DOT_RED, conf=0.35, region=_MAIL_ZONE)
            if dot is not None:
                self.log("opening mail via red-dot anchor (envelope cls missed)")
                return action_click(dot.cx, min(0.06, dot.cy + 0.02), "open mailbox (dot)")
            self.log("on lobby but no 邮件箱/红点 — YOLO gap; waiting")
            return action_wait(400, "waiting for 邮件箱 cls")

        if self._phase_ticks > _ENTER_MAX:
            self.log("enter budget exhausted, giving up")
            return action_done("could not reach mailbox")
        if len(screen.yolo_boxes or []) < 2:
            return action_wait(700, "no UI detected, likely loading")
        return action_back("mail: recover toward lobby")

    def _claim(self, screen: ScreenState) -> Dict[str, Any]:
        if not self._on_mail(screen):
            if screen.is_lobby():
                self._goto("enter")
                return action_wait(300, "claim: back on lobby, re-enter")
            if self._phase_ticks > _CLAIM_MAX:
                self._goto("exit")
                return action_wait(300, "claim lost mail → exit")
            return action_wait(400, "waiting for mail UI (claim)")

        # Claim-all (一次领取黄色) — one tap claims the whole unclaimed queue.
        claim_all = self.find_cls(screen, UC.CLAIM_ONCE_YELLOW, conf=_CLS_CONF)
        if claim_all is not None:
            self._claims += 1
            self.log(f"claim all mail (#{self._claims}, YOLO 一次领取黄色)")
            return action_click_box(claim_all, "claim all mail")

        # Claim-all greyed = queue drained = DONE.
        if self.find_cls(screen, UC.CLAIM_ONCE_GREY, conf=_CLS_CONF) is not None:
            self.log(f"一次领取灰色 → mailbox drained (claims={self._claims})")
            self._goto("exit")
            return action_wait(300, "mail drained → exit")

        # Neither yellow nor grey claim-all visible yet — settle, then bail.
        if self._phase_ticks > _CLAIM_MAX:
            self.log("no claim-all cls found, exiting")
            self._goto("exit")
            return action_wait(300, "no claim cls → exit")
        return action_wait(400, "waiting for claim-all cls")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log(f"done (claims={self._claims})")
            return action_done(f"mail complete (claims={self._claims})")
        if self._phase_ticks > _EXIT_MAX:
            self.log("exit budget exhausted, reporting done")
            return action_done("mail exit timeout")
        home = self.find_cls(screen, UC.BTN_HOME, conf=_CLS_CONF)
        if home is not None:
            return action_click_box(home, "mail exit: home button")
        back = self.find_cls(screen, UC.BTN_BACK, conf=_CLS_CONF)
        if back is not None:
            return action_click_box(back, "mail exit: back button")
        return action_back("mail exit: ESC toward lobby")
