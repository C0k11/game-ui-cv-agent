"""MailSkill: Claim all mail rewards. YOLO-only clicks, OCR for digits only.

Design (2026-05-28 full YOLO rewrite):
- Every click target resolved via self.find_cls/click_cls (ui_classes cls).
- The ONLY OCR call is _read_mail_count() reading "X/200" digits — used to
  verify each claim actually drains the queue (real progress, not timeout).
- No OCR button fallback, no detect_current_screen (OCR). Page state is
  inferred from YOLO cls: seeing CLAIM_* cls = inside mail; seeing NAV_MAIL
  or its 红点 = on lobby.
- If YOLO can't see a needed cls, log + wait (surface gap) — don't fake-finish.

States: enter → claim → exit
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC


_COUNT_RE = re.compile(r"^(\d{1,3})\s*/\s*\d{2,3}$")


class MailSkill(BaseSkill):
    _LOBBY_DOT_ENTRIES = [UC.NAV_MAIL]

    def should_run(self, screen):
        return self.dot_on_entry(screen, self._LOBBY_DOT_ENTRIES)

    def __init__(self):
        super().__init__("Mail")
        self.max_ticks = 60
        self._last_count: Optional[int] = None
        self._stuck_streak: int = 0
        self._claim_attempts: int = 0
        self._claimed_count: int = 0
        self._post_claim_wait: int = 0
        self._enter_click_cooldown: int = 0

    def reset(self) -> None:
        super().reset()
        self._last_count = None
        self._stuck_streak = 0
        self._claim_attempts = 0
        self._claimed_count = 0
        self._post_claim_wait = 0
        self._enter_click_cooldown = 0

    # ── OCR: digits only (the one allowed OCR use) ────────────────────
    def _read_mail_count(self, screen: ScreenState) -> Optional[int]:
        """Read X/200 unclaimed-mail counter (top-right). ONLY OCR call."""
        boxes = screen.find_text(
            r"^\d{1,3}\s*/\s*\d{2,3}$",
            region=(0.85, 0.04, 1.0, 0.13), min_conf=0.5,
        )
        for b in boxes:
            m = _COUNT_RE.match((b.text or "").replace(" ", ""))
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
        return None

    # ── page inference via YOLO cls (no OCR) ──────────────────────────
    def _on_mail_page(self, screen: ScreenState) -> bool:
        """Inside mail iff any claim cls (active or grey) is visible."""
        return self.find_cls(
            screen, UC.CLAIM_ACTIVE + UC.CLAIM_DONE, conf=0.20,
        ) is not None or self._read_mail_count(screen) is not None

    # ── state machine ─────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        # Battle-active guard (orphan battle) — keep as OCR since AUTO/timer
        # text isn't a ui_v1 cls; cheap + rare.
        if screen.find_any_text(["AUTO", "Auto", "戰鬥時間", "战斗时间"], min_conf=0.6):
            return action_wait(800, "mail: battle in progress, waiting")

        self.ticks += 1
        if self.ticks >= self.max_ticks:
            self.log(f"timeout (claimed={self._claimed_count}, last_count={self._last_count})")
            return action_done(f"mail timeout (last count={self._last_count})")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "mail loading")

        if self.sub_state == "":
            self.sub_state = "enter"
        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "claim":
            return self._claim(screen)
        if self.sub_state == "exit":
            return self._exit(screen)
        return action_wait(300, "mail unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._on_mail_page(screen):
            self.log("inside mail")
            self.sub_state = "claim"
            self._post_claim_wait = 2
            return action_wait(500, "entered mail")

        # Cooldown so we don't re-tap mail icon (becomes ⚙ Settings in-page).
        if self._enter_click_cooldown > 0:
            self._enter_click_cooldown -= 1
            return action_wait(500, f"mail: enter cooldown ({self._enter_click_cooldown})")

        # YOLO 邮件箱 cls (envelope icon).
        mail_btn = self.find_cls(screen, UC.NAV_MAIL, conf=0.30)
        if mail_btn is not None:
            self._enter_click_cooldown = 4
            return action_click_box(mail_btn, "open mail (YOLO 邮件箱)")

        # Fallback: ui_v1 misses the small envelope but catches its 红点
        # badge. Clicking the badge corner opens a tooltip, so click the
        # envelope center (0.91, 0.04) when a 红点 sits in the mail zone.
        mail_dot = self.find_cls(
            screen, UC.DOT_RED, conf=0.6, region=(0.86, 0.0, 0.96, 0.08),
        )
        if mail_dot is not None:
            self._enter_click_cooldown = 4
            self.log(f"mail 红点 at ({mail_dot.cx:.3f},{mail_dot.cy:.3f}) → envelope center")
            return action_click(0.91, 0.04, "open mail (envelope center, 红点 confirmed)")

        # Not lobby + not mail → back out toward lobby.
        if self.detect_screen_yolo(screen) not in (None, "Lobby"):
            return action_back("mail: backing toward lobby")
        self.log("no 邮件箱/红点 in mail zone — YOLO gap; waiting")
        return action_wait(400, "mail: waiting for 邮件箱 detection")

    def _claim(self, screen: ScreenState) -> Dict[str, Any]:
        self._claim_attempts += 1

        # Preemptive: close any popup covering the claim button (选项/通知).
        # Central X only — avoid the top-right close-mail X.
        cover_x = self.find_cls(
            screen, UC.BTN_CLOSE_X, conf=0.40, region=(0.30, 0.05, 0.92, 0.50),
        )
        if cover_x is not None:
            self._post_claim_wait = 2
            return action_click_box(cover_x, "close covering popup (mail YOLO X)")

        if self._post_claim_wait > 0:
            self._post_claim_wait -= 1
            return action_wait(500, f"mail: settling ({self._post_claim_wait})")

        # OCR count — real progress signal.
        cur = self._read_mail_count(screen)
        if cur is not None:
            if self._last_count is not None and cur < self._last_count:
                self._claimed_count += (self._last_count - cur)
                self._stuck_streak = 0
            elif self._last_count is not None and cur == self._last_count and cur > 0:
                self._stuck_streak += 1
            self._last_count = cur
            if cur == 0:
                self.log("mailbox drained")
                self.sub_state = "exit"
                return action_wait(300, "all mail claimed")
            if self._stuck_streak >= 2:
                self.log(f"BLOCKED — count stuck at {cur}/200 (likely inventory full)")
                self.sub_state = "exit"
                return action_wait(300, f"mail blocked at {cur}/200")

        # Claim-all (一次領取). ui_v1 mislabels it; accept the cluster of
        # yellow CTA cls in the bottom-right zone. v2 (cls 417 一次领取黄色
        # oversampled) will make this clean.
        claim_btn = self.find_cls(
            screen,
            [UC.CLAIM_ONCE_YELLOW, UC.CLAIM_ALL_YELLOW, UC.TASK_START,
             UC.SWEEP_START, UC.CLAIM_REWARD_YELLOW],
            conf=0.15, region=(0.80, 0.85, 1.00, 1.00),
        )
        if claim_btn is not None:
            self.log(f"claim all (#{self._claim_attempts}, count={cur}, cls={claim_btn.cls_name})")
            self._post_claim_wait = 3
            return action_click_box(claim_btn, f"claim all mail (YOLO {claim_btn.cls_name})")

        # Per-row single claim.
        single = self.find_cls(
            screen, [UC.CLAIM_YELLOW, UC.CLAIM_REWARD_YELLOW, UC.CLAIM_BLUE],
            conf=0.25, region=(0.82, 0.10, 1.00, 0.78),
        )
        if single is not None:
            self.log(f"claim single (#{self._claim_attempts}, count={cur}, cls={single.cls_name})")
            self._post_claim_wait = 3
            return action_click_box(single, f"claim single mail (YOLO {single.cls_name})")

        # No claim cls visible.
        if cur == 0:
            self.sub_state = "exit"
            return action_wait(300, "no claim cls + count=0")
        if self._claim_attempts > 15:
            self.log(f"giving up — no claim cls 15 attempts (count={self._last_count})")
            self.sub_state = "exit"
            return action_wait(300, "no claim cls found")
        return action_wait(400, "mail: waiting for claim cls (YOLO gap)")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        # On lobby iff we see >=2 nav icons.
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log(f"done (claimed={self._claimed_count}, final={self._last_count})")
            return action_done(
                f"mail complete (claimed={self._claimed_count}, remaining={self._last_count})"
            )
        # Prefer YOLO home/back button over blind ESC.
        home = self.find_cls(screen, UC.BTN_HOME, conf=0.30)
        if home is not None:
            return action_click_box(home, "mail exit: home button")
        back = self.find_cls(screen, UC.BTN_BACK, conf=0.30)
        if back is not None:
            return action_click_box(back, "mail exit: back button")
        return action_back("mail exit: ESC toward lobby")
