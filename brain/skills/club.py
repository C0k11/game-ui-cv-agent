"""ClubSkill — daily 社團 sign-in for free 10 AP (pure-YOLO rewrite).

Verified flow (interactive probe 2026-06-01, data/_social_probe_log.md). The
old skill was WRONG: it looked for a 領取 claim cls and relied on the OCR popup
handler (dead in pure-YOLO mode). The probe found the real flow:

  社交入口 → 社團 card → auto-pops 「社團簽到獎勵」 → tap 确认键 (centered, no
  cancel) → sign-in done. The 10 AP goes to the MAILBOX (mail skill claims it),
  not immediately to the balance.

Two YOLO gaps the skill works around (documented, fix in v6 — task #22):
- The 3 social cards (社團/好友/幫手) have no reliable cls (社團/CLUB cls 51
  recall-missed at conf<0.20). We click the 社團 card via CLUB cls when seen,
  else via a normalized offset DOWN-LEFT of its red dot (probe: card body
  ~(0.235,0.52), dot ~(0.364,0.415) ⇒ Δ(-0.13,+0.105)). Tapping the dot itself
  is too high → lands above the card → the overlay collapses back to lobby.

State machine
-------------
enter    checkin dialog up → checkin. social overlay (社團 dot / CLUB cls) →
         tap the card. lobby → tap NAV_SOCIAL. (cooldown to avoid spam.)
checkin  「社團簽到獎勵」 = 确认键 present with NO 取消键 → tap it → done.
exit     BTN_HOME / BTN_BACK → lobby → done.

Detectors: base "ui" only.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC

_CLS_CONF = 0.30
# 社團 card red dot lives here on the social overlay (probe: ~0.364,0.415).
_CARD_DOT_REGION = (0.28, 0.34, 0.46, 0.50)
# Normalized offset from the 社團 red dot to the card BODY center (probe).
_DOT_TO_CARD = (-0.129, 0.105)
# Centered confirm button band for the 社團簽到獎勵 dialog (probe: ~0.499,0.676).
_CHECKIN_REGION = (0.34, 0.58, 0.66, 0.78)

_ENTER_MAX = 26
_CHECKIN_MAX = 12
_EXIT_MAX = 14


class ClubSkill(BaseSkill):
    def should_run(self, screen: ScreenState) -> bool:
        # Badge-gating is a LOBBY decision only (schedule踩坑 2026-06-09: 复跑
        # 停在中间屏时入口锚点误判 "no dot" → 整个skill被skip)。
        if not screen.is_lobby():
            return True
        return self.dot_on_entry(screen, [UC.NAV_SOCIAL])

    def __init__(self):
        super().__init__("Club")
        self.max_ticks = 50
        self._init_state()

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._nav_cooldown: int = 0      # ticks to wait after a navigation tap
        self._card_taps: int = 0
        self._checked_in: bool = False
        self._checkin_clicked: bool = False  # 点過確認键; latch _checked_in 仅凭到达证据
        # badge-verified one-shot re-entry (cafe 同款, deep-dive r2 M3)
        self._verify_reentered: bool = False
        self._exit_lobby_ticks: int = 0

    def reset(self) -> None:
        super().reset()
        self._init_state()

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0

    # ── helpers ──────────────────────────────────────────────────────────
    def _checkin_dialog(self, screen: ScreenState):
        """社團簽到獎勵 = a centered 确认键 with NO 取消键 (single-button)."""
        confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_CHECKIN_REGION)
        if confirm is None:
            return None
        if self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF) is not None:
            return None  # 2-button dialog = not the checkin (don't mishandle)
        return confirm

    def _card_dot(self, screen: ScreenState):
        return self.find_cls(screen, UC.DOT_RED, conf=0.35, region=_CARD_DOT_REGION)

    def _on_social_overlay(self, screen: ScreenState) -> bool:
        """Social overlay marker: the 社團 card dot (or CLUB cls) is present and
        we're not on a deeper page."""
        return (self._card_dot(screen) is not None
                or self.find_cls(screen, UC.CLUB, conf=_CLS_CONF) is not None)

    # ── tick ────────────────────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log(f"timeout (checked_in={self._checked_in})")
            return action_done("club timeout")

        if screen.is_loading():
            return action_wait(700, "club loading")

        if self.sub_state == "":
            self._goto("enter")

        handler = {
            "enter": self._enter,
            "checkin": self._checkin,
            "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "club unknown state")
        return handler(screen)

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        # Sign-in dialog already up (card auto-pops it) → checkin.
        if self._checkin_dialog(screen) is not None:
            self._goto("checkin")
            return action_wait(200, "checkin dialog up → checkin")

        if self._nav_cooldown > 0:
            self._nav_cooldown -= 1
            return action_wait(450, f"club: nav cooldown ({self._nav_cooldown})")

        # Social overlay open → tap the 社團 card.
        if self._on_social_overlay(screen):
            club = self.find_cls(screen, UC.CLUB, conf=_CLS_CONF)
            if club is not None:
                self._card_taps += 1
                self._nav_cooldown = 3
                self.log("tapping 社團 card (YOLO 社团)")
                return action_click_box(club, "open 社團 (cls)")
            dot = self._card_dot(screen)
            if dot is not None and self._card_taps < 5:
                # Card body is DOWN-LEFT of the dot (dot alone is too high →
                # overlay collapses). Documented under-trained-cls fallback.
                cx = max(0.05, min(0.95, dot.cx + _DOT_TO_CARD[0]))
                cy = max(0.05, min(0.95, dot.cy + _DOT_TO_CARD[1]))
                self._card_taps += 1
                self._nav_cooldown = 3
                self.log(f"tapping 社團 card body via dot offset ({cx:.2f},{cy:.2f})")
                return action_click(cx, cy, "open 社團 (dot-relative card body)")
            if self._card_taps >= 5:
                self.log("社團 card taps exhausted, exiting")
                self._goto("exit")
                return action_wait(300, "card taps exhausted → exit")
            return action_wait(350, "waiting for 社團 card cls/dot")

        # On lobby → open the social overlay.
        if screen.is_lobby():
            social = self.find_cls(screen, UC.NAV_SOCIAL, conf=_CLS_CONF)
            if social is not None:
                self._nav_cooldown = 3
                self.log("opening social overlay (YOLO 社交入口)")
                return action_click_box(social, "open social")
            # A stray popup (e.g. buy_pyroxene's free-pack leftovers) hides the
            # bottom nav while is_lobby() stays true → close it instead of
            # starving (live 2026-06-12: 50 ticks waiting under a popup).
            x = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
            if x is not None:
                return action_click_box(x, "close stray popup blocking lobby nav")
            self.log("on lobby but no 社交入口 — YOLO gap; waiting")
            return action_wait(400, "waiting for 社交入口 cls")

        if self._phase_ticks > _ENTER_MAX:
            self.log("enter budget exhausted, exiting")
            self._goto("exit")
            return action_wait(300, "enter timeout → exit")
        return self.nav_home(screen, "club recover")

    def _checkin(self, screen: ScreenState) -> Dict[str, Any]:
        confirm = self._checkin_dialog(screen)
        if confirm is not None:
            # 不提前 latch _checked_in(2026-07-21 mutate-before-ack 根治: 点击被吞
            # 时旧码已置 _checked_in=True → _exit 的 badge-verify 兜底(L202 not
            # _checked_in)被骗过, 签到没成仍报 done)。reason 加"確認键"稳定门豁免。
            self._checkin_clicked = True
            self.log("社團簽到 → 確認键 (10AP to mailbox)")
            return action_click_box(confirm, "confirm club sign-in (確認键)")

        # Dialog gone. 若点過確認 → 签到落地 latch(到达证据, post-ack)。
        if self._checkin_clicked:
            self._checked_in = True
            self.log("确认框消失 → 签到落地 → exit")
            self._goto("exit")
            return action_wait(300, "checkin committed → exit")
        if self._phase_ticks > _CHECKIN_MAX:
            self.log("checkin dialog never appeared → exit")
            self._goto("exit")
            return action_wait(300, "no checkin dialog → exit")
        return action_wait(300, "waiting for 社團簽到 dialog")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if self.detect_screen_yolo(screen) == "Lobby":
            # BADGE-VERIFIED completeness (cafe 同款): 社交入口还挂红点 = 签到
            # 没成 → 重进一次。3-tick 驻留防 badge 渐入时序闪失。
            # ⚠2026-06-16 实测: 签到成功后 navbar 社交的蓝「+」徽标被 v12 误判成
            # 红点 → 假"dot persists"触发无谓重进(社团→空转5次才自退, 浪费~40s)。
            # 既然本轮已签到成功(_checked_in), navbar dot 必是假阳/残留 → 不重进,
            # 直接收尾。真·漏签到(_checked_in=False)的安全重进保留。
            if not self._verify_reentered and not self._checked_in:
                entry = self.find_cls(screen, UC.NAV_SOCIAL, conf=0.40)
                if entry is not None and self.dot_in_region(
                        screen,
                        (entry.x1 - 0.005, entry.y1 - 0.05, entry.x2 + 0.02, entry.y2)):
                    self.log("⚠ 社交入口 still has dot → sign-in missed, re-entering once")
                    self._init_state()
                    self._verify_reentered = True
                    self._goto("enter")
                    return action_wait(300, "social dot persists → re-enter")
                self._exit_lobby_ticks += 1
                if self._exit_lobby_ticks < 3:
                    return action_wait(350, f"lobby badge dwell ({self._exit_lobby_ticks}/3)")
            self.log(f"done (checked_in={self._checked_in})")
            return action_done(f"club complete (checked_in={self._checked_in})")
        if self._phase_ticks > _EXIT_MAX:
            return action_done("club exit timeout")
        home = self.find_cls(screen, UC.BTN_HOME, conf=_CLS_CONF)
        if home is not None:
            return action_click_box(home, "club exit: home button")
        back = self.find_cls(screen, UC.BTN_BACK, conf=_CLS_CONF)
        if back is not None:
            return action_click_box(back, "club exit: back button")
        return self.nav_home(screen, "club exit")
