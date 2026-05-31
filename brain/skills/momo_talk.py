"""MomoTalkSkill: Auto-complete all unread MomoTalk conversations.

Design (2026-05-28 full YOLO rewrite — mirrors mail.py):
- Every click target resolved via self.find_cls/click_cls (ui_classes cls).
- NO OCR anywhere. MomoTalk has no digit counters worth reading, so this
  skill is 100% YOLO. Page state is inferred from YOLO cls signatures.
- If YOLO can't see a needed cls, log + wait (surface the gap) rather than
  fall back to OCR or blind hardcoded clicks.

Flow:
1. enter  : from lobby click NAV_MOMOTALK (cls 8). Inside iff
             detect_screen_yolo()=="MomoTalk" (MOMO_UNREAD/REPLY_OPT/SENDING).
2. scan   : find an MOMO_UNREAD badge → click that conversation. The badge
             sits at the right edge of a conversation row; clicking the row
             body (left of the badge, same y) opens it. No sort menu needed —
             we locate unread directly by its YOLO cls.
3. dialogue: pick a reply via MOMO_REPLY_OPT (cls 440). MOMO_SENDING means a
             message is mid-send — wait for the next reply prompt. When the
             bond-story buttons appear (GOTO_BOND_STORY / ENTER_BOND_STORY)
             switch to story.
4. story  : skip the bond-story cutscene — STORY_SKIP → confirm → GOT_REWARD
             tap-through. STORY_TAP_CONTINUE advances stuck narration.
5. exit   : detect_screen_yolo()=="Lobby" → done. Prefer BTN_HOME/BTN_BACK.

KNOWN cls gaps (need more training data — see report):
- Several momo cls are weak: GOTO_BOND_STORY (12f), ENTER_BOND_STORY (12f).
- "select unread sort" menu has NO cls. We sidestep it via MOMO_UNREAD.
- Bond-story skip-confirm dialog uses generic BTN_CONFIRM (82f, solid).
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box,
    action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC


class MomoTalkSkill(BaseSkill):
    _LOBBY_DOT_ENTRIES = [UC.NAV_MOMOTALK]

    def should_run(self, screen):
        return self.dot_on_entry(screen, self._LOBBY_DOT_ENTRIES)

    def __init__(self):
        super().__init__("MomoTalk")
        self.max_ticks = 120
        self._conversations_completed: int = 0
        self._scan_ticks: int = 0
        self._dialogue_ticks: int = 0
        self._story_ticks: int = 0
        self._enter_ticks: int = 0
        self._enter_click_cooldown: int = 0
        self._scan_misses: int = 0
        self._story_taps: int = 0
        # De-dup guard for the completion counter. MOMO_SENDING (28f) flickers,
        # so _in_conversation toggles True/False and the dialogue↔scan/story
        # transitions can fire repeatedly for ONE conversation. We count a
        # completion at most once per opened conversation: set this True when
        # we tally, clear it only when a NEW unread conversation is opened.
        self._counted_current: bool = False

    def reset(self) -> None:
        super().reset()
        self._conversations_completed = 0
        self._scan_ticks = 0
        self._dialogue_ticks = 0
        self._story_ticks = 0
        self._enter_ticks = 0
        self._enter_click_cooldown = 0
        self._scan_misses = 0
        self._story_taps = 0
        self._counted_current = False

    # ── completion counter (de-duplicated) ────────────────────────────
    def _count_completion(self, where: str) -> None:
        """Tally one completed conversation, but only once per opened
        conversation (guards against MOMO_SENDING flicker double-counting)."""
        if self._counted_current:
            return
        self._counted_current = True
        self._conversations_completed += 1
        self.log(f"conversation #{self._conversations_completed} completed ({where})")

    # ── page inference via YOLO cls (no OCR) ──────────────────────────
    def _on_momotalk(self, screen: ScreenState) -> bool:
        """Inside MomoTalk iff any momo cls is visible (== detect_screen_yolo
        MomoTalk signature: MOMO_UNREAD / MOMO_REPLY_OPT / MOMO_SENDING)."""
        return self.find_cls(
            screen, [UC.MOMO_UNREAD, UC.MOMO_REPLY_OPT, UC.MOMO_SENDING],
            conf=0.25,
        ) is not None

    def _in_conversation(self, screen: ScreenState) -> bool:
        """Right panel shows a live conversation: a reply option, a sending
        bubble, or a bond-story CTA."""
        return self.find_cls(
            screen,
            [UC.MOMO_REPLY_OPT, UC.MOMO_SENDING,
             UC.GOTO_BOND_STORY, UC.ENTER_BOND_STORY],
            conf=0.25,
        ) is not None

    def _in_story(self, screen: ScreenState) -> bool:
        """Inside a bond-story cutscene (skip key visible, or reward splash)."""
        return self.find_cls(
            screen,
            [UC.STORY_SKIP, UC.STORY_SKIP_DISABLED,
             UC.STORY_TAP_CONTINUE, UC.GOT_REWARD],
            conf=0.30,
        ) is not None

    # ── state machine ─────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        if self.ticks >= self.max_ticks:
            self.log(f"timeout ({self._conversations_completed} conversations completed)")
            return action_done(f"momotalk timeout ({self._conversations_completed} done)")

        # Bond level-up full-screen splash (羁绊升级) — YOLO cls, tap to dismiss.
        levelup = self.find_cls(screen, [UC.BOND_LEVELUP, UC.REGION_LEVELUP], conf=0.30)
        if levelup is not None:
            return action_click(0.5, 0.5, f"dismiss level-up ({levelup.cls_name})")

        # Reward splash (獲得獎勵) — tap to dismiss.
        reward = self.find_cls(screen, UC.GOT_REWARD, conf=0.30)
        if reward is not None:
            return action_click_box(reward, "dismiss reward popup (YOLO 获得奖励)")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "momotalk loading")

        if self.sub_state == "":
            self.sub_state = "enter"
        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "scan":
            return self._scan(screen)
        if self.sub_state == "dialogue":
            return self._dialogue(screen)
        if self.sub_state == "story":
            return self._story(screen)
        if self.sub_state == "exit":
            return self._exit(screen)
        return action_wait(300, "momotalk unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks += 1

        if self._on_momotalk(screen):
            self.log("inside MomoTalk")
            self.sub_state = "scan"
            self._scan_ticks = 0
            self._scan_misses = 0
            return action_wait(400, "entered MomoTalk")

        # Cooldown so we don't re-tap the entry icon while it transitions.
        if self._enter_click_cooldown > 0:
            self._enter_click_cooldown -= 1
            return action_wait(500, f"momotalk: enter cooldown ({self._enter_click_cooldown})")

        # YOLO MomoTalk entry icon (cls 8) on the lobby sidebar.
        momo_btn = self.find_cls(screen, UC.NAV_MOMOTALK, conf=0.30)
        if momo_btn is not None:
            self._enter_click_cooldown = 4
            return action_click_box(momo_btn, "open MomoTalk (YOLO NAV_MOMOTALK)")

        # Not lobby + not momotalk → back out toward lobby.
        if self.detect_screen_yolo(screen) not in (None, "Lobby"):
            return action_back("momotalk: backing toward lobby")

        if self._enter_ticks > 20:
            self.log("can't reach MomoTalk (no NAV_MOMOTALK cls) — YOLO gap; giving up")
            return action_done("momotalk unavailable")
        self.log("no MomoTalk entry cls — YOLO gap; waiting")
        return action_wait(400, "momotalk: waiting for NAV_MOMOTALK detection")

    def _scan(self, screen: ScreenState) -> Dict[str, Any]:
        """Find an unread conversation badge and open it."""
        self._scan_ticks += 1

        # Already inside a conversation (auto-opened) → handle it.
        if self._in_conversation(screen):
            self.sub_state = "dialogue"
            self._dialogue_ticks = 0
            return action_wait(300, "conversation already open")

        # Left MomoTalk entirely (e.g. into a story) → re-route.
        if not self._on_momotalk(screen):
            if self._in_story(screen):
                self.sub_state = "story"
                self._story_ticks = 0
                self._story_taps = 0
                return action_wait(300, "story detected during scan")
            if self._scan_ticks > 4:
                self.sub_state = "enter"
                self._enter_ticks = 0
                return action_wait(400, "lost MomoTalk, re-entering")
            return action_wait(400, "waiting for MomoTalk UI")

        # Locate an unread badge. It sits at the right edge of a conversation
        # row in the LEFT list panel; click the row body (left of badge, same
        # y) to open the conversation.
        unread = self.find_cls(
            screen, UC.MOMO_UNREAD, conf=0.30, region=(0.0, 0.15, 0.55, 0.95),
        )
        if unread is not None:
            self._scan_misses = 0
            # Opening a NEW conversation → re-arm the completion de-dup flag.
            self._counted_current = False
            self.log(f"unread conversation at y={unread.cy:.3f} → opening")
            self.sub_state = "dialogue"
            self._dialogue_ticks = 0
            # Click the conversation ROW body, not a blind x=0.30. The unread
            # badge sits at the right edge of the row; the row body is to its
            # left, so tap a bit left of the badge cx (clamped to the panel).
            row_x = max(0.06, unread.cx - 0.15)
            return action_click(row_x, unread.cy, "open unread conversation (YOLO MOMO_UNREAD row)")

        # No unread badge anywhere — done (give it a couple ticks to settle
        # in case the list is still loading after a completed conversation).
        self._scan_misses += 1
        if self._scan_misses < 3:
            return action_wait(400, f"no unread badge (settle {self._scan_misses})")
        self.log(f"no more unread conversations ({self._conversations_completed} completed)")
        self.sub_state = "exit"
        return action_wait(300, "scan complete")

    def _dialogue(self, screen: ScreenState) -> Dict[str, Any]:
        """Drive one conversation: pick replies until the bond-story CTA or
        the conversation ends and we drop back to the list."""
        self._dialogue_ticks += 1
        if self._dialogue_ticks > 50:
            self.log("dialogue timeout, back to scan")
            self.sub_state = "scan"
            self._scan_ticks = 0
            return action_back("dialogue timeout")

        # Bond-story flow has priority: 前往羁绊剧情 → 进入羁绊剧情.
        goto_bond = self.find_cls(screen, UC.GOTO_BOND_STORY, conf=0.30)
        if goto_bond is not None:
            self._dialogue_ticks = 0
            self.log("bond story available → 前往羁绊剧情")
            return action_click_box(goto_bond, "goto bond story (YOLO GOTO_BOND_STORY)")
        enter_bond = self.find_cls(screen, UC.ENTER_BOND_STORY, conf=0.30)
        if enter_bond is not None:
            self.log("entering bond story → 进入羁绊剧情")
            self.sub_state = "story"
            self._story_ticks = 0
            self._story_taps = 0
            return action_click_box(enter_bond, "enter bond story (YOLO ENTER_BOND_STORY)")

        # If a story cutscene already started, switch.
        if self._in_story(screen):
            self.sub_state = "story"
            self._story_ticks = 0
            self._story_taps = 0
            return action_wait(300, "story started during dialogue")

        # Reply option — the core interaction. Pick the highest-conf choice.
        reply = self.find_cls(screen, UC.MOMO_REPLY_OPT, conf=0.30)
        if reply is not None:
            self._dialogue_ticks = 0  # reset timeout on each interaction
            return action_click_box(reply, "pick reply (YOLO MOMO_REPLY_OPT)")

        # Message mid-send → wait for the next reply prompt.
        if self.find_cls(screen, UC.MOMO_SENDING, conf=0.25) is not None:
            return action_wait(500, "message sending, waiting for next reply")

        # Back on the conversation list with no reply/story CTA → this
        # conversation is done; rescan for the next unread.
        if self._on_momotalk(screen) and not self._in_conversation(screen):
            self._count_completion("back to list")
            self.sub_state = "scan"
            self._scan_ticks = 0
            self._scan_misses = 0
            return action_wait(300, "conversation done, scanning for more")

        # On MomoTalk but no actionable cls yet — wait (YOLO gap / loading).
        if self._on_momotalk(screen):
            return action_wait(400, "momotalk: waiting for reply cls (YOLO gap)")
        return action_wait(400, "dialogue: waiting for UI")

    def _story(self, screen: ScreenState) -> Dict[str, Any]:
        """Skip the bond-story cutscene: STORY_SKIP → confirm → reward."""
        self._story_ticks += 1
        if self._story_ticks > 60:
            self.log("story timeout, back to scan")
            self.sub_state = "scan"
            self._scan_ticks = 0
            return action_back("story timeout")

        # Reward splash → conversation+story complete (also handled in tick,
        # but counts the completion here).
        reward = self.find_cls(screen, UC.GOT_REWARD, conf=0.30)
        if reward is not None:
            self._count_completion("story reward")
            self.sub_state = "scan"
            self._scan_ticks = 0
            self._scan_misses = 0
            return action_click_box(reward, "dismiss story reward (YOLO 获得奖励)")

        # Skip-confirm dialog (是否略過此劇情) uses the generic blue 确认.
        confirm = self.find_cls(
            screen, UC.BTN_CONFIRM, conf=0.40, region=(0.30, 0.55, 0.85, 0.88),
        )
        if confirm is not None:
            self.log("confirming story skip (YOLO 确认)")
            return action_click_box(confirm, "confirm story skip (YOLO BTN_CONFIRM)")

        # Skip key (跳過故事键). If disabled, the cutscene must advance first.
        skip = self.find_cls(screen, UC.STORY_SKIP, conf=0.30)
        if skip is not None:
            self._story_taps = 0
            return action_click_box(skip, "skip story (YOLO STORY_SKIP)")

        # Skip disabled → tap-to-continue to roll the scene until skip enables.
        tap = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=0.30)
        if tap is not None:
            return action_click_box(tap, "advance story (YOLO STORY_TAP_CONTINUE)")
        if self.find_cls(screen, UC.STORY_SKIP_DISABLED, conf=0.30) is not None:
            self._story_taps += 1
            return action_click(0.5, 0.5, "advance story (skip disabled, tap center)")

        # Back on the conversation list → story ended.
        if self._on_momotalk(screen) and not self._in_conversation(screen):
            self._count_completion("story back to list")
            self.sub_state = "scan"
            self._scan_ticks = 0
            self._scan_misses = 0
            return action_wait(300, "story done, scanning for more")
        if self._in_conversation(screen):
            self.sub_state = "dialogue"
            self._dialogue_ticks = 0
            return action_wait(300, "back in conversation after story")

        # No story cls visible — wait (YOLO gap / transition).
        return action_wait(400, "story: waiting for skip cls (YOLO gap)")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log(f"done ({self._conversations_completed} conversations)")
            return action_done(f"momotalk complete ({self._conversations_completed} done)")
        # Prefer YOLO home/back button over blind ESC.
        home = self.find_cls(screen, UC.BTN_HOME, conf=0.30)
        if home is not None:
            return action_click_box(home, "momotalk exit: home button")
        back = self.find_cls(screen, UC.BTN_BACK, conf=0.30)
        if back is not None:
            return action_click_box(back, "momotalk exit: back button")
        return action_back("momotalk exit: ESC toward lobby")
