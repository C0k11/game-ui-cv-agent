"""MomoTalkSkill — mine all unread MomoTalk conversations (pure-YOLO rewrite).

Verified flow (interactive probe 2026-06-01, data/_mining_probe_log.md). Mining
MomoTalk = reading unread conversations to unlock 羁绊剧情 (each story ≈ 80
pyroxene — a top free-pyroxene source).

Probe refinements over the old skill:
- After entering MomoTalk, click the 对话区域 tab (MOMO_CHAT_TAB) to reach the
  未讀訊息 list (the default name-sorted view doesn't show unread directly).
- Open a student by tapping the row's LEFT (avatar, x≈0.22), NOT the unread
  badge (x≈0.505) — tapping the badge is unreliable (probe: 莉 didn't open).
- 学生发送信息中 (MOMO_SENDING) is a TRANSIENT "student is typing" cls — it only
  flickers. WGC polls ~55fps so we catch it; we only declare a student done
  after STABLE_N consecutive frames with NO sending / reply / goto-bond.
- 前往羁绊剧情 does NOT retrigger consecutively — post-bond chatter is just
  more reply options to clear.

State machine
-------------
enter     lobby → NAV_MOMOTALK → MomoTalk → click MOMO_CHAT_TAB → unread list.
scan      tap the top 学生momotalk信息未读 row (via avatar-x). None → done.
dialogue  metronome: 发送信息中→wait; 回复选项→tap; 前往羁绊剧情→story. STABLE_N
          empty frames → student done → scan next.
story     进入羁绊剧情 → 剧情menu→跳过故事键→确认键→获得奖励→点击继续字样 → list.
exit      BTN_HOME / BTN_BACK → lobby → done.

Detectors: base "ui" only (SKILL_YOLO_MAP MomoTalk = base).
"""
from __future__ import annotations

from typing import Any, Dict, List

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_wait, action_back, action_done,
    action_swipe,
)
from brain.skills import ui_classes as UC

_CLS_CONF = 0.30
_UNREAD_LIST_REGION = (0.0, 0.15, 0.55, 0.95)   # left conversation-list panel
_AVATAR_DX = 0.28          # avatar sits ~this far LEFT of the unread badge
# Switch students ONLY when the screen has NO 学生信息回复选项 (reply) AND NO 学生发
# 送信息中 (sending) AND NO bond CTA — driven purely by cls detection, NOT a timer.
# This tiny window only bridges a 1-2 frame render gap (sending vanishes before
# the reply renders) and the weak cls flickering (sending 28f / reply 32f → an
# occasional missed frame). It is NOT a per-student cooldown — any reply/sending
# frame instantly resets it, so a still-talking student is never abandoned.
_STABLE_EMPTY = 14         # weak cls (sending 28f / reply 32f) misses a few frames +
                           # students pause between consecutive messages — bridge it.
                           # Still cls-driven: ANY reply/sending frame resets it.
                           # 8 was too impatient (live 2026-06-10: students were
                           # abandoned mid-chat → 横跳/re-opens).
_ROW_OPEN_CAP = 2          # re-open a still-badged row at most this many times
                           # (badge = ground truth; the cap only guards the 一花
                           # class of badges that never clear).
_MAX_SENDING = 30          # sending stuck this long = mis-detect (a real msg types <13s)
# Lowered detection floors for the two weak chat cls so a faint reply/sending
# frame still registers (was missing → false "done"). v6 should add samples.
_SENDING_CONF = 0.15
_REPLY_CONF = 0.18
_MAX_SCROLLS = 6           # list swipes before giving up (mine visible, then scroll down)

_ENTER_MAX = 22
_TAB_MAX = 12
_DIALOGUE_MAX = 60
_STORY_MAX = 70
_EXIT_MAX = 14


class MomoTalkSkill(BaseSkill):
    def should_run(self, screen: ScreenState) -> bool:
        # ★ The unread-MomoTalk dot appears on the NAVBAR 社交入口 (NAV_SOCIAL),
        # NOT on the left-side MomoTalk widget — NAV_MOMOTALK has NO dot, so
        # gating on it alone ALWAYS false-skipped (live 2026-06-15: 社交入口 红点
        # clearly present + LobbyBadge saw social=red, but dot_on_entry anchored
        # the wrong icon → skipped every run). The 社交入口 dot is the real signal
        # (user: 红点在社交旁边 = 进). Check both icons so the dot is found
        # wherever it renders; momo_talk still clicks NAV_MOMOTALK to enter.
        return self.dot_on_entry(screen, [UC.NAV_SOCIAL, UC.NAV_MOMOTALK])

    def __init__(self):
        super().__init__("MomoTalk")
        # 400: a first-run BACKLOG (live 2026-06-10: 22 unread) measures ~22
        # ticks/student incl. bond stories — 200 timed out at 9/22 mined.
        # Steady-state dailies (1-3 unread) finish in <60.
        self.max_ticks = 400
        self._init_state()

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._enter_ticks: int = 0
        self._tab_opened: bool = False
        self._students_done: int = 0
        self._empty_streak: int = 0
        self._sending_streak: int = 0        # consecutive sending frames (mis-detect cap)
        self._scan_misses: int = 0
        self._scrolls: int = 0               # list swipes done
        self._row_opens: List[List[float]] = []  # [row-cy, opens] — re-open cap per view
        self._reply_positions: set = set()     # tapped reply spots this student (skip mis-detect repeats)
        self._reply_gone: int = 0              # consecutive frames with NO reply option
        self._story_taps: int = 0
        self._story_cut: int = 0

    def reset(self) -> None:
        super().reset()
        self._init_state()

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0

    def _open_count(self, cy: float) -> int:
        """How many times we've opened a student at ~this row-cy this view."""
        for entry in self._row_opens:
            if abs(entry[0] - cy) < 0.05:
                return int(entry[1])
        return 0

    def _bump_open(self, cy: float) -> None:
        for entry in self._row_opens:
            if abs(entry[0] - cy) < 0.05:
                entry[0] = cy
                entry[1] += 1
                return
        self._row_opens.append([cy, 1])

    # ── page predicates ──────────────────────────────────────────────────
    def _on_momotalk(self, screen: ScreenState) -> bool:
        return self.find_cls(
            screen, [UC.MOMO_CHAT_TAB, UC.MOMO_CHAT_TAB_SEL, UC.MOMO_UNREAD,
                     UC.MOMO_REPLY_OPT, UC.MOMO_SENDING, UC.FAVORITE_ICON],
            conf=0.25,
        ) is not None

    def _on_unread_list(self, screen: ScreenState) -> bool:
        return (self.find_cls(screen, UC.MOMO_CHAT_TAB_SEL, conf=_CLS_CONF) is not None
                or self.find_cls(screen, UC.MOMO_UNREAD, conf=_CLS_CONF, region=_UNREAD_LIST_REGION) is not None)

    def _in_conversation(self, screen: ScreenState) -> bool:
        return self.find_cls(
            screen, [UC.MOMO_REPLY_OPT, UC.MOMO_SENDING,
                     UC.GOTO_BOND_STORY, UC.ENTER_BOND_STORY], conf=0.25,
        ) is not None

    def _in_story(self, screen: ScreenState) -> bool:
        return self.find_cls(
            screen, [UC.STORY_MENU, UC.STORY_SKIP, UC.STORY_SKIP_DISABLED,
                     UC.STORY_TAP_CONTINUE, UC.ENTER_BOND_STORY], conf=0.30,
        ) is not None

    # ── tick ────────────────────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log(f"timeout ({self._students_done} students)")
            return action_done(f"momotalk timeout ({self._students_done})")

        # Global: bond level-up splash + reward popup.
        levelup = self.find_cls(screen, [UC.BOND_LEVELUP, UC.REGION_LEVELUP], conf=_CLS_CONF)
        if levelup is not None:
            return action_click(0.5, 0.5, f"dismiss level-up ({levelup.cls_name})")

        if screen.is_loading():
            return action_wait(700, "momotalk loading")

        if self.sub_state == "":
            self._goto("enter")

        handler = {
            "enter": self._enter,
            "open_tab": self._open_tab,
            "scan": self._scan,
            "dialogue": self._dialogue,
            "story": self._story,
            "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "momotalk unknown state")
        return handler(screen)

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks += 1
        if self._on_momotalk(screen):
            self.log("inside MomoTalk → open 对话区域 tab")
            self._goto("open_tab")
            return action_wait(350, "entered MomoTalk")

        if screen.is_lobby():
            act = self.click_cls(screen, UC.NAV_MOMOTALK, "open MomoTalk", conf=_CLS_CONF)
            if act is not None:
                return act
            return action_wait(400, "waiting for MomoTalk entry cls")

        if self._enter_ticks > _ENTER_MAX:
            return action_done("momotalk unreachable")
        if self.detect_screen_yolo(screen) not in (None, "Lobby"):
            return action_back("momotalk: recover toward lobby")
        return action_wait(400, "entering MomoTalk")

    def _open_tab(self, screen: ScreenState) -> Dict[str, Any]:
        """Switch to the 对话区域 (未讀) tab so the unread list shows."""
        if self._on_unread_list(screen) or self._in_conversation(screen):
            self._goto("scan")
            return action_wait(250, "unread list ready → scan")

        if not self._tab_opened:
            tab = self.find_cls(screen, UC.MOMO_CHAT_TAB, conf=_CLS_CONF)
            if tab is not None:
                self._tab_opened = True
                self.log("clicking 对话区域 tab (MOMO_CHAT_TAB)")
                return action_click_box(tab, "open 对话区域 tab")

        if self._phase_ticks > _TAB_MAX:
            # Tab cls missed but maybe already on the list — proceed.
            self._goto("scan")
            return action_wait(300, "tab timeout → scan")
        return action_wait(350, "waiting for 对话区域 tab")

    def _scan(self, screen: ScreenState) -> Dict[str, Any]:
        # Story splash surfaced mid-scan?
        if self._in_story(screen):
            self._goto("story")
            return action_wait(250, "story detected → story")

        # ★ Do NOT treat a lingering right-pane conversation as work. After a
        # student finishes, the right pane keeps showing its last messages
        # (reply/bond cls stay detected) — the old `_in_conversation → dialogue`
        # branch re-entered dialogue forever (total ran 14→47, instant-done loop,
        # only 3 students actually mined before max_ticks). The open student's own
        # multi-turn dialogue is fully handled INSIDE _dialogue; scan only ever
        # opens a FRESH unread badge (below). No residual-conversation shortcut.

        if not self._on_momotalk(screen):
            if self._phase_ticks > 6:
                self._goto("enter")
                self._enter_ticks = 0
                return action_wait(400, "lost MomoTalk → re-enter")
            return action_wait(400, "waiting for MomoTalk UI")

        # Badge = GROUND TRUTH (user 2026-06-10: 从上到下挨个打完, 全部清空才
        # scroll). Open the TOP-MOST visible badge, top-to-bottom, re-opening a
        # still-badged row up to _ROW_OPEN_CAP times (an unfinished student's
        # badge stays until truly done; the cap only guards 一花-class badges
        # that never clear). NEVER scroll while a workable badge is visible.
        all_unread = [u for u in self.find_all_cls(screen, UC.MOMO_UNREAD, conf=_CLS_CONF)
                      if 0.0 <= u.cx <= 0.62 and 0.15 <= u.cy <= 0.95]
        fresh = [u for u in all_unread if self._open_count(u.cy) < _ROW_OPEN_CAP]
        if fresh:
            unread = min(fresh, key=lambda b: b.cy)  # top-most, strict top-down
            self._bump_open(unread.cy)
            self._scan_misses = 0
            self._empty_streak = 0
            self._sending_streak = 0
            self._reply_positions = set()
            self._reply_gone = 0
            row_x = min(0.30, max(0.12, unread.cx - _AVATAR_DX))
            self.log(f"open unread student y={unread.cy:.3f} (avatar x={row_x:.2f}, "
                     f"open #{self._open_count(unread.cy)})")
            self._goto("dialogue")
            return action_click(row_x, unread.cy, "open unread student (avatar, not badge)")

        # No workable badge (none visible, or the visible ones are 一花-class
        # stale after CAP re-opens). Settle (badges may still render), then
        # SCROLL DOWN for more. Rows shift after a swipe → reset open counts.
        self._scan_misses += 1
        if self._scan_misses < 3:
            return action_wait(400, f"no workable unread (settle {self._scan_misses})")
        if self._scrolls < _MAX_SCROLLS:
            self._scrolls += 1
            self._scan_misses = 0
            self._row_opens = []   # rows shifted by the swipe
            self.log(f"visible list cleared → scroll down ({self._scrolls}/{_MAX_SCROLLS})")
            return action_swipe(0.25, 0.72, 0.25, 0.42, 700,
                                f"scroll unread list down ({self._scrolls})")
        self.log(f"no more unread after {self._scrolls} scrolls ({self._students_done} mined)")
        self._goto("exit")
        return action_wait(300, "scan complete → exit")

    def _dialogue(self, screen: ScreenState) -> Dict[str, Any]:
        if self._phase_ticks > _DIALOGUE_MAX:
            self.log("dialogue timeout → scan")
            self._goto("scan")
            self._scan_misses = 0
            return action_back("dialogue timeout")

        # Bond-story CTA (priority — the 80-pyroxene payoff).
        goto_bond = self.find_cls(screen, UC.GOTO_BOND_STORY, conf=_CLS_CONF)
        if goto_bond is not None:
            self._empty_streak = 0
            self.log("前往羁绊剧情")
            return action_click_box(goto_bond, "goto bond story")
        enter_bond = self.find_cls(screen, UC.ENTER_BOND_STORY, conf=_CLS_CONF)
        if enter_bond is not None:
            self._empty_streak = 0
            self.log("进入羁绊剧情 → story")
            self._story_taps = 0
            self._story_cut = 0
            self._goto("story")
            return action_click_box(enter_bond, "enter bond story")

        # ★ Metronome: student typing → wait. Cap consecutive sending: a real
        # "typing" lasts a few frames; if it sticks (mis-detect) treat it as empty
        # so the student can finish instead of waiting forever.
        if self.find_cls(screen, UC.MOMO_SENDING, conf=_SENDING_CONF) is not None:
            self._sending_streak += 1
            if self._sending_streak <= _MAX_SENDING:
                self._empty_streak = 0
                # A new message wave makes previously-tapped reply spots STALE —
                # the next option legitimately renders at the SAME fixed spot.
                self._reply_positions.clear()
                return action_wait(450, f"学生发送信息中 — waiting ({self._sending_streak}/{_MAX_SENDING})")
            # sending stuck → mis-detect; fall through to empty handling.
        else:
            self._sending_streak = 0

        # Reply option → tap, with DYNAMIC position-dedup. ⚠️ The 回覆 box
        # renders at a FIXED spot, so consecutive turns reuse the same position
        # — a permanent dedup ate the 2nd+ option and ABANDONED the student
        # mid-chat (live 2026-06-10 横跳 root cause: 40 opens for ~24 students).
        # Dedup now invalidates when the option DISAPPEARS ≥2 frames (consumed)
        # or a sending wave arrives. A mis-detected static chat bubble (一花
        # class) never disappears → stays deduped → empty-streak still ends the
        # student.
        reply = self.find_cls(screen, UC.MOMO_REPLY_OPT, conf=_REPLY_CONF)
        if reply is not None:
            self._reply_gone = 0
            rpos = (round(reply.cx, 2), round(reply.cy, 2))
            if rpos not in self._reply_positions:
                self._reply_positions.add(rpos)
                self._empty_streak = 0
                self._sending_streak = 0
                return action_click_box(reply, "pick reply option")
            # same spot already tapped → mis-detect; fall through to empty.
        elif self._reply_positions:
            self._reply_gone += 1
            if self._reply_gone >= 2:
                self._reply_positions.clear()
                self._reply_gone = 0

        # Nothing fresh to do → empty. The student is FULLY done only after
        # STABLE_EMPTY *consecutive* empties (post-bond chatter + typing pauses all
        # cleared). No cumulative shortcut — we never switch students early.
        self._empty_streak += 1
        if self._empty_streak >= _STABLE_EMPTY:
            self._students_done += 1
            self.log(f"student fully done (#{self._students_done}) → scan next")
            self._goto("scan")
            self._scan_misses = 0
            return action_wait(300, "student done → scan")
        return action_wait(350, f"dialogue settle (empty {self._empty_streak}/{_STABLE_EMPTY})")

    def _story(self, screen: ScreenState) -> Dict[str, Any]:
        """Skip the bond-story cutscene → claim the ~80-pyroxene reward."""
        if self._phase_ticks > _STORY_MAX:
            self.log("story timeout → scan")
            self._goto("scan")
            self._scan_misses = 0
            return action_back("story timeout")

        # Reward splash → claim, then RETURN TO DIALOGUE (not scan): the bond
        # story is mined but the student still has post-bond chatter to clear.
        # Jumping to scan here (+ row-dedup) skipped it and switched students —
        # the "剧情打完没打后续就换人" bug. dialogue clears the post-bond replies,
        # THEN scans the next student.
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
        got = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if got is not None or cont is not None:
            self._goto("dialogue")
            self._empty_streak = 0
            self._sending_streak = 0
            self._reply_positions = set()   # post-bond replies are fresh options
            if cont is not None:
                return action_click_box(cont, "dismiss bond reward → post-bond chatter")
            return action_click_box(got, "dismiss bond reward (header) → post-bond")

        # Skip-confirm dialog (是否略過) → 确认键.
        if self._story_cut > 0:
            confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=(0.30, 0.55, 0.85, 0.85))
            if confirm is not None:
                self._story_cut = 0
                return action_click_box(confirm, "confirm story skip")

        # MENU → 跳过故事键 (skip ASAP — story auto-plays).
        menu = self.find_cls(screen, UC.STORY_MENU, conf=_CLS_CONF)
        skip = self.find_cls(screen, UC.STORY_SKIP, conf=_CLS_CONF)
        if skip is not None:
            self._story_cut += 1
            return action_click_box(skip, "跳过故事键")
        if menu is not None:
            self._story_cut += 1
            return action_click_box(menu, "open 剧情menu")

        # Skip greyed → advance narration via tap-continue.
        if self.find_cls(screen, UC.STORY_SKIP_DISABLED, conf=_CLS_CONF) is not None:
            self._story_taps += 1
            return action_click(0.5, 0.5, "advance story (skip disabled)")

        # Bond story done — the game returns to the student's chat. The left list
        # tab is still visible so _on_unread_list trips, but the RIGHT pane is the
        # student's POST-BOND chatter. Go to DIALOGUE to clear it, NOT scan. The
        # reward popup is usually eaten by the global interceptor before our
        # reward branch above fires, so THIS is the real "story done" exit — and
        # it must return to the SAME student, else we jump to the next (横跳 bug).
        if self._on_unread_list(screen):
            self._goto("dialogue")
            self._empty_streak = 0
            self._sending_streak = 0
            self._reply_positions = set()   # post-bond replies are fresh options
            return action_wait(300, "story done → post-bond chatter (same student)")
        return action_wait(400, "story: waiting for menu/skip cls")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log(f"done ({self._students_done} students)")
            return action_done(f"momotalk complete ({self._students_done})")
        if self._phase_ticks > _EXIT_MAX:
            return action_done("momotalk exit timeout")
        # Standard exit kit (2026-06-10): cancel-first (quit-prompt / cost
        # dialogs), then home/back cls, then PACED blind ESC.
        cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=0.20)
        if cancel is not None:
            return action_click_box(cancel, "momotalk exit: cancel pending dialog")
        home = self.find_cls(screen, UC.BTN_HOME, conf=_CLS_CONF)
        if home is not None:
            return action_click_box(home, "momotalk exit: home")
        back = self.find_cls(screen, UC.BTN_BACK, conf=_CLS_CONF)
        if back is not None:
            return action_click_box(back, "momotalk exit: back")
        if self._phase_ticks % 3 != 0:
            return action_wait(600, "exit: settle before next ESC")
        return action_back("momotalk exit: ESC toward lobby")
