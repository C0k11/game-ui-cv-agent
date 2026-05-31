"""DailyTasksSkill: Claim daily quest rewards. YOLO-only clicks.

Design (2026-05-28 full YOLO rewrite, mirrors mail.py):
- Every click target resolved via self.find_cls/click_cls (ui_classes cls).
- NO OCR button fallback, NO detect_current_screen (OCR), NO hardcoded blind
  coordinates. If YOLO can't see a needed cls, log + wait (surface the gap)
  rather than fake-finish.
- The ONLY OCR is the battle-active guard (AUTO / 戰鬥時間) — there's no
  ui_v1 cls for that full-screen text and it's cheap + rare (same as mail).

States: enter → claim → exit

Flow:
1. ENTER:  From lobby, click 任務大廳入口 (UC.NAV_TASKS) to open the task hall.
           Cooldown after click so we don't re-tap the entry icon.
2. CLAIM:  Click 全部領取_黃 (UC.CLAIM_ALL_YELLOW) to sweep all completed
           tasks; fall through to per-row 領取_黃 / 領取獎勵_黃 for tier-bonus
           rows the sweep misses. Dismiss 獲得獎勵 result popups.
3. EXIT:   Back to lobby (detect_screen_yolo() == "Lobby").

KNOWN cls GAPS (need training data — see report):
- daily_tasks page has NO dedicated page-signature cls. We infer "on tasks
  page" heuristically: detect_screen_yolo() != "Lobby" AND a CLAIM_* cls is
  visible. A dedicated 任務大廳 header/title cls would make this clean.
- 活躍度寶箱 (activity-milestone chests) have NO YOLO cls — the entire chest
  phase was REMOVED. Re-add once a chest cls is trained.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box,
    action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC


class DailyTasksSkill(BaseSkill):
    _LOBBY_DOT_ENTRIES = [UC.NAV_TASKS]

    def should_run(self, screen):
        return self.dot_on_entry(screen, self._LOBBY_DOT_ENTRIES)

    def __init__(self):
        super().__init__("DailyTasks")
        self.max_ticks = 50
        self._claim_attempts: int = 0
        self._claimed_count: int = 0
        self._post_claim_wait: int = 0
        self._enter_click_cooldown: int = 0
        # Entry-clicked gate: only treat a claim-cls screen as "our page"
        # AFTER we've clicked our own entry icon. Stops串页 — a residual claim
        # page from the previous skill, or a transient YOLO claim-cls misfire,
        # no longer false-trips _on_tasks_page while we're still on lobby.
        self._entered: bool = False

    def reset(self) -> None:
        super().reset()
        self._claim_attempts = 0
        self._claimed_count = 0
        self._post_claim_wait = 0
        self._enter_click_cooldown = 0
        self._entered = False

    # ── page inference via YOLO cls (no OCR) ──────────────────────────
    def _on_tasks_page(self, screen: ScreenState) -> bool:
        """Inside daily-tasks iff we've clicked our own entry (self._entered)
        AND we're NOT on lobby AND a claim cls (active or grey) is visible.
        No dedicated page cls exists (gap); the entry gate is what keeps this
        heuristic from串页 onto a leftover/other-skill claim screen. Pure
        predicate — the _entered re-arm on lobby happens in _enter."""
        if not self._entered:
            return False
        if self.detect_screen_yolo(screen) == "Lobby":
            return False
        return self.find_cls(
            screen, UC.CLAIM_ACTIVE + UC.CLAIM_DONE, conf=0.20,
        ) is not None

    # ── state machine ─────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        # Battle-active guard (orphan battle) — kept as OCR since AUTO/timer
        # text isn't a ui_v1 cls; cheap + rare (same as mail).
        if screen.find_any_text(["AUTO", "Auto", "戰鬥時間", "战斗时间"], min_conf=0.6):
            return action_wait(800, "daily_tasks: battle in progress, waiting")

        self.ticks += 1
        if self.ticks >= self.max_ticks:
            self.log(f"timeout (claimed={self._claimed_count})")
            return action_done("daily_tasks timeout")

        # Reward result popup (獲得獎勵) — dismiss via YOLO X, else tap center.
        reward = self.find_cls(screen, UC.GOT_REWARD, conf=0.30)
        if reward is not None:
            close_x = self.find_cls(screen, UC.BTN_CLOSE_X, conf=0.40)
            if close_x is not None:
                self._post_claim_wait = 2
                return action_click_box(close_x, "dismiss reward popup (YOLO X)")
            self._post_claim_wait = 2
            return action_click(0.5, 0.9, "dismiss reward popup (獲得獎勵, tap center)")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "daily_tasks loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "claim":
            return self._claim(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "daily_tasks unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        # Re-arm the entry gate only on a CONFIRMED return to lobby and only
        # when not mid-entry (cooldown==0), so a still-visible lobby nav bar
        # during the page transition doesn't clear _entered prematurely.
        if self._enter_click_cooldown == 0 and self.detect_screen_yolo(screen) == "Lobby":
            self._entered = False

        if self._on_tasks_page(screen):
            self.log("inside daily tasks")
            self.sub_state = "claim"
            self._post_claim_wait = 2
            return action_wait(500, "entered daily tasks")

        # Cooldown so we don't re-tap the 任務大廳 entry icon.
        if self._enter_click_cooldown > 0:
            self._enter_click_cooldown -= 1
            return action_wait(500, f"daily_tasks: enter cooldown ({self._enter_click_cooldown})")

        # YOLO 任務大廳入口 cls (sidebar tasks icon).
        tasks_btn = self.find_cls(screen, UC.NAV_TASKS, conf=0.30)
        if tasks_btn is not None:
            self._enter_click_cooldown = 4
            self._entered = True  # entry-clicked gate: now in-page is trusted
            return action_click_box(tasks_btn, "open daily tasks (YOLO 任務大廳入口)")

        # Not lobby + not tasks → back out toward lobby.
        if self.detect_screen_yolo(screen) not in (None, "Lobby"):
            return action_back("daily_tasks: backing toward lobby")
        self.log("no 任務大廳入口 visible — YOLO gap; waiting")
        return action_wait(400, "daily_tasks: waiting for 任務大廳入口 detection")

    def _claim(self, screen: ScreenState) -> Dict[str, Any]:
        self._claim_attempts += 1

        if self._post_claim_wait > 0:
            self._post_claim_wait -= 1
            return action_wait(500, f"daily_tasks: settling ({self._post_claim_wait})")

        # Lowered 25→12: claim-all keeps rendering yellow for a frame or two
        # after the queue drains, so the old cap burned ~100 ticks re-clicking
        # an already-empty list. With _post_claim_wait settling each claim, 12
        # real claim attempts is plenty.
        if self._claim_attempts > 12:
            self.log(f"claim loop done ({self._claimed_count} claimed)")
            self.sub_state = "exit"
            return action_wait(300, "claim attempts exhausted")

        # Claim-all (全部領取_黃). Only the YELLOW (active) state — grey means
        # already swept. The tier-bonus bar sits high (y≈0.02-0.10), so allow
        # the full vertical span on the right edge.
        claim_all = self.find_cls(
            screen, UC.CLAIM_ALL_YELLOW, conf=0.25, region=(0.55, 0.02, 1.0, 0.98),
        )
        if claim_all is not None:
            self.log(f"claim all tasks (#{self._claim_attempts}, cls={claim_all.cls_name})")
            self._claimed_count += 1
            self._post_claim_wait = 3
            return action_click_box(claim_all, f"claim all tasks (YOLO {claim_all.cls_name})")

        # Per-row 領取_黃 / 領取獎勵_黃 — catches tier-bonus rows the sweep
        # didn't get. Only yellow (active) variants; grey = already claimed.
        single = self.find_cls(
            screen, [UC.CLAIM_YELLOW, UC.CLAIM_REWARD_YELLOW],
            conf=0.25, region=(0.55, 0.02, 1.0, 0.98),
        )
        if single is not None:
            self.log(f"claim single task (#{self._claim_attempts}, cls={single.cls_name})")
            self._claimed_count += 1
            self._post_claim_wait = 3
            return action_click_box(single, f"claim single task (YOLO {single.cls_name})")

        # No active claim cls. If a grey claim-all is present, everything's
        # swept → exit cleanly.
        if self.find_cls(screen, UC.CLAIM_ALL_GREY, conf=0.25) is not None:
            self.log(f"all claimed (全部領取_灰 present, {self._claimed_count} claimed)")
            self.sub_state = "exit"
            return action_wait(300, "all tasks claimed (grey claim-all)")

        # Nothing to claim and no grey marker — give it a couple of ticks in
        # case a popup/animation is settling, then conclude.
        if self._claim_attempts > 6:
            self.log(f"no more claim cls ({self._claimed_count} claimed)")
            self.sub_state = "exit"
            return action_wait(300, "no claim cls found")
        return action_wait(400, "daily_tasks: waiting for claim cls (YOLO gap)")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        # On lobby iff we see >=2 nav icons.
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log(f"done ({self._claimed_count} tasks claimed)")
            return action_done(f"daily_tasks complete ({self._claimed_count} claimed)")
        # Prefer YOLO home/back button over blind ESC.
        home = self.find_cls(screen, UC.BTN_HOME, conf=0.30)
        if home is not None:
            return action_click_box(home, "daily_tasks exit: home button")
        back = self.find_cls(screen, UC.BTN_BACK, conf=0.30)
        if back is not None:
            return action_click_box(back, "daily_tasks exit: back button")
        return action_back("daily_tasks exit: ESC toward lobby")
