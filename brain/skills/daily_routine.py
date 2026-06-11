"""DailyRoutineSkill: one-shot dispatcher for all daily-harvest sub-flows.

Replaces the old "skill_order has 12 separate harvest skills" UX where the
user had to toggle each one. This single skill cycles through the daily
HARVEST sub-flows in fixed order:

    buy_pyroxene → club → craft → shop → cafe → schedule → mail → daily_mission

momo_talk / story_mining are registered but NOT in the default harvest
(user 2026-06-11: bond-story grinding ≠ 收菜) — run them via sub_only.

Per sub-flow:
- Check sub.should_run(screen) (most have a red/yellow dot check on the
  lobby entry icon — skip if no dot).
- CraftSkill explicitly DOES NOT have a dot check — it always enters,
  per user spec ("制造不需要靠黄点识别").
- Run sub.tick(screen) until it returns action 'done', then advance.

Battle / sweep / arena / bounty etc. are NOT in here — those stay as
their own entries in skill_order because user wants explicit control
over battle skills (different AP budgets, ticket priorities, etc.).
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Type

from brain.skills.base import BaseSkill, ScreenState, action_done


class DailyRoutineSkill(BaseSkill):
    """Single skill bundling all daily harvest sub-flows."""

    # (sub_skill_class, force_run_even_without_dot)
    # force_run=True for sub-flows that should ALWAYS enter (no dot check):
    #   - craft: 快速制造 may need to be triggered even when no dot
    #     (per user spec)
    # The dot check itself is encapsulated in each sub-skill's should_run()
    # override (set in commit 9eb9aa0).
    SUB_PLAN: List[Tuple[Type[BaseSkill], bool]] = []  # populated lazily below

    def __init__(self, sub_only=None):
        super().__init__("DailyRoutine")
        # Larger than any single sub-skill's max_ticks (cafe alone = 160,
        # event_activity = 800, plus all the others). Total budget cap so
        # pipeline can break out if something's truly broken.
        self.max_ticks = 2500

        # Lazy import to avoid circular dependency at module load time
        # (pipeline.py imports DailyRoutineSkill, this imports the others)
        from brain.skills.buy_pyroxene import BuyPyroxeneSkill
        from brain.skills.mail import MailSkill
        from brain.skills.cafe import CafeSkill
        from brain.skills.schedule import ScheduleSkill
        from brain.skills.club import ClubSkill
        from brain.skills.craft import CraftSkill
        from brain.skills.momo_talk import MomoTalkSkill
        from brain.skills.story_mining import StoryMiningSkill
        from brain.skills.shop import ShopSkill
        from brain.skills.daily_mission import DailyMissionSkill

        # Order matters — runs top to bottom (user-defined daily order, probe).
        # `force_run` = True means skip the dot check entirely (always enter).
        # Full plan with stable snake-case sub-ids.  `sub_only` (a list of
        # sub-ids) restricts the run to JUST those subs — used for SAFE
        # single-sub live walk-throughs: e.g. schedule (青辉石买票) never even
        # enters the plan unless its id is explicitly whitelisted.  This is a
        # money-safety isolation layer on top of schedule's own 3 guards.
        # (sub_id, skill_instance, force_run, in_default_daily)
        # in_default_daily=False → registered (runnable via sub_only) but NOT
        # part of the unattended daily harvest. User 2026-06-11: 剧情挖矿 +
        # momotalk 挖矿 are bond-story grinding, not 收菜 — separate triggers.
        # Order (user 2026-06-11): 购买青辉石 → 社团 → 制造 → 商店 → 课程表 →
        # 咖啡厅(LAST: cafe earnings grant AP, segueing straight into the task
        # hall block that spends it). mail / daily_mission moved OUT to the
        # TOP-LEVEL order (they run AFTER the hall block so hall rewards are
        # in the mailbox; daily_mission gates on n/8≥7).
        _full: List[Tuple[str, BaseSkill, bool, bool]] = [
            ("buy_pyroxene",  BuyPyroxeneSkill(), False, True),  # 免费组合包 — 红点才进
            ("club",          ClubSkill(), False, True),         # 社交 — 红点才进 (10AP→信箱)
            ("craft",         CraftSkill(), False, True),        # 制造 — 红点才进(造好可领)
            ("shop",          ShopSkill(), False, True),         # 普通商店日购(动态预算)
            ("schedule",      ScheduleSkill(), False, True),     # 课程表 — 黄点才进 (⚠️青辉石买票)
            ("cafe",          CafeSkill(), False, True),         # cafe 最后 — 收益给AP, 衔接任务大厅
            ("momo_talk",     MomoTalkSkill(), False, False),    # MomoTalk 挖矿 — 单独开(非收菜)
            ("story_mining",  StoryMiningSkill(), False, False), # 剧情挖矿 — 单独开(非收菜)
            ("mail",          MailSkill(), False, False),        # → top-level(厅后收口)
            ("daily_mission", DailyMissionSkill(), False, False),# → top-level(n/8≥7, 最后)
        ]
        if sub_only:
            allow = {str(s).strip() for s in sub_only}
            self._sub_only: Optional[List[str]] = sorted(allow)
            # sub_only = the user EXPLICITLY asked for these subs → force-run
            # them (skip the dot gate AND the in_default filter, so momo_talk /
            # story_mining run when explicitly requested). Live 2026-06-10:
            # momo_talk's counted badge ("22") isn't a DOT_RED cls, so the dot
            # gate silently skipped the very sub the walk-through targeted.
            self._plan: List[Tuple[BaseSkill, bool]] = [
                (sk, True) for (sid, sk, _fr, _d) in _full if sid in allow
            ]
        else:
            self._sub_only = None
            # Default unattended daily = harvest subs only (in_default=True).
            self._plan = [(sk, fr) for (_sid, sk, fr, d) in _full if d]
        self._cur_idx: int = 0
        self._cur_started: bool = False

    def reset(self) -> None:
        super().reset()
        self._cur_idx = 0
        self._cur_started = False
        for sub, _ in self._plan:
            try:
                sub.reset()
            except Exception:
                pass

    def should_run(self, screen: ScreenState) -> bool:
        """DailyRoutine itself always runs when user enables it. The internal
        sub-flows have their own dot checks."""
        return True

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        # Global timeout safeguard
        if self.ticks >= self.max_ticks:
            self.log(f"timeout at sub {self._cur_idx}/{len(self._plan)}")
            return action_done("daily routine timeout")

        # Loop through subs that need to be entered but can be skipped via
        # should_run, so we don't waste a tick per skip.
        while self._cur_idx < len(self._plan):
            sub, force_run = self._plan[self._cur_idx]

            # First time we touch this sub — decide whether to enter
            if not self._cur_started:
                if not force_run:
                    try:
                        if not sub.should_run(screen):
                            self.log(f"skip '{sub.name}' (no dot)")
                            self._cur_idx += 1
                            continue
                    except Exception as e:
                        self.log(f"should_run({sub.name}) error: {e}; entering anyway")
                # Enter sub: reset its state, expose its name in our sub_state
                sub.reset()
                self._cur_started = True
                self.sub_state = sub.name
                self.log(f"→ entering '{sub.name}'")
                # ★ YOLO context follows the ACTIVE SUB, not the static
                # DailyRoutine loadout (which carried +cafe+avatar for the
                # whole routine — the emoticon model then ran and drew boxes
                # on every non-cafe screen, e.g. 每日任務, live 2026-06-10).
                # Cafe gets +cafe+avatar, Schedule +avatar, the rest base ui.
                try:
                    from brain.pipeline import (SKILL_YOLO_MAP, BASE_DETECTORS,
                                                set_yolo_context)
                    set_yolo_context(SKILL_YOLO_MAP.get(sub.name, BASE_DETECTORS))
                except Exception:
                    pass

            # Delegate the tick to the current sub-skill
            try:
                action = sub.tick(screen)
            except Exception as e:
                self.log(f"sub '{sub.name}' tick error: {e}; advancing")
                self._cur_idx += 1
                self._cur_started = False
                continue

            # If the sub finished, advance and try the next one in the SAME
            # outer tick (to avoid wasting wall-clock waiting for next frame).
            act = action.get("action", "")
            if act == "done":
                self.log(f"✓ '{sub.name}' done")
                self._cur_idx += 1
                self._cur_started = False
                # If there's another sub, keep looping to enter it; if this
                # was the last one, fall out and return done below.
                if self._cur_idx >= len(self._plan):
                    break
                continue

            # Sub still running — pass its action up to the pipeline
            return action

        # All subs done
        return action_done("daily routine complete — all subs handled")
