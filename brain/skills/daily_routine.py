"""DailyRoutineSkill: one-shot dispatcher for all daily-harvest sub-flows.

Replaces the old "skill_order has 12 separate harvest skills" UX where the
user had to toggle each one. This single skill cycles through every
sub-flow in fixed order:

    mail → event_activity → cafe → schedule → club → daily_tasks
    → craft → pass_reward → momo_talk → story_mining → shop → ap_planning

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
        _full: List[Tuple[str, BaseSkill, bool]] = [
            # (sub_id, skill_instance, force_run)
            ("buy_pyroxene",  BuyPyroxeneSkill(), False),  # 免费组合包 — 红点才进
            ("club",          ClubSkill(), False),         # 社交 — 红点才进 (10AP→信箱)
            ("craft",         CraftSkill(), True),         # 制造 — ALWAYS enter (user spec)
            ("shop",          ShopSkill(), False),         # 普通商店日购(动态预算)
            ("cafe",          CafeSkill(), False),         # cafe — 收益/邀请/摸头 dot
            ("schedule",      ScheduleSkill(), False),     # 课程表 — 黄点才进 (⚠️青辉石买票)
            ("momo_talk",     MomoTalkSkill(), False),     # MomoTalk 挖矿 — 红/黄点
            ("story_mining",  StoryMiningSkill(), False),  # 剧情挖矿(主线/短篇/支线)
            # mail 是收口：bounty/jfd/arena/club 奖励都汇入信箱 → 放挖矿后、
            # daily_mission前,确保本轮所有奖励都领到(probe: mail最后跑)。
            ("mail",          MailSkill(), False),         # 邮件收口 — 红点才进
            # 每日任务领奖 —— 必须最后跑(其他日常完成才解锁奖励)。
            ("daily_mission", DailyMissionSkill(), False), # 每日任务领奖(收口,最后)
        ]
        if sub_only:
            allow = {str(s).strip() for s in sub_only}
            self._sub_only: Optional[List[str]] = sorted(allow)
            self._plan: List[Tuple[BaseSkill, bool]] = [
                (sk, fr) for (sid, sk, fr) in _full if sid in allow
            ]
        else:
            self._sub_only = None
            self._plan = [(sk, fr) for (_sid, sk, fr) in _full]
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
