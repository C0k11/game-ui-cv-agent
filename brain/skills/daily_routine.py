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

    def __init__(self):
        super().__init__("DailyRoutine")
        # Larger than any single sub-skill's max_ticks (cafe alone = 160,
        # event_activity = 800, plus all the others). Total budget cap so
        # pipeline can break out if something's truly broken.
        self.max_ticks = 2500

        # Lazy import to avoid circular dependency at module load time
        # (pipeline.py imports DailyRoutineSkill, this imports the others)
        from brain.skills.mail import MailSkill
        from brain.skills.event_activity import EventActivitySkill
        from brain.skills.cafe import CafeSkill
        from brain.skills.schedule import ScheduleSkill
        from brain.skills.club import ClubSkill
        from brain.skills.daily_tasks import DailyTasksSkill
        from brain.skills.craft import CraftSkill
        from brain.skills.pass_reward import PassRewardSkill
        from brain.skills.momo_talk import MomoTalkSkill
        from brain.skills.story_mining import StoryMiningSkill
        from brain.skills.shop import ShopSkill
        from brain.skills.ap_planning import ApPlanningSkill

        # Order matters — runs top to bottom.
        # `force_run` = True means skip the dot check entirely (always enter).
        # NOTE 2026-05-28: EventActivity 临时跳过 —— 周年庆活动页面反常，
        # 先把其他日常都跑通再回头处理活动。跑完恢复这一行。
        self._plan: List[Tuple[BaseSkill, bool]] = [
            # (skill_instance, force_run)
            (MailSkill(), False),            # 邮件 — 红点才进
            # (EventActivitySkill(), False),   # 活动 — 周年庆反常临时跳过
            (CafeSkill(), False),            # cafe — 收益/邀请/摸头 dot
            (ScheduleSkill(), False),        # 课程表 — 黄点才进
            (ClubSkill(), False),            # 社交 — 红点才进 (AP)
            (DailyTasksSkill(), False),      # 任务大厅 — 红点
            (CraftSkill(), True),            # 制造 — ALWAYS enter (user spec)
            (PassRewardSkill(), False),      # 战令 — 红点
            (MomoTalkSkill(), False),        # MomoTalk — 红/黄点
            (StoryMiningSkill(), False),     # 主线挖矿
            (ShopSkill(), False),            # 普通商店日购
            (ApPlanningSkill(), True),       # 补给/免费AP — 总是 check
        ]
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
