"""CampaignSweepSkill: one-shot sweep of the campaign hub.

User directive (2026-05-15): "活动入口战术悬赏通缉这些都在任务里面，没必
要强制回主页。把收菜流程的skill聚合，统一管理还能优化顺序逻辑。然后个
别部分的跳过通过检测黄点红点来。"

Replaces the bounty / arena / event_activity trio in skill_order when
the user wants one consolidated entry that:

  1. Enters the campaign hub ONCE via the right-sidebar 任務 tile.
  2. Scans each visible tile (戰術大賽 / 懸賞通緝 / 學園交流會 / 大決戰
     / etc.) for the red/yellow dot that BA paints over actionable
     tiles.
  3. For each dotted tile, delegates to the matching sub-skill
     instance from the pipeline registry.  Sub-skills are already
     hub-aware (their _enter detects current == "Mission" and clicks
     the right tile, their _exit stays on hub when the next campaign
     skill is queued) so delegation is essentially:
        sub_skill.reset(); loop sub_skill.tick() until "done"
  4. Stays on the hub between sub-skills — no lobby round-trip per
     tile (saves 10-20 ticks each).
  5. After all dotted tiles processed, ESCs back to lobby.

This is a META-skill: it does not duplicate the sub-skills' logic, it
just orchestrates them.  Individual sub-skills (BountySkill,
ArenaSkill, EventActivitySkill) remain in the registry for users who
want fine-grained control via the dashboard skill list.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_wait, action_back, action_done,
)


# Tile-text → sub-skill registry name.  When a tile shows a red/yellow
# dot we look up the sub-skill here and delegate.  Order in this list
# is the priority order for execution (reward-claim first, then
# tickets, then event grinding which can consume a lot of AP).
_TILE_TO_SUBSKILL: List[Tuple[Tuple[str, ...], str, str]] = [
    # tile-text-variants, sub-skill registry key, display label
    (("懸賞通緝", "悬赏通缉"),         "bounty",         "Bounty 悬赏通缉"),
    (("戰術大賽", "战术大赛", "对抗"),  "arena",          "Arena 战术对抗"),
    (("學園交流", "学园交流", "活動進行中", "活动进行中"),
                                       "event_activity", "Event 活动"),
]


class CampaignSweepSkill(BaseSkill):
    def __init__(self):
        super().__init__("CampaignSweep")
        # Pipeline injects this after construction (see DailyPipeline
        # __init__).  Keeps the meta-skill from importing sibling skills
        # directly and creating a cycle.
        self._registry: Dict[str, BaseSkill] = {}
        self.max_ticks = 600  # large budget — covers up to 3 sub-skills

        # Orchestration state
        self._enter_ticks: int = 0
        self._sub_queue: List[Tuple[str, str]] = []  # [(sub_name, display)]
        self._current_sub: Optional[BaseSkill] = None
        self._current_sub_name: str = ""
        self._scanned: bool = False

    def reset(self) -> None:
        super().reset()
        self._enter_ticks = 0
        self._sub_queue = []
        self._current_sub = None
        self._current_sub_name = ""
        self._scanned = False

    def set_registry(self, registry: Dict[str, BaseSkill]) -> None:
        """Pipeline calls this once at construction so the meta-skill
        can delegate without importing sibling skill modules."""
        self._registry = registry

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        if self.ticks >= self.max_ticks:
            self.log("campaign sweep timeout")
            return action_done("campaign sweep timeout")

        # If currently delegating to a sub-skill, forward the tick.
        if self._current_sub is not None:
            sub_action = self._current_sub.tick(screen)
            sub_type = str(sub_action.get("action", ""))
            if sub_type == "done":
                self.log(f"sub-skill '{self._current_sub_name}' returned done: "
                         f"{sub_action.get('reason', '')[:50]}")
                self._current_sub = None
                self._current_sub_name = ""
                # Brief pause so the screen stabilises on hub before
                # we scan tiles again or pick the next sub-skill.
                return action_wait(300, "campaign sweep: sub-skill done, returning to hub")
            return sub_action

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "campaign sweep loading")

        # If we still have queued sub-skills, dispatch the next one.
        if self._sub_queue:
            sub_name, display = self._sub_queue.pop(0)
            sub = self._registry.get(sub_name)
            if sub is None:
                self.log(f"sub-skill '{sub_name}' not in registry, skipping")
                return action_wait(150, f"sub-skill {sub_name} missing")
            try:
                sub.reset()
            except Exception as exc:
                self.log(f"sub-skill '{sub_name}' reset failed: {exc}")
                return action_wait(150, f"sub-skill {sub_name} reset failed")
            self._current_sub = sub
            self._current_sub_name = sub_name
            self.log(f"delegating to sub-skill '{sub_name}' ({display})")
            # Sub-skill's first tick will fire on next iteration.
            return action_wait(200, f"starting {display}")

        # No queue, no current — either we haven't scanned yet OR
        # we're done (all sub-skills processed).
        if not self._scanned:
            return self._enter_and_scan(screen)

        # All done — exit to lobby.
        return self._exit(screen)

    # ── helpers ─────────────────────────────────────────────────────

    def _enter_and_scan(self, screen: ScreenState) -> Dict[str, Any]:
        """Navigate to campaign hub (if not already there), then scan
        tiles for dots and build the sub-skill queue."""
        self._enter_ticks += 1
        current = self.detect_current_screen(screen)

        # If we're not on the campaign hub yet, click the lobby sidebar
        # 任務 button to enter.
        if current != "Mission":
            if current == "Lobby":
                campaign_btn = screen.find_any_text(
                    ["任務", "任务"],
                    region=(0.80, 0.70, 1.0, 0.90),
                    min_conf=0.6,
                )
                if campaign_btn:
                    return action_click_box(campaign_btn, "campaign sweep: open campaign from lobby")
                if self._enter_ticks > 3:
                    return action_click(0.95, 0.83, "campaign sweep: open campaign (hardcoded)")
                return action_wait(400, "campaign sweep: waiting for lobby UI")
            # Unknown screen — try ESC out, but give up after a few attempts
            if self._enter_ticks > 10:
                self.log(f"can't reach campaign hub from {current!r}, giving up")
                self._scanned = True  # short-circuit to exit
                return action_wait(200, "campaign sweep: can't reach hub")
            if current:
                return action_back(f"campaign sweep: back from {current}")
            return action_wait(500, "campaign sweep: entering")

        # On campaign hub — scan tiles for red/yellow dots.
        tiles_with_dot = self._scan_tiles(screen)
        self.log(f"campaign hub scan: {len(tiles_with_dot)} actionable tiles "
                 f"{[d for _, d, _ in tiles_with_dot] or '(none)'}")
        self._scanned = True
        if not tiles_with_dot:
            self.log("no actionable campaign tiles — nothing to do")
            return action_wait(200, "campaign sweep: no dots, nothing to do")
        # Build queue in priority order (tickets first, event last).
        self._sub_queue = [(sub_name, display) for _, display, sub_name in tiles_with_dot]
        return action_wait(200, f"campaign sweep: {len(self._sub_queue)} sub-skills queued")

    def _scan_tiles(self, screen: ScreenState) -> List[Tuple[Any, str, str]]:
        """Return list of (anchor_box, display_label, sub_skill_name)
        for tiles that currently have a red OR yellow badge.

        Order follows _TILE_TO_SUBSKILL priority.
        """
        out: List[Tuple[Any, str, str]] = []
        for variants, sub_name, display in _TILE_TO_SUBSKILL:
            anchor = None
            for v in variants:
                anchor = screen.find_text_one(v, min_conf=0.55)
                if anchor:
                    break
            if anchor is None:
                continue
            # Tile dots live at the tile's top-right corner.  Probe a
            # tight box anchored on the tile-label position.  Variants
            # in tile size are small (templated grid) so a single
            # offset is enough.
            state = screen.badge_state(anchor)
            if state in ("red", "yellow"):
                out.append((anchor, display, sub_name))
                self.log(f"tile {display!r}: {state} dot")
        return out

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log("campaign sweep complete")
            return action_done("campaign sweep complete")
        return action_back("campaign sweep: back to lobby")
