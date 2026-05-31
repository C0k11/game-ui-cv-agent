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
from brain.skills import ui_classes as UC


# Tile-cls → (sub_skill_name, display_label, gate_on_dot).
#
# PURE-YOLO (2026-05-29): hub tiles are located by the UI model's cls
# (HUB_BOUNTY / HUB_ARENA), not OCR text. The model was trained on the
# stylized tile labels directly, so this is more robust than the old
# chopped-OCR variants.
#
# gate_on_dot semantics (per user 2026-05-17):
#   - Bounty / Arena: ALWAYS queue. Tile dot doesn't reflect ticket
#     availability — tickets reset daily and the bot should always try to
#     spend them. Sub-skill exits quickly when 0 tickets so cost is bounded.
#   - Event: gate on a red/yellow dot near the tile (DOT_RED/DOT_YELLOW).
_TILE_TO_SUBSKILL: List[Tuple[str, str, str, bool]] = [
    # (tile cls_name, sub-skill registry key, display label, gate_on_dot)
    (UC.HUB_BOUNTY, "bounty", "Bounty 悬赏通缉", False),
    (UC.HUB_ARENA,  "arena",  "Arena 战术对抗", False),
    # NOTE 2026-05-28: 周年庆活动页面反常，临时禁用 event_activity sub。
    # 跑完恢复这一项。
    # (UC.HUB_SCHOOL_EXCHANGE, "event_activity", "Event 活动", True),
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

    def _set_yolo_for(self, skill_name: str) -> None:
        """Set the global YOLO detector context to match ``skill_name``'s
        loadout (e.g. Arena → ui+battle+avatar so opponent heads are
        detected). Lazy-imports from pipeline to avoid an import cycle
        (pipeline imports this module at top level). Best-effort: a failure
        just leaves the previous context — degrades to fewer detectors,
        never crashes. Without this, sub-skills delegated by the sweep run
        with the sweep's base 'ui' context and can't see avatar/battle
        boxes (H2: arena selected 0 opponents via the sweep)."""
        try:
            from brain.pipeline import (
                set_yolo_context, SKILL_YOLO_MAP, BASE_DETECTORS,
            )
            set_yolo_context(SKILL_YOLO_MAP.get(skill_name, BASE_DETECTORS))
        except Exception as exc:  # pragma: no cover - defensive
            self.log(f"_set_yolo_for({skill_name!r}) failed: {exc}")

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
                # H2: restore base detector loadout ("ui") for the hub tile scan
                # — drops the heavy avatar/battle nets when not in a sub-skill.
                self._set_yolo_for(self.name)
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
            # H2: load the sub-skill's detector loadout (arena needs avatar to
            # find opponent heads). Context is read at DETECTION time, so
            # setting it now applies from the sub's first real tick next loop.
            self._set_yolo_for(sub.name)
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

        # If we're not on the campaign hub yet, click the lobby
        # 任务大厅入口 tile (YOLO) to enter.
        if current != "Mission":
            if current == "Lobby":
                act = self.click_cls(
                    screen, UC.NAV_TASKS,
                    "campaign sweep: open hub from lobby", conf=0.30,
                )
                if act:
                    return act
                return action_wait(400, "campaign sweep: 任务大厅入口 cls not seen yet")
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
        for tiles that should be visited this run.

        Sub-skills with gate_on_dot=False (Bounty, Arena) are queued
        unconditionally if their tile is found on the hub — tickets
        reset daily, dot only signals REWARDS not tickets, sub-skill
        exits in ~10 ticks when no work.

        Sub-skills with gate_on_dot=True (Event) require a red/yellow
        dot on the tile.

        Order follows _TILE_TO_SUBSKILL priority.
        """
        out: List[Tuple[Any, str, str]] = []
        for tile_cls, sub_name, display, gate_on_dot in _TILE_TO_SUBSKILL:
            anchor = self.find_cls(screen, tile_cls, conf=0.30)
            if anchor is None:
                self.log(f"tile {display!r}: cls {tile_cls!r} not on hub")
                continue
            if not gate_on_dot:
                out.append((anchor, display, sub_name))
                self.log(f"tile {display!r}: queued (cls {tile_cls!r} {anchor.confidence:.2f})")
                continue
            # Gated: queue only if a red/yellow dot sits over the tile.
            if self._tile_has_dot(screen, anchor):
                out.append((anchor, display, sub_name))
                self.log(f"tile {display!r}: dot present, queued")
            else:
                self.log(f"tile {display!r}: no dot, skipping")
        return out

    def _tile_has_dot(self, screen: ScreenState, tile) -> bool:
        """True if a DOT_RED/DOT_YELLOW cls sits inside / just above the
        tile box (BA paints the dot at the tile's top-right corner)."""
        for d in self.find_all_cls(screen, [UC.DOT_RED, UC.DOT_YELLOW], conf=0.30):
            if (tile.x1 - 0.01) <= d.cx <= (tile.x2 + 0.03) and \
               (tile.y1 - 0.05) <= d.cy <= (tile.y2 + 0.01):
                return True
        return False

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log("campaign sweep complete")
            return action_done("campaign sweep complete")
        return action_back("campaign sweep: back to lobby")
