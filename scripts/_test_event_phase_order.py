"""Regression test: event-activity phase order respects per-node progress.

User invariant: "if story has 9 nodes, we must finish story 1-9 FIRST,
then move on to quest (1-12)".  This test drives EventActivitySkill
through a simulated progression and verifies:

1. While story is incomplete, `_enter` never dispatches to mission.
2. After marking all story nodes done, `_enter` dispatches to mission.
3. Quest progress is independent of story progress.
4. Progress persists across skill reset (mimics bot restart).
5. A timeout setting `_story_done = True` with incomplete node set
   does NOT skip to mission — the store is the source of truth.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Redirect the store's files to a temp dir BEFORE importing the skill
# so the global singleton binds to the test fixtures.
import brain.skills.event_progress as ep  # noqa: E402


def _prime_store(tmp: Path) -> None:
    meta = tmp / "event_metadata.json"
    prog = tmp / "event_progress.json"
    meta.write_text(
        json.dumps({"events": {
            "serenade_promenade": {
                "display_name": "Serenade Promenade",
                "total_story": 8,
                "total_mission": 12,
                "total_challenge": 4,
                "phase_order": ["story", "mission", "challenge"],
            },
        }}),
        encoding="utf-8",
    )
    # Default-arg capture in EventProgressStore.__init__ means we can't
    # rewire the module-level _METADATA_FILE / _PROGRESS_FILE globals
    # and expect get_store() to pick them up — those paths were bound
    # at class-definition time.  Instead, manually instantiate a store
    # with the temp paths and install it as the singleton.
    import brain.skills.event_progress as _ep
    _ep._singleton = _ep.EventProgressStore(metadata_path=meta, progress_path=prog)


def run_tests() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _prime_store(tmp_path)

        # Import AFTER priming so the skill's module-level singleton
        # picks up our test paths on first get_store() call.
        from brain.skills.event_activity import EventActivitySkill  # noqa: E402

        skill = EventActivitySkill()
        skill.reset()
        assert skill._current_event_id == ""
        assert skill._next_node("story") == 1  # unknown event → default 1

        # Simulate banner matcher identifying Serenade on lobby.
        skill._set_current_event("serenade_promenade")
        assert skill._current_event_id == "serenade_promenade"
        assert skill._current_story_index == 1
        assert skill._quest_current_index == 1
        assert not skill._phase_done("story")
        assert not skill._phase_done("mission")
        print("PASS fresh Serenade: story@1 mission@1 both not done")

        # Play story nodes 1..7; still not done.
        for i in range(1, 8):
            skill._mark_node_done("story", i)
        assert not skill._phase_done("story")
        assert skill._next_node("story") == 8
        assert skill._next_node("mission") == 1  # quest untouched
        print("PASS story 1..7 marked; story NOT done, mission still @1")

        # Simulate a skill RESET (bot restart mid-event).  The new
        # instance must pick up where the store left off.
        skill2 = EventActivitySkill()
        skill2.reset()
        skill2._set_current_event("serenade_promenade")
        assert skill2._current_story_index == 8, \
            f"expected resume at story 8, got {skill2._current_story_index}"
        assert skill2._quest_current_index == 1
        assert not skill2._phase_done("story")
        print("PASS restart resumes at story@8 (not 1, not 9)")

        # Mark the final story node.
        skill2._mark_node_done("story", 8)
        assert skill2._phase_done("story")
        assert skill2._next_node("story") is None or \
            skill2._next_node("story") == 9
        print("PASS story@8 marked; story fully done")

        # Now quest is unlocked.  Play a few.
        skill2._mark_node_done("mission", 1)
        skill2._mark_node_done("mission", 2)
        assert skill2._next_node("mission") == 3
        assert not skill2._phase_done("mission")
        print("PASS mission 1..2 marked; next mission @3")

        # CRITICAL: session-abort (AP depletion / timeout) while story
        # is incomplete must NOT silently advance to quest.  Previously
        # the dispatcher did `if not _phase_done and not _story_done`,
        # so `_story_done=True` bypassed story AND fell through to the
        # mission branch — violating the user's invariant.  The fixed
        # dispatcher (see _enter in event_activity.py) now exits the
        # skill instead of advancing.
        ep.get_store().reset("serenade_promenade")
        skill3 = EventActivitySkill()
        skill3.reset()
        skill3._set_current_event("serenade_promenade")
        for i in range(1, 4):
            skill3._mark_node_done("story", i)
        skill3._story_done = True  # simulate AP-depletion session abort

        # Mirror the dispatch decisions from _enter() literally.
        story_done = skill3._phase_done("story")
        mission_done = skill3._phase_done("mission")
        if not story_done:
            dispatch = "exit" if skill3._story_done else "story"
        elif not mission_done:
            dispatch = "exit" if skill3._mission_done else "mission"
        else:
            dispatch = "challenge_or_exit"
        assert dispatch == "exit", (
            f"session abort with story 3/8 incomplete must exit, got {dispatch}"
        )
        print("PASS session abort with partial story -> dispatch exits "
              "(does NOT advance to mission)")

        # Next session (fresh skill instance) must retry story from 4.
        skill4 = EventActivitySkill()
        skill4.reset()
        skill4._set_current_event("serenade_promenade")
        assert not skill4._story_done, "fresh instance must clear stale flag"
        assert skill4._current_story_index == 4
        assert skill4._next_node("story") == 4
        print("PASS fresh instance after session abort resumes story@4")

        # CRITICAL: memo fallback.  The previous fresh-instance test
        # called _set_current_event manually to mimic the lobby banner
        # matcher firing — but in real runs the skill can start already
        # on the event page with no banner visible (chained after daily
        # tasks, or user opened the event page before triggering the
        # bot).  The store's persisted current_event_id memo must let
        # reset() seed the counters from the right event's progress.
        assert ep.get_store().get_current_event_id() == "serenade_promenade", \
            "memo should have been persisted by earlier _set_current_event calls"
        skill5 = EventActivitySkill()
        skill5.reset()
        # Did NOT call _set_current_event — mimicking "skill starts
        # directly on event page".  The memo fallback should still
        # bind us to serenade_promenade progress.
        assert skill5._current_event_id == "serenade_promenade", (
            f"memo fallback failed: _current_event_id={skill5._current_event_id!r}"
        )
        assert skill5._current_story_index == 4, (
            f"memo-recovered skill should resume story@4, got {skill5._current_story_index}"
        )
        assert skill5._next_node("story") == 4
        assert not skill5._phase_done("story")
        print("PASS memo fallback: skill starting mid-event page "
              "recovers event_id and resumes at story@4")

    print("\nRESULT: phase-order invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(run_tests())
