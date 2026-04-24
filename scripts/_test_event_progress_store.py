"""Smoke test for brain/skills/event_progress.py.

Uses a temp directory so it doesn't touch the real data/ folder.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from brain.skills.event_progress import EventProgressStore, PhaseProgress  # noqa: E402


def run_phase_progress_tests() -> int:
    pp = PhaseProgress()
    assert pp.next_node(total=5) == 1
    pp.add(1)
    pp.add(3)
    assert pp.completed == [1, 3]
    assert pp.next_node(total=5) == 2
    pp.add(3)  # idempotent
    assert pp.completed == [1, 3]
    assert not pp.done(total=5)
    for i in range(1, 6):
        pp.add(i)
    assert pp.done(total=5)
    assert pp.next_node(total=5) is None
    print("PASS PhaseProgress: add, next_node, done")
    return 0


def run_store_tests() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        meta_file = tmp_path / "event_metadata.json"
        prog_file = tmp_path / "event_progress.json"
        meta_file.write_text(
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

        s = EventProgressStore(metadata_path=meta_file, progress_path=prog_file)

        # Metadata loaded
        meta = s.metadata("serenade_promenade")
        assert meta.total_story == 8
        assert meta.total_mission == 12
        assert meta.phase_order == ["story", "mission", "challenge"]
        print("PASS metadata loaded correctly")

        # Unknown event returns safe default (zero totals = phase_done)
        unknown = s.metadata("unknown_event")
        assert unknown.total_story == 0
        assert s.phase_done("unknown_event", "story")
        print("PASS unknown event returns safe default (phase_done=True)")

        # Progress marking
        assert s.next_node("serenade_promenade", "story") == 1
        s.mark_done("serenade_promenade", "story", 1)
        s.mark_done("serenade_promenade", "story", 2)
        s.mark_done("serenade_promenade", "story", 1)  # idempotent
        assert s.next_node("serenade_promenade", "story") == 3
        prog = s.progress("serenade_promenade")
        assert prog.story.completed == [1, 2]
        assert prog.last_phase == "story"
        print("PASS mark_done is idempotent and advances next_node")

        # Persistence: reload from disk
        s2 = EventProgressStore(metadata_path=meta_file, progress_path=prog_file)
        prog2 = s2.progress("serenade_promenade")
        assert prog2.story.completed == [1, 2]
        assert s2.next_node("serenade_promenade", "story") == 3
        print("PASS progress persists across process restarts")

        # Finish the whole story phase
        for i in range(3, 9):
            s2.mark_done("serenade_promenade", "story", i)
        assert s2.phase_done("serenade_promenade", "story")
        assert s2.next_node("serenade_promenade", "story") is None
        # And mission hasn't been touched
        assert not s2.phase_done("serenade_promenade", "mission")
        assert s2.next_node("serenade_promenade", "mission") == 1
        print("PASS story_done doesn't leak into mission")

        # Summary is human-readable
        summary = s2.summary("serenade_promenade")
        assert summary == {"story": "8/8", "mission": "0/12", "challenge": "0/4"}
        print(f"PASS summary reads: {summary}")

        # Reset wipes progress but keeps metadata
        s2.reset("serenade_promenade")
        assert s2.next_node("serenade_promenade", "story") == 1
        assert s2.metadata("serenade_promenade").total_story == 8
        print("PASS reset wipes progress but preserves metadata")

    return 0


def main() -> int:
    run_phase_progress_tests()
    run_store_tests()
    print("\nRESULT: all event_progress store tests PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
