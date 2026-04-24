"""Regression test: the global interceptor must NOT treat a lobby's
event-widget "指南任務" stamp label as the tutorial-guide panel.

In run_20260423_231649 tick_0001 the Serenade Promenade top-right event
widget had "指南任務" stamped on the character portrait at x≈0.90, y≈0.31.
The interceptor used a region-less text match and fired, clicking
(0.98, 0.03) — which on the lobby is the event widget's fullscreen-
expand button, not a close-panel home icon. That opened a splash and
derailed the event-entry flow for 16 ticks.

This test loads the OCR boxes from that exact tick and asserts that
`_global_interceptor` does NOT produce an action for it after the fix.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from brain.pipeline import DailyPipeline  # noqa: E402
from brain.skills.base import OcrBox, ScreenState  # noqa: E402


TICK = REPO / "data" / "trajectories" / "run_20260423_231649" / "tick_0001.json"


def _build_screen() -> ScreenState:
    data = json.loads(TICK.read_text(encoding="utf-8"))
    screen = ScreenState(
        screenshot_path=str(TICK.with_suffix(".jpg")),
        image_w=data.get("image_w", 0),
        image_h=data.get("image_h", 0),
    )
    for b in data.get("ocr_boxes", []):
        screen.ocr_boxes.append(OcrBox(
            text=b["text"], confidence=b.get("conf", 1.0),
            x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"],
        ))
    return screen


class _StubSkill:
    name = "EventActivity"
    _enter_ticks = 0


def main() -> int:
    screen = _build_screen()
    has_guide_label = any(
        "指南任" in b.text for b in screen.ocr_boxes
    )
    guide_boxes = [
        (b.text, round(b.x1, 3), round(b.y1, 3))
        for b in screen.ocr_boxes if "指南任" in b.text
    ]
    print(f"OCR contains '指南任務' label: {has_guide_label}")
    print(f"  boxes: {guide_boxes}")

    p = DailyPipeline.__new__(DailyPipeline)
    p._interceptor_streak = 0
    action = p._global_interceptor(screen, _StubSkill())

    if action is None:
        print("PASS: interceptor did not fire on lobby frame with event-widget 指南任務")
        return 0

    reason = action.get("reason", "")
    target = action.get("target")
    print(f"FAIL: interceptor fired. reason={reason!r} target={target}")
    if "guide mission" in reason:
        print("  -> this is the exact false-positive the fix was meant to eliminate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
