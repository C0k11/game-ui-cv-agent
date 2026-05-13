"""Template regression test — replay historical trajectory ticks through
templates to verify the new template-first logic catches what OCR missed.

For each test case (tick + template + expected outcome), this loads the
saved screenshot, runs the template matcher against the configured
region, and prints whether the expected behavior would fire.

Usage:
    py scripts/test_template_regression.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
TRAJ_DIR = REPO / "data" / "trajectories"

from vision.template_matcher import get_template_matcher  # noqa: E402


# Test cases derived from observed failure ticks in trajectories.
# Each entry verifies that the template-first path now catches what OCR
# previously missed.  Format:
#   tick_jpg:    trajectory tick image (relative to TRAJ_DIR)
#   template:    name in _TEMPLATE_DEFS
#   region:      (x1, y1, x2, y2) normalized search region (None = full)
#   threshold:   min match score
#   expected:    "hit" or "miss"
#   note:        what real-world bug this prevents
TESTS = [
    # cafe invite list — t77 visible students, should detect 邀請 buttons
    {
        "tick_jpg": "run_20260513_112359/tick_0077.jpg",
        "template": "cafe_invite_button",
        "region":   (0.50, 0.20, 0.70, 0.90),
        "threshold":0.65,
        "expected": "hit",
        "note":     "5-6 邀請 row buttons should be detected on cafe invite list",
    },
    # cafe t114 — OCR misread 邀請 as 返明; template should still see real buttons
    {
        "tick_jpg": "run_20260513_112359/tick_0114.jpg",
        "template": "cafe_invite_button",
        "region":   (0.50, 0.20, 0.70, 0.90),
        "threshold":0.65,
        "expected": "hit",
        "note":     "邀請 buttons visible despite OCR returning '返明' — prevents tutorial false-positive",
    },
    # event_activity DEFEAT screen — fight-success-confirm template
    {
        "tick_jpg": "run_20260513_112359/tick_0688.jpg",
        "template": "activity_fight_confirm",
        "region":   (0.30, 0.80, 0.70, 0.99),
        "threshold":0.65,
        "expected": "hit",
        "note":     "DEFEAT screen yellow 確認 button — fixes scripted-loss stuck",
    },
    # shop tab0 (gacha single-item) — purchase button visible
    {
        "tick_jpg": "run_20260504_231557/tick_0149.jpg",
        "template": "shop_purchase_avail",
        "region":   (0.45, 0.30, 1.0, 0.95),
        "threshold":0.65,
        "expected": "hit",
        "note":     "gacha tab single 購買 button — confirms shop single-item-tab detection",
    },
    # auto-cropped buttons regression on the very frame they were cropped from
    {
        "tick_jpg": "run_20260318_121602/tick_0012.jpg",
        "template": "task_start_button",
        "region":   (0.55, 0.55, 0.95, 0.90),
        "threshold":0.70,
        "expected": "hit",
        "note":     "任務開始 yellow sortie btn — self-test on crop source",
    },
    {
        "tick_jpg": "run_20260428_184236/tick_0011.jpg",
        "template": "sweep_start_button",
        "region":   (0.55, 0.40, 0.95, 0.85),
        "threshold":0.70,
        "expected": "hit",
        "note":     "掃蕩開始 cyan sweep btn — self-test on crop source",
    },
    {
        "tick_jpg": "run_20260305_171137/tick_0437.jpg",
        "template": "sortie_button",
        "region":   (0.65, 0.70, 1.00, 0.99),
        "threshold":0.70,
        "expected": "hit",
        "note":     "出擊 formation sortie btn — self-test on crop source",
    },
]


def load_img(jpg: Path) -> Optional[np.ndarray]:
    if not jpg.exists():
        return None
    return cv2.imdecode(np.fromfile(jpg, np.uint8), cv2.IMREAD_COLOR)


def run_test(test: dict) -> Tuple[bool, str]:
    jpg = TRAJ_DIR / test["tick_jpg"]
    img = load_img(jpg)
    if img is None:
        return False, f"image not found: {jpg}"
    matcher = get_template_matcher(test["template"])
    if matcher is None:
        return False, f"template '{test['template']}' not registered"
    hits = matcher.match(img, region=test["region"], threshold=test["threshold"])
    has_hit = len(hits) > 0
    expected_hit = test["expected"] == "hit"
    ok = has_hit == expected_hit
    if hits:
        top_score = hits[0].confidence
        detail = f"{len(hits)} hits (top: {top_score:.2f})"
        if len(hits) <= 5:
            positions = [f"({h.cx:.2f},{h.cy:.2f}):{h.confidence:.2f}" for h in hits]
            detail += f" — {positions}"
    else:
        detail = "0 hits"
    return ok, detail


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    passed = failed = 0
    for i, test in enumerate(TESTS, 1):
        ok, detail = run_test(test)
        mark = "✓" if ok else "✗"
        status = "PASS" if ok else "FAIL"
        print(f"[{i}] {mark} {status}  {test['template']:<28} @ {test['tick_jpg']}")
        print(f"     expected: {test['expected']:<5}  got: {detail}")
        print(f"     ({test['note']})")
        if ok:
            passed += 1
        else:
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
