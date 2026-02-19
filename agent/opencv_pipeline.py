"""
OpenCV-driven deterministic pipeline for Blue Archive daily routine.

Architecture:
  - PipelineController: state machine that drives routine steps sequentially
  - PipelineStep (base): each step detects state, emits actions, checks completion
  - VLM is NOT called here — only Cerebellum template matching + OpenCV color detection
  - The parent agent (VlmPolicyAgent) calls pipeline.tick() each loop iteration;
    if the pipeline returns an action, it is used directly (no VLM needed).
    If the pipeline returns None, VLM fallback kicks in.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore

try:
    from cerebellum import Cerebellum, TemplateMatch
except ImportError:
    Cerebellum = None  # type: ignore
    TemplateMatch = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _center(bbox) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) // 2), int((y1 + y2) // 2)


def _detect_yellow_markers(screenshot_path: str, *, min_area: int = 80, max_area: int = 12000) -> List[Tuple[int, int, int, int]]:
    """Detect yellow exclamation-mark interaction markers in cafe."""
    if cv2 is None or np is None:
        return []
    img = cv2.imread(screenshot_path)
    if img is None:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Yellow range in HSV
    lo = np.array([18, 120, 180], dtype=np.uint8)
    hi = np.array([35, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        if aspect > 3.0 or aspect < 0.2:
            continue
        out.append((x, y, x + w, y + h))
    out.sort(key=lambda b: (b[1], b[0]))
    return out[:12]


# ---------------------------------------------------------------------------
# Pipeline State
# ---------------------------------------------------------------------------

class Phase(Enum):
    IDLE = auto()
    STARTUP = auto()
    LOBBY_CLEANUP = auto()
    CAFE = auto()
    CAFE_EARNINGS = auto()
    CAFE_HEADPAT = auto()
    CAFE_EXIT = auto()
    SCHEDULE = auto()
    CLUB = auto()
    BOUNTIES = auto()
    CRAFT = auto()
    MAIL_TASKS = auto()
    DONE = auto()


@dataclass
class StepState:
    """Per-phase mutable state, reset when entering a new phase."""
    ticks: int = 0
    sub_state: str = ""
    retries: int = 0
    last_click_xy: Optional[Tuple[int, int]] = None
    headpat_done: List[Tuple[int, int]] = field(default_factory=list)
    earnings_claimed: bool = False


@dataclass
class PipelineConfig:
    assets_dir: str = "data/captures"
    confidence: float = 0.20
    max_ticks_per_phase: int = 25
    step_sleep_ms: int = 400
    headpat_offset_x: int = 20
    headpat_offset_y: int = 30


# ---------------------------------------------------------------------------
# PipelineController
# ---------------------------------------------------------------------------

class PipelineController:
    """Deterministic OpenCV pipeline for Blue Archive daily routine."""

    def __init__(self, cerebellum: Optional[Any] = None, cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self._cerebellum = cerebellum
        self._phase = Phase.IDLE
        self._state = StepState()
        self._phase_order: List[Phase] = [
            Phase.STARTUP,
            Phase.LOBBY_CLEANUP,
            Phase.CAFE,
            Phase.CAFE_EARNINGS,
            Phase.CAFE_HEADPAT,
            Phase.CAFE_EXIT,
            # Future phases:
            # Phase.SCHEDULE,
            # Phase.CLUB,
            # Phase.BOUNTIES,
            # Phase.CRAFT,
            # Phase.MAIL_TASKS,
            Phase.DONE,
        ]
        self._phase_index: int = -1
        self._last_action_ts: float = 0.0
        self._started: bool = False
        self._cafe_confirmed: bool = False  # Set by supervision or headpat template

    # -- Public API --

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def is_active(self) -> bool:
        return self._started and self._phase != Phase.DONE and self._phase != Phase.IDLE

    def start(self) -> None:
        self._started = True
        self._phase_index = 0
        self._enter_phase(self._phase_order[0])

    def stop(self) -> None:
        self._started = False
        self._phase = Phase.IDLE
        self._state = StepState()

    def tick(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """
        Main entry point called every agent loop iteration.
        Returns an action dict if the pipeline wants to act, or None to defer to VLM.
        """
        if not self._started or self._phase in (Phase.IDLE, Phase.DONE):
            return None

        self._state.ticks += 1

        # Timeout → skip to next phase
        if self._state.ticks > self.cfg.max_ticks_per_phase:
            print(f"[Pipeline] Phase {self._phase.name} timed out after {self._state.ticks} ticks, skipping.")
            self._advance_phase()
            if self._phase in (Phase.IDLE, Phase.DONE):
                return None
            # Let next tick handle new phase
            return {"action": "wait", "duration_ms": 300, "reason": f"Pipeline: skipped to {self._phase.name} after timeout.", "_pipeline": True}

        # Dispatch to phase handler
        handler = self._get_handler()
        if handler is None:
            return None
        act = handler(screenshot_path=screenshot_path)
        if act is not None:
            act["_pipeline"] = True
            act["_pipeline_phase"] = self._phase.name
            act["_pipeline_tick"] = self._state.ticks
            self._last_action_ts = time.time()
        return act

    def notify_supervision(self, state: str) -> None:
        """Called by the parent agent when VLM supervision returns a state."""
        # Use supervision to validate/correct phase
        if state == "Cafe_Inside":
            self._cafe_confirmed = True
            if self._phase == Phase.CAFE:
                print(f"[Pipeline] Supervision confirms Cafe_Inside, advancing to CAFE_EARNINGS.")
                self._enter_phase(Phase.CAFE_EARNINGS)
        elif state == "Lobby":
            if self._phase in (Phase.CAFE_EARNINGS, Phase.CAFE_HEADPAT):
                # We expected to be in cafe but supervision says lobby — cafe entry failed
                print(f"[Pipeline] Supervision says Lobby but phase is {self._phase.name}, resetting to CAFE (retry).")
                self._cafe_confirmed = False
                self._enter_phase(Phase.CAFE)
            elif self._phase == Phase.CAFE_EXIT:
                # Good — we wanted to return to lobby
                print(f"[Pipeline] Supervision confirms Lobby during CAFE_EXIT, advancing.")
                self._cafe_confirmed = False
                self._advance_phase()

    # -- Phase management --

    def _enter_phase(self, phase: Phase) -> None:
        print(f"[Pipeline] Entering phase: {phase.name}")
        self._phase = phase
        self._state = StepState()

    def _advance_phase(self) -> None:
        try:
            idx = self._phase_order.index(self._phase)
        except ValueError:
            idx = self._phase_index
        idx += 1
        self._phase_index = idx
        if idx >= len(self._phase_order):
            self._enter_phase(Phase.DONE)
        else:
            self._enter_phase(self._phase_order[idx])

    def _get_handler(self):
        return {
            Phase.STARTUP: self._handle_startup,
            Phase.LOBBY_CLEANUP: self._handle_lobby_cleanup,
            Phase.CAFE: self._handle_cafe_enter,
            Phase.CAFE_EARNINGS: self._handle_cafe_earnings,
            Phase.CAFE_HEADPAT: self._handle_cafe_headpat,
            Phase.CAFE_EXIT: self._handle_cafe_exit,
        }.get(self._phase)

    # -- Cerebellum helpers --

    def _match(self, screenshot_path: str, template: str, *, roi: Optional[Tuple[int, int, int, int]] = None, min_score: float = 0.0) -> Optional[TemplateMatch]:
        c = self._cerebellum
        if c is None:
            return None
        m = c.best_match(screenshot_path=screenshot_path, template_name=template, roi=roi)
        if m is None:
            return None
        threshold = max(min_score, self.cfg.confidence)
        if m.score < threshold:
            return None
        return m

    def _click(self, x: int, y: int, reason: str) -> Dict[str, Any]:
        self._state.last_click_xy = (x, y)
        return {"action": "click", "target": [int(x), int(y)], "reason": reason}

    def _wait(self, ms: int, reason: str) -> Dict[str, Any]:
        return {"action": "wait", "duration_ms": int(ms), "reason": reason}

    def _get_size(self, screenshot_path: str) -> Tuple[int, int]:
        if Image is not None:
            try:
                with Image.open(screenshot_path) as im:
                    return im.size  # (w, h)
            except Exception:
                pass
        if cv2 is not None:
            img = cv2.imread(screenshot_path)
            if img is not None:
                return img.shape[1], img.shape[0]
        return 0, 0

    def _is_lobby(self, screenshot_path: str) -> bool:
        """Quick check: are we in the lobby? (nav bar visible at bottom)
        
        Requirements to avoid false positives:
        - Only check bottom 20% of screen (nav bar area)
        - Need 3+ templates matched at high confidence (0.50+)
        - Also verify NO sub-screen indicators (back arrow, home button)
        """
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return False
        # Check for sub-screen indicators first — if present, NOT lobby
        if self._is_subscreen(screenshot_path):
            return False
        roi_nav = (0, int(sh * 0.80), sw, sh)
        hits = 0
        for tmpl in ["咖啡厅.png", "学生.png", "课程表.png", "社交.png", "制造.png", "招募.png"]:
            m = self._match(screenshot_path, tmpl, roi=roi_nav, min_score=0.50)
            if m is not None:
                hits += 1
        return hits >= 3

    def _is_subscreen(self, screenshot_path: str) -> bool:
        """Detect if we're on a sub-screen (活動任務, 劇情, etc.)
        Sub-screens have a Home icon (🏠) AND/OR gear (⚙) at top-right.
        The lobby and cafe do NOT have these top-right indicators.
        NOTE: Do NOT check back arrow alone — cafe also has back arrow."""
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return False
        # Check for Home button at top-right (unique to sub-screens)
        home_roi = (int(sw * 0.90), 0, sw, int(sh * 0.10))
        m = self._match(screenshot_path, "主界面.png", roi=home_roi, min_score=0.50)
        if m is not None:
            return True
        # Check for settings gear at top-right (unique to sub-screens)
        gear_roi = (int(sw * 0.85), 0, sw, int(sh * 0.10))
        m = self._match(screenshot_path, "齿轮.png", roi=gear_roi, min_score=0.50)
        if m is not None:
            return True
        return False

    def _try_go_back(self, screenshot_path: str, reason_prefix: str) -> Optional[Dict[str, Any]]:
        """Try to navigate back to lobby from a sub-screen."""
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return None
        # Try Home button first (most reliable)
        home_roi = (int(sw * 0.90), 0, sw, int(sh * 0.10))
        m = self._match(screenshot_path, "主界面.png", roi=home_roi, min_score=0.40)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"{reason_prefix}: click Home. score={m.score:.3f}")
        # Try back arrow
        back_roi = (0, 0, int(sw * 0.10), int(sh * 0.10))
        m = self._match(screenshot_path, "返回.png", roi=back_roi, min_score=0.40)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"{reason_prefix}: click back arrow. score={m.score:.3f}")
        # Fallback: press ESC
        return {"action": "back", "reason": f"{reason_prefix}: press ESC to go back.", "_pipeline": True}

    def _is_cafe_interior(self, screenshot_path: str) -> bool:
        """Check if we're inside the cafe.
        
        ONLY use headpat template (unique to cafe).
        DO NOT use yellow marker detection — yellow appears on many screens
        (活動任務 立即前往 buttons, lobby event banners, etc.)
        """
        # Check for headpat marker (unique to cafe)
        m = self._match(screenshot_path, "可摸头的标志.png", min_score=0.40)
        if m is not None:
            return True
        # Check for cafe earnings button
        m = self._match(screenshot_path, "咖啡厅收益.png", min_score=0.40)
        if m is not None:
            return True
        # Check for "移动至2号店" button (unique to cafe)
        m = self._match(screenshot_path, "移动至2号店.png", min_score=0.40)
        if m is not None:
            return True
        # Check for invitation ticket (unique to cafe)
        m = self._match(screenshot_path, "邀请卷（带黄点）.png", min_score=0.40)
        if m is not None:
            return True
        return False

    # -----------------------------------------------------------------------
    # Phase handlers — each returns Optional[action_dict]
    # -----------------------------------------------------------------------

    def _handle_startup(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Close startup popups/notices until we reach the lobby."""
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(500, "Pipeline(startup): waiting for screenshot.")

        # Lobby detection: need at least 5 ticks before trusting _is_lobby()
        # to avoid false positives on loading screens
        if self._state.ticks >= 5 and self._is_lobby(screenshot_path):
            print("[Pipeline] Lobby detected, startup complete.")
            self._advance_phase()
            return self._wait(300, "Pipeline(startup): lobby detected, advancing.")

        # Try confirm button (skip dialog) — high confidence only
        confirm_roi = (int(sw * 0.30), int(sh * 0.50), int(sw * 0.85), int(sh * 0.98))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.55)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(startup): click confirm. score={m.score:.3f}")

        # Try dismiss-today checkbox
        dismiss_roi = (0, int(sh * 0.55), int(sw * 0.50), int(sh * 0.85))
        m = self._match(screenshot_path, "今日不再提示（点完今日不再有这个弹窗）.png", roi=dismiss_roi, min_score=0.55)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(startup): dismiss today. score={m.score:.3f}")

        # Try close buttons (X) — require HIGH confidence (0.70+) to avoid false positives
        close_roi = (int(sw * 0.55), 0, sw, int(sh * 0.25))
        best = None
        best_score = 0.0
        for tmpl, min_s in [("内嵌公告的叉.png", 0.70), ("游戏内很多页面窗口的叉.png", 0.65), ("公告叉叉.png", 0.60)]:
            m = self._match(screenshot_path, tmpl, roi=close_roi, min_score=min_s)
            if m is not None and m.score > best_score:
                best = m
                best_score = m.score
        if best is not None:
            return self._click(best.center[0], best.center[1],
                f"Pipeline(startup): close popup. template={best.template} score={best.score:.3f}")

        # Try tap-to-start
        m = self._match(screenshot_path, "点击开始.png", min_score=0.50)
        if m is not None:
            return self._click(sw // 2, int(sh * 0.82),
                f"Pipeline(startup): tap to start. score={m.score:.3f}")

        return self._wait(800, "Pipeline(startup): waiting for game to load.")

    def _handle_lobby_cleanup(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """In lobby — close any remaining popups, then advance to cafe."""
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(lobby): no screenshot.")

        # SAFETY: If we're on a sub-screen, go back to lobby first
        if self._is_subscreen(screenshot_path):
            print(f"[Pipeline] lobby_cleanup: on sub-screen, going back.")
            act = self._try_go_back(screenshot_path, "Pipeline(lobby_cleanup)")
            if act is not None:
                return act

        # Check if there's a REAL popup to close — require VERY high score (0.75+)
        # The popup X button should be clearly visible, not a faint match on lobby UI
        close_roi = (int(sw * 0.55), 0, sw, int(sh * 0.20))
        best = None
        best_score = 0.0
        for tmpl, min_s in [("内嵌公告的叉.png", 0.75), ("游戏内很多页面窗口的叉.png", 0.70)]:
            m = self._match(screenshot_path, tmpl, roi=close_roi, min_score=min_s)
            if m is not None and m.score > best_score:
                best = m
                best_score = m.score
        if best is not None:
            return self._click(best.center[0], best.center[1],
                f"Pipeline(lobby): close popup. template={best.template} score={best.score:.3f}")

        # If lobby nav bar is visible, move to cafe
        if self._is_lobby(screenshot_path):
            self._advance_phase()  # → CAFE
            return self._wait(200, "Pipeline(lobby): clean, advancing to cafe.")

        # Not in lobby — maybe still loading or on wrong screen
        return self._wait(500, "Pipeline(lobby): waiting for lobby.")

    def _handle_cafe_enter(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Click cafe button from lobby, wait for cafe to load."""
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(cafe): no screenshot.")

        # SAFETY: If we're on a sub-screen, go back first
        if self._is_subscreen(screenshot_path):
            print(f"[Pipeline] cafe_enter: on sub-screen, going back to lobby.")
            act = self._try_go_back(screenshot_path, "Pipeline(cafe_enter)")
            if act is not None:
                return act

        # Check if supervision confirmed cafe (more reliable than template)
        if self._cafe_confirmed:
            print("[Pipeline] Cafe confirmed by supervision.")
            self._advance_phase()  # → CAFE_EARNINGS
            return self._wait(200, "Pipeline(cafe): supervision confirmed cafe.")

        # Check via template (headpat marker only — no yellow marker detection)
        if self._is_cafe_interior(screenshot_path):
            print("[Pipeline] Cafe interior detected by template.")
            self._cafe_confirmed = True
            self._advance_phase()  # → CAFE_EARNINGS
            return self._wait(200, "Pipeline(cafe): already in cafe.")

        # In lobby → click cafe button
        if self._is_lobby(screenshot_path):
            roi_nav = (0, int(sh * 0.80), sw, sh)
            m = self._match(screenshot_path, "咖啡厅.png", roi=roi_nav, min_score=0.50)
            if m is not None:
                return self._click(m.center[0], m.center[1],
                    f"Pipeline(cafe): click cafe button. score={m.score:.3f}")
            return self._wait(400, "Pipeline(cafe): cafe button not found, waiting.")

        # Loading screen — just wait
        return self._wait(600, "Pipeline(cafe): waiting for cafe to load.")

    def _handle_cafe_earnings(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Claim cafe earnings if available."""
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(cafe_earnings): no screenshot.")

        # SAFETY: verify we're actually in cafe, not some random screen
        if not self._cafe_confirmed:
            if self._is_lobby(screenshot_path) or self._is_subscreen(screenshot_path):
                print("[Pipeline] cafe_earnings: not in cafe! Going back to lobby.")
                self._enter_phase(Phase.LOBBY_CLEANUP)
                return self._wait(300, "Pipeline(cafe_earnings): not in cafe, resetting.")

        if self._state.earnings_claimed:
            self._advance_phase()  # → CAFE_HEADPAT
            return self._wait(200, "Pipeline(cafe_earnings): done, advancing to headpat.")

        # Look for confirm button (earnings claim dialog) — high confidence
        confirm_roi = (int(sw * 0.25), int(sh * 0.40), int(sw * 0.75), int(sh * 0.90))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.55)
        if m is not None:
            self._state.earnings_claimed = True
            return self._click(m.center[0], m.center[1],
                f"Pipeline(cafe_earnings): confirm earnings. score={m.score:.3f}")

        # Try cafe earnings button template
        m = self._match(screenshot_path, "咖啡厅收益.png", min_score=0.40)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(cafe_earnings): click earnings button. score={m.score:.3f}")

        # Blind click: only if cafe is confirmed by supervision AND first tick
        if self._cafe_confirmed and self._state.ticks <= 1:
            tx = int(sw * 0.08)
            ty = int(sh * 0.18)
            return self._click(tx, ty,
                "Pipeline(cafe_earnings): click earnings area (top-left).")

        # After a couple ticks, assume earnings done or not available
        self._state.earnings_claimed = True
        self._advance_phase()  # → CAFE_HEADPAT
        return self._wait(200, "Pipeline(cafe_earnings): no earnings dialog, advancing.")

    def _handle_cafe_headpat(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Tap all students with yellow interaction markers.
        
        ONLY runs when we're actually inside the cafe (confirmed by supervision
        or headpat template). Yellow marker detection is used HERE (inside cafe)
        but NOT for cafe detection.
        """
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(headpat): no screenshot.")

        # SAFETY: verify we're in cafe, not lobby or sub-screen
        if not self._cafe_confirmed:
            print("[Pipeline] headpat: cafe not confirmed, skipping to exit.")
            self._enter_phase(Phase.CAFE_EXIT)
            return self._wait(200, "Pipeline(headpat): cafe not confirmed.")

        if self._is_lobby(screenshot_path):
            print("[Pipeline] headpat: unexpectedly in lobby, skipping.")
            self._cafe_confirmed = False
            self._enter_phase(Phase.CAFE_EXIT)
            return self._wait(200, "Pipeline(headpat): back in lobby unexpectedly.")

        if self._is_subscreen(screenshot_path):
            print("[Pipeline] headpat: on sub-screen, going back.")
            act = self._try_go_back(screenshot_path, "Pipeline(headpat)")
            if act is not None:
                return act

        # Look for confirm button (interaction dialog)
        confirm_roi = (int(sw * 0.25), int(sh * 0.40), int(sw * 0.75), int(sh * 0.90))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.40)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(headpat): confirm dialog. score={m.score:.3f}")

        # Detect yellow markers
        marks = _detect_yellow_markers(screenshot_path)
        # Filter to cafe area (not nav bar, not top bar)
        marks = [(x1, y1, x2, y2) for x1, y1, x2, y2 in marks
                 if 0.12 * sh < (y1 + y2) / 2 < 0.80 * sh]

        if not marks:
            # Also try the headpat template
            m = self._match(screenshot_path, "可摸头的标志.png", min_score=0.25)
            if m is not None:
                # Click slightly below and right of the marker (on the character's head)
                tx = m.center[0] + self.cfg.headpat_offset_x
                ty = m.center[1] + self.cfg.headpat_offset_y
                if (tx, ty) not in self._state.headpat_done:
                    self._state.headpat_done.append((tx, ty))
                    return self._click(tx, ty,
                        f"Pipeline(headpat): click headpat marker. score={m.score:.3f}")

            # No markers left — done with headpat
            self._advance_phase()  # → CAFE_EXIT
            return self._wait(200, "Pipeline(headpat): no more markers, advancing.")

        # Find a marker we haven't clicked yet
        for x1, y1, x2, y2 in marks:
            cx, cy = _center((x1, y1, x2, y2))
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            tx = int(cx + max(self.cfg.headpat_offset_x, int(bw * 0.6)))
            ty = int(cy + max(self.cfg.headpat_offset_y, int(bh * 0.9)))
            # Check if we already clicked near this position
            already = False
            for px, py in self._state.headpat_done:
                if abs(tx - px) + abs(ty - py) < 60:
                    already = True
                    break
            if not already:
                self._state.headpat_done.append((tx, ty))
                return self._click(tx, ty,
                    f"Pipeline(headpat): click yellow marker at ({cx},{cy}).")

        # All markers clicked
        self._advance_phase()  # → CAFE_EXIT
        return self._wait(200, "Pipeline(headpat): all markers done, advancing.")

    def _handle_cafe_exit(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Return to lobby from cafe."""
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(cafe_exit): no screenshot.")

        self._cafe_confirmed = False

        # Already in lobby?
        if self._is_lobby(screenshot_path):
            print("[Pipeline] Back in lobby after cafe.")
            self._advance_phase()  # → next phase or DONE
            return self._wait(300, "Pipeline(cafe_exit): back in lobby.")

        # Try Home button (top-right, most reliable for returning to lobby)
        act = self._try_go_back(screenshot_path, "Pipeline(cafe_exit)")
        if act is not None:
            return act

        # Fallback: click back arrow area directly (always at top-left corner)
        if self._state.ticks % 3 == 1:
            return self._click(int(sw * 0.03), int(sh * 0.04),
                "Pipeline(cafe_exit): click back arrow area.")

        return {"action": "back", "reason": "Pipeline(cafe_exit): press back to return to lobby.", "_pipeline": True}

    # -----------------------------------------------------------------------
    # Debug / status
    # -----------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "phase": self._phase.name,
            "tick": self._state.ticks,
            "active": self.is_active,
            "sub_state": self._state.sub_state,
            "headpat_done": len(self._state.headpat_done),
            "earnings_claimed": self._state.earnings_claimed,
            "cafe_confirmed": self._cafe_confirmed,
        }
