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
    CAFE_INVITE = auto()
    CAFE_HEADPAT = auto()
    CAFE_SWITCH = auto()       # move to cafe 2F
    CAFE_2_EARNINGS = auto()   # collect earnings in cafe 2F
    CAFE_2_INVITE = auto()     # invite student in cafe 2F
    CAFE_2_HEADPAT = auto()    # headpat in cafe 2F
    CAFE_EXIT = auto()
    SCHEDULE_ENTER = auto()
    SCHEDULE_EXECUTE = auto()
    SCHEDULE = auto()          # legacy alias
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
    last_popup_close_tick: int = -10  # tick at which last popup was closed


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
            Phase.CAFE_INVITE,
            Phase.CAFE_HEADPAT,
            Phase.CAFE_SWITCH,
            Phase.CAFE_2_EARNINGS,
            Phase.CAFE_2_INVITE,
            Phase.CAFE_2_HEADPAT,
            Phase.CAFE_EXIT,
            Phase.SCHEDULE_ENTER,
            Phase.SCHEDULE_EXECUTE,
            # Future phases:
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
        self._last_sup_state: str = ""  # Last supervision state from VLM
        self._cafe_actually_entered: bool = False  # True only if we actually entered cafe
        self._done_restart_count: int = 0  # prevent infinite DONE→restart loops

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
            self._cafe_actually_entered = True
            if self._phase == Phase.CAFE:
                print(f"[Pipeline] Supervision confirms Cafe_Inside, advancing to CAFE_EARNINGS.")
                self._enter_phase(Phase.CAFE_EARNINGS)
        elif state == "Lobby":
            if self._phase == Phase.STARTUP:
                # VLM says Lobby during STARTUP — advance past startup
                print(f"[Pipeline] Supervision says Lobby during STARTUP, advancing to LOBBY_CLEANUP.")
                self._enter_phase(Phase.LOBBY_CLEANUP)
            elif self._phase == Phase.LOBBY_CLEANUP:
                # VLM says Lobby during LOBBY_CLEANUP — no popups blocking, advance to CAFE
                if self._state.ticks >= 2:
                    print(f"[Pipeline] Supervision says Lobby during LOBBY_CLEANUP (tick {self._state.ticks}), advancing to CAFE.")
                    self._advance_phase()
            elif self._phase in (Phase.CAFE_EARNINGS, Phase.CAFE_INVITE, Phase.CAFE_HEADPAT,
                                Phase.CAFE_SWITCH, Phase.CAFE_2_EARNINGS,
                                Phase.CAFE_2_INVITE, Phase.CAFE_2_HEADPAT):
                # We expected to be in cafe but supervision says lobby — cafe entry failed
                print(f"[Pipeline] Supervision says Lobby but phase is {self._phase.name}, resetting to CAFE_EXIT.")
                self._cafe_confirmed = False
                self._enter_phase(Phase.CAFE_EXIT)
            elif self._phase == Phase.CAFE_EXIT:
                # Good — we wanted to return to lobby
                print(f"[Pipeline] Supervision confirms Lobby during CAFE_EXIT, advancing.")
                self._cafe_confirmed = False
                self._advance_phase()
            elif self._phase == Phase.DONE:
                # Pipeline reached DONE but cafe was never actually entered — restart (max 2 times)
                if not self._cafe_actually_entered and self._done_restart_count < 2:
                    self._done_restart_count += 1
                    print(f"[Pipeline] Supervision says Lobby, phase is DONE but cafe never entered. Restart #{self._done_restart_count} from LOBBY_CLEANUP.")
                    self._cafe_confirmed = False
                    self._enter_phase(Phase.LOBBY_CLEANUP)
        elif state == "Popup":
            # VLM says there's a popup — if pipeline is DONE, restart to handle it
            if self._phase == Phase.DONE and not self._cafe_actually_entered and self._done_restart_count < 2:
                self._done_restart_count += 1
                print(f"[Pipeline] Supervision says Popup, phase is DONE but cafe never entered. Restart #{self._done_restart_count} from LOBBY_CLEANUP.")
                self._cafe_confirmed = False
                self._enter_phase(Phase.LOBBY_CLEANUP)
        # Track last supervision state for phase handlers to use
        self._last_sup_state = state

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
            Phase.CAFE_INVITE: self._handle_cafe_invite,
            Phase.CAFE_HEADPAT: self._handle_cafe_headpat,
            Phase.CAFE_SWITCH: self._handle_cafe_switch,
            Phase.CAFE_2_EARNINGS: self._handle_cafe_earnings,  # reuse
            Phase.CAFE_2_INVITE: self._handle_cafe_invite,      # reuse (picks 2nd student)
            Phase.CAFE_2_HEADPAT: self._handle_cafe_headpat,    # reuse
            Phase.CAFE_EXIT: self._handle_cafe_exit,
            Phase.SCHEDULE_ENTER: self._handle_schedule_enter,
            Phase.SCHEDULE_EXECUTE: self._handle_schedule_execute,
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

    def _find_all_matches(
        self, screenshot_path: str, template: str, *,
        roi: Optional[Tuple[int, int, int, int]] = None,
        min_score: float = 0.50, nms_dist: int = 30,
        use_color: bool = False,
    ) -> List[TemplateMatch]:
        """Find ALL occurrences of a template in the screenshot (not just the best).

        Uses cv2.matchTemplate and non-maximum suppression to return multiple hits.
        When use_color=True, matches on BGR image (better for color-specific templates).
        """
        if cv2 is None or np is None or self._cerebellum is None:
            return []
        info = self._cerebellum._load_template(str(template))
        if info is None:
            return []
        tmpl_g = info.get("gray")
        tmpl_mask = info.get("mask")
        if tmpl_g is None:
            return []

        scr = self._cerebellum._imread(Path(screenshot_path).resolve(), cv2.IMREAD_COLOR)
        if scr is None:
            return []
        th, tw = tmpl_g.shape[:2]
        sh_full, sw_full = scr.shape[:2]

        # Choose matching image: BGR (color) or grayscale
        if use_color:
            tmpl_bgr = info.get("bgr")
            if tmpl_bgr is None:
                return []
            match_scr = scr.copy()
            match_tmpl = tmpl_bgr
        else:
            match_scr = cv2.cvtColor(scr, cv2.COLOR_BGR2GRAY)
            match_tmpl = tmpl_g

        x_off, y_off = 0, 0
        if roi is not None:
            try:
                x0, y0, x1, y1 = [int(v) for v in roi]
                x0 = max(0, min(sw_full - 1, x0))
                y0 = max(0, min(sh_full - 1, y0))
                x1 = max(x0 + 1, min(sw_full, x1))
                y1 = max(y0 + 1, min(sh_full, y1))
                if (x1 - x0) >= tw and (y1 - y0) >= th:
                    match_scr = match_scr[y0:y1, x0:x1]
                    x_off, y_off = x0, y0
            except Exception:
                x_off, y_off = 0, 0

        # Use mask (from alpha channel) when available for both color and grayscale.
        # TM_CCORR_NORMED supports mask; TM_CCOEFF_NORMED does not.
        use_mask = tmpl_mask is not None
        method = cv2.TM_CCORR_NORMED if use_mask else cv2.TM_CCOEFF_NORMED
        if use_mask:
            res = cv2.matchTemplate(match_scr, match_tmpl, method, mask=tmpl_mask)
        else:
            res = cv2.matchTemplate(match_scr, match_tmpl, method)

        # Find all locations above threshold
        locs = np.where(res >= min_score)
        results: List[TemplateMatch] = []
        candidates = sorted(zip(locs[0], locs[1], res[locs]), key=lambda t: -t[2])

        # Simple NMS: skip candidates too close to already-accepted ones
        accepted: List[Tuple[int, int]] = []
        for cy_raw, cx_raw, score_val in candidates:
            cx_abs = int(cx_raw) + x_off + tw // 2
            cy_abs = int(cy_raw) + y_off + th // 2
            too_close = False
            for ax, ay in accepted:
                if abs(cx_abs - ax) < nms_dist and abs(cy_abs - ay) < nms_dist:
                    too_close = True
                    break
            if too_close:
                continue
            accepted.append((cx_abs, cy_abs))
            x1 = int(cx_raw) + x_off
            y1 = int(cy_raw) + y_off
            results.append(TemplateMatch(
                template=str(template),
                score=float(score_val),
                bbox=(x1, y1, x1 + tw, y1 + th),
                center=(cx_abs, cy_abs),
            ))
            if len(results) >= 20:  # cap results
                break
        return results

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
        - Do NOT call _is_subscreen() here — causes circular dependency
          and false negatives when subscreen detector has false positives
        """
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return False
        roi_nav = (0, int(sh * 0.80), sw, sh)
        hits = 0
        for tmpl in ["咖啡厅.png", "学生.png", "课程表.png", "社交.png", "制造.png", "招募.png"]:
            m = self._match(screenshot_path, tmpl, roi=roi_nav, min_score=0.35)
            if m is not None:
                hits += 1
        return hits >= 3

    def _is_subscreen(self, screenshot_path: str) -> bool:
        """Detect if we're on a sub-screen (活動任務, 劇情, etc.)
        Sub-screens have a Home icon (🏠) AND/OR gear (⚙) at top-right.
        The lobby and cafe do NOT have these top-right indicators.
        NOTE: Do NOT check back arrow alone — cafe also has back arrow.
        
        IMPORTANT: High threshold (0.70+) to avoid false positives!
        The lobby has small icons (⊞ grid, ↗ expand) at top-right that can
        match Home/gear templates at low scores (0.50)."""
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return False
        # Check for Home button at top-right (unique to sub-screens)
        # Use 0.70+ threshold — lobby ⊞ grid icon matched at 0.50!
        home_roi = (int(sw * 0.90), 0, sw, int(sh * 0.10))
        m = self._match(screenshot_path, "Home按钮.png", roi=home_roi, min_score=0.70)
        if m is not None:
            return True
        # Check for settings gear at top-right (unique to sub-screens)
        gear_roi = (int(sw * 0.85), 0, sw, int(sh * 0.10))
        m = self._match(screenshot_path, "设置齿轮.png", roi=gear_roi, min_score=0.70)
        if m is not None:
            return True
        return False

    def _try_go_back(self, screenshot_path: str, reason_prefix: str) -> Optional[Dict[str, Any]]:
        """Try to navigate back to lobby from a sub-screen."""
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return None
        # Try Home button first (most reliable) — need 0.60+ to avoid lobby ⊞ false positive
        home_roi = (int(sw * 0.90), 0, sw, int(sh * 0.10))
        m = self._match(screenshot_path, "Home按钮.png", roi=home_roi, min_score=0.60)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"{reason_prefix}: click Home. score={m.score:.3f}")
        # Try back arrow
        back_roi = (0, 0, int(sw * 0.10), int(sh * 0.10))
        m = self._match(screenshot_path, "返回按钮.png", roi=back_roi, min_score=0.50)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"{reason_prefix}: click back arrow. score={m.score:.3f}")
        # Fallback: press ESC
        return {"action": "back", "reason": f"{reason_prefix}: press ESC to go back.", "_pipeline": True}

    def _is_cafe_interior(self, screenshot_path: str) -> bool:
        """Check if we're inside the cafe.
        
        Uses cafe-specific templates. Does NOT use Emoticon_Action.png
        (scores 0.94 on lobby without ROI — too many false positives).
        """
        # Screenshot-based headpat marker
        # Threshold 0.55: lobby false-positives score 0.39-0.43, cafe real matches should be 0.65+
        m = self._match(screenshot_path, "可摸头的标志.png", min_score=0.55)
        if m is not None:
            return True
        # Check for cafe earnings button
        m = self._match(screenshot_path, "咖啡厅收益按钮.png", min_score=0.55)
        if m is not None:
            return True
        # Check for "移动至2号店" button (unique to cafe)
        m = self._match(screenshot_path, "移动至2号店.png", min_score=0.55)
        if m is not None:
            return True
        # Check for invitation ticket (unique to cafe)
        m = self._match(screenshot_path, "邀请卷（带黄点）.png", min_score=0.55)
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

        # Lobby detection: _is_lobby() requires 3+ nav bar templates matched,
        # which is robust enough to avoid false positives on loading screens.
        # Check from tick 1 so we skip ahead immediately if starting from lobby.
        if self._is_lobby(screenshot_path):
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

        # Try tap-to-start template
        m = self._match(screenshot_path, "点击开始.png", min_score=0.30)
        if m is not None:
            return self._click(sw // 2, int(sh * 0.82),
                f"Pipeline(startup): tap to start (template). score={m.score:.3f}")

        # After several ticks with nothing detected, click center to advance
        # Handles: title screen (TAP TO START), loading transitions, etc.
        if self._state.ticks >= 2 and self._state.ticks % 2 == 0:
            return self._click(sw // 2, int(sh * 0.70),
                f"Pipeline(startup): blind tap center to advance (tick {self._state.ticks}).")

        return self._wait(800, "Pipeline(startup): waiting for game to load.")

    def _handle_lobby_cleanup(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """In lobby — close any remaining popups, then advance to cafe."""
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(lobby): no screenshot.")

        # PRIORITY 1: Check lobby FIRST — if nav bar visible, advance
        lobby_detected = self._is_lobby(screenshot_path)

        # PRIORITY 2: Close any popup (選單 popup, announcement, etc.)
        # Check for 選單 popup X button (center-right area where 選單 X appears)
        menu_close_roi = (int(sw * 0.25), int(sh * 0.05), int(sw * 0.75), int(sh * 0.20))
        m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", roi=menu_close_roi, min_score=0.65)
        if m is not None:
            self._state.last_popup_close_tick = self._state.ticks
            return self._click(m.center[0], m.center[1],
                f"Pipeline(lobby): close menu/popup. template={m.template} score={m.score:.3f}")

        # Check for announcement X button (top-right)
        close_roi = (int(sw * 0.55), 0, sw, int(sh * 0.20))
        best = None
        best_score = 0.0
        for tmpl, min_s in [("内嵌公告的叉.png", 0.75), ("游戏内很多页面窗口的叉.png", 0.70)]:
            m = self._match(screenshot_path, tmpl, roi=close_roi, min_score=min_s)
            if m is not None and m.score > best_score:
                best = m
                best_score = m.score
        if best is not None:
            self._state.last_popup_close_tick = self._state.ticks
            return self._click(best.center[0], best.center[1],
                f"Pipeline(lobby): close popup. template={best.template} score={best.score:.3f}")

        # PRIORITY 3: If lobby nav bar visible and no popups, advance to cafe
        # Require at least 2 clean ticks after last popup close to catch follow-up popups
        if lobby_detected:
            clean_gap = self._state.ticks - self._state.last_popup_close_tick
            if clean_gap >= 2:
                self._advance_phase()  # → CAFE
                return self._wait(200, "Pipeline(lobby): clean, advancing to cafe.")
            else:
                return self._wait(400, "Pipeline(lobby): waiting for follow-up popups after close.")

        # PRIORITY 4: If NOT lobby AND on a sub-screen, go back
        if self._is_subscreen(screenshot_path):
            print(f"[Pipeline] lobby_cleanup: on sub-screen, going back.")
            act = self._try_go_back(screenshot_path, "Pipeline(lobby_cleanup)")
            if act is not None:
                return act

        # Not in lobby — maybe still loading or on wrong screen
        return self._wait(500, "Pipeline(lobby): waiting for lobby.")

    def _handle_cafe_enter(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Click cafe button from lobby, wait for cafe to load.
        
        NOTE: Do NOT use _is_subscreen() here — the cafe IS a subscreen
        (has Home button + gear at top-right). Using _is_subscreen() would
        always send us back to lobby, creating an infinite loop.
        """
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(cafe): no screenshot.")

        # Check if supervision confirmed cafe (more reliable than template)
        if self._cafe_confirmed:
            print("[Pipeline] Cafe confirmed by supervision.")
            self._cafe_actually_entered = True
            self._advance_phase()  # → CAFE_EARNINGS
            return self._wait(200, "Pipeline(cafe): supervision confirmed cafe.")

        # Check lobby FIRST — _is_cafe_interior() can false-positive on lobby
        # (Emoticon_Action.png matches yellow elements like event banners).
        is_lobby = self._is_lobby(screenshot_path)

        # Only trust cafe interior detection if NOT on lobby
        if not is_lobby and self._is_cafe_interior(screenshot_path):
            print("[Pipeline] Cafe interior detected by template.")
            self._cafe_confirmed = True
            self._cafe_actually_entered = True
            # Close any popup inside cafe (e.g. 說明/訪問學生目錄 dialog)
            m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", min_score=0.80)
            if m is not None:
                return self._click(m.center[0], m.center[1],
                    f"Pipeline(cafe): close cafe popup. template={m.template} score={m.score:.3f}")
            self._advance_phase()  # → CAFE_EARNINGS
            return self._wait(200, "Pipeline(cafe): already in cafe.")

        # In lobby → close popups if any, then click cafe button
        if is_lobby:
            # Close announcement X buttons first (top-right)
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
                    f"Pipeline(cafe): close popup first. template={best.template} score={best.score:.3f}")
            # Close menu X button (center area)
            menu_close_roi = (int(sw * 0.25), int(sh * 0.05), int(sw * 0.75), int(sh * 0.20))
            m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", roi=menu_close_roi, min_score=0.65)
            if m is not None:
                return self._click(m.center[0], m.center[1],
                    f"Pipeline(cafe): close menu first. template={m.template} score={m.score:.3f}")
            # No popups — click cafe button
            roi_nav = (0, int(sh * 0.80), sw, sh)
            m = self._match(screenshot_path, "咖啡厅.png", roi=roi_nav, min_score=0.35)
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
        m = self._match(screenshot_path, "咖啡厅收益按钮.png", min_score=0.40)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(cafe_earnings): click earnings button. score={m.score:.3f}")

        # NOTE: Removed blind click at (sw*0.08, sh*0.18) — it hits 公告 button
        # on lobby if cafe detection was a false positive. Only use template matching.

        # After a couple ticks, assume earnings done or not available
        self._state.earnings_claimed = True
        self._advance_phase()  # → CAFE_INVITE
        return self._wait(200, "Pipeline(cafe_earnings): no earnings dialog, advancing.")

    def _handle_cafe_invite(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Invite featured (精選) students via MomoTalk.

        Cafe 1 (CAFE_INVITE): invite 1st featured student.
        Cafe 2 (CAFE_2_INVITE): invite 2nd featured student.

        Sub-state machine:
          "" (init)           → click invite ticket button in cafe
          "momotalk_open"     → MomoTalk detected; check sort
          "sort_opening"      → clicked sort dropdown, waiting for sort dialog
          "sort_selecting"    → sort dialog open, click 精選 option
          "sort_confirming"   → 精選 selected, click 確認
          "check_direction"   → verify sort direction is descending (下排序)
          "picking"           → find Nth featured student → click 邀請
          "confirming"        → clicked invite, waiting for confirm dialog
          "done"              → advance to next phase
        """
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(cafe_invite): no screenshot.")
        ss = self._state.sub_state
        cafe_num = 2 if self._phase == Phase.CAFE_2_INVITE else 1

        # Global timeout
        if self._state.ticks >= 30:
            self._advance_phase()
            return self._wait(300, "Pipeline(cafe_invite): timeout, advancing.")

        # ── DONE state ──
        if ss == "done":
            self._advance_phase()
            return self._wait(300, f"Pipeline(cafe_invite): invite done (cafe {cafe_num}), advancing.")

        # ── PRIORITY 1: Confirm dialogs ──
        confirm_roi = (int(sw * 0.25), int(sh * 0.35), int(sw * 0.75), int(sh * 0.95))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.40)
        if m is not None:
            if ss == "confirming":
                self._state.sub_state = "done"
            elif ss == "sort_confirming":
                self._state.sub_state = "check_direction"
            # else: just dismiss (bag full, etc.) — keep current sub_state
            return self._click(m.center[0], m.center[1],
                f"Pipeline(cafe_invite): confirm. sub={ss} score={m.score:.3f}")

        # ── Close generic popups (NOT when in MomoTalk) ──
        momotalk_states = ("momotalk_open", "sort_opening", "sort_selecting",
                           "sort_confirming", "check_direction", "picking")
        if ss not in momotalk_states:
            m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", min_score=0.80)
            if m is not None:
                return self._click(m.center[0], m.center[1],
                    f"Pipeline(cafe_invite): close popup. score={m.score:.3f}")

        # ── PRIORITY 2: Sort dialog (排列) ──
        # Templates are whole-dialog screenshots (~835x440). Each matches best
        # when its option is selected (pink). Low-scoring matches have shifted
        # bboxes (~100px off), so we MUST use the highest-scoring template.
        # Dialog layout (relative to bbox): 精選=(0.75, 0.58), 確認=(0.50, 0.88)
        sort_dialog_roi = (int(sw * 0.20), int(sh * 0.10), int(sw * 0.80), int(sh * 0.80))
        sort_templates = [
            "咖啡厅momotalk排序选项_精选.png",
            "咖啡厅momotalk排序选项_名字.png",
            "咖啡厅momotalk排序选项_学院.png",
            "咖啡厅momotalk排序选项_羁绊等级.png",
        ]
        best_sort = None
        for tmpl in sort_templates:
            m_sort = self._match(screenshot_path, tmpl, roi=sort_dialog_roi, min_score=0.55)
            if m_sort is not None and (best_sort is None or m_sort.score > best_sort.score):
                best_sort = m_sort
        if best_sort is not None:
            bx1, by1, bx2, by2 = best_sort.bbox
            dw, dh = bx2 - bx1, by2 - by1
            is_featured = best_sort.template == "咖啡厅momotalk排序选项_精选.png"
            if is_featured or ss == "sort_confirming":
                # 精選 already selected OR we just clicked it → click 確認
                confirm_x = bx1 + int(dw * 0.50)
                confirm_y = by1 + int(dh * 0.88)
                self._state.sub_state = "sort_confirming"
                return self._click(confirm_x, confirm_y,
                    f"Pipeline(cafe_invite): click 確認. best={best_sort.template} score={best_sort.score:.3f}")
            else:
                # 精選 NOT selected → click 精選 (bottom-right of 2x2 grid)
                feat_x = bx1 + int(dw * 0.75)
                feat_y = by1 + int(dh * 0.58)
                self._state.sub_state = "sort_confirming"
                return self._click(feat_x, feat_y,
                    f"Pipeline(cafe_invite): click 精選 option. best={best_sort.template} score={best_sort.score:.3f}")

        # ── PRIORITY 3: MomoTalk list (邀請 button visible) ──
        invite_btn_roi = (int(sw * 0.35), int(sh * 0.10), int(sw * 0.75), int(sh * 0.85))
        m_invite_btn = self._match(screenshot_path, "邀请.png",
            roi=invite_btn_roi, min_score=0.55)
        if m_invite_btn is not None:
            # MomoTalk is open
            sort_indicator_roi = (int(sw * 0.25), int(sh * 0.05), int(sw * 0.80), int(sh * 0.25))

            # ── Step A: Open sort dialog (unless already past sort setup) ──
            # Don't try to read the sort dropdown text (精选.png false-positives
            # at 0.612 on "名字" sort). Always open the sort dialog to verify.
            if ss in ("", "momotalk_open"):
                # Find sort direction icon to click the sort TEXT to its left
                m_asc = self._match(screenshot_path, "上排序.png",
                    roi=sort_indicator_roi, min_score=0.55)
                m_desc = self._match(screenshot_path, "下排列.png",
                    roi=sort_indicator_roi, min_score=0.55)
                sort_icon = m_asc or m_desc
                if sort_icon is not None:
                    click_x = sort_icon.center[0] - int(sw * 0.06)
                    click_y = sort_icon.center[1]
                    self._state.sub_state = "sort_opening"
                    return self._click(click_x, click_y,
                        f"Pipeline(cafe_invite): click sort dropdown (left of icon). cafe={cafe_num}")
                # Try specific sort text template
                m_bond = self._match(screenshot_path, "momotalk羁绊等级.png",
                    roi=sort_indicator_roi, min_score=0.60)
                if m_bond is not None:
                    self._state.sub_state = "sort_opening"
                    return self._click(m_bond.center[0], m_bond.center[1],
                        f"Pipeline(cafe_invite): click sort text. score={m_bond.score:.3f}")
                # Can't locate sort dropdown → proceed to picking
                self._state.sub_state = "picking"

            # ── Step B: Check sort direction (click 上排序 ONCE to toggle) ──
            if ss == "check_direction":
                m_asc = self._match(screenshot_path, "上排序.png",
                    roi=sort_indicator_roi, min_score=0.55)
                if m_asc is not None:
                    # Ascending → click once to toggle to descending, then pick
                    self._state.sub_state = "picking"
                    return self._click(m_asc.center[0], m_asc.center[1],
                        f"Pipeline(cafe_invite): toggle ascending→descending (once). score={m_asc.score:.3f}")
                # Already descending → proceed to picking
                self._state.sub_state = "picking"

            # ── Step C: Pick featured student ──
            if ss == "picking":
                badge_roi = (int(sw * 0.05), int(sh * 0.10), int(sw * 0.55), int(sh * 0.90))
                badges = self._find_all_matches(screenshot_path, "精选标志.png",
                    roi=badge_roi, min_score=0.50, nms_dist=50)
                badges.sort(key=lambda b: b.center[1])

                target_idx = cafe_num - 1  # 0 for cafe 1, 1 for cafe 2
                if target_idx < len(badges):
                    badge = badges[target_idx]
                    badge_cy = badge.center[1]
                    all_invites = self._find_all_matches(screenshot_path, "邀请.png",
                        roi=invite_btn_roi, min_score=0.50)
                    best_invite = None
                    best_dist = 9999
                    for inv in all_invites:
                        dist = abs(inv.center[1] - badge_cy)
                        if dist < best_dist:
                            best_dist = dist
                            best_invite = inv
                    if best_invite is not None and best_dist < sh * 0.10:
                        self._state.sub_state = "confirming"
                        return self._click(best_invite.center[0], best_invite.center[1],
                            f"Pipeline(cafe_invite): invite featured #{cafe_num}. badge_y={badge_cy} btn_y={best_invite.center[1]}")

                # Not enough featured students or no matching invite button
                if self._state.ticks >= 12:
                    self._state.sub_state = "confirming"
                    return self._click(m_invite_btn.center[0], m_invite_btn.center[1],
                        f"Pipeline(cafe_invite): fallback invite first visible (cafe {cafe_num}).")
                return self._wait(400, f"Pipeline(cafe_invite): looking for featured #{cafe_num}.")

            # Waiting for sort dialog or other transition
            return self._wait(400, f"Pipeline(cafe_invite): MomoTalk open, sub={ss}.")

        # ── PRIORITY 4: Cafe interior — look for invite ticket ──
        invite_roi = (int(sw * 0.50), int(sh * 0.75), sw, sh)
        m = self._match(screenshot_path, "邀请卷（带黄点）.png", roi=invite_roi, min_score=0.55)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(cafe_invite): click invite ticket. score={m.score:.3f}")

        # Positional fallback: invite ticket at fixed position (no yellow dot)
        # Position from previous successful match: ~(sw*0.70, sh*0.91)
        if ss == "" and self._state.ticks in (3, 4):
            tx, ty = int(sw * 0.70), int(sh * 0.91)
            return self._click(tx, ty,
                f"Pipeline(cafe_invite): positional click invite ticket area ({tx},{ty}).")

        # No invite ticket found
        if self._state.ticks >= 6:
            self._advance_phase()
            return self._wait(300, "Pipeline(cafe_invite): no invite ticket, advancing.")

        return self._wait(500, "Pipeline(cafe_invite): waiting for invite UI.")

    def _handle_cafe_headpat(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Tap all students with Emoticon_Action (yellow speech bubble) markers.

        Uses HSV color thresholding + morphological contour extraction:
        HSV→filter UI yellow→close/open morphology→find contours→filter by
        area (1500-15000) and aspect ratio (1.5-5.0 horizontal pill shape).
        <2ms on CPU, immune to scaling/animation/background changes.
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

        # NOTE: Do NOT use _is_subscreen() — cafe has Home/gear buttons.
        # Close any unexpected popup via X button instead.
        m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", min_score=0.80)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(headpat): close popup. template={m.template} score={m.score:.3f}")

        # Look for confirm button (interaction dialog)
        confirm_roi = (int(sw * 0.25), int(sh * 0.40), int(sw * 0.75), int(sh * 0.90))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.40)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(headpat): confirm dialog. score={m.score:.3f}")

        # ── Detect emoticons via HSV color thresholding + contour extraction ──
        # Pure color-based approach: converts to HSV, isolates the specific "UI
        # yellow" of Emoticon_Action bubbles, applies morphological cleanup, then
        # filters contours by area and aspect ratio (horizontal pill shape).
        # Runs <2ms on CPU. Immune to scaling, animation, and background.
        emoticons: list = []
        if cv2 is not None and np is not None:
            img = cv2.imread(screenshot_path)
            if img is not None:
                # ROI: cafe play area (exclude top nav and bottom UI bar)
                ry0, ry1 = int(sh * 0.08), int(sh * 0.72)
                roi_bgr = img[ry0:ry1, 0:sw]
                roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
                # Threshold for Emoticon_Action yellow (H≈19, S≈192, V≈236)
                yellow_mask = cv2.inRange(roi_hsv, (15, 130, 180), (30, 255, 255))
                # Morphological close (fill outline gaps) then open (remove noise)
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kern)
                yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kern)
                contours, _ = cv2.findContours(
                    yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 2500 or area > 15000:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect = w / max(h, 1)
                    if aspect < 1.5 or aspect > 5.0:
                        continue
                    # Convert back to absolute coordinates
                    cx = x + w // 2
                    cy = ry0 + y + h // 2
                    emoticons.append((cx, cy, area, aspect))

        if not emoticons:
            if self._state.ticks >= 3:
                self._advance_phase()
                return self._wait(200, "Pipeline(headpat): no emoticons found, advancing.")
            return self._wait(500, "Pipeline(headpat): waiting for emoticons to appear.")

        # Click each emoticon (character body is below the bubble).
        for (ex, ey, area, aspect) in emoticons:
            tx = ex
            ty = ey + max(self.cfg.headpat_offset_y, int(sh * 0.05))
            already = False
            for px, py in self._state.headpat_done:
                if abs(tx - px) + abs(ty - py) < 80:
                    already = True
                    break
            if not already:
                self._state.headpat_done.append((tx, ty))
                return self._click(tx, ty,
                    f"Pipeline(headpat): click below emoticon at ({ex},{ey}). area={area} ar={aspect:.1f}")

        self._advance_phase()
        return self._wait(200, "Pipeline(headpat): all emoticons done, advancing.")

    def _handle_cafe_switch(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Switch from cafe 1F to cafe 2F by clicking 移动至2号店."""
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(cafe_switch): no screenshot.")

        # Close any popup first (Rank Up, etc.)
        m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", min_score=0.80)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(cafe_switch): close popup. score={m.score:.3f}")

        # Confirm dialogs
        confirm_roi = (int(sw * 0.25), int(sh * 0.40), int(sw * 0.75), int(sh * 0.90))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.40)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(cafe_switch): confirm. score={m.score:.3f}")

        # Click 移动至2号店 button (top-left area of cafe, no ROI — template is unique)
        m = self._match(screenshot_path, "移动至2号店.png", min_score=0.50)
        if m is not None:
            if self._state.ticks >= 8:
                # Clicked 8+ times but still seeing button → click isn't working
                print("[Pipeline] cafe_switch: clicked 8 times but button persists, skipping.")
                self._advance_phase()
                return self._wait(300, "Pipeline(cafe_switch): click not working, skipping.")
            return self._click(m.center[0], m.center[1],
                f"Pipeline(cafe_switch): click switch to cafe 2. score={m.score:.3f}")

        # Button disappeared → we either switched successfully or it's not available
        if self._state.ticks >= 2:
            # Give it a couple ticks to detect the new cafe
            self._advance_phase()
            return self._wait(500, "Pipeline(cafe_switch): button gone, advancing.")

        # If button not found on first tick, wait a bit
        if self._state.ticks >= 5:
            print("[Pipeline] cafe_switch: button not found, skipping.")
            self._advance_phase()
            return self._wait(300, "Pipeline(cafe_switch): timeout, skipping.")

        return self._wait(500, "Pipeline(cafe_switch): waiting for switch button.")

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
    # Schedule handlers
    # -----------------------------------------------------------------------

    def _handle_schedule_enter(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Navigate from lobby to schedule (课程表)."""
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(schedule_enter): no screenshot.")

        # PRIORITY 1: Close announcement popups (公告) — these block everything
        # Check wide area since announcement X can be at various positions
        close_roi = (int(sw * 0.30), 0, sw, int(sh * 0.20))
        best = None
        best_score = 0.0
        for tmpl, min_s in [("内嵌公告的叉.png", 0.70), ("公告叉叉.png", 0.55),
                            ("游戏内很多页面窗口的叉.png", 0.70)]:
            m = self._match(screenshot_path, tmpl, roi=close_roi, min_score=min_s)
            if m is not None and m.score > best_score:
                best = m
                best_score = m.score
        if best is not None:
            return self._click(best.center[0], best.center[1],
                f"Pipeline(schedule_enter): close popup. template={best.template} score={best.score:.3f}")

        # Also check center-area X button (menu popups)
        menu_close_roi = (int(sw * 0.25), int(sh * 0.05), int(sw * 0.75), int(sh * 0.20))
        m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", roi=menu_close_roi, min_score=0.65)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(schedule_enter): close menu popup. score={m.score:.3f}")

        # Confirm dialogs
        confirm_roi = (int(sw * 0.25), int(sh * 0.40), int(sw * 0.75), int(sh * 0.90))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.40)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(schedule_enter): confirm. score={m.score:.3f}")

        # Already inside schedule? Check for back button + no lobby nav
        # (schedule is a sub-screen with Home button)
        if not self._is_lobby(screenshot_path) and self._is_subscreen(screenshot_path):
            # Likely inside schedule already
            if self._state.ticks >= 2:
                self._advance_phase()  # → SCHEDULE_EXECUTE
                return self._wait(300, "Pipeline(schedule_enter): inside sub-screen, advancing.")

        # In lobby → click schedule button
        if self._is_lobby(screenshot_path):
            roi_nav = (0, int(sh * 0.80), sw, sh)
            m = self._match(screenshot_path, "课程表.png", roi=roi_nav, min_score=0.40)
            if m is not None:
                return self._click(m.center[0], m.center[1],
                    f"Pipeline(schedule_enter): click schedule. score={m.score:.3f}")
            # Timeout: if in lobby but schedule not found after many ticks, skip
            if self._state.ticks >= 10:
                print("[Pipeline] schedule_enter: schedule not found after 10 ticks, skipping.")
                self._advance_phase()  # → SCHEDULE_EXECUTE (will detect lobby → DONE)
                return self._wait(300, "Pipeline(schedule_enter): schedule button not found, skipping.")
            return self._wait(400, "Pipeline(schedule_enter): schedule button not found.")

        # Not in lobby, not in subscreen — might be transitioning
        if self._state.ticks >= 15:
            print("[Pipeline] schedule_enter: timeout, skipping.")
            self._advance_phase()
            return self._wait(300, "Pipeline(schedule_enter): timeout, skipping.")
        return self._wait(500, "Pipeline(schedule_enter): waiting for lobby.")

    def _handle_schedule_execute(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Execute schedule: click rooms, confirm, handle ticket exhaustion.

        Strategy: Use template matching for confirm buttons and popups.
        For room selection, click the leftmost/first available room area.
        VLM fallback handles complex decisions (return None to defer).
        """
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(schedule_exec): no screenshot.")

        # If we ended up in lobby, schedule is done
        if self._is_lobby(screenshot_path):
            print("[Pipeline] schedule_execute: back in lobby, schedule done.")
            self._advance_phase()  # → DONE
            return self._wait(300, "Pipeline(schedule_exec): back in lobby, done.")

        # Close any popup (ticket exhausted, reward, etc.)
        m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", min_score=0.75)
        if m is not None:
            self._state.sub_state = "popup_closed"
            return self._click(m.center[0], m.center[1],
                f"Pipeline(schedule_exec): close popup. score={m.score:.3f}")

        # After closing a popup, if we see lobby nav, we're done
        if self._state.sub_state == "popup_closed" and self._is_lobby(screenshot_path):
            self._advance_phase()
            return self._wait(300, "Pipeline(schedule_exec): done after popup close.")

        # Confirm button (start schedule, result confirm, etc.)
        confirm_roi = (int(sw * 0.25), int(sh * 0.40), int(sw * 0.85), int(sh * 0.95))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.40)
        if m is not None:
            self._state.sub_state = "confirmed"
            return self._click(m.center[0], m.center[1],
                f"Pipeline(schedule_exec): confirm. score={m.score:.3f}")

        # Tap center to skip animations (schedule result, etc.)
        if self._state.sub_state == "confirmed" and self._state.ticks % 2 == 0:
            return self._click(sw // 2, sh // 2,
                "Pipeline(schedule_exec): tap to skip animation.")

        # Defer to VLM for room selection and complex decisions
        # Return None so VLM takes over
        if self._state.ticks >= 20:
            # Timeout — go back to lobby
            print("[Pipeline] schedule_execute: timeout, going back.")
            act = self._try_go_back(screenshot_path, "Pipeline(schedule_exec)")
            if act is not None:
                return act
            self._advance_phase()
            return self._wait(300, "Pipeline(schedule_exec): timeout.")

        # Let VLM handle room selection for the first several ticks
        return None

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
