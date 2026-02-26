"""
OpenCV-driven deterministic pipeline for Blue Archive daily routine.

Architecture:
  - PipelineController: state machine that drives routine steps sequentially
  - PipelineStep (base): each step detects state, emits actions, checks completion
  - VLM is called for headpat emoticon detection (semantic understanding);
    all other phases use Cerebellum template matching + OpenCV.
  - The parent agent (VlmPolicyAgent) calls pipeline.tick() each loop iteration;
    if the pipeline returns an action, it is used directly.
    If the pipeline returns None, VLM fallback kicks in.
"""

from __future__ import annotations

import json
import math
import re
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

try:
    from vision.yolo_detector import YoloDetector, get_yolo_detector
except ImportError:
    YoloDetector = None  # type: ignore
    get_yolo_detector = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _center(bbox) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) // 2), int((y1 + y2) // 2)


def _detect_yellow_markers(
    screenshot_path: str,
    *,
    min_area: int = 150,
    max_area: int = 12000,
    safe_roi: Optional[Tuple[int, int, int, int]] = None,
) -> List[Tuple[int, int, int, int]]:
    """Detect yellow exclamation-mark interaction markers in cafe.

    Args:
        safe_roi: (x1, y1, x2, y2) absolute pixel region to restrict detection
                  to. Only contours whose center falls inside this box are kept.
                  This is the "safe zone" that avoids all UI chrome.
    """
    if cv2 is None or np is None:
        return []
    img = cv2.imread(screenshot_path)
    if img is None:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Yellow range in HSV — tightened to avoid furniture/decoration false positives.
    # Real headpat bubbles are bright saturated yellow; furniture is duller.
    lo = np.array([18, 180, 200], dtype=np.uint8)
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
        
        # Filter out tiny noise blobs that passed the area check (e.g. 10x20 pixels)
        if w < 20 or h < 20:
            continue
            
        aspect = w / max(h, 1)
        if aspect > 3.0 or aspect < 0.2:
            continue
        cx, cy = x + w // 2, y + h // 2
        if safe_roi is not None:
            rx1, ry1, rx2, ry2 = safe_roi
            if cx < rx1 or cx > rx2 or cy < ry1 or cy > ry2:
                continue
        # Fill-ratio check: real bubbles are solid yellow blobs.
        # Furniture has yellow mixed with other colors → low fill ratio.
        roi_mask = mask[y:y+h, x:x+w]
        fill_ratio = float(cv2.countNonZero(roi_mask)) / max(1, w * h)
        if fill_ratio < 0.35:
            continue
        out.append((x, y, x + w, y + h))
    out.sort(key=lambda b: (b[1], b[0]))
    return out[:12]


# Safe ROI margins — avoids top resource bar, bottom menu, side buttons.
# Values are fractions of screen width/height.
SAFE_ROI_LEFT   = 0.08   # skip left sidebar (指定訪問, 隨機訪問)
SAFE_ROI_RIGHT  = 0.92   # skip right edge (gear, home buttons)
SAFE_ROI_TOP    = 0.15   # skip top resource bar + cafe header + timer icon
SAFE_ROI_BOTTOM = 0.78   # skip bottom menu (編輯模式, 禮物, etc.)


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
    CLUB_ENTER = auto()        # navigate to club from lobby
    CLUB_CLAIM = auto()        # claim AP reward inside club
    CLUB = auto()              # legacy alias
    CAMPAIGN_ENTER = auto()    # lobby → click 業務區
    CAMPAIGN_BOUNTIES = auto() # 悬赏通缉 sweep loop
    CAMPAIGN_SCRIMMAGES = auto()  # 学院交流会 sweep loop
    CAMPAIGN_PVP = auto()      # 战术对抗赛 claim + optional fight
    CAMPAIGN_EXIT = auto()     # back to lobby from campaign
    BOUNTIES = auto()          # legacy alias
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
    invite_skip: int = 0  # students skipped due to 隔壁 warning
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

    def __init__(self, cerebellum: Optional[Any] = None, cfg: Optional[PipelineConfig] = None,
                 vlm_engine: Optional[Any] = None):
        self.cfg = cfg or PipelineConfig()
        self._cerebellum = cerebellum
        self._vlm_engine = vlm_engine
        # YOLO detector (primary headpat detector, falls back to VLM if not trained)
        self._yolo: Optional[Any] = None
        try:
            if get_yolo_detector is not None:
                self._yolo = get_yolo_detector(skill_name="cafe")
                if self._yolo is not None:
                    print("[Pipeline] YOLO detector loaded for cafe.")
                else:
                    print("[Pipeline] YOLO model not found (not trained yet), using VLM fallback.")
        except Exception as e:
            print(f"[Pipeline] YOLO init failed: {e}")
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
            # CAFE_2_EARNINGS skipped: earnings are shared between cafe 1 and 2
            Phase.CAFE_2_INVITE,
            Phase.CAFE_2_HEADPAT,
            Phase.CAFE_EXIT,
            Phase.SCHEDULE_ENTER,
            Phase.SCHEDULE_EXECUTE,
            Phase.CLUB_ENTER,
            Phase.CLUB_CLAIM,
            Phase.CAMPAIGN_ENTER,
            Phase.CAMPAIGN_BOUNTIES,
            Phase.CAMPAIGN_SCRIMMAGES,
            Phase.CAMPAIGN_PVP,
            Phase.CAMPAIGN_EXIT,
            # Future phases:
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
        # Headpat phases need more ticks for pan-and-scan (5 viewports × ~5 ticks each)
        phase_tick_limit = self.cfg.max_ticks_per_phase
        if self._phase in (Phase.CAFE_HEADPAT, Phase.CAFE_2_HEADPAT):
            phase_tick_limit = 60
        if self._state.ticks > phase_tick_limit:
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
            Phase.CLUB_ENTER: self._handle_club_enter,
            Phase.CLUB_CLAIM: self._handle_club_claim,
            Phase.CAMPAIGN_ENTER: self._handle_campaign_enter,
            Phase.CAMPAIGN_BOUNTIES: self._handle_campaign_bounties,
            Phase.CAMPAIGN_SCRIMMAGES: self._handle_campaign_scrimmages,
            Phase.CAMPAIGN_PVP: self._handle_campaign_pvp,
            Phase.CAMPAIGN_EXIT: self._handle_campaign_exit,
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
        """Close startup popups/notices until we reach the lobby.

        Also detects if we're already in a known screen (cafe, lobby, subscreen)
        and fast-forwards to the correct phase instead of blindly tapping.
        """
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(500, "Pipeline(startup): waiting for screenshot.")

        # Fast-forward: already in cafe → skip to CAFE_EARNINGS
        if self._is_cafe_interior(screenshot_path):
            print("[Pipeline] Cafe detected during STARTUP, jumping to CAFE_EARNINGS.")
            self._cafe_confirmed = True
            self._cafe_actually_entered = True
            self._enter_phase(Phase.CAFE_EARNINGS)
            return self._wait(200, "Pipeline(startup): cafe detected, jumping to CAFE_EARNINGS.")

        # Fast-forward: already in lobby
        if self._is_lobby(screenshot_path):
            print("[Pipeline] Lobby detected, startup complete.")
            self._advance_phase()
            return self._wait(300, "Pipeline(startup): lobby detected, advancing.")

        # Fast-forward: already in schedule
        m_all = self._match(screenshot_path, "全体课程表.png", min_score=0.30)
        m_tickets = self._match(screenshot_path, "课程表票持有数量.png", min_score=0.30)
        if m_all or m_tickets:
            print("[Pipeline] Schedule UI detected during STARTUP, jumping to SCHEDULE_EXECUTE.")
            self._enter_phase(Phase.SCHEDULE_EXECUTE)
            return self._wait(200, "Pipeline(startup): schedule detected, jumping to SCHEDULE_EXECUTE.")

        # Fast-forward: on a subscreen (schedule, mission, etc.) → go Home
        if self._is_subscreen(screenshot_path):
            print("[Pipeline] Subscreen detected during STARTUP, navigating Home.")
            act = self._try_go_back(screenshot_path, "Pipeline(startup)")
            if act is not None:
                return act

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

        # Try tap-to-start template — require 0.50+ to avoid false positives
        m = self._match(screenshot_path, "点击开始.png", min_score=0.50)
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

        # PRIORITY 0: Already in cafe → skip lobby entirely, jump to CAFE_EARNINGS
        if self._is_cafe_interior(screenshot_path):
            print("[Pipeline] Cafe detected during LOBBY_CLEANUP, jumping to CAFE_EARNINGS.")
            self._cafe_confirmed = True
            self._cafe_actually_entered = True
            self._enter_phase(Phase.CAFE_EARNINGS)
            return self._wait(200, "Pipeline(lobby): cafe detected, jumping to CAFE_EARNINGS.")

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

        # Check if earnings are 0% — no point clicking
        earnings_roi = (int(sw * 0.75), int(sh * 0.80), sw, sh)
        m_zero = self._match(screenshot_path, "咖啡厅收益为0.png", roi=earnings_roi, min_score=0.85)
        if m_zero is not None:
            self._state.earnings_claimed = True
            self._advance_phase()
            return self._wait(200, f"Pipeline(cafe_earnings): earnings 0%, skipping. score={m_zero.score:.3f}")

        # Try cafe earnings button template
        # The button is usually in the bottom-right corner
        earnings_roi = (int(sw * 0.70), int(sh * 0.75), sw, sh)
        m = self._match(screenshot_path, "咖啡厅收益按钮.png", roi=earnings_roi, min_score=0.70)
        if m is not None:
            # We found the button. Click it.
            return self._click(m.center[0], m.center[1],
                f"Pipeline(cafe_earnings): click earnings button. score={m.score:.3f}")

        # If we have waited a few ticks and still don't see the earnings button or confirm,
        # assume it's done or not available.
        if self._state.ticks >= 5:
            self._state.earnings_claimed = True
            self._advance_phase()  # → CAFE_INVITE
            return self._wait(200, "Pipeline(cafe_earnings): no earnings dialog, advancing.")
        
        return self._wait(400, "Pipeline(cafe_earnings): waiting for earnings button.")

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

        # ── CLOSING MOMOTALK state ──
        # After invite confirmed, MomoTalk may still be open (showing animation
        # or student list). Close it before advancing to headpat.
        if ss == "closing_momotalk":
            # Try template X button first
            close_roi = (int(sw * 0.30), int(sh * 0.01), int(sw * 0.80), int(sh * 0.20))
            m_x = self._match(screenshot_path, "游戏内很多页面窗口的叉.png",
                roi=close_roi, min_score=0.50)
            if m_x is not None:
                self._state.sub_state = "done"
                return self._click(m_x.center[0], m_x.center[1],
                    f"Pipeline(cafe_invite): close MomoTalk X. score={m_x.score:.3f}")
            # Check if MomoTalk is still visible (邀請 button in student list)
            momo_roi = (int(sw * 0.35), int(sh * 0.10), int(sw * 0.75), int(sh * 0.75))
            m_inv = self._match(screenshot_path, "邀请.png", roi=momo_roi, min_score=0.55)
            if m_inv is not None:
                # Blind-click MomoTalk X at known position
                x_btn_x = int(sw * 0.66)
                x_btn_y = int(sh * 0.075)
                return self._click(x_btn_x, x_btn_y,
                    f"Pipeline(cafe_invite): blind-click MomoTalk X at ({x_btn_x},{x_btn_y}).")
            # MomoTalk seems closed → advance
            self._state.sub_state = "done"
            return self._wait(300, f"Pipeline(cafe_invite): MomoTalk closed, advancing.")

        # ── DONE state ──
        if ss == "done":
            self._advance_phase()
            return self._wait(300, f"Pipeline(cafe_invite): invite done (cafe {cafe_num}), advancing.")

        # ── PRIORITY 1: Confirm dialogs ──
        confirm_roi = (int(sw * 0.25), int(sh * 0.35), int(sw * 0.75), int(sh * 0.95))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.50)
        if m is not None:
            if ss == "confirming":
                # 隔壁咖啡廳 warning or normal confirm → always click 確認 to proceed
                self._state.sub_state = "closing_momotalk"
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

            # Sort dialog just closed → transition to direction check
            if ss == "sort_confirming":
                self._state.sub_state = "check_direction"
                ss = "check_direction"

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

            # ── Step B: Check sort direction (compare 上排序 vs 下排列 scores) ──
            if ss == "check_direction":
                m_asc = self._match(screenshot_path, "上排序.png",
                    roi=sort_indicator_roi, min_score=0.55)
                m_desc = self._match(screenshot_path, "下排列.png",
                    roi=sort_indicator_roi, min_score=0.55)
                asc_score = m_asc.score if m_asc else 0
                desc_score = m_desc.score if m_desc else 0
                if m_asc is not None and asc_score > desc_score:
                    # Ascending → click once to toggle to descending, then pick
                    self._state.sub_state = "picking"
                    ss = "picking"
                    return self._click(m_asc.center[0], m_asc.center[1],
                        f"Pipeline(cafe_invite): toggle asc→desc. asc={asc_score:.3f} desc={desc_score:.3f}")
                # Already descending (or no icon) → proceed to picking
                self._state.sub_state = "picking"
                ss = "picking"

            # ── Step C: Pick featured student ──
            if ss == "picking":
                badge_roi = (int(sw * 0.05), int(sh * 0.10), int(sw * 0.55), int(sh * 0.90))
                badges = self._find_all_matches(screenshot_path, "精选标志.png",
                    roi=badge_roi, min_score=0.50, nms_dist=50)
                badges.sort(key=lambda b: b.center[1])

                target_idx = cafe_num - 1 + self._state.invite_skip
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

        m_active = self._match(screenshot_path, "邀请卷（带黄点）.png", roi=invite_roi, min_score=0.55)
        if m_active is not None:
            return self._click(m_active.center[0], m_active.center[1],
                f"Pipeline(cafe_invite): click invite ticket. score={m_active.score:.3f}")

        # Active template didn't match — check if ticket is used (greyed out)
        m_used = self._match(screenshot_path, "邀请卷已用.png", roi=invite_roi, min_score=0.60)
        if m_used is not None:
            self._advance_phase()
            return self._wait(200,
                f"Pipeline(cafe_invite): invite ticket already used, skipping. score={m_used.score:.3f}")

        # No invite ticket found
        if self._state.ticks >= 6:
            self._advance_phase()
            return self._wait(300, "Pipeline(cafe_invite): no invite ticket, advancing.")

        return self._wait(500, "Pipeline(cafe_invite): waiting for invite UI.")

    # -- VLM emoticon detection helper ------------------------------------------

    _VLM_HEADPAT_PROMPT_TEMPLATE = (
        "You are a game UI detector. Find all bright yellow interaction icons "
        "(emoticon bubbles, starburst markers, exclamation marks) floating above "
        "characters' heads in this cafe screenshot. These are yellow UI elements. "
        "Return JSON only. "
        'Return format: {{"items": [{{"label": "emoticon", "bbox": [x1,y1,x2,y2]}}]}}. '
        "bbox must be pixel coordinates in the original image. "
        "Image size: width={w}, height={h}. "
        "If no yellow icons found, return {{\"items\": []}}. "
        "Do not include any extra keys. Do not wrap in markdown."
    )

    def _vlm_detect_emoticons(self, screenshot_path: str, sw: int, sh: int
                              ) -> Optional[List[Tuple[int, int]]]:
        """Ask VLM to find yellow emoticon markers. Returns list of (cx, cy) or None on failure."""
        if self._vlm_engine is None:
            return None
        try:
            prompt = self._VLM_HEADPAT_PROMPT_TEMPLATE.format(w=sw, h=sh)
            res = self._vlm_engine.ocr(
                image_path=screenshot_path,
                prompt=prompt,
                max_new_tokens=512,
            )
            raw = str(res.get("raw") or "").strip()
            print(f"[Pipeline] VLM headpat raw: {raw[:300]}")
            if not raw:
                return []
            # Clean markdown wrapping
            cleaned = raw.strip().strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            # Try to parse as JSON object with items+bbox (primary format)
            result: List[Tuple[int, int]] = []
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    items = parsed.get("items", []) if isinstance(parsed, dict) else []
                    for item in items:
                        bbox = item.get("bbox") if isinstance(item, dict) else None
                        if bbox and len(bbox) >= 4:
                            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            if 0 <= cx <= sw and 0 <= cy <= sh:
                                result.append((cx, cy))
                            else:
                                print(f"[Pipeline] VLM headpat: bbox out of range: {bbox}")
                    if result:
                        return result
                except json.JSONDecodeError:
                    pass
            # Fallback: try parsing as array of [x, y] pairs
            arr_match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if arr_match:
                try:
                    coords = json.loads(arr_match.group(0))
                    if isinstance(coords, list):
                        for item in coords:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                nx, ny = float(item[0]), float(item[1])
                                # Handle both normalized (0-1) and pixel coordinates
                                if nx <= 1.0 and ny <= 1.0:
                                    result.append((int(nx * sw), int(ny * sh)))
                                elif 0 <= nx <= sw and 0 <= ny <= sh:
                                    result.append((int(nx), int(ny)))
                except json.JSONDecodeError:
                    pass
            if not result:
                print(f"[Pipeline] VLM headpat: no valid coords parsed from: {raw[:200]}")
            return result
        except Exception as e:
            print(f"[Pipeline] VLM headpat error: {type(e).__name__}: {e}")
            return None

    # -- Headpat handler -------------------------------------------------------

    # -- Pan-and-Scan viewport definitions ------------------------------------
    # 2-point horizontal: zoom out first, then one swipe right (covers left half)
    # and one swipe left (covers right half). The cafe is wide horizontally.
    _PAN_SCAN_VIEWPORTS = [
        ("left_half",  +0.40, 0.0),   # swipe right → expose left side of map
        ("right_half", -0.40, 0.0),   # swipe left  → expose right side of map
    ]
    _SWIPE_DURATION_MS = 600  # slow swipe to avoid map inertia fly-away
    _ZOOM_OUT_CLICKS = -6     # mouse wheel clicks to zoom out (negative = zoom out)

    def _handle_cafe_headpat(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Tap all students with yellow interaction markers using HSV pan-and-scan.

        Strategy:
        1. Zoom out to minimum (mouse wheel scroll) for maximum visibility.
        2. Define a "safe zone" ROI that avoids UI chrome.
        3. Use HSV yellow detection ONLY inside this safe zone.
        4. Pan the cafe map to 3 viewports (center + left half + right half)
           by swiping, scanning each for yellow bubbles.

        Sub-state machine:
          "zoom_out"      → scroll wheel to zoom out for max visibility
          "zoom_wait"     → wait for zoom animation to settle
          "scan_center"   → initial scan at default (zoomed-out) view
          "pan_N"         → swiping to viewport N (0=left_half, 1=right_half)
          "wait_N"        → waiting for swipe inertia to settle
          "scan_N"        → scanning viewport N for bubbles
          "reverse_N"     → swiping back to center after scanning
          "rwait_N"       → waiting for reverse swipe inertia
          "done"          → all viewports scanned, advance
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

        # Global timeout — prevent infinite loops
        if self._state.ticks >= 50:
            print("[Pipeline] headpat: global timeout, advancing.")
            self._advance_phase()
            return self._wait(200, "Pipeline(headpat): timeout, advancing.")

        # ── Close MomoTalk / popups that may still be open ──
        # MomoTalk can stay open after invite phase; its yellow level badges
        # trigger massive HSV false positives. Close it first.
        close_roi = (int(sw * 0.30), int(sh * 0.01), int(sw * 0.80), int(sh * 0.20))
        m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", roi=close_roi, min_score=0.70)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(headpat): close popup/MomoTalk. score={m.score:.3f}")

        # Also check for a wider close button area (羈絆升級 popups, etc.)
        wide_close_roi = (int(sw * 0.45), 0, sw, int(sh * 0.30))
        m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", roi=wide_close_roi, min_score=0.80)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(headpat): close wide popup. score={m.score:.3f}")

        confirm_roi = (int(sw * 0.25), int(sh * 0.40), int(sw * 0.75), int(sh * 0.90))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.50)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(headpat): confirm dialog. score={m.score:.3f}")

        # Detect MomoTalk still open (邀請 button or invitation animation).
        # NEVER press Escape here — it exits the cafe entirely!
        # Instead, blind-click the MomoTalk X at its known position.
        momo_invite_roi = (int(sw * 0.35), int(sh * 0.10), int(sw * 0.75), int(sh * 0.75))
        m_momo = self._match(screenshot_path, "邀请.png", roi=momo_invite_roi, min_score=0.55)
        if m_momo is not None:
            # MomoTalk X is always at top-right of dialog: ~66% width, ~7.5% height
            x_btn_x = int(sw * 0.66)
            x_btn_y = int(sh * 0.075)
            return self._click(x_btn_x, x_btn_y,
                f"Pipeline(headpat): blind-click MomoTalk X at ({x_btn_x},{x_btn_y}).")

        # Dismiss fullscreen popups (羈絆升級 Rank Up, etc.)
        if not self._is_cafe_interior(screenshot_path):
            return self._click(sw // 2, sh // 2,
                f"Pipeline(headpat): tap to dismiss fullscreen popup (tick {self._state.ticks}).")

        # ── Compute safe ROI ──
        safe_roi = (
            int(sw * SAFE_ROI_LEFT),
            int(sh * SAFE_ROI_TOP),
            int(sw * SAFE_ROI_RIGHT),
            int(sh * SAFE_ROI_BOTTOM),
        )

        ss = self._state.sub_state

        # Initialize sub_state
        if ss == "":
            self._state.sub_state = "zoom_out"
            ss = "zoom_out"

        # ── ZOOM OUT: scroll wheel to zoom out for max cafe visibility ──
        if ss == "zoom_out":
            self._state.sub_state = "zoom_wait"
            cx, cy = sw // 2, sh // 2
            print(f"[Pipeline] headpat: zooming out ({self._ZOOM_OUT_CLICKS} clicks) at ({cx},{cy})")
            return {
                "action": "scroll",
                "target": [cx, cy],
                "clicks": self._ZOOM_OUT_CLICKS,
                "reason": f"Pipeline(headpat): zoom out cafe view ({self._ZOOM_OUT_CLICKS} clicks).",
                "_pipeline": True,
            }

        if ss == "zoom_wait":
            self._state.sub_state = "scan_center"
            return self._wait(800, "Pipeline(headpat): waiting for zoom animation to settle.")

        # ── SCAN states: detect and click yellow bubbles ──
        if ss.startswith("scan_"):
            markers = _detect_yellow_markers(screenshot_path, safe_roi=safe_roi)
            if markers:
                # Click the first unvisited marker
                for (x1, y1, x2, y2) in markers:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    # Offset: click slightly below/right of bubble (character body)
                    tx = cx + max(self.cfg.headpat_offset_x, int(sw * 0.03))
                    ty = cy + max(self.cfg.headpat_offset_y, int(sh * 0.02))
                    already = False
                    for px, py in self._state.headpat_done:
                        if abs(tx - px) + abs(ty - py) < 80:
                            already = True
                            break
                    if not already:
                        self._state.headpat_done.append((tx, ty))
                        # After clicking, stay in same scan state to re-scan
                        # (new bubbles may appear after interaction)
                        print(f"[Pipeline] headpat HSV: click at ({tx},{ty}), viewport={ss}")
                        return self._click(tx, ty,
                            f"Pipeline(headpat): HSV click bubble at ({cx},{cy}), viewport={ss}.")
                # All markers in this viewport already visited — fall through to advance

            # No (new) markers in this viewport → advance to next
            if ss == "scan_center":
                if not self._PAN_SCAN_VIEWPORTS:
                    self._state.sub_state = "done"
                else:
                    self._state.sub_state = "pan_0"
            elif ss.startswith("scan_"):
                # scan_N → reverse swipe back to center, then next viewport
                try:
                    idx = int(ss.split("_")[1])
                except (ValueError, IndexError):
                    idx = len(self._PAN_SCAN_VIEWPORTS)
                self._state.sub_state = f"reverse_{idx}"
            return self._wait(300, f"Pipeline(headpat): no new bubbles in {ss}, moving on.")

        # ── PAN states: swipe to expose a corner ──
        if ss.startswith("pan_"):
            try:
                idx = int(ss.split("_")[1])
            except (ValueError, IndexError):
                self._state.sub_state = "done"
                return self._wait(200, "Pipeline(headpat): pan index error, done.")

            if idx >= len(self._PAN_SCAN_VIEWPORTS):
                self._state.sub_state = "done"
                return self._wait(200, "Pipeline(headpat): all viewports done.")

            name, dx_frac, dy_frac = self._PAN_SCAN_VIEWPORTS[idx]
            cx, cy = sw // 2, sh // 2
            to_x = cx + int(sw * dx_frac)
            to_y = cy + int(sh * dy_frac)
            self._state.sub_state = f"wait_{idx}"
            print(f"[Pipeline] headpat: panning to {name} ({cx},{cy})→({to_x},{to_y})")
            return {
                "action": "swipe",
                "from": [cx, cy],
                "to": [to_x, to_y],
                "duration_ms": self._SWIPE_DURATION_MS,
                "reason": f"Pipeline(headpat): swipe to expose {name}.",
                "_pipeline": True,
            }

        # ── WAIT states: let swipe inertia settle ──
        if ss.startswith("wait_"):
            try:
                idx = int(ss.split("_")[1])
            except (ValueError, IndexError):
                idx = 0
            self._state.sub_state = f"scan_{idx}"
            return self._wait(800, f"Pipeline(headpat): waiting for swipe inertia (viewport {idx}).")

        # ── REVERSE states: swipe back to center after scanning a corner ──
        # This prevents cumulative drift that would make later viewports wrong.
        if ss.startswith("reverse_"):
            try:
                idx = int(ss.split("_")[1])
            except (ValueError, IndexError):
                idx = 0
            if idx < len(self._PAN_SCAN_VIEWPORTS):
                name, dx_frac, dy_frac = self._PAN_SCAN_VIEWPORTS[idx]
                cx, cy = sw // 2, sh // 2
                # Reverse: swipe in the OPPOSITE direction
                to_x = cx - int(sw * dx_frac)
                to_y = cy - int(sh * dy_frac)
                self._state.sub_state = f"rwait_{idx}"
                print(f"[Pipeline] headpat: reversing from {name} ({cx},{cy})→({to_x},{to_y})")
                return {
                    "action": "swipe",
                    "from": [cx, cy],
                    "to": [to_x, to_y],
                    "duration_ms": self._SWIPE_DURATION_MS,
                    "reason": f"Pipeline(headpat): reverse swipe from {name} back to center.",
                    "_pipeline": True,
                }
            # Index out of range → just advance
            self._state.sub_state = "done"
            return self._wait(200, "Pipeline(headpat): reverse index error, done.")

        # ── RWAIT states: wait for reverse swipe inertia ──
        if ss.startswith("rwait_"):
            try:
                idx = int(ss.split("_")[1])
            except (ValueError, IndexError):
                idx = 0
            next_idx = idx + 1
            if next_idx < len(self._PAN_SCAN_VIEWPORTS):
                self._state.sub_state = f"pan_{next_idx}"
            else:
                self._state.sub_state = "done"
            return self._wait(600, f"Pipeline(headpat): reverse inertia settling (viewport {idx}).")

        # ── DONE ──
        if ss == "done":
            n = len(self._state.headpat_done)
            print(f"[Pipeline] headpat: pan-and-scan complete. Patted {n} students.")
            self._advance_phase()
            return self._wait(200, f"Pipeline(headpat): all viewports scanned ({n} patted), advancing.")

    def _handle_cafe_switch(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Switch from cafe 1F to cafe 2F by clicking 移动至2号店.

        IMPORTANT: The template 移动至2号店.png also matches 移動至1號店 at
        high score (0.96+) because the buttons look almost identical. So we
        must click ONLY ONCE, then wait for the scene to reload.

        Sub-states:
          "" (init)     → find and click 移動至2號店
          "switching"   → button clicked; wait for cafe to reload (handle
                          TAP TO START screen, loading screens, etc.)
        """
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(cafe_switch): no screenshot.")
        ss = self._state.sub_state

        # Global timeout
        if self._state.ticks >= 15:
            print("[Pipeline] cafe_switch: timeout, advancing.")
            self._advance_phase()
            return self._wait(300, "Pipeline(cafe_switch): timeout, advancing.")

        # ── Close popups in any state ──
        m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", min_score=0.80)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(cafe_switch): close popup. score={m.score:.3f}")
        confirm_roi = (int(sw * 0.25), int(sh * 0.40), int(sw * 0.75), int(sh * 0.90))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.50)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(cafe_switch): confirm. score={m.score:.3f}")

        # ── INIT: Click the switch button once ──
        if ss == "":
            m = self._match(screenshot_path, "移动至2号店.png", min_score=0.50)
            if m is not None:
                self._state.sub_state = "switching"
                return self._click(m.center[0], m.center[1],
                    f"Pipeline(cafe_switch): click switch to cafe 2. score={m.score:.3f}")
            # Button not visible yet — maybe a popup is blocking
            if self._state.ticks >= 5:
                self._advance_phase()
                return self._wait(300, "Pipeline(cafe_switch): button not found, advancing.")
            return self._wait(500, "Pipeline(cafe_switch): waiting for switch button.")

        # ── SWITCHING: Wait for cafe to reload after click ──
        # During transition the game may show TAP TO START or loading screen.
        # Once we see cafe interior again, we've arrived at cafe 2.
        if self._is_cafe_interior(screenshot_path):
            print("[Pipeline] cafe_switch: cafe interior detected after switch.")
            self._advance_phase()
            return self._wait(400, "Pipeline(cafe_switch): arrived at cafe 2, advancing.")

        # Handle TAP TO START screen during transition — tap to proceed
        m_tap = self._match(screenshot_path, "点击开始.png", min_score=0.40)
        if m_tap is not None:
            return self._click(sw // 2, int(sh * 0.82),
                f"Pipeline(cafe_switch): tap to start during transition. score={m_tap.score:.3f}")

        # Unknown screen during transition — just wait (do NOT tap center,
        # that was causing the infinite loop by re-clicking the switch button)
        return self._wait(600, f"Pipeline(cafe_switch): waiting for cafe to load (tick {self._state.ticks}).")

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
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.50)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(schedule_enter): confirm. score={m.score:.3f}")

        # Check for Schale Office entry screen (课程表夏莱办公室入口.png)
        # This sometimes appears after clicking Schedule from lobby before entering the actual rooms
        m_schale = self._match(screenshot_path, "课程表夏莱办公室入口.png", min_score=0.60)
        if m_schale is not None:
            return self._click(m_schale.center[0], m_schale.center[1],
                f"Pipeline(schedule_enter): enter Schale Office. score={m_schale.score:.3f}")

        # Fallback for Schale Office if template fails (e.g. resolution differences)
        # We can detect we are in the Schale Office entry screen if we are not in lobby, 
        # not inside the actual schedule rooms, and we just clicked Schedule from lobby.
        # But to be safe, we'll just click the coordinate if we timeout while waiting for schedule to load.
        if not self._is_lobby(screenshot_path):
            if self._state.ticks >= 8 and self._state.ticks % 3 == 0:
                # Click the center of the Schale Office entry button area (approx ~65% width, 35% height in 16:9)
                return self._click(int(sw * 0.65), int(sh * 0.35),
                    f"Pipeline(schedule_enter): fallback click Schale Office entry (tick {self._state.ticks}).")

        # Already inside schedule? Check for back button + no lobby nav
        # (schedule is a sub-screen with Home button)
        # Wait, Schale Office doesn't have a home button. We also need to check if we are in the main Schedule UI directly.
        m_all = self._match(screenshot_path, "全体课程表.png", min_score=0.35)
        m_tickets = self._match(screenshot_path, "课程表票持有数量.png", min_score=0.35)
        
        if (not self._is_lobby(screenshot_path) and self._is_subscreen(screenshot_path)) or m_all or m_tickets:
            # Likely inside schedule already
            if self._state.ticks >= 2:
                self._advance_phase()  # → SCHEDULE_EXECUTE
                return self._wait(300, "Pipeline(schedule_enter): inside schedule already, advancing.")

        # In lobby → click schedule button
        if self._is_lobby(screenshot_path):
            roi_nav = (0, int(sh * 0.80), sw, sh)
            m = self._match(screenshot_path, "课程表.png", roi=roi_nav, min_score=0.30)
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
        """Execute schedule: check locks, YOLO snipe targets, confirm, handle ticket exhaustion."""
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(schedule_exec): no screenshot.")

        if self._is_lobby(screenshot_path):
            print("[Pipeline] schedule_execute: back in lobby, schedule done.")
            self._advance_phase()
            return self._wait(300, "Pipeline(schedule_exec): back in lobby, done.")

        # 1. Handle "Ticket Exhausted / Purchase" popup
        m_cancel = self._match(screenshot_path, "取消（可点Esc）.png", min_score=0.60)
        if m_cancel:
            print("[Pipeline] schedule: ticket exhausted popup detected, finishing schedule.")
            self._advance_phase()
            return self._click(m_cancel.center[0], m_cancel.center[1], "Pipeline(schedule): cancel purchase, schedule done.")

        # 2. General popups (e.g., level up, rewards)
        m_close = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", min_score=0.75)
        if m_close is not None:
            return self._click(m_close.center[0], m_close.center[1], f"Pipeline(schedule_exec): close popup. score={m_close.score:.3f}")

        # 3. Room entry or reward confirm
        confirm_roi = (int(sw * 0.25), int(sh * 0.40), int(sw * 0.85), int(sh * 0.95))
        m_confirm = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.50)
        if m_confirm is not None:
            self._state.sub_state = "confirmed"
            return self._click(m_confirm.center[0], m_confirm.center[1], f"Pipeline(schedule_exec): confirm. score={m_confirm.score:.3f}")

        # 4. Main Schedule Page Check
        m_all = self._match(screenshot_path, "全体课程表.png", min_score=0.30)
        m_tickets = self._match(screenshot_path, "课程表票持有数量.png", min_score=0.30)
        if not m_all and not m_tickets:
            # Tap center to skip animations if we just confirmed
            if self._state.sub_state == "confirmed":
                return self._click(sw // 2, sh // 2, "Pipeline(schedule_exec): tap to skip animation.")
            # Otherwise wait for UI to settle
            if self._state.ticks >= 40:
                print("[Pipeline] schedule_execute: timeout, going back.")
                act = self._try_go_back(screenshot_path, "Pipeline(schedule_exec)")
                if act is not None:
                    return act
                self._advance_phase()
                return self._wait(300, "Pipeline(schedule_exec): timeout.")
            return self._wait(400, "Pipeline(schedule_exec): waiting for schedule UI.")

        # Clear confirmed state since we are back on the main schedule UI
        self._state.sub_state = "scan"

        # 5. Check for Locks (Priority 1: Level up academies)
        m_lock = self._match(screenshot_path, "课程表锁.png", min_score=0.60)
        if m_lock is not None:
            # Click unlocked room above the lock
            cx = m_lock.center[0]
            cy = max(int(sh * 0.15), m_lock.center[1] - int(sh * 0.22))
            return self._click(cx, cy, "Pipeline(schedule_exec): found lock, clicking unlocked room above it.")

        # 6. YOLO + Avatar Sniping
        import cv2
        import numpy as np
        import json
        from pathlib import Path
        
        if getattr(self, "_avatar_matcher", None) is None:
            from vision.avatar_matcher import AvatarMatcher
            self._avatar_matcher = AvatarMatcher("data/captures/角色头像")
        
        from vision.yolo_detector import get_yolo_detector
        yolo = get_yolo_detector(skill_name="schedule")

        favorites = []
        try:
            config_path = Path("data/app_config.json")
            if config_path.exists():
                fav_data = json.loads(config_path.read_text("utf-8"))
                favorites = fav_data.get("target_favorites", [])
        except Exception:
            pass

        room_scores = [0] * 6
        room_centers = [
            (0.24 * sw, 0.34 * sh), (0.52 * sw, 0.34 * sh), (0.80 * sw, 0.34 * sh),
            (0.24 * sw, 0.66 * sh), (0.52 * sw, 0.66 * sh), (0.80 * sw, 0.66 * sh)
        ]
        rooms = [
            (0.05 * sw, 0.15 * sh, 0.38 * sw, 0.48 * sh),
            (0.38 * sw, 0.15 * sh, 0.66 * sw, 0.48 * sh),
            (0.66 * sw, 0.15 * sh, 0.95 * sw, 0.48 * sh),
            (0.05 * sw, 0.50 * sh, 0.38 * sw, 0.85 * sh),
            (0.38 * sw, 0.50 * sh, 0.66 * sw, 0.85 * sh),
            (0.66 * sw, 0.50 * sh, 0.95 * sw, 0.85 * sh),
        ]

        img_bgr = cv2.imdecode(np.fromfile(screenshot_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if yolo and img_bgr is not None:
            dets = yolo.detect_student_avatars(screenshot_path)
            for d in dets:
                cx, cy = d["center"]
                bx1, by1, bx2, by2 = d["bbox"]
                room_idx = -1
                for i, (rx1, ry1, rx2, ry2) in enumerate(rooms):
                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                        room_idx = i
                        break
                if room_idx == -1: continue

                room_scores[room_idx] += 10
                
                if favorites:
                    roi = img_bgr[int(by1):int(by2), int(bx1):int(bx2)]
                    matched_name, score = self._avatar_matcher.match_avatar(roi, favorites)
                    if matched_name and score > 0.65:
                        print(f"[Schedule] Sniped '{matched_name}' in room {room_idx} (score: {score:.2f})")
                        room_scores[room_idx] += 1000

        max_score = max(room_scores)

        # 7. Flip Page or Select Room
        if max_score < 1000 and self._state.retries < 6:
            # Try to flip to next academy
            m_left = self._match(screenshot_path, "左切换.png", min_score=0.60)
            if m_left is not None:
                self._state.retries += 1
                return self._click(m_left.center[0], m_left.center[1], f"Pipeline(schedule_exec): no targets, flip page ({self._state.retries}/6).")

        # Found a target or checked all pages. Click best room!
        self._state.retries = 0
        if max_score > 0:
            best_room = room_scores.index(max_score)
            rx, ry = room_centers[best_room]
            return self._click(int(rx), int(ry), f"Pipeline(schedule_exec): click room {best_room} (score {max_score}).")
        
        # Fallback if literally zero students anywhere
        return self._click(int(room_centers[0][0]), int(room_centers[0][1]), "Pipeline(schedule_exec): no students found, click room 0.")

    # -----------------------------------------------------------------------
    # Club handlers
    # -----------------------------------------------------------------------

    def _handle_club_enter(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Navigate from lobby to club (社交) to claim daily AP."""
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(club_enter): no screenshot.")

        if self._state.ticks >= 15:
            self._advance_phase()
            return self._wait(300, "Pipeline(club_enter): timeout, skipping.")

        # Close popups
        m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", min_score=0.70)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(club_enter): close popup. score={m.score:.3f}")

        # Confirm dialogs
        confirm_roi = (int(sw * 0.25), int(sh * 0.40), int(sw * 0.75), int(sh * 0.90))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.50)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(club_enter): confirm. score={m.score:.3f}")

        # Already inside club? (not lobby, is subscreen)
        if not self._is_lobby(screenshot_path) and self._is_subscreen(screenshot_path):
            if self._state.ticks >= 2:
                self._advance_phase()
                return self._wait(500, "Pipeline(club_enter): inside club, advancing to claim.")

        # In lobby → click club button
        if self._is_lobby(screenshot_path):
            roi_nav = (0, int(sh * 0.80), sw, sh)
            m = self._match(screenshot_path, "社交.png", roi=roi_nav, min_score=0.40)
            if m is not None:
                return self._click(m.center[0], m.center[1],
                    f"Pipeline(club_enter): click club. score={m.score:.3f}")
            if self._state.ticks >= 8:
                print("[Pipeline] club_enter: club button not found, skipping.")
                self._advance_phase()
                return self._wait(300, "Pipeline(club_enter): club not found, skipping.")
            return self._wait(400, "Pipeline(club_enter): looking for club button.")

        return self._wait(500, "Pipeline(club_enter): waiting.")

    def _handle_club_claim(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Inside club: dismiss AP reward popup, then exit back to lobby.

        The club page auto-shows a +10 AP popup on entry. Just tap to dismiss,
        then click back/Home to return to lobby.
        """
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(club_claim): no screenshot.")

        if self._state.ticks >= 15:
            self._advance_phase()
            return self._wait(300, "Pipeline(club_claim): timeout, advancing.")

        # If back in lobby, we're done
        if self._is_lobby(screenshot_path):
            print("[Pipeline] club_claim: back in lobby.")
            self._advance_phase()
            return self._wait(300, "Pipeline(club_claim): back in lobby, done.")

        # Close any popup / dismiss reward
        m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", min_score=0.65)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(club_claim): close popup. score={m.score:.3f}")

        # Confirm
        confirm_roi = (int(sw * 0.25), int(sh * 0.40), int(sw * 0.75), int(sh * 0.90))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.50)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(club_claim): confirm. score={m.score:.3f}")

        # Tap to dismiss any fullscreen reward overlay (first few ticks)
        if self._state.ticks <= 3:
            return self._click(int(sw * 0.10), int(sh * 0.10),
                "Pipeline(club_claim): tap to dismiss reward overlay.")

        # Try to go back to lobby
        act = self._try_go_back(screenshot_path, "Pipeline(club_claim)")
        if act is not None:
            return act

        return self._wait(400, "Pipeline(club_claim): waiting.")

    # -----------------------------------------------------------------------
    # Campaign handlers
    # -----------------------------------------------------------------------

    def _handle_campaign_enter(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Navigate from lobby into the Campaign (業務區) menu.

        Sub-states:
          "" (init)     → in lobby, click 業務區 / Campaign button
          "inside"      → inside campaign menu, advance to bounties
        """
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(campaign_enter): no screenshot.")

        if self._state.ticks >= 20:
            self._advance_phase()
            return self._wait(300, "Pipeline(campaign_enter): timeout, skipping.")

        # Close popups
        m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", min_score=0.70)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(campaign_enter): close popup. score={m.score:.3f}")

        # Confirm
        confirm_roi = (int(sw * 0.25), int(sh * 0.40), int(sw * 0.75), int(sh * 0.90))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.50)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(campaign_enter): confirm. score={m.score:.3f}")

        # Already inside campaign sub-screen?
        if not self._is_lobby(screenshot_path) and self._is_subscreen(screenshot_path):
            if self._state.ticks >= 2:
                self._advance_phase()
                return self._wait(300, "Pipeline(campaign_enter): inside campaign, advancing.")

        # In lobby → defer to VLM for clicking campaign button
        # (we don't have a 業務區 template yet)
        if self._is_lobby(screenshot_path):
            # Return None to let VLM handle navigation
            return None

        return self._wait(500, "Pipeline(campaign_enter): waiting.")

    def _handle_campaign_bounties(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """悬赏通缉 (Bounties) sweep loop.

        Strategy: Click each bounty type → select highest difficulty →
        sweep max → confirm → close results → next type → back.

        For now, defer to VLM since we need specific templates.
        Return None to let VLM handle.
        """
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(bounties): no screenshot.")

        if self._state.ticks >= 40:
            print("[Pipeline] bounties: timeout, advancing.")
            self._advance_phase()
            return self._wait(300, "Pipeline(bounties): timeout, advancing.")

        # If back in lobby, skip remaining campaign phases
        if self._is_lobby(screenshot_path):
            print("[Pipeline] bounties: back in lobby unexpectedly.")
            self._enter_phase(Phase.CAMPAIGN_EXIT)
            return self._wait(300, "Pipeline(bounties): back in lobby.")

        # Close popups / confirm dialogs
        m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", min_score=0.70)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(bounties): close popup. score={m.score:.3f}")

        confirm_roi = (int(sw * 0.25), int(sh * 0.40), int(sw * 0.75), int(sh * 0.90))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.50)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(bounties): confirm. score={m.score:.3f}")

        # Defer to VLM for navigation and sweep actions
        return None

    def _handle_campaign_scrimmages(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """学院交流会 (Scrimmages) sweep loop.

        Same pattern as bounties. Defer to VLM.
        """
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(scrimmages): no screenshot.")

        if self._state.ticks >= 40:
            print("[Pipeline] scrimmages: timeout, advancing.")
            self._advance_phase()
            return self._wait(300, "Pipeline(scrimmages): timeout, advancing.")

        if self._is_lobby(screenshot_path):
            print("[Pipeline] scrimmages: back in lobby unexpectedly.")
            self._enter_phase(Phase.CAMPAIGN_EXIT)
            return self._wait(300, "Pipeline(scrimmages): back in lobby.")

        m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", min_score=0.70)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(scrimmages): close popup. score={m.score:.3f}")

        confirm_roi = (int(sw * 0.25), int(sh * 0.40), int(sw * 0.75), int(sh * 0.90))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.50)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(scrimmages): confirm. score={m.score:.3f}")

        return None

    def _handle_campaign_pvp(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """战术对抗赛 (Tactical PvP): claim rewards + optional battle.

        Strategy:
          1. Enter PvP screen
          2. Click claim reward buttons (time reward + daily reward)
          3. Optional: fight one match (for daily quest)
          4. Exit

        Defer to VLM for now.
        """
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(pvp): no screenshot.")

        if self._state.ticks >= 40:
            print("[Pipeline] pvp: timeout, advancing.")
            self._advance_phase()
            return self._wait(300, "Pipeline(pvp): timeout, advancing.")

        if self._is_lobby(screenshot_path):
            print("[Pipeline] pvp: back in lobby unexpectedly.")
            self._enter_phase(Phase.CAMPAIGN_EXIT)
            return self._wait(300, "Pipeline(pvp): back in lobby.")

        m = self._match(screenshot_path, "游戏内很多页面窗口的叉.png", min_score=0.70)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(pvp): close popup. score={m.score:.3f}")

        confirm_roi = (int(sw * 0.25), int(sh * 0.40), int(sw * 0.75), int(sh * 0.90))
        m = self._match(screenshot_path, "确认(可以点space）.png", roi=confirm_roi, min_score=0.50)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(pvp): confirm. score={m.score:.3f}")

        return None

    def _handle_campaign_exit(self, *, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """Exit campaign area back to lobby."""
        sw, sh = self._get_size(screenshot_path)
        if sw <= 0 or sh <= 0:
            return self._wait(300, "Pipeline(campaign_exit): no screenshot.")

        if self._state.ticks >= 15:
            self._advance_phase()
            return self._wait(300, "Pipeline(campaign_exit): timeout, advancing.")

        # Already in lobby → done
        if self._is_lobby(screenshot_path):
            print("[Pipeline] campaign_exit: in lobby.")
            self._advance_phase()
            return self._wait(300, "Pipeline(campaign_exit): back in lobby, done.")

        # Click Home button
        m = self._match(screenshot_path, "Home按钮.png", min_score=0.60)
        if m is not None:
            return self._click(m.center[0], m.center[1],
                f"Pipeline(campaign_exit): click Home. score={m.score:.3f}")

        # Press Escape to go back
        return {"action": "back", "reason": "Pipeline(campaign_exit): press ESC to go back.", "_pipeline": True}

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
