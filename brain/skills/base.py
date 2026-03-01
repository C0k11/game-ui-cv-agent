"""Base skill framework for Blue Archive automation.

Core concepts:
- ScreenState: OCR + metadata snapshot of current game screen
- Action: dict returned by skills telling the pipeline what to do
- BaseSkill: abstract class all skills inherit from

OCR text matching is the PRIMARY navigation method (portable across resolutions).
All coordinates are normalized 0-1 ratios.
"""
from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Detection boxes ────────────────────────────────────────────────────

@dataclass
class YoloBox:
    """YOLO detection result (pixel coords + normalized)."""
    cls_id: int
    cls_name: str
    confidence: float
    # Normalized 0-1
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass
class OcrBox:
    text: str
    confidence: float
    x1: float  # normalized 0-1
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


# ── Screen state ────────────────────────────────────────────────────────

@dataclass
class ScreenState:
    """Snapshot of what's on screen right now."""
    ocr_boxes: List[OcrBox] = field(default_factory=list)
    yolo_boxes: List[YoloBox] = field(default_factory=list)
    image_w: int = 0
    image_h: int = 0
    screenshot_path: str = ""
    timestamp: float = field(default_factory=time.time)

    # ── OCR text search helpers ──

    def find_text(self, pattern: str, *, min_conf: float = 0.5,
                  region: Optional[Tuple[float, float, float, float]] = None) -> List[OcrBox]:
        """Find OCR boxes matching a text pattern (substring or regex).

        Args:
            pattern: text to search for (case-insensitive substring match,
                     or regex if it contains special chars)
            min_conf: minimum confidence threshold
            region: optional (x1, y1, x2, y2) normalized region filter
        """
        results = []
        for box in self.ocr_boxes:
            if box.confidence < min_conf:
                continue
            if region:
                rx1, ry1, rx2, ry2 = region
                if box.cx < rx1 or box.cx > rx2 or box.cy < ry1 or box.cy > ry2:
                    continue
            # Try substring match first, then regex
            if pattern.lower() in box.text.lower():
                results.append(box)
            else:
                try:
                    if re.search(pattern, box.text, re.IGNORECASE):
                        results.append(box)
                except re.error:
                    pass
        return results

    def find_text_one(self, pattern: str, **kwargs) -> Optional[OcrBox]:
        """Find first matching OCR box."""
        hits = self.find_text(pattern, **kwargs)
        return hits[0] if hits else None

    def has_text(self, pattern: str, **kwargs) -> bool:
        """Check if text exists on screen."""
        return len(self.find_text(pattern, **kwargs)) > 0

    def find_any_text(self, patterns: List[str], **kwargs) -> Optional[OcrBox]:
        """Find first match from multiple patterns."""
        for p in patterns:
            hit = self.find_text_one(p, **kwargs)
            if hit:
                return hit
        return None

    # ── YOLO detection helpers ──

    def find_yolo(self, cls_name: str, *, min_conf: float = 0.3,
                  region: Optional[Tuple[float, float, float, float]] = None) -> List[YoloBox]:
        """Find YOLO detections by class name (substring match)."""
        results = []
        for box in self.yolo_boxes:
            if box.confidence < min_conf:
                continue
            if cls_name.lower() not in box.cls_name.lower():
                continue
            if region:
                rx1, ry1, rx2, ry2 = region
                if box.cx < rx1 or box.cx > rx2 or box.cy < ry1 or box.cy > ry2:
                    continue
            results.append(box)
        return results

    def find_yolo_one(self, cls_name: str, **kwargs) -> Optional[YoloBox]:
        """Find best (highest conf) YOLO detection by class name."""
        hits = self.find_yolo(cls_name, **kwargs)
        if not hits:
            return None
        return max(hits, key=lambda b: b.confidence)

    def has_yolo(self, cls_name: str, **kwargs) -> bool:
        return len(self.find_yolo(cls_name, **kwargs)) > 0

    # ── Region constants for Blue Archive ──

    # Bottom navigation bar (咖啡廳, 課程表, 學生, 编辑, 社交, 製造, 商店, 招募)
    NAV_BAR = (0.0, 0.90, 1.0, 1.0)
    # Top status bar (AP, credits, gems)
    TOP_BAR = (0.0, 0.0, 1.0, 0.10)
    # Left sidebar (公告, MomoTalk, 任務)
    LEFT_SIDE = (0.0, 0.20, 0.20, 0.50)
    # Center dialog area
    CENTER = (0.25, 0.15, 0.75, 0.85)

    def is_lobby(self) -> bool:
        """Detect if we're on the main lobby screen.

        Lobby has bottom nav with multiple buttons: 咖啡廳/咖啡厅, 課程表, 商店, etc.
        """
        nav_hits = 0
        nav_texts = ["咖啡", "課程", "课程", "學生", "学生", "社交", "製造", "制造", "商店", "招募"]
        for t in nav_texts:
            if self.find_text_one(t, region=self.NAV_BAR, min_conf=0.5):
                nav_hits += 1
        return nav_hits >= 3

    def is_loading(self) -> bool:
        """Detect loading screen."""
        return self.has_text("Loading", min_conf=0.7) or self.has_text("loading", min_conf=0.7)

    def is_dialog(self) -> bool:
        """Detect a popup/dialog (has 確認/取消 or 確定 buttons)."""
        return (self.has_text("確認", region=self.CENTER, min_conf=0.8) or
                self.has_text("确认", region=self.CENTER, min_conf=0.8) or
                self.has_text("確定", region=self.CENTER, min_conf=0.8))


# ── Action helpers ──────────────────────────────────────────────────────

def action_click(nx: float, ny: float, reason: str = "") -> Dict[str, Any]:
    """Click at normalized coordinates (0-1)."""
    return {"action": "click", "target": [nx, ny], "reason": reason}


def action_click_box(box: OcrBox, reason: str = "") -> Dict[str, Any]:
    """Click center of an OCR box."""
    return action_click(box.cx, box.cy, reason or f"click '{box.text}'")


def action_click_yolo(box: YoloBox, reason: str = "") -> Dict[str, Any]:
    """Click center of a YOLO detection box."""
    return action_click(box.cx, box.cy, reason or f"click yolo '{box.cls_name}'")


def action_wait(ms: int = 500, reason: str = "") -> Dict[str, Any]:
    """Wait for specified duration."""
    return {"action": "wait", "duration_ms": ms, "reason": reason}


def action_back(reason: str = "") -> Dict[str, Any]:
    """Press Escape / Back."""
    return {"action": "back", "reason": reason}


def action_swipe(fx: float, fy: float, tx: float, ty: float,
                 duration_ms: int = 400, reason: str = "") -> Dict[str, Any]:
    """Swipe between two normalized points."""
    return {"action": "swipe", "from": [fx, fy], "to": [tx, ty],
            "duration_ms": duration_ms, "reason": reason}


def action_scroll(nx: float, ny: float, clicks: int = -3, reason: str = "") -> Dict[str, Any]:
    """Mouse wheel scroll at normalized position."""
    return {"action": "scroll", "target": [nx, ny], "clicks": clicks, "reason": reason}


def action_done(reason: str = "") -> Dict[str, Any]:
    """Signal that this skill is finished."""
    return {"action": "done", "reason": reason}


# ── Base Skill ──────────────────────────────────────────────────────────

class BaseSkill(ABC):
    """Abstract base for all agent skills.

    Each skill handles one game section (e.g. cafe, schedule, bounties).
    Skills are stateful - they track their sub-state across ticks.

    Lifecycle:
        1. Pipeline calls reset() before starting the skill
        2. Pipeline calls tick(screen) each frame
        3. Skill returns an action dict
        4. If action is "done", pipeline moves to next skill
    """

    def __init__(self, name: str):
        self.name = name
        self.sub_state: str = ""
        self.ticks: int = 0
        self.max_ticks: int = 60  # timeout per skill
        self._log_lines: List[str] = []

    def reset(self) -> None:
        """Reset skill state for a fresh run."""
        self.sub_state = ""
        self.ticks = 0
        self._log_lines = []

    def log(self, msg: str) -> None:
        line = f"[{self.name}] {msg}"
        self._log_lines.append(line)
        print(line)

    @abstractmethod
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        """Process one frame and return an action.

        Returns action dict. Return action_done() when skill is complete.
        """
        ...

    def _handle_common_popups(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Handle common popups that can appear in any skill.

        Returns an action if a popup was handled, None otherwise.
        """
        # Close generic X buttons (叉叉) - common popups
        # Confirm dialogs (確認/Space)
        confirm = screen.find_any_text(
            ["確認", "确认", "確定", "确定"],
            region=screen.CENTER, min_conf=0.8
        )
        if confirm:
            # Check if there's also a 取消 (cancel) - if so, be careful
            cancel = screen.find_any_text(
                ["取消"],
                region=screen.CENTER, min_conf=0.8
            )
            if cancel:
                # Dialog with confirm+cancel - only confirm if safe
                # (subclass should handle specific dialogs)
                return None
            # Simple confirm-only popup (rewards, level up, etc.)
            self.log(f"confirm popup: '{confirm.text}'")
            return action_click_box(confirm, f"confirm popup '{confirm.text}'")

        # "是否跳過" (skip) dialog - always confirm
        skip = screen.find_text_one("跳過", region=screen.CENTER, min_conf=0.8)
        if skip:
            confirm2 = screen.find_any_text(["確", "确"], region=screen.CENTER, min_conf=0.8)
            if confirm2:
                self.log("skip dialog: confirming")
                return action_click_box(confirm2, "confirm skip dialog")

        return None

    def _try_go_lobby(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Try to navigate back to lobby."""
        if screen.is_lobby():
            return None
        return action_back(f"{self.name}: press back toward lobby")

    def _nav_to(self, screen: ScreenState, nav_texts: List[str]) -> Optional[Dict[str, Any]]:
        """Click a bottom nav bar button by text."""
        for t in nav_texts:
            hit = screen.find_text_one(t, region=screen.NAV_BAR, min_conf=0.5)
            if hit:
                self.log(f"nav click '{hit.text}'")
                return action_click_box(hit, f"nav to '{hit.text}'")
        return None
