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
from pathlib import Path
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
class TemplateHitBox:
    """Template matching result (normalized 0-1 coords)."""
    label: str
    confidence: float
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
class ScreenState:
    """Snapshot of what's on screen right now."""
    ocr_boxes: List[OcrBox] = field(default_factory=list)
    yolo_boxes: List[YoloBox] = field(default_factory=list)
    template_hits: List[TemplateHitBox] = field(default_factory=list)
    florence_boxes: List[OcrBox] = field(default_factory=list)
    image_w: int = 0
    image_h: int = 0
    screenshot_path: str = ""
    timestamp: float = field(default_factory=time.time)

    def add_florence_boxes(self, boxes: List[OcrBox]) -> None:
        """Append Florence detection boxes."""
        self.florence_boxes.extend(boxes)

    # ── OCR text search helpers ──

    def find_text(self, pattern: str, *, min_conf: float = 0.5,
                  region: Optional[Tuple[float, float, float, float]] = None) -> List[OcrBox]:
        """Find OCR boxes matching a text pattern (substring or regex).

        Matching runs on *normalized* text: OCR output and the pattern are
        both passed through `vision.ocr_normalize.normalize` (known misreads
        fixed, Traditional chars folded to Simplified). Skills no longer
        need to enumerate Trad/Simp/mixed permutations of a keyword.

        Args:
            pattern: text to search for (case-insensitive substring match,
                     or regex if it contains special chars)
            min_conf: minimum confidence threshold
            region: optional (x1, y1, x2, y2) normalized region filter
        """
        # Lazy import so vision module tests don't require brain.
        try:
            from vision.ocr_normalize import normalize as _norm
        except Exception:
            _norm = lambda s: s  # fall through to raw match
        norm_pattern = _norm(pattern).lower()
        # Only fall through to regex if the pattern contains *clearly intentional*
        # regex syntax. A bare `?` / `+` after a literal char (e.g. "次?") is
        # syntactically valid regex but almost always a keyword author's literal-
        # punctuation typo, and `次?` matches the empty string at every position
        # → false-positive on every OCR box. Keep regex opt-in via clear markers.
        _regex_markers = (".*", ".+", "[", "\\", "|", "^", "$", "(?")
        treat_as_regex = any(m in norm_pattern for m in _regex_markers)
        results = []
        for box in self.ocr_boxes:
            if box.confidence < min_conf:
                continue
            if region:
                rx1, ry1, rx2, ry2 = region
                if box.cx < rx1 or box.cx > rx2 or box.cy < ry1 or box.cy > ry2:
                    continue
            norm_text = _norm(box.text).lower()
            # Try substring match first, then regex (only if pattern looks regex-y).
            if norm_pattern in norm_text:
                results.append(box)
            elif treat_as_regex:
                try:
                    if re.search(norm_pattern, norm_text, re.IGNORECASE):
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

    def find_clickable_text(
        self,
        patterns: List[str],
        *,
        min_conf: float = 0.5,
        region: Optional[Tuple[float, float, float, float]] = None,
        disabled_saturation: float = 0.20,
    ) -> Optional[OcrBox]:
        """Find a text match AND verify the pixel region isn't grayed-out.

        A common failure mode: a button's text is OCR'd (e.g. "掃蕩開始")
        but the button is disabled/locked (greyed). Clicking does nothing
        and the skill stalls. We avoid that by sampling the box's pixels
        from the saved screenshot and rejecting matches whose saturation
        is below `disabled_saturation` — disabled UI is desaturated in BA.

        If the screenshot isn't available, falls back to plain text match
        (no verification). Returns the first match passing both checks.
        """
        # Plain text hit first
        for p in patterns:
            box = self.find_text_one(p, min_conf=min_conf, region=region)
            if box is None:
                continue
            if not self.screenshot_path:
                return box  # no screenshot to verify; accept
            try:
                import cv2  # noqa: PLC0415
                import numpy as np  # noqa: PLC0415
                from vision.io_utils import imread_any  # noqa: PLC0415
                img = imread_any(self.screenshot_path)
                if img is None:
                    return box
                H, W = img.shape[:2]
                ix1 = max(0, int(box.x1 * W)); iy1 = max(0, int(box.y1 * H))
                ix2 = min(W, int(box.x2 * W)); iy2 = min(H, int(box.y2 * H))
                if ix2 - ix1 < 4 or iy2 - iy1 < 4:
                    return box
                crop = img[iy1:iy2, ix1:ix2]
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                mean_sat = float(np.mean(hsv[:, :, 1])) / 255.0
                if mean_sat < disabled_saturation:
                    # Grayed-out — skip this match, try next pattern
                    continue
                return box
            except Exception:
                return box
        return None

    def reocr_region(
        self,
        region: Tuple[float, float, float, float],
        *,
        min_conf: float = 0.5,
        pad_frac: float = 0.02,
    ) -> List[OcrBox]:
        """Re-run OCR on a cropped region of the current frame.

        Use when you need a higher-confidence reading on a specific area
        (e.g. a mission popup's start-button row) than the full-frame
        pass provided. Crops the saved screenshot, runs OCR, and returns
        boxes with coordinates re-mapped to the FULL-frame normalized
        space (so downstream region filters work unchanged).

        Cost on RTX 4090 with CUDA: ~100–200ms per call for a popup-
        sized region. Not for per-tick use — reserve for cases where the
        full-frame pass produced ambiguous or sub-threshold results.
        """
        x1, y1, x2, y2 = region
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            return []
        if not self.screenshot_path:
            return []
        try:
            import cv2  # noqa: PLC0415
            from vision.io_utils import imread_any  # noqa: PLC0415
            img = imread_any(self.screenshot_path)
            if img is None:
                return []
        except Exception:
            return []
        H, W = img.shape[:2]
        # Add small padding so text near edges isn't clipped
        px = pad_frac * (x2 - x1)
        py = pad_frac * (y2 - y1)
        ix1 = max(0, int((x1 - px) * W))
        iy1 = max(0, int((y1 - py) * H))
        ix2 = min(W, int((x2 + px) * W))
        iy2 = min(H, int((y2 + py) * H))
        if ix2 - ix1 < 8 or iy2 - iy1 < 8:
            return []
        crop = img[iy1:iy2, ix1:ix2]
        try:
            from brain.pipeline import _get_ocr  # noqa: PLC0415
            ocr = _get_ocr()
            result, _ = ocr(crop)
        except Exception:
            return []
        if not result:
            return []
        # Map crop-local coords back to full-frame normalized coords
        crop_w = ix2 - ix1
        crop_h = iy2 - iy1
        boxes: List[OcrBox] = []
        for item in result:
            try:
                poly, text, conf = item
                conf = float(conf)
            except (ValueError, TypeError):
                continue
            if conf < min_conf or not text:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            cx1 = (ix1 + min(xs)) / W
            cy1 = (iy1 + min(ys)) / H
            cx2 = (ix1 + max(xs)) / W
            cy2 = (iy1 + max(ys)) / H
            boxes.append(OcrBox(
                text=str(text),
                confidence=float(conf),
                x1=float(cx1), y1=float(cy1),
                x2=float(cx2), y2=float(cy2),
            ))
        return boxes

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

    # ── Template match helpers ──

    def find_template(self, label: str, *, min_conf: float = 0.5,
                      region: Optional[Tuple[float, float, float, float]] = None) -> List[TemplateHitBox]:
        """Find template hits by label."""
        results = []
        for h in self.template_hits:
            if h.confidence < min_conf:
                continue
            if label.lower() not in h.label.lower():
                continue
            if region:
                rx1, ry1, rx2, ry2 = region
                if h.cx < rx1 or h.cx > rx2 or h.cy < ry1 or h.cy > ry2:
                    continue
            results.append(h)
        return sorted(results, key=lambda h: h.confidence, reverse=True)

    def find_template_one(self, label: str, **kwargs) -> Optional[TemplateHitBox]:
        """Find best template hit by label."""
        hits = self.find_template(label, **kwargs)
        return hits[0] if hits else None

    # ── Pixel color sampling ──
    #
    # reference uses RGB pixel checks for button states. We sample at normalized
    # positions on the screenshot for resolution-independent state detection.

    def load_image(self):
        """Return the decoded BGR frame of this tick's screenshot.

        The decoded ``np.ndarray`` is cached on the ``ScreenState`` so
        multiple callers (sample_color, template matchers, HSV checks,
        avatar matchers, …) share one JPEG decode per tick.

        Returns ``None`` when no screenshot is available or decoding
        fails.
        """
        cached = getattr(self, "_cached_bgr_image", None)
        if cached is not None:
            return cached
        if not self.screenshot_path:
            return None
        try:
            import cv2
            import numpy as np
            img = cv2.imdecode(
                np.fromfile(self.screenshot_path, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        except Exception:
            img = None
        # Cache even None so repeated failures don't re-attempt decode.
        try:
            object.__setattr__(self, "_cached_bgr_image", img)
        except Exception:
            pass
        return img

    def sample_color(self, nx: float, ny: float, patch: int = 3) -> Optional[Tuple[int, int, int]]:
        """Sample average BGR color at normalized position (0-1).

        Args:
            nx, ny: normalized (0-1) position in the screenshot.
            patch: half-size of the sampling square (in pixels at native res).
                   Averages a (2*patch+1)² area for noise robustness.

        Returns (B, G, R) tuple, or None if screenshot not available.
        """
        img = self.load_image()
        if img is None:
            return None
        try:
            h, w = img.shape[:2]
            px = max(patch, min(w - patch - 1, int(nx * w)))
            py = max(patch, min(h - patch - 1, int(ny * h)))
            roi = img[py - patch:py + patch + 1, px - patch:px + patch + 1]
            mean = roi.mean(axis=(0, 1)).astype(int)
            return (int(mean[0]), int(mean[1]), int(mean[2]))
        except Exception:
            return None

    def check_color(
        self,
        nx: float,
        ny: float,
        *,
        rgb_min: Optional[Tuple[int, int, int]] = None,
        rgb_max: Optional[Tuple[int, int, int]] = None,
        hsv_min: Optional[Tuple[int, int, int]] = None,
        hsv_max: Optional[Tuple[int, int, int]] = None,
        patch: int = 3,
    ) -> bool:
        """Check if pixel color at (nx, ny) falls within given range.

        Specify rgb_min/rgb_max for RGB range check, or hsv_min/hsv_max for
        HSV range check (OpenCV scale: H 0-179, S 0-255, V 0-255).
        Both can be specified (all conditions must pass).

        Returns False if screenshot unavailable.
        """
        bgr = self.sample_color(nx, ny, patch=patch)
        if bgr is None:
            return False
        b, g, r = bgr
        if rgb_min is not None and rgb_max is not None:
            if not (rgb_min[0] <= r <= rgb_max[0] and
                    rgb_min[1] <= g <= rgb_max[1] and
                    rgb_min[2] <= b <= rgb_max[2]):
                return False
        if hsv_min is not None and hsv_max is not None:
            import cv2
            import numpy as np
            pixel = np.uint8([[[b, g, r]]])
            hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
            h_val, s_val, v_val = int(hsv[0]), int(hsv[1]), int(hsv[2])
            if not (hsv_min[0] <= h_val <= hsv_max[0] and
                    hsv_min[1] <= s_val <= hsv_max[1] and
                    hsv_min[2] <= v_val <= hsv_max[2]):
                return False
        return True

    def is_button_yellow(self, nx: float, ny: float, patch: int = 5) -> bool:
        """Check if a button at (nx, ny) is yellow/gold (active state in BA).

        Blue Archive active buttons are bright yellow (~RGB 255,210,50).
        Greyed/disabled buttons are grey (~RGB 180,180,180).
        """
        return self.check_color(nx, ny, patch=patch,
                                hsv_min=(18, 80, 160), hsv_max=(38, 255, 255))

    def is_button_grey(self, nx: float, ny: float, patch: int = 5) -> bool:
        """Check if a button at (nx, ny) is grey (disabled state in BA)."""
        return self.check_color(nx, ny, patch=patch,
                                hsv_min=(0, 0, 100), hsv_max=(179, 40, 210))

    # ── Universal notification badge detection ──
    #
    # Blue Archive uses two kinds of tiny circular badges over icons /
    # buttons to signal the user:
    #
    #   🔴 red dot   = "you have something unclaimed here"
    #                  (reward waiting, new mail, unread momotalk, pass
    #                  rewards, craft finished, event reward threshold)
    #
    #   🟡 yellow dot = "you can DO something here now"
    #                  (stage unlocked, new task available, craft craftable,
    #                  ticket refilled, daily reset)
    #
    # Skills should use the helpers below rather than hard-coding HSV per
    # skill. Typical call site::
    #
    #     btn = screen.find_text_one("奖励资讯")
    #     if btn and screen.has_red_badge(btn):
    #         click(btn)   # go claim
    #
    # Badges are ~12-20px diameter and sit at the top-right corner of
    # the parent button/icon, slightly overlapping the edge. We sample a
    # small patch at the anchor's top-right with a configurable offset
    # (default +1.2% x / -0.5% y from the top-right corner, tuned so the
    # patch lands on the badge's filled center, not the border).

    # HSV envelopes for the badge colors (OpenCV H: 0-179).
    # Tight thresholds: alert badges are SATURATED pure colors, not the
    # soft orange/pink gradients of nearby buttons/icons. Loosening too
    # much false-positives on the 獎勵資訊 button's own orange outline
    # and the pink `P` currency icon.
    #
    # Real red badge pixels have S≥200 & V≥180. Previous loose H≤15
    # caught orange button edges; restrict to true red H≤8 / ≥172.
    _RED_BADGE_HSV = [
        ((0,   200, 180), (8,   255, 255)),    # red hue near 0 (strict)
        ((172, 200, 180), (179, 255, 255)),    # red hue near 180 (wraparound)
    ]
    # Yellow badges are distinct from the yellow active-button state —
    # a badge is a small saturated dot, not the whole button face.
    _YELLOW_BADGE_HSV = ((18, 200, 200), (32, 255, 255))

    def _sample_badge_color(
        self,
        anchor: Optional["OcrBox"] = None,
        *,
        nx: Optional[float] = None,
        ny: Optional[float] = None,
        offset: Tuple[float, float] = (0.012, -0.005),
        patch: int = 3,
    ) -> Optional[Tuple[int, int, int]]:
        """Sample the small patch where a notification badge would sit.

        Give EITHER an ``anchor`` (OcrBox — sample at its top-right + offset)
        OR explicit ``(nx, ny)``.
        """
        if anchor is not None:
            x = min(0.995, anchor.x2 + offset[0])
            y = max(0.005, anchor.y1 + offset[1])
        elif nx is not None and ny is not None:
            x, y = nx, ny
        else:
            return None
        return self.sample_color(x, y, patch=patch)

    def _hsv_matches(self, bgr: Tuple[int, int, int], envelopes) -> bool:
        try:
            import cv2
            import numpy as np
        except Exception:
            return False
        b, g, r = bgr
        pixel = np.uint8([[[b, g, r]]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        # Accept single envelope or list of envelopes
        if isinstance(envelopes, tuple) and len(envelopes) == 2 and all(
                isinstance(e, tuple) for e in envelopes):
            mn, mx = envelopes
            return (mn[0] <= h <= mx[0] and mn[1] <= s <= mx[1] and mn[2] <= v <= mx[2])
        for mn, mx in envelopes:
            if (mn[0] <= h <= mx[0] and mn[1] <= s <= mx[1] and mn[2] <= v <= mx[2]):
                return True
        return False

    # Region search: badge position varies per button style, but it's
    # always within a rectangle near the anchor's top-right corner. Mask
    # the region for badge-color HSV and return True when the matching
    # pixel cluster is large enough to be a real badge (not stray noise).
    #
    # Region spans slightly outside the button frame (for badges that
    # overlap the edge) and into the upper half (where alert dots sit).
    # Narrow to just the top-right corner where badge actually sits.
    # Before this was too wide and caught the button body/gradient.
    _BADGE_REGION_OFFSET = (-0.03, -0.08, +0.03, -0.02)   # (dx1,dy1,dx2,dy2)
    _BADGE_MIN_PIXELS = 20   # minimum saturated pixel cluster to count

    def _badge_matches_any_offset(
        self,
        envelopes,
        anchor: Optional["OcrBox"] = None,
        *,
        nx: Optional[float] = None,
        ny: Optional[float] = None,
        offset: Optional[Tuple[float, float]] = None,
    ) -> bool:
        # Explicit single-point mode when nx/ny or offset explicitly given
        if nx is not None and ny is not None:
            bgr = self._sample_badge_color(None, nx=nx, ny=ny)
            return bgr is not None and self._hsv_matches(bgr, envelopes)
        if offset is not None:
            bgr = self._sample_badge_color(anchor, offset=offset)
            return bgr is not None and self._hsv_matches(bgr, envelopes)
        # Default: region mask scan around anchor's top-right corner
        if anchor is None:
            return False
        try:
            import cv2
            import numpy as np
        except Exception:
            return False
        img = self.load_image()
        if img is None:
            return False
        h, w = img.shape[:2]
        dx1, dy1, dx2, dy2 = self._BADGE_REGION_OFFSET
        x0 = max(0.0, min(0.995, anchor.x2 + dx1))
        y0 = max(0.0, min(0.995, anchor.y1 + dy1))
        x1 = max(0.005, min(1.0, anchor.x2 + dx2))
        y1 = max(0.005, min(1.0, anchor.y1 + dy2))
        if x1 <= x0 or y1 <= y0:
            return False
        px0, py0 = int(x0 * w), int(y0 * h)
        px1, py1 = int(x1 * w), int(y1 * h)
        roi = img[py0:py1, px0:px1]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Accept envelope as list or single tuple
        envs = envelopes if isinstance(envelopes, list) else [envelopes]
        total = 0
        for mn, mx in envs:
            m = cv2.inRange(hsv, np.array(mn, dtype=np.uint8),
                                  np.array(mx, dtype=np.uint8))
            total += int(m.sum() // 255)
            if total >= self._BADGE_MIN_PIXELS:
                return True
        return total >= self._BADGE_MIN_PIXELS

    def has_red_badge(
        self,
        anchor: Optional["OcrBox"] = None,
        *,
        nx: Optional[float] = None,
        ny: Optional[float] = None,
        offset: Optional[Tuple[float, float]] = None,
    ) -> bool:
        """True if a red unclaimed-reward badge overlays the anchor.

        Default: probes 5 offsets around anchor's top-right corner.
        Pass explicit ``offset`` or ``(nx, ny)`` for single-point mode.
        """
        return self._badge_matches_any_offset(
            self._RED_BADGE_HSV, anchor, nx=nx, ny=ny, offset=offset)

    def has_yellow_badge(
        self,
        anchor: Optional["OcrBox"] = None,
        *,
        nx: Optional[float] = None,
        ny: Optional[float] = None,
        offset: Optional[Tuple[float, float]] = None,
    ) -> bool:
        """True if a yellow actionable badge overlays the anchor."""
        return self._badge_matches_any_offset(
            self._YELLOW_BADGE_HSV, anchor, nx=nx, ny=ny, offset=offset)

    def badge_state(
        self,
        anchor: Optional["OcrBox"] = None,
        *,
        nx: Optional[float] = None,
        ny: Optional[float] = None,
        offset: Optional[Tuple[float, float]] = None,
    ) -> str:
        """Return 'red' | 'yellow' | 'none' for a badge at anchor."""
        if self._badge_matches_any_offset(
                self._RED_BADGE_HSV, anchor, nx=nx, ny=ny, offset=offset):
            return "red"
        if self._badge_matches_any_offset(
                self._YELLOW_BADGE_HSV, anchor, nx=nx, ny=ny, offset=offset):
            return "yellow"
        return "none"

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

    # Names → (TC, SC) text variants for the 8 bottom-nav icons.  Used by
    # scan_lobby_nav_badges() so callers get a stable english key
    # regardless of which OCR character variant fired on a given frame.
    # 編輯 sometimes mis-OCRs (small icon, no Chinese-character anchor)
    # so we also accept partial / known mis-reads.
    _LOBBY_NAV_VARIANTS = (
        ("cafe",     ["咖啡廳", "咖啡厅", "咖啡"]),
        ("schedule", ["課程表", "课程表", "課程", "课程"]),
        ("student",  ["學生", "学生"]),
        ("edit",     ["編輯", "编辑"]),
        ("social",   ["社交"]),
        ("craft",    ["製造", "制造"]),
        ("shop",     ["商店"]),
        ("recruit",  ["招募"]),
    )

    # HSV envelopes used by scan_lobby_nav_badges.  Saturation/value
    # floors are intentionally high to reject muted anime-art reds and
    # the icon body's own decorative fills (e.g. cafe heart cup).  Real
    # nav badges are punchy, saturated, and high-contrast.
    _NAV_DOT_RED_HSV = [
        ((0, 120, 160),   (12, 255, 255)),
        ((168, 120, 160), (180, 255, 255)),
    ]
    _NAV_DOT_YELLOW_HSV = [
        ((10, 120, 180), (45, 255, 255)),
    ]
    # Min pixel count for a real dot — under this is anti-alias / noise.
    # On a 2255×1268 frame a real dot is ~100-200px (verified: student
    # yellow=122, social red=118, craft yellow=120).  Floor at 80 to
    # catch faint dots while rejecting stray icon-art pixel clusters.
    _NAV_DOT_MIN_PIXELS = 80

    def scan_lobby_nav_badges(self) -> Dict[str, str]:
        """Scan the 8 bottom-nav icons for red/yellow badges.

        Returns a dict {nav_name: state} where state ∈ {"red", "yellow",
        "none"} and nav_name is the stable english key (cafe / schedule /
        student / edit / social / craft / shop / recruit).  Names not
        currently visible on screen are omitted from the dict — callers
        should treat absence as "unknown" (we're probably not on lobby).

        Where the dot lives: BA renders the badge as a small saturated
        dot just BELOW the icon body and to the RIGHT of the label.  In
        the standard 4K-window the dot centroid is at roughly
        (label_cx + 0.012, ~0.93).  We scan a tight ROI just above the
        label top and right of label centre (`label_cx + [0.005, 0.045]`,
        y `[0.905, 0.945]`).  This ROI deliberately excludes the icon
        body (above) — that part is full of decorative colors like the
        cafe heart, recruit gold stars, etc. — to avoid false positives.

        Caller should ensure they're on lobby (is_lobby() == True) before
        calling — running on non-lobby screens may catch false positives
        from whatever happens to live in the bottom strip.
        """
        try:
            import cv2
            import numpy as np
        except Exception:
            return {}
        img = self.load_image()
        if img is None:
            return {}
        h, w = img.shape[:2]
        results: Dict[str, str] = {}
        for key, variants in self._LOBBY_NAV_VARIANTS:
            anchor = None
            for v in variants:
                anchor = self.find_text_one(v, region=self.NAV_BAR, min_conf=0.5)
                if anchor:
                    break
            if not anchor:
                continue
            cx = (anchor.x1 + anchor.x2) / 2.0
            # Tight ROI: right of label, between icon bottom and label top.
            x0 = max(0, int((cx + 0.005) * w))
            x1 = min(w, int((cx + 0.045) * w))
            y0 = max(0, int(0.905 * h))
            y1 = min(h, int(0.945 * h))
            if x1 <= x0 or y1 <= y0:
                continue
            roi = img[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            red_total = 0
            for mn, mx in self._NAV_DOT_RED_HSV:
                mask = cv2.inRange(hsv, np.array(mn, dtype=np.uint8),
                                          np.array(mx, dtype=np.uint8))
                red_total += int(mask.sum() // 255)
            yellow_total = 0
            for mn, mx in self._NAV_DOT_YELLOW_HSV:
                mask = cv2.inRange(hsv, np.array(mn, dtype=np.uint8),
                                          np.array(mx, dtype=np.uint8))
                yellow_total += int(mask.sum() // 255)
            # Red dominates yellow if both fire (red is the stronger
            # "claim me" signal; yellow is "go look at this").
            if red_total >= self._NAV_DOT_MIN_PIXELS:
                results[key] = "red"
            elif yellow_total >= self._NAV_DOT_MIN_PIXELS:
                results[key] = "yellow"
            else:
                results[key] = "none"
        # Also scan extra fixed-position badges (top-right mail icon,
        # left sidebar 任務, right sidebar campaign 任務) so the badge-
        # skip routing extends GLOBALLY across all "go here if there's
        # something to claim" skills, not just the bottom-nav skills.
        results.update(self._scan_extra_badges(img, h, w))
        return results

    # Extra badge regions (NOT bottom-nav).  Each entry is:
    #   (key, (x0, y0, x1, y1)) — fixed normalised region to scan.
    # Use a tight ROI around the dot's actual pixel cluster, not the
    # whole icon — otherwise icon-internal colors leak as false positives.
    _EXTRA_BADGE_REGIONS = (
        # Top-right mail envelope.  Anchor centered around (0.91, 0.04).
        # Red dot when unclaimed mail.
        ("mail",            (0.890, 0.020, 0.940, 0.075)),
        # Left sidebar 任務 8/8 indicator — daily-tasks unclaimed when
        # red dot above the badge.  Anchor near (0.045, 0.18).
        ("daily_tasks_nav", (0.020, 0.140, 0.075, 0.210)),
        # Right sidebar 活動進行中 / 任務 stack — campaign or event
        # tasks with unclaimed rewards.  Tile sits around y=0.70-0.95.
        ("campaign_nav",    (0.890, 0.640, 0.980, 0.940)),
    )

    def _scan_extra_badges(self, img, h: int, w: int) -> Dict[str, str]:
        try:
            import cv2
            import numpy as np
        except Exception:
            return {}
        out: Dict[str, str] = {}
        for key, (rx0, ry0, rx1, ry1) in self._EXTRA_BADGE_REGIONS:
            x0 = max(0, int(rx0 * w))
            y0 = max(0, int(ry0 * h))
            x1 = min(w, int(rx1 * w))
            y1 = min(h, int(ry1 * h))
            if x1 <= x0 or y1 <= y0:
                continue
            roi = img[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            red_total = 0
            for mn, mx in self._NAV_DOT_RED_HSV:
                mask = cv2.inRange(hsv, np.array(mn, dtype=np.uint8),
                                          np.array(mx, dtype=np.uint8))
                red_total += int(mask.sum() // 255)
            yellow_total = 0
            for mn, mx in self._NAV_DOT_YELLOW_HSV:
                mask = cv2.inRange(hsv, np.array(mn, dtype=np.uint8),
                                          np.array(mx, dtype=np.uint8))
                yellow_total += int(mask.sum() // 255)
            # Extra regions are larger so the floor scales up slightly
            # to avoid catching anime-art reds in the campaign sidebar
            # which sits on top of the school-uniform character art.
            floor = self._NAV_DOT_MIN_PIXELS
            if key == "campaign_nav":
                floor = 180  # larger ROI, more bleed-through possible
            if red_total >= floor:
                out[key] = "red"
            elif yellow_total >= floor:
                out[key] = "yellow"
            else:
                out[key] = "none"
        return out

    def is_loading(self) -> bool:
        """Detect loading screen, download progress, or game update.

        Covers the FULL startup sequence:
        1. "Now Loading..." — post-title-screen loading
        2. "正在更新。[N/M] X.XX%" — game asset download progress
        3. "驗證下載檔案中" — file verification after download
        4. "重置遊戲資料中" — game data reset / cache rebuild
        5. "下載中" / "Downloading" — generic download state
        """
        if self.has_text("Loading", min_conf=0.7) or self.has_text("loading", min_conf=0.7):
            return True
        # Bottom-of-screen progress text during startup / update.
        # OCR garbles these regularly (e.g. "驗證下载檔案中" → mixed Trad/Simp).
        if self.find_any_text(
            [
                # Download in progress
                "下載中", "下载中", "Downloading", "ダウンロード",
                "資料下載", "资料下载",
                # Game update ("正在更新。 [N/M] X%")
                "正在更新", "正在更新。", "正在更新°",
                # File verification
                "檔案驗證", "档案验证", "驗證下載", "验证下载",
                "驗證下载", "驗證檔案", "验证档案",
                # Data reset
                "重置遊戲", "重置游戏", "重置游資", "重置游资",
                "資料中", "资料中",
                # Generic download completion
                "下載檔案", "下载档案",
            ],
            region=(0.0, 0.80, 0.60, 1.0), min_conf=0.40,
        ):
            return True
        return False

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


def action_scroll(nx: float, ny: float, clicks: int = -3, reason: str = "",
                  with_ctrl: bool = False) -> Dict[str, Any]:
    """Mouse wheel scroll at normalized position.

    with_ctrl=True holds the Ctrl key during the wheel event — required by
    some Android emulators (e.g. MuMu, LDPlayer) to trigger pinch-zoom.
    """
    return {"action": "scroll", "target": [nx, ny], "clicks": clicks,
            "reason": reason, "with_ctrl": with_ctrl}


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

    # ── Shared helpers: reusable mini-flows used by multiple skills ──

    _CLAIM_ALL_TEXTS = [
        "一鍵領取", "一键领取", "全部領取", "全部领取",
        "一次領取", "一次领取", "一键领", "一鍵领", "Claim All",
    ]
    _SINGLE_CLAIM_TEXTS = ["領取", "领取", "Claim"]
    _EMPTY_LIST_TEXTS = [
        "沒有郵件", "没有郵件", "沒有邮件", "没有邮件",
        "没有任務", "没有任务", "沒有任務", "沒有任务",
        "暫無", "暂无", "No Mail", "No Tasks",
    ]

    def find_claim_all_button(
        self,
        screen: ScreenState,
        *,
        min_conf: float = 0.6,
        region: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[OcrBox]:
        """Locate a 一鍵領取 / 全部領取 / Claim All button in any of its
        Trad/Simp/English forms. Returns OcrBox or None.
        """
        return screen.find_any_text(
            self._CLAIM_ALL_TEXTS, min_conf=min_conf, region=region,
        )

    def find_single_claim_button(
        self,
        screen: ScreenState,
        *,
        min_conf: float = 0.7,
        region: Optional[Tuple[float, float, float, float]] = (0.6, 0.1, 1.0, 0.9),
    ) -> Optional[OcrBox]:
        """Locate an individual 領取 / Claim button (default region: right
        side where per-item rewards usually sit).

        Substring matching on 領取 also matches the bulk-claim variants
        全部領取 / 一鍵領取 — which is wrong here, those are handled by
        find_claim_all_button.  Filter the candidate list to drop boxes
        whose normalised text *contains* a bulk-claim prefix.  Without
        this filter, daily-tasks per-row claims kept clicking the
        bottom-right 全部領取 button instead of the tier-bonus row's
        own 領取 (run 2026-05-13 ~21:25: 20 throttled clicks at
        (0.90,0.93) which was 全部領取, not the 8/8 領取 at (0.55,0.93)).
        """
        candidates = []
        for pat in self._SINGLE_CLAIM_TEXTS:
            candidates.extend(screen.find_text(pat, min_conf=min_conf, region=region))
        # Drop boxes whose text is actually a bulk-claim variant.
        _BULK_PREFIXES = ("全部", "一鍵", "一键", "一次")
        for box in candidates:
            txt = (box.text or "").strip()
            if any(p in txt for p in _BULK_PREFIXES):
                continue
            return box
        return None

    def is_empty_reward_list(self, screen: ScreenState) -> bool:
        """True if the current list view shows 'nothing to claim'."""
        return bool(screen.find_any_text(self._EMPTY_LIST_TEXTS, min_conf=0.6))

    @abstractmethod
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        """Process one frame and return an action.

        Returns action dict. Return action_done() when skill is complete.
        """
        ...

    def detect_current_screen(self, screen: ScreenState) -> Optional[str]:
        """Detect current screen based on OCR header and nav bar cues."""
        if screen.is_lobby():
            return "Lobby"

        header_region = (0.0, 0.0, 0.3, 0.15)

        # Check "任務" header first — but disambiguate Daily Tasks vs Campaign.
        # Daily Tasks has tabs: 全體/每天/每週/成就/挑戰任務 in the tab bar area.
        # Campaign (mission hub) does NOT have these tabs.
        mission_header = screen.find_any_text(
            ["任務", "任务"], region=header_region, min_conf=0.5
        )
        if mission_header:
            daily_tabs = screen.find_any_text(
                ["全體", "全体", "每天", "每週", "每周", "挑戰任務", "挑战任务"],
                region=(0.35, 0.10, 1.0, 0.20), min_conf=0.6
            )
            if daily_tabs:
                return "DailyTasks"
            return "Mission"

        headers = {
            "Cafe": ["咖啡廳", "咖啡厅"],
            "Schedule": ["課程表", "课程表", "全体課程", "全体课程"],
            "Shop": ["商店"],
            "Club": ["社團", "社团", "Club"],
            "Bounty": ["懸賞通緝", "悬赏通缉", "懸賞", "悬赏", "通緝", "通缉", "悬通", "Bounty"],
            "PVP": ["戰術對抗", "战术对抗", "戰術大賽", "战术大赛", "術大赛", "術大賽", "大賽", "大赛"],
            "TotalAssault": ["總力戰", "总力战", "大決戰", "大决战"],
            "Pass": ["通行證", "通行证", "Pass", "PASS"],
            "Mail": ["郵件", "邮件", "郵箱", "邮箱", "信箱", "Mail"],
            "Event": ["活動", "活动"],
            "Craft": ["製造", "制造", "Craft"],
            "Student": ["學生", "学生", "Student"],
            "Formation": ["部隊", "编队", "部隊編成"],
        }
        for screen_name, texts in headers.items():
            if screen.find_any_text(texts, region=header_region, min_conf=0.6):
                return screen_name

        # Fallback: campaign hub detection via grid markers (OCR often misses 任務 header)
        # Keep this AFTER specific header checks to avoid classifying total assault/pass as Mission.
        hub_markers = screen.find_any_text(
            ["懸賞通緝", "悬赏通缉", "學園交流會", "学园交流会",
             "戰術大賽", "战术大赛", "特殊任務", "特殊任务",
             "制約解除", "劇情", "剧情"],
            min_conf=0.5
        )
        if hub_markers:
            return "Mission"
        area_marker = screen.find_text_one(r"Area\s*\d+", min_conf=0.5)
        if area_marker:
            return "Mission"
        stage_tabs = screen.find_any_text(
            ["Normal", "Hard"],
            region=(0.50, 0.16, 0.98, 0.28),
            min_conf=0.6,
        )
        stage_id = screen.find_text_one(r"\d+\-\d+", min_conf=0.5)
        entry_btn = screen.find_any_text(
            ["入場", "入场"],
            region=(0.78, 0.22, 0.98, 0.78),
            min_conf=0.5,
        )
        if stage_tabs and (stage_id or entry_btn or area_marker):
            return "Mission"
        return None

    def _handle_common_popups(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Handle common popups that can appear in any skill.

        Returns an action if a popup was handled, None otherwise.
        """
        # Notification modal headers — anything that sits at the top of
        # a center-screen dialog and has a 確認 button at the bottom is
        # safe to dismiss.  Original detection was "通知" only, which
        # missed sign-in/reward popups that use different headers:
        #   - 社團簽到獎勵 (club daily sign-in, observed in
        #     run_20260513_211309 t74: bot exited Club without clicking
        #     the popup's 確認, leaving x10 AP rewards "stuck" because
        #     the popup also describes the reward going to mail).
        #   - 提示 (generic hint/tip dialog)
        #   - 系統公告 (system announcement, rare)
        notification = screen.find_any_text(
            ["通知", "提示", "簽到獎勵", "签到奖励",
             "系統公告", "系统公告", "公告"],
            region=(0.20, 0.10, 0.80, 0.32), min_conf=0.55
        )
        if notification:
            # Check if this notification MUST be confirmed (not canceled):
            # 1. Cafe invite: "邀.*咖啡"
            # 2. Game update download: title "更新通知" or body keywords
            #    Actual body example: "需要下載遊戲所需的檔案6.27GB"
            invite_hint = screen.find_text_one(
                r"邀.*咖啡", region=screen.CENTER, min_conf=0.5
            )
            # Detect "更新" in the title text itself (OCR: "更新通知")
            update_title = "更新" in (notification.text or "")
            update_hint = screen.find_any_text(
                ["下載必要", "下载必要", "更新資源", "更新资源",
                 "下載資源", "下载资源", "下載內容", "下载内容",
                 "需要下載", "需要下载", "遊戲所需", "游戏所需",
                 "下載遊戲", "下载游戏", "檔案"],
                region=screen.CENTER, min_conf=0.45,
            )
            # Bounty/commission/event sweep confirm:
            #   "要使用X AP掃蕩Y次嗎？"          (commission AP sweep)
            #   "要使用6通票券6次？"              (bounty ticket sweep)
            #   "要使用600AP30次？"               (event activity sweep)
            # OCR drops spaces between "使用" and digits, so "使用AP"
            # doesn't match the event variant. Use broader anchors:
            # `要使用` is distinctive to BA sweep confirms; `次？` +
            # digit-before-AP catch the same pattern.
            sweep_hint = screen.find_any_text(
                ["掃蕩", "扫荡", "捅荡", "捅蕩",
                 "使用AP", "使用 AP", "要使用",
                 "次？", "次?", "次嗎", "次吗",
                 "通票券", "通缉票券", "通緝票券", "票券",
                 # Sweep-reset prompts: "掃蕩次數設定為2次以上 / 是否重置 / 次數設定 / 重置次數"
                 # OCR sometimes drops the leading 掃蕩, so use the
                 # post-prefix anchors as alternates (run_20260504_224706
                 # t322 saw "次數設定為2次以上" without the 掃蕩 prefix).
                 "重置", "次數設定", "次數", "次数设定", "次数",
                 "次以上", "次以下", "若想開始", "若想开始"],
                region=screen.CENTER, min_conf=0.45,
            )
            # Friend-cafe visit confirm ("要訪問好友的咖啡廳嗎？") must NEVER be
            # confirmed — we want to stay in our own cafe. Force cancel.
            friend_hint = screen.find_any_text(
                ["訪問好友", "访问好友", "朋友的咖啡廳", "朋友的咖啡厅",
                 "前往好友", "前往訪問", "前往访问", "要訪問", "要访问",
                 "指定訪問", "指定访问", "隨機訪問", "随机访问"],
                region=screen.CENTER, min_conf=0.5,
            )
            # Resume-battle prompt: "有進行中的戰鬥。是否繼續活動關卡？"
            # appears at login after a prior run abandoned mid-battle.
            # Buttons are 中斷 (abort) and 繼續 (resume).  MUST click 繼續
            # to resume the orphaned battle; clicking 中斷 forfeits it.
            # OCR routinely garbles 戰鬥→門 and drops 繼 from 繼續, so
            # match on the most stable substrings only.
            resume_battle_hint = screen.find_any_text(
                ["進行中", "进行中", "活動關卡", "活动关卡",
                 "繼續活動", "继续活动", "續活動"],
                region=screen.CENTER, min_conf=0.5,
            )
            # Exit-game prompt: "是否結束？" / "是否结束？" — appears when
            # ESC is pressed on lobby OR when over-ESC'ing inside a sub-
            # screen pops up the "exit?" confirmation.  Clicking 確認
            # WILL CLOSE THE GAME or back out of the current sub-screen
            # to lobby.  User feedback (2026-05-15): "经常点进一个地方
            # 然后就喜欢退回主界面" — bot was blindly confirming exit
            # prompts.  Force CANCEL so the bot stays where it is.
            exit_hint = screen.find_any_text(
                ["是否結束", "是否结束", "結束遊戲", "结束游戏",
                 "是否退出", "退出遊戲", "退出游戏"],
                region=screen.CENTER, min_conf=0.5,
            )
            must_confirm = (invite_hint or update_title or update_hint
                            or sweep_hint or resume_battle_hint) and not (friend_hint or exit_hint)
            must_cancel = bool(friend_hint or exit_hint)

            confirm_btn = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "繼續", "继续",
                 "確", "确", "續", "OK"],   # `續` covers OCR-dropped 繼
                region=(0.30, 0.55, 0.74, 0.82),
                min_conf=0.40,
            )
            cancel_btn = screen.find_any_text(
                ["取消", "中斷", "中断"],
                region=(0.28, 0.60, 0.56, 0.82),
                min_conf=0.55,
            )

            # Must-cancel notifications: NEVER confirm.
            #   friend_hint → don't visit friend cafe
            #   exit_hint   → don't exit the current screen / game
            if must_cancel:
                tag = "exit-prompt" if exit_hint else "friend-cafe"
                if cancel_btn:
                    self.log(f"notification popup ({tag}): clicking cancel")
                    return action_click_box(cancel_btn, f"cancel {tag}")
                # OCR missed the button — hardcoded cancel position (left of confirm)
                self.log(f"notification popup ({tag}): cancel fallback click")
                return action_click(0.402, 0.701, f"cancel {tag} fallback")

            # Must-confirm notifications: always click 確認 — UNLESS the
            # confirm button is grayed out (insufficient resources, e.g.
            # shop purchase without enough currency). In that case clicking
            # does nothing → infinite loop. Sample the button color; if
            # grayed, click cancel and give up.
            if must_confirm and confirm_btn:
                if screen.is_button_grey(confirm_btn.cx, confirm_btn.cy):
                    self.log(
                        "notification popup: confirm button is grayed "
                        "(insufficient / disabled) — clicking cancel instead"
                    )
                    if cancel_btn:
                        return action_click_box(cancel_btn, "grayed confirm → cancel")
                    return action_click(0.402, 0.701, "grayed confirm → cancel (fallback)")
                tag = ("invite" if invite_hint
                       else "resume-battle" if resume_battle_hint
                       else "update")
                self.log(f"notification popup ({tag}): clicking confirm")
                return action_click_box(confirm_btn, f"confirm {tag} notification")
            if must_confirm and cancel_btn:
                # OCR missed confirm but found cancel → use hardcoded confirm position
                self.log("notification popup (must-confirm): confirm fallback click")
                return action_click(0.598, 0.701, "confirm notification fallback")

            # Regular notification: prefer cancel to dismiss
            if cancel_btn:
                self.log("notification popup: clicking cancel to close")
                return action_click_box(cancel_btn, "dismiss notification popup (cancel)")

            if confirm_btn:
                self.log("notification popup: clicking confirm")
                return action_click_box(confirm_btn, "dismiss notification popup (confirm)")

            if must_confirm:
                self.log("notification popup (must-confirm): no buttons, fallback click confirm area")
                return action_click(0.598, 0.701, "confirm notification (no btn fallback)")

            # No buttons detected by OCR — try template-match the cyan
            # 確認 button before falling back to a hardcoded click.
            # The inventory-full popup ("因持有數限制 / 請確認信箱")
            # is a TALL popup whose 確認 button sits at y≈0.78, lower
            # than the old hardcoded (0.50, 0.64) fallback which fell
            # into the popup body and did nothing — burned the entire
            # skill's tick budget (DailyTasks timeout, run_20260515_
            # 230517 t890+).
            try:
                tpl = screen.find_template_one("task_fight_confirm", min_conf=0.65)
                if tpl is None:
                    tpl = screen.find_template_one("activity_fight_confirm", min_conf=0.65)
                if tpl is not None:
                    self.log(
                        f"notification popup: OCR missed buttons, "
                        f"template-matched 確認 at ({tpl.cx:.2f},{tpl.cy:.2f})"
                    )
                    return action_click(tpl.cx, tpl.cy, "dismiss notification popup (template match)")
            except Exception:
                pass
            # Last-resort hardcoded fallback: BA's tall popups (inventory
            # full, daily reward summary) put 確認 at center-bottom around
            # (0.50, 0.78).  Shorter popups put it at (0.50, 0.64).  We
            # try the BOTTOM coord first since the tall popup is the
            # newer / more common failure case.
            self.log("notification popup: no buttons detected, clicking confirm-area fallback (0.50, 0.78)")
            return action_click(0.50, 0.78, "dismiss notification popup (bottom fallback)")

        # Confirm dialogs: full two-char buttons (確認/確定)
        confirm = screen.find_any_text(
            ["確認", "确认", "確定", "确定", "確", "确"],
            region=screen.CENTER, min_conf=0.8
        )
        if confirm:
            cancel = screen.find_any_text(
                ["取消"],
                region=screen.CENTER, min_conf=0.8
            )
            if cancel:
                return None
            self.log(f"confirm popup: '{confirm.text}'")
            return action_click_box(confirm, f"confirm popup '{confirm.text}'")

        # Single-char confirm button (確/确) — game sometimes renders
        # only one character for the confirm button (e.g. inventory-full popup).
        # Only click if there's supporting popup body text nearby.
        single_confirm = screen.find_any_text(
            ["確", "确"],
            region=(0.35, 0.65, 0.65, 0.80), min_conf=0.9
        )
        if single_confirm:
            popup_body = screen.find_any_text(
                ["背包已满", "背包已滿", "整理背包", "道具背包",
                 "已達上限", "已达上限", "空間不足", "空间不足",
                 "超過上限", "超过上限"],
                region=screen.CENTER, min_conf=0.6
            )
            if popup_body:
                self.log(f"single-char confirm popup: '{single_confirm.text}' (body: '{popup_body.text}')")
                return action_click_box(single_confirm, f"confirm popup '{single_confirm.text}'")

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
