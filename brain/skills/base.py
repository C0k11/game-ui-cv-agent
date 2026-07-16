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
    # Which detector produced this box: "ui" / "avatar" / "battle" / "cafe".
    # Lets skills filter by source model (e.g. arena opponent heads come from
    # the avatar model, not the ui model) without guessing from cls_name.
    model_tag: str = ""

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
    # Raw BGR frame (numpy) kept for on-demand digit-OCR cropping. Not always
    # populated (e.g. read_screen from a path skips it); digit-OCR helpers must
    # null-check. Excluded from any serialization (it's a big array).
    frame: Any = field(default=None, repr=False, compare=False)
    # 高频 DXcam 线程的最新检出(2026-07-11 工业级链路): 帧龄≤0.5s@2FPS,
    # 主 tick 帧龄 ~2.2s 对轮播类时敏目标必错位 — skill 做"有目标就点"
    # 判定时优先读这里。None = 线程未跑/未接线, 调用方必须 null-check。
    fresh_boxes: Any = field(default=None, repr=False, compare=False)
    fresh_frame: Any = field(default=None, repr=False, compare=False)
    fresh_ts: float = 0.0

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
        """Detect if we're on the main lobby screen — pure YOLO (no OCR).

        Lobby shows the 8 bottom-nav entry icons (咖啡厅入口/课程表入口/...).
        Seeing >=2 of those cls = lobby. v3 detects all 8 at 0.91-0.99 on
        real frames, so 2 is a safe floor that still rejects sub-pages
        (which only carry a lingering nav bar partially, if at all).
        """
        from brain.skills.ui_classes import LOBBY_NAV_ICONS
        want = set(LOBBY_NAV_ICONS)
        hits = sum(
            1 for b in (self.yolo_boxes or [])
            if b.cls_name in want and b.confidence >= 0.30
        )
        return hits >= 2

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

        PURE-YOLO (2026-05-29): maps DOT_RED / DOT_YELLOW detections to the
        nearest nav-entry cls by horizontal proximity (a badge sits just
        above-right of its icon). Only entries WITH a nearby dot are
        returned ('red'/'yellow'); entries with no dot are OMITTED rather
        than marked 'none', so the badge-skip optimiser treats them as
        'unknown' → runs the skill. That's the safe choice during bring-up
        (we never wrongly skip real work). The 'none'-marking optimisation
        comes back once the YOLO flow is verified end-to-end.
        """
        boxes = [b for b in (self.yolo_boxes or []) if b.confidence >= 0.30]
        if not boxes:
            return {}
        from brain.skills import ui_classes as UC
        # nav-entry cls -> stable english key (matches _SKILL_BADGE_MAP)
        entry_key = {
            UC.NAV_CAFE: "cafe", UC.NAV_SCHEDULE: "schedule",
            UC.NAV_STUDENT: "student", UC.NAV_EDIT: "edit",
            UC.NAV_SOCIAL: "social", UC.NAV_CRAFT: "craft",
            UC.NAV_SHOP: "shop", UC.NAV_RECRUIT: "recruit",
            UC.NAV_MAIL: "mail", UC.NAV_TASKS: "campaign_nav",
        }
        entries = [(entry_key[b.cls_name], b)
                   for b in boxes if b.cls_name in entry_key]
        reds = [b for b in boxes if b.cls_name == UC.DOT_RED]
        yellows = [b for b in boxes if b.cls_name == UC.DOT_YELLOW]
        results: Dict[str, str] = {}
        for key, eb in entries:
            # campaign tile (任务大厅入口) is large — its dots sit well above
            # the tile centre; bottom-nav dots sit just above their icon.
            dx = 0.06 if key == "campaign_nav" else 0.05
            dy_above = 0.30 if key == "campaign_nav" else 0.03
            def near(d):
                return abs(d.cx - eb.cx) <= dx and (eb.cy - dy_above) <= d.cy <= eb.cy + 0.02
            if any(near(d) for d in reds):
                results[key] = "red"
            elif any(near(d) for d in yellows):
                results[key] = "yellow"
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
        """Detect in-game loading spinner — pure YOLO (加载中 cls, no OCR).

        KNOWN GAP (pure-YOLO bring-up): the startup download/verify/reset
        screens ("Now Loading", "正在更新", "驗證下載檔案中") have NO YOLO
        cls yet, so with OCR disabled they are not detected here. In-game
        transient scene loads carry the 加载中 spinner (cls 22, 45f trained),
        which this catches. If a startup/download screen ever stalls the
        pipeline, the fix is to train a cls for it (NOT re-enable OCR here).
        """
        for b in (self.yolo_boxes or []):
            if b.cls_name == "加载中" and b.confidence >= 0.35:
                return True
        return False

    def is_dialog(self) -> bool:
        """Detect a popup/dialog — pure YOLO (确认键 button present)."""
        for b in (self.yolo_boxes or []):
            if b.cls_name == "确认键" and b.confidence >= 0.30:
                return True
        return False


# ── Dot color posterior (2026-07-08) ────────────────────────────────────
# v12 红点/黄点位置先验强: 社交入口的蓝「+」badge 被标红点 conf0.72, 已爬进
# 真点 conf 区间(0.85-0.92, 闪烁真点低至 0.69) → conf 阈值不可分。但点类是
# 纯色小圆, 颜色占比完美分离(同帧实测: 真红点 red%=0.67-0.75 / 真黄点
# yellow%=0.72-0.74 / 假点(蓝+) 两者=0.00)。所有 dot 判定过此闸。

def classify_dot_color(frame, x1: float, y1: float, x2: float, y2: float
                       ) -> Optional[str]:
    """HSV posterior for a dot bbox (normalized coords). Returns '红点' /
    '黄点' / None (neither red nor yellow = position-prior false fire).
    frame None → caller should treat as "cannot verify" (pass-through)."""
    if frame is None:
        return None
    try:
        import cv2
        import numpy as np
        h, w = frame.shape[:2]
        px1, py1 = max(0, int(x1 * w)), max(0, int(y1 * h))
        px2, py2 = min(w, int(x2 * w)), min(h, int(y2 * h))
        crop = frame[py1:py2, px1:px2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        vivid = (sat > 120) & (val > 120)
        red = float(np.mean(((hue < 10) | (hue > 170)) & vivid))
        yel = float(np.mean((hue >= 15) & (hue <= 35) & vivid))
        if max(red, yel) < 0.25:
            return None
        return "红点" if red >= yel else "黄点"
    except Exception:
        return None


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


def action_swipe_tap(fx: float, fy: float, tx: float, ty: float,
                     cx: float, cy: float, duration_ms: int = 150,
                     reason: str = "") -> Dict[str, Any]:
    """原子 swipe→tap(一条 adb shell 连发, 间隔<0.5s)。

    对自动轮播 UI: swipe 把轮播拉停数秒并翻到确定项, tap 在静止期内落点
    无时序竞争(hub banner 帧龄 2.2s vs 项周期 2.6s 的错位问题唯一硬解)。
    """
    return {"action": "swipe_tap", "from": [fx, fy], "to": [tx, ty],
            "target": [cx, cy], "duration_ms": duration_ms, "reason": reason}


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

    # ── Dot-driven skip check (overridden by daily-harvest skills) ──
    # When pipeline is about to start this skill, it first calls should_run()
    # on the current ScreenState. Default = True (always run). Daily-harvest
    # skills (cafe / mail / schedule / club / daily_tasks / event_activity)
    # override this to look for their associated red/yellow dot on the lobby
    # screen and return False if there's no work to do.
    #
    # Battle / sweep / arena / bounty skills DO NOT override — they always
    # run when the user enables them in skill_order.
    def should_run(self, screen: ScreenState) -> bool:
        """Return False to make pipeline skip this skill entirely.
        Called once at skill entry before tick(). Default = always run."""
        return True

    def hall_tile_dot(self, screen: ScreenState, tile_cls: str,
                      *, dot_classes: Tuple[str, ...] = ("红点", "黄点")
                      ) -> Optional[bool]:
        """Task-hall per-activity work check (user iron rule 2026-06-11: the
        LOBBY entry dot must never gate these skills — enter the hall and scan
        each activity's own dot).

        Returns None when the tile isn't visible (not in the hall — can't
        decide), True when a red/yellow dot sits at the tile's TOP-RIGHT
        (live-measured 2026-06-11: 悬赏 tile (0.561,0.550) → dot (0.634,0.512)),
        False when the tile is visible with no dot (= no work today)."""
        tile = self.find_cls(screen, tile_cls, conf=0.40)
        if tile is None:
            return None
        region = (tile.x1, tile.y1 - 0.08, tile.x2 + 0.11, tile.y2)
        return self.dot_in_region(screen, region, dot_classes=dot_classes)

    def dot_in_region(self, screen: ScreenState,
                       region: Tuple[float, float, float, float],
                       *, dot_classes: Tuple[str, ...] = ("红点", "黄点"),
                       min_conf: float = 0.35) -> bool:
        """Helper for should_run: is there a red/yellow dot inside this
        normalized rect (x1, y1, x2, y2)?  Uses ui_yolo26m_v1 detections
        already on screen.yolo_boxes (no extra inference)."""
        if not screen.yolo_boxes:
            return False
        x1, y1, x2, y2 = region
        for b in screen.yolo_boxes:
            if b.confidence < min_conf:
                continue
            if b.cls_name not in dot_classes:
                continue
            cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                # 颜色后验 (2026-07-08): 位置先验假点(社交蓝+ conf0.72)过滤。
                # frame 缺失时放行(验色只做增量过滤, 不引入新失败模式)。
                if screen.frame is not None:
                    seen = classify_dot_color(
                        screen.frame, b.x1, b.y1, b.x2, b.y2)
                    if seen is None or seen not in dot_classes:
                        continue
                return True
        return False

    def dot_on_entry(self, screen: ScreenState,
                      entry_class_names,
                      *, dot_classes: Tuple[str, ...] = ("红点", "黄点"),
                      min_conf_entry: float = 0.4,
                      min_conf_dot: float = 0.35) -> bool:
        """Stronger should_run helper: are any of the listed entry icons
        currently on screen AND covered by a red/yellow dot?

        Returns True when:
          (a) Entry icon NOT visible (we're probably not on lobby, can't
              decide here — defer to skill's own logic), OR
          (b) Entry icon visible AND a red/yellow dot center sits inside it.
        Returns False ONLY when entry IS visible but NO dot covers it
        (clean "no work to do" signal → skill can be skipped).
        """
        if not screen.yolo_boxes:
            return True  # detector disabled, can't decide → pass
        targets = list(entry_class_names) if isinstance(entry_class_names, (list, tuple)) else [entry_class_names]
        target_set = set(targets)
        target_lower = [t.lower() for t in targets]
        entries = []
        dots = []
        for b in screen.yolo_boxes:
            cn = b.cls_name
            cn_low = (cn or "").lower()
            if b.confidence >= min_conf_entry and (
                cn in target_set or any(t in cn_low or cn_low in t for t in target_lower)
            ):
                entries.append(b)
            elif b.confidence >= min_conf_dot and cn in dot_classes:
                dots.append(b)
        if not entries:
            return True  # not on lobby (entry not visible) → defer
        # Margin: red/yellow badges sit at the entry's TOP-RIGHT corner, often
        # a hair OUTSIDE the icon bbox. A strict inside-bbox test false-skips
        # (live 2026-06-02: cafe/schedule yellow dots present but skipped). Allow
        # a small expansion so a badge near the entry counts.
        mx, my = 0.03, 0.06
        for d in dots:
            # 颜色后验 (2026-07-08): 蓝+/灰位假点过滤 (club 假重进同根)。
            if screen.frame is not None:
                seen = classify_dot_color(screen.frame, d.x1, d.y1, d.x2, d.y2)
                if seen is None or seen not in dot_classes:
                    continue
            dcx = (d.x1 + d.x2) / 2
            dcy = (d.y1 + d.y2) / 2
            for e in entries:
                if (e.x1 - mx) <= dcx <= (e.x2 + mx) and (e.y1 - my) <= dcy <= (e.y2 + my):
                    return True
        return False  # entry visible, no dot → no work

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

    # ════════════════════════════════════════════════════════════════
    # YOLO-only UI resolution (canonical — use these, NOT find_ui/OCR).
    # cls names come from brain/skills/ui_classes.py. Exact-match only
    # (ScreenState.find_yolo uses substring, which mis-matches e.g.
    # "领取_黄" vs "全部领取_黄"). No OCR fallback by design: if YOLO
    # can't see a cls, surface the gap (log+wait) instead of hiding it.
    # ════════════════════════════════════════════════════════════════
    def find_cls(
        self,
        screen: ScreenState,
        cls_names,
        *,
        conf: float = 0.30,
        region: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[YoloBox]:
        """Return highest-conf YOLO box whose cls_name EXACTLY equals one of
        cls_names (str or list), optionally constrained to a region. None if
        no match. This is the primary click-target resolver for all skills."""
        if isinstance(cls_names, str):
            cls_names = [cls_names]
        want = set(cls_names)
        best = None
        for b in (screen.yolo_boxes or []):
            if b.confidence < conf:
                continue
            if b.cls_name not in want:
                continue
            if region is not None:
                if not (region[0] <= b.cx <= region[2] and region[1] <= b.cy <= region[3]):
                    continue
            if best is None or b.confidence > best.confidence:
                best = b
        return best

    def read_count(self, screen: ScreenState, icon_cls, *, conf: float = 0.30,
                   side: str = "right", span: float = 0.10, pad: float = 0.005):
        """DIGIT-ONLY read of the number next to a currency/count icon.

        YOLO locates the icon (e.g. TICKET_BOUNTY, TOPBAR_AP); we crop the
        digit strip beside it and OCR only the digits. This is the ONLY place
        OCR is used (per spec: YOLO for everything, OCR for digits) and works
        regardless of the global _OCR_ENABLED nav-OCR switch.

        Args:
            icon_cls: cls name(s) of the icon box to anchor on.
            side: which side the number sits — "right" (top-bar currencies,
                  tickets) or "left".
            span: width of the digit strip as a fraction of screen width.
            pad: gap between icon edge and strip start (fraction of width).
        Returns:
            (current, total) from pipeline.parse_count — total may be None;
            or None if the icon isn't found / nothing read. Caller decides.
        """
        box = self.find_cls(screen, icon_cls, conf=conf)
        if box is None or screen.frame is None:
            return None
        # vertical band aligned to the icon (a touch taller for glyph margin)
        bh = box.y2 - box.y1
        y1 = max(0.0, box.y1 - bh * 0.25)
        y2 = min(1.0, box.y2 + bh * 0.25)
        if side == "right":
            x1 = min(1.0, box.x2 + pad)
            x2 = min(1.0, x1 + span)
        else:  # left
            x2 = max(0.0, box.x1 - pad)
            x1 = max(0.0, x2 - span)
        try:
            from brain.pipeline import run_digit_ocr, parse_count
        except Exception:
            return None
        raw = run_digit_ocr(screen.frame, (x1, y1, x2, y2))
        result = parse_count(raw)
        if result is not None:
            self.log(f"read_count({icon_cls})={result} (raw {raw!r})")
        return result

    def find_all_cls(
        self,
        screen: ScreenState,
        cls_names,
        *,
        conf: float = 0.30,
        region: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[YoloBox]:
        """Like find_cls but returns ALL exact matches sorted by conf desc."""
        if isinstance(cls_names, str):
            cls_names = [cls_names]
        want = set(cls_names)
        hits = []
        for b in (screen.yolo_boxes or []):
            if b.confidence < conf:
                continue
            if b.cls_name not in want:
                continue
            if region is not None:
                if not (region[0] <= b.cx <= region[2] and region[1] <= b.cy <= region[3]):
                    continue
            hits.append(b)
        return sorted(hits, key=lambda b: -b.confidence)

    def click_cls(
        self,
        screen: ScreenState,
        cls_names,
        reason: str,
        *,
        conf: float = 0.30,
        region: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find a cls by exact name + click it. Returns action dict or None
        (caller decides what to do on miss — usually log+wait)."""
        box = self.find_cls(screen, cls_names, conf=conf, region=region)
        if box is None:
            return None
        return action_click_box(box, f"{reason} (YOLO {box.cls_name} {box.confidence:.2f})")

    def nav_home(self, screen: ScreenState, reason: str = "回大厅") -> Dict[str, Any]:
        """Navigate toward the lobby using ONLY in-game buttons — NEVER a blind
        ESC / back keyevent (user 2026-06-15 iron rule: 反复 ESC-spam recovery
        多次触发 Unity ANR「Blue Archive没有响应」, freezing the game; "只点基于游戏
        内的返回大厅还是叉叉"). Preference: 回大厅按钮(home → lobby directly) →
        弹窗叉叉(close popup) → 返回键(back one screen). If NONE is detected this
        frame, WAIT — do not blind-tap a guessed position and do not ESC; the
        caller's own _phase_ticks timeout ends the skill cleanly if truly stuck.
        """
        from brain.skills import ui_classes as UC
        home = self.find_cls(screen, UC.BTN_HOME, conf=0.30)
        if home is not None:
            return action_click_box(home, f"{reason}: 回大厅按钮")
        x = self.find_cls(screen, UC.BTN_CLOSE_X, conf=0.30)
        if x is not None:
            return action_click_box(x, f"{reason}: 弹窗叉叉")
        back = self.find_cls(screen, UC.BTN_BACK, conf=0.30)
        if back is not None:
            return action_click_box(back, f"{reason}: 返回键")
        return action_wait(450, f"{reason}: 无 home/X/返回键 → 等待 (绝不瞎按 ESC)")

    def detect_screen_yolo(self, screen: ScreenState) -> Optional[str]:
        """Detect current page from YOLO cls signatures (no OCR).

        Returns page name (Lobby/Mail/Schedule/Cafe/Craft/MomoTalk/Story/
        Battle) or None if no signature matches. See ui_classes.PAGE_SIGNATURES.
        Lobby is checked last so a sub-page's own cls wins over a lingering
        nav-bar (nav icons are visible inside many pages)."""
        from brain.skills import ui_classes as UC
        # Non-lobby pages first (more specific)
        for page, (cls_list, min_n) in UC.PAGE_SIGNATURES.items():
            if page == "Lobby":
                continue
            n = sum(1 for c in cls_list if self.find_cls(screen, c, conf=0.30) is not None)
            if n >= min_n:
                return page
        # Lobby last
        lobby_cls, lobby_min = UC.PAGE_SIGNATURES["Lobby"]
        n = sum(1 for c in lobby_cls if self.find_cls(screen, c, conf=0.30) is not None)
        if n >= lobby_min:
            return "Lobby"
        return None

    def find_ui(
        self,
        screen: ScreenState,
        yolo_classes: List[str],
        ocr_texts: Optional[List[str]] = None,
        *,
        yolo_conf: float = 0.40,
        ocr_conf: float = 0.6,
        region: Optional[Tuple[float, float, float, float]] = None,
    ):
        """[LEGACY] Find a UI element using YOLO first, OCR fallback.

        DEPRECATED for new code — use find_cls/click_cls (YOLO-only). Kept
        for skills not yet migrated off OCR fallback.

        This is the canonical button-finder for sub-skills. Pass a list of
        YOLO cls_name strings (from ui_v1 model — see data/yolo_datasets/
        ui_v1_auto/classes.txt names referenced by trajectory yolo_boxes):
        e.g. ['咖啡厅入口', '弹窗叉叉', '全部领取_黄', '邀请键', '确认键'].

        Falls back to OCR text-matching when YOLO misses, so legacy skills
        keep working. Returns whichever box type matched (YoloBox or OcrBox)
        — both have .x1/.y1/.x2/.y2 + .cx/.cy via cx_ny accessors so the
        caller's action_click_box() doesn't care which it got.

        Args:
            yolo_classes: list of cls_name to try first (exact or substring).
            ocr_texts: OCR substrings to try if YOLO misses. None = no fallback.
            yolo_conf / ocr_conf: per-method confidence floors.
            region: (x1,y1,x2,y2) normalized 0..1 to constrain YOLO+OCR hits.
        """
        # YOLO first (already in screen.yolo_boxes — no extra inference).
        if yolo_classes and screen.yolo_boxes:
            name_set = set(yolo_classes)
            name_lower = [str(n).lower() for n in yolo_classes]
            hits = []
            for b in screen.yolo_boxes:
                if b.confidence < yolo_conf:
                    continue
                # region clip
                if region is not None:
                    cx = (b.x1 + b.x2) / 2
                    cy = (b.y1 + b.y2) / 2
                    if not (region[0] <= cx <= region[2] and region[1] <= cy <= region[3]):
                        continue
                # exact match
                if b.cls_name in name_set:
                    hits.append(b)
                    continue
                # substring fallback
                bn = (b.cls_name or "").lower()
                if any(q in bn or bn in q for q in name_lower):
                    hits.append(b)
            if hits:
                return max(hits, key=lambda x: x.confidence)
        # OCR fallback
        if ocr_texts:
            return screen.find_any_text(ocr_texts, min_conf=ocr_conf, region=region)
        return None

    def find_claim_all_button(
        self,
        screen: ScreenState,
        *,
        min_conf: float = 0.6,
        region: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[YoloBox]:
        """Locate a bulk-claim button (全部領取 / 一次領取 / 一鍵領取) —
        pure YOLO cls, no OCR text matching (2026-07 YOLO migration).

        Yellow (actionable) cls first so callers that click get a live
        button; grey (already-claimed) is returned as a fallback because
        legacy callers also used this as a "button exists here" probe.

        NOTE: min_conf was the legacy OCR threshold — kept in the
        signature for caller compatibility, NOT applied to YOLO
        (cls conf floor 0.35, same as the other cls helpers).
        """
        from brain.skills import ui_classes as UC
        box = self.find_cls(
            screen,
            [UC.CLAIM_ALL_YELLOW, UC.CLAIM_ONCE_YELLOW],  # 107 / 417
            conf=0.35, region=region,
        )
        if box is not None:
            return box
        return self.find_cls(
            screen,
            [UC.CLAIM_ALL_GREY, UC.CLAIM_ONCE_GREY,
             UC.CLAIM_ONEKEY_GREY],                       # 413 / 416 / 415
            conf=0.35, region=region,
        )

    def find_single_claim_button(
        self,
        screen: ScreenState,
        *,
        min_conf: float = 0.7,
        region: Optional[Tuple[float, float, float, float]] = (0.6, 0.1, 1.0, 0.9),
    ) -> Optional[YoloBox]:
        """Locate an individual per-row claim button — pure YOLO cls
        (default region: right side where per-item rewards sit).

        cls: 领取_黄 (106, single-row actionable) / 领取奖励_黄 (89,
        領取獎勵-style rows e.g. arena). Bulk variants (全部领取_黄 107 /
        一次领取黄色 417 / grey states) are DISTINCT cls, so exact-name
        matching already excludes them — the OCR-era substring bug (領取
        also matching 全部領取; run 2026-05-13 ~21:25 clicked 全部領取 at
        (0.90,0.93) instead of the row's 領取 at (0.55,0.93)) cannot
        recur by name. Belt-and-braces: drop any candidate whose center
        sits inside a detected bulk-claim box, guarding against the model
        double-firing 领取_黄 on the 全部領取 button's 领取 glyphs.

        Only yellow (actionable) states are returned — grey rows are not
        click targets. NOTE: min_conf was the legacy OCR threshold —
        kept for signature compatibility, NOT applied to YOLO (floor 0.35).
        """
        from brain.skills import ui_classes as UC
        candidates = self.find_all_cls(
            screen,
            [UC.CLAIM_YELLOW, UC.CLAIM_REWARD_YELLOW],    # 106 / 89
            conf=0.35, region=region,
        )
        if not candidates:
            return None
        # Bulk boxes fetched WITHOUT region: the bulk button usually sits
        # outside the per-row region, but a stray 领取_黄 double-detection
        # on top of it would be inside — compare against all of them.
        bulk = self.find_all_cls(
            screen,
            [UC.CLAIM_ALL_YELLOW, UC.CLAIM_ALL_GREY, UC.CLAIM_ONCE_YELLOW,
             UC.CLAIM_ONCE_GREY, UC.CLAIM_ONEKEY_GREY],   # 107/413/417/416/415
            conf=0.30,
        )
        for cand in candidates:  # find_all_cls: conf-desc order
            if any(b.x1 <= cand.cx <= b.x2 and b.y1 <= cand.cy <= b.y2
                   for b in bulk):
                continue
            return cand
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
        """Detect current screen — pure YOLO (OCR-free per 2026-05-29 spec).

        Delegates to detect_screen_yolo (PAGE_SIGNATURES). Returns the same
        legacy vocabulary skills already branch on: Lobby / Mail / Schedule /
        Cafe / Craft / MomoTalk / Story / Battle / PVP / Bounty / Mission.
        Returns None when no signature matches (caller decides: usually
        wait or ESC). DailyTasks has no YOLO signature yet — returns None
        there (skills that special-cased it now just fall through, harmless).
        """
        return self.detect_screen_yolo(screen)

    def _detect_current_screen_ocr(self, screen: ScreenState) -> Optional[str]:
        """[DEAD — OCR disabled] Legacy OCR header detection, kept for the
        digit-OCR re-enable phase. Not called while _OCR_ENABLED is False."""
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

        # Fallback: campaign hub / stage-select detection via YOLO cls
        # (2026-07 YOLO migration: OCR grid markers 懸賞通緝/Area N/Normal/
        # 入場 replaced by ui-model hub tiles + stage-select cls). >=2
        # distinct cls required so one tile leaking into another screen
        # can't misfire (mirrors PAGE_SIGNATURES["Mission"]). Keep this
        # AFTER specific header checks to avoid classifying total assault/
        # pass as Mission.
        from brain.skills import ui_classes as UC
        _MISSION_CLS = (
            UC.HUB_CAMPAIGN,         # 67 任务关卡推图
            UC.HUB_STORY,            # 68 剧情
            UC.HUB_BOUNTY,           # 69 悬赏通缉
            UC.HUB_SPECIAL,          # 70 特殊任务
            UC.HUB_SCHOOL_EXCHANGE,  # 71 学院交流会
            UC.HUB_ARENA,            # 75 战术大赛
            "制约解除决战",           # 76 (no ui_classes constant yet)
            UC.STAGE_ENTER,          # 79 入场键 (stage-select right panel)
            UC.STAGE_NORMAL_SEL,     # 80 普通关卡选中
            UC.STAGE_HARD,           # 81 困难关卡
            UC.STAGE_HARD_SEL,       # 419 (Hard-tab-active variant of 80/81)
            UC.STAGE_NORMAL,         # 420
        )
        n_hits = sum(
            1 for c in _MISSION_CLS
            if self.find_cls(screen, c, conf=0.30) is not None
        )
        if n_hits >= 2:
            return "Mission"
        return None

    def _handle_common_popups(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Handle common popups that can appear in any skill.

        Returns an action if a popup was handled, None otherwise.
        """
        # Bond level-up screen (羈絆升級！) — full-screen transition that
        # any affinity-earning skill can trigger (cafe headpat, schedule
        # lesson, club AP claim, event battles, etc.).  Tap anywhere
        # advances.  Run_20260516_234050 t232: bot got stuck here for
        # 29 ticks waiting because no skill-specific handler caught it.
        # 2026-07-16 纯 cls 化: 词表有专类 398羁绊升级/399地区升级,
        # OCR 文字匹配(羈絆升級/治癒力 fallback)全删。
        bond_screen = self.find_cls(
            screen, ["羁绊升级", "地区升级"], conf=0.35)
        if bond_screen is not None:
            self.log(f"bond level-up screen detected "
                     f"'{bond_screen.cls_name}', tap to dismiss")
            return action_click(0.5, 0.5, "dismiss bond level-up")

        # ── 通知弹窗 (pure YOLO, 2026-07-16 重构) ────────────────────────
        # 旧版在这里用 OCR 文字给弹窗分类(通知/提示标题 + 邀.*咖啡=确认 /
        # 更新通知+下載=确认 / 掃蕩=确认 / 訪問好友=取消 / 是否結束=取消 …),
        # _OCR_ENABLED=False 后全是死代码, 且"读文字定动作"违反感知铁律
        # (判断一律 cls)。新策略(用户 spec 2026-07-16):
        #   • 语境内的"该确认"由拥有语境的 skill 自己处理, 并且必须排在调用
        #     本 helper 之前:
        #       - 咖啡厅邀请确认 → cafe._invite stage2 / _recover_invite_overlay
        #       - 课程表报告确认 → schedule.tick PRIORITY 1 (调本 helper 前)
        #       - 扫荡确认      → 各 sweep skill 的结构闸
        #       - 强更下载确认   → pipeline._global_interceptor 启动期结构闸
        #   • 因此能落到这个通用 helper 的「确认+取消/叉」结构弹窗, 定义上就是
        #     当前 skill 没预期的通知弹窗 → 默认安全路径: 一律点取消/叉掉。
        #     绝不盲点确认(2026-06-02 买票事故根因 = 盲确认, 见 money_safety)。
        #   • 只有确认键、无取消/叉的弹窗: 这里不动 (fail-closed — 交给 skill
        #     自己的 handler / tick 预算; 启动期强更框由 interceptor 接)。
        from brain.skills import ui_classes as UC
        _POPUP_BTN_BAND = (0.20, 0.45, 0.80, 0.95)  # 居中对话框按钮带
        popup_confirm = self.find_cls(
            screen, [UC.BTN_CONFIRM, UC.BTN_CONFIRM_GREY],
            conf=0.30, region=_POPUP_BTN_BAND,
        )
        if popup_confirm is not None:
            popup_cancel = self.find_cls(
                screen, UC.BTN_CANCEL, conf=0.30, region=_POPUP_BTN_BAND
            )
            if popup_cancel is not None:
                self.log("通知弹窗(确认+取消结构) → 默认安全路径: 取消")
                return action_click_box(
                    popup_cancel, "dismiss notification popup (取消键)")
            popup_x = self.find_cls(screen, UC.BTN_CLOSE_X, conf=0.30)
            if popup_x is not None:
                self.log("通知弹窗(确认+叉结构) → 默认安全路径: 叉掉")
                return action_click_box(
                    popup_x, "dismiss notification popup (弹窗叉叉)")
            # 只有确认、无取消/叉 → fail-closed, 这里不碰。

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

        # "是否跳過" (story-skip confirm) — cls + context, no OCR text match.
        # This dialog only appears right after 跳过故事键 (cls 141) was
        # clicked, i.e. while a story scene is up.  Structural gate:
        #   1) current page signature == "Story" (剧情menu 等 cls 仍在帧上), AND
        #   2) CENTER shows BOTH 确认键 (cls 20) and 取消键 (cls 118).
        # Outside a story flow a bare confirm+cancel pair is ambiguous
        # (exit / purchase prompts) → fall through, let the branches above
        # or the owning skill decide (fail-closed).
        if self.detect_screen_yolo(screen) == "Story":
            skip_confirm = self.find_cls(
                screen, "确认键", conf=0.30, region=screen.CENTER)
            skip_cancel = self.find_cls(
                screen, "取消键", conf=0.30, region=screen.CENTER)
            if skip_confirm is not None and skip_cancel is not None:
                self.log("story skip-confirm (Story page + 确认键/取消键 cls): confirming")
                return action_click_box(skip_confirm, "confirm story skip dialog")

        return None

    def _try_go_lobby(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Try to navigate back to lobby."""
        if screen.is_lobby():
            return None
        return self.nav_home(screen, f"{self.name} go-lobby")

    # OCR-text → YOLO ui_v1 cls_name map for bottom-nav entries.
    # YOLO cls is more reliable than OCR for these icons (consistent location,
    # icon-based not text-based). Added 2026-05-28 — user asked to drive
    # click logic from UI model cls instead of OCR strings.
    _NAV_OCR_TO_YOLO: Dict[str, List[str]] = {
        "咖啡廳": ["咖啡厅入口"], "咖啡厅": ["咖啡厅入口"], "咖啡": ["咖啡厅入口"],
        "課程表": ["课程表入口"], "课程表": ["课程表入口"],
        "學生": ["学生入口"],   "学生": ["学生入口"],
        "編輯": ["编辑入口"],   "编辑": ["编辑入口"],
        "社交": ["社交入口"],   "社團": ["社交入口"], "社团": ["社交入口"],
        "製造": ["制造入口"],   "制造": ["制造入口"],
        "商店": ["商店入口"],
        "招募": ["招募入口"],
        "任務": ["任务大厅入口"], "任务": ["任务大厅入口"],
        "信箱": ["邮件箱"],     "郵箱": ["邮件箱"], "邮箱": ["邮件箱"],
        "郵件": ["邮件箱"],     "邮件": ["邮件箱"], "Mail": ["邮件箱"],
    }

    def _nav_to(self, screen: ScreenState, nav_texts: List[str]) -> Optional[Dict[str, Any]]:
        """Click a bottom nav bar button — YOLO cls ONLY, fail-closed.

        nav_texts keeps the legacy OCR vocabulary for callers; it is mapped
        to ui-model cls names via _NAV_OCR_TO_YOLO. If the target cls is not
        detected on this frame, return None and let the caller wait for the
        next frame — NO OCR fallback (iron rule: navigation by cls only).
        """
        # Gather YOLO cls candidates from the legacy text list.
        yolo_cls: List[str] = []
        for t in nav_texts:
            yolo_cls.extend(self._NAV_OCR_TO_YOLO.get(t, []))
        # Dedupe preserving order.
        seen = set()
        yolo_cls = [c for c in yolo_cls if not (c in seen or seen.add(c))]
        if yolo_cls and screen.yolo_boxes:
            hit = self.find_ui(screen, yolo_cls, None, yolo_conf=0.35)
            if hit is not None:
                self.log(f"nav YOLO click '{hit.cls_name}'")
                return action_click_box(hit, f"nav to '{hit.cls_name}'")
        return None
