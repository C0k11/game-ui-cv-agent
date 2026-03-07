"""Fast multi-scale template matching for game UI elements.

Uses OpenCV matchTemplate with TM_CCOEFF_NORMED for sub-millisecond
detection of known icons (headpat bubble, close button, etc.).

All returned coordinates are **normalized 0-1** relative to the input frame.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TemplateHit:
    """A single template match result with normalized coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str = ""

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


def _nms(hits: List[TemplateHit], iou_thresh: float = 0.3) -> List[TemplateHit]:
    """Non-maximum suppression to remove overlapping detections."""
    if not hits:
        return []
    hits = sorted(hits, key=lambda h: h.confidence, reverse=True)
    keep: List[TemplateHit] = []
    for h in hits:
        overlaps = False
        for k in keep:
            # IoU check
            ix1 = max(h.x1, k.x1)
            iy1 = max(h.y1, k.y1)
            ix2 = min(h.x2, k.x2)
            iy2 = min(h.y2, k.y2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_h = h.w * h.h
            area_k = k.w * k.h
            union = area_h + area_k - inter
            if union > 0 and inter / union > iou_thresh:
                overlaps = True
                break
        if not overlaps:
            keep.append(h)
    return keep


class TemplateMatcher:
    """Multi-scale template matcher for a single icon.

    Strategy for speed:
      - Downscale the search frame to a fixed working width (e.g. 960px)
        so matchTemplate runs on ~1/16 the pixels of a 3840px 4K frame.
      - Pre-compute a small set of scaled templates at the working resolution.
      - No alpha mask (unreliable with TM_CCOEFF_NORMED); just BGR matching.
    """

    # Working width for search — balance between speed and accuracy
    _WORK_W = 960

    def __init__(
        self,
        template_path: str,
        label: str = "template",
        scales: Optional[List[float]] = None,
        threshold: float = 0.70,
    ):
        self.label = label
        self.threshold = threshold
        self.scales = scales or [0.7, 0.85, 1.0, 1.2, 1.5]

        # Load template as BGR (ignore alpha)
        raw = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Template not found: {template_path}")
        if len(raw.shape) > 2 and raw.shape[2] == 4:
            self._template_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        else:
            self._template_bgr = raw
        self._th, self._tw = self._template_bgr.shape[:2]

    def _build_scaled_templates(self, frame_w: int) -> List[Tuple[np.ndarray, int, int]]:
        """Build scaled templates for the working resolution."""
        ratio = self._WORK_W / max(1, frame_w)
        base_tw = max(4, int(self._tw * ratio))
        base_th = max(4, int(self._th * ratio))
        result = []
        for s in self.scales:
            sw = max(4, int(base_tw * s))
            sh = max(4, int(base_th * s))
            tmpl = cv2.resize(self._template_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
            result.append((tmpl, sw, sh))
        return result

    def match(
        self,
        frame_bgr: np.ndarray,
        threshold: Optional[float] = None,
        max_hits: int = 10,
        region: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[TemplateHit]:
        """Find all instances of the template in frame_bgr.

        Downscales frame to working width for speed, then maps results
        back to normalized 0-1 coordinates.
        """
        thresh = threshold if threshold is not None else self.threshold
        fh, fw = frame_bgr.shape[:2]
        if fw <= 0 or fh <= 0:
            return []

        # Downscale frame for speed
        ratio = self._WORK_W / fw
        work_h = max(1, int(fh * ratio))
        work = cv2.resize(frame_bgr, (self._WORK_W, work_h), interpolation=cv2.INTER_AREA)

        # Crop to region if specified (on working-resolution image)
        rx1_px = ry1_px = 0
        search = work
        if region:
            rx1_px = max(0, int(region[0] * self._WORK_W))
            ry1_px = max(0, int(region[1] * work_h))
            rx2_px = min(self._WORK_W, int(region[2] * self._WORK_W))
            ry2_px = min(work_h, int(region[3] * work_h))
            if rx2_px <= rx1_px or ry2_px <= ry1_px:
                return []
            search = work[ry1_px:ry2_px, rx1_px:rx2_px]

        sh, sw = search.shape[:2]
        scaled_templates = self._build_scaled_templates(fw)
        all_hits: List[TemplateHit] = []

        for tmpl, tw, th in scaled_templates:
            if tw >= sw or th >= sh:
                continue
            try:
                result = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue

            locs = np.where(result >= thresh)
            for py, px in zip(*locs):
                conf = float(result[py, px])
                # Map from working-resolution crop back to normalized 0-1
                abs_x1 = (rx1_px + px) / self._WORK_W
                abs_y1 = (ry1_px + py) / work_h
                abs_x2 = (rx1_px + px + tw) / self._WORK_W
                abs_y2 = (ry1_px + py + th) / work_h
                all_hits.append(TemplateHit(
                    x1=abs_x1, y1=abs_y1,
                    x2=abs_x2, y2=abs_y2,
                    confidence=conf,
                    label=self.label,
                ))

        # NMS + sort + limit
        all_hits = _nms(all_hits, iou_thresh=0.3)
        all_hits.sort(key=lambda h: h.confidence, reverse=True)
        return all_hits[:max_hits]


# ── Headpat bubble detection via template matching ─────────────────────
#
# Uses Emoticon_Action.png as template, matched at small scales on a
# 960px downscaled frame.  ~30ms per frame, accurate, no false positives
# from yellow furniture/UI elements.

# ── Named template registry ─────────────────────────────────────────
#
# Each entry: (filename_in_captures, label, scales, threshold)
# Templates are lazy-loaded on first use.

_TEMPLATE_DEFS = {
    "headpat": ("Emoticon_Action.png", "headpat_bubble", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 0.78),
    "红点": ("红点.png", "红点", [0.3, 0.5, 0.7, 1.0], 0.80),
    "黄点": ("黄点.png", "黄点", [0.3, 0.5, 0.7, 1.0], 0.80),
    "内嵌公告": ("内嵌公告状态特征.png", "内嵌公告", [0.3, 0.5, 0.7, 1.0], 0.70),
    "gold": ("UI/Currency_Icon_Gold.png", "gold", [0.3, 0.5, 0.7, 1.0], 0.80),
    "gem": ("UI/Currency_Icon_Gem.png", "gem", [0.3, 0.5, 0.7, 1.0], 0.80),
    "ap": ("UI/Currency_Icon_AP.png", "ap", [0.3, 0.5, 0.7, 1.0], 0.80),
    "exp": ("UI/Currency_Icon_Exp.png", "exp", [0.3, 0.5, 0.7, 1.0], 0.80),
    "raid_ticket": ("UI/Currency_Icon_RaidTicket.png", "raid_ticket", [0.3, 0.5, 0.7, 1.0], 0.80),
    "academy_ticket": ("UI/Currency_Icon_AcademyTicket.png", "academy_ticket", [0.3, 0.5, 0.7, 1.0], 0.80),
    "arena_ticket": ("UI/Currency_Icon_ArenaTicket.png", "arena_ticket", [0.3, 0.5, 0.7, 1.0], 0.80),
    "chaser_a_ticket": ("UI/Currency_Icon_ChaserATicket.png", "chaser_a_ticket", [0.3, 0.5, 0.7, 1.0], 0.80),
    "chaser_b_ticket": ("UI/Currency_Icon_ChaserBTicket.png", "chaser_b_ticket", [0.3, 0.5, 0.7, 1.0], 0.80),
    "chaser_c_ticket": ("UI/Currency_Icon_ChaserCTicket.png", "chaser_c_ticket", [0.3, 0.5, 0.7, 1.0], 0.80),
}

_template_cache: dict = {}


def get_template_matcher(name: str) -> Optional[TemplateMatcher]:
    """Get a named template matcher. Returns None if template file not found."""
    if name in _template_cache:
        return _template_cache[name]
    defn = _TEMPLATE_DEFS.get(name)
    if defn is None:
        return None
    filename, label, scales, threshold = defn
    tmpl_path = _REPO_ROOT / "data" / "captures" / filename
    if not tmpl_path.exists():
        _template_cache[name] = None
        return None
    try:
        m = TemplateMatcher(str(tmpl_path), label=label, scales=scales, threshold=threshold)
        _template_cache[name] = m
        return m
    except Exception:
        _template_cache[name] = None
        return None


def find_headpat_bubbles(
    frame_bgr: np.ndarray,
    threshold: float = 0.78,
    region: Optional[Tuple[float, float, float, float]] = None,
) -> List[TemplateHit]:
    """Find headpat bubbles via template matching on downscaled frame."""
    matcher = get_template_matcher("headpat")
    if matcher is None:
        return []
    return matcher.match(frame_bgr, threshold=threshold, max_hits=8, region=region)


def find_template_by_name(
    name: str,
    frame_bgr: np.ndarray,
    threshold: Optional[float] = None,
    max_hits: int = 10,
    region: Optional[Tuple[float, float, float, float]] = None,
) -> List[TemplateHit]:
    """Find any named template in a frame."""
    matcher = get_template_matcher(name)
    if matcher is None:
        return []
    return matcher.match(frame_bgr, threshold=threshold, max_hits=max_hits, region=region)
