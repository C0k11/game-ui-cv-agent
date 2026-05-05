"""Identify which event is currently showing in each lobby banner
position by template-matching pre-cropped PNGs.

This is the self-developed equivalent of reference's ``compare_image``
approach — RedDeadDepresso/BAAuto documents it as:

    "For Event, you need to upload a cropped image of the event banner
     (without a background) in the assets/EN/goto or assets/CN/goto
     directory and save it as event_banner.png."

We extend that by supporting MULTIPLE concurrent events (so the
pipeline can decide which one to prioritise) and by caching templates
in a ``data/event_banners/`` folder populated via
``scripts/_build_event_banner_templates.py``.

Template file naming:
    tr_<event>.png  — top-right cycling widget
    bl_<event>.png  — bottom-left main EVENT! card

Events whose name starts with ``schale_`` are treated as the 夏莱 grind
banner that the pipeline always DEFERS to daily tasks; everything else
is an "enterable" event that we want to click through.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


_DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "data" / "event_banners"

# Canonical match sizes — both template and ROI are resized to these
# before cv2.matchTemplate so the score is resolution-agnostic.
_CANONICAL: Dict[str, Tuple[int, int]] = {
    "top_right": (240, 240),     # portrait-ish square
    "bottom_left": (480, 180),   # wide event card
}

# Banner regions in NORMALISED frame coordinates (x1, y1, x2, y2).
# These must match the regions the extraction script cropped from.
_REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    "top_right": (0.83, 0.15, 1.00, 0.32),
    "bottom_left": (0.01, 0.63, 0.33, 0.84),
}

# Event names (file-stem part after the tr_/bl_ prefix) that we treat as
# the 夏莱 grind banner and actively SKIP in the skill.
_SCHALE_PREFIXES = ("schale_",)


@dataclass(frozen=True)
class BannerMatch:
    position: str          # "top_right" | "bottom_left"
    event: Optional[str]   # matched template name, or None
    score: float           # TM_CCOEFF_NORMED in [-1, 1]

    @property
    def is_schale(self) -> bool:
        return bool(self.event and self.event.startswith(_SCHALE_PREFIXES))

    @property
    def is_known_event(self) -> bool:
        return self.event is not None


class EventBannerMatcher:
    """Loads event-banner templates once and classifies a live frame's
    top-right and bottom-left banner positions against them."""

    def __init__(self, template_dir: Path | str = _DEFAULT_TEMPLATE_DIR) -> None:
        self.template_dir = Path(template_dir)
        self._templates: Dict[str, Dict[str, np.ndarray]] = {
            "top_right": {},
            "bottom_left": {},
        }
        self._load()

    # ── loading ────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not self.template_dir.exists():
            return
        for p in sorted(self.template_dir.glob("*.png")):
            stem = p.stem
            if stem.startswith("tr_"):
                pos, name = "top_right", stem[3:]
            elif stem.startswith("bl_"):
                pos, name = "bottom_left", stem[3:]
            else:
                continue
            from vision.io_utils import imread_any  # noqa: PLC0415
            img = imread_any(str(p), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                continue
            canonical = _CANONICAL[pos]
            resized = cv2.resize(img, canonical, interpolation=cv2.INTER_AREA)
            self._templates[pos][name] = resized

    def all_events(self, position: str) -> list[str]:
        return sorted(self._templates.get(position, {}).keys())

    # ── helpers ────────────────────────────────────────────────────────
    @staticmethod
    def region(position: str) -> Tuple[float, float, float, float]:
        return _REGIONS[position]

    def _crop_region(
        self, frame_bgr: np.ndarray, position: str
    ) -> Optional[np.ndarray]:
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return None
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = _REGIONS[position]
        cx1, cy1 = max(0, int(x1 * w)), max(0, int(y1 * h))
        cx2, cy2 = min(w, int(x2 * w)), min(h, int(y2 * h))
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        crop = frame_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        canonical = _CANONICAL[position]
        return cv2.resize(crop, canonical, interpolation=cv2.INTER_AREA)

    # ── scoring ────────────────────────────────────────────────────────
    def classify(
        self, frame_bgr: np.ndarray, position: str
    ) -> BannerMatch:
        """Return the best-matching event template for the given banner
        position, plus its TM_CCOEFF_NORMED score.  ``event=None`` if no
        templates are loaded or the frame cannot be cropped."""
        templates = self._templates.get(position, {})
        if not templates:
            return BannerMatch(position, None, -1.0)
        roi = self._crop_region(frame_bgr, position)
        if roi is None:
            return BannerMatch(position, None, -1.0)
        best_name: Optional[str] = None
        best_score = -1.0
        for name, template in templates.items():
            # Same canonical size ⇒ matchTemplate returns a 1x1 result.
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            s = float(max_val)
            if s > best_score:
                best_score = s
                best_name = name
        return BannerMatch(position, best_name, best_score)

    # ── convenience predicates ────────────────────────────────────────
    def is_schale(
        self,
        frame_bgr: np.ndarray,
        position: str,
        *,
        min_score: float = 0.55,
    ) -> bool:
        m = self.classify(frame_bgr, position)
        return m.is_schale and m.score >= min_score

    def has_non_schale_event(
        self,
        frame_bgr: np.ndarray,
        position: str,
        *,
        min_score: float = 0.55,
    ) -> bool:
        m = self.classify(frame_bgr, position)
        return (
            m.is_known_event
            and not m.is_schale
            and m.score >= min_score
        )


# Module-level singleton for cheap reuse across ticks.
_shared: Optional[EventBannerMatcher] = None


def get_matcher() -> EventBannerMatcher:
    global _shared
    if _shared is None:
        _shared = EventBannerMatcher()
    return _shared
