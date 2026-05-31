"""ScheduleSkill: Handle daily schedule (課程表).

Game UI flow (observed from trajectory):
  - Schedule screen = "Location Select" showing locations like
    夏莱辦公室 RANK12, 夏莱居住區 RANK12, etc.
  - Click a location → enters the location detail view (isometric building)
  - Inside location: right ">" arrow to switch to next location
  - Click "全體課程表" button → opens roster overlay (rooms + students)
  - Check roster for target characters via YOLO26n-cls avatar classifier
    (vision.avatar_classifier — 4-layer fast/vote/mutex/unknown pipeline)
  - Close roster via X → click ">" to switch location if no target
  - To start: click center building area or "開始日程" button
  - Repeat until tickets exhausted (持有票券 = 0)

Avatar identification migrated 2026-05-17 from template+histogram match
(vision.avatar_matcher) to YOLO26n-cls (vision.avatar_classifier).
Reasons: template match against full-body CG portraits at ~50px crop
size produced false positives at 0.55 threshold; classifier hits
top1=86.55% / top5=95.32% on val and emits temporal-voted verdicts.

Button/navigation layer migrated 2026-05-28 from OCR text matching to
YOLO ui cls (find_cls / detect_screen_yolo).  See ui_classes.py — all
click targets resolve through named cls constants; OCR is retained ONLY
for the ticket digit read (持有票券 X/Y).  Avatar recognition, roster
overlay scan, room-slot coords, and the dispatch state machine are
UNCHANGED — only the "find a button / which page am I on" plumbing moved
to YOLO.

  IMPORTANT (interceptor dead-loop guard): SCHED_ALL (全體課程表) is the
  schedule WORK surface, not a popup.  We never treat it as a dismissable
  popup inside this skill — the pipeline interceptor previously closed it
  and trapped the skill in a re-open loop (task #3).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote

from brain.skills.base import (
    BaseSkill, ScreenState, OcrBox, YoloBox,
    action_click, action_click_yolo,
    action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC
from vision.avatar_classifier import get_default as get_avatar_classifier

# Schedule location-select tiles that HAVE a YOLO ui cls.  Ordered the
# same way the Location-Select screen lays them out (办公室 / 居住区 /
# 格黑娜 / 阿拜多斯 / 千年).  Clicking + visited-tracking now key off
# cls_name instead of OCR'd location names.
#
# GAP: the remaining schedule regions (百鬼夜行 / 紅冬 / 山海經 /
# 瓦爾基里 / SRT / 聯邦學園) have NO ui cls yet — see migration report.
# We cycle past them via the right-arrow switch (which also has no cls and
# is a documented gap) rather than OCR-clicking their tiles.
_SCHOOL_TILES = [
    UC.SCHOOL_OFFICE,      # 夏莱办公室   (first location — full-circle anchor)
    UC.SCHOOL_DORM,        # 夏莱居住区
    UC.SCHOOL_GEHENNA,     # 格黑娜学院中央区
    UC.SCHOOL_ABYDOS,      # 阿拜多斯高中
    UC.SCHOOL_MILLENNIUM,  # 千年研究所
]

# ── Room grid positions in 全體課程表 popup ──
# Measured from actual MuMu 3840×2160 screenshots (normalized 0-1).
# Room names: row1 y≈0.29, row2 y≈0.50, row3 y≈0.70
# Columns: x≈0.19, x≈0.46, x≈0.73
# Click center of each room card (slightly below the name).
_NUM_ROOMS = 7  # Only 7 rooms exist in the popup (row3 has 1)
_ROOM_CLICK_POS = [
    (0.19, 0.34), (0.46, 0.34), (0.73, 0.34),  # row 1: 視聽室, 體育館, 圖書館
    (0.19, 0.55), (0.46, 0.55), (0.73, 0.55),  # row 2: 教室, 實驗室, 射擊場
    (0.19, 0.74),                                # row 3: 載具庫 (only 1)
]
_ROOM_SLOT_NAMES = [
    "視聽室", "體育館", "圖書館",
    "教室", "實驗室", "射擊場",
    "載具庫",
]
# Status check positions: top-left area of each room card border.
# White=available, grey=done, dark=locked.
_ROOM_STATUS_POS = [
    (0.11, 0.27), (0.38, 0.27), (0.65, 0.27),  # row 1
    (0.11, 0.48), (0.38, 0.48), (0.65, 0.48),  # row 2
    (0.11, 0.68), (0.38, 0.68), (0.65, 0.68),  # row 3
]

# Strip-to-room mapping (fixes bug discovered via trajectory audit 2026-05-17).
# The user drew the avatar strips COLUMN-MAJOR in the dashboard:
#   col 1 top→bottom (strips 0,3,4), then col 2 top→bottom (strips 1,5,7),
#   then col 3 top→bottom (strips 2,6,8).
# But _ROOM_CLICK_POS / _ROOM_SLOT_NAMES are ROW-MAJOR.  The naive
# `enumerate(strips)` → room_idx mapping wires strips 4-6 to the wrong
# rooms (e.g. strip 4 = 載具庫 area mapped to room_idx 4 = 實驗室 →
# detector finds a character in 載具庫 but click goes to 實驗室).
#
# Mapping verified by nearest-neighbor matching strip centers (cx,cy) to
# _ROOM_CLICK_POS:
#   strip 4 (0.19, 0.82) -> room 6 載具庫 (0.19, 0.74)  dist=0.08
#   strip 5 (0.46, 0.61) -> room 4 實驗室 (0.46, 0.55)  dist=0.06
#   strip 6 (0.73, 0.61) -> room 5 射擊場 (0.73, 0.55)  dist=0.06
_STRIP_TO_ROOM_MAP = {
    0: 0,  # strip 0 row1-col1  -> room 0 視聽室
    1: 1,  # strip 1 row1-col2  -> room 1 體育館
    2: 2,  # strip 2 row1-col3  -> room 2 圖書館
    3: 3,  # strip 3 row2-col1  -> room 3 教室
    4: 6,  # strip 4 row3-col1  -> room 6 載具庫    ← was wrong (→4 實驗室)
    5: 4,  # strip 5 row2-col2  -> room 4 實驗室    ← was wrong (→5 射擊場)
    6: 5,  # strip 6 row2-col3  -> room 5 射擊場    ← was wrong (→6 載具庫)
}


def _load_target_favorites() -> List[str]:
    """Load target character names from app_config.json.

    Generates multiple name variants for each entry so that both config-style
    names (e.g. 'Toki_(Bunny_Girl).png') and reference template-style names
    (e.g. 'Toki (Bunny Girl)') are included for matching.
    """
    try:
        cfg_path = Path(__file__).resolve().parents[2] / "data" / "app_config.json"
        if cfg_path.exists():
            data = json.loads(cfg_path.read_text("utf-8"))
            raw = data.get("target_favorites", [])
            normalized: List[str] = []
            seen: Set[str] = set()
            for item in raw:
                name = str(item or "").strip()
                if not name:
                    continue
                candidates = [name]
                decoded = unquote(name)
                if decoded and decoded != name:
                    candidates.append(decoded)
                # Generate variants: strip .png, replace _ with space
                expanded: List[str] = []
                for c in candidates:
                    expanded.append(c)
                    stripped = c
                    if stripped.lower().endswith(".png"):
                        stripped = stripped[:-4]
                    if stripped != c:
                        expanded.append(stripped)
                    spaced = stripped.replace("_", " ")
                    if spaced != stripped:
                        expanded.append(spaced)
                for candidate in expanded:
                    key = candidate.lower()
                    if key in seen:
                        continue
                    normalized.append(candidate)
                    seen.add(key)
            return normalized
    except Exception:
        pass
    return []


# Minimum match score for lesson affection template matching.
# reference uses 0.75; we use the same threshold for per-room student detection.
_AFFECTION_MATCH_THRESHOLD = 0.75
# Avatar identification: handled by YOLO26n-cls (vision.avatar_classifier).
# Acceptance is internal to the classifier — fast-path conf>=0.95+margin>=0.30
# or 3-frame temporal vote.  No threshold lives here anymore.
# Max locations to cycle through before executing at current (full circle).
# Blue Archive has ~10 schedule locations; 12 gives safety margin.
_MAX_LOCATIONS_FULL_CIRCLE = 12

# ── Per-room student detect regions (from reference lesson.py) ──
# Base resolution 1280×720; each room card in the 3×3 roster popup grid
# has a student avatar area at these pixel offsets.
# Room j: x1 = 285 + 344*(j%3), y1 = 240 + 152*(j//3), w=161, h=52
_ROOM_DETECT_REGIONS_1280 = []
for _j in range(9):
    _rx1 = 285 + 344 * (_j % 3)
    _ry1 = 240 + 152 * (_j // 3)
    _ROOM_DETECT_REGIONS_1280.append((_rx1, _ry1, _rx1 + 161, _ry1 + 52))
# Normalized 0-1 versions
_ROOM_DETECT_REGIONS_NORM = [
    (r[0]/1280, r[1]/720, r[2]/1280, r[3]/720) for r in _ROOM_DETECT_REGIONS_1280
]


class ScheduleSkill(BaseSkill):
    _LOBBY_DOT_ENTRIES = [UC.NAV_SCHEDULE, UC.SCHED_TICKET]

    def should_run(self, screen):
        return self.dot_on_entry(screen, self._LOBBY_DOT_ENTRIES)

    def __init__(self):
        super().__init__("Schedule")
        self.max_ticks = 120
        self._tickets_used: int = 0
        self._tickets_remaining: int = -1  # -1 = unknown
        self._locations_checked: int = 0
        self._animation_ticks: int = 0
        self._roster_open: bool = False
        self._target_found: bool = False
        self._target_favorites: List[str] = []
        self._wait_ticks: int = 0
        self._visited_locations: Set[str] = set()
        self._starting_location: str = ""
        self._switch_ticks: int = 0
        self._execute_ticks: int = 0
        self._start_clicked: bool = False
        self._roster_scan_ticks: int = 0
        self._avatar_clf = None  # vision.avatar_classifier singleton (lazy)
        self._stale_ticks: int = 0
        self._matched_room_idx: int = -1  # room index where favorite was found
        self._clicked_rooms_this_location: Set[int] = set()  # rooms clicked since last location switch
        self._avatar_regions_cfg = self._load_avatar_regions_config()

    @staticmethod
    def _default_strips() -> List[Dict[str, float]]:
        """Derive default flat strip list from class-level rows × cols grid."""
        return [
            {"x1": cx1, "y1": ry1, "x2": cx2, "y2": ry2}
            for (ry1, ry2) in ScheduleSkill._DEFAULT_AVATAR_ROWS_Y
            for (cx1, cx2) in ScheduleSkill._DEFAULT_AVATAR_COLS_X
        ]

    @staticmethod
    def _load_avatar_regions_config() -> Dict[str, Any]:
        """Load roster avatar region overrides from data/schedule_avatar_regions.json.

        Preferred format (canvas UI)::

            {
              "strips": [{"x1":0.09,"y1":0.35,"x2":0.30,"y2":0.47}, ...],
              "cells_per_room": 4
            }

        Legacy grid format (still supported for back-compat)::

            {
              "rows_y": [[0.35,0.47], ...],
              "cols_x": [[0.09,0.30], ...],
              "cells_per_room": 4
            }

        Missing keys fall back to class defaults. Invalid files are ignored.
        """
        try:
            from pathlib import Path
            import json as _json
            p = Path(__file__).resolve().parents[2] / "data" / "schedule_avatar_regions.json"
            if p.exists():
                raw = _json.loads(p.read_text(encoding="utf-8"))
                cpr = int(raw.get("cells_per_room") or ScheduleSkill._DEFAULT_CELLS_PER_ROOM)
                # Preferred: flat strip list
                strips_raw = raw.get("strips")
                if isinstance(strips_raw, list) and strips_raw:
                    strips: List[Dict[str, float]] = []
                    for s in strips_raw:
                        try:
                            x1, y1, x2, y2 = float(s["x1"]), float(s["y1"]), float(s["x2"]), float(s["y2"])
                            if x2 > x1 and y2 > y1:
                                strips.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
                        except (KeyError, TypeError, ValueError):
                            continue
                    if strips:
                        return {"strips": strips, "cells_per_room": max(1, cpr)}
                # Legacy: rows × cols cross-product
                rows = raw.get("rows_y")
                cols = raw.get("cols_x")
                if rows and cols:
                    rows_t = [tuple(r) for r in rows if len(r) == 2]
                    cols_t = [tuple(c) for c in cols if len(c) == 2]
                    if rows_t and cols_t:
                        strips = [
                            {"x1": float(cx1), "y1": float(ry1), "x2": float(cx2), "y2": float(ry2)}
                            for (ry1, ry2) in rows_t for (cx1, cx2) in cols_t
                        ]
                        return {"strips": strips, "cells_per_room": max(1, cpr)}
        except Exception:
            pass
        return {
            "strips": ScheduleSkill._default_strips(),
            "cells_per_room": ScheduleSkill._DEFAULT_CELLS_PER_ROOM,
        }

    def reset(self) -> None:
        super().reset()
        self._tickets_used = 0
        self._tickets_remaining = -1
        self._locations_checked = 0
        self._animation_ticks = 0
        self._roster_open = False
        self._target_found = False
        self._start_clicked = False
        self._target_favorites = _load_target_favorites()
        self._wait_ticks = 0
        self._visited_locations = set()
        self._starting_location = ""
        self._switch_ticks = 0
        self._execute_ticks = 0
        self._stale_ticks = 0
        self._roster_scan_ticks = 0
        self._matched_room_idx = -1
        self._clicked_rooms_this_location = set()
        self._avatar_regions_cfg = self._load_avatar_regions_config()
        if self._target_favorites:
            self.log(f"target favorites loaded: {len(self._target_favorites)} characters")
        # Lazy-init avatar classifier (YOLO26n-cls, ~290-line wrapper in vision/)
        if self._avatar_clf is None and self._target_favorites:
            self._avatar_clf = get_avatar_classifier()
            if not self._avatar_clf.available:
                self.log(
                    f"avatar classifier model missing at {self._avatar_clf.model_path} — "
                    f"falling back to pixel-only room picker"
                )
                self._avatar_clf = None
            else:
                self.log(f"avatar classifier ready ({self._avatar_clf.model_path.name})")
        # Reset temporal vote buffers — this is a fresh schedule run
        if self._avatar_clf is not None:
            self._avatar_clf.reset_buffer(reason="schedule reset")
        # Normalized favorite name set (matches classifier output format:
        # 'Wakamo.png' → 'Wakamo', 'Toki_(Bunny_Girl).png' → 'Toki_(Bunny_Girl)')
        self._fav_classes: Set[str] = {
            unquote(f[:-4] if f.lower().endswith(".png") else f)
            for f in self._target_favorites
        }
        self._matched_avatar_pos: Optional[Tuple[float, float]] = None  # where the matched avatar is on popup

    # ── Ticket parsing ──

    def _parse_tickets(self, screen: ScreenState) -> None:
        """Parse remaining ticket count from '持有票券 X/Y' OCR text."""
        for box in screen.ocr_boxes:
            if box.confidence < 0.6:
                continue
            m = re.search(r'(\d+)\s*/\s*(\d+)', box.text)
            if m and '票券' in box.text:
                remaining = int(m.group(1))
                total = int(m.group(2))
                if self._tickets_remaining != remaining:
                    self.log(f"tickets: {remaining}/{total}")
                self._tickets_remaining = remaining
                return

    # ── Avatar matching ──

    def _fallback_roster_avatar_boxes(self, screen: ScreenState) -> List[OcrBox]:
        candidates: List[OcrBox] = []
        seen: Set[Tuple[int, int]] = set()
        level_hits = [
            box for box in screen.ocr_boxes
            if box.confidence >= 0.6 and re.match(r"^L[vV]\.?(\d+)$", box.text.strip())
        ]
        for level in level_hits:
            card_x1 = max(0.06, level.x1 - 0.02)
            card_y1 = max(0.18, level.y1 - 0.01)
            card_x2 = min(0.93, card_x1 + 0.28)
            card_y2 = min(0.84, card_y1 + 0.18)
            locked = any(
                hit.confidence >= 0.6
                and "需要RANK" in hit.text
                and card_x1 <= hit.cx <= card_x2
                and card_y1 <= hit.cy <= card_y2
                for hit in screen.ocr_boxes
            )
            if locked:
                continue
            usable_x1 = card_x1 + (card_x2 - card_x1) * 0.02
            usable_x2 = card_x2 - (card_x2 - card_x1) * 0.02
            avatar_y1 = card_y1 + (card_y2 - card_y1) * 0.45
            avatar_y2 = card_y1 + (card_y2 - card_y1) * 0.92
            cell_w = max(0.02, (usable_x2 - usable_x1) / 4.0)
            for idx in range(4):
                x1 = usable_x1 + idx * cell_w
                x2 = min(usable_x2, usable_x1 + (idx + 1) * cell_w)
                key = (int((x1 + x2) * 500), int((avatar_y1 + avatar_y2) * 500))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    OcrBox(
                        text=f"roster_avatar_{len(candidates) + 1}",
                        confidence=0.3,
                        x1=x1,
                        y1=avatar_y1,
                        x2=x2,
                        y2=avatar_y2,
                    )
                )
        return candidates

    def _get_room_statuses(self, screen: ScreenState) -> List[str]:
        """Detect status of all 9 rooms in 全體課程表 popup.

        Uses green checkmark detection on avatar areas: completed rooms
        have green ✓ overlays on student avatars (HSV green pixels > 0.5%).
        The template-based border pixel check doesn't work for our resolution
        (all borders read as white=255 regardless of room status).

        Returns list of 9 status strings: "available", "done", or "unknown".
        """
        import cv2
        import numpy as np
        try:
            img = cv2.imdecode(
                np.fromfile(screen.screenshot_path, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if img is None:
                return ["unknown"] * _NUM_ROOMS
            h, w = img.shape[:2]
        except Exception:
            return ["unknown"] * _NUM_ROOMS

        # Use the same strip list that _check_roster_avatars uses — this
        # comes from data/schedule_avatar_regions.json when present, or
        # falls back to the class-level defaults (_default_strips()).
        # NOTE: _AVATAR_ROWS_Y / _AVATAR_COLS_X were removed when the
        # region config was flattened to strips; referencing them here
        # crashed with AttributeError and aborted the pipeline worker
        # (see run_20260422_213826 tick 8).
        statuses: List[str] = []
        strips = (self._avatar_regions_cfg or {}).get("strips") \
            or self._default_strips()
        for room_idx, s in enumerate(strips):
            if room_idx >= _NUM_ROOMS:
                break
            try:
                px1 = int(float(s["x1"]) * w)
                py1 = int(float(s["y1"]) * h)
                px2 = int(float(s["x2"]) * w)
                py2 = int(float(s["y2"]) * h)
            except (KeyError, TypeError, ValueError):
                statuses.append("unknown")
                continue
            # The canvas-tuned strips cover only the avatar FACE area, but
            # BA's 'done' green ✓ overlay sits on the TOP-RIGHT of each
            # avatar — that is, ABOVE the face strip.  Extend the strip
            # upward by 2x its own height for the status check so the
            # checkmark pixels actually fall inside the ROI.  We only do
            # this for green detection; avatar matching still uses the
            # tight strip (wide aspect ratio preserved).
            strip_h = max(1, py2 - py1)
            sy1 = max(0, py1 - 2 * strip_h)
            roi = img[sy1:py2, px1:px2]
            if roi.size == 0:
                statuses.append("unknown")
                continue
            # Green checkmark: H=35-85, S>80, V>100 (bright green overlay)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, np.array([35, 80, 100]),
                                     np.array([85, 255, 255]))
            green_ratio = (green_mask.sum() / 255) / max(1, roi.shape[0] * roi.shape[1])
            statuses.append("done" if green_ratio > 0.005 else "available")
        while len(statuses) < _NUM_ROOMS:
            statuses.append("unknown")
        return statuses

    def _choose_best_room(self, statuses: List[str]) -> int:
        """Choose the best available room to schedule.

        Adapted from reference lesson.py choose_lesson():
        - Prefer rooms with higher tier (bottom rows = higher tier rewards)
        - Among available rooms, pick the last one (higher index = better tier)

        Returns room index (0-8) or -1 if none available.
        """
        best = -1
        for i in range(min(len(statuses), _NUM_ROOMS) - 1, -1, -1):
            if statuses[i] == "available":
                best = i
                break
        return best

    # Avatar region positions in 全體課程表 popup (normalized 0-1).
    # Each room card's avatar strip: where the small face icons appear.
    # Row/column grid: 3×3 layout, last row only has 1 room.
    # Default values can be overridden by data/schedule_avatar_regions.json;
    # see _load_avatar_regions_config() below.
    _DEFAULT_AVATAR_ROWS_Y = [(0.35, 0.47), (0.54, 0.66), (0.73, 0.84)]
    _DEFAULT_AVATAR_COLS_X = [(0.09, 0.30), (0.34, 0.55), (0.58, 0.79)]
    _DEFAULT_CELLS_PER_ROOM = 4

    def _check_roster_avatars(self, screen: ScreenState) -> bool:
        """Check roster overlay for favorite characters via YOLO26n-cls.

        For each available room in the 全體課程表 popup:
          1. Crop each avatar cell from the configured strip (3 cells/room).
          2. Batch them through AvatarClassifier (4-layer pipeline:
             fast-path / temporal vote / Hungarian spatial mutex / unknown).
          3. If any verdict's class name is in our favorite set with
             source in {fast, vote, mutex(*)}, mark the room.

        Side effects on hit:
          self._target_found = True
          self._matched_room_idx = <room_idx>
        """
        if not self._target_favorites:
            return False
        if self._avatar_clf is None:
            # classifier failed to load; pixel-only fallback runs in _check_roster
            return False

        try:
            import cv2
            import numpy as np
            img = cv2.imdecode(
                np.fromfile(screen.screenshot_path, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if img is None:
                return False
            h, w = img.shape[:2]
        except Exception:
            return False

        statuses = self._get_room_statuses(screen)
        strips = self._avatar_regions_cfg.get("strips") or []
        cells_per_room = int(self._avatar_regions_cfg.get("cells_per_room", 3))

        # ── 1. Build cell crops (only for rooms we'd actually want to click) ──
        # strip_idx is the position in schedule_avatar_regions.json (column-major
        # draw order).  room_idx is the canonical row-major index used by
        # _ROOM_CLICK_POS / _ROOM_SLOT_NAMES.  Map via _STRIP_TO_ROOM_MAP.
        cells: List[Tuple[Any, int, int]] = []  # (crop_bgr, room_idx, cell_idx)
        room_skipped: Dict[int, str] = {}       # room_idx -> reason (for log)
        for strip_idx, s in enumerate(strips):
            room_idx = _STRIP_TO_ROOM_MAP.get(strip_idx)
            if room_idx is None or room_idx >= _NUM_ROOMS:
                # Strip outside the 7-room popup (legacy 9-strip configs had
                # bottom-mid/bottom-right entries that don't match any room).
                continue
            if statuses[room_idx] not in ("available", "unknown"):
                room_skipped[room_idx] = f"status={statuses[room_idx]}"
                continue
            # Don't re-target a room we've already dispatched at this location
            # (observed loop bug in run_20260422_214941 ticks 19–59 where the
            # post-dispatch roster re-scan kept clicking the same favorite).
            if room_idx in self._clicked_rooms_this_location:
                room_skipped[room_idx] = "already-clicked"
                continue
            px1 = max(0, int(s["x1"] * w))
            py1 = max(0, int(s["y1"] * h))
            px2 = min(w, int(s["x2"] * w))
            py2 = min(h, int(s["y2"] * h))
            strip = img[py1:py2, px1:px2]
            if strip.size == 0:
                continue
            strip_h, strip_w = strip.shape[:2]
            cell_w = max(1, strip_w // cells_per_room)
            cell_size = min(cell_w, strip_h)
            if cell_size < 16:  # too tiny to be a real avatar
                continue
            sy = max(0, (strip_h - cell_size) // 2)
            for slot in range(cells_per_room):
                sx = slot * cell_w + max(0, (cell_w - cell_size) // 2)
                if sx + cell_size > strip_w:
                    break
                cell = strip[sy:sy + cell_size, sx:sx + cell_size]
                if cell.size == 0:
                    continue
                cells.append((cell, room_idx, slot))

        if not cells:
            scanned = sum(1 for s in statuses if s in ("available", "unknown"))
            self.log(
                f"no avatar cells to scan (scanned={scanned} skipped={room_skipped})"
            )
            return False

        # ── 2. Batch-classify all cells in this frame ──
        frame_id = getattr(screen, "tick_id", -1) or self._roster_scan_ticks
        verdicts = self._avatar_clf.classify_cells(cells, frame_id=frame_id)

        # ── 3. Search verdicts for favorites; pick highest-confidence match ──
        ACCEPT_SOURCES = {"fast", "vote"}  # mutex(*) also OK — handled below
        best_room = -1
        best_name: Optional[str] = None
        best_conf = -1.0
        per_room_log: Dict[int, List[str]] = {}
        for v in verdicts:
            per_room_log.setdefault(v.room_idx, []).append(
                f"c{v.cell_idx}[{v.name}/{v.conf:.2f}/{v.source}]"
            )
            if v.name is None or v.name == "__unknown__":
                continue
            is_accepted = (
                v.source in ACCEPT_SOURCES
                or v.source.startswith("mutex(")
            )
            if not is_accepted:
                continue
            if v.name in self._fav_classes and v.conf > best_conf:
                best_room = v.room_idx
                best_name = v.name
                best_conf = v.conf

        if best_room >= 0 and best_name:
            self._target_found = True
            self._matched_room_idx = best_room
            self.log(
                f"★FAVORITE MATCH: '{best_name}' conf={best_conf:.3f} "
                f"in room {best_room} ({_ROOM_SLOT_NAMES[best_room]})"
            )
            return True

        # No favorite: emit per-room diag (one line per room) for calibration
        for room_idx, slots in sorted(per_room_log.items()):
            self.log(
                f"room {room_idx} ({_ROOM_SLOT_NAMES[room_idx]}): "
                f"no favorite | " + " ".join(slots)
            )
        return False

    def _find_lowest_affection_card(self, screen: ScreenState):
        """Find the location card with lowest affection number in roster popup.

        Affection numbers appear as small digits (10-30) near avatar icons.
        Returns (cx, cy) of the lowest-numbered avatar, or None.
        """
        import re
        # Find all small numbers in the roster popup area (affection scores)
        candidates = []
        for box in screen.ocr_boxes:
            text = box.text.strip()
            # Affection numbers are 1-2 digit numbers, typically 10-30
            if not re.match(r"^\d{1,2}$", text):
                continue
            val = int(text)
            if val < 1 or val > 50:
                continue
            # Must be in the roster popup area
            if box.cy < 0.15 or box.cy > 0.85 or box.cx < 0.08 or box.cx > 0.92:
                continue
            candidates.append((val, box.cx, box.cy))

        if not candidates:
            return None

        # Pick the lowest affection number
        candidates.sort(key=lambda c: c[0])
        val, cx, cy = candidates[0]
        self.log(f"lowest affection: {val} at ({cx:.2f},{cy:.2f})")
        return (cx, cy)

    # ── Screen detection helpers (YOLO cls signatures) ──
    #
    # All page judgement runs off the schedule ui cls instead of OCR
    # headers.  Region splits distinguish the two faces of SCHED_ALL
    # (全體課程表): top-center = roster overlay header (open); bottom-right
    # = the button that OPENS the roster (inside-location detail view).

    # SCHED_ALL when it is the roster overlay TITLE (top-center band).
    _ROSTER_HEADER_REGION = (0.15, 0.0, 0.85, 0.22)
    # SCHED_ALL when it is the open-roster BUTTON (bottom-right).
    _ROSTER_OPEN_BTN_REGION = (0.55, 0.78, 1.0, 1.0)

    def _on_schedule_page(self, screen: ScreenState) -> bool:
        """Any schedule surface (location-select / location-detail / roster).

        Signature = detect_screen_yolo()=="Schedule" (SCHED_ALL / SCHED_TICKET
        / SCHED_START visible) OR a school-area tile is on screen (the
        Location-Select list shows the SCHOOL_* tiles but may carry none of
        the SCHED_* cls)."""
        if self.detect_screen_yolo(screen) == "Schedule":
            return True
        return self.find_cls(screen, _SCHOOL_TILES, conf=0.30) is not None

    # Backwards-compatible alias used throughout the state machine.
    def _is_schedule(self, screen: ScreenState) -> bool:
        return self._on_schedule_page(screen)

    def _is_location_select(self, screen: ScreenState) -> bool:
        """Detect Location Select screen (the list of all school areas).

        Signature = at least one SCHOOL_* tile cls visible AND the roster
        overlay is NOT open (the roster header would otherwise sit on top)."""
        if self._find_roster_header(screen) is not None:
            return False
        return self.find_cls(screen, _SCHOOL_TILES, conf=0.30) is not None

    def _find_roster_header(self, screen: ScreenState) -> Optional[YoloBox]:
        """SCHED_ALL appearing as the roster overlay TITLE (top-center)."""
        return self.find_cls(
            screen, UC.SCHED_ALL, conf=0.30, region=self._ROSTER_HEADER_REGION,
        )

    def _find_roster_open_btn(self, screen: ScreenState) -> Optional[YoloBox]:
        """SCHED_ALL appearing as the open-roster BUTTON (bottom-right)."""
        return self.find_cls(
            screen, UC.SCHED_ALL, conf=0.30, region=self._ROSTER_OPEN_BTN_REGION,
        )

    def _is_roster_overlay(self, screen: ScreenState) -> bool:
        """Detect the full roster overlay (全體課程表 title at top-center).

        NOTE: SCHED_ALL here is the schedule WORK surface — NOT a popup to
        close.  The pipeline interceptor must not dismiss it (task #3)."""
        return self._find_roster_header(screen) is not None

    def _is_inside_location(self, screen: ScreenState) -> bool:
        """Detect the location detail view (inside one school, iso building).

        Has the 全體課程表 OPEN button (bottom-right) but the roster overlay
        is not yet open, and it is not the Location-Select list."""
        if self._find_roster_header(screen) is not None:
            return False  # roster open, not the plain detail view
        if self._is_location_select(screen):
            return False
        return self._find_roster_open_btn(screen) is not None

    def _find_schedule_close(self, screen: ScreenState) -> Optional[YoloBox]:
        """Find the roster close-X via YOLO BTN_CLOSE_X cls (top-right)."""
        return self.find_cls(
            screen, UC.BTN_CLOSE_X, conf=0.30, region=(0.70, 0.0, 0.98, 0.30),
        )

    def _find_schedule_switch_arrow(self, screen: ScreenState) -> Optional[YoloBox]:
        """Find the right location-switch arrow via YOLO ARROW_RIGHT cls."""
        return self.find_cls(
            screen, UC.ARROW_RIGHT, conf=0.30, region=(0.88, 0.30, 1.0, 0.70),
        )

    def _close_roster_action(self, screen: ScreenState, next_state: str, reason: str) -> Dict[str, Any]:
        """Close the roster overlay and transition to next_state.

        Resolves the close-X via YOLO BTN_CLOSE_X cls.  If the cls isn't
        visible we surface the gap (log + wait) rather than blind-clicking a
        hardcoded corner."""
        self._roster_open = False
        close_btn = self._find_schedule_close(screen)
        if close_btn:
            self.sub_state = next_state
            return action_click_yolo(close_btn, f"close roster X ({reason}, YOLO 弹窗叉叉)")
        self.log(f"close roster: no BTN_CLOSE_X cls ({reason}) — YOLO gap; waiting")
        self.sub_state = next_state
        return action_wait(400, f"waiting for roster close-X cls ({reason})")

    # ── State handlers ──

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("schedule timeout")

        # ── Popup handling ──

        # Schedule result / reward popup — tap the GOT_REWARD cls to dismiss
        # (replaces OCR 獲得獎勵/日程結果 + blind center-tap).
        got_reward = self.find_cls(screen, UC.GOT_REWARD, conf=0.30)
        if got_reward is not None:
            self.log("schedule result/reward popup, tapping to dismiss (YOLO 获得奖励)")
            self._roster_open = False
            return action_click_yolo(got_reward, "dismiss schedule result (YOLO 获得奖励)")

        # Bond / region level-up full-screen splash (羈絆升級！/ 地區升級).
        # These are tap-to-dismiss过场 surfaces the schedule can trigger by
        # raising affinity.  Resolve via the dedicated splash cls (same as
        # story_mining); tap anywhere advances.
        splash = self.find_cls(screen, [UC.BOND_LEVELUP, UC.REGION_LEVELUP], conf=0.30)
        if splash is not None:
            self.log(f"level-up splash ({splash.cls_name}), tapping to dismiss")
            return action_click(0.5, 0.5, f"dismiss splash ({splash.cls_name})")

        # Ticket-exhausted popup.  The body text "日程券不足" has no ui cls,
        # so the OCR guard stays (full-screen text guard — allowed); the
        # button it dismisses is resolved via YOLO BTN_CONFIRM / BTN_CLOSE_X.
        no_ticket = screen.find_any_text(
            ["日程券不足", "日程券已用完", "沒有日程券", "没有日程券"],
            region=screen.CENTER,
            min_conf=0.6,
        )
        if no_ticket:
            confirm = self.find_cls(
                screen, UC.BTN_CONFIRM, conf=0.30, region=screen.CENTER,
            )
            if confirm is not None:
                self.log(f"tickets exhausted ({self._tickets_used} used)")
                self.sub_state = "exit"
                return action_click_yolo(confirm, "confirm no tickets (YOLO 确认键)")
            close_btn = self._find_schedule_close(screen)
            if close_btn is not None:
                self.sub_state = "exit"
                return action_click_yolo(close_btn, "close ticket popup (YOLO 弹窗叉叉)")
            self.log("ticket-exhausted popup but no BTN_CONFIRM/X cls — YOLO gap; waiting")
            self.sub_state = "exit"
            return action_wait(400, "waiting for ticket-popup button cls")

        # Generic popups (confirm/cancel dialogs, notifications) — base helper
        # already resolves buttons via OCR+YOLO bottom-up fallback.
        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "schedule loading")

        # ── State machine ──

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "select_location":
            return self._select_location(screen)
        if self.sub_state == "check_roster":
            return self._check_roster(screen)
        if self.sub_state == "execute":
            return self._execute(screen)
        if self.sub_state == "switch_location":
            return self._switch_location(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "schedule unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        current = self.detect_screen_yolo(screen)
        roster_overlay = self._is_roster_overlay(screen)
        inside_location = self._is_inside_location(screen)
        location_select = self._is_location_select(screen)
        in_schedule = self._is_schedule(screen)

        if current == "Schedule" or in_schedule or roster_overlay or inside_location or location_select:
            self.log("inside schedule")
            self.sub_state = "check_roster" if (roster_overlay or inside_location) else "select_location"
            return action_wait(500, "entered schedule")

        if current == "Lobby":
            # YOLO 课程表入口 cls (lobby bottom-nav schedule icon).
            nav = self.click_cls(screen, UC.NAV_SCHEDULE, "click schedule nav", conf=0.30)
            if nav:
                return nav
            self.log("on lobby but no 课程表入口 cls — YOLO gap; waiting")
            return action_wait(400, "waiting for 课程表入口 cls")

        if current is not None and current != "Schedule":
            self.log(f"wrong screen '{current}', backing out")
            return action_back(f"back from {current}")

        return action_wait(500, "entering schedule")

    def _select_location(self, screen: ScreenState) -> Dict[str, Any]:
        """On Location Select screen, click an unvisited location."""
        if not self._is_schedule(screen):
            # If we're back on lobby, the skill ended — don't keep pressing
            # back (which would open the exit-game dialog and loop forever).
            if self.detect_screen_yolo(screen) == "Lobby":
                self.log("on lobby, schedule already exited — finishing skill")
                self._stale_ticks = 0
                self.sub_state = "exit"
                return action_done("schedule done (returned to lobby)")
            self._stale_ticks += 1
            if self._stale_ticks > 10:
                self.log("schedule UI lost for too long, trying back")
                self._stale_ticks = 0
                return action_back("back (schedule UI lost)")
            return action_wait(500, "waiting for schedule UI")
        self._stale_ticks = 0

        if self._is_roster_overlay(screen):
            self.log("roster already open while selecting location, switching back to check_roster")
            self.sub_state = "check_roster"
            self._roster_open = True
            self._roster_scan_ticks = 0
            self._wait_ticks = 0
            return action_wait(300, "roster already open")

        # If we're actually inside a location (not Location Select), go check it
        if self._is_inside_location(screen):
            self.log("already inside a location, checking roster")
            self.sub_state = "check_roster"
            self._roster_open = False
            self._wait_ticks = 0
            return action_wait(300, "inside location, switching to check_roster")

        # If tickets are known and 0, exit
        if self._tickets_remaining == 0:
            self.log("no tickets remaining, exiting")
            self.sub_state = "exit"
            return action_wait(300, "no tickets")

        # Full-circle safety: if we've cycled through too many locations, execute
        if self._locations_checked >= _MAX_LOCATIONS_FULL_CIRCLE:
            self.log(f"full circle done ({self._locations_checked} locations), entering for execute")
            tile = self.find_cls(screen, _SCHOOL_TILES, conf=0.30)
            if tile is not None:
                self.sub_state = "execute"
                self._execute_ticks = 0
                return action_click_yolo(tile, f"enter location for fallback execute (YOLO {tile.cls_name})")
            self.log("no school-tile cls found for fallback execute — YOLO gap; exiting")
            self.sub_state = "exit"
            return action_wait(300, "no school tiles found")

        # First entry: always start at 夏莱办公室 (the full-circle anchor).
        if not self._starting_location:
            office = self.find_cls(screen, UC.SCHOOL_OFFICE, conf=0.30)
            if office is not None:
                self._starting_location = office.cls_name
                self._visited_locations.add(office.cls_name)
                self.log(f"starting at '{office.cls_name}' (will full-circle back)")
                self.sub_state = "check_roster"
                self._roster_open = False
                self._wait_ticks = 0
                return action_click_yolo(office, f"enter location '{office.cls_name}'")
            # 办公室 tile not visible — start at the first visible school tile.
            tile = self._first_visible_school_tile(screen)
            if tile is not None:
                self._starting_location = tile.cls_name
                self._visited_locations.add(tile.cls_name)
                self.log(f"starting at '{tile.cls_name}' (办公室 not visible)")
                self.sub_state = "check_roster"
                self._roster_open = False
                self._wait_ticks = 0
                return action_click_yolo(tile, f"enter location '{tile.cls_name}'")
            self.log("no school-tile cls visible to start — YOLO gap; waiting")
            return action_wait(400, "waiting for school-tile cls")

        # Subsequent entries from Location Select: click the first UNVISITED
        # school tile (top-to-bottom). Visited tracked by cls_name.
        tile = self._first_visible_school_tile(screen, skip_visited=True)
        if tile is not None:
            self._visited_locations.add(tile.cls_name)
            self.log(f"clicking location '{tile.cls_name}' (new)")
            self.sub_state = "check_roster"
            self._roster_open = False
            self._wait_ticks = 0
            return action_click_yolo(tile, f"enter location '{tile.cls_name}'")

        # All cls-backed school tiles visited. The remaining regions
        # (百鬼夜行 / 紅冬 / 山海經 / 瓦爾基里 / SRT / 聯邦學園) have no cls —
        # documented gap. Re-enter the first visible tile so we still execute
        # whatever rooms remain rather than stalling.
        tile = self._first_visible_school_tile(screen)
        if tile is not None:
            self.log(f"all cls tiles visited, re-entering '{tile.cls_name}' (no-cls regions are a gap)")
            self.sub_state = "check_roster"
            self._roster_open = False
            self._wait_ticks = 0
            return action_click_yolo(tile, f"enter location '{tile.cls_name}' (all visited)")

        self.log("no school-tile cls visible — YOLO gap; waiting")
        return action_wait(500, "looking for school-tile cls")

    def _first_visible_school_tile(
        self, screen: ScreenState, *, skip_visited: bool = False,
    ) -> Optional[YoloBox]:
        """Return the top-most school-area tile cls on the Location-Select
        screen (optionally skipping ones already in _visited_locations).

        Ordered top-to-bottom (cy then cx) so cycling is deterministic."""
        tiles = self.find_all_cls(screen, _SCHOOL_TILES, conf=0.30)
        if skip_visited:
            tiles = [t for t in tiles if t.cls_name not in self._visited_locations]
        if not tiles:
            return None
        return min(tiles, key=lambda b: (round(b.cy, 3), b.cx))

    def _detect_current_location(self, screen: ScreenState) -> str:
        """Detect the current location from the detail-view via school tile cls.

        Best-effort bookkeeping for _visited_locations only (never a click).
        Returns the cls_name of any school tile visible, or ""."""
        tile = self.find_cls(screen, _SCHOOL_TILES, conf=0.30)
        return tile.cls_name if tile is not None else ""

    def _check_roster(self, screen: ScreenState) -> Dict[str, Any]:
        """Inside a location: open roster, check for targets/locks, decide."""
        # Cooldown after location switch: negative _wait_ticks counts up to 0
        if self._wait_ticks < 0:
            self._wait_ticks += 1
            return action_wait(500, f"location switch cooldown ({self._wait_ticks})")

        roster_overlay = self._is_roster_overlay(screen)
        if not roster_overlay and not self._is_schedule(screen):
            self._wait_ticks += 1
            if self._wait_ticks > 5:
                self.sub_state = "select_location"
            return action_wait(500, "waiting for location view")

        self._wait_ticks = 0

        # Recovery: if roster was open but now it's gone (accidentally closed),
        # try to re-open the 全體課程表 button (YOLO SCHED_ALL, bottom-right).
        if not roster_overlay and self._roster_open:
            self.log("roster was open but closed unexpectedly, re-opening")
            self._roster_open = False
            self._roster_scan_ticks = 0
            full_tab = self._find_roster_open_btn(screen)
            if full_tab is not None:
                return action_click_yolo(full_tab, "re-open roster after accidental close (YOLO 全体课程表)")
            self.log("re-open: no SCHED_ALL button cls — YOLO gap; waiting")
            return action_wait(300, "looking for roster button to re-open")

        # Detect current location name (visible in building detail header)
        cur_loc = self._detect_current_location(screen)

        # ── Roster overlay: scan rooms and click an available one directly ──
        # Click OCR-detected room names in the popup to navigate to room info.
        if roster_overlay:
            self._roster_open = True
            self._roster_scan_ticks += 1

            if self._roster_scan_ticks == 1:
                self._target_found = False
                self._parse_tickets(screen)
                return action_wait(300, "scanning roster tickets")

            # Ticks 2-4: scan avatars up to 3 times so the classifier's
            # temporal vote (3-frame buffer) can converge on ambiguous cells.
            # Fast-path matches (conf>=0.95 + margin>=0.30) cause immediate
            # _target_found=True and we fall through to room-pick on this
            # same tick — no extra latency for the easy 80% case.
            if 2 <= self._roster_scan_ticks <= 4 and self._target_favorites:
                self._check_roster_avatars(screen)
                if not self._target_found:
                    return action_wait(
                        300,
                        f"scanning roster avatars (pass {self._roster_scan_ticks - 1}/3)",
                    )

            # Tick 5+ (or earlier on fast-path hit): pick a room
            self._roster_scan_ticks = 0
            self._locations_checked += 1
            if cur_loc:
                self._visited_locations.add(cur_loc)
            self._switch_ticks = 0

            # PRIORITY: if favorite student was found in a specific room, click that room
            if self._target_found and self._matched_room_idx >= 0:
                fav_slot = self._matched_room_idx
                slot_name = _ROOM_SLOT_NAMES[fav_slot]
                click_x, click_y = _ROOM_CLICK_POS[fav_slot]
                self.log(f"clicking FAVORITE room '{slot_name}' idx={fav_slot}")
                self._roster_open = False
                self._matched_room_idx = -1
                self._clicked_rooms_this_location.add(fav_slot)
                self.sub_state = "execute"
                self._execute_ticks = 0
                self._start_clicked = False
                return action_click(click_x, click_y, f"click favorite room {slot_name}")

            # template-based primary path: use fixed-position pixel color checks to find
            # the best available room slot, then click its normalized slot position.
            statuses = self._get_room_statuses(screen)
            # Mask out rooms we've already clicked (handles max-affinity case
            # where schedule runs but no green check ever appears)
            masked_statuses = list(statuses)
            for idx in self._clicked_rooms_this_location:
                if 0 <= idx < len(masked_statuses):
                    masked_statuses[idx] = "done"
            best_slot = self._choose_best_room(masked_statuses)
            if any(s != "unknown" for s in statuses):
                skip = sorted(self._clicked_rooms_this_location)
                self.log(f"room statuses: {statuses} (skipping clicked: {skip})")

            # PRIORITY HUNT: when favorites are configured and no favorite was
            # found in this location's roster, try the next location before
            # falling back to pixel mode. Only accept a pixel-picked room after
            # we've cycled through every location once (or after enough ticks).
            _locations_to_try = max(1, _MAX_LOCATIONS_FULL_CIRCLE // 2)
            if (
                self._target_favorites
                and not self._target_found
                and self._locations_checked < _locations_to_try
                and best_slot >= 0
            ):
                self.log(
                    f"no favorite in '{cur_loc or '?'}' (checked "
                    f"{self._locations_checked}/{_locations_to_try} locations), "
                    f"switching to next location to hunt favorites"
                )
                return self._close_roster_action(
                    screen, "switch_location",
                    "hunt favorites at next location",
                )

            if best_slot >= 0:
                slot_name = _ROOM_SLOT_NAMES[best_slot]
                click_x, click_y = _ROOM_CLICK_POS[best_slot]
                hunt_done = (
                    not self._target_favorites
                    or self._locations_checked >= _locations_to_try
                )
                self.log(
                    f"pixel-selected room '{slot_name}' idx={best_slot} at "
                    f"({click_x:.3f},{click_y:.3f}) (fav-hunt exhausted={hunt_done})"
                )
                self._roster_open = False
                self._clicked_rooms_this_location.add(best_slot)
                self.sub_state = "execute"
                self._execute_ticks = 0
                self._start_clicked = False
                return action_click(click_x, click_y, f"click room slot {slot_name}")

            # All rooms done/locked/already-clicked, OR room-status pixel
            # check returned all-unknown (couldn't read the cards) → switch
            # to next location. The room picker is the preserved pixel/coord
            # logic above; we no longer OCR-click room names as a fallback
            # (iron rule: no OCR button finding).
            any_known = any(s != "unknown" for s in statuses)
            if any_known:
                done_count = sum(1 for s in statuses if s == "done")
                clicked_count = len(self._clicked_rooms_this_location)
                self.log(f"no available room (done={done_count}/{_NUM_ROOMS}, clicked={clicked_count}), switching location")
                return self._close_roster_action(screen, "switch_location", "no available rooms")

            self.log("room statuses all-unknown (pixel read failed), switching location")
            return self._close_roster_action(screen, "switch_location", "no rooms detected")

        # Not in roster overlay — open it if we haven't yet (YOLO SCHED_ALL btn).
        if not self._roster_open:
            full_tab = self._find_roster_open_btn(screen)
            if full_tab is not None:
                self.log(f"clicking '{full_tab.cls_name}' to view roster (YOLO)")
                self._roster_open = True
                self._roster_scan_ticks = 0
                return action_click_yolo(full_tab, "open full schedule roster (YOLO 全体课程表)")

            if self._wait_ticks < 2:
                self._wait_ticks += 1
                return action_wait(300, "looking for 全体课程表 cls")

            # Button cls not visible after grace ticks — surface the gap
            # rather than blind-clicking a hardcoded tab corner.
            self.log("full roster button cls not found — YOLO gap; switching location")
            self._roster_scan_ticks = 0
            self._wait_ticks = 0
            self._locations_checked += 1
            self.sub_state = "switch_location"
            return action_wait(300, "roster button cls missing, switching")

        # Roster was opened but overlay detection failed (YOLO miss).
        # Give it a few ticks grace period before giving up.
        self._roster_scan_ticks += 1
        retry_tab = self._find_roster_open_btn(screen)
        if retry_tab is not None and self._roster_scan_ticks <= 2:
            self.log("roster overlay missing after open, retrying roster button (YOLO)")
            return action_click_yolo(retry_tab, "retry open full schedule roster (YOLO 全体课程表)")
        if self._roster_scan_ticks <= 3:
            return action_wait(400, f"waiting for roster overlay to appear ({self._roster_scan_ticks}/3)")

        self.log("roster overlay not detected after opening, switching location")
        self._roster_open = False
        self._roster_scan_ticks = 0
        self._locations_checked += 1
        self.sub_state = "switch_location"
        return action_wait(300, "roster detection failed, switching")

    def _switch_location(self, screen: ScreenState) -> Dict[str, Any]:
        """Switch to next location using right arrow or back to Location Select."""
        self._switch_ticks += 1

        # If roster overlay is still showing, close it first
        if self._is_roster_overlay(screen):
            return self._close_roster_action(screen, "switch_location", "before switch")

        # If we somehow landed at Location Select, pick a different location
        if self._is_location_select(screen):
            self.sub_state = "select_location"
            self._clicked_rooms_this_location = set()
            if self._avatar_clf is not None:
                self._avatar_clf.reset_buffer(reason="back to location select")
            return action_wait(300, "back at location select")

        right = self._find_schedule_switch_arrow(screen)
        if right is not None:
            self.log("clicking right switch (YOLO 右切换)")
            self.sub_state = "check_roster"
            self._roster_open = False
            self._wait_ticks = -2  # cooldown: skip 2 ticks for location transition
            self._roster_scan_ticks = 0
            self._clicked_rooms_this_location = set()
            if self._avatar_clf is not None:
                self._avatar_clf.reset_buffer(reason="right-arrow location switch")
            return action_click_yolo(right, "right switch (YOLO 右切换)")

        # ARROW_RIGHT cls not visible. Rather than blind-click a hardcoded
        # right-edge coord (iron rule), back out to Location Select and pick
        # the next unvisited tile there — a clean state-machine path.
        if self._switch_ticks > 3:
            self.log("switch arrow cls missing, going back to location select")
            self.sub_state = "select_location"
            return action_back("back to location select (no ARROW_RIGHT cls)")

        self.log("no ARROW_RIGHT cls — YOLO gap; waiting")
        return action_wait(400, "looking for ARROW_RIGHT switch cls")

    def _execute(self, screen: ScreenState) -> Dict[str, Any]:
        """Wait for room info popup → click 開始 → animation → report → confirm → loop.

        Adapted from reference lesson.py start_lesson():
        After clicking a room card in 全體課程表 popup, the game opens
        a room info popup (課程表資訊) with a 開始 (Start) button at the bottom.
        Click 開始 → schedule animation plays → report popup → confirm → done.
        """
        self._execute_ticks += 1

        # Hard limit: if no start button after many building clicks, this location
        # likely has all rooms done. Switch to next location instead of looping.
        if self._execute_ticks > 8:
            self.log("no start button found after 8 ticks, switching location")
            self.sub_state = "switch_location"
            self._execute_ticks = 0
            self._start_clicked = False
            self._roster_open = False
            self._roster_scan_ticks = 0
            self._switch_ticks = 0
            return action_wait(300, "execute timeout, try next location")

        # ── Handle non-schedule screens (animation, bond popups, etc.) ──
        if not self._is_schedule(screen):
            if self.detect_screen_yolo(screen) == "Lobby":
                self.log("drifted to lobby during schedule execute, re-entering schedule")
                self._animation_ticks = 0
                self._execute_ticks = 0
                self._start_clicked = False
                self._roster_open = False
                self._roster_scan_ticks = 0
                self.sub_state = "enter"
                return action_wait(300, "re-enter schedule from lobby")

            # Start button (SCHED_START) visible mid-transition → proceed.
            start_visible = self.find_cls(screen, UC.SCHED_START, conf=0.30)
            if start_visible is not None:
                pass  # Fall through to start-button logic
            else:
                self._animation_ticks += 1
                if self._animation_ticks > 8:
                    return action_click(0.5, 0.5, "tap to skip animation")
                if self._animation_ticks > 15:
                    self._animation_ticks = 0
                    self._start_clicked = False
                    self.sub_state = "check_roster"
                    self._roster_open = False
                    self._roster_scan_ticks = 0
                    return action_wait(500, "animation done, back to roster")
                return action_wait(500, "waiting for schedule UI")

        self._animation_ticks = 0
        self._parse_tickets(screen)
        if self._tickets_remaining == 0:
            self.log("no tickets remaining, exiting")
            self.sub_state = "exit"
            return action_wait(300, "tickets exhausted")

        # ── PRIORITY 1: Report popup (schedule just finished) ──
        # The 課程表報告 result popup's only actionable is its blue 確認
        # button. Resolve it via YOLO BTN_CONFIRM in the center-bottom band.
        # (No SCHED_START/SCHED_ALL here yet — so a BTN_CONFIRM in this band
        # means the report popup, not a room-info start.)
        report_confirm = self.find_cls(
            screen, UC.BTN_CONFIRM, conf=0.30, region=(0.30, 0.55, 0.70, 0.92),
        )
        if report_confirm is not None and self.find_cls(screen, UC.SCHED_START, conf=0.30) is None:
            self.log(f"schedule report confirmed (ticket #{self._tickets_used}, YOLO 确认键)")
            self._start_clicked = False
            self._execute_ticks = 0
            self._roster_open = True  # game re-opens roster after report
            self._roster_scan_ticks = 0
            self.sub_state = "check_roster"
            return action_click_yolo(report_confirm, "confirm schedule report (YOLO 确认键)")

        # If we already clicked 開始, do not click it repeatedly every tick.
        # Wait a few ticks for transition/animation before retrying.
        if self._start_clicked:
            if self._execute_ticks > 4:
                self.log("start click didn't trigger, retrying")
                self._start_clicked = False
                self._execute_ticks = 0
            else:
                return action_wait(500, "waiting for schedule to start")

        # ── PRIORITY 2: Start button (room info popup open) — YOLO SCHED_START.
        start = self.find_cls(screen, UC.SCHED_START, conf=0.30)
        if start is not None:
            self.log(f"starting schedule #{self._tickets_used + 1} (YOLO 课程表开始)")
            self._tickets_used += 1
            self._roster_open = False
            self._start_clicked = True
            self._execute_ticks = 0
            return action_click_yolo(start, "start schedule (YOLO 课程表开始)")

        # ── PRIORITY 3: Location Select → re-enter ──
        if self._is_location_select(screen):
            self.sub_state = "select_location"
            self._execute_ticks = 0
            return action_wait(300, "back at location select")

        # ── PRIORITY 4: Roster visible → re-scan or wait ──
        if self._is_roster_overlay(screen):
            # Grace period: roster lingers briefly after room click.
            if self._execute_ticks <= 4:
                return action_wait(400, "roster still closing, waiting")
            # After grace: roster reappeared (game re-opens it after
            # schedule completes). Go back to check_roster to find
            # the next available room instead of closing it.
            self.log("roster reappeared after schedule, re-scanning")
            self._roster_open = True
            self._roster_scan_ticks = 0
            self._execute_ticks = 0
            self._start_clicked = False
            self.sub_state = "check_roster"
            return action_wait(300, "back to check_roster (roster reappeared)")

        # ── PRIORITY 5: No SCHED_START / report / roster cls visible ──
        # The room-info popup should have opened from the _ROOM_CLICK_POS
        # click in _check_roster. If SCHED_START never appears, the
        # _execute_ticks>8 guard above kicks us to the next location. We do
        # NOT blind-click avatar icons / map center (iron rule) — surface
        # the gap and wait.
        self.log("no SCHED_START/report/roster cls in execute — YOLO gap; waiting")
        return action_wait(400, "execute: waiting for SCHED_START cls")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        # On lobby iff YOLO nav-icon signature present.
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log("back in lobby, done")
            return action_done("schedule complete")
        # Prefer YOLO home/back button over blind ESC.
        home = self.find_cls(screen, UC.BTN_HOME, conf=0.30)
        if home is not None:
            return action_click_yolo(home, "schedule exit: home button (YOLO 回大厅)")
        back = self.find_cls(screen, UC.BTN_BACK, conf=0.30)
        if back is not None:
            return action_click_yolo(back, "schedule exit: back button (YOLO 返回键)")
        return action_back("schedule exit: ESC toward lobby")
