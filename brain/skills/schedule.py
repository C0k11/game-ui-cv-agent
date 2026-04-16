"""ScheduleSkill: Handle daily schedule (課程表).

Game UI flow (observed from trajectory):
  - Schedule screen = "Location Select" showing locations like
    夏莱辦公室 RANK12, 夏莱居住區 RANK12, etc.
  - Click a location → enters the location detail view (isometric building)
  - Inside location: right ">" arrow to switch to next location
  - Click "全體課程表" button → opens roster overlay (rooms + students)
  - Check roster for target characters via avatar template matching
  - Close roster via X → click ">" to switch location if no target
  - To start: click center building area or "開始日程" button
  - Repeat until tickets exhausted (持有票券 = 0)

YOLO classes used (full.pt):
  - 角色头像 (character portrait) — used for avatar matching
  - 左切換 / 右切換 (switch arrows)
  - 锁 (lock icon)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote

from brain.skills.base import (
    BaseSkill, ScreenState, OcrBox,
    action_click, action_click_box,
    action_wait, action_back, action_done,
)

# Known location names (Traditional + Simplified Chinese)
_LOCATION_NAMES = [
    "夏莱公室", "夏莱辦公室", "夏莱办公室",
    "夏莱居住區", "夏莱居住区",
    "格黑娜學園", "格黑娜学园",
    "阿拜多斯", "千年研究",
    "百鬼夜行", "紅冬", "红冬",
    "山海經", "山海经", "瓦爾基里", "瓦尔基里",
    "SRT", "聯邦學園", "联邦学园",
]

# Hardcoded UI positions (normalized) observed from screenshots
_RIGHT_ARROW_POS = (0.99, 0.50)   # Right edge ">" arrow for location switch
_ROSTER_X_POS = (0.89, 0.14)      # X close button on roster overlay
_BUILD_CENTER_POS = (0.50, 0.50)  # Center hexagon in location detail
_ROSTER_TAB_POS = (0.84, 0.91)

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
# Legacy avatar matcher threshold (kept as fallback).
_AVATAR_MATCH_THRESHOLD = 0.45
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
        self._avatar_matcher = None
        self._best_match_score: float = -1.0
        self._stale_ticks: int = 0
        self._matched_room_idx: int = -1  # room index where favorite was found
        self._clicked_rooms_this_location: Set[int] = set()  # rooms clicked since last location switch

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
        self._best_match_score = -1.0
        self._stale_ticks = 0
        self._roster_scan_ticks = 0
        self._matched_room_idx = -1
        self._clicked_rooms_this_location = set()
        if self._target_favorites:
            self.log(f"target favorites loaded: {len(self._target_favorites)} characters")
        # Lazy-init avatar matcher
        if self._avatar_matcher is None and self._target_favorites:
            try:
                from vision.avatar_matcher import AvatarMatcher
                avatar_dir = Path(__file__).resolve().parents[2] / "data" / "captures" / "角色头像"
                self._avatar_matcher = AvatarMatcher(str(avatar_dir))
                self.log(f"avatar matcher loaded from {avatar_dir}")
            except Exception as e:
                self.log(f"avatar matcher init failed: {e}")
                self._avatar_matcher = None
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

        statuses = []
        room_idx = 0
        for ri, (ry1, ry2) in enumerate(self._AVATAR_ROWS_Y):
            for ci, (cx1, cx2) in enumerate(self._AVATAR_COLS_X):
                if room_idx >= _NUM_ROOMS:
                    break
                px1 = int(cx1 * w)
                py1 = int(ry1 * h)
                px2 = int(cx2 * w)
                py2 = int(ry2 * h)
                roi = img[py1:py2, px1:px2]
                if roi.size == 0:
                    statuses.append("unknown")
                    room_idx += 1
                    continue
                # Green checkmark: H=35-85, S>80, V>100 (bright green overlay)
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                green_mask = cv2.inRange(hsv, np.array([35, 80, 100]),
                                         np.array([85, 255, 255]))
                green_ratio = (green_mask.sum() / 255) / max(1, roi.shape[0] * roi.shape[1])
                if green_ratio > 0.005:
                    statuses.append("done")
                else:
                    statuses.append("available")
                room_idx += 1
            if room_idx >= _NUM_ROOMS:
                break
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

    def _check_avatar_status(self, roi) -> dict:
        """Detect green checkmark and heart+number on an avatar ROI via HSV.

        Returns dict with:
            'has_checkmark': bool  — green tick overlay (schedule already running)
            'has_heart': bool      — pink/red heart overlay (affection indicator)
        """
        import cv2
        import numpy as np
        if roi is None or roi.size == 0:
            return {"has_checkmark": False, "has_heart": False}
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, w = roi.shape[:2]
        pixels = h * w
        # Green checkmark: H=35-85, S>80, V>100 (bright green)
        green_mask = cv2.inRange(hsv, np.array([35, 80, 100]), np.array([85, 255, 255]))
        green_ratio = green_mask.sum() / 255 / max(1, pixels)
        # Red/pink heart: H=0-10 or H=160-180, S>80, V>80
        red_lo = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
        red_hi = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
        heart_mask = cv2.bitwise_or(red_lo, red_hi)
        heart_ratio = heart_mask.sum() / 255 / max(1, pixels)
        return {
            "has_checkmark": green_ratio > 0.02,  # 2% green pixels = checkmark
            "has_heart": heart_ratio > 0.01,       # 1% red pixels = heart
        }

    # reference template names differ from our config: Arisu→Aris, Bunny_Girl→Bunny, etc.
    # This map bridges reference template base names to our config base names.
    _reference_NAME_ALIASES = {
        "Aris": "Arisu", "Arisu": "Aris",
        "Hina": "Hina", "Ako": "Ako",
    }
    # reference suffix aliases: template suffix → our config suffix
    _reference_SUFFIX_ALIASES = {
        "Bunny": "Bunny_Girl", "Bunny_Girl": "Bunny",
        "Track": "Sportswear", "Sportswear": "Track",
        "Cheer Squad": "Cheerleader", "Cheerleader": "Cheer Squad",
        "Camp": "Camping", "Camping": "Camp",
        "Cycling": "Riding", "Riding": "Cycling",
    }

    def _build_fav_lookup(self) -> Set[str]:
        """Build a set of all name variants that count as 'favorite'.

        reference lesson_affection templates use names like 'Aris (Maid)',
        while our config uses 'Arisu_(Maid).png'. Build a bridge so
        template match results can be checked against favorites.
        """
        lookup: Set[str] = set()
        for raw in self._target_favorites:
            name = raw
            if name.lower().endswith(".png"):
                name = name[:-4]
            name = unquote(name)
            # Add as-is and common variants
            for variant in [name, name.replace("_", " ")]:
                lookup.add(variant)
                lookup.add(variant.lower())
            # Base name without suffix
            base = name.split("(")[0].split("_")[0].strip()
            if base:
                lookup.add(base)
                lookup.add(base.lower())
                # Add reference alias for base name
                alias = self._reference_NAME_ALIASES.get(base)
                if alias:
                    lookup.add(alias)
                    lookup.add(alias.lower())
            # Generate variants with base-name aliases and suffix aliases
            import re as _re
            m = _re.search(r"\((.+)\)", name)
            suffix = m.group(1) if m else None
            bases = [base]
            alias = self._reference_NAME_ALIASES.get(base)
            if alias:
                bases.append(alias)
            suffixes = [suffix] if suffix else []
            if suffix:
                alt_suffix = self._reference_SUFFIX_ALIASES.get(suffix)
                if alt_suffix:
                    suffixes.append(alt_suffix)
            # Cross-product: all base names × all suffixes
            for b in bases:
                for s in suffixes:
                    for fmt in [f"{b} ({s})", f"{b}_({s})"]:
                        lookup.add(fmt)
                        lookup.add(fmt.lower())
        return lookup

    # Avatar region positions in 全體課程表 popup (normalized 0-1).
    # Each room card's avatar strip: where the small face icons appear.
    # Row/column grid: 3×3 layout, last row only has 1 room.
    _AVATAR_ROWS_Y = [(0.35, 0.47), (0.54, 0.66), (0.73, 0.84)]
    _AVATAR_COLS_X = [(0.09, 0.30), (0.34, 0.55), (0.58, 0.79)]

    def _check_roster_avatars(self, screen: ScreenState) -> bool:
        """Check roster overlay for favorite characters using 角色头像_crop.

        For each available room in the 全體課程表 popup:
        1. Crop the avatar strip (where student faces are displayed)
        2. Match only favorite students via AvatarMatcher (template+histogram)
        3. If a favorite scores above threshold, mark that room

        Uses our own 角色头像_crop assets (not reference templates).
        """
        if not self._target_favorites:
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

        # Get room statuses to only scan available rooms
        statuses = self._get_room_statuses(screen)

        best_name = None
        best_score = -1.0
        best_room = -1

        # PRIMARY: use AvatarMatcher with 角色头像_crop (only favorites)
        if self._avatar_matcher is not None:
            fav_names = []
            for raw in self._target_favorites:
                name = raw
                if name.lower().endswith(".png"):
                    name = name[:-4]
                name = unquote(name)
                fav_names.append(name)

            room_idx = 0
            for ri, (ry1, ry2) in enumerate(self._AVATAR_ROWS_Y):
                for ci, (cx1, cx2) in enumerate(self._AVATAR_COLS_X):
                    if room_idx >= _NUM_ROOMS:
                        break
                    if statuses[room_idx] not in ("available", "unknown"):
                        room_idx += 1
                        continue
                    # Crop the avatar strip region
                    px1 = int(cx1 * w)
                    py1 = int(ry1 * h)
                    px2 = int(cx2 * w)
                    py2 = int(ry2 * h)
                    roi = img[py1:py2, px1:px2]
                    if roi.size == 0:
                        room_idx += 1
                        continue
                    # Match favorites against this region
                    matched, score = self._avatar_matcher.match_avatar(
                        roi, fav_names
                    )
                    if matched and score > _AVATAR_MATCH_THRESHOLD:
                        self.log(
                            f"room {room_idx} ({_ROOM_SLOT_NAMES[room_idx]}): "
                            f"★FAV '{matched}' score={score:.2f}"
                        )
                        if score > best_score:
                            best_name = matched
                            best_score = score
                            best_room = room_idx
                    room_idx += 1
                if room_idx >= _NUM_ROOMS:
                    break

        if best_name and best_room >= 0:
            self._target_found = True
            self._matched_room_idx = best_room
            self.log(
                f"FAVORITE MATCH: '{best_name}' score={best_score:.2f} "
                f"in room {best_room} ({_ROOM_SLOT_NAMES[best_room]})"
            )
            return True

        self.log(f"no favorite found in roster (scanned {sum(1 for s in statuses if s in ('available','unknown'))} rooms)")
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

    # ── Screen detection helpers ──

    def _is_schedule(self, screen: ScreenState) -> bool:
        """Detect schedule UI: header shows 課程 (traditional) or 课程 (simplified)."""
        return (
            screen.has_text("課程", region=(0.0, 0.0, 0.3, 0.10), min_conf=0.5)
            or screen.has_text("课程", region=(0.0, 0.0, 0.3, 0.10), min_conf=0.5)
            or screen.has_text("程表", region=(0.0, 0.0, 0.3, 0.10), min_conf=0.5)
        )

    def _is_location_select(self, screen: ScreenState) -> bool:
        """Detect Location Select screen (list of all locations).

        Must distinguish from location detail (inside one location).
        Location Select has "LocationSelect" text and/or multiple RANK entries.
        Location Detail only has ONE RANK at the top-right.
        """
        if screen.has_text("LocationSelect", min_conf=0.6):
            return True
        # Multiple RANK entries at different y positions → list of locations
        rank_hits = screen.find_text("RANK", min_conf=0.7)
        if len(rank_hits) >= 2:
            ys = sorted(b.cy for b in rank_hits)
            if ys[-1] - ys[0] > 0.10:
                return True
        return False

    def _is_roster_overlay(self, screen: ScreenState) -> bool:
        """Detect full roster overlay (全體課程表 shown as title at top center).

        Wider region and partial matches for MuMu emulator compatibility.
        """
        # Full title match (widened region)
        title = screen.find_any_text(
            ["全體課程表", "全体课程表", "全體課程", "全体课程", "全體程表", "全体程表"],
            region=(0.20, 0.02, 0.80, 0.25),
            min_conf=0.5
        )
        if title:
            return True
        partial_title = screen.find_any_text(
            ["程表"],
            region=(0.20, 0.02, 0.80, 0.25),
            min_conf=0.6
        )
        if partial_title and self._roster_open:
            return True
        if self._is_location_select(screen):
            return False
        # Fallback: roster overlay shows room rows ("教室" / "房間" text in center)
        # plus the overlay has a dark background with student avatars
        room_label = screen.find_any_text(
            ["教室", "房間", "房间"],
            region=(0.05, 0.15, 0.50, 0.85),
            min_conf=0.55
        )
        if room_label and self._roster_open:
            return True
        return False

    def _is_inside_location(self, screen: ScreenState) -> bool:
        """Detect location detail view (inside one location, isometric building).

        Has "全體課程表" button at bottom-right and schedule header, but is NOT
        the Location Select list (which has multiple RANK entries).
        """
        if not self._is_schedule(screen):
            return False
        if self._is_location_select(screen):
            return False
        btn = screen.find_any_text(
            ["全體課程表", "全体课程表"],
            region=(0.75, 0.85, 1.0, 1.0),
            min_conf=0.5
        )
        if btn:
            return True
        if screen.has_text("RANK", min_conf=0.7):
            return True
        return False

    def _find_schedule_close(self, screen: ScreenState) -> Optional[OcrBox]:
        """Find roster close button via OCR only (no Florence)."""
        return screen.find_text_one(r"^[Xx×]$", region=(0.72, 0.02, 0.94, 0.28), min_conf=0.7)

    def _find_schedule_switch_arrow(self, screen: ScreenState) -> Optional[OcrBox]:
        """Find right switch arrow via OCR only (no Florence)."""
        return screen.find_text_one(">", region=(0.90, 0.35, 1.0, 0.65), min_conf=0.7)

    def _close_roster_action(self, screen: ScreenState, next_state: str, reason: str) -> Dict[str, Any]:
        """Helper to close the roster overlay and transition to next_state.

        Uses YOLO X detection first, then hardcoded position fallback.
        (OCR X detection skipped per project rule: all X buttons via YOLO only.)
        """
        self._roster_open = False
        close_btn = self._find_schedule_close(screen)
        if close_btn:
            self.sub_state = next_state
            return action_click_box(close_btn, f"close roster X ({reason})")
        self.sub_state = next_state
        return action_click(*_ROSTER_X_POS, f"close roster X fallback ({reason})")

    # ── State handlers ──

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("schedule timeout")

        # ── Popup handling ──

        # Ticket exhausted popup
        no_ticket = screen.find_any_text(
            ["日程券不足", "日程券已用完", "沒有日程券", "没有日程券"],
            region=screen.CENTER,
            min_conf=0.6,
        )
        if no_ticket:
            confirm = screen.find_any_text(
                ["確認", "确认", "確", "确"],
                region=screen.CENTER, min_conf=0.7
            )
            if confirm:
                self.log(f"tickets exhausted ({self._tickets_used} used)")
                self.sub_state = "exit"
                return action_click_box(confirm, "confirm no tickets")
            close_btn = self._find_schedule_close(screen)
            if close_btn:
                self.sub_state = "exit"
                return action_click_box(close_btn, "close ticket popup")
            self.sub_state = "exit"
            return action_click(0.5, 0.8, "dismiss ticket popup")

        # Schedule result / reward popup — tap to dismiss
        result = screen.find_any_text(
            ["日程結果", "日程结果", "獲得獎勵", "获得奖励"],
            region=screen.CENTER, min_conf=0.6
        )
        if result:
            self.log("schedule result popup, tapping to dismiss")
            self._roster_open = False
            return action_click(0.5, 0.9, "dismiss schedule result")

        # Rank-up / bond level up popup (好感度升級 / 羈絆升級)
        # Schedule also raises bond, so these popups can appear here too.
        if screen.find_any_text(["好感度", "Rank Up"], min_conf=0.6):
            self.log("rank-up popup, tapping to dismiss")
            return action_click(0.5, 0.5, "dismiss rankup popup")

        # Bond level up screen (羈絆升級！) — full-screen animation
        bond_stat = screen.find_any_text(
            ["治愈力", "治癒力", "最大體力", "最大体力"],
            min_conf=0.6
        )
        if bond_stat:
            self.log("bond level up screen (stat text), tapping to dismiss")
            return action_click(0.5, 0.5, "dismiss bond level up")

        # Pre-level-up / bond notification screen
        if not self._is_schedule(screen):
            bond_notif = screen.find_any_text(
                ["羈絆升級", "鲜升級", "羈絆點數", "羈絆"],
                min_conf=0.6
            )
            if bond_notif:
                self.log("bond notification screen, tapping to dismiss")
                return action_click(0.5, 0.5, "dismiss bond notification")

        # Generic popups
        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        # Skip animation
        skip = screen.find_any_text(["SKIP", "Skip", "跳過", "跳过"], min_conf=0.7)
        if skip:
            return action_click_box(skip, "skip schedule animation")

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
        current = self.detect_current_screen(screen)
        roster_overlay = self._is_roster_overlay(screen)
        inside_location = self._is_inside_location(screen)
        location_select = self._is_location_select(screen)
        in_schedule = self._is_schedule(screen)

        if current == "Schedule" or in_schedule or roster_overlay or inside_location or location_select:
            self.log("inside schedule")
            self.sub_state = "check_roster" if (roster_overlay or inside_location) else "select_location"
            return action_wait(500, "entered schedule")

        if current == "Lobby":
            nav = self._nav_to(screen, ["課程表", "课程表", "課程", "课程", "程表"])
            if nav:
                return nav
            return action_wait(300, "waiting for schedule button")

        if current and current != "Schedule":
            self.log(f"wrong screen '{current}', backing out")
            return action_back(f"back from {current}")

        return action_wait(500, "entering schedule")

    def _select_location(self, screen: ScreenState) -> Dict[str, Any]:
        """On Location Select screen, click an unvisited location."""
        if not self._is_schedule(screen):
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
            for loc_name in _LOCATION_NAMES:
                loc = screen.find_text_one(loc_name, min_conf=0.6)
                if loc:
                    self.sub_state = "execute"
                    self._execute_ticks = 0
                    return action_click_box(loc, "enter location for fallback execute")
            rank_hits = screen.find_text("RANK", min_conf=0.7)
            if rank_hits:
                rank_hits.sort(key=lambda b: b.cy)
                target = rank_hits[0]
                self.sub_state = "execute"
                self._execute_ticks = 0
                return action_click(min(target.cx + 0.18, 0.75), target.cy, "enter location for fallback execute")
            self.log("no locations found, exiting")
            self.sub_state = "exit"
            return action_wait(300, "no locations found")

        # First entry: always start at 夏莱办公室 (the first location)
        if not self._starting_location:
            for loc_name in ["夏莱公室", "夏莱辦公室", "夏莱办公室"]:
                loc = screen.find_text_one(loc_name, min_conf=0.6)
                if loc:
                    self._starting_location = loc.text
                    self._visited_locations.add(loc.text)
                    self.log(f"starting at '{loc.text}' (will full-circle back)")
                    self.sub_state = "check_roster"
                    self._roster_open = False
                    self._wait_ticks = 0
                    return action_click_box(loc, f"enter location '{loc.text}'")
            # 夏莱 not visible — click first available location
            for loc_name in _LOCATION_NAMES:
                loc = screen.find_text_one(loc_name, min_conf=0.6)
                if loc:
                    self._starting_location = loc.text
                    self._visited_locations.add(loc.text)
                    self.log(f"starting at '{loc.text}' (夏莱 not visible)")
                    self.sub_state = "check_roster"
                    self._roster_open = False
                    self._wait_ticks = 0
                    return action_click_box(loc, f"enter location '{loc.text}'")
            # Fallback: click first RANK entry
            rank_hits = screen.find_text("RANK", min_conf=0.7)
            if rank_hits:
                rank_hits.sort(key=lambda b: b.cy)
                self._starting_location = "RANK"
                self.sub_state = "check_roster"
                self._roster_open = False
                self._wait_ticks = 0
                return action_click_box(rank_hits[0], "enter first location via RANK")

        # Subsequent entries from Location Select (after back from switch_location)
        for loc_name in _LOCATION_NAMES:
            loc = screen.find_text_one(loc_name, min_conf=0.6)
            if loc and loc.text not in self._visited_locations:
                self._visited_locations.add(loc.text)
                self.log(f"clicking location '{loc.text}' (new)")
                self.sub_state = "check_roster"
                self._roster_open = False
                self._wait_ticks = 0
                return action_click_box(loc, f"enter location '{loc.text}'")

        # All known names visited — click any RANK entry
        rank_hits = screen.find_text("RANK", min_conf=0.7)
        if rank_hits:
            rank_hits.sort(key=lambda b: b.cy)
            target = rank_hits[0]
            self.log(f"all locations visited, clicking RANK at ({target.cx:.2f},{target.cy:.2f})")
            self.sub_state = "check_roster"
            self._roster_open = False
            self._wait_ticks = 0
            return action_click(min(target.cx + 0.18, 0.75), target.cy, "enter location via RANK (all visited)")

        return action_wait(500, "looking for locations")

    def _detect_current_location(self, screen: ScreenState) -> str:
        """Detect the current location name from the building detail view header."""
        for loc_name in _LOCATION_NAMES:
            hit = screen.find_text_one(loc_name, region=(0.60, 0.10, 1.0, 0.25), min_conf=0.6)
            if hit:
                return hit.text
        return ""

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
        # try to re-open the 全體課程表 button
        if not roster_overlay and self._roster_open:
            self.log("roster was open but closed unexpectedly, re-opening")
            self._roster_open = False
            self._roster_scan_ticks = 0
            full_tab = screen.find_any_text(
                ["全體課程表", "全体课程表"],
                region=(0.60, 0.80, 1.0, 1.0), min_conf=0.5
            )
            if full_tab:
                return action_click_box(full_tab, "re-open roster after accidental close")
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

            if self._roster_scan_ticks == 2:
                if self._target_favorites:
                    self._check_roster_avatars(screen)
                return action_wait(300, "scanning roster avatars")

            # Tick 3+: click a room name in the popup to open room info
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
            if best_slot >= 0:
                slot_name = _ROOM_SLOT_NAMES[best_slot]
                click_x, click_y = _ROOM_CLICK_POS[best_slot]
                self.log(f"pixel-selected room '{slot_name}' idx={best_slot} at ({click_x:.3f},{click_y:.3f})")
                self._roster_open = False
                self._clicked_rooms_this_location.add(best_slot)
                self.sub_state = "execute"
                self._execute_ticks = 0
                self._start_clicked = False
                return action_click(click_x, click_y, f"click room slot {slot_name}")

            # All rooms done/locked/already-clicked → switch to next location
            any_known = any(s != "unknown" for s in statuses)
            if any_known:
                done_count = sum(1 for s in statuses if s == "done")
                clicked_count = len(self._clicked_rooms_this_location)
                self.log(f"no available room (done={done_count}/{_NUM_ROOMS}, clicked={clicked_count}), switching location")
                return self._close_roster_action(screen, "switch_location", "no available rooms")

            # OCR fallback: find room names via OCR and click one (prefer higher tier = later in list)
            _ROOM_NAMES = [
                "視聽室", "體育館", "圖書館",
                "教室", "實驗室", "射擊場",
                "載具庫",
                # Simplified variants
                "视听室", "体育馆", "图书馆",
                "实验室", "射击场", "载具库",
            ]
            found_rooms = []
            for name in _ROOM_NAMES:
                hit = screen.find_text_one(name, region=(0.08, 0.15, 0.92, 0.85), min_conf=0.40)
                if hit:
                    found_rooms.append(hit)

            if found_rooms:
                # Click the last found room (higher tier rooms are listed later)
                target = found_rooms[-1]
                self.log(f"clicking room '{target.text}' at ({target.cx:.3f},{target.cy:.3f}) in roster popup")
                self._roster_open = False
                self.sub_state = "execute"
                self._execute_ticks = 0
                self._start_clicked = False
                return action_click_box(target, f"click room {target.text}")

            # No room names found — switch to next location
            self.log("no room names found in roster, switching location")
            return self._close_roster_action(screen, "switch_location", "no rooms found")

        # Not in roster overlay — open it if we haven't yet
        if not self._roster_open:
            full_tab = screen.find_any_text(
                ["全體課程表", "全体课程表"],
                region=(0.60, 0.80, 1.0, 1.0),
                min_conf=0.5
            )
            if full_tab:
                self.log(f"clicking '{full_tab.text}' to view roster")
                self._roster_open = True
                self._roster_scan_ticks = 0
                return action_click_box(full_tab, "open full schedule roster")

            if self._wait_ticks < 2:
                self._wait_ticks += 1
                return action_wait(300, "looking for 全體課程表")

            self.log("full roster button not found, clicking roster fallback area")
            self._roster_open = True
            self._roster_scan_ticks = 0
            self._wait_ticks = 0
            return action_click(*_ROSTER_TAB_POS, "open full schedule roster fallback")

        # Roster was opened but overlay detection failed (YOLO/OCR miss).
        # Give it a few ticks grace period before giving up.
        self._roster_scan_ticks += 1
        retry_tab = screen.find_any_text(
            ["全體課程表", "全体课程表"],
            region=(0.60, 0.80, 1.0, 1.0),
            min_conf=0.5
        )
        if retry_tab and self._roster_scan_ticks <= 2:
            self.log("roster overlay missing after open, retrying roster button")
            return action_click_box(retry_tab, "retry open full schedule roster")
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
            return action_wait(300, "back at location select")

        right = self._find_schedule_switch_arrow(screen)
        if right:
            self.log("clicking right switch")
            self.sub_state = "check_roster"
            self._roster_open = False
            self._wait_ticks = -2  # cooldown: skip 2 ticks for location transition
            self._roster_scan_ticks = 0
            self._clicked_rooms_this_location = set()
            return action_click_box(right, "right switch")

        # Hardcoded fallback: click right arrow position (visible as ">" on right edge)
        if self._switch_ticks <= 3:
            self.log("clicking right arrow at hardcoded position")
            self.sub_state = "check_roster"
            self._roster_open = False
            self._wait_ticks = -2  # cooldown: skip 2 ticks for location transition
            self._roster_scan_ticks = 0
            self._clicked_rooms_this_location = set()
            return action_click(*_RIGHT_ARROW_POS, "right arrow (hardcoded)")

        # If still stuck after many attempts, go back to Location Select
        if self._switch_ticks > 6:
            self.log("switch stuck, going back to location select")
            self.sub_state = "select_location"
            return action_back("back to location select (switch stuck)")

        return action_wait(400, "looking for switch arrows")

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
            if screen.is_lobby():
                self.log("drifted to lobby during schedule execute, re-entering schedule")
                self._animation_ticks = 0
                self._execute_ticks = 0
                self._start_clicked = False
                self._roster_open = False
                self._roster_scan_ticks = 0
                self.sub_state = "enter"
                return action_wait(300, "re-enter schedule from lobby")

            start_visible = screen.find_any_text(
                ["課程表開始", "课程表開始", "课程表开始", "開始", "开始"],
                min_conf=0.5,
            )
            if start_visible:
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
        # OCR often misses the stylized 確認 button on blue background.
        # If we see the report title, click the confirm button at known position.
        report_popup = screen.find_any_text(
            ["課程表報告", "课程表报告"],
            region=(0.30, 0.10, 0.70, 0.30), min_conf=0.5
        )
        report_confirm = screen.find_any_text(
            ["確認", "确认"],
            region=(0.30, 0.60, 0.70, 0.90), min_conf=0.5
        )
        if report_popup or report_confirm:
            self.log(f"schedule report confirmed (ticket #{self._tickets_used})")
            self._start_clicked = False
            self._execute_ticks = 0
            self._roster_open = True  # game re-opens roster after report
            self._roster_scan_ticks = 0
            self.sub_state = "check_roster"
            if report_confirm:
                return action_click_box(report_confirm, "confirm schedule report")
            # OCR missed 確認 button — click at known position (center-bottom of popup)
            return action_click(0.50, 0.82, "confirm schedule report (hardcoded)")

        # If we already clicked 開始, do not click it repeatedly every tick.
        # Wait a few ticks for transition/animation before retrying.
        if self._start_clicked:
            if self._execute_ticks > 4:
                self.log("start click didn't trigger, retrying")
                self._start_clicked = False
                self._execute_ticks = 0
            else:
                return action_wait(500, "waiting for schedule to start")

        # ── PRIORITY 2: Start button (room info popup open) ──
        start = screen.find_any_text(
            ["課程表開始", "课程表開始", "课程表开始",
             "開始日程", "开始日程", "START"],
            min_conf=0.5,
        )
        if not start:
            start = screen.find_any_text(
                ["開始", "开始"],
                region=(0.25, 0.60, 0.75, 0.95),
                min_conf=0.6,
            )
        if start:
            self.log(f"starting schedule #{self._tickets_used + 1}: '{start.text}'")
            self._tickets_used += 1
            self._roster_open = False
            self._start_clicked = True
            self._execute_ticks = 0
            return action_click_box(start, "start schedule")

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

        # ── PRIORITY 5: Click avatar icons on location map ──
        # Avatar icons on the isometric map are the clickable elements.
        # Each avatar has a small affection number (10-30) below it.
        # Click directly ON the avatar icon (slightly above the number).
        avatar_nums = screen.find_text(
            r"^\d{1,2}$", region=(0.05, 0.15, 0.95, 0.90), min_conf=0.45
        )
        avatar_nums = [b for b in avatar_nums if b.text.isdigit() and 5 <= int(b.text) <= 40]
        if avatar_nums:
            idx = (self._execute_ticks - 1) % len(avatar_nums)
            target = avatar_nums[idx]
            # Click slightly above the number — the avatar icon is just above it
            click_y = max(0.08, target.cy - 0.04)
            self.log(f"clicking avatar icon '{target.text}' at ({target.cx:.2f},{click_y:.2f})")
            return action_click(target.cx, click_y,
                                f"click avatar {target.text}")

        # Fallback: click center area
        return action_click(0.50, 0.45, "click map center fallback")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log("back in lobby, done")
            return action_done("schedule complete")
        return action_back("schedule exit: back to lobby")
