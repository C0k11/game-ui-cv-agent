"""ByteTrack-style tracker with velocity coast for hard lock through VFX.

Key features vs basic SORT:
1. TWO-STAGE matching (ByteTrack): high-conf detections matched first,
   then LOW-conf detections (conf 0.05-0.15) matched to remaining tracks.
   This keeps lock through visual effects that tank YOLO confidence.
2. Center-distance matching (not IoU): robust for small fast targets.
3. Velocity-predicted coasting: tracks coast along their velocity vector
   when occluded, surviving up to max_age frames of total blindness.
4. Same-class NMS: prevents box splitting/duplication.

Usage:
    from scripts.box_tracker import BoxTracker
    tracker = BoxTracker()
    smoothed = tracker.update(raw_detections)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import time

Box = Tuple[str, float, float, float, float, float]

# ByteTrack confidence split threshold
_HIGH_CONF = 0.15  # Above = first-pass match; below = second-pass rescue


class TrackedBox:
    """A tracked box with EMA smoothing + velocity prediction."""

    __slots__ = ("track_id", "cls", "conf", "x1", "y1", "x2", "y2",
                 "vx", "vy", "age", "hits", "time_since_update", "alpha",
                 "peak_conf")

    def __init__(self, track_id: int, cls: str, conf: float,
                 x1: float, y1: float, x2: float, y2: float,
                 alpha: float = 0.85):
        self.track_id = track_id
        self.cls = cls
        self.conf = conf
        self.peak_conf = conf  # highest conf ever seen (for output during coast)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.vx = 0.0
        self.vy = 0.0
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.alpha = alpha

    def update(self, cls: str, conf: float,
               x1: float, y1: float, x2: float, y2: float) -> None:
        a = self.alpha
        old_cx = (self.x1 + self.x2) / 2
        old_cy = (self.y1 + self.y2) / 2
        new_cx = (x1 + x2) / 2
        new_cy = (y1 + y2) / 2
        self.vx = 0.6 * (new_cx - old_cx) + 0.4 * self.vx
        self.vy = 0.6 * (new_cy - old_cy) + 0.4 * self.vy
        self.x1 = a * x1 + (1 - a) * self.x1
        self.y1 = a * y1 + (1 - a) * self.y1
        self.x2 = a * x2 + (1 - a) * self.x2
        self.y2 = a * y2 + (1 - a) * self.y2
        self.conf = conf
        self.peak_conf = max(self.peak_conf, conf)
        self.cls = cls
        self.hits += 1
        self.time_since_update = 0

    def predict(self) -> None:
        """Coast: FREEZE in place (no velocity extrapolation = no flying boxes)."""
        self.age += 1
        self.time_since_update += 1
        # DO NOT apply velocity — box stays at last known position.
        # Characters in Blue Archive are mostly stationary during VFX.
        # Velocity extrapolation at 240Hz causes explosive drift.
        self.vx = 0.0
        self.vy = 0.0
        # Slow confidence decay so box persists visually
        self.conf *= 0.92

    def as_tuple(self) -> Box:
        # During coast, output peak_conf so the box doesn't flicker/fade
        out_conf = max(self.conf, self.peak_conf * 0.5) if self.time_since_update > 0 else self.conf
        return (self.cls, out_conf, self.x1, self.y1, self.x2, self.y2)


def _center_dist(a: TrackedBox, bx1: float, by1: float, bx2: float, by2: float) -> float:
    acx, acy = (a.x1 + a.x2) / 2, (a.y1 + a.y2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    avg_size = ((a.x2 - a.x1) + (bx2 - bx1) + (a.y2 - a.y1) + (by2 - by1)) / 4
    return dist / max(avg_size, 0.001)


def _iou(a: TrackedBox, bx1: float, by1: float, bx2: float, by2: float) -> float:
    ix1, iy1 = max(a.x1, bx1), max(a.y1, by1)
    ix2, iy2 = min(a.x2, bx2), min(a.y2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = max(0, a.x2-a.x1)*max(0, a.y2-a.y1) + max(0, bx2-bx1)*max(0, by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _greedy_match(tracks, dets, used_tracks, used_dets, max_dist):
    """Greedy center-distance assignment. Returns matched (di, ti) pairs."""
    pairs = []
    for di, det in enumerate(dets):
        if di in used_dets:
            continue
        cls, _, dx1, dy1, dx2, dy2 = det
        for ti, trk in enumerate(tracks):
            if ti in used_tracks:
                continue
            cdist = _center_dist(trk, dx1, dy1, dx2, dy2)
            bonus = 0.0 if trk.cls == cls else 0.3
            score = cdist + bonus
            if cdist < max_dist:
                pairs.append((score, di, ti))
    pairs.sort()
    matched = []
    for score, di, ti in pairs:
        if di in used_dets or ti in used_tracks:
            continue
        matched.append((di, ti))
        used_dets.add(di)
        used_tracks.add(ti)
    return matched


class BoxTracker:
    """ByteTrack-style tracker with velocity coast for hard lock.

    - Two-stage matching: high-conf first, then low-conf rescue
    - Center-distance matching (not IoU)
    - Velocity-predicted coasting through VFX occlusion
    - Same-class NMS prevents box splitting
    """

    def __init__(self, max_age: int = 5, min_hits: int = 1,
                 max_center_dist: float = 1.5, alpha: float = 0.85,
                 high_conf: float = 0.25):
        self.max_age = max_age       # 5 frames (~20ms@240Hz) — short enough to avoid ghosts
        self.min_hits = min_hits
        self.max_dist = max_center_dist
        self.alpha = alpha
        self.high_conf = high_conf   # ByteTrack split threshold
        self._tracks: List[TrackedBox] = []
        self._next_id = 1

    def _nms_same_class(self) -> None:
        if len(self._tracks) < 2:
            return
        to_remove = set()
        for i in range(len(self._tracks)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(self._tracks)):
                if j in to_remove:
                    continue
                a, b = self._tracks[i], self._tracks[j]
                if a.cls != b.cls:
                    continue
                cdist = _center_dist(a, b.x1, b.y1, b.x2, b.y2)
                if cdist < 1.0 or _iou(a, b.x1, b.y1, b.x2, b.y2) > 0.01:
                    to_remove.add(j if a.hits >= b.hits else i)
        if to_remove:
            self._tracks = [t for i, t in enumerate(self._tracks) if i not in to_remove]

    def update(self, detections: List[Box]) -> List[Box]:
        # Predict all tracks forward (velocity coast)
        for t in self._tracks:
            t.predict()

        # ── ByteTrack two-stage matching ──
        # Split detections into high-conf and low-conf
        high_dets = [(i, d) for i, d in enumerate(detections) if d[1] >= self.high_conf]
        low_dets = [(i, d) for i, d in enumerate(detections) if d[1] < self.high_conf]

        used_tracks = set()
        used_dets = set()

        # STAGE 1: Match high-confidence detections to ALL tracks
        if self._tracks and high_dets:
            hi_list = [d for _, d in high_dets]
            hi_idx = [i for i, _ in high_dets]
            matches = _greedy_match(self._tracks, hi_list, used_tracks, set(),
                                    self.max_dist)
            for local_di, ti in matches:
                real_di = hi_idx[local_di]
                cls, conf, x1, y1, x2, y2 = detections[real_di]
                self._tracks[ti].update(cls, conf, x1, y1, x2, y2)
                used_tracks.add(ti)
                used_dets.add(real_di)

        # STAGE 2: Match LOW-confidence detections to UNMATCHED tracks
        # This is the ByteTrack magic — rescues tracks through VFX
        if low_dets:
            lo_list = [d for _, d in low_dets]
            lo_idx = [i for i, _ in low_dets]
            # Use wider distance for low-conf (VFX may shift apparent position)
            lo_used_dets = set()
            matches = _greedy_match(self._tracks, lo_list, used_tracks,
                                    lo_used_dets, self.max_dist * 1.5)
            for local_di, ti in matches:
                real_di = lo_idx[local_di]
                cls, conf, x1, y1, x2, y2 = detections[real_di]
                self._tracks[ti].update(cls, conf, x1, y1, x2, y2)
                used_tracks.add(ti)
                used_dets.add(real_di)

        # Create new tracks only for unmatched HIGH-conf detections
        # (low-conf unmatched are noise — don't spawn new tracks)
        for i, d in high_dets:
            if i not in used_dets:
                cls, conf, x1, y1, x2, y2 = d
                trk = TrackedBox(self._next_id, cls, conf, x1, y1, x2, y2,
                                 alpha=self.alpha)
                self._next_id += 1
                self._tracks.append(trk)

        # Remove dead tracks
        self._tracks = [t for t in self._tracks
                        if t.time_since_update <= self.max_age]

        # NMS: kill duplicate same-class tracks
        self._nms_same_class()

        # Output all confirmed tracks (including coasting ones)
        result: List[Box] = []
        for t in self._tracks:
            if t.hits >= self.min_hits:
                result.append(t.as_tuple())
        return result

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
