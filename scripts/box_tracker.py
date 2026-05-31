"""Industrial-grade lock tracker: ByteTrack association + One-Euro smoothing +
Kalman-style constant-velocity coast + predictive lead-aim.

This is the "aimbot-feel" tracker. Three things make a lock look external-grade:

1. ASSOCIATION that survives VFX — ByteTrack two-stage (high-conf first, then
   low-conf 0.05-0.15 rescue) + center-distance matching. (Kept from v1.)
2. SMOOTH yet RESPONSIVE position — the One-Euro filter: heavy smoothing when
   the target is slow (kills jitter), light smoothing when fast (kills lag).
   This is the AR/VR-industry standard answer to the jitter-vs-latency tradeoff,
   replacing the old fixed-alpha EMA.
3. PREDICTION that hides end-to-end latency — the display box is drawn at where
   the target WILL be (pos + velocity × latency), not where it was N ms ago.
   Velocity is tracked in px/sec (time-aware, frame-rate independent) and the
   lead is clamped so noisy velocity can't fling the box ("explosive drift" the
   old freeze-coast was avoiding — solved properly with cap+decay, not by
   disabling prediction).

Occlusion handling: when a track gets no detection it COASTS along its (decayed,
capped) velocity instead of freezing — short occlusions (a few frames) keep
gliding naturally; the velocity decays so a long blind coast settles instead of
flying off; after max_age with no detection the track dies.

Backward-compat: lead_ms=0 + coast_decay=0 + a fixed beta≈0 reproduces the old
freeze-EMA behaviour, so this is a safe drop-in.

Usage:
    from scripts.box_tracker import BoxTracker
    tracker = BoxTracker()                       # sensible aimbot defaults
    smoothed = tracker.update(raw_detections)    # call once per frame
    # smoothed boxes are lead-predicted; pass dt for true time-awareness:
    smoothed = tracker.update(raw_detections, dt=elapsed_seconds)
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import math

Box = Tuple[str, float, float, float, float, float]

# ByteTrack confidence split threshold (module-level default)
_HIGH_CONF = 0.25  # Above = first-pass match; below = second-pass rescue

# Nominal frame interval used when caller doesn't pass dt. 240Hz overlay.
_DEFAULT_DT = 1.0 / 240.0


class _OneEuro:
    """One-Euro filter for a single scalar (Casiez et al. 2012).

    Cutoff frequency rises with speed: slow signal → low cutoff → strong
    smoothing (no jitter); fast signal → high cutoff → little smoothing (no
    lag). `min_cutoff` sets the floor smoothing; `beta` sets how aggressively
    smoothing relaxes with speed.
    """

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "_x_prev", "_dx_prev", "_started")

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.7,
                 d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev = 0.0
        self._dx_prev = 0.0
        self._started = False

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def reset(self, x: float) -> None:
        self._x_prev = x
        self._dx_prev = 0.0
        self._started = True

    def filter(self, x: float, dt: float) -> float:
        if not self._started:
            self.reset(x)
            return x
        # derivative of the signal, low-pass filtered
        dx = (x - self._x_prev) / max(dt, 1e-6)
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        # speed-dependent cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class TrackedBox:
    """A tracked box: One-Euro-smoothed corners + px/sec velocity + lead-aim.

    Position is stored as the smoothed top-left/bottom-right corners. Velocity
    is the center velocity in normalized-units/second (time-aware). `predict`
    advances the box by its velocity (coast) when no detection arrives.
    """

    __slots__ = ("track_id", "cls", "conf", "x1", "y1", "x2", "y2",
                 "vx", "vy", "age", "hits", "time_since_update",
                 "peak_conf", "_fx1", "_fy1", "_fx2", "_fy2",
                 "coast_decay", "vel_cap")

    def __init__(self, track_id: int, cls: str, conf: float,
                 x1: float, y1: float, x2: float, y2: float,
                 min_cutoff: float = 1.0, beta: float = 0.7,
                 coast_decay: float = 0.85, vel_cap: float = 3.0):
        self.track_id = track_id
        self.cls = cls
        self.conf = conf
        self.peak_conf = conf
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.vx = 0.0
        self.vy = 0.0
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.coast_decay = coast_decay   # per-frame velocity decay while coasting
        self.vel_cap = vel_cap           # max |velocity| in units/sec (anti-fling)
        # One-Euro filter per corner
        self._fx1 = _OneEuro(min_cutoff, beta)
        self._fy1 = _OneEuro(min_cutoff, beta)
        self._fx2 = _OneEuro(min_cutoff, beta)
        self._fy2 = _OneEuro(min_cutoff, beta)
        for f, v in ((self._fx1, x1), (self._fy1, y1), (self._fx2, x2), (self._fy2, y2)):
            f.reset(v)

    def update(self, cls: str, conf: float,
               x1: float, y1: float, x2: float, y2: float, dt: float) -> None:
        old_cx = (self.x1 + self.x2) / 2
        old_cy = (self.y1 + self.y2) / 2
        # One-Euro smooth each corner (adaptive: slow=smooth, fast=responsive)
        self.x1 = self._fx1.filter(x1, dt)
        self.y1 = self._fy1.filter(y1, dt)
        self.x2 = self._fx2.filter(x2, dt)
        self.y2 = self._fy2.filter(y2, dt)
        new_cx = (self.x1 + self.x2) / 2
        new_cy = (self.y1 + self.y2) / 2
        # velocity in units/second (time-aware), IIR-smoothed, capped
        inst_vx = (new_cx - old_cx) / max(dt, 1e-6)
        inst_vy = (new_cy - old_cy) / max(dt, 1e-6)
        self.vx = 0.6 * inst_vx + 0.4 * self.vx
        self.vy = 0.6 * inst_vy + 0.4 * self.vy
        self._clamp_velocity()
        self.conf = conf
        self.peak_conf = max(self.peak_conf, conf)
        self.cls = cls
        self.hits += 1
        self.time_since_update = 0

    def _clamp_velocity(self) -> None:
        speed = math.hypot(self.vx, self.vy)
        if speed > self.vel_cap and speed > 1e-9:
            s = self.vel_cap / speed
            self.vx *= s
            self.vy *= s

    def predict(self) -> None:
        """Bookkeeping only: age the track. Called on EVERY track each frame
        BEFORE matching. Position/velocity are NOT touched here — a matched
        track is corrected in update(), an unmatched track coasts in coast().
        (Standard SORT predict/update separation — keeps velocity from
        decaying while we're still tracking.)"""
        self.age += 1
        self.time_since_update += 1

    def coast(self, dt: float) -> None:
        """Advance an UNMATCHED track along its (decayed, capped) velocity.

        Unlike the old freeze (vx=vy=0), this keeps the box gliding through a
        short occlusion. Velocity decays each coast frame so a long blind coast
        settles instead of flying off; the cap already bounds a single step."""
        step_x = self.vx * dt
        step_y = self.vy * dt
        self.x1 += step_x
        self.y1 += step_y
        self.x2 += step_x
        self.y2 += step_y
        # decay velocity so coasting settles; keep One-Euro state in sync so a
        # re-acquired detection doesn't snap from a stale filter position
        self.vx *= self.coast_decay
        self.vy *= self.coast_decay
        self._fx1.reset(self.x1)
        self._fy1.reset(self.y1)
        self._fx2.reset(self.x2)
        self._fy2.reset(self.y2)
        # confidence decay so a coasted box fades if it never re-acquires
        self.conf *= 0.92

    def as_tuple(self, lead_sec: float = 0.0, lead_cap: float = 0.06) -> Box:
        """Output box, optionally lead-predicted by `lead_sec` of velocity.

        lead_sec = total end-to-end latency to compensate (capture+infer+render+
        display). The center is pushed forward by velocity*lead_sec, clamped to
        ±lead_cap (normalized units) so noisy velocity can't fling the box.
        """
        out_conf = (max(self.conf, self.peak_conf * 0.5)
                    if self.time_since_update > 0 else self.conf)
        if lead_sec <= 0.0:
            return (self.cls, out_conf, self.x1, self.y1, self.x2, self.y2)
        dx = self.vx * lead_sec
        dy = self.vy * lead_sec
        # clamp lead so a noisy velocity spike can't throw the box across screen
        dx = max(-lead_cap, min(lead_cap, dx))
        dy = max(-lead_cap, min(lead_cap, dy))
        return (self.cls, out_conf,
                self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)


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
    """Industrial lock tracker.

    - Two-stage ByteTrack association (high-conf then low-conf rescue)
    - One-Euro adaptive smoothing per corner (jitter-free + responsive)
    - Time-aware px/sec velocity with cap + decay coast through occlusion
    - Predictive lead-aim on output (compensates end-to-end latency)
    - Same-class NMS prevents box splitting

    Tuning knobs (aimbot feel):
      lead_ms        — how far ahead to predict; set ≈ measured end-to-end
                       latency (capture+infer+render+display). 0 = no lead.
      min_cutoff     — One-Euro floor: lower = smoother at rest (more jitter
                       kill), higher = snappier.
      beta           — One-Euro speed relax: higher = less lag when moving.
      coast_decay    — per-frame velocity decay while coasting (occluded).
      max_age        — frames a track survives with no detection.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 1,
                 max_center_dist: float = 1.5,
                 high_conf: float = _HIGH_CONF,
                 lead_ms: float = 40.0,
                 min_cutoff: float = 1.2, beta: float = 0.6,
                 coast_decay: float = 0.85, vel_cap: float = 3.0,
                 lead_cap: float = 0.06):
        self.max_age = max_age            # 30 frames (~125ms@240Hz) coast budget
        self.min_hits = min_hits
        self.max_dist = max_center_dist
        self.high_conf = high_conf
        self.lead_sec = max(0.0, lead_ms / 1000.0)
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.coast_decay = coast_decay
        self.vel_cap = vel_cap
        self.lead_cap = lead_cap
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

    def update(self, detections: List[Box], dt: Optional[float] = None) -> List[Box]:
        """Advance one frame. `dt` = seconds since last update (time-aware
        velocity / smoothing / lead). If omitted, assumes 240Hz."""
        if dt is None or dt <= 0:
            dt = _DEFAULT_DT

        # Age all tracks (bookkeeping only — no position/velocity change yet)
        for t in self._tracks:
            t.predict()

        # ── ByteTrack two-stage matching ──
        high_dets = [(i, d) for i, d in enumerate(detections) if d[1] >= self.high_conf]
        low_dets = [(i, d) for i, d in enumerate(detections) if d[1] < self.high_conf]

        used_tracks = set()
        used_dets = set()

        # STAGE 1: high-confidence detections → all tracks
        if self._tracks and high_dets:
            hi_list = [d for _, d in high_dets]
            hi_idx = [i for i, _ in high_dets]
            matches = _greedy_match(self._tracks, hi_list, used_tracks, set(),
                                    self.max_dist)
            for local_di, ti in matches:
                real_di = hi_idx[local_di]
                cls, conf, x1, y1, x2, y2 = detections[real_di]
                self._tracks[ti].update(cls, conf, x1, y1, x2, y2, dt)
                used_tracks.add(ti)
                used_dets.add(real_di)

        # STAGE 2: low-confidence detections → unmatched tracks (ByteTrack rescue)
        if low_dets:
            lo_list = [d for _, d in low_dets]
            lo_idx = [i for i, _ in low_dets]
            matches = _greedy_match(self._tracks, lo_list, used_tracks,
                                    set(), self.max_dist * 1.5)
            for local_di, ti in matches:
                real_di = lo_idx[local_di]
                cls, conf, x1, y1, x2, y2 = detections[real_di]
                self._tracks[ti].update(cls, conf, x1, y1, x2, y2, dt)
                used_tracks.add(ti)
                used_dets.add(real_di)

        # Coast tracks that got NO detection this frame (matched ones were
        # already corrected by update()). This is the occlusion glide.
        for ti, t in enumerate(self._tracks):
            if ti not in used_tracks:
                t.coast(dt)

        # New tracks only for unmatched HIGH-conf detections
        for i, d in high_dets:
            if i not in used_dets:
                cls, conf, x1, y1, x2, y2 = d
                trk = TrackedBox(self._next_id, cls, conf, x1, y1, x2, y2,
                                 min_cutoff=self.min_cutoff, beta=self.beta,
                                 coast_decay=self.coast_decay, vel_cap=self.vel_cap)
                self._next_id += 1
                self._tracks.append(trk)

        # Remove dead tracks
        self._tracks = [t for t in self._tracks
                        if t.time_since_update <= self.max_age]

        # NMS: kill duplicate same-class tracks
        self._nms_same_class()

        # Output confirmed tracks, lead-predicted
        result: List[Box] = []
        for t in self._tracks:
            if t.hits >= self.min_hits:
                result.append(t.as_tuple(self.lead_sec, self.lead_cap))
        return result

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
