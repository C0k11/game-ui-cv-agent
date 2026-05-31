"""Textbook Kalman filter for bbox tracking (constant-velocity model).

The REAL thing — full predict/correct with a covariance matrix P and a
dynamically-computed Kalman gain K — not the constant-velocity + One-Euro
approximation in box_tracker.py.

State (8-D):  x = [cx, cy, w, h, vcx, vcy, vw, vh]
    box center x/y, width, height, and their velocities (units / second).
Measurement (4-D):  z = [cx, cy, w, h]
    YOLO gives position only; the filter infers velocity.

Per frame:
    PREDICT   x⁻ = F·x ;  P⁻ = F·P·Fᵀ + Q       (motion model + growing uncertainty)
    CORRECT   y  = z − H·x⁻                       (innovation: obs − prediction)
              S  = H·P⁻·Hᵀ + R                    (innovation covariance)
              K  = P⁻·Hᵀ·S⁻¹                      (Kalman gain — the magic)
              x  = x⁻ + K·y
              P  = (I − K·H)·P⁻

The gain K automatically balances trust between prediction and measurement:
  • noisy detections (large R) → small K → smoother (trust the model)
  • long occlusion (P grew)    → large K → snap onto the new detection fast
That adaptiveness is what fixed-alpha EMA cannot do and One-Euro only
approximates heuristically.

F and Q are rebuilt each step from the REAL dt, so the filter is frame-rate
independent. Lead-aim is free: extrapolate the state by F(lead_dt).

Coordinates are normalized 0-1, so the default noise scales are small. Tune
std_meas (how noisy YOLO boxes are) and std_vel (how fast targets accelerate)
to taste — those two dominate the feel.
"""
from __future__ import annotations
import numpy as np


class KalmanBox:
    """Single-target constant-velocity Kalman filter over a normalized bbox."""

    def __init__(self, cx: float, cy: float, w: float, h: float,
                 std_meas: float = 0.02, std_vel: float = 0.5,
                 std_proc_pos: float = 0.01):
        # state: [cx, cy, w, h, vcx, vcy, vw, vh]
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # H: observe position only (first 4 of 8)
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        # P: initial covariance — position known to ~std_meas, velocity unknown (big)
        self.P = np.eye(8, dtype=np.float64)
        self.P[:4, :4] *= std_meas ** 2
        self.P[4:, 4:] *= 10.0          # large initial velocity uncertainty

        # noise scales (Q/F rebuilt per step from dt)
        self._q_pos = std_proc_pos      # process noise on position
        self._q_vel = std_vel           # process noise on velocity (target maneuver)
        self.R = np.eye(4, dtype=np.float64) * (std_meas ** 2)
        self._I = np.eye(8, dtype=np.float64)

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(8, dtype=np.float64)
        F[0, 4] = F[1, 5] = F[2, 6] = F[3, 7] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        # diagonal discrete white-noise: pos & vel process variance scale with dt
        Q = np.zeros((8, 8), dtype=np.float64)
        qp = (self._q_pos * dt) ** 2
        qv = (self._q_vel * dt) ** 2
        for i in range(4):
            Q[i, i] = qp
            Q[i + 4, i + 4] = qv
        return Q

    def predict(self, dt: float) -> None:
        """Advance the motion model; uncertainty grows by Q."""
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def correct(self, cx: float, cy: float, w: float, h: float) -> None:
        """Fuse a measurement; gain K decides how much to trust it."""
        z = np.array([cx, cy, w, h], dtype=np.float64)
        y = z - self.H @ self.x                       # innovation
        S = self.H @ self.P @ self.H.T + self.R       # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain
        self.x = self.x + K @ y
        # Joseph-form covariance update (numerically stable, stays symmetric PSD)
        IKH = self._I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T

    @property
    def velocity(self) -> tuple:
        return float(self.x[4]), float(self.x[5])

    def box(self, lead_sec: float = 0.0) -> tuple:
        """Return (x1,y1,x2,y2). If lead_sec>0, extrapolate state forward by it
        (predictive lead-aim) — same F used by predict(), so it's consistent."""
        x = self.x
        if lead_sec > 0.0:
            x = self._F(lead_sec) @ x
        cx, cy, w, h = x[0], x[1], max(1e-4, x[2]), max(1e-4, x[3])
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def position_uncertainty(self) -> float:
        """Trace of the position covariance — grows during occlusion, shrinks
        on correction. Usable as an association gate or a 'lock quality' meter."""
        return float(self.P[0, 0] + self.P[1, 1])


# ── self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.RandomState(0)

    def center(b):
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    print("=== 1. static target + measurement noise → converges, low jitter ===")
    kf = KalmanBox(0.5, 0.5, 0.1, 0.1)
    last = []
    for _ in range(60):
        zx = 0.5 + rng.randn() * 0.01   # noisy obs around 0.5
        zy = 0.5 + rng.randn() * 0.01
        kf.predict(1 / 60)
        kf.correct(zx, zy, 0.1, 0.1)
        last.append(center(kf.box()))
    cx, cy = last[-1]
    jitter = np.std([c[0] for c in last[-20:]])
    print(f"   center=({cx:.4f},{cy:.4f}) expect ~0.5 | last-20 jitter={jitter:.5f}")
    print(f"   {'PASS' if abs(cx-0.5)<0.01 and jitter<0.008 else 'FAIL'}")

    print("=== 2. constant-velocity target → filter learns velocity ===")
    kf = KalmanBox(0.2, 0.5, 0.1, 0.1, std_vel=1.0)
    x = 0.2
    for _ in range(40):
        x += 0.01                       # 0.01/frame = 0.6/sec @60fps
        kf.predict(1 / 60)
        kf.correct(x, 0.5, 0.1, 0.1)
    vx, vy = kf.velocity
    print(f"   learned vx={vx:.3f} units/s (expect ~0.6) | {'PASS' if abs(vx-0.6)<0.15 else 'FAIL'}")

    print("=== 3. occlusion → predict-only coast keeps gliding, P grows ===")
    kf = KalmanBox(0.2, 0.5, 0.1, 0.1, std_vel=1.0)
    x = 0.2
    for _ in range(30):
        x += 0.01
        kf.predict(1 / 60); kf.correct(x, 0.5, 0.1, 0.1)
    p_before = kf.position_uncertainty()
    coast = []
    for _ in range(8):
        kf.predict(1 / 60)              # NO correct = occluded
        coast.append(center(kf.box())[0])
    p_after = kf.position_uncertainty()
    print(f"   coast x: {[round(c,3) for c in coast]}")
    print(f"   moved forward: {'PASS' if coast[-1] > coast[0] else 'FAIL'} | "
          f"P grew {p_before:.2e}→{p_after:.2e}: {'PASS' if p_after > p_before else 'FAIL'}")

    print("=== 4. lead-aim → box() with lead is ahead of current state ===")
    kf = KalmanBox(0.2, 0.5, 0.1, 0.1, std_vel=1.0)
    x = 0.2
    for _ in range(40):
        x += 0.01; kf.predict(1 / 60); kf.correct(x, 0.5, 0.1, 0.1)
    now_cx = center(kf.box(0.0))[0]
    lead_cx = center(kf.box(0.05))[0]   # 50ms lead
    print(f"   now={now_cx:.4f} lead50ms={lead_cx:.4f} ahead by {lead_cx-now_cx:+.4f} | "
          f"{'PASS' if lead_cx > now_cx else 'FAIL'}")

    print("=== 5. adaptive gain → occluded track snaps HARDER than a steady one ===")
    # steady-state: well-tracked, then 1 detection jumps to 0.7
    steady = KalmanBox(0.5, 0.5, 0.1, 0.1)
    for _ in range(40):
        steady.predict(1 / 60); steady.correct(0.5, 0.5, 0.1, 0.1)
    steady.predict(1 / 60); steady.correct(0.7, 0.5, 0.1, 0.1)
    steady_step = center(steady.box())[0] - 0.5
    # occluded: P grows for 10 frames, then the same 0.7 detection
    occl = KalmanBox(0.5, 0.5, 0.1, 0.1)
    for _ in range(40):
        occl.predict(1 / 60); occl.correct(0.5, 0.5, 0.1, 0.1)
    for _ in range(10):
        occl.predict(1 / 60)           # occlusion → P grows → K grows
    occl.predict(1 / 60); occl.correct(0.7, 0.5, 0.1, 0.1)
    occl_step = center(occl.box())[0] - 0.5
    print(f"   step toward new detection: steady={steady_step:.4f}  occluded={occl_step:.4f}")
    print(f"   occluded snaps harder (adaptive K): "
          f"{'PASS' if occl_step > steady_step else 'FAIL'}")

    print("\nALL KALMAN TESTS DONE")
