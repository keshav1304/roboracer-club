"""Reactive controllers for f1tenth_gym agents.

FollowTheGap: disparity extender + safety bubble + widest gap, steering via a
pure-pursuit law toward the chosen gap point, and a speed law from steering
angle and front clearance. Opponents show up in the scan, so it also steers
around cars ahead (a crude reactive overtake).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class FTGParams:
    v_max: float = 6.0            # m/s, straight-line cap
    v_min: float = 1.5            # m/s, floor while moving
    a_brake: float = 6.0          # m/s^2, used for the clearance speed cap
    wheelbase: float = 0.3302     # m (gym default lf + lr)
    max_steer: float = 0.4189     # rad (gym default s_max)
    fov_deg: float = 150.0        # only beams within ±fov/2 are considered (targets beyond ~75° are not drivable)
    max_range: float = 10.0       # m, ranges clipped to this
    smooth: int = 5               # moving-average window (beams)
    disparity: float = 0.4        # m, range jump treated as an edge
    car_half_width: float = 0.155 # m (gym width 0.31 / 2)
    margin: float = 0.25          # m, extra clearance added to the half-width (0.12 let passes rub)
    bubble: float = 0.35          # m, radius blanked around the closest point
    free_thresh: float = 1.0      # m, beams farther than this count as free
    deep_frac: float = 0.9        # target = centre of beams ≥ deep_frac·max in gap
    lookahead_min: float = 1.0    # m, pure-pursuit distance clip
    lookahead_max: float = 3.0
    front_cone_deg: float = 10.0  # cone used for the clearance speed cap
    stop_dist: float = 0.4        # m, clearance where the speed cap reaches 0
    side_safe: float = 0.50       # m, side clearance below which we steer away
    side_gain: float = 1.5        # rad per m of clearance deficit
    corridor_margin: float = 0.15 # m beyond half-width; anything inside this lane ahead caps speed
    clear_ok: float = 1.0         # m of arc clearance considered unblocked
    n_fan: int = 15               # candidate steering angles tried when blocked
    reverse_time: float = 0.6     # s of reversing when every arc is blocked
    reverse_speed: float = -1.0   # m/s
    blocked_wait: float = 0.3     # s blocked before reversing (a car in the way usually moves on)
    rear_clear: float = 0.6       # m, rear-side beams (|angle| > 100°) must be clear to reverse
    dt: float = 0.01              # s per plan() call (gym timestep)
    side_lo_deg: float = 25.0     # side sectors span side_lo..side_hi degrees
    side_hi_deg: float = 110.0


class FollowTheGap:
    def __init__(self, scan_angles: np.ndarray, params: FTGParams | None = None):
        self.p = params or FTGParams()
        self.angles = np.asarray(scan_angles, dtype=np.float64)
        self.inc = float(np.mean(np.diff(self.angles)))
        half = np.deg2rad(self.p.fov_deg) / 2
        self.window = np.where(np.abs(self.angles) <= half)[0]
        cone = np.deg2rad(self.p.front_cone_deg)
        self.front = np.where(np.abs(self.angles) <= cone)[0]
        lo, hi = np.deg2rad(self.p.side_lo_deg), np.deg2rad(self.p.side_hi_deg)
        ahead = np.abs(self.angles) < np.pi / 2
        self.cos_a, self.sin_a, self.ahead = np.cos(self.angles), np.sin(self.angles), ahead
        self.left = np.where((self.angles >= lo) & (self.angles <= hi))[0]
        self.right = np.where((self.angles <= -lo) & (self.angles >= -hi))[0]
        self.last_target_angle = 0.0
        self.reverse_left = 0.0  # s of recovery reversing remaining
        self.reverse_steer = 0.0
        self.blocked_for = 0.0
        self.rear = np.where(np.abs(self.angles) > np.deg2rad(100))[0]

    def _preprocess(self, scan: np.ndarray) -> np.ndarray:
        r = np.clip(scan[self.window].astype(np.float64), 0.0, self.p.max_range)
        k = self.p.smooth
        if k > 1:
            r = np.convolve(r, np.ones(k) / k, mode="same")
        return r

    def _extend_disparities(self, r: np.ndarray) -> np.ndarray:
        """At each range jump, copy the nearer range sideways over the beams
        the car's half-width (plus margin) would sweep at that distance."""
        out = r.copy()
        w = self.p.car_half_width + self.p.margin
        jumps = np.where(np.abs(np.diff(r)) > self.p.disparity)[0]
        for i in jumps:
            near_i, far_dir = (i, 1) if r[i] < r[i + 1] else (i + 1, -1)
            near = max(r[near_i], 0.05)
            n = int(np.ceil(np.arctan2(w, near) / self.inc))
            lo, hi = (near_i, min(near_i + n, len(r) - 1)) if far_dir > 0 else (max(near_i - n, 0), near_i)
            out[lo:hi + 1] = np.minimum(out[lo:hi + 1], near)
        return out

    def _arc_clearance(self, scan: np.ndarray, steer: float) -> float:
        """Distance along the steering arc to the first scan point inside a
        car-width lane around that arc."""
        p = self.p
        x = scan * self.cos_a
        y = scan * self.sin_a
        half = p.car_half_width + p.corridor_margin
        kappa = np.tan(steer) / p.wheelbase
        if abs(kappa) < 1e-3:
            lat, along = np.abs(y), x
        else:
            R = 1.0 / kappa                       # signed radius, centre at (0, R)
            lat = np.abs(np.hypot(x, y - R) - abs(R))
            along = np.abs(R) * np.arctan2(x, np.sign(R) * (R - y))
        lane = self.ahead & (lat < half) & (along > 0)
        return float(np.min(along[lane])) if np.any(lane) else p.max_range

    def plan(self, scan: np.ndarray, speed: float) -> tuple[float, float]:
        """Return (steering_angle [rad], speed [m/s])."""
        p = self.p
        if self.reverse_left > 0.0:   # recovery in progress
            self.reverse_left -= p.dt
            return self.reverse_steer, p.reverse_speed
        r = self._extend_disparities(self._preprocess(scan))

        # safety bubble around the closest point
        ci = int(np.argmin(r))
        n = int(np.ceil(np.arctan2(p.bubble, max(r[ci], 0.05)) / self.inc))
        r[max(ci - n, 0):ci + n + 1] = 0.0

        # widest run of free beams; ties broken toward the previous target
        free = r > p.free_thresh
        best, best_score = None, -np.inf
        i = 0
        while i < len(free):
            if free[i]:
                j = i
                while j + 1 < len(free) and free[j + 1]:
                    j += 1
                width = j - i + 1
                centre_ang = self.angles[self.window[(i + j) // 2]]
                score = width - 20.0 * abs(centre_ang - self.last_target_angle)
                if score > best_score:
                    best, best_score = (i, j), score
                i = j + 1
            else:
                i += 1

        if best is None:  # nothing free: aim at the farthest beam, crawl
            ti = int(np.argmax(r))
            target_ang, target_r = self.angles[self.window[ti]], r[ti]
        else:
            i, j = best
            seg = r[i:j + 1]
            deep = np.where(seg >= p.deep_frac * seg.max())[0]
            ti = i + int(deep[len(deep) // 2])
            target_ang, target_r = self.angles[self.window[ti]], r[ti]
        self.last_target_angle = target_ang

        # pure pursuit toward the target point
        ld = float(np.clip(target_r, p.lookahead_min, p.lookahead_max))
        steer = float(np.arctan2(2.0 * p.wheelbase * np.sin(target_ang), ld))

        # side-clearance repulsion: FTG alone steers away from a wall it is
        # running parallel to far too gently (this caused post-pass wall hits)
        dl = float(np.min(scan[self.left]))
        dr = float(np.min(scan[self.right]))
        if dl < p.side_safe:
            steer -= p.side_gain * (p.side_safe - dl)
        if dr < p.side_safe:
            steer += p.side_gain * (p.side_safe - dr)
        steer = float(np.clip(steer, -p.max_steer, p.max_steer))

        # speed: tiers by steering angle, capped by braking distance to the front
        a = abs(steer)
        frac = 1.0 if a < 0.05 else 0.8 if a < 0.12 else 0.6 if a < 0.22 else 0.45
        v = p.v_max * frac
        # clearance along our own lane: scan points inside a car-width corridor
        # ahead (a ±10° cone misses a car slightly off-centre, which made the
        # fast car squeeze past with ~5 cm to spare)
        # The lane follows the arc we're about to steer along; a straight lane
        # sees the outer wall of a hairpin as "in the way" and deadlocks.
        front = self._arc_clearance(scan, steer)

        # Blocked arc: try a fan of steering angles and take the clear one
        # closest to what we wanted (a hard stop here never recovered).
        if front < p.clear_ok:
            cands = np.linspace(-p.max_steer, p.max_steer, p.n_fan)
            clears = np.array([self._arc_clearance(scan, c) for c in cands])
            ok = clears >= p.clear_ok
            j = int(np.argmin(np.abs(cands - steer) + 10.0 * ~ok)) if ok.any() else int(np.argmax(clears))
            steer, front = float(cands[j]), float(clears[j])

        # Every arc blocked: hold, and if still blocked after blocked_wait with
        # the rear sides clear, reverse briefly with opposite lock. The LiDAR
        # can't see straight back, so never reverse with anything beside/behind.
        if front < p.stop_dist:
            self.blocked_for += p.dt
            if self.blocked_for >= p.blocked_wait and float(np.min(scan[self.rear])) > p.rear_clear:
                self.blocked_for = 0.0
                self.reverse_left = p.reverse_time
                self.reverse_steer = -steer
            return steer, 0.0
        self.blocked_for = 0.0
        v_cap = np.sqrt(max(2.0 * p.a_brake * (front - p.stop_dist), 0.0))
        v = float(np.clip(min(v, v_cap), p.v_min if front > 1.0 else 0.5, p.v_max))
        return steer, v
