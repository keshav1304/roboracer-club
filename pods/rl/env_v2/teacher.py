"""The teacher: a good race line for any track, a speed plan that respects the friction, and pure pursuit.

It is used three ways, and none of them limits the final policy:
1. behaviour cloning, so PPO starts from a driver that already finishes laps (learning from scratch
   worked on 1 seed in 3 in our sweep),
2. a shaping term toward the teacher's line early in training (potential based, so it does not change
   which policy is best, and it is turned down to 0),
3. a baseline: the lap time the policy has to beat on each track.

Race line. Minimum curvature inside the track (keeping MARGIN from each wall): the line is
c_i + e_i n_i, e_i bounded by the widths, and the sum of squared second differences is minimised with a
bounded least squares. That is the classic outside-inside-outside line. Minimum curvature is close to
minimum time on tracks like ours; RL is left to find the rest (late apexes, where to brake).

Speed plan. v <= sqrt(f * lim(wheel) / |kappa|) where lim is the nominal grip by wheel angle
(contract.nominal_limit) and f a safety factor, then a forward pass with the friction-limited drive
(5.4 - 0.273 v m/s^2) and a backward pass with the throttle-0 brake (4.1 + 0.273 v), both reduced by
the share of grip the corner already uses (friction ellipse).
"""
from __future__ import annotations

import math
import os

import numpy as np

from contract import MAX_STEER, MAX_THROTTLE, WHEELBASE, commands_to_action, nominal_limit

MARGIN = 0.35            # m from the wall to the line's centre (car half width 0.155 + room)
V_PER_THR = 24.4         # settled speed per unit throttle (twin and AutoDRIVE)
A_DRIVE, A_BRAKE, DRAG = 5.4, 4.1, 0.273


def min_curvature_offsets(cx, cy, wr, wl, step=2, margin=MARGIN):
    """Lateral offsets e (m, + left) of the minimum-curvature line at every centre-line point."""
    from scipy.optimize import lsq_linear
    from scipy.sparse import lil_matrix
    idx = np.arange(0, len(cx), step)
    n = len(idx)
    px, py = cx[idx], cy[idx]
    t = np.stack([np.roll(px, -1) - np.roll(px, 1), np.roll(py, -1) - np.roll(py, 1)], 1)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nx, ny = -t[:, 1], t[:, 0]
    A = lil_matrix((2 * n, n))
    b = np.zeros(2 * n)
    for i in range(n):
        im, ip = (i - 1) % n, (i + 1) % n
        for comp, nn, pp in ((0, nx, px), (1, ny, py)):
            r = 2 * i + comp
            A[r, im] += nn[im]
            A[r, i] += -2 * nn[i]
            A[r, ip] += nn[ip]
            b[r] = -(pp[ip] - 2 * pp[i] + pp[im])
    lo = -(wr[idx] - margin)
    hi = wl[idx] - margin
    bad = lo > hi                                                      # narrower than 2 x margin: stay central
    lo[bad], hi[bad] = -0.01, 0.01
    sol = lsq_linear(A.tocsr(), b, bounds=(lo, hi), method='trf', lsmr_tol='auto', max_iter=200)
    e = np.interp(np.arange(len(cx)), np.append(idx, len(cx)), np.append(sol.x, sol.x[0]))
    return e


def speed_plan(xy, safety=0.85, v_top=0.32 * V_PER_THR):
    """Planned speed along a closed line (points 0.1 m apart)."""
    a, m, z = np.roll(xy, 3, 0), xy, np.roll(xy, -3, 0)
    cross = (m - a)[:, 0] * (z - a)[:, 1] - (m - a)[:, 1] * (z - a)[:, 0]
    k = 2 * np.abs(cross) / np.maximum(1e-9, np.linalg.norm(m - a, axis=1) * np.linalg.norm(z - m, axis=1)
                                       * np.linalg.norm(z - a, axis=1))
    wheel = np.arctan(WHEELBASE * k)
    lim = np.array([safety * nominal_limit(w) for w in wheel])
    v = np.minimum(v_top, np.sqrt(lim / np.maximum(k, 1e-6)))
    ds = np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1)
    n = len(xy)
    for _ in range(2):
        for i in range(n):                                             # accelerate
            j = (i + 1) % n
            use = min(1.0, v[i] ** 2 * k[i] / lim[i])
            acc = max(0.0, (A_DRIVE - DRAG * v[i]) * math.sqrt(max(0.0, 1 - use ** 2)))
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2 * acc * ds[i]))
        for i in range(n - 1, -1, -1):                                # brake
            j = (i + 1) % n
            use = min(1.0, v[j] ** 2 * k[j] / lim[j])
            dec = (A_BRAKE + DRAG * v[j]) * math.sqrt(max(0.05, 1 - use ** 2))
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * dec * ds[i]))
    return v, k


class Teacher:
    """Race line, speed plan and pure pursuit for one RaceTrack (any variant)."""

    def __init__(self, track, safety=0.85, margin=MARGIN, cache_dir=None):
        self.track = track
        self.ey = None
        path = None
        if cache_dir:                                      # workers share the optimised line through a file
            os.makedirs(cache_dir, exist_ok=True)
            path = os.path.join(cache_dir, f'{track.name}_n{track.n}_m{margin:.2f}.npy')
            if os.path.isfile(path):
                try:
                    self.ey = np.load(path)
                except (OSError, ValueError):
                    self.ey = None
        if self.ey is None or len(self.ey) != track.n:
            self.ey = min_curvature_offsets(track.cx, track.cy, track.wr, track.wl, margin=margin)
            if path:
                tmp = path + f'.{os.getpid()}.npy'
                np.save(tmp, self.ey)
                os.replace(tmp, path)
        lx = track.cx - self.ey * track.ty
        ly = track.cy + self.ey * track.tx
        xy = np.stack([lx, ly], 1)
        # resample the line itself every 0.1 m
        q = np.vstack([xy, xy[:1]])
        sd = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
        t = np.arange(0, sd[-1] - 1e-9, 0.1)
        self.xy = np.stack([np.interp(t, sd, q[:, 0]), np.interp(t, sd, q[:, 1])], 1)
        self.s = t
        self.length = float(sd[-1])
        self.v, self.k = speed_plan(self.xy, safety)
        self.lap_time = float(np.sum(0.1 / np.maximum(self.v, 0.1)))
        self._i = None

    def ey_at(self, i_centre: int) -> float:
        """Teacher's lateral offset at a centre-line index (for the shaping term)."""
        return float(self.ey[i_centre % len(self.ey)])

    def reset(self):
        self._i = None

    def commands(self, x, y, yaw, v, speed_scale=1.0):
        """Pure pursuit on the teacher line: (steering_command, throttle_command). Lookahead 0.5 m + 0.25 s
        of travel, as the club's pure pursuit."""
        n = len(self.xy)
        if self._i is None:
            d2 = (self.xy[:, 0] - x) ** 2 + (self.xy[:, 1] - y) ** 2
            idx = np.arange(n)
        else:
            idx = np.arange(self._i - 30, self._i + 31) % n
            d2 = (self.xy[idx, 0] - x) ** 2 + (self.xy[idx, 1] - y) ** 2
        i = int(idx[int(np.argmin(d2))])
        self._i = i
        la = 0.5 + 0.25 * max(0.0, v)
        tx, ty = self.xy[int((i + round(la / 0.1)) % n)]
        dx, dy = tx - x, ty - y
        xc, yc = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
        steer = math.atan(2.0 * yc / max(1e-6, xc * xc + yc * yc) * WHEELBASE) / MAX_STEER
        # look a little ahead for the speed target (delays)
        target = speed_scale * float(self.v[int((i + round(0.15 * max(v, 1.0) / 0.1)) % n)])
        thr = 0.0 if v > target + 0.4 else min(MAX_THROTTLE, target / V_PER_THR)
        return float(np.clip(steer, -1.0, 1.0)), float(thr)

    def action(self, x, y, yaw, v, use_envelope=True, speed_scale=1.0):
        steer, thr = self.commands(x, y, yaw, v, speed_scale)
        return commands_to_action(steer, thr, v, use_envelope)
