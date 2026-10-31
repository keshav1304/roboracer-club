"""Shared pieces: gym env, Frenet geometry on the raceline/centerline, and
ground-truth opponent perception with realistic noise.

Reuses the baseline in ../gym-two-car-reactive (its f1tenth_gym install, map
path, env factory and the FollowTheGap controller) instead of copying them.
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
BASELINE = HERE.parent / "gym-two-car-reactive"
sys.path.insert(0, str(BASELINE))

import run_two_cars as baseline  # noqa: E402  (make_env, start_poses, GYM_MAPS)
from reactive import FollowTheGap, FTGParams  # noqa: E402,F401

CAR_W, CAR_L = 0.31, 0.58
DT = 0.01


def make_env(map_name: str, render: bool = False, num_agents: int = 2):
    """Like the baseline's make_env, but the number of cars is configurable."""
    import gymnasium as gym
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": map_name, "num_agents": num_agents, "timestep": DT, "integrator": "rk4",
            "model": "st", "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": "features", "features": baseline.FEATURES},
        },
        render_mode="human" if render else None,
    )


class Line:
    """A closed reference line with Frenet projection (s along, d left-positive)."""

    def __init__(self, xs, ys, yaws, ss, ks=None, vxs=None):
        self.xy = np.stack([np.asarray(xs, float), np.asarray(ys, float)], 1)
        self.yaw = np.asarray(yaws, float)
        self.s = np.asarray(ss, float)
        seg = np.hypot(*(self.xy[0] - self.xy[-1]))
        self.L = float(self.s[-1] + seg)
        self.k = None if ks is None else np.asarray(ks, float)
        self.v = None if vxs is None else np.asarray(vxs, float)
        self.n = len(self.s)

    def project(self, x, y):
        """Return (s, d, idx) of the nearest point."""
        i = int(np.argmin(np.sum((self.xy - [x, y]) ** 2, 1)))
        dx, dy = x - self.xy[i, 0], y - self.xy[i, 1]
        c, sn = np.cos(self.yaw[i]), np.sin(self.yaw[i])
        s = (self.s[i] + dx * c + dy * sn) % self.L
        d = -dx * sn + dy * c
        return s, d, i

    def index_at(self, s):
        return int(np.searchsorted(self.s, s % self.L, side="right") - 1) % self.n

    def ds(self, s_from, s_to):
        """Signed shortest arc distance s_to - s_from in (-L/2, L/2]."""
        d = (s_to - s_from) % self.L
        return d - self.L if d > self.L / 2 else d


class TrackModel:
    """Raceline (what we drive) + centerline (track bounds) for a gym map.

    The shipped min-curvature raceline passes 0.2–0.3 m from walls (centre to
    wall), i.e. ~5 cm of body clearance, so any approach angle clips the wall.
    It is pulled in to keep `wall_margin` of body clearance (ForzaETH plans with
    a 0.7 m safety width for the same reason). The speed profile is kept.
    """

    def __init__(self, track, map_name, wall_margin=0.30):
        rl, cl = track.raceline, track.centerline
        self.center = Line(cl.xs, cl.ys, cl.yaws, cl.ss)
        csv = baseline.GYM_MAPS / map_name / f"{map_name}_centerline.csv"
        w = np.loadtxt(csv, delimiter=",", comments="#")
        # widths are per raw CSV point; map them onto the gym's resampled centerline
        raw = Line(w[:, 0], w[:, 1], np.zeros(len(w)), np.r_[0, np.cumsum(np.hypot(*np.diff(w[:, :2], axis=0).T))])
        idx = [raw.project(x, y)[2] for x, y in self.center.xy]
        self.w_right = w[idx, 2]
        self.w_left = w[idx, 3]

        xy = np.stack([np.asarray(rl.xs, float), np.asarray(rl.ys, float)], 1)
        for j, (x, y) in enumerate(xy.copy()):
            _, d, i = self.center.project(x, y)
            lo = -(self.w_right[i] - CAR_W / 2 - wall_margin)
            hi = self.w_left[i] - CAR_W / 2 - wall_margin
            dc = float(np.clip(d, lo, hi))
            n = np.array([-np.sin(self.center.yaw[i]), np.cos(self.center.yaw[i])])
            xy[j] = (x, y) - (d - dc) * n
        seg = np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T)
        ss = np.r_[0.0, np.cumsum(seg)[:-1]]
        yaw = np.arctan2(np.roll(xy[:, 1], -1) - np.roll(xy[:, 1], 1), np.roll(xy[:, 0], -1) - np.roll(xy[:, 0], 1))
        self.race = Line(xy[:, 0], xy[:, 1], yaw, ss, rl.ks, rl.vxs)

    def free_width(self, x, y):
        """Free lateral space on each side of a car centred at (x, y): (left, right) in m."""
        _, d, i = self.center.project(x, y)
        return self.w_left[i] - (d + CAR_W / 2), self.w_right[i] - (-d + CAR_W / 2)


class OpponentSensor:
    """Opponent perception stand-in until LiDAR detection + tracking is built.

    Ground truth plus slowly varying noise at ForzaETH's measured tracking
    accuracy (position RMSE 0.17 m, velocity RMSE 0.49 m/s), and, with
    fov=True, only when the real LiDAR could see it: some corner of the car
    inside the 270° fan, within range, and not hidden behind a wall (the beam
    toward it must reach it). Otherwise sense() returns None, like a tracker
    that has lost the car, e.g. when it drops into the blind wedge behind us.
    """

    FOV = 2.35619          # ±135°, AutoDRIVE / Hokuyo UST-10LX
    RANGE = 10.0           # m
    LIDAR_X = 0.0          # m ahead of the pose; the gym scans from the pose (the real car: 0.27)

    def __init__(self, pos_sigma=0.17, vel_sigma=0.49, tau=0.5, seed=0, fov=True):
        self.rng = np.random.default_rng(seed)
        self.ps, self.vs, self.a = pos_sigma, vel_sigma, np.exp(-DT / tau)
        self.e = np.zeros(3)  # x, y, v errors (AR(1), stationary std = sigma)
        self.fov = fov

    def visible(self, o, ego):
        from f1tenth_gym.envs.collision_models import get_vertices
        yaw = float(ego["pose_theta"])
        lx = float(ego["pose_x"]) + self.LIDAR_X * np.cos(yaw)
        ly = float(ego["pose_y"]) + self.LIDAR_X * np.sin(yaw)
        corners = get_vertices(np.array([o["pose_x"], o["pose_y"], o["pose_theta"]], np.float64), CAR_L, CAR_W)
        scan = np.asarray(ego["scan"], float)
        n = len(scan)
        for cx, cy in corners:
            dist = float(np.hypot(cx - lx, cy - ly))
            bearing = (np.arctan2(cy - ly, cx - lx) - yaw + np.pi) % (2 * np.pi) - np.pi
            if abs(bearing) > self.FOV or dist > self.RANGE:
                continue
            i = int(round((bearing + self.FOV) / (2 * self.FOV) * (n - 1)))
            if scan[i] > dist - 0.4:      # the beam gets to the car: not behind a wall
                return True
        return False

    def sense(self, o, ego=None):
        sig = np.array([self.ps, self.ps, self.vs])
        self.e = self.a * self.e + np.sqrt(1 - self.a ** 2) * sig * self.rng.standard_normal(3)
        if self.fov and ego is not None and not self.visible(o, ego):
            return None
        return {
            "x": float(o["pose_x"]) + self.e[0],
            "y": float(o["pose_y"]) + self.e[1],
            "yaw": float(o["pose_theta"]),
            "v": float(o["linear_vel_x"]) + self.e[2],
        }
