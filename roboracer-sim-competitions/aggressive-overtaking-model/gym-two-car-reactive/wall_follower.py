"""Left-wall distance/heading PD control with local LiDAR corner preview.

The PD loop is adapted from the repository's ROS wall_follow.py node.
WallPreview checks feasible turns while retaining a wall-standoff objective.
No gap selection, saved map, waypoint or vehicle pose is used for control.
"""
from dataclasses import dataclass
import numpy as np
from reactive import FollowTheGap, FTGParams
from wall_preview import WallPreview


@dataclass
class WallParams:
    v_max: float = 2.0
    desired_dist: float = 0.7
    lookahead: float = 0.65
    kp: float = 1.4
    kd: float = 0.02
    dt: float = 0.01
    derivative_tau: float = 0.1
    max_steer: float = 0.4189
    brake_accel: float = 3.0
    stop_dist: float = 0.5


class WallFollower:
    def __init__(self, scan_angles, params=None):
        self.p = params or WallParams()
        self.angles = np.asarray(scan_angles)
        # Only the arc geometry helper is used; its gap planner is not called.
        self.clearance = FollowTheGap(scan_angles, FTGParams(corridor_margin=.04))
        self.wall_sector = (self.angles > np.deg2rad(25)) & (self.angles < np.deg2rad(160))
        self.kernel = np.ones(5) / 5
        self.previous_error = None
        self.derivative = 0.
        self.preview = WallPreview(self.p.desired_dist)
        self.tick = 0
        self.command = (0., 0.)

    def plan(self, scan, speed=0.):
        p = self.p
        ranges = np.clip(np.nan_to_num(np.asarray(scan, float), nan=10., posinf=10., neginf=0.), 0, 30)
        if np.max(ranges) < .35:
            self.tick = 0
            self.command = (0., 0.)
            return self.command

        # Estimate distance and tangent from the same local left-wall patch.
        filtered = np.convolve(np.pad(ranges, 2, mode="edge"), self.kernel, mode="valid")
        nearest = float(np.min(filtered[self.wall_sector]))
        patch = self.wall_sector & (filtered < nearest + .015)
        bearing = float(np.mean(self.angles[patch]))
        heading = bearing - np.pi / 2
        distance = float(np.mean(filtered[patch]))
        error = p.desired_dist - distance - p.lookahead * np.sin(heading)
        raw_derivative = 0. if self.previous_error is None else (error - self.previous_error) / p.dt
        self.derivative += p.dt / (p.derivative_tau + p.dt) * (raw_derivative - self.derivative)
        self.previous_error = error
        wall_steering = float(np.clip(-p.kp * error - p.kd * self.derivative,
                                      -p.max_steer, p.max_steer))
        if self.tick % 10 == 0:
            self.command = self.preview.plan(ranges, self.angles, wall_steering, p.v_max)
        self.tick += 1
        steering, preview_speed = self.command
        # Check fresh scan clearance even between 10 Hz preview updates.
        clearance = self.clearance._arc_clearance(ranges, steering)
        braking_cap = np.sqrt(max(2 * p.brake_accel * (clearance - p.stop_dist), 0.))
        return steering, float(min(preview_speed, braking_cap))
