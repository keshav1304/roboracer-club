"""Independent progress and checkpoint measurements; no control outputs.

ForwardRoute and CheckpointMonitor are evaluation tools, never control inputs.
These observers have no control output.
"""
import numpy as np


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class ForwardRoute:
    def __init__(self, track):
        line = track.raceline
        self.xy = np.column_stack([line.xs, line.ys]).astype(float)
        segments = np.linalg.norm(np.roll(self.xy, -1, axis=0) - self.xy, axis=1)
        self.s = np.r_[0., np.cumsum(segments)[:-1]]
        self.length = float(segments.sum())
        self.yaw = np.asarray(line.yaws, float)
        self.station = None
        self.start = None
        self.progress = 0.
        self.lap_times = []
        self.last_lap = 0.
        self.wrong_way_seconds = 0.
        self.max_heading_error = 0.

    def observe(self, observation, time):
        xy = np.array([observation["pose_x"], observation["pose_y"]], float)
        yaw = float(observation["pose_theta"])
        delta = None
        if self.station is None:
            candidates = np.arange(len(self.s))
        else:
            delta = wrap((self.s - self.station) / self.length * 2 * np.pi) * self.length / (2 * np.pi)
            candidates = np.flatnonzero((delta >= -1.) & (delta <= 3.))
        # Restrict projection locally so a nearby parallel lane cannot reverse
        # the reference direction or jump the car to another part of the lap.
        error = np.sum((self.xy[candidates] - xy) ** 2, axis=1)
        i = int(candidates[np.argmin(error)])
        tangent = np.array([np.cos(self.yaw[i]), np.sin(self.yaw[i])])
        projected = (self.s[i] + np.dot(xy - self.xy[i], tangent)) % self.length
        if self.station is not None:
            ds = float(wrap((projected - self.station) / self.length * 2 * np.pi) * self.length / (2 * np.pi))
            self.progress += ds
        else:
            self.start = projected
        self.station = projected
        heading_error = float(wrap(yaw - self.yaw[i]))
        self.max_heading_error = max(self.max_heading_error, abs(heading_error))
        forward_velocity = float(observation["linear_vel_x"]) * np.cos(heading_error)
        if forward_velocity < -.05:
            self.wrong_way_seconds += .01
        if self.progress >= (len(self.lap_times) + 1) * self.length:
            self.lap_times.append(round(time - self.last_lap, 2))
            self.last_lap = time

class CheckpointMonitor:
    """Independent ordered-gate validation, not used to produce controls."""
    def __init__(self, positions, headings, starting_xy):
        self.positions = np.asarray(positions, float)
        self.headings = np.asarray(headings, float)
        nearest = int(np.argmin(np.sum((self.positions - starting_xy) ** 2, axis=1)))
        signed = float((starting_xy - self.positions[nearest]) @ self.headings[nearest])
        self.next = nearest if signed < -.05 else (nearest + 1) % len(self.positions)
        self.previous = float((starting_xy - self.positions[self.next]) @ self.headings[self.next])
        self.crossings = 0
        self.laps = 0

    def observe(self, observation):
        xy = np.array([observation["pose_x"], observation["pose_y"]], float)
        relative = xy - self.positions[self.next]
        signed = float(relative @ self.headings[self.next])
        if self.previous < 0 <= signed and np.linalg.norm(relative) < 2.:
            if self.next == 0:
                self.laps += 1
            self.crossings += 1
            self.next = (self.next + 1) % len(self.positions)
            signed = float((xy - self.positions[self.next]) @ self.headings[self.next])
        self.previous = signed
