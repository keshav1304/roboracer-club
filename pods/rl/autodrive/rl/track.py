"""Centre line geometry: replaces f1tenth_gym's Track for the real simulator.

The reward needs s (arc length along the centre line) and ey (sideways offset, + left), which
race_env.py got from f1tenth_gym. The centre line files are the club's, unchanged, so s and ey
mean the same here as they did in the gym.
"""
from __future__ import annotations

import os

import numpy as np

TRACKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tracks')


class Track:
    def __init__(self, xy: np.ndarray):
        self.points = np.asarray(xy, dtype=np.float64)
        self.next_points = np.roll(self.points, -1, axis=0)
        self.seg = self.next_points - self.points
        self.seg_len = np.hypot(self.seg[:, 0], self.seg[:, 1])
        self.seg_len2 = self.seg_len ** 2
        self.s_at = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])
        self.length = float(self.seg_len.sum())

    @classmethod
    def load(cls, name: str, tracks_dir: str = TRACKS_DIR) -> 'Track':
        path = os.path.join(tracks_dir, name, f'{name}_centerline.csv')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'no centre line for track {name!r} at {path}')
        return cls(np.loadtxt(path, delimiter=',', comments='#')[:, :2])

    @classmethod
    def from_csv(cls, path: str) -> 'Track':
        """A line from a CSV of x_m, y_m (more columns are ignored), such as a race line."""
        return cls(np.loadtxt(path, delimiter=',', comments='#', ndmin=2)[:, :2])

    def _project(self, x: float, y: float, idx: np.ndarray) -> tuple[float, float]:
        px, py = x - self.points[idx, 0], y - self.points[idx, 1]
        t = np.clip((px * self.seg[idx, 0] + py * self.seg[idx, 1]) / self.seg_len2[idx], 0.0, 1.0)
        dx, dy = px - t * self.seg[idx, 0], py - t * self.seg[idx, 1]
        best = int(np.argmin(dx * dx + dy * dy))
        i = idx[best]
        s = float(self.s_at[i] + t[best] * self.seg_len[i])
        cross = self.seg[i, 0] * py[best] - self.seg[i, 1] * px[best]    # > 0 when the car is left of the line
        ey = float(np.hypot(dx[best], dy[best])) * (1.0 if cross >= 0 else -1.0)
        return s, ey

    def cartesian_to_frenet(self, x: float, y: float, s_guess: float | None = None,
                            window_m: float = 3.0) -> tuple[float, float]:
        """(s, ey). With s_guess, only look within window_m of it, so that where the track runs
        close to itself a car cannot jump to the neighbouring section."""
        n = len(self.points)
        if s_guess is None:
            return self._project(x, y, np.arange(n))
        centre = int(np.searchsorted(self.s_at, s_guess % self.length, side='right')) - 1
        half = max(1, int(window_m / (self.length / n)))
        return self._project(x, y, (centre + np.arange(-half, half + 1)) % n)

    def wrap_ds(self, s: float, prev_s: float) -> float:
        """Signed progress, handling the start line, as in race_env.py."""
        ds = s - prev_s
        if ds < -self.length / 2:
            ds += self.length
        elif ds > self.length / 2:
            ds -= self.length
        return ds
