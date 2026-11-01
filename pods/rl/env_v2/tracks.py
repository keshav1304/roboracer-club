"""Tracks for env_v2: the three AutoDRIVE tracks we have, their mirror and reverse copies, and generated ones.

Why so many. A policy trained on one track learns that track (which LiDAR picture means which action),
not how to drive. Training on many shapes, with each one also mirrored and run backwards, forces the
policy to learn corners in general, and evens out left and right turns. ICRA stays out of training by
default (HOLDOUT), so its lap time tells us how the policy does on a track it has never seen.

Files use the gym's track format: NAME_map.png, NAME_map.yaml, NAME_centerline.csv (x_m, y_m,
w_tr_right_m, w_tr_left_m). Generated tracks go to GEN_DIR the first time they are asked for, from a
fixed seed per track, so every machine builds the same ones.

    python tracks.py --build NAME MAP.pgm MAP.yaml GUIDE.csv   # a new AutoDRIVE map (needs opencv)
    python tracks.py --generate 200      # pre-builds generated tracks and prints their stats
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
V1_TRACKS = os.path.join(HERE, '..', 'env', 'tracks')
V2_TRACKS = os.path.join(HERE, 'tracks')
GEN_DIR = os.environ.get('ENV_V2_TRACK_CACHE', os.path.join(V2_TRACKS, 'generated'))
# IROS 2026 track-20260731 is a second SLAM map of the practice track (same 30.8 m loop), so it is not listed.
REAL = {'icra': os.path.join(V1_TRACKS, 'icra'), 'practice': os.path.join(V1_TRACKS, 'practice')}
HOLDOUT = ('icra',)
RES = 0.05


# ---- geometry helpers ------------------------------------------------------------------------------
def resample(p, h=0.1):
    q = np.vstack([p, p[:1]])
    sd = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
    t = np.arange(0, sd[-1] - 1e-9, h)
    return np.stack([np.interp(t, sd, q[:, 0]), np.interp(t, sd, q[:, 1])], 1)


def smooth(p, k):
    if k <= 1:
        return p
    pad = np.vstack([p[-k:], p, p[:k]])
    return np.stack([np.convolve(pad[:, j], np.ones(k) / k, 'same')[k:-k] for j in (0, 1)], 1)


def tangents(p):
    t = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    return t / np.maximum(1e-9, np.linalg.norm(t, axis=1, keepdims=True))


def curvature(p):
    a, m, z = np.roll(p, 2, 0), p, np.roll(p, -2, 0)
    cross = (m - a)[:, 0] * (z - a)[:, 1] - (m - a)[:, 1] * (z - a)[:, 0]
    den = np.linalg.norm(m - a, axis=1) * np.linalg.norm(z - m, axis=1) * np.linalg.norm(z - a, axis=1)
    return 2 * cross / np.maximum(den, 1e-9)                                  # signed, + = left turn


def _catmull_rom_closed(ctrl, n_per=40):
    out = []
    n = len(ctrl)
    for i in range(n):
        p0, p1, p2, p3 = ctrl[(i - 1) % n], ctrl[i], ctrl[(i + 1) % n], ctrl[(i + 2) % n]
        for t in np.linspace(0, 1, n_per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


# ---- generated tracks ----------------------------------------------------------------------------
def _hull(pts):
    """Convex hull, counter-clockwise (monotone chain)."""
    p = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def generate_shape(rng: np.random.Generator):
    """A random closed centre line with AutoDRIVE-like proportions: 30 to 80 m long, 1.3 to 2.4 m wide,
    long straights, sweepers and hairpins down to ~0.6 m radius. Returns (xy every 0.1 m, widths) or None."""
    for _ in range(400):
        # random points -> convex hull -> push edge midpoints in and out (S-bends, hairpins, chicanes)
        n = int(rng.integers(8, 20))
        pts = rng.uniform(0, 1, (n, 2)) * np.array([1.0, rng.uniform(0.25, 1.0)])
        ctrl = _hull(pts)
        for _round in range(int(rng.integers(1, 3))):
            new = []
            for i in range(len(ctrl)):
                p, q = ctrl[i], ctrl[(i + 1) % len(ctrl)]
                e = q - p
                nrm = np.array([e[1], -e[0]])                                 # outward for a CCW hull
                new.append(p)
                new.append(0.5 * (p + q) + nrm * rng.uniform(-0.45, 0.25))
            ctrl = np.array(new)
        a = rng.uniform(0, np.pi)
        ctrl = ctrl @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        xy = _catmull_rom_closed(ctrl, 20)
        length = np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T).sum()
        xy = xy * (rng.uniform(28.0, 75.0) / length)
        xy = smooth(resample(xy, 0.1), 7)
        xy = resample(xy, 0.1)
        if len(xy) < 250:
            continue
        w0 = rng.uniform(1.3, 2.4)
        s = np.arange(len(xy)) * 0.1
        w = w0 * (1 + 0.15 * np.sin(2 * np.pi * rng.integers(1, 5) * s / s[-1] + rng.uniform(0, 6.3)))
        k = np.abs(curvature(xy))
        if k.max() > 1 / 0.45 or k.max() * (w.max() / 2) > 1.15:         # ICRA's hairpin is 0.49 m at the centre
            continue
        # walls of far-apart parts must not touch: centre lines at least w + 0.4 m apart
        idx = np.arange(0, len(xy), 3)
        d = np.linalg.norm(xy[idx, None, :] - xy[None, idx, :], axis=2)
        sep = np.abs(idx[:, None] - idx[None, :])
        sep = np.minimum(sep, len(xy) - sep) * 0.1
        need = 0.5 * (w[idx, None] + w[None, idx]) + 0.4
        if np.any((sep > 4.0) & (d < need)):
            continue
        return xy, w
    return None


def write_track(out_dir, name, xy, w_right, w_left, margin=1.5):
    """Rasterise a tube of the given widths into the gym's files."""
    os.makedirs(out_dir, exist_ok=True)
    lo = np.minimum(xy - np.maximum(w_right, w_left)[:, None], 1e9).min(0) - margin
    hi = (xy + np.maximum(w_right, w_left)[:, None]).max(0) + margin
    W, H = (int(math.ceil((hi[0] - lo[0]) / RES)), int(math.ceil((hi[1] - lo[1]) / RES)))
    img = Image.new('L', (W, H), 0)
    drw = ImageDraw.Draw(img)
    fine = resample(xy, 0.03)
    j = np.minimum(np.round(np.arange(len(fine)) * len(xy) / len(fine)).astype(int), len(xy) - 1)
    nrm = np.stack([-tangents(fine)[:, 1], tangents(fine)[:, 0]], 1)
    for p, n, wr, wl in zip(fine, nrm, w_right[j], w_left[j]):
        r = 0.5 * (wr + wl)
        c = p + 0.5 * (wl - wr) * n                                           # off-centre if widths differ
        cx, cy = (c[0] - lo[0]) / RES, H - 1 - (c[1] - lo[1]) / RES
        drw.ellipse([cx - r / RES, cy - r / RES, cx + r / RES, cy + r / RES], fill=255)
    img.save(os.path.join(out_dir, f'{name}_map.png'))
    with open(os.path.join(out_dir, f'{name}_map.yaml'), 'w') as f:
        f.write(f'image: {name}_map.png\nresolution: {RES}\norigin: [{lo[0]:.4f}, {lo[1]:.4f}, 0.0]\n'
                'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
    np.savetxt(os.path.join(out_dir, f'{name}_centerline.csv'), np.column_stack([xy, w_right, w_left]),
               delimiter=', ', fmt='%.4f', header='x_m, y_m, w_tr_right_m, w_tr_left_m')


def generated_name(i: int) -> str:
    return f'gen{i:04d}'


def ensure_generated(i: int) -> str:
    """Folder of generated track i, built on first use from seed 7919 * i."""
    name = generated_name(i)
    folder = os.path.join(GEN_DIR, name)
    if not os.path.isfile(os.path.join(folder, f'{name}_centerline.csv')):
        rng = np.random.default_rng(7919 * i + 17)
        res = None
        while res is None:
            res = generate_shape(rng)
        xy, w = res
        tmp = folder + f'.tmp{os.getpid()}'
        write_track(tmp, name, xy, w / 2, w / 2)
        try:
            os.replace(tmp, folder)                                   # another worker may have won the race
        except OSError:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    return folder


# ---- IROS track from the competition map -------------------------------------------------------
def build_from_map(name, pgm, yaml_path, guide_csv, out_dir):
    """Same method as ../env/tracks/build_tracks.py (Rubi), for any AutoDRIVE map with a guide line:
    keep the free region the guide line is in, take the centre line from the distance-transform ridge
    along the guide's normals, widths along the normals, start at the guide's first point."""
    import cv2
    meta = dict(ln.split(': ', 1) for ln in open(yaml_path).read().splitlines() if ': ' in ln)
    res = float(meta['resolution'])
    ox, oy = (float(v) for v in meta['origin'].strip('[]').split(',')[:2])
    grid = cv2.imread(pgm, cv2.IMREAD_UNCHANGED)
    H, W = grid.shape
    to_rc = lambda x, y: (int(round(H - 1 - (y - oy) / res)), int(round((x - ox) / res)))  # noqa: E731
    line = np.loadtxt(guide_csv, delimiter=',', comments='s')[:, 1:3]
    line = resample(line, 0.1)
    _, lab = cv2.connectedComponents((grid > 250).astype(np.uint8), connectivity=4)
    seed = to_rc(*line[0])
    free = (lab == lab[seed]).astype(np.uint8)
    n_, blob, stats, _ = cv2.connectedComponentsWithStats(1 - free, connectivity=8)
    specks = [k for k in range(1, n_) if stats[k, cv2.CC_STAT_AREA] * res * res < 0.1]
    free[np.isin(blob, specks)] = 1
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * res

    def inside(x, y):
        r, c = to_rc(x, y)
        return 0 <= r < H and 0 <= c < W and free[r, c]

    def walk(p, n, sign, limit=1.5):
        out = []
        for k in range(1, int(limit / 0.025) + 1):
            t = sign * 0.025 * k
            x, y = p + t * n
            if not inside(x, y):
                return out, abs(t)
            r, c = to_rc(x, y)
            out.append((t, dist[r, c]))
        return out, limit

    c = line.copy()
    for _ in range(3):
        nrm = np.stack([-tangents(c)[:, 1], tangents(c)[:, 0]], 1)
        new = []
        for p, n in zip(c, nrm):
            r, cc = to_rc(*p)
            cand = [(0.0, dist[r, cc])] + walk(p, n, +1)[0] + walk(p, n, -1)[0]
            new.append(p + max(cand, key=lambda tv: tv[1])[0] * n)
        c = resample(smooth(np.array(new), 9))
    i0 = int(np.argmin(np.hypot(*(c - line[0]).T)))
    c = np.roll(c, -i0, axis=0)
    if np.dot(c[3] - c[0], line[3] - line[0]) < 0:
        c = c[::-1]
    nrm = np.stack([-tangents(c)[:, 1], tangents(c)[:, 0]], 1)
    w_left = np.array([walk(p, n, +1, 3.0)[1] for p, n in zip(c, nrm)])
    w_right = np.array([walk(p, n, -1, 3.0)[1] for p, n in zip(c, nrm)])
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f'{name}_map.png'), free * 255)
    with open(os.path.join(out_dir, f'{name}_map.yaml'), 'w') as f:
        f.write(f'image: {name}_map.png\nresolution: {res}\norigin: [{ox}, {oy}, 0.0]\n'
                'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
    np.savetxt(os.path.join(out_dir, f'{name}_centerline.csv'), np.column_stack([c, w_right, w_left]),
               delimiter=', ', fmt='%.4f', header='x_m, y_m, w_tr_right_m, w_tr_left_m')
    np.savetxt(os.path.join(out_dir, f'{name}_line.csv'), line, delimiter=', ', fmt='%.4f', header='x_m, y_m')
    return c, w_right, w_left


# ---- runtime track -----------------------------------------------------------------------------
class RaceTrack:
    """A track as env_v2 uses it: occupancy, LiDAR, centre line with Frenet projection, optional
    mirror (x -> -x) and reverse (run the other way). The teacher line is attached by teacher.py."""

    def __init__(self, folder: str, mirror: bool = False, reverse: bool = False, lidar_beams: int = 1081,
                 fov: float = 4.712389, max_range: float = 10.0):
        from f1tenth_gym.envs.lidar import ScanSimulator2D
        from types import SimpleNamespace

        from f1tenth_gym.envs.track.track import Track
        stem = os.path.basename(os.path.normpath(folder))
        self.base_name, self.folder = stem, folder
        self.name = stem + ('_m' if mirror else '') + ('_r' if reverse else '')
        self.mirror, self.reverse = mirror, reverse
        meta = dict(ln.split(': ', 1) for ln in open(os.path.join(folder, f'{stem}_map.yaml')).read().splitlines()
                    if ': ' in ln)
        res = float(meta['resolution'])
        ox, oy = (float(v) for v in meta['origin'].strip('[]').split(',')[:2])
        img = Image.open(os.path.join(folder, meta['image'].strip())).convert('L').transpose(Image.FLIP_TOP_BOTTOM)
        occ = np.array(img).astype(np.float32)
        occ = np.where(occ > 128, 255.0, 0.0).astype(np.float32)
        d = np.loadtxt(os.path.join(folder, f'{stem}_centerline.csv'), delimiter=',', comments='#')
        xy, wr, wl = d[:, :2].copy(), d[:, 2].copy(), d[:, 3].copy()
        if mirror:                                                     # x -> -x: left and right swap
            occ = occ[:, ::-1].copy()
            ox = -ox - occ.shape[1] * res
            xy[:, 0] = -xy[:, 0]
            wr, wl = wl, wr
        if reverse:                                                    # run the other way: order flips, sides swap
            xy, wr, wl = xy[::-1].copy(), wl[::-1].copy(), wr[::-1].copy()
        self.res, self.ox, self.oy = res, ox, oy
        spec = SimpleNamespace(name=self.name, resolution=res, origin=(ox, oy, 0.0))   # all the LiDAR reads
        self.gym_track = Track(spec=spec, occupancy_map=occ)
        self.scanner = ScanSimulator2D(lidar_beams, fov, max_range=max_range)
        self.scanner.set_map(self.gym_track)
        self.edt = np.ascontiguousarray(self.scanner.dt, dtype=np.float64)
        # centre line every 0.1 m
        self._set_centerline(xy, wr, wl)
        self.teacher = None                                           # set by teacher.attach_teacher

    def _set_centerline(self, xy, wr, wl):
        q = np.vstack([xy, xy[:1]])
        sd = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
        t = np.arange(0, sd[-1] - 1e-9, 0.1)
        self.cx = np.interp(t, sd, q[:, 0])
        self.cy = np.interp(t, sd, q[:, 1])
        self.wr = np.interp(t, sd, np.append(wr, wr[0]))
        self.wl = np.interp(t, sd, np.append(wl, wl[0]))
        self.s = t
        self.length = float(sd[-1])
        tg = tangents(np.stack([self.cx, self.cy], 1))
        self.tx, self.ty = tg[:, 0], tg[:, 1]
        self.n = len(self.cx)

    def project(self, x, y, i_guess=None, window=60):
        """(s, ey, index) of the closest centre-line point; ey > 0 is left of the line."""
        if i_guess is None:
            idx = np.arange(self.n)
        else:
            idx = np.arange(i_guess - window, i_guess + window + 1) % self.n
        d2 = (self.cx[idx] - x) ** 2 + (self.cy[idx] - y) ** 2
        i = int(idx[int(np.argmin(d2))])
        dx, dy = x - self.cx[i], y - self.cy[i]
        along = dx * self.tx[i] + dy * self.ty[i]
        ey = -dx * self.ty[i] + dy * self.tx[i]
        return (self.s[i] + along) % self.length, ey, i

    def pose_at(self, s, ey=0.0, dpsi=0.0):
        i = int(round(s / 0.1)) % self.n
        x = self.cx[i] - ey * self.ty[i]
        y = self.cy[i] + ey * self.tx[i]
        return x, y, math.atan2(self.ty[i], self.tx[i]) + dpsi, i

    def free(self, x, y, clearance=0.0) -> bool:
        r, c = int((y - self.oy) / self.res), int((x - self.ox) / self.res)
        if r < 0 or c < 0 or r >= self.edt.shape[0] or c >= self.edt.shape[1]:
            return False
        return self.edt[r, c] > clearance

    def scan(self, x, y, yaw):
        return self.scanner.scan(np.array([x, y, yaw]), None)


def training_bank(n_generated: int = 200, holdout=HOLDOUT, mirror: bool = True, reverse: bool = True):
    """List of (folder, mirror, reverse) for training. Real tracks minus the hold-out ones, plus the
    generated ones, each in the 4 variants (as is, mirrored, reversed, both)."""
    folders = [REAL[k] for k in REAL if k not in holdout and os.path.isdir(REAL[k])]
    folders += [os.path.join(GEN_DIR, generated_name(i)) for i in range(n_generated)]
    out = []
    for f in folders:
        for m in ((False, True) if mirror else (False,)):
            for r in ((False, True) if reverse else (False,)):
                out.append((f, m, r))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', nargs=4, metavar=('NAME', 'PGM', 'YAML', 'GUIDE_CSV'))
    ap.add_argument('--generate', type=int, default=0)
    a = ap.parse_args()
    if a.build:
        name, pgm, yml, guide = a.build
        c, wr, wl = build_from_map(name, pgm, yml, guide, os.path.join(V2_TRACKS, name))
        k = np.abs(curvature(c))
        print(f'{name}: {len(c) * 0.1:.1f} m, width {np.min(wr + wl):.2f} to {np.max(wr + wl):.2f} m, '
              f'tightest radius {1 / k.max():.2f} m. Add it to REAL in tracks.py to train on it.')
    for i in range(a.generate):
        f = ensure_generated(i)
        d = np.loadtxt(os.path.join(f, f'{generated_name(i)}_centerline.csv'), delimiter=',', comments='#')
        k = np.abs(curvature(d[:, :2]))
        print(f'{generated_name(i)}: {len(d) * 0.1:5.1f} m, width {2 * d[0, 2]:.2f} m, tightest radius {1 / k.max():.2f} m')
