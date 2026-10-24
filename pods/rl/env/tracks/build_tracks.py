"""Build f1tenth_gym track folders for the AutoDRIVE tracks from the club's own map files.

Run from pods/rl/env:  python tracks/build_tracks.py
Source: roboracer-sim-competitions/icra-2026/autodrive_devkit/ in this repo.
    icra       comp_track.pgm      guided by comp_waypoints_works.csv   (ICRA 2026 compete track)
    practice   practice_track.pgm  guided by waypoints.csv              (AutoDRIVE practice track)
Writes tracks/NAME/NAME_map.png, NAME_map.yaml, NAME_centerline.csv (the gym's format, x_m, y_m,
w_tr_right_m, w_tr_left_m) and NAME_line.csv (the club race line, for baselines only), plus a
check picture NAME_view.png.

1. Map. The gym treats any pixel above 128 as free, so the grey "unknown" cells (205) would be
   drivable and LiDAR would see through them. Only the free region the car is in stays white;
   everything else becomes wall. Wall specks under 0.1 m^2 inside the track (SLAM noise the gym
   would count as obstacles) are cleared.
2. Centre line. For each point of the club race line, look along its normal (up to 1.5 m each way,
   without crossing a wall) and take the point farthest from both walls (the ridge of the distance
   transform). Smooth, resample every 0.1 m, repeat with the new normals. The race line only
   gives the order around the loop, so no skeleton is needed.
3. Widths. Left and right width = distance along the normal to the first wall cell.
4. Start. Index 0 is the AutoDRIVE spawn (0.8, 3.16) and the line runs the way the car faces (-y).
"""
import math
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', '..', '..', 'roboracer-sim-competitions', 'icra-2026', 'autodrive_devkit')
SPAWN = (0.80, 3.16)
TRACKS = {'icra': ('comp_track', 'comp_waypoints_works.csv'),
          'practice': ('practice_track', 'waypoints.csv')}


def read_map(stem):
    meta = dict(ln.split(': ', 1) for ln in open(os.path.join(SRC, stem + '.yaml')).read().splitlines() if ': ' in ln)
    res = float(meta['resolution'])
    ox, oy = (float(v) for v in meta['origin'].strip('[]').split(',')[:2])
    return cv2.imread(os.path.join(SRC, stem + '.pgm'), cv2.IMREAD_UNCHANGED), res, ox, oy


def read_line(name):
    d = np.loadtxt(os.path.join(SRC, name), delimiter=',', comments='s')   # header starts with 's'
    return d[:, 1:3]


def smooth(p, k):
    pad = np.vstack([p[-k:], p, p[:k]])
    return np.stack([np.convolve(pad[:, j], np.ones(k) / k, 'same')[k:-k] for j in (0, 1)], 1)


def resample(p, h=0.1):
    q = np.vstack([p, p[:1]])
    sd = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))])
    t = np.arange(0, sd[-1], h)
    return np.stack([np.interp(t, sd, q[:, 0]), np.interp(t, sd, q[:, 1])], 1)


def normals(p):
    t = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.stack([-t[:, 1], t[:, 0]], 1)                                # left normal


def build(name, stem, line_file):
    grid, res, ox, oy = read_map(stem)
    H, W = grid.shape
    to_rc = lambda x, y: (int(round(H - 1 - (y - oy) / res)), int(round((x - ox) / res)))  # noqa: E731

    _, lab = cv2.connectedComponents((grid > 250).astype(np.uint8), connectivity=4)
    free = (lab == lab[to_rc(*SPAWN)]).astype(np.uint8)
    # specks: wall blobs under 0.1 m^2 that sit inside the track are SLAM noise, not walls
    n, blob, stats, _ = cv2.connectedComponentsWithStats(1 - free, connectivity=8)
    specks = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] * res * res < 0.1]
    free[np.isin(blob, specks)] = 1
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * res

    def inside(x, y):
        r, c = to_rc(x, y)
        return 0 <= r < H and 0 <= c < W and free[r, c]

    def walk(p, n, sign, limit=1.5):
        """Offsets along +/-n until the first wall cell, with distance-transform values."""
        out = []
        for k in range(1, int(limit / 0.025) + 1):
            t = sign * 0.025 * k
            x, y = p + t * n
            if not inside(x, y):
                return out, abs(t)
            r, c = to_rc(x, y)
            out.append((t, dist[r, c]))
        return out, limit

    line = resample(read_line(line_file))
    c = line.copy()
    for _ in range(3):
        nrm = normals(c)
        new = []
        for p, n in zip(c, nrm):
            r, cc = to_rc(*p)
            cand = [(0.0, dist[r, cc])] + walk(p, n, +1)[0] + walk(p, n, -1)[0]
            new.append(p + max(cand, key=lambda tv: tv[1])[0] * n)
        c = resample(smooth(np.array(new), 9))
    i0 = int(np.argmin(np.hypot(c[:, 0] - SPAWN[0], c[:, 1] - SPAWN[1])))
    if c[(i0 + 3) % len(c), 1] > c[i0, 1]:                                  # make it run -y at the spawn
        c = c[::-1]
        i0 = len(c) - 1 - i0
    c = np.roll(c, -i0, axis=0)
    nrm = normals(c)
    w_left = np.array([walk(p, n, +1, 3.0)[1] for p, n in zip(c, nrm)])
    w_right = np.array([walk(p, n, -1, 3.0)[1] for p, n in zip(c, nrm)])

    out = os.path.join(HERE, name)
    os.makedirs(out, exist_ok=True)
    cv2.imwrite(os.path.join(out, f'{name}_map.png'), free * 255)
    with open(os.path.join(out, f'{name}_map.yaml'), 'w') as f:
        f.write(f'image: {name}_map.png\nresolution: {res}\norigin: [{ox}, {oy}, 0.0]\n'
                'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
    np.savetxt(os.path.join(out, f'{name}_centerline.csv'), np.column_stack([c, w_right, w_left]),
               delimiter=', ', fmt='%.4f', header='x_m, y_m, w_tr_right_m, w_tr_left_m')
    np.savetxt(os.path.join(out, f'{name}_line.csv'), line, delimiter=', ', fmt='%.4f', header='x_m, y_m')

    S = 3
    view = cv2.resize(cv2.cvtColor(free * 255, cv2.COLOR_GRAY2BGR), (W * S, H * S), interpolation=cv2.INTER_NEAREST)
    px = lambda q: (int((q[0] - ox) / res * S), int((H - 1 - (q[1] - oy) / res) * S))  # noqa: E731
    for p, col in ((line, (0, 0, 255)), (c, (0, 160, 0))):
        for a, b in zip(p, np.roll(p, -1, 0)):
            cv2.line(view, px(a), px(b), col, 1)
    cv2.circle(view, px(c[0]), 6, (255, 0, 0), -1)
    cv2.imwrite(os.path.join(out, f'{name}_view.png'), view)
    L = np.hypot(*np.diff(np.vstack([c, c[:1]]), axis=0).T).sum()
    k = np.abs(np.gradient(np.unwrap(np.arctan2(*np.gradient(c, axis=0).T[::-1]))) / 0.1)
    print(f'{name}: centre line {L:.1f} m, {len(c)} points, width {np.min(w_left + w_right):.2f} to '
          f'{np.max(w_left + w_right):.2f} m, tightest radius {1 / max(k.max(), 1e-6):.2f} m, '
          f'start ({c[0, 0]:.2f}, {c[0, 1]:.2f}) heading {math.degrees(math.atan2(*(c[1] - c[0])[::-1])):.0f} deg')


if __name__ == '__main__':
    for name, (stem, line_file) in TRACKS.items():
        build(name, stem, line_file)
