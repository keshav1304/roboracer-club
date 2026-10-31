"""Convert a playground / iros-2026 track (map.pgm + map.yaml + raceline
waypoints.csv) into an f1tenth_gym map folder.

  python tools/port_track.py track-20260731   # our mapped IROS 2026 track
  python tools/port_track.py icra2026-comp    # ICRA 2026 competition track

Writes ../gym-two-car-reactive/f1tenth_gym/maps/<name>/:
  <name>.png             occupancy image (unknown cells → occupied, so the LiDAR sees walls)
  <name>_map.yaml        gym map spec
  <name>_raceline.csv    s;x;y;psi;kappa;vx;ax
  <name>_centerline.csv  x, y, w_right, w_left: midpoints between the walls,
                         measured along the raceline normals
"""
import pathlib
import sys

import numpy as np
import yaml
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parents[1]
IROS = HERE.parent / "iros-2026" / "autodrive_devkit"
ICRA = HERE.parent / "icra-2026" / "autodrive_devkit"
GYM_MAPS = HERE.parent / "gym-two-car-reactive" / "f1tenth_gym" / "maps"
# name → (map yaml, raceline csv with columns s,x,y,theta,velocity)
TRACKS = {
    "track-20260731": (IROS / "maps" / "track-20260731" / "map.yaml",
                       IROS / "racelines" / "track-20260731-line" / "waypoints.csv"),
    "icra2026-comp": (ICRA / "comp_track.yaml", ICRA / "comp_waypoints.csv"),
}


def ray(free, res, ox, oy, x, y, nx, ny, max_d=3.0):
    """Distance from (x, y) along (nx, ny) to the first non-free cell."""
    h = free.shape[0]
    for d in np.arange(0.0, max_d, res / 2):
        c = int((x + d * nx - ox) / res)
        r = h - 1 - int((y + d * ny - oy) / res)
        if r < 0 or r >= h or c < 0 or c >= free.shape[1] or not free[r, c]:
            return d
    return max_d


def main(name):
    map_yaml, raceline = TRACKS[name]
    spec = yaml.safe_load(open(map_yaml))
    img = np.array(Image.open(map_yaml.parent / spec["image"]))
    res, (ox, oy, _) = spec["resolution"], spec["origin"]
    free = img >= 250                                   # trinary pgm: 254 free, 205 unknown, 0 occupied
    # SLAM leaves specks of "occupied" inside the track; the LiDAR would see them
    # as obstacles and the car brakes for nothing. Drop occupied blobs < 0.5 m².
    from scipy import ndimage
    lab, n = ndimage.label(~free)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    small = np.isin(lab, 1 + np.nonzero(sizes * res * res < 0.5)[0])
    free = free | small
    out = GYM_MAPS / name
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(free, 255, 0).astype(np.uint8)).save(out / f"{name}.png")
    yaml.safe_dump({"image": f"{name}.png", "resolution": res, "origin": [ox, oy, 0.0], "negate": 0,
                    "occupied_thresh": 0.45, "free_thresh": 0.196}, open(out / f"{name}_map.yaml", "w"))

    w = np.loadtxt(raceline, delimiter=",", skiprows=1)
    s, x, y, th, v = w.T
    bad = v <= 0                                        # comp_waypoints.csv has 7 sign glitches (3.3 → −5.2 → 5.1)
    if bad.any():
        v = v.copy()
        v[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), v[~bad])
        print(f"fixed {bad.sum()} non-positive raceline speeds by interpolation")
    # curvature from heading change per metre (closed loop)
    dth = (np.roll(th, -1) - np.roll(th, 1) + np.pi) % (2 * np.pi) - np.pi
    dsl = np.roll(s, -1) - np.roll(s, 1)
    dsl[dsl <= 0] += s[-1] + np.hypot(x[0] - x[-1], y[0] - y[-1])
    k = dth / dsl
    ax = np.zeros_like(v)
    np.savetxt(out / f"{name}_raceline.csv", np.c_[s, x, y, th, k, v, ax], delimiter=";", fmt="%.7f",
               header="s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2")

    cl = []
    for xi, yi, ti in zip(x, y, th):
        nx, ny = -np.sin(ti), np.cos(ti)                # left normal
        dl, dr = ray(free, res, ox, oy, xi, yi, nx, ny), ray(free, res, ox, oy, xi, yi, -nx, -ny)
        off = (dl - dr) / 2                             # shift to the midpoint between the walls
        half = (dl + dr) / 2
        cl.append((xi + off * nx, yi + off * ny, half, half))
    cl = np.array(cl)
    # smooth the midpoints and widths around the loop (wall pixels are jagged)
    kern = np.ones(15) / 15
    cl = np.stack([np.convolve(np.r_[c[-7:], c, c[:7]], kern, "valid") for c in cl.T], 1)
    np.savetxt(out / f"{name}_centerline.csv", cl, delimiter=", ", fmt="%.6f", header="x_m, y_m, w_tr_right_m, w_tr_left_m")
    print(f"{name}: raceline {len(w)} pts, {s[-1]:.1f} m; track width min {2 * cl[:, 2].min():.2f} m, "
          f"median {2 * np.median(cl[:, 2]):.2f} m, max {2 * cl[:, 2].max():.2f} m; v {v.min():.1f}–{v.max():.1f} m/s → {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "track-20260731")
