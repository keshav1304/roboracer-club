"""Custom test track: the Porto layout scaled up, constant width, static obstacles.

Porto's straights are only 1.85 m apart, so the shape is scaled before widening
(scale 1.8 → 3.3 m apart, leaving a ~0.9 m island at width 2.4 m).

Usage:
  python build_track.py                         # PortoWide: scale 1.8, width 2.4 m, 3 boxes
  python build_track.py --name PortoWideClear --no-obstacles
Writes f1tenth_gym/maps/<name>/ (png, <name>_map.yaml, <name>.yaml,
<name>_centerline.csv, obstacles.json).
"""
import argparse
import json
import pathlib

import cv2
import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
MAPS = HERE / "f1tenth_gym" / "maps"              # created by setup.sh
SRC_CL = HERE / "tracks" / "porto_centerline.csv" # AutoDRIVE Porto centerline (Unity x,z → x,y)
RES = 0.025
PAD = 1.0

# (fraction of lap, lateral offset [m, + = left of travel], length, width [m])
DEFAULT_OBSTACLES = [
    (0.22, +0.55, 0.40, 0.35),   # on the left: pass on the right
    (0.52, -0.55, 0.40, 0.35),   # on the right: pass on the left
    (0.80, 0.00, 0.40, 0.35),    # dead centre: either side (~1 m each)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="PortoWide")
    ap.add_argument("--scale", type=float, default=1.8)
    ap.add_argument("--width", type=float, default=2.4, help="road width [m]")
    ap.add_argument("--no-obstacles", action="store_true")
    args = ap.parse_args()

    cl = np.loadtxt(SRC_CL, delimiter=",", comments="#")[:, :2] * args.scale
    seg = np.hypot(*np.diff(np.vstack([cl, cl[:1]]), axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)[:-1]]
    L = float(seg.sum())
    yaw = np.arctan2(np.roll(cl[:, 1], -1) - np.roll(cl[:, 1], 1), np.roll(cl[:, 0], -1) - np.roll(cl[:, 0], 1))

    half = args.width / 2
    xmin, ymin = cl.min(0) - half - PAD
    xmax, ymax = cl.max(0) + half + PAD
    W, H = int(np.ceil((xmax - xmin) / RES)), int(np.ceil((ymax - ymin) / RES))

    def to_px(xy):
        xy = np.atleast_2d(xy)
        return np.stack([(xy[:, 0] - xmin) / RES, (ymax - xy[:, 1]) / RES], 1)

    # road = closed centerline drawn with the road width; everything else is wall
    img = np.zeros((H, W), np.uint8)
    pts = to_px(cl).round().astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, 255, thickness=int(round(args.width / RES)), lineType=cv2.LINE_AA)
    img = np.where(img > 127, 255, 0).astype(np.uint8)

    # sanity: the road must not merge across the island
    dist = cv2.distanceTransform((img > 0).astype(np.uint8), cv2.DIST_L2, 5) * RES
    px = to_px(cl).round().astype(int)
    hw = dist[px[:, 1].clip(0, H - 1), px[:, 0].clip(0, W - 1)]
    if hw.max() > half + 0.15:
        raise SystemExit(f"road merged with itself (half-width up to {hw.max():.2f} m): increase --scale")

    obstacles = []
    if not args.no_obstacles:
        for frac, off, ln, wd in DEFAULT_OBSTACLES:
            i = int(np.argmin(np.abs(s - frac * L)))
            th = yaw[i]
            c = cl[i] + off * np.array([-np.sin(th), np.cos(th)])
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            corners = (R @ np.array([[ln, wd], [ln, -wd], [-ln, -wd], [-ln, wd]]).T / 2).T + c
            cv2.fillPoly(img, [to_px(corners).round().astype(np.int32)], 0)
            obstacles.append({"s": float(s[i]), "offset": off, "corners": corners.round(4).tolist()})

    out = MAPS / args.name
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / f"{args.name}.png"), img)
    spec = {"image": f"{args.name}.png", "resolution": RES, "origin": [float(xmin), float(ymin), 0.0],
            "negate": 0, "occupied_thresh": 0.45, "free_thresh": 0.196}
    for fn in (f"{args.name}_map.yaml", f"{args.name}.yaml"):
        with open(out / fn, "w") as f:
            yaml.safe_dump(spec, f, default_flow_style=None)
    np.savetxt(out / f"{args.name}_centerline.csv", np.column_stack([cl, np.full(len(cl), half), np.full(len(cl), half)]),
               delimiter=", ", fmt="%.4f", header="x_m, y_m, w_tr_right_m, w_tr_left_m", comments="# ")
    with open(out / "obstacles.json", "w") as f:
        json.dump(obstacles, f, indent=1)
    print(f"{args.name}: lap {L:.1f} m, width {args.width} m, {W}x{H} px, {len(obstacles)} obstacles -> {out}")


if __name__ == "__main__":
    main()
