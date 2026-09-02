#!/usr/bin/env python3
"""
generate_waypoints.py  –  CMA-ES Raceline Optimiser
====================================================
Reads a ROS map (YAML + PGM), extracts a clean closed-loop centreline via
skeleton graph-pruning, then runs CMA-ES to find the minimum-curvature
raceline within the track.

Based on:
  • TUNERCAR (O'Kelly et al., ICRA 2020)
  • Gradient-free Multi-domain Optimisation (Zheng et al., arXiv 2202.13525)
  • Minimum Curvature Trajectory Planning (Heilmeier et al., Veh. Syst. Dyn. 2019)

Install:
    pip install cma opencv-python-headless scikit-image pyyaml scipy numpy

Usage:
    python generate_waypoints.py --map comp_track.yaml --out comp_waypoints.csv

    For gentler curvature (less tight spline bends / lower κ spikes), try e.g.:
    ``--spl-smooth 40`` and/or ``--frenet-sigma 2`` (tune to taste).

Output CSV columns:  s, x, y, theta, velocity
"""

import argparse, csv, os, sys
import cv2, cma
import numpy as np, yaml
from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from skimage.morphology import skeletonize


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MAP LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_map(yaml_path: str):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    pgm_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), cfg["image"])
    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load PGM: {pgm_path}")
    return (img,
            float(cfg["resolution"]),
            cfg["origin"],
            float(cfg.get("occupied_thresh", 0.65)),
            float(cfg.get("free_thresh",    0.25)),
            int  (cfg.get("negate",         0)))


def build_free_mask(img, occ_thresh, free_thresh, negate):
    """True where navigable (ROS convention: bright pixel = free space)."""
    norm = img.astype(np.float32) / 255.0
    if negate:
        norm = 1.0 - norm
    return norm > (1.0 - free_thresh)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CENTRELINE EXTRACTION  –  skeleton → graph prune → ordered loop
# ─────────────────────────────────────────────────────────────────────────────

def extract_centreline(free_mask: np.ndarray, erode_iters: int = 2):
    """
    Returns
    -------
    loop_px  : (N, 2) float  [col, row] pixel coords, loop-ordered
    dist_map : (H, W) float  Euclidean distance to nearest wall (pixels)
    """
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(
        free_mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)

    # Keep largest connected free-space component (the track itself)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if n < 2:
        raise RuntimeError("No connected free-space region found in map.")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)

    dist_map = distance_transform_edt(mask)

    # Erode before skeletonising: thickens walls slightly → cleaner skeleton
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=erode_iters)
    skel   = skeletonize(eroded.astype(bool))

    loop_px = _skeleton_to_loop(skel)
    if loop_px is None or len(loop_px) < 20:
        raise RuntimeError(
            "Could not extract a valid closed loop from the skeleton.\n"
            "Try --erode 1 or check map thresholds.")

    return loop_px, dist_map


def _skeleton_to_loop(skel: np.ndarray):
    """
    Convert skeleton image → single ordered closed loop.

    Steps
    -----
    1. Build 8-connected pixel graph.
    2. Prune degree-0 / degree-1 nodes (removes all spurious branches).
    3. Keep only the largest connected component (the main loop).
    4. Traverse: at junctions (degree > 2) prefer the direction that
       minimises turning angle ("go straight").

    Returns (N, 2) float [col, row].
    """
    ys, xs = np.where(skel)
    if len(xs) == 0:
        return None

    # Build adjacency
    nodes = set(zip(ys.tolist(), xs.tolist()))
    adj: dict = {n: set() for n in nodes}
    for y, x in nodes:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = (y + dy, x + dx)
                if nb in nodes:
                    adj[(y, x)].add(nb)

    # Prune branches iteratively until only cycle nodes remain
    changed = True
    while changed:
        changed = False
        dead = [k for k, v in adj.items() if len(v) <= 1]
        for k in dead:
            for nb in list(adj.get(k, set())):
                adj[nb].discard(k)
            del adj[k]
            changed = True

    if not adj:
        return None

    # Keep largest connected component
    seen: set = set()
    components: list = []
    for seed in adj:
        if seed in seen:
            continue
        comp: set = set()
        stk = [seed]
        while stk:
            n = stk.pop()
            if n in comp:
                continue
            comp.add(n)
            stk.extend(adj[n] - comp)
        components.append(comp)
        seen |= comp

    main_comp = max(components, key=len)
    adj = {k: (v & main_comp) for k, v in adj.items() if k in main_comp}

    # Traverse: prefer degree-2 start, go-straight at junctions
    start = next((k for k, v in adj.items() if len(v) == 2), next(iter(adj)))
    path    = [start]
    visited = {start}
    prev    = None
    cur     = start

    while True:
        unvisited = [n for n in adj[cur] if n not in visited]
        half_done = len(path) > len(adj) // 2

        if not unvisited:
            break

        if len(unvisited) == 1 or prev is None:
            nxt = unvisited[0]
        else:
            # Junction: pick direction closest to current heading
            pv = np.array([cur[1] - prev[1], cur[0] - prev[0]], dtype=float)
            m  = np.linalg.norm(pv)
            if m > 0:
                pv /= m

            def alignment(n):
                d = np.array([n[1] - cur[1], n[0] - cur[0]], dtype=float)
                nd = np.linalg.norm(d)
                return float(np.dot(pv, d / nd)) if nd > 0 else -1.0

            nxt = max(unvisited, key=alignment)

        path.append(nxt)
        visited.add(nxt)
        prev = cur
        cur  = nxt

        # Stop once we can close the loop back to start
        if half_done and start in adj[cur]:
            break

    return np.array([[p[1], p[0]] for p in path], dtype=np.float64)  # [col,row]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SPLINE + LATERAL PERTURBATION  (search-space Θ_dm)
# ─────────────────────────────────────────────────────────────────────────────

def fit_closed_spline(pts_px: np.ndarray, n_ctrl: int) -> np.ndarray:
    """
    Subsample the raw skeleton loop, smooth-fit a closed cubic spline,
    return n_ctrl evenly-spaced control points.
    """
    # Subsample before fitting to avoid over-fitting noisy skeleton pixels
    if len(pts_px) > 600:
        idx    = np.round(np.linspace(0, len(pts_px) - 1, 600)).astype(int)
        pts_px = pts_px[idx]

    x, y = pts_px[:, 0], pts_px[:, 1]
    # s ~ 2×N gives heavy smoothing; reduce toward 0 for less smoothing
    tck, _ = splprep([x, y], s=len(x) * 2.0, per=True, k=3)
    u      = np.linspace(0, 1, n_ctrl, endpoint=False)
    cx, cy = splev(u, tck)
    return np.stack([cx, cy], axis=1)


def _left_normals(pts: np.ndarray) -> np.ndarray:
    """Unit left-perpendicular to tangent at each point."""
    t   = np.gradient(pts, axis=0)
    t  /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    return np.stack([-t[:, 1], t[:, 0]], axis=1)


def perturb_and_sample(
    ctrl_pts: np.ndarray,
    offsets: np.ndarray,
    n_out: int = 500,
    spl_smooth: float = 0.0,
):
    """
    Shift each control point laterally by offsets[i] pixels (Θ_dm),
    re-spline, return n_out uniform samples.  None if spline fails.

    spl_smooth
        scipy ``splprep(..., s=...)`` smoothing. 0 = interpolate exactly
        through knots (can produce sharp local curvature). Larger values
        (e.g. 0.5–3× number of control points) yield gentler curves that
        approximate the lateral offsets.
    """
    perturbed = ctrl_pts + _left_normals(ctrl_pts) * offsets[:, None]
    n_k = len(perturbed)
    s = float(spl_smooth)
    if s < 0.0:
        s = 0.0
    try:
        tck, _ = splprep(
            [perturbed[:, 0], perturbed[:, 1]],
            s=s,
            per=True,
            k=3,
        )
    except Exception:
        return None
    u = np.linspace(0, 1, n_out, endpoint=False)
    rx, ry = splev(u, tck)
    return np.stack([rx, ry], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  OBJECTIVE  f(θ)
# ─────────────────────────────────────────────────────────────────────────────

def _curvature(pts: np.ndarray) -> np.ndarray:
    dx  = np.gradient(pts[:, 0])
    dy  = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    return (dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-9)


def objective(
    offsets: np.ndarray,
    ctrl_pts: np.ndarray,
    dist_map: np.ndarray,
    margin_px: float,
    n_eval: int = 250,
    spl_smooth: float = 0.0,
) -> float:
    """
    f(θ) = ∫κ²ds  +  1e4 · Σ max(0, margin_px − clearance)²

    Lower = smoother path that stays ≥ margin_px pixels from any wall.
    """
    rl = perturb_and_sample(ctrl_pts, offsets, n_out=n_eval, spl_smooth=spl_smooth)
    if rl is None:
        return 1e9

    H, W  = dist_map.shape
    rows  = np.clip(rl[:, 1].astype(int), 0, H - 1)
    cols  = np.clip(rl[:, 0].astype(int), 0, W - 1)
    clear = dist_map[rows, cols]

    wall_penalty   = np.sum(np.maximum(0.0, margin_px - clear) ** 2)
    curvature_cost = np.sum(_curvature(rl) ** 2)

    return curvature_cost + 1e4 * wall_penalty


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COORDINATE CONVERSION + VELOCITY PROFILE + FRENET FRAME
# ─────────────────────────────────────────────────────────────────────────────

def pixels_to_world(pts_px: np.ndarray,
                    img_h:  int,
                    res:    float,
                    origin: list) -> np.ndarray:
    """
    Pixel (col, row) → world (x, y) metres.
    ROS convention: origin = bottom-left corner; row 0 = top of image.
    """
    wx = origin[0] + pts_px[:, 0] * res
    wy = origin[1] + (img_h - 1 - pts_px[:, 1]) * res
    return np.stack([wx, wy], axis=1)


def velocity_profile(kappa: np.ndarray,
                     v_min: float,
                     v_max: float,
                     a_lat: float,
                     a_lon_accel: float = 3.0,
                     a_lon_brake: float = 5.0,
                     ds: np.ndarray = None) -> np.ndarray:
    """
    Physics-based velocity with forward-backward acceleration smoothing.

    1. Cornering limit:  v_corner = sqrt(a_lat / |κ|)
    2. Forward pass:     enforce max longitudinal acceleration
    3. Backward pass:    enforce max braking deceleration

    This lets the car hit v_max on straights quickly, while braking
    hard just before corners — much faster than naive linear scaling.
    """
    N = len(kappa)

    # Step 1: Raw cornering speed limit
    v_corner = np.sqrt(a_lat / (np.abs(kappa) + 1e-6))
    v_corner = np.clip(v_corner, v_min, v_max)

    # Arc-length between consecutive waypoints
    if ds is None:
        ds = np.ones(N)

    # Step 2: Forward pass — limit acceleration
    v_fwd = np.copy(v_corner)
    for i in range(1, N):
        v_fwd[i] = min(v_fwd[i],
                       np.sqrt(v_fwd[i-1]**2 + 2 * a_lon_accel * ds[i]))
    # Wrap-around: propagate from last point back to first
    for i in range(1, N):
        v_fwd[i] = min(v_fwd[i],
                       np.sqrt(v_fwd[i-1]**2 + 2 * a_lon_accel * ds[i]))

    # Step 3: Backward pass — limit braking
    v_smooth = np.copy(v_fwd)
    for i in range(N - 2, -1, -1):
        v_smooth[i] = min(v_smooth[i],
                          np.sqrt(v_smooth[i+1]**2 + 2 * a_lon_brake * ds[i+1]))
    # Wrap-around: propagate backward once more
    for i in range(N - 2, -1, -1):
        v_smooth[i] = min(v_smooth[i],
                          np.sqrt(v_smooth[i+1]**2 + 2 * a_lon_brake * ds[i+1]))

    return np.clip(v_smooth, v_min, v_max)


def smooth_closed_polyline_xy(world_pts: np.ndarray, sigma: float) -> np.ndarray:
    """Light Gaussian smoothing along arc index (periodic). Reduces κ noise."""
    if sigma <= 0.0 or world_pts.shape[0] < 4:
        return world_pts
    sig = float(sigma)
    x = gaussian_filter1d(world_pts[:, 0], sig, mode="wrap")
    y = gaussian_filter1d(world_pts[:, 1], sig, mode="wrap")
    return np.column_stack([x, y])


def compute_frenet(
    world_pts: np.ndarray,
    v_min: float,
    v_max: float,
    a_lat: float,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """Build [s, x, y, theta, velocity] table from world-frame raceline."""
    if smooth_sigma > 0.0:
        world_pts = smooth_closed_polyline_xy(world_pts, smooth_sigma)
    dx = np.gradient(world_pts[:, 0])
    dy = np.gradient(world_pts[:, 1])
    ds = np.hypot(dx, dy)
    s = np.cumsum(ds)
    s -= s[0]
    theta = np.arctan2(dy, dx)
    kappa = _curvature(world_pts)            # 1/m in world space
    vel = velocity_profile(kappa, v_min, v_max, a_lat, ds=ds)
    return np.stack([s, world_pts[:, 0], world_pts[:, 1], theta, vel], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 5b. PROGRAMMATIC API (GUI / notebooks)
# ─────────────────────────────────────────────────────────────────────────────


def load_track_and_ctrl(map_yaml: str, erode: int, n_ctrl: int) -> dict:
    """
    Load map, extract centreline, build CMA control polygon.
    Returns dict with keys: img, res, origin, img_h, img_w, dist_map, ctrl_pts.
    """
    img, res, origin, occ_t, free_t, neg = load_map(map_yaml)
    img_h, img_w = img.shape
    free_mask = build_free_mask(img, occ_t, free_t, neg)
    centreline_px, dist_map = extract_centreline(free_mask, erode_iters=erode)
    ctrl_pts = fit_closed_spline(centreline_px, n_ctrl=n_ctrl)
    return {
        "img": img,
        "res": res,
        "origin": origin,
        "img_h": img_h,
        "img_w": img_w,
        "dist_map": dist_map,
        "ctrl_pts": ctrl_pts,
    }


def build_offset_bounds(
    ctrl_pts: np.ndarray,
    dist_map: np.ndarray,
    margin_px: float,
) -> list[float]:
    img_h, img_w = dist_map.shape
    cp_r = np.clip(ctrl_pts[:, 1].astype(int), 0, img_h - 1)
    cp_c = np.clip(ctrl_pts[:, 0].astype(int), 0, img_w - 1)
    avail = dist_map[cp_r, cp_c] - margin_px
    return np.maximum(avail, 0.5).tolist()


def run_cma_optimize(
    ctrl_pts: np.ndarray,
    dist_map: np.ndarray,
    margin_px: float,
    spl_smooth: float,
    budget: int,
    sigma0: float,
) -> tuple[np.ndarray, float]:
    """
    Run CMA-ES on lateral offsets. Returns (best_offsets, fbest).
    """
    n_ctrl = int(ctrl_pts.shape[0])
    bound = build_offset_bounds(ctrl_pts, dist_map, margin_px)
    spl_s = max(0.0, float(spl_smooth))
    opts = cma.CMAOptions()
    opts["maxfevals"] = int(budget)
    opts["verbose"] = -9
    opts["tolx"] = 1e-4
    opts["tolfun"] = 1e-5
    opts["bounds"] = [[-b for b in bound], bound]
    opts["popsize"] = max(24, n_ctrl)

    es = cma.CMAEvolutionStrategy(np.zeros(n_ctrl), float(sigma0), opts)
    while not es.stop():
        candidates = es.ask()
        fitnesses = [
            objective(
                np.array(c),
                ctrl_pts,
                dist_map,
                margin_px,
                n_eval=200,
                spl_smooth=spl_s,
            )
            for c in candidates
        ]
        es.tell(candidates, fitnesses)

    return np.array(es.result.xbest), float(es.result.fbest)


def frenet_from_offsets(
    ctrl_pts: np.ndarray,
    offsets: np.ndarray,
    img_h: int,
    res: float,
    origin: list,
    *,
    spl_smooth: float,
    n_out: int,
    frenet_sigma: float,
    v_min: float,
    v_max: float,
    a_lat: float,
) -> np.ndarray | None:
    """Pixels raceline → world Frenet table, or None if spline fails."""
    raceline_px = perturb_and_sample(
        ctrl_pts, offsets, n_out=int(n_out), spl_smooth=max(0.0, float(spl_smooth))
    )
    if raceline_px is None:
        return None
    world_pts = pixels_to_world(raceline_px, img_h, res, origin)
    return compute_frenet(
        world_pts,
        v_min,
        v_max,
        a_lat,
        smooth_sigma=max(0.0, float(frenet_sigma)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="CMA-ES raceline optimiser — outputs Frenet-frame CSV")
    p.add_argument("--map",    default="comp_track.yaml")
    p.add_argument("--out",    default="comp_waypoints.csv")
    p.add_argument("--ctrl",   type=int,   default=80,
                   help="CMA-ES control-point count (default 80)")
    p.add_argument("--budget", type=int,   default=20000,
                   help="Max objective evaluations (default 8000)")
    p.add_argument("--sigma0", type=float, default=2.0,
                   help="Initial CMA-ES step size in pixels (default 2.0)")
    p.add_argument("--margin", type=float, default=0.25,
                   help="Min wall clearance in METRES (default 0.33)")
    p.add_argument("--vmin",   type=float, default=0.5,
                   help="Min target velocity m/s (default 0.5)")
    p.add_argument("--vmax",   type=float, default=6.5,
                   help="Max target velocity m/s (default 3.0)")
    p.add_argument("--alat",   type=float, default=3.5,
                   help="Max lateral acceleration m/s^2 (default 3.0)")
    p.add_argument("--nout",   type=int,   default=800,
                   help="Output waypoint count (default 500)")
    p.add_argument("--erode",  type=int,   default=2,
                   help="Erosion iterations before skeletonise (default 2)")
    p.add_argument(
        "--spl-smooth",
        type=float,
        default=0.0,
        metavar="S",
        help="Raceline splprep smoothing s (0=exact through knots; try ~0.3–1.5× "
             "--ctrl to reduce sharp bends, e.g. 40 for ctrl=80)",
    )
    p.add_argument(
        "--frenet-sigma",
        type=float,
        default=0.0,
        metavar="SIGMA",
        help="Gaussian smooth world (x,y) along closed polyline before κ/velocity "
             "(0=off; try 1–4 in waypoint index units to clip numerical κ spikes)",
    )
    args = p.parse_args()

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading map …  ({args.map})")
    img, res, origin, occ_t, free_t, neg = load_map(args.map)
    img_h, img_w = img.shape
    margin_px = args.margin / res
    print(f"      {img_w}×{img_h} px | {res} m/px | "
          f"margin = {args.margin} m ({margin_px:.1f} px)")

    # ── 2. Occupancy ──────────────────────────────────────────────────────────
    print("[2/6] Building free-space mask …")
    free_mask = build_free_mask(img, occ_t, free_t, neg)
    print(f"      Free: {100 * free_mask.mean():.1f}%")

    # ── 3–4. Centreline + control points ───────────────────────────────────────
    print("[3/6] Extracting closed-loop centreline …")
    centreline_px, dist_map = extract_centreline(free_mask,
                                                  erode_iters=args.erode)
    print(f"      Loop: {len(centreline_px)} ordered pixels")

    print(f"[4/6] Fitting smooth spline → {args.ctrl} control points …")
    ctrl_pts = fit_closed_spline(centreline_px, n_ctrl=args.ctrl)

    avail = dist_map[
        np.clip(ctrl_pts[:, 1].astype(int), 0, img_h - 1),
        np.clip(ctrl_pts[:, 0].astype(int), 0, img_w - 1),
    ] - margin_px
    print(f"      Mean available offset = {np.mean(avail) * res:.3f} m  "
          f"(min = {np.min(avail) * res:.3f} m)")

    # ── 5. CMA-ES optimisation ────────────────────────────────────────────────
    spl_s = max(0.0, float(args.spl_smooth))
    print(f"[5/6] Running CMA-ES  (budget={args.budget}, "
          f"popsize={max(24, args.ctrl)}, σ₀={args.sigma0} px, spl_smooth={spl_s}) …")
    best, fbest = run_cma_optimize(
        ctrl_pts,
        dist_map,
        margin_px,
        spl_s,
        int(args.budget),
        float(args.sigma0),
    )
    print(f"      Optimisation complete — best cost = {fbest:.6f}")

    # Clearance audit on best solution
    rl_audit = perturb_and_sample(ctrl_pts, best, n_out=300, spl_smooth=spl_s)
    if rl_audit is not None:
        r = np.clip(rl_audit[:, 1].astype(int), 0, img_h - 1)
        c = np.clip(rl_audit[:, 0].astype(int), 0, img_w - 1)
        min_m = float(dist_map[r, c].min()) * res
        ok    = "✓" if min_m >= args.margin else "⚠  below target"
        print(f"      Min wall clearance on optimised path: {min_m:.3f} m  {ok}")

    # ── 6. Output ─────────────────────────────────────────────────────────────
    print(f"[6/6] Sampling {args.nout} waypoints and writing CSV …")
    frenet = frenet_from_offsets(
        ctrl_pts,
        best,
        img_h,
        res,
        origin,
        spl_smooth=spl_s,
        n_out=int(args.nout),
        frenet_sigma=max(0.0, float(args.frenet_sigma)),
        v_min=float(args.vmin),
        v_max=float(args.vmax),
        a_lat=float(args.alat),
    )
    if frenet is None:
        sys.exit("ERROR: best solution produced an invalid spline. "
                 "Try a smaller --margin or larger --budget.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s", "x", "y", "theta", "velocity"])
        w.writerows(frenet.tolist())

    print(f"      Track length   = {frenet[-1, 0]:.2f} m")
    print(f"      Velocity range = [{frenet[:,4].min():.2f}, "
          f"{frenet[:,4].max():.2f}] m/s  "
          f"(mean {frenet[:,4].mean():.2f})")
    print(f"      Wrote → {args.out}\n\nDone ✓\n")


if __name__ == "__main__":
    main()