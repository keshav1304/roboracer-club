"""Two-car race on the test track: both cars run reactive follow-the-gap.

Checks overtaking (fast ego vs slow opponent) and obstacle avoidance (static
boxes on the track). Prints per-lap times, every overtake, collisions, the
closest car-to-car gap and the closest approach to each box; saves a plot.

  python race.py                      # 10 laps on PortoWide, opponent 1.5 m/s
  python race.py --render             # watch live
  python race.py --opp-vmax 2.5 --laps 10
"""
import argparse
import json
import os
import pathlib
import time
import warnings

import gymnasium as gym
import numpy as np
import yaml

from reactive import FollowTheGap, FTGParams

warnings.filterwarnings("ignore")
FEATURES = ["scan", "pose_x", "pose_y", "pose_theta", "linear_vel_x", "collision", "lap_time", "lap_count"]
HERE = pathlib.Path(__file__).resolve().parent
GYM_MAPS = HERE / "f1tenth_gym" / "maps"   # created by setup.sh


def make_env(map_name: str, render: bool):
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": map_name,
            "num_agents": 2,
            "timestep": 0.01,
            "integrator": "rk4",
            "model": "st",
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": "features", "features": FEATURES},
        },
        render_mode="human" if render else None,
    )


def start_poses(track, gap_m: float) -> np.ndarray:
    """Ego (agent_0) at centerline index 0; opponent (agent_1) gap_m ahead along s."""
    cl = track.centerline
    j = int(np.argmin(np.abs(cl.ss - (cl.ss[0] + gap_m))))
    return np.array([[cl.xs[0], cl.ys[0], cl.yaws[0]], [cl.xs[j], cl.ys[j], cl.yaws[j]]])


def plot(map_name, logs, events, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    d = GYM_MAPS / map_name
    spec = yaml.safe_load(open(d / f"{map_name}_map.yaml"))
    img = np.array(Image.open(d / spec["image"]).convert("L"))
    res, (ox, oy, _) = spec["resolution"], spec["origin"]
    h, w = img.shape
    fig, ax = plt.subplots(figsize=(10, 7), dpi=130)
    ax.imshow(img, cmap="gray", extent=[ox, ox + w * res, oy, oy + h * res], origin="upper")
    colors = {"agent_0": "#d62728", "agent_1": "#1f77b4"}
    names = {"agent_0": "ego (fast)", "agent_1": "opponent (slow)"}
    for a, xy in logs.items():
        xy = np.array(xy)
        ax.plot(xy[:, 0], xy[:, 1], lw=1.0, color=colors[a], label=names[a])
        ax.plot(xy[0, 0], xy[0, 1], "o", color=colors[a], ms=6)
    obs_file = d / "obstacles.json"
    if obs_file.exists():
        for ob in json.load(open(obs_file)):
            c = np.array(ob["corners"] + [ob["corners"][0]])
            ax.fill(c[:, 0], c[:, 1], color="#ff7f0e", zorder=5)
    for (a, x, y, kind) in events:
        ax.plot(x, y, "x" if kind == "collision" else "^", color="k", ms=9, mew=2)
    allxy = np.vstack([np.array(v) for v in logs.values()])
    pad = 3.0
    ax.set_xlim(allxy[:, 0].min() - pad, allxy[:, 0].max() + pad)
    ax.set_ylim(allxy[:, 1].min() - pad, allxy[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title(f"{map_name}: two cars, follow-the-gap  (x = collision, ▲ = overtake, orange = obstacle)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="PortoWide")
    ap.add_argument("--laps", type=int, default=10, help="stop when the ego has done this many laps")
    ap.add_argument("--seconds", type=float, default=600.0, help="sim-time limit")
    ap.add_argument("--ego-vmax", type=float, default=6.0)
    ap.add_argument("--opp-vmax", type=float, default=1.5)
    ap.add_argument("--gap", type=float, default=3.0, help="opponent start distance ahead [m]")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--plot", default="race.png")
    args = ap.parse_args()

    env = make_env(args.map, args.render)
    u = env.unwrapped
    angles = u.sim.agents[0].scan_angles
    ctrl = {
        "agent_0": FollowTheGap(angles, FTGParams(v_max=args.ego_vmax)),
        "agent_1": FollowTheGap(angles, FTGParams(v_max=args.opp_vmax)),
    }
    obs, _ = env.reset(options={"poses": start_poses(u.track, args.gap)})
    if args.render:
        env.render()

    cl = u.track.centerline
    L = float(cl.ss[-1])
    xy_cl = np.stack([cl.xs, cl.ys], 1)

    def progress(o):  # arc length of the nearest centerline point
        i = int(np.argmin(np.sum((xy_cl - [o["pose_x"], o["pose_y"]]) ** 2, 1)))
        return cl.ss[i]

    names = {"agent_0": "ego", "agent_1": "opp"}
    from shapely.geometry import Polygon
    from f1tenth_gym.envs.collision_models import get_vertices
    obs_path = GYM_MAPS / args.map / "obstacles.json"
    obstacles = [Polygon(o["corners"]) for o in json.load(open(obs_path))] if obs_path.exists() else []
    ob_clear = {a: [np.inf] * len(obstacles) for a in obs}   # closest body-to-box distance
    car_gap_min = np.inf

    def body(o):
        return Polygon(get_vertices(np.array([o["pose_x"], o["pose_y"], o["pose_theta"]], dtype=np.float64), 0.58, 0.31))
    logs = {a: [] for a in obs}
    speeds = {a: [] for a in obs}
    s_un = {a: progress(obs[a]) for a in obs}            # unwrapped progress
    if s_un["agent_1"] < s_un["agent_0"]:
        s_un["agent_1"] += L
    s_start = dict(s_un)
    s_prev = {a: progress(obs[a]) for a in obs}
    lap_marks = {a: [0.0] for a in obs}                   # sim times each car completed a lap
    crashed = {a: False for a in obs}
    events = []

    # overtakes: n = how many times the ego is "ahead" of the opponent, counting
    # lappings; 0.6 m hysteresis so side-by-side running doesn't flicker
    MARGIN = 0.6
    rel = s_un["agent_0"] - s_un["agent_1"]
    n = int(np.floor(rel / L))
    passes = []                                           # (time, who, ego lap)

    dt = 0.01
    t0 = time.time()
    for k in range(int(args.seconds / dt)):
        t = (k + 1) * dt
        act = np.zeros((2, 2))
        for i, a in enumerate(obs):
            if not crashed[a]:
                act[i] = ctrl[a].plan(obs[a]["scan"], float(obs[a]["linear_vel_x"]))
        obs, _, _, _, _ = env.step(act)
        if args.render:
            env.render()

        for a in obs:
            o = obs[a]
            logs[a].append((float(o["pose_x"]), float(o["pose_y"])))
            speeds[a].append(float(o["linear_vel_x"]))
            s = progress(o)
            ds = s - s_prev[a]
            ds += L if ds < -L / 2 else -L if ds > L / 2 else 0.0
            s_un[a] += ds
            s_prev[a] = s
            if s_un[a] - s_start[a] >= len(lap_marks[a]) * L:
                lap_marks[a].append(t)
                if a == "agent_0":
                    lt = lap_marks[a][-1] - lap_marks[a][-2]
                    print(f"[t={t:6.2f}s] ego lap {len(lap_marks[a]) - 1:2d}: {lt:5.2f} s")
            if o["collision"] > 0 and not crashed[a]:
                crashed[a] = True
                events.append((a, float(o["pose_x"]), float(o["pose_y"]), "collision"))
                print(f"[t={t:6.2f}s] {names[a]} COLLISION")

        bodies = {a: body(obs[a]) for a in obs}
        car_gap_min = min(car_gap_min, bodies["agent_0"].distance(bodies["agent_1"]))
        for a in obs:
            for j, ob in enumerate(obstacles):
                ob_clear[a][j] = min(ob_clear[a][j], bodies[a].distance(ob))

        rel = s_un["agent_0"] - s_un["agent_1"]
        if rel > (n + 1) * L + MARGIN:
            n += 1
            who = "agent_0"
        elif rel < n * L - MARGIN:
            n -= 1
            who = "agent_1"
        else:
            who = None
        if who and not any(crashed.values()):
            o = obs[who]
            passes.append((t, who, len(lap_marks["agent_0"]) - 1))
            events.append((who, float(o["pose_x"]), float(o["pose_y"]), "pass"))
            print(f"[t={t:6.2f}s] OVERTAKE #{len(passes)}: {names[who]} passes")

        if crashed["agent_0"] or len(lap_marks["agent_0"]) - 1 >= args.laps:
            break

    wall = time.time() - t0
    print(f"\nsim {t:.1f} s in {wall:.1f} s wall ({t / wall:.1f}x real time), track {args.map} ({L:.1f} m)")
    for a in obs:
        lt = np.diff(lap_marks[a])
        lt_str = " ".join(f"{x:.2f}" for x in lt)
        print(f"{names[a]}: v_max {ctrl[a].p.v_max:.1f} | mean {np.mean(speeds[a]):.2f} m/s | "
              f"laps {len(lt)} [{lt_str}] | crashed {crashed[a]}")
        if len(lt):
            print(f"      best {lt.min():.2f} s, mean {lt.mean():.2f} s")
    ego_p = sum(1 for p in passes if p[1] == "agent_0")
    opp_p = len(passes) - ego_p
    print(f"overtakes: {len(passes)} total (ego {ego_p}, opp {opp_p}) | closest car-to-car gap {car_gap_min:.2f} m")
    if obstacles:
        for a in obs:
            cl_str = ", ".join(f"{c:.2f}" for c in ob_clear[a])
            print(f"obstacle clearance {names[a]} (closest body-to-box per obstacle, m): {cl_str}")
    plot(args.map, logs, events, args.plot)
    print("plot ->", os.path.abspath(args.plot))


if __name__ == "__main__":
    main()
