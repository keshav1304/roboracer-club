"""Two cars on an f1tenth_gym track, both driven by reactive follow-the-gap.

Usage (after ./setup.sh):
  .venv/bin/python run_two_cars.py               # headless, writes a plot
  .venv/bin/python run_two_cars.py --render       # watch it live
  .venv/bin/python run_two_cars.py --opp-vmax 3 --gap 4 --seconds 90
"""
import argparse
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
    for (a, x, y, kind) in events:
        ax.plot(x, y, "x" if kind == "collision" else "^", color="k", ms=9, mew=2)
    allxy = np.vstack([np.array(v) for v in logs.values()])
    pad = 3.0
    ax.set_xlim(allxy[:, 0].min() - pad, allxy[:, 0].max() + pad)
    ax.set_ylim(allxy[:, 1].min() - pad, allxy[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title(f"{map_name}: two cars, follow-the-gap  (x = collision, ▲ = pass)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="Spielberg")
    ap.add_argument("--seconds", type=float, default=60.0, help="sim time")
    ap.add_argument("--ego-vmax", type=float, default=6.0)
    ap.add_argument("--opp-vmax", type=float, default=3.5)
    ap.add_argument("--gap", type=float, default=5.0, help="opponent start distance ahead [m]")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--plot", default="two_cars.png")
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

    def progress(o):  # unwrapped arc length along the centerline
        i = int(np.argmin(np.sum((xy_cl - [o["pose_x"], o["pose_y"]]) ** 2, 1)))
        return cl.ss[i]

    logs = {a: [] for a in obs}
    speeds = {a: [] for a in obs}
    s_unwrapped = {a: progress(obs[a]) for a in obs}
    s_unwrapped["agent_1"] += 0.0 if s_unwrapped["agent_1"] >= s_unwrapped["agent_0"] else L
    s_start = dict(s_unwrapped)
    s_prev = {a: progress(obs[a]) for a in obs}
    laps = {a: [] for a in obs}
    last_lap_count = {a: 0 for a in obs}
    crashed = {a: False for a in obs}
    events, lead = [], "agent_1"

    steps = int(args.seconds / 0.01)
    t0 = time.time()
    for k in range(steps):
        act = np.zeros((2, 2))
        for i, a in enumerate(obs):
            if crashed[a]:
                continue
            steer, v = ctrl[a].plan(obs[a]["scan"], float(obs[a]["linear_vel_x"]))
            act[i] = [steer, v]
        obs, _, done, _, _ = env.step(act)
        if args.render:
            env.render()

        for a in obs:
            o = obs[a]
            logs[a].append((float(o["pose_x"]), float(o["pose_y"])))
            speeds[a].append(float(o["linear_vel_x"]))
            s = progress(o)
            ds = s - s_prev[a]
            if ds < -L / 2:
                ds += L
            elif ds > L / 2:
                ds -= L
            s_unwrapped[a] += ds
            s_prev[a] = s
            if int(o["lap_count"]) > last_lap_count[a]:
                last_lap_count[a] = int(o["lap_count"])
                laps[a].append(float(o["lap_time"]))
            if o["collision"] > 0 and not crashed[a]:
                crashed[a] = True
                events.append((a, float(o["pose_x"]), float(o["pose_y"]), "collision"))
                print(f"[t={k*0.01:6.2f}s] {a} COLLISION")

        other = "agent_0" if lead == "agent_1" else "agent_1"
        new_lead = other if s_unwrapped[other] > s_unwrapped[lead] + 0.6 else lead  # 0.6 m margin vs flicker
        if new_lead != lead and not any(crashed.values()):
            o = obs[new_lead]
            events.append((new_lead, float(o["pose_x"]), float(o["pose_y"]), "pass"))
            print(f"[t={k*0.01:6.2f}s] {new_lead} takes the lead")
            lead = new_lead
        if all(crashed.values()) or crashed["agent_0"]:
            break

    wall = time.time() - t0
    sim_t = (k + 1) * 0.01
    print(f"\nsim {sim_t:.1f} s in {wall:.1f} s wall ({sim_t / wall:.1f}x real time)")
    for a in obs:
        lt = laps[a]
        cum = np.diff([0.0] + lt) if lt else []
        print(f"{a}: v_max {ctrl[a].p.v_max:.1f} | mean speed {np.mean(speeds[a]):.2f} m/s | "
              f"distance {s_unwrapped[a] - s_start[a]:.1f} m | "
              f"laps {len(lt)} {['%.2f' % x for x in cum]} | crashed {crashed[a]}")
    passes = sum(1 for e in events if e[3] == "pass")
    print(f"lead changes: {passes}")
    plot(args.map, logs, events, args.plot)
    print("plot ->", os.path.abspath(args.plot))


if __name__ == "__main__":
    main()
