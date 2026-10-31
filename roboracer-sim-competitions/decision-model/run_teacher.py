"""Race the teacher state machine (or plain FTG) as the ego against a slower FTG car.

  python run_teacher.py                       # teacher ego, Spielberg, 125 s, writes teacher.png
  python run_teacher.py --render
  python run_teacher.py --ego ftg             # same race with the plain FTG ego, for comparison
  python run_teacher.py --solo --seconds 125  # ego alone: lap times only
  python run_teacher.py --opp-vmax 0 --gap 30 --opp-on-raceline   # parked car on our line → OBSTACLE
"""
import argparse
import os
import time
import warnings
from collections import Counter

import numpy as np
from shapely.geometry import Polygon

from common import CAR_L, CAR_W, DT, FollowTheGap, FTGParams, OpponentSensor, TrackModel, baseline, make_env
from teacher import STATES, Teacher, TeacherParams, params_for

warnings.filterwarnings("ignore")
STATE_COLORS = {"CAR_STOPPED": "#7f7f7f", "OBSTACLE": "#ff7f0e", "OVERTAKE": "#d62728",
                "CORNERING": "#9467bd", "SPEED_UP": "#2ca02c", "SPEED_DOWN": "#1f77b4"}


def body(o):
    from f1tenth_gym.envs.collision_models import get_vertices
    return Polygon(get_vertices(np.array([o["pose_x"], o["pose_y"], o["pose_theta"]], dtype=np.float64), CAR_L, CAR_W))


def race(ego="teacher", map_name="Spielberg", seconds=125.0, opp_v=3.5, gap=5.0, solo=False,
         render=False, seed=0, params=None, verbose=True, opp_on_raceline=False, model=None):
    """model: optional .pt (RaceDecider) or .txt (LightGBM) that picks the state
    each tick instead of the teacher's rules (the teacher's executors still drive)."""
    env = make_env(map_name, render, num_agents=1 if solo else 2)
    u = env.unwrapped
    ang = u.sim.agents[0].scan_angles
    tm = TrackModel(u.track, map_name)
    p = params or params_for(map_name)
    ego_ctrl = Teacher(tm, ang, p) if ego == "teacher" else FollowTheGap(ang, FTGParams(v_max=p.v_max))
    opp_ctrl = FollowTheGap(ang, FTGParams(v_max=opp_v))
    decide, fx = None, None
    if model and ego == "teacher":
        from features import STATES as FSTATES, FeatureExtractor
        fx = FeatureExtractor(tm, p.v_max, p.aggression, p.static_window, p.stuck_window, p.static_min)
        if model.endswith(".pt"):
            from race_decider import make_decider
            decide = make_decider(model)
        else:
            import lightgbm as lgb
            booster = lgb.Booster(model_file=model)
            decide = lambda f: FSTATES[int(booster.predict(f[None], num_threads=1).argmax())]
    sensor = OpponentSensor(seed=seed)
    # same start as the baseline (centerline), so teacher vs FTG is a fair comparison;
    # the raceline start is 0.27 m from a wall and plain FTG can't pull away from it
    poses = baseline.start_poses(u.track, gap)[:1 if solo else 2]
    if opp_on_raceline and not solo:      # put the opponent on the (safe) raceline, gap m ahead
        s0 = tm.race.project(*poses[0, :2])[0]
        j = tm.race.index_at(s0 + gap)
        poses[1] = [*tm.race.xy[j], tm.race.yaw[j]]
    obs, _ = env.reset(options={"poses": poses})
    if render:
        env.render()

    line = tm.center
    s_un = {a: line.project(obs[a]["pose_x"], obs[a]["pose_y"])[0] for a in obs}
    if not solo and s_un["agent_1"] < s_un["agent_0"]:
        s_un["agent_1"] += line.L
    start, prev = dict(s_un), {a: line.project(obs[a]["pose_x"], obs[a]["pose_y"])[0] for a in obs}
    lap_marks = {a: [0.0] for a in obs}
    n = 0 if solo else int(np.floor((s_un["agent_0"] - s_un["agent_1"]) / line.L))
    passes, states, trace, events = [], [], [], []
    collision, car_gap = None, np.inf
    t0 = time.time()

    for k in range(int(seconds / DT)):
        t = (k + 1) * DT
        o0, o1 = obs["agent_0"], obs.get("agent_1")
        if ego == "teacher":
            sensed = None if solo else sensor.sense(o1, o0)
            if decide is None:
                steer, v, st, _ = ego_ctrl.plan(o0, sensed)
            else:
                feat, _ = fx(o0, sensed)
                steer, v, st, _ = ego_ctrl.plan(o0, sensed, lambda sit: decide(feat))
                fx.observe_decision(st)
        else:
            steer, v = ego_ctrl.plan(o0["scan"], 0.0)
            st = "FTG"
        act = [[steer, v]] if solo else [[steer, v], list(opp_ctrl.plan(o1["scan"], 0.0))]
        obs, *_ = env.step(np.array(act))
        if render:
            env.render()
        states.append(st)
        trace.append((float(obs["agent_0"]["pose_x"]), float(obs["agent_0"]["pose_y"]), st))

        if not solo:
            car_gap = min(car_gap, body(obs["agent_0"]).distance(body(obs["agent_1"])))
        for a in obs:
            s = line.project(obs[a]["pose_x"], obs[a]["pose_y"])[0]
            d = s - prev[a]
            d += line.L if d < -line.L / 2 else -line.L if d > line.L / 2 else 0.0
            s_un[a] += d
            prev[a] = s
            if s_un[a] - start[a] >= len(lap_marks[a]) * line.L:
                lap_marks[a].append(t)
            if obs[a]["collision"] > 0 and collision is None:
                collision = (a, round(t, 2))
                events.append(("collision", float(obs[a]["pose_x"]), float(obs[a]["pose_y"])))
                if verbose:
                    print(f"[t={t:6.2f}s] {a} COLLISION")
        if not solo:
            rel = s_un["agent_0"] - s_un["agent_1"]
            if rel > (n + 1) * line.L + 0.6:
                n += 1
                passes.append(round(t, 2))
                events.append(("pass", float(obs["agent_0"]["pose_x"]), float(obs["agent_0"]["pose_y"])))
                if verbose:
                    print(f"[t={t:6.2f}s] ego overtakes (state {st})")
            elif rel < n * line.L - 0.6:
                n -= 1
        if collision:
            break

    env.close()
    laps = np.diff(lap_marks["agent_0"])
    res = {
        "ego": ego, "time": round(t, 1), "wall": round(time.time() - t0, 1),
        "laps": [round(x, 2) for x in laps], "passes": passes, "collision": collision,
        "car_gap": None if solo else round(float(car_gap), 3),
        "distance": round(s_un["agent_0"] - start["agent_0"], 1),
        "time_in_state": {k2: round(v2 * DT, 1) for k2, v2 in Counter(states).items()},
    }
    return res, trace, events


def plot(map_name, trace, events, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml
    from PIL import Image

    d = baseline.GYM_MAPS / map_name
    spec = yaml.safe_load(open(d / f"{map_name}_map.yaml"))
    img = np.array(Image.open(d / spec["image"]).convert("L"))
    res, (ox, oy, _) = spec["resolution"], spec["origin"]
    h, w = img.shape
    fig, ax = plt.subplots(figsize=(10, 7), dpi=130)
    ax.imshow(img, cmap="gray", extent=[ox, ox + w * res, oy, oy + h * res], origin="upper")
    xy = np.array([(x, y) for x, y, _ in trace])
    st = np.array([s for _, _, s in trace])
    for name, c in STATE_COLORS.items():
        m = st == name
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=1.5, color=c, label=name)
    if (st == "FTG").any():
        ax.scatter(xy[st == "FTG", 0], xy[st == "FTG", 1], s=1.5, color="k", label="FTG")
    for kind, x, y in events:
        ax.plot(x, y, "x" if kind == "collision" else "^", color="k", ms=9, mew=2)
    pad = 3.0
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", markerscale=6)
    ax.set_title(f"{map_name}: ego path coloured by decision state (▲ = overtake, x = collision)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ego", choices=["teacher", "ftg"], default="teacher")
    ap.add_argument("--map", default="Spielberg")
    ap.add_argument("--seconds", type=float, default=125.0)
    ap.add_argument("--opp-vmax", type=float, default=3.5)
    ap.add_argument("--gap", type=float, default=5.0)
    ap.add_argument("--solo", action="store_true", help="ego alone on track (lap times)")
    ap.add_argument("--opp-on-raceline", action="store_true", help="start the opponent on the raceline")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--model", default=None, help="drive with a learned decider: models/race_decider_xs.pt or models/lgbm_state.txt")
    args = ap.parse_args()

    res, trace, events = race(args.ego, args.map, args.seconds, args.opp_vmax, args.gap, args.solo, args.render,
                              opp_on_raceline=args.opp_on_raceline, model=args.model)
    print(f"\n{res['ego']} ego: sim {res['time']} s in {res['wall']} s wall | distance {res['distance']} m")
    print(f"laps: {res['laps']}")
    print(f"overtakes: {len(res['passes'])} at {res['passes']} | collision: {res['collision']} | closest car gap: {res['car_gap']} m")
    tot = sum(res["time_in_state"].values())
    print("time in state: " + ", ".join(f"{k} {v:.1f}s ({100 * v / tot:.0f}%)"
                                        for k, v in sorted(res["time_in_state"].items(), key=lambda kv: -kv[1])))
    out = args.plot or f"{args.ego}.png"
    plot(args.map, trace, events, out)
    print("plot ->", os.path.abspath(out))


if __name__ == "__main__":
    main()
