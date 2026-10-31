"""Step 2: drive the teacher through randomized scenarios and log
(features, teacher state, action, outcomes) for training.

Each episode samples a scenario (random start on the track, opponent type,
speed, gap, placement), runs the teacher at 100 Hz, logs every `log_every`-th
tick, and then fills in the outcome columns by looking ahead in the episode:

  y_collide_1s   ego or opponent collision within the next 1 s
  y_progress_2s  raceline metres gained over the next 2 s (to episode end if it stops)
  y_min_gap_2s   closest body-to-body gap over the next 2 s (NaN without an opponent)
  y_pass_3s      ego completes a pass within the next 3 s
  y_horizon_ok   the full 3 s look-ahead fits inside the episode

One parquet per episode in <out>/episodes/, plus <out>/manifest.parquet
with one row per episode. Episode id = seed, so any episode is reproducible.

  python generate.py --episodes 400 --workers 9 --out data/v1
  python generate.py --episodes 4 --workers 1 --out data/smoke   # quick check
"""
import argparse
import multiprocessing as mp
import os
import pathlib
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# scenario mix: (kind, probability)
KINDS = [("race", 0.30), ("close_race", 0.10), ("parked", 0.20), ("stopgo", 0.15), ("solo", 0.10), ("hard_start", 0.15)]
LOG_EVERY = 5       # 100 Hz sim → 20 Hz rows (neighbouring ticks are near-duplicates)


def sample_scenario(seed, n_center):
    r = np.random.default_rng(seed)
    kind = r.choice([k for k, _ in KINDS], p=[p for _, p in KINDS])
    sc = {
        "seed": int(seed), "kind": str(kind),
        "start_idx": int(r.integers(n_center)),
        "ego_d": float(r.uniform(-0.3, 0.3)),
        "ego_yaw": float(r.normal(0, 0.1)),
        "opp_v": float(r.uniform(1.5, 5.5)),
        "gap": float(r.uniform(2.0, 15.0)),
        "opp_on_raceline": bool(r.random() < 0.5),
        "opp_d": 0.0,
        "stop_every": float(r.uniform(3.0, 8.0)), "stop_for": float(r.uniform(1.0, 3.0)),
        "seconds": 40.0,
    }
    if kind == "parked":
        sc.update(opp_v=0.0, gap=float(r.uniform(9.0, 40.0)), opp_on_raceline=True,
                  opp_d=float(r.uniform(-0.3, 0.3)))
    elif kind == "close_race":   # opponent nearly as fast: no overtake possible → trailing (SPEED_DOWN)
        sc.update(opp_v=float(r.uniform(5.0, 6.0)), gap=float(r.uniform(1.5, 4.0)))
    elif kind == "hard_start":   # facing away from the track direction, maybe near a wall
        sc.update(ego_yaw=float(r.choice([-1, 1]) * r.uniform(0.6, 2.0)), ego_d=float(r.uniform(-0.6, 0.6)),
                  seconds=20.0, gap=float(r.uniform(5.0, 15.0)))
    return sc


def _pose_on(line, idx, d, dyaw):
    x, y = line.xy[idx]
    yaw = line.yaw[idx]
    return [x - d * np.sin(yaw), y + d * np.cos(yaw), yaw + dyaw]


def run_episode(sc, log_every=LOG_EVERY, decider=None):
    """decider: optional callable(features) → state. When given, it picks the
    state each tick instead of the teacher's rules (closed-loop evaluation);
    the logged `state` is then the decider's and `teacher_state` what the rules
    would have picked."""
    from shapely.geometry import Polygon
    from f1tenth_gym.envs.collision_models import get_vertices

    from common import CAR_L, CAR_W, DT, FollowTheGap, FTGParams, OpponentSensor, TrackModel, make_env
    from features import FeatureExtractor
    from teacher import Teacher, TeacherParams

    solo = sc["kind"] == "solo"
    env = make_env("Spielberg", num_agents=1 if solo else 2)
    u = env.unwrapped
    ang = u.sim.agents[0].scan_angles
    tm = TrackModel(u.track, "Spielberg")
    p = TeacherParams()
    teacher = Teacher(tm, ang, p)
    fx = FeatureExtractor(tm, p.v_max, p.aggression, p.static_window, p.stuck_window, p.static_min)
    sensor = OpponentSensor(seed=sc["seed"])
    opp_ctrl = FollowTheGap(ang, FTGParams(v_max=max(sc["opp_v"], 0.5)))

    c, r = tm.center, tm.race
    poses = [_pose_on(c, sc["start_idx"], sc["ego_d"], sc["ego_yaw"])]
    if not solo:
        s0 = c.s[sc["start_idx"]]
        if sc["opp_on_raceline"]:
            poses.append(_pose_on(r, r.index_at(r.project(*poses[0][:2])[0] + sc["gap"]), sc["opp_d"], 0.0))
        else:
            poses.append(_pose_on(c, c.index_at(s0 + sc["gap"]), 0.0, 0.0))
    obs, _ = env.reset(options={"poses": np.array(poses)})

    def body(o):
        return Polygon(get_vertices(np.array([o["pose_x"], o["pose_y"], o["pose_theta"]], np.float64), CAR_L, CAR_W))

    n = int(sc["seconds"] / DT)
    X, rows = [], []
    s_ego, gap, coll = np.zeros(n), np.full(n, np.nan), None
    rel_un, rel_prev, passes = np.zeros(n), None, []
    k = -1
    for k in range(n):
        t = k * DT
        o0, o1 = obs["agent_0"], obs.get("agent_1")
        sensed = None if solo else sensor.sense(o1, o0)
        feat, aux = fx(o0, sensed)
        if decider is None:
            steer, v, st, _ = teacher.plan(o0, sensed)
            rule = st
        else:
            seen = {}
            def decide(sit):
                seen["rule"] = teacher._choose(sit)
                return decider(feat)
            steer, v, st, _ = teacher.plan(o0, sensed, decide)
            rule = seen["rule"]
        fx.observe_decision(st)
        s_ego[k] = aux["s_un"]
        if k % log_every == 0:
            X.append(feat)
            rows.append((k, round(t, 2), st, rule, steer, v))
        if solo:
            act = [[steer, v]]
        else:
            if sc["kind"] == "parked":
                oa = [0.0, 0.0]
            else:
                oa = list(opp_ctrl.plan(o1["scan"], 0.0))
                if sc["kind"] == "stopgo" and (t % sc["stop_every"]) > sc["stop_every"] - sc["stop_for"]:
                    oa[1] = 0.0
            act = [[steer, v], oa]
        obs, *_ = env.step(np.array(act))
        if not solo:
            a, b = obs["agent_0"], obs["agent_1"]
            gap[k] = body(a).distance(body(b))
            rel = r.ds(r.project(a["pose_x"], a["pose_y"])[0], r.project(b["pose_x"], b["pose_y"])[0])
            if rel_prev is not None:   # unwrap the relative s so a lap doesn't look like a pass
                rel_un[k] = rel_un[k - 1] + r.ds(rel_prev, rel) if abs(rel - rel_prev) < r.L / 2 else rel_un[k - 1]
            else:
                rel_un[k] = rel
            rel_prev = rel
        if any(obs[a_]["collision"] > 0 for a_ in obs):
            coll = (k, "ego" if obs["agent_0"]["collision"] > 0 else "opp")
            break
    env.close()
    n_run = k + 1

    # passes: opponent goes from ahead to ≥ CAR_L + 0.6 m behind
    pass_ticks = []
    if not solo:
        ahead = rel_un[0] > 0
        for j in range(1, n_run):
            if ahead and rel_un[j] < -(CAR_L + 0.6):
                pass_ticks.append(j)
                ahead = False
            elif not ahead and rel_un[j] > CAR_L + 0.6:
                ahead = True
    pass_ticks = np.array(pass_ticks, int)

    H1, H2, H3 = int(1 / DT), int(2 / DT), int(3 / DT)
    out = {c_: [] for c_ in ("y_collide_1s", "y_progress_2s", "y_min_gap_2s", "y_pass_3s", "y_horizon_ok")}
    for k_, *_ in rows:
        j2 = min(k_ + H2, n_run - 1)
        out["y_collide_1s"].append(coll is not None and k_ < coll[0] <= k_ + H1)
        out["y_progress_2s"].append(float(s_ego[j2] - s_ego[k_]))
        out["y_min_gap_2s"].append(np.nan if solo else float(np.nanmin(gap[k_:j2 + 1])))
        out["y_pass_3s"].append(bool(np.any((pass_ticks > k_) & (pass_ticks <= k_ + H3))))
        # a crash ends the episode but its outcome is known: those rows are the collision positives
        out["y_horizon_ok"].append(k_ + H3 < n_run or coll is not None)

    import pandas as pd
    from features import FeatureExtractor as F
    df = pd.DataFrame(np.vstack(X), columns=F.names())
    meta = pd.DataFrame(rows, columns=["tick", "t", "state", "teacher_state", "act_steer", "act_speed"])
    df = pd.concat([meta, df, pd.DataFrame(out)], axis=1)
    df.insert(0, "episode", sc["seed"])
    df.insert(1, "kind", sc["kind"])
    summary = {**sc, "ticks": n_run, "rows": len(df), "passes": len(pass_ticks),
               "collision": None if coll is None else coll[1],
               "collision_t": None if coll is None else round(coll[0] * DT, 2),
               "distance": round(float(s_ego[n_run - 1]), 1),
               "min_gap": None if solo else round(float(np.nanmin(gap[:n_run])), 3)}
    return df, summary


def _work(job):
    seed, out = job
    t0 = time.time()
    from common import TrackModel  # noqa: F401  (warm imports inside the worker)
    import gymnasium  # noqa: F401
    n_center = _work.n_center
    sc = sample_scenario(seed, n_center)
    df, summary = run_episode(sc)
    df.to_parquet(out / "episodes" / f"ep_{seed:06d}.parquet", index=False)
    summary["wall_s"] = round(time.time() - t0, 1)
    return summary


def _init(n_center):
    warnings.filterwarnings("ignore")
    _work.n_center = n_center


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="data/v1")
    args = ap.parse_args()

    import pandas as pd
    from common import TrackModel, make_env
    env = make_env("Spielberg")
    n_center = TrackModel(env.unwrapped.track, "Spielberg").center.n
    env.close()

    out = pathlib.Path(args.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    seeds = [s for s in range(args.first_seed, args.first_seed + args.episodes)
             if not (out / "episodes" / f"ep_{s:06d}.parquet").exists()]   # resumable
    print(f"{len(seeds)} episodes to run on {args.workers} workers → {out}", flush=True)
    t0, done = time.time(), []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(n_center,)) as pool:
        for i, s in enumerate(pool.imap_unordered(_work, [(sd, out) for sd in seeds]), 1):
            done.append(s)
            print(f"[{i}/{len(seeds)} {time.time() - t0:6.0f}s] ep {s['seed']:5d} {s['kind']:10s} "
                  f"rows {s['rows']:4d} passes {s['passes']} coll {s['collision']}", flush=True)

    man = out / "manifest.parquet"
    new = pd.DataFrame(done)
    if man.exists() and len(new):
        new = pd.concat([pd.read_parquet(man), new]).drop_duplicates("seed", keep="last")
    elif man.exists():
        new = pd.read_parquet(man)
    new.sort_values("seed").to_parquet(man, index=False)
    print(f"done in {time.time() - t0:.0f}s; manifest {man} ({len(new)} episodes)")


if __name__ == "__main__":
    main()
