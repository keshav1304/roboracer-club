"""Step 5: DAgger + counterfactual labels.

The learned decider drives fresh scenarios (seeds from 1000, so none overlap
the step-2 test split). Every logged row gets two labels:

  teacher_label  what the teacher's rules (with their 0.3 s dwell) would pick in
                 the state the *model* got itself into. This is DAgger: the model
                 learns to recover from its own mistakes, not just the teacher's
                 good trajectories.
  label          teacher_label, unless a counterfactual says otherwise. At
                 interesting moments (opponent in view, and the model disagrees
                 with the rules or a random 15%), the sim is snapshotted and each
                 candidate state is forced for `branch_s`, then the rules drive for
                 `settle_s`. Score = raceline progress − 50 per collision −
                 20 per metre of contact closer than 0.10 m. If the best candidate
                 beats the teacher's choice by > `margin` m, it becomes the label.
                 That is how the model can end up better than the teacher.

  python dagger.py --model models/race_decider_xs.pt --episodes 400 --out data/dagger1
"""
import argparse
import copy
import multiprocessing as mp
import os
import pathlib
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

CANDIDATES = ("OVERTAKE", "SPEED_DOWN", "SPEED_UP", "OBSTACLE")


class Episode:
    """One scenario with every piece of mutable state in one place, so it can
    be snapshotted and restored exactly (gym sim + controllers + sensor RNG)."""

    def __init__(self, sc):
        from shapely.geometry import Polygon  # noqa: F401  (warm import)
        from common import FollowTheGap, FTGParams, OpponentSensor, TrackModel, make_env
        from features import FeatureExtractor
        from generate import _pose_on
        from teacher import Teacher, TeacherParams

        self.sc, self.solo = sc, sc["kind"] == "solo"
        self.env = make_env("Spielberg", num_agents=1 if self.solo else 2)
        u = self.env.unwrapped
        ang = u.sim.agents[0].scan_angles
        self.tm = TrackModel(u.track, "Spielberg")
        p = TeacherParams()
        self.p = p
        self.teacher = Teacher(self.tm, ang, p)
        self.fx = FeatureExtractor(self.tm, p.v_max, p.aggression, p.static_window, p.stuck_window, p.static_min)
        self.sensor = OpponentSensor(seed=sc["seed"])
        self.opp = FollowTheGap(ang, FTGParams(v_max=max(sc["opp_v"], 0.5)))
        c, r = self.tm.center, self.tm.race
        poses = [_pose_on(c, sc["start_idx"], sc["ego_d"], sc["ego_yaw"])]
        if not self.solo:
            if sc["opp_on_raceline"]:
                poses.append(_pose_on(r, r.index_at(r.project(*poses[0][:2])[0] + sc["gap"]), sc["opp_d"], 0.0))
            else:
                poses.append(_pose_on(c, c.index_at(c.s[sc["start_idx"]] + sc["gap"]), 0.0, 0.0))
        self.obs, _ = self.env.reset(options={"poses": np.array(poses)})
        self.k, self.collided = 0, False

    # ------------------------------------------------------------ snapshot
    def snapshot(self):
        u = self.env.unwrapped
        return copy.deepcopy((u.sim, self.obs, self.teacher, self.fx, self.sensor, self.opp, self.k, self.collided))

    def restore(self, snap):
        u = self.env.unwrapped
        (u.sim, self.obs, self.teacher, self.fx, self.sensor, self.opp, self.k, self.collided) = copy.deepcopy(snap)

    # ---------------------------------------------------------------- step
    def body_gap(self):
        from shapely.geometry import Polygon
        from f1tenth_gym.envs.collision_models import get_vertices
        from common import CAR_L, CAR_W
        if self.solo:
            return np.inf
        b = [Polygon(get_vertices(np.array([o["pose_x"], o["pose_y"], o["pose_theta"]], np.float64), CAR_L, CAR_W))
             for o in (self.obs["agent_0"], self.obs["agent_1"])]
        return b[0].distance(b[1])

    def step(self, choose):
        """choose(feat, teacher_label) → state, or None for the teacher's rules.
        Returns (feat, state_used, teacher_label)."""
        from teacher import SAFETY
        from common import DT
        o0, o1 = self.obs["agent_0"], self.obs.get("agent_1")
        sensed = None if self.solo else self.sensor.sense(o1, o0)
        feat, _ = self.fx(o0, sensed)
        T, p = self.teacher, self.p
        out = {}

        def decide(sit):
            want = T._choose(sit)
            # the label the teacher would actually commit to, dwell included
            out["teacher"] = want if (want != T.state and (want in SAFETY or T.state in SAFETY
                                                          or T.t_in_state >= p.dwell)) else T.state
            pick = choose(feat, out["teacher"]) if choose else None
            return want if pick is None else pick
        steer, v, st, _ = T.plan(o0, sensed, decide)
        self.fx.observe_decision(st)
        if self.solo:
            act = [[steer, v]]
        else:
            t = self.k * DT
            if self.sc["kind"] == "parked":
                oa = [0.0, 0.0]
            else:
                oa = list(self.opp.plan(o1["scan"], 0.0))
                if self.sc["kind"] == "stopgo" and (t % self.sc["stop_every"]) > self.sc["stop_every"] - self.sc["stop_for"]:
                    oa[1] = 0.0
            act = [[steer, v], oa]
        self.obs, *_ = self.env.step(np.array(act))
        self.k += 1
        if any(self.obs[a]["collision"] > 0 for a in self.obs):
            self.collided = True
        return feat, st, out["teacher"]

    def progress(self):
        return self.fx.s_un


def score_branch(ep, snap, state, branch_s, settle_s):
    """Force `state` for branch_s, then the teacher's rules for settle_s."""
    from common import DT
    ep.restore(snap)
    ep.teacher.state, ep.teacher.t_in_state, ep.teacher.pass_done_at = state, 0.0, None
    s0, gap_min = ep.progress(), np.inf
    n1, n2 = int(branch_s / DT), int(settle_s / DT)
    for j in range(n1 + n2):
        ep.step((lambda f, t: state) if j < n1 else None)
        gap_min = min(gap_min, ep.body_gap())
        if ep.collided:
            return ep.progress() - s0 - 50.0
    return ep.progress() - s0 - 20.0 * max(0.0, 0.10 - gap_min)


def run(seed, n_center, model_path, branch_s=1.5, settle_s=1.5, margin=0.3, max_branches=40, log_every=5):
    import pandas as pd
    from features import STATES, FeatureExtractor
    from generate import sample_scenario
    from race_decider import make_decider

    sc = sample_scenario(seed, n_center)
    model = make_decider(model_path)
    ep = Episode(sc)
    rng = np.random.default_rng(seed + 7)
    rows, X = [], []
    n = int(sc["seconds"] / 0.01)
    branches, overridden = 0, 0
    while ep.k < n and not ep.collided:
        k = ep.k
        log = k % log_every == 0
        snap = ep.snapshot() if log and branches < max_branches and not ep.solo else None
        feat, st, lab = ep.step(lambda f, t: model(f))
        if not log:
            continue
        cf, gain, scores = lab, 0.0, ""
        opp_seen = feat[FeatureExtractor.names().index("opp0_present")] > 0
        if snap is not None and opp_seen and (st != lab or rng.random() < 0.15):
            after = ep.snapshot()
            res = {s: score_branch(ep, snap, s, branch_s, settle_s) for s in dict.fromkeys((lab, st) + CANDIDATES)}
            ep.restore(after)
            best = max(res, key=res.get)
            branches += 1
            if best != lab and res[best] > res[lab] + margin:
                cf, gain = best, res[best] - res[lab]
                overridden += 1
            scores = ";".join(f"{s}:{v:.2f}" for s, v in res.items())
        X.append(feat)
        rows.append((k, round(k * 0.01, 2), st, lab, cf, gain, scores))

    df = pd.DataFrame(np.vstack(X), columns=FeatureExtractor.names())
    meta = pd.DataFrame(rows, columns=["tick", "t", "state", "teacher_label", "label", "cf_gain", "cf_scores"])
    df = pd.concat([meta, df], axis=1)
    df.insert(0, "episode", seed)
    df.insert(1, "kind", sc["kind"])
    df["y_horizon_ok"] = False          # no outcome labels here: aux heads train on step-2 data only
    for c in ("y_collide_1s", "y_pass_3s"):
        df[c] = False
    df["y_progress_2s"] = 0.0
    df["y_min_gap_2s"] = np.nan
    ep.env.close()
    agree = float((df.state == df.teacher_label).mean())
    return df, {"seed": seed, "kind": sc["kind"], "rows": len(df), "collision": ep.collided,
                "branches": branches, "overridden": overridden, "agree": round(agree, 4),
                "distance": round(float(ep.progress()), 1), "ticks": ep.k}


def _work(job):
    seed, n_center, model_path, out = job
    warnings.filterwarnings("ignore")
    t0 = time.time()
    df, s = run(seed, n_center, model_path)
    df.to_parquet(out / "episodes" / f"ep_{seed:06d}.parquet", index=False)
    s["wall_s"] = round(time.time() - t0, 1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/race_decider_xs.pt")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--out", default="data/dagger1")
    args = ap.parse_args()

    import pandas as pd
    from common import TrackModel, make_env
    env = make_env("Spielberg")
    n_center = TrackModel(env.unwrapped.track, "Spielberg").center.n
    env.close()
    out = pathlib.Path(args.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    seeds = [s for s in range(args.first_seed, args.first_seed + args.episodes)
             if not (out / "episodes" / f"ep_{s:06d}.parquet").exists()]
    print(f"{len(seeds)} DAgger episodes on {args.workers} workers → {out}", flush=True)
    t0, done = time.time(), []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, s in enumerate(pool.imap_unordered(_work, [(sd, n_center, args.model, out) for sd in seeds]), 1):
            done.append(s)
            if i % 20 == 0 or i == len(seeds):
                print(f"[{i}/{len(seeds)} {time.time() - t0:5.0f}s] last: ep {s['seed']} {s['kind']} "
                      f"branches {s['branches']} overridden {s['overridden']} agree {s['agree']}", flush=True)
    man = pd.DataFrame(done)
    if (out / "manifest.parquet").exists():
        man = pd.concat([pd.read_parquet(out / "manifest.parquet"), man]).drop_duplicates("seed", keep="last")
    man.sort_values("seed").to_parquet(out / "manifest.parquet", index=False)
    print(f"done in {time.time() - t0:.0f}s: {man.rows.sum():,} rows, {man.branches.sum():,} counterfactual branch "
          f"points, {man.overridden.sum():,} labels overridden, {man.collision.sum()} crash episodes, "
          f"agreement with rules {100 * man.agree.mean():.1f}%")


if __name__ == "__main__":
    main()
