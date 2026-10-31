"""Step 3 closed-loop check: let a learned decider pick the state every tick
(the teacher's executors still drive) on the 150 held-out test episodes, and
compare against the teacher's own rules on the same scenarios.

Offline accuracy flatters a model that copies its previous state; this is the
test that counts: does it crash, get stuck, or pass less than the teacher?

  python eval_closed_loop.py --model models/race_decider_xs.pt
  python eval_closed_loop.py --model models/lgbm_state.txt --episodes 60
  python eval_closed_loop.py --model models/race_decider_xs.pt --model models/lgbm_state.txt
"""
import argparse
import multiprocessing as mp
import time
import warnings

import numpy as np
import pandas as pd

from dataset_stats import split
from features import STATES

warnings.filterwarnings("ignore")


def _run(job):
    seed, n_center, model_path = job
    warnings.filterwarnings("ignore")
    from generate import run_episode, sample_scenario
    decider = None
    if model_path and model_path.endswith(".pt"):
        from race_decider import make_decider
        decider = make_decider(model_path)
    elif model_path:
        import lightgbm as lgb
        from features import FeatureExtractor
        m = lgb.Booster(model_file=model_path)
        cols = [FeatureExtractor.names().index(f) for f in m.feature_name()]

        def decider(feat):
            return STATES[int(m.predict(feat[cols][None], num_threads=1).argmax())]
    df, s = run_episode(sample_scenario(seed, n_center), decider=decider)
    s["agree"] = float((df.state == df.teacher_state).mean())
    for st in ("CAR_STOPPED", "OBSTACLE", "SPEED_DOWN"):
        s[f"share_{st}"] = float((df.state == st).mean())
    return s


def evaluate(seeds, n_center, model_path, workers):
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        return pd.DataFrame(pool.map(_run, [(s, n_center, model_path) for s in seeds]))


def summary(r):
    return {
        "episodes": len(r),
        "crashes": int(r.collision.notna().sum()),
        "stuck (<50 m)": int((r.distance < 50).sum()),
        "passes": int(r.passes.sum()),
        "mean distance m": round(r.distance.mean(), 1),
        "min car gap m": round(r.min_gap.min(), 3),
        "agree w/ rules %": round(100 * r.agree.mean(), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="LightGBM .txt or RaceDecider .pt model file (repeatable)")
    ap.add_argument("--episodes", type=int, default=150, help="first N test-split episodes")
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()

    from common import TrackModel, make_env
    env = make_env("Spielberg")
    n_center = TrackModel(env.unwrapped.track, "Spielberg").center.n
    env.close()
    cand = np.arange(1000)
    seeds = [int(s) for s in cand[split(cand) == "test"]][: args.episodes]

    out = {}
    for name, path in [("teacher", None)] + [(p.split("/")[-1].rsplit(".", 1)[0], p) for p in args.model]:
        t0 = time.time()
        r = evaluate(seeds, n_center, path, args.workers)
        out[name] = summary(r)
        print(f"{name}: {out[name]} ({time.time() - t0:.0f}s)", flush=True)
        if path:
            by = r.groupby("kind").agg(crashes=("collision", lambda c: int(c.notna().sum())),
                                       passes=("passes", "sum"), dist=("distance", "mean")).round(1)
            print(by.to_string(), "\n")
    print("\n" + pd.DataFrame(out).to_string())


if __name__ == "__main__":
    main()
