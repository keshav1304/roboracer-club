"""Pass/fail check: does the reactive stack overtake and clear obstacles?

Runs 10-lap races on PortoWide against a slow opponent at three speeds and
checks each run:
  - no collision (car-car or wall/box) for either car
  - the ego completes all laps
  - at least --min-overtakes overtakes by the ego
  - no body contact with any box (closest body-to-box distance > 0)
Clearances under 0.10 m are flagged as WARN (tight but not contact).

  python check.py                     # 3 scenarios, ~1 min
  python check.py --speeds 1.5 2.0 --laps 10 --min-overtakes 6
"""
import argparse
import json
import sys
import warnings

import numpy as np
from shapely.geometry import Polygon

import race
from reactive import FollowTheGap, FTGParams

warnings.filterwarnings("ignore")


def run(opp_v, laps, track, ego_v=6.0, gap=3.0, seconds=400.0):
    from f1tenth_gym.envs.collision_models import get_vertices

    env = race.make_env(track, False)
    u = env.unwrapped
    ang = u.sim.agents[0].scan_angles
    ctrl = {"agent_0": FollowTheGap(ang, FTGParams(v_max=ego_v)),
            "agent_1": FollowTheGap(ang, FTGParams(v_max=opp_v))}
    obs, _ = env.reset(options={"poses": race.start_poses(u.track, gap)})
    boxes = [Polygon(o["corners"]) for o in json.load(open(race.GYM_MAPS / track / "obstacles.json"))]

    def body(o):
        return Polygon(get_vertices(np.array([o["pose_x"], o["pose_y"], o["pose_theta"]], dtype=np.float64), 0.58, 0.31))

    cl = u.track.centerline
    L = float(cl.ss[-1])
    xy = np.stack([cl.xs, cl.ys], 1)
    prog = lambda o: cl.ss[int(np.argmin(((xy - [o["pose_x"], o["pose_y"]]) ** 2).sum(1)))]
    s = {a: prog(obs[a]) for a in obs}
    if s["agent_1"] < s["agent_0"]:
        s["agent_1"] += L
    start = dict(s)
    prev = {a: prog(obs[a]) for a in obs}
    n = int(np.floor((s["agent_0"] - s["agent_1"]) / L))
    overtakes = 0
    car_gap, box_gap = np.inf, {a: np.inf for a in obs}
    collision = None

    for k in range(int(seconds / 0.01)):
        act = np.array([ctrl[a].plan(obs[a]["scan"], 0.0) for a in obs])
        obs, *_ = env.step(act)
        bodies = {a: body(obs[a]) for a in obs}
        car_gap = min(car_gap, bodies["agent_0"].distance(bodies["agent_1"]))
        for a in obs:
            box_gap[a] = min(box_gap[a], min(bodies[a].distance(b) for b in boxes))
            p = prog(obs[a])
            d = p - prev[a]
            d += L if d < -L / 2 else -L if d > L / 2 else 0.0
            s[a] += d
            prev[a] = p
            if obs[a]["collision"] > 0 and collision is None:
                collision = (a, round((k + 1) * 0.01, 2))
        rel = s["agent_0"] - s["agent_1"]
        if rel > (n + 1) * L + 0.6:
            n += 1
            overtakes += 1
        elif rel < n * L - 0.6:
            n -= 1
        if collision or s["agent_0"] - start["agent_0"] >= laps * L:
            break
    env.close()
    return {
        "opp_v": opp_v,
        "time": round((k + 1) * 0.01, 1),
        "ego_laps": int((s["agent_0"] - start["agent_0"]) // L),
        "overtakes": overtakes,
        "collision": collision,
        "car_gap": round(float(car_gap), 3),
        "ego_box_gap": round(float(box_gap["agent_0"]), 3),
        "opp_box_gap": round(float(box_gap["agent_1"]), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="PortoWide")
    ap.add_argument("--speeds", type=float, nargs="+", default=[1.5, 2.0, 2.5], help="opponent v_max values")
    ap.add_argument("--laps", type=int, default=10)
    ap.add_argument("--min-overtakes", type=int, default=6)
    args = ap.parse_args()

    all_ok = True
    print(f"{'opp m/s':>7} {'laps':>5} {'overtakes':>9} {'car gap':>8} {'ego-box':>8} {'opp-box':>8}  result")
    for v in args.speeds:
        r = run(v, args.laps, args.track)
        fails, warns = [], []
        if r["collision"]:
            fails.append(f"collision {r['collision']}")
        if r["ego_laps"] < args.laps:
            fails.append(f"only {r['ego_laps']} laps")
        if r["overtakes"] < args.min_overtakes:
            fails.append(f"{r['overtakes']} < {args.min_overtakes} overtakes")
        for key in ("car_gap", "ego_box_gap", "opp_box_gap"):
            if r[key] <= 0.0:
                fails.append(f"{key} contact")
            elif r[key] < 0.10:
                warns.append(f"{key} {r[key]:.2f} m")
        status = "FAIL: " + "; ".join(fails) if fails else ("PASS (warn: " + ", ".join(warns) + ")" if warns else "PASS")
        all_ok &= not fails
        print(f"{v:7.1f} {r['ego_laps']:5d} {r['overtakes']:9d} {r['car_gap']:8.2f} {r['ego_box_gap']:8.2f} {r['opp_box_gap']:8.2f}  {status}", flush=True)
    print("\nOVERALL:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
