"""Grid of two-car scenarios (opponent speed x start gap); counts crashes and passes."""
import itertools, sys, warnings
import numpy as np
import run_two_cars as R
from reactive import FollowTheGap, FTGParams
warnings.filterwarnings("ignore")

def run(opp_v, gap, ego_v=6.0, seconds=40.0, map_name="Spielberg"):
    env = R.make_env(map_name, False); u = env.unwrapped
    ang = u.sim.agents[0].scan_angles
    c = {"agent_0": FollowTheGap(ang, FTGParams(v_max=ego_v)), "agent_1": FollowTheGap(ang, FTGParams(v_max=opp_v))}
    obs, _ = env.reset(options={"poses": R.start_poses(u.track, gap)})
    cl = u.track.centerline; L = float(cl.ss[-1]); xy = np.stack([cl.xs, cl.ys], 1)
    prog = lambda o: cl.ss[int(np.argmin(((xy - [o["pose_x"], o["pose_y"]]) ** 2).sum(1)))]
    s = {a: prog(obs[a]) for a in obs}; s["agent_1"] += 0 if s["agent_1"] >= s["agent_0"] else L
    prev = {a: prog(obs[a]) for a in obs}
    crash = {a: None for a in obs}; pass_t = None
    for k in range(int(seconds / 0.01)):
        act = np.zeros((2, 2))
        for i, a in enumerate(obs):
            if crash[a] is None:
                act[i] = c[a].plan(obs[a]["scan"], 0.0)
        obs, *_ = env.step(act)
        for a in obs:
            p = prog(obs[a]); d = p - prev[a]; d += L if d < -L/2 else -L if d > L/2 else 0; s[a] += d; prev[a] = p
            if obs[a]["collision"] > 0 and crash[a] is None: crash[a] = k * 0.01
        if pass_t is None and s["agent_0"] > s["agent_1"] + 0.6: pass_t = k * 0.01
        if crash["agent_0"] is not None: break
    env.close()
    return crash, pass_t

if __name__ == "__main__":
    rows = []
    for opp_v, gap in itertools.product([2.0, 3.0, 4.0, 5.0], [3.0, 5.0, 8.0]):
        crash, pt = run(opp_v, gap)
        rows.append((opp_v, gap, crash["agent_0"], crash["agent_1"], pt))
        print(f"opp_v={opp_v:.1f} gap={gap:.0f}m | ego crash={crash['agent_0']} opp crash={crash['agent_1']} | pass at={pt}", flush=True)
    ok = sum(1 for r in rows if r[2] is None and r[3] is None)
    passed = sum(1 for r in rows if r[4] is not None and r[2] is None and r[3] is None)
    print(f"\nclean runs {ok}/{len(rows)} | clean passes {passed}/{len(rows)}")
