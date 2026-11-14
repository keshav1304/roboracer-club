"""Measure car-car contact: min body-to-body distance, contacts, who hit whom."""
import itertools, sys, warnings
import numpy as np
import run_two_cars as R
from reactive import FollowTheGap, FTGParams
from shapely.geometry import Polygon
from f1tenth_gym.envs.collision_models import get_vertices
warnings.filterwarnings("ignore")

def body(o, L=0.58, W=0.31):
    return Polygon(get_vertices(np.array([o["pose_x"], o["pose_y"], o["pose_theta"]], dtype=np.float64), L, W))

def run(opp_v, gap, seconds=125.0, ego_v=6.0, verbose=False):
    env = R.make_env("Spielberg", False); u = env.unwrapped
    ang = u.sim.agents[0].scan_angles
    c = {"agent_0": FollowTheGap(ang, FTGParams(v_max=ego_v)), "agent_1": FollowTheGap(ang, FTGParams(v_max=opp_v))}
    obs, _ = env.reset(options={"poses": R.start_poses(u.track, gap)})
    dmin, contacts, in_contact, wall = 9e9, [], False, {a: 0 for a in obs}
    stalled = {a: 0 for a in obs}
    for k in range(int(seconds / 0.01)):
        act = np.array([c[a].plan(obs[a]["scan"], 0.0) for a in obs])
        obs, *_ = env.step(act)
        b0, b1 = body(obs["agent_0"]), body(obs["agent_1"])
        d = b0.distance(b1); dmin = min(dmin, d)
        touching = d < 1e-3 or u.sim.collision_idx[0] >= 0
        if touching and not in_contact:
            o0, o1 = obs["agent_0"], obs["agent_1"]
            # who is behind: project relative position on ego heading
            rel = np.array([o1["pose_x"] - o0["pose_x"], o1["pose_y"] - o0["pose_y"]])
            ahead = rel @ [np.cos(o0["pose_theta"]), np.sin(o0["pose_theta"])]
            lat = rel @ [-np.sin(o0["pose_theta"]), np.cos(o0["pose_theta"])]
            contacts.append((round(k * 0.01, 2), "opp ahead" if ahead > 0 else "opp behind", round(float(lat), 2),
                             round(float(o0["linear_vel_x"]), 2), round(float(o1["linear_vel_x"]), 2)))
        in_contact = touching
        for i, a in enumerate(obs):
            if u.sim.agents[i].in_collision: wall[a] += 1
            if abs(obs[a]["linear_vel_x"]) < 0.1 and k > 100: stalled[a] += 1
    env.close()
    return dmin, contacts, wall, stalled

if __name__ == "__main__":
    tot = 0
    for opp_v, gap in itertools.product([2.0, 3.0, 3.5, 4.0, 5.0], [3.0, 5.0, 8.0]):
        dmin, contacts, wall, stalled = run(opp_v, gap)
        tot += len(contacts)
        print(f"opp_v={opp_v} gap={gap}: min body gap {dmin:.3f} m | contacts {len(contacts)} {contacts[:3]} | wall-stop steps {sum(wall.values())} | stalled s {[round(v*0.01,1) for v in stalled.values()]}", flush=True)
    print("total contacts:", tot)
