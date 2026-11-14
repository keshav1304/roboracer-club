"""Audit the real stage-2 rival and optionally measure its response to ego.

No-threat runs place a line-following car half a lap away. The matched ablation
leaves both cars' physics and collision scoring unchanged, but removes ego's
LiDAR returns from the rival only. A pass with and without those returns does
not establish induced yielding. The audit never calls an LLM.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np

from bootstrap import ARTIFACTS
from run_experiment import Scenario, episode
from track_geometry import PathTracker, TrackGeometry
from world import DT, make_env, make_stage2_opponent, reset_scenario


def source_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("opponent_audit.py", "world.py", "run_experiment.py",
                         "maneuver_core.py", "perception.py", "track_geometry.py")}


def no_threat_trial(env, geometry, start, seconds, seed):
    for i, agent in enumerate(env.unwrapped.sim.agents):
        agent.seed = seed + i
    obs, _ = reset_scenario(env, geometry, start, geometry.length / 2, 2.4, ego_speed=2.4)
    rival = make_stage2_opponent(geometry, env.unwrapped.sim.agents[1].scan_angles,
                                2.4, obs["agent_1"], env.unwrapped.sim.agents[1].state[2])
    other = PathTracker(geometry)
    other.last_steer = float(env.unwrapped.sim.agents[0].state[2])
    speeds, offsets, frames, collisions = [], [], [], []
    target_ticks, action_counts = 0, {}
    for tick in range(round(seconds / DT)):
        command = rival.plan(obs["agent_1"])
        baseline = other.control(obs["agent_0"], vmax=2.4)
        obs, *_ = env.step(np.array([baseline, command]))
        o = obs["agent_1"]
        _, d = geometry.project(float(o["pose_x"]), float(o["pose_y"]))
        speeds.append(float(o["linear_vel_x"]))
        offsets.append(abs(d))
        target_ticks += int(rival.telemetry["target"] is not None)
        action_counts[rival.action] = action_counts.get(rival.action, 0) + 1
        if tick % 10 == 0:
            frames.append({"t": (tick + 1) * DT, "speed": speeds[-1], "offset": d,
                           "action": rival.action, "target": rival.telemetry["target"],
                           "command": list(command)})
        if bool(o["collision"]):
            collisions.append((tick + 1) * DT)
            break
    return {"start": start, "seed": seed, "seconds": len(speeds) * DT,
            "minimum_speed": min(speeds), "maximum_line_offset": max(offsets),
            "stopped_seconds": sum(v < .1 for v in speeds) * DT,
            "collision_times": collisions, "target_ticks": target_ticks,
            "action_counts": action_counts, "opponent_model_sha256": rival.model_sha256,
            "passed_audit": not collisions and min(speeds) > .3 and max(offsets) < .4
                            and target_ticks == 0 and set(action_counts) == {"WAIT"},
            "frames": frames}


def paired_influence_trial(env, geometry, start, gap, seconds, seed):
    scene = Scenario(3, seed, start, gap, 0., 2.4, 2.4, seconds, False, "reactive")
    outputs = []
    for hide_ego in (False, True):
        # The simulator first raycasts static walls, then inserts other-car
        # returns. Disable only that insertion for agent_1 in this diagnostic.
        context = (patch.object(env.unwrapped.sim.agents[1], "ray_cast_agents", lambda scan: scan)
                   if hide_ego else nullcontext())
        with context:
            result, frames, _, _ = episode(scene, env=env, geometry=geometry)
        result["opponent_perception"] = "static_wall_lidar_only" if hide_ego else "full_lidar"
        result["diagnostic_only"] = True
        result["collision_geometry_unchanged"] = True
        result["opponent_detected_target_frames"] = sum(f.get("opponent_target") is not None for f in frames)
        outputs.append({"result": result, "frames": frames})
    full, hidden = [output["result"] for output in outputs]
    difference = None
    if full["passed_at"] is not None and hidden["passed_at"] is not None:
        difference = hidden["passed_at"] - full["passed_at"]
    return {"full_lidar": outputs[0], "wall_only": outputs[1],
            "pass_time_advantage_from_opponent_perception_seconds": difference,
            "interpretation": "Matched perception ablation; a difference measures dependence on rival observation, not a general guarantee of induced yielding."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20.)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--ablation-position", type=float, default=46.)
    parser.add_argument("--ablation-gap", type=float, default=1.4)
    parser.add_argument("--ablation-seconds", type=float, default=35.)
    parser.add_argument("--output", default="opponent_audit")
    args = parser.parse_args()
    if args.seconds <= 0 or args.starts < 1 or args.ablation_seconds <= 0:
        parser.error("Use positive durations and at least one start.")
    before = source_hashes()
    env = make_env(seed=713)
    geometry = TrackGeometry(env.unwrapped.track)
    report = {"source_sha256": before, "track_length": geometry.length,
              "llm_calls": 0, "network_access": "blocked", "no_threat_results": []}
    output = ARTIFACTS / f"{args.output}.json"
    def forbidden(*a, **k):
        raise RuntimeError("Network access is forbidden during the opponent audit")
    try:
        with patch.object(socket.socket, "connect", forbidden), patch.object(socket.socket, "connect_ex", forbidden), patch.object(socket, "create_connection", forbidden):
            for i in range(args.starts):
                result = no_threat_trial(env, geometry, i * geometry.length / args.starts, args.seconds, 713 + i * 10)
                report["no_threat_results"].append(result)
                print(json.dumps({k: v for k, v in result.items() if k != "frames"}), flush=True)
                output.write_text(json.dumps(report, indent=2) + "\n")
            if args.ablation:
                report["matched_influence_ablation"] = paired_influence_trial(
                    env, geometry, args.ablation_position, args.ablation_gap,
                    args.ablation_seconds, 741)
    finally:
        env.close()
    report["sources_unchanged_during_run"] = source_hashes() == before
    report["completed"] = True
    report["all_no_threat_audits_passed"] = all(r["passed_audit"] for r in report["no_threat_results"])
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}", flush=True)
    if not report["all_no_threat_audits_passed"] or not report["sources_unchanged_during_run"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
