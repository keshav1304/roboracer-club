"""Comparable local decision latency on an identical recorded sensor stream.

This is an open-loop compute benchmark, not a closed-loop driving score. The
complete neural pipeline includes scan processing, motion tracking, candidate
generation, inference, feasibility filtering and actuator commands. Pure-network
inference is reported separately and excludes those necessary operations.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import platform
from pathlib import Path
import socket
import time
from unittest.mock import patch

import numpy as np

from bootstrap import ARTIFACTS, MODELS, DT
from world import make_env, reset_scenario, restore_base_map, body_polygon, timing_summary
from track_geometry import TrackGeometry, PathTracker
from maneuver_core import NeuralDriver, FEATURE_NAMES
from reactive import FollowTheGap, FTGParams
from wall_follower import WallFollower, WallParams
from gap_follower import GapFollower
from reactive_safety import ReactiveDriver


def deny_network(*args, **kwargs):
    raise AssertionError("Network access is forbidden in evaluation.")


def offline_only():
    """Fail visibly if runtime code attempts an outbound connection."""
    stack = ExitStack()
    stack.enter_context(patch.object(socket.socket, "connect", deny_network))
    stack.enter_context(patch.object(socket.socket, "connect_ex", deny_network))
    stack.enter_context(patch.object(socket, "create_connection", deny_network))
    return stack


def record_streams(geometry, env, measured_per_scene=500, warmup=100):
    """Two contiguous scenes with actual LiDAR and motion, without NN calls.

    Both vehicles use the same line tracker while recording. This fixed trace
    makes every timed controller receive identical data; its actions do not
    alter the replay. Collision checks protect the benchmark's scene quality.
    """
    streams, summaries = [], []
    count = measured_per_scene + warmup
    for scene, position in enumerate((0., geometry.length * .45)):
        obs, _ = reset_scenario(env, geometry, position, 4., 1.8, ego_speed=1.8)
        controllers = [PathTracker(geometry), PathTracker(geometry)]
        sequence = []
        minimum_gap = float("inf")
        for step in range(count):
            o = obs["agent_0"]
            sequence.append({key: np.asarray(value).copy() if key == "scan" else float(value)
                             for key, value in o.items()
                             if key in ("scan", "pose_x", "pose_y", "pose_theta", "linear_vel_x")})
            commands = [controllers[i].control(obs[f"agent_{i}"], vmax=2. if i == 0 else 1.8)
                        for i in (0, 1)]
            obs, *_ = env.step(np.asarray(commands))
            minimum_gap = min(minimum_gap, body_polygon(obs["agent_0"]).distance(body_polygon(obs["agent_1"])))
            if any(bool(obs[f"agent_{i}"]["collision"]) for i in (0, 1)) or minimum_gap < .001:
                raise RuntimeError(f"Benchmark recording scene {scene} collided; choose a valid stream.")
        streams.append(sequence)
        summaries.append({"scene": scene, "reference_position": position, "samples": count,
                          "warmup_samples": warmup, "minimum_body_gap": minimum_gap,
                          "simulated_seconds": count * DT})
    return streams, summaries


def _timed_sequence(factory, streams, warmup, *, neural=False, split_preview=False):
    durations, preview, cached, feature_rows = [], [], [], []
    for sequence in streams:
        controller = factory()
        for index, obs in enumerate(sequence):
            is_preview = (getattr(controller.controller, "tick", 0) % 10 == 0) if split_preview else True
            started = time.perf_counter_ns()
            if neural:
                command = controller.plan(obs)
            else:
                command = controller.plan(obs["scan"], obs["linear_vel_x"])
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if not np.all(np.isfinite(command)):
                raise AssertionError("Controller produced a nonfinite command during benchmark.")
            if index >= warmup:
                durations.append(elapsed)
                (preview if is_preview else cached).append(elapsed)
                if neural:
                    feature_rows.append(np.asarray(controller.telemetry["features"]).copy())
    result = {"full_controller": timing_summary(durations)}
    if split_preview:
        result.update(preview_replan=timing_summary(preview), cached_tick=timing_summary(cached))
    return result, feature_rows


def run_benchmark(stages=(1, 2, 3), measured_per_scene=500, warmup=100, output=None):
    if measured_per_scene < 50 or warmup < 10:
        raise ValueError("Use at least 50 measured and 10 warmup ticks per scene.")
    paths = {int(stage): MODELS / f"stage{int(stage)}.npz" for stage in stages}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Train the requested models before timing them: " + ", ".join(missing))
    directory = Path(__file__).parent
    source_hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                     for name in ("maneuver_core.py", "perception.py", "track_geometry.py", "benchmark.py")}
    env = make_env(seed=90210)
    geometry = TrackGeometry(env.unwrapped.track)
    angles = env.unwrapped.sim.agents[0].scan_angles.copy()
    try:
        streams, scenes = record_streams(geometry, env, measured_per_scene, warmup)
    finally:
        restore_base_map(env)
        env.close()

    results = {}
    with offline_only():
        for stage, path in paths.items():
            factory = lambda stage=stage, path=path: NeuralDriver(geometry, angles, stage=stage,
                                                                 vmax=2.4, model_path=path)
            values, features = _timed_sequence(factory, streams, warmup, neural=True)
            model = factory().model
            for features_one in features[:100]:
                model.proba(features_one)
            pure = []
            for features_one in features:
                start = time.perf_counter_ns()
                model.proba(features_one)
                pure.append((time.perf_counter_ns() - start) / 1e6)
            values["network_only"] = timing_summary(pure)
            values["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            results[f"stage_{stage}"] = values

        factories = {
            "plain_follow_the_gap": (lambda: FollowTheGap(angles, FTGParams(v_max=2.4)), False),
            "wall_follower_with_preview_and_safety": (
                lambda: ReactiveDriver(WallFollower(angles, WallParams(v_max=2.4)), angles, 2.4,
                                       corridor_margin=.04), True),
            "gap_follower_with_preview_and_safety": (
                lambda: ReactiveDriver(GapFollower(angles, v_max=2.4), angles, 2.4,
                                       steering_rate=3., acceleration=2., lateral_acceleration=2.5,
                                       stop_distance=.45, corridor_margin=.04), True),
        }
        for name, (factory, split) in factories.items():
            results[name], _ = _timed_sequence(factory, streams, warmup, split_preview=split)

    try:
        from threadpoolctl import threadpool_info
        threadpools = threadpool_info()
    except ImportError:
        threadpools = None
    report = {
        "benchmark": "Identical contiguous observation replay, single process, CPU wall-clock",
        "source_sha256": source_hashes,
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "processor": platform.processor(), "python": platform.python_version(),
                     "numpy": np.__version__, "threadpools": threadpools},
        "scenes": scenes, "measured_samples_per_controller": measured_per_scene * len(streams),
        "warmup_samples_per_scene": warmup, "model_loading_timed": False,
        "rendering_timed": False, "physics_timed": False,
        "full_neural_includes": ["scan processing", "target tracking", "candidate trajectories",
                                  "feature extraction", "MLP", "safety filtering", "control output", "telemetry"],
        "network_only_excludes": ["all perception", "features", "planning", "control", "safety filtering"],
        "test_network_access": "Socket connections patched to raise; zero successful outbound requests",
        "all_controller_speed_caps": 2.4, "results": results,
        "limits": ["Open-loop timing replay does not measure driving success.",
                   "These measurements apply to this computer and these scenes, not an onboard guarantee.",
                   "Hosted Jev is not called; its network round-trip latency is not a direct local benchmark.",
                   "Preview controllers run a full preview every tenth tick; both tick types are reported."]}
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", type=int, choices=(1, 2, 3), nargs="+", default=[1, 2, 3])
    parser.add_argument("--samples-per-scene", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ARTIFACTS / "benchmark.json")
    args = parser.parse_args()
    report = run_benchmark(args.stages, args.samples_per_scene, args.warmup, args.output)
    for name, value in report["results"].items():
        full = value["full_controller"]
        print(f"{name}: full p50={full['p50_ms']:.3f} ms, p95={full['p95_ms']:.3f} ms, p99={full['p99_ms']:.3f} ms")
        if "network_only" in value:
            pure = value["network_only"]
            print(f"  MLP only p50={pure['p50_ms']:.3f} ms, p95={pure['p95_ms']:.3f} ms")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
