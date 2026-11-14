"""Measure the deployed moving-car controllers locally, without rival comparisons."""
import json
import platform
import time
from pathlib import Path
import numpy as np

from benchmark import offline_only, record_streams
from bootstrap import ARTIFACTS, MODELS
from maneuver_core import NeuralDriver
from rollout_training import file_hash, sources
from track_geometry import TrackGeometry
from world import make_env, timing_summary


def main():
    env = make_env(seed=90210)
    try:
        geometry = TrackGeometry(env.unwrapped.track)
        angles = env.unwrapped.sim.agents[0].scan_angles.copy()
        streams, scenes = record_streams(geometry, env, measured_per_scene=600, warmup=100)
    finally:
        env.close()
    results = {}
    with offline_only():
        for stage in (2, 3):
            durations, network, heads, deployed_network = [], [], [], []
            for stream in streams:
                driver = NeuralDriver(geometry, angles, stage=stage, model_path=MODELS/f"stage{stage}.npz")
                # Inputs were physically recorded at 100 Hz. Evaluate every
                # fifth sample to match the qualified 20 Hz decision cadence.
                for i in range(0, len(stream), 5):
                    start = time.perf_counter_ns()
                    command = driver.plan(stream[i], dt=.05)
                    elapsed = (time.perf_counter_ns()-start)/1e6
                    if not np.isfinite(command).all():
                        raise RuntimeError("Nonfinite control output")
                    if i < 100:
                        continue
                    durations.append(elapsed)
                    deployed_network.append(driver.network_times[-1])
                    features = driver.telemetry["features"]
                    start = time.perf_counter_ns()
                    driver.model.proba(features)
                    network.append((time.perf_counter_ns()-start)/1e6)
                    if driver.value_model is not None:
                        count = 4 if stage == 3 else 3
                        value_input = np.column_stack([np.repeat(features[None,:],count,axis=0),np.eye(4)[:count]])
                        start = time.perf_counter_ns()
                        driver.value_model.predict_proba(value_input)
                        heads.append((time.perf_counter_ns()-start)/1e6)
            results[str(stage)] = dict(full_controller=timing_summary(durations),
                deployed_network=timing_summary(deployed_network),
                maneuver_network=timing_summary(network), outcome_network=timing_summary(heads),
                model_sha256=driver.model_sha256, value_model_sha256=driver.value_model_sha256)
    report = dict(results=results, scenes=scenes, reference_digest=geometry.reference_digest,
        source_sha256=sources(), benchmark_sha256=file_hash(__file__),
        hardware=dict(platform=platform.platform(),python=platform.python_version(),numpy=np.__version__),
        control_period_ms=50, physics_period_ms=10, network_access="blocked",
        scope="Single-process warmed compute on identical recorded observations; excludes physics, rendering and model loading. Sensor scheduling can add up to 50 ms beyond computation; these are not physical reaction-time guarantees.")
    path=ARTIFACTS/"benchmark_v3.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
