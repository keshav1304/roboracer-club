"""Train the three local maneuver networks from this simulator's LiDAR scenes.

Example:
  ../gym-two-car-reactive/.venv/bin/python train_models.py --stage all \
      --samples 1000 --epochs 100 --llm-calls 45

LLM calls occur only here, with an explicit total ceiling. Geometric bootstrap
labels and genuine API reviews are recorded separately. Neither source is
called ground truth: scenario rollout results remain an independent test.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from bootstrap import ARTIFACTS, MODELS, DT
from llm_teacher import LLMTeacher, TeacherError, load_local_env
from maneuver_core import ACTIONS, FEATURE_NAMES, CandidatePlanner
from perception import LidarTracker, Target
from tiny_network import fit_network
from track_geometry import TrackGeometry, PathTracker
from world import make_env, reset_scenario, restore_base_map, body_polygon, make_stage2_opponent


HOLDOUT_SECTORS = (1, 5, 9)
VALIDATION_SECTORS = (2, 6, 10)
TRAIN_SECTORS = (0, 3, 4, 7, 8, 11)
DATASETS = ARTIFACTS / "datasets"


def sector_at(s, length):
    return min(11, int((float(s) % float(length)) / float(length) * 12))


def split_for_sector(sector):
    return "holdout" if sector in HOLDOUT_SECTORS else "validation" if sector in VALIDATION_SECTORS else "train"


def bootstrap_probabilities(candidates):
    """Soft preference for low estimated cost, with zero infeasible weight."""
    valid = np.array([bool(c["feasible"]) for c in candidates])
    costs = np.array([float(c["estimated_time"]) for c in candidates])
    if not valid.any():
        probabilities = np.zeros(len(candidates))
        probabilities[[c["action"] for c in candidates].index("WAIT")] = 1.
        return probabilities
    probabilities = np.zeros(len(candidates))
    probabilities[valid] = np.exp(-2.5 * (costs[valid] - costs[valid].min()))
    return probabilities / probabilities.sum()


def mirror_examples(X, y):
    """Reflect local training features and swap LEFT/RIGHT targets.

    These are synthetic symmetry augmentations of real snapshots, not extra
    simulator rollouts. Validation and evaluation scenes are never mirrored.
    Vehicle kinematics and the action definitions are left/right symmetric.
    """
    X, y = np.asarray(X, np.float32), np.asarray(y, np.float32)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or y.shape not in ((len(X), 3), (len(X), 4)):
        raise ValueError("Unexpected feature/label schema for symmetry augmentation.")
    mirrored = X.copy()
    indices = {name: i for i, name in enumerate(FEATURE_NAMES)}
    scan = [indices[f"lidar_sector_{i:02d}"] for i in range(18)]
    mirrored[:, scan] = X[:, scan[::-1]]
    for name in ("line_offset", "heading_error", "curvature_now", "curvature_1m", "curvature_3m", "target_offset"):
        mirrored[:, indices[name]] *= -1
    for name in ("feasible", "cost", "clearance", "steering", "speed"):
        left, right = indices[f"left_{name}"], indices[f"right_{name}"]
        mirrored[:, left], mirrored[:, right] = X[:, right], X[:, left]
    for action in ACTIONS:
        mirrored[:, indices[f"{action.lower()}_steering"]] *= -1
    order = [1, 0, 2] + ([3] if y.shape[1] == 4 else [])
    return mirrored, y[:, order].copy()


def _scenario_record(stage, seed, s, gap, lateral, ego_speed, opponent_speed, blocked, sector):
    return {"stage": int(stage), "seed": int(seed), "s": float(s), "gap": float(gap),
            "lateral": float(lateral), "ego_speed": float(ego_speed),
            "opponent_speed": float(opponent_speed), "blocked": bool(blocked),
            "sector": int(sector), "opponent": "stage2_neural" if stage == 3 else "line"}


def reactive_opponent(geometry, angles, speed_limit, observation, steering):
    """Use exactly the same saved stage-2 opponent in training and testing."""
    return make_stage2_opponent(geometry, angles, speed_limit, observation, steering)


def generate_snapshots(stage, count, seed, *, progress_every=100, gap_range=(.9, 5.)):
    """Actual scan snapshots after a short temporal-tracker warmup.

    Ground truth is used to arrange and reject intersecting initial scenes.
    Planner inputs come exclusively from the ego observation and scan tracker.
    Both the initial and final ego sector must match, preventing warmup motion
    across the declared training/validation/holdout boundaries.
    """
    env = make_env(seed=seed)
    g = TrackGeometry(env.unwrapped.track)
    angles = env.unwrapped.sim.agents[0].scan_angles
    rng = np.random.default_rng(seed)
    rows, rejected = [], Counter()
    started = time.monotonic()
    attempt = 0
    try:
        while len(rows) < count:
            attempt += 1
            if attempt > count * 25:
                raise RuntimeError("Too many invalid scene samples; inspect scenario geometry.")
            s = float(rng.uniform(0, g.length))
            sector = sector_at(s, g.length)
            if split_for_sector(sector) == "holdout":
                rejected["reserved_holdout_sector"] += 1
                continue
            gap, lateral = float(rng.uniform(*gap_range)), float(rng.uniform(-.2, .2))
            ego_lateral = float(rng.uniform(-.5, .5)) if rng.random() < .40 else 0.
            ego_speed = float(rng.uniform(1., 2.4))
            opponent_speed = 0. if stage == 1 else float(rng.uniform(1.6, 2.2)) if stage == 2 else 2.4
            if stage == 3 and rng.random() < .70:
                ego_speed = opponent_speed
            blocked = bool(stage == 1 and rng.random() < .15)
            scene_seed = int(seed * 100000 + attempt)
            scene = _scenario_record(stage, scene_seed, s, gap, lateral, ego_speed, opponent_speed, blocked, sector)
            scene["ego_lateral"] = ego_lateral
            try:
                obs, meta = reset_scenario(env, g, s, gap, opponent_speed, ego_speed=ego_speed,
                                           lateral=lateral, blocked=blocked)
            except ValueError:
                rejected["invalid_initial_geometry"] += 1
                continue
            if ego_lateral:
                # Off-policy recovery states use the same world and real scans;
                # only scenario initialization sees the ground-truth poses.
                xy, yaw, _ = g.sample(s, ego_lateral)
                if float(g.body_clearance(xy, yaw)) < .04:
                    rejected["invalid_lateral_spawn"] += 1
                    continue
                poses = np.array(meta["initial_poses"], float)
                poses[0] = np.r_[xy, yaw]
                obs, _ = env.reset(options={"poses": poses})
                for agent, speed in zip(env.unwrapped.sim.agents, (ego_speed, opponent_speed)):
                    agent.state[3] = speed
                obs, *_ = env.step(np.asarray([[0., ego_speed], [0., opponent_speed]]))
                obstacle_body = meta["obstacle_polygon"] if blocked else body_polygon(obs["agent_1"])
                if body_polygon(obs["agent_0"]).distance(obstacle_body) < .05:
                    rejected["invalid_lateral_spawn"] += 1
                    continue
            sensor = LidarTracker(g, angles)
            ego_tracker, other_tracker = PathTracker(g), PathTracker(g)
            if stage == 3:
                other_tracker = reactive_opponent(g, angles, opponent_speed, obs["agent_1"],
                                                   env.unwrapped.sim.agents[1].state[2])
                scene["opponent_model_sha256"] = other_tracker.model_sha256
            target = sensor.update(obs["agent_0"], dt=DT)
            warm_ticks = int(rng.integers(15, 31))
            collision = False
            for _ in range(warm_ticks):
                ego_action = ego_tracker.control(obs["agent_0"], vmax=ego_speed)
                if stage == 1:
                    other_action = (0., 0.)
                elif stage == 3:
                    other_action = other_tracker.plan(obs["agent_1"])
                else:
                    other_action = other_tracker.control(obs["agent_1"], vmax=opponent_speed)
                obs, *_ = env.step(np.asarray([ego_action, other_action]))
                target = sensor.update(obs["agent_0"], dt=DT)
                if any(bool(o["collision"]) for o in obs.values()):
                    collision = True
                    break
            final_s, _ = g.project(float(obs["agent_0"]["pose_x"]), float(obs["agent_0"]["pose_y"]))
            if collision or sector_at(final_s, g.length) != sector:
                rejected["warmup_collision_or_sector_crossing"] += 1
                continue
            ego_body = body_polygon(obs["agent_0"])
            obstacle_body = meta["obstacle_polygon"] if blocked else body_polygon(obs["agent_1"])
            if ego_body.distance(obstacle_body) < .04:
                rejected["warmup_near_contact"] += 1
                continue
            if stage == 1 and target is not None:
                target = replace(target, speed=0., lateral_speed=0.)
            planner = CandidatePlanner(g, vmax=2.4, stage=stage)
            candidates = planner.candidates(obs["agent_0"], target)
            features = planner.features(obs["agent_0"], target, candidates)
            nclasses = 4 if stage == 3 else 3
            records = [c.record() for c in candidates[:nclasses]]
            probabilities = bootstrap_probabilities(records)
            row = {"id": f"stage{stage}-snapshot-{seed}-{attempt}", "stage": stage,
                   "features": features.tolist(), "candidates": records, "max_speed": 2.4,
                   "state": None if target is None else asdict(target),
                   "scenario": scene, "final_s": float(final_s), "split": split_for_sector(sector),
                   "warmup_seconds": warm_ticks * DT, "label_source": "geometric_bootstrap",
                   "action": ACTIONS[int(np.argmax(probabilities))],
                   "probabilities": probabilities.tolist(), "rollout_verified": False,
                   "generation": "simulated_lidar_snapshot"}
            rows.append(row)
            if progress_every and len(rows) % progress_every == 0:
                print(json.dumps({"stage": stage, "snapshots": len(rows), "requested": count,
                                  "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    finally:
        restore_base_map(env)
        env.close()
    return rows, {"attempts": attempt, "rejections": dict(rejected), "track_length": g.length,
                  "opponent_controller_initialized_to_physical_state": stage == 3,
                  "environment_seed": seed, "initial_gap_range_m": list(gap_range),
                  "generation_seconds": time.monotonic() - started}


def append_rollout_samples(rows, paths, stage, track_length):
    """Append explicitly located rollout rows; ambiguous split provenance is rejected.

    Accepted rows need scenario.s (or initial_s/s), plus stage and 50 features.
    Existing .samples.json files can obtain scenario metadata from their sibling
    .json result. A holdout or validation rollout cannot enter the training set.
    """
    added, rejected = 0, Counter()
    seen = {r["id"] for r in rows}
    for source in paths:
        path = Path(source)
        if path.suffix == ".jsonl":
            incoming = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            data = json.loads(path.read_text())
            incoming = data.get("samples", []) if isinstance(data, dict) else data
        sibling = path.with_name(path.name.replace(".samples.json", ".json"))
        source_scenario = {}
        if sibling != path and sibling.exists():
            source_scenario = json.loads(sibling.read_text()).get("scenario", {})
        for i, sample in enumerate(incoming):
            if int(sample.get("stage", source_scenario.get("stage", -1))) != stage:
                continue
            scene = sample.get("scenario", source_scenario)
            # Initial episode position does not locate later frames: a long
            # rollout can cross into a reserved evaluation sector.
            s = sample.get("ego_s", sample.get("s"))
            if s is None:
                rejected["missing_observation_location"] += 1
                continue
            split = split_for_sector(sector_at(s, track_length))
            initial_s = scene.get("s")
            if split != "train" or (initial_s is not None and split_for_sector(sector_at(initial_s, track_length)) != "train"):
                rejected["not_training_sector"] += 1
                continue
            features = np.asarray(sample.get("features", []), float)
            nclasses = 4 if stage == 3 else 3
            records = sample.get("candidates", [])[:nclasses]
            if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all() or len(records) != nclasses:
                rejected["invalid_feature_or_candidate_schema"] += 1
                continue
            uid = f"rollout-{path.name}-{i}"
            if uid in seen:
                continue
            probabilities = bootstrap_probabilities(records)
            rows.append({**sample, "id": uid, "scenario": scene, "split": "train",
                         "stage": stage, "candidates": records,
                         "max_speed": float(sample.get("max_speed", scene.get("ego_speed", 2.4))),
                         "label_source": "geometric_bootstrap", "action": ACTIONS[int(probabilities.argmax())],
                         "probabilities": probabilities.tolist(), "rollout_verified": False,
                         "generation": "recorded_rollout_snapshot", "source_file": str(path)})
            seen.add(uid)
            added += 1
    return {"added": added, "rejections": dict(rejected)}


def refresh_geometry_labels(rows, stage):
    """Recompute planner-derived fields from preserved measured snapshot state.

    No scan is regenerated. The original 18 LiDAR summary features are kept
    verbatim. Localized position/heading are reconstructed from saved Frenet
    s/d and heading error; full saved scan-tracker state supplies the target.
    Any earlier LLM advice is retained as superseded, not falsely revalidated.
    """
    env = make_env(seed=0)
    g = TrackGeometry(env.unwrapped.track)
    planner = CandidatePlanner(g, vmax=2.4, stage=stage)
    refreshed, skipped, retired_reviews = 0, 0, 0
    feature_index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    try:
        for row in rows:
            if row.get("generation") != "simulated_lidar_snapshot" or "final_s" not in row:
                skipped += 1
                continue
            original = np.array(row["features"], np.float32)
            xy, yaw, _ = g.sample(float(row["final_s"]), float(original[feature_index["line_offset"]]))
            # CandidatePlanner itself never reads scan. The placeholder is only
            # used to let features() fill its geometry fields; measured sectors
            # replace that temporary prefix immediately below.
            obs = {"pose_x": float(xy[0]), "pose_y": float(xy[1]),
                   "pose_theta": float(yaw + original[feature_index["heading_error"]]),
                   "linear_vel_x": float(original[feature_index["ego_speed"]] * 3.),
                   "scan": np.zeros(1080)}
            target = None if row["state"] is None else Target(**row["state"])
            candidates = planner.candidates(obs, target)
            features = planner.features(obs, target, candidates)
            features[:18] = original[:18]
            records = [c.record() for c in candidates[:4 if stage == 3 else 3]]
            probabilities = bootstrap_probabilities(records)
            if row.get("llm_review") is not None:
                row["superseded_llm_review"] = row.pop("llm_review")
                retired_reviews += 1
            row.update(features=features.tolist(), candidates=records,
                       action=ACTIONS[int(probabilities.argmax())], probabilities=probabilities.tolist(),
                       label_source="geometric_bootstrap", geometry_refreshed=True)
            refreshed += 1
    finally:
        env.close()
    directory = Path(__file__).parent
    return {"refreshed": refreshed, "skipped_non_snapshot_rows": skipped,
            "superseded_llm_reviews": retired_reviews,
            "lidar_features_preserved": True, "raw_scans_regenerated": False,
            "pose_reconstructed_from_measured_frenet_features": True,
            "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                              for name in ("maneuver_core.py", "track_geometry.py", "perception.py")}}


def review_with_llm(rows, teacher, budget, seed, require_llm=False):
    """Review diverse training rows only; leave validation labels independent."""
    rng = np.random.default_rng(seed)
    queues = {}
    for i, row in enumerate(rows):
        if row["split"] != "train" or row.get("label_source") == "openai_responses":
            continue
        p = np.array(row["probabilities"])
        uncertainty = float(-np.sum(p * np.log(np.clip(p, 1e-8, 1))))
        queues.setdefault(row["action"], []).append((uncertainty + float(rng.uniform(0, .3)), i))
    for queue in queues.values():
        queue.sort(reverse=True)
    choices = []
    while len(choices) < budget and any(queues.values()):
        for action in ACTIONS:
            queue = queues.get(action)
            if queue and len(choices) < budget:
                choices.append(queue.pop(0)[1])
    report = {"requested_reviews": len(choices), "accepted": 0, "rejected_or_failed": 0,
              "cache_hits": 0, "disagreements_with_bootstrap": 0, "errors": []}
    for i in choices:
        row = rows[i]
        scenario = {"id": row["id"], "stage": row["stage"], "state": row.get("state"),
                    "max_speed": row["max_speed"], "candidates": row["candidates"],
                    "motion_model": "constant tracked Frenet velocity with uncertainty margin",
                    "reference": "locally curvature-reduced racing line",
                    "objective": "earliest safe completed pass; preserve body clearance; do not assume yielding",
                    "observation_features": dict(zip(FEATURE_NAMES, row["features"]))}
        try:
            response = teacher.label(scenario)
        except TeacherError as exc:
            report["rejected_or_failed"] += 1
            report["errors"].append({"id": row["id"], "error": str(exc)})
            if require_llm:
                raise
            # Stop on connectivity/configuration/quota errors; do not spend the
            # remaining budget repeating an unavailable service.
            if any(word in str(exc).lower() for word in ("network", "not configured", "http", "limit")):
                break
            continue
        report["accepted"] += 1
        report["cache_hits"] += int(response["cache_hit"])
        report["disagreements_with_bootstrap"] += int(response["action"] != row["action"])
        row["bootstrap_action"] = row["action"]
        row["bootstrap_probabilities"] = row["probabilities"]
        confidence = float(np.clip(response["confidence"], .55, .98))
        label = np.zeros(len(row["candidates"]))
        chosen = [c["action"] for c in row["candidates"]].index(response["action"])
        remainder = np.array(row["probabilities"])
        remainder[chosen] = 0.
        if remainder.sum() > 1e-8:
            label = (1 - confidence) * remainder / remainder.sum()
            label[chosen] = confidence
        else:
            label[chosen] = 1.
        row.update(action=response["action"], probabilities=label.tolist(),
                   label_source="openai_responses", llm_review=response)
        print(json.dumps({"stage": row["stage"], "llm_reviews_accepted": report["accepted"],
                          "api_calls_used": teacher.calls, "action": response["action"]}), flush=True)
    if require_llm and report["accepted"] == 0:
        raise TeacherError("No actual LLM review accepted; --require-llm prevents local-only training.")
    return report


def save_dataset(rows, stage, generation):
    DATASETS.mkdir(parents=True, exist_ok=True)
    truth_path = DATASETS / f"stage{stage}.truth.jsonl"
    truth_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    sources = Counter(row["label_source"] for row in rows)
    splits = Counter(row["split"] for row in rows)
    summary = {"stage": stage, "rows": len(rows), "features": FEATURE_NAMES,
               "classes": ACTIONS[:4 if stage == 3 else 3], "sources": dict(sources),
               "split_counts": dict(splits), "holdout_sectors": HOLDOUT_SECTORS,
               "validation_sectors": VALIDATION_SECTORS, "train_sectors": TRAIN_SECTORS,
               "action_counts": dict(Counter(row["action"] for row in rows)),
               "no_target_count": sum(row.get("state") is None for row in rows),
               "full_barrier_count": sum(bool(row.get("scenario", {}).get("blocked")) for row in rows),
               "equal_initial_speed_count": sum(abs(float(row.get("scenario", {}).get("ego_speed", -10.))
                                                     - float(row.get("scenario", {}).get("opponent_speed", -20.))) < 1e-6
                                                for row in rows),
               "labels_are_rollout_ground_truth": False, "generation": generation}
    for split in ("train", "validation"):
        subset = [row for row in rows if row["split"] == split]
        if not subset:
            raise ValueError(f"No {split} rows; increase the sample budget.")
        np.savez_compressed(DATASETS / f"stage{stage}.{split}.npz",
                            X=np.array([r["features"] for r in subset], dtype=np.float32),
                            y=np.array([r["probabilities"] for r in subset], dtype=np.float32),
                            ids=np.array([r["id"] for r in subset]),
                            sources=np.array([r["label_source"] for r in subset]),
                            feature_names=np.array(FEATURE_NAMES),
                            classes=np.array(summary["classes"]))
    (DATASETS / f"stage{stage}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def train_stage(stage, args, teacher=None, llm_budget=0):
    truth_path = DATASETS / f"stage{stage}.truth.jsonl"
    if args.reuse_data:
        if not truth_path.exists():
            raise FileNotFoundError(f"No saved dataset at {truth_path}")
        rows = [json.loads(line) for line in truth_path.read_text().splitlines() if line.strip()]
        generation = {"reused": True}
        existing = DATASETS / f"stage{stage}.summary.json"
        if existing.exists():
            generation["original"] = json.loads(existing.read_text()).get("generation")
        for row in rows:
            if row["split"] == "holdout" or len(row["features"]) != len(FEATURE_NAMES):
                raise ValueError("Saved dataset contains reserved holdout rows or wrong feature schema.")
    else:
        rows, generation = generate_snapshots(stage, args.samples, args.seed + stage * 100)
    if args.append_snapshots:
        additional, extra_generation = generate_snapshots(stage, args.append_snapshots,
                                                          args.seed + stage * 100 + 10000)
        ids = {row["id"] for row in rows}
        rows.extend(row for row in additional if row["id"] not in ids)
        generation["additional_simulated_snapshots"] = extra_generation
    if args.append_samples:
        length = generation.get("track_length")
        if length is None:
            env = make_env(seed=args.seed)
            try:
                length = TrackGeometry(env.unwrapped.track).length
            finally:
                env.close()
        generation["appended_rollouts"] = append_rollout_samples(rows, args.append_samples, stage, length)
    if args.refresh_geometry:
        generation["geometry_refresh"] = refresh_geometry_labels(rows, stage)
    llm_report = None
    if llm_budget and teacher is not None:
        llm_report = review_with_llm(rows, teacher, llm_budget, args.seed + stage, args.require_llm)
    elif args.require_llm and not any(r.get("label_source") == "openai_responses" for r in rows):
        raise TeacherError("--require-llm needs actual API labels or a previously reviewed dataset.")
    generation["llm_review"] = llm_report
    summary = save_dataset(rows, stage, generation)
    train = np.load(DATASETS / f"stage{stage}.train.npz", allow_pickle=False)
    validation = np.load(DATASETS / f"stage{stage}.validation.npz", allow_pickle=False)
    prior = MODELS / f"stage{stage-1}.npz"
    warmstart = prior if stage > 1 and prior.exists() and not args.no_transfer else None
    train_X, train_y = train["X"], train["y"]
    original_train_rows = len(train_X)
    if not args.no_symmetry:
        mirrored_X, mirrored_y = mirror_examples(train_X, train_y)
        train_X, train_y = np.vstack([train_X, mirrored_X]), np.vstack([train_y, mirrored_y])
    result = fit_network(train_X, train_y, MODELS / f"stage{stage}.npz",
                         validation=(validation["X"], validation["y"]),
                         classes=ACTIONS[:4 if stage == 3 else 3], feature_names=FEATURE_NAMES,
                         epochs=args.epochs, seed=args.seed + stage, warmstart=warmstart,
                         metadata={"stage": stage, "map": "autodrive-default",
                                   "label_sources": summary["sources"], "dataset": str(truth_path),
                                   "holdout_sectors": list(HOLDOUT_SECTORS),
                                   "mirror_training_augmentation": not args.no_symmetry,
                                   "original_training_rows": original_train_rows,
                                   "not_rollout_ground_truth": True})
    report = {"stage": stage, "dataset": summary, "train": result["train"],
              "validation": result["validation"], "selected_epoch": result["selected_epoch"],
              "warmstart": result["warmstart"], "checkpoints": result["checkpoints"],
              "mirror_training_augmentation": not args.no_symmetry,
              "original_training_rows": original_train_rows,
              "llm_review": llm_report, "api_calls_used_total": 0 if teacher is None else teacher.calls}
    (ARTIFACTS / f"stage{stage}.training.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stage": stage, "training_complete": True, "rows": len(rows),
                      "sources": summary["sources"], "validation_accuracy": result["validation"]["accuracy"],
                      "api_calls_used_total": report["api_calls_used_total"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--llm-calls", type=int, default=0,
                        help="TOTAL API request ceiling across all selected stages; defaults to offline only")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-max-cost", type=float, default=1., help="Conservative USD guard, cumulative for this review log")
    parser.add_argument("--llm-log", type=Path, default=ARTIFACTS / "llm_training_reviews.jsonl")
    parser.add_argument("--require-llm", action="store_true")
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--refresh-geometry", action="store_true",
                        help="Recompute candidate features/labels from stored measured snapshot state, preserving LiDAR summaries")
    parser.add_argument("--no-transfer", action="store_true")
    parser.add_argument("--no-symmetry", action="store_true", help="Disable synthetic left/right training augmentation")
    parser.add_argument("--append-samples", type=Path, action="append", default=[])
    parser.add_argument("--append-snapshots", type=int, default=0,
                        help="Generate and append this many additional actual scan snapshots using a separate seed")
    args = parser.parse_args()
    if args.samples < 20 or args.epochs < 1 or args.llm_calls < 0 or args.append_snapshots < 0:
        parser.error("Use at least 20 samples, positive epochs, and a nonnegative API call limit.")
    load_local_env()
    stages = (1, 2, 3) if args.stage == "all" else (int(args.stage),)
    teacher = None
    if args.llm_calls:
        teacher = LLMTeacher(args.llm_model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
                             args.llm_log, max_calls=args.llm_calls, max_cost_usd=args.llm_max_cost)
    reports = []
    for position, stage in enumerate(stages):
        budget = args.llm_calls // len(stages) + int(position < args.llm_calls % len(stages))
        reports.append(train_stage(stage, args, teacher, budget))
    # A one-stage refresh must not make previously trained stages disappear
    # from the project overview. Per-stage files retain their own provenance.
    available = []
    for stage in (1, 2, 3):
        path = ARTIFACTS / f"stage{stage}.training.json"
        if path.exists():
            available.append(json.loads(path.read_text()))
    (ARTIFACTS / "training_summary.json").write_text(json.dumps({"stages": available,
        "stages_updated_this_run": stages,
        "api_request_ceiling": args.llm_calls, "api_calls_attempted": 0 if teacher is None else teacher.calls,
        "api_estimated_usage_cost_usd": 0 if teacher is None else teacher.usage_cost_usd,
        "api_reserved_cost_usd": 0 if teacher is None else teacher.reserved_cost_usd,
        "api_cost_ceiling_usd": args.llm_max_cost,
        "holdout_sectors": HOLDOUT_SECTORS, "final_rollout_validation_performed_here": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
