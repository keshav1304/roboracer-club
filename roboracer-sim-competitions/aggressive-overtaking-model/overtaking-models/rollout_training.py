"""Train on completed overtaking episodes, with sealed scenario partitions.

Default: 36 random training encounters x three controlled preference policies
= 108 complete training rollouts per stage. Ten different validation encounters
have their own three policies; 30 further test encounters are never used by fit.
Failures train an empirical PASS/FAIL value head. Successful trajectories teach
the imitation network, optionally supplemented by explicitly marked LLM advice;
failure does not invent a winning action or a successful outcome label.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np

from bootstrap import ARTIFACTS, MODELS
from maneuver_core import ACTIONS, FEATURE_NAMES
from tiny_network import TinyNetwork, fit_network

SECTORS = {"train": (0, 3, 4, 7, 8, 11), "validation": (2, 6, 10), "test": (1, 5, 9)}
STAGE3_GAP_RANGES_M = {"close": (.70,1.05), "approach": (1.3,2.0)}
VALUE_FEATURE_NAMES = list(FEATURE_NAMES) + [f"action_{a.lower()}" for a in ACTIONS]
SOURCE_FILES = ("maneuver_core.py", "perception.py", "track_geometry.py", "world.py", "run_experiment.py")
DEFAULT_ROOT = ARTIFACTS / "v3_rollouts"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return {name: file_hash(Path(__file__).with_name(name)) for name in SOURCE_FILES}


def sector_at(s, length):
    return min(11, int((float(s) % length) / length * 12))


def policy_variants(stage):
    if stage == 2:
        return (("neutral", [0., 0., 0.]), ("left", [-4., 0., 0.]), ("right", [0., -4., 0.]))
    return (("inside", [0., 0., 0., -4.]), ("left", [-4., 0., 0., 0.]), ("right", [0., -4., 0., 0.]))


def validate_rollout_timing(seconds, control_dt, warmup_seconds):
    """A short physical sensor prelude must end on a full control period."""
    if not all(np.isfinite(v) for v in (seconds,control_dt,warmup_seconds)):
        raise ValueError("Rollout timing must be finite")
    if (not .01 <= control_dt <= .1
            or abs(round(control_dt/.01)*.01-control_dt)>1e-8):
        raise ValueError("Control period must be a 0.01-second multiple in [0.01, 0.10]")
    if (seconds<=0 or not 0<=warmup_seconds<=.4 or warmup_seconds>=seconds
            or abs(round(warmup_seconds/control_dt)*control_dt-warmup_seconds)>1e-8):
        raise ValueError("Warmup must be between 0 and 0.4 seconds, a whole control-period multiple, and below the episode deadline")


def make_manifest(length, reference_digest, *, seed=20261119, train_cases=36,
                  validation_cases=10, test_cases=30, seconds=18., control_dt=.05,warmup_seconds=.4):
    """Create all split scenarios before collection or model selection."""
    if min(train_cases, validation_cases, test_cases) < 1 or length <= 0 or seconds <= 0:
        raise ValueError("Positive scene counts, length and duration are required")
    validate_rollout_timing(seconds,control_dt,warmup_seconds)
    rng = np.random.default_rng(seed)
    used_seeds, jobs = set(), []
    for stage in (2, 3):
        for split, count in (("train", train_cases), ("validation", validation_cases), ("test", test_cases)):
            for index in range(count):
                sector = SECTORS[split][index % len(SECTORS[split])]
                # Cycle the scenario stratum *within* each sector. Using just
                # index % 3 makes speed/gap perfectly confounded with sector
                # when the sector count is three or six.
                scenario_group = (index // len(SECTORS[split]) + index % len(SECTORS[split])) % 3
                s = float((sector + rng.uniform(.12, .85)) * length / 12)
                physical_seed = int(rng.integers(100_000, 2**30))
                while physical_seed in used_seeds or physical_seed + 1 in used_seeds:
                    physical_seed = int(rng.integers(100_000, 2**30))
                used_seeds.update((physical_seed, physical_seed + 1))
                gap = float(rng.uniform(1.15, 2.4) if stage == 2 else
                            rng.uniform(*STAGE3_GAP_RANGES_M["close" if scenario_group else "approach"]))
                opponent_speed = float(rng.uniform(*((1.5, 1.75), (1.85, 2.05), (2.1, 2.25))[scenario_group])) if stage == 2 else 2.4
                scenario = dict(stage=stage, seed=physical_seed, s=s, gap=gap, lateral=0.,
                                ego_speed=2.4, opponent_speed=opponent_speed, seconds=seconds,
                                blocked=False, opponent="line" if stage == 2 else "reactive",
                                warmup_seconds=warmup_seconds)
                case_id = f"stage{stage}-{split}-{index:03d}"
                variants = (("learned", None),) if split == "test" else policy_variants(stage)
                for variant, bias in variants:
                    jobs.append(dict(trial_id=f"{case_id}-{variant}", case_id=case_id,
                                     stage=stage, split=split, start_sector=sector, scenario=scenario,
                                     scenario_digest=digest(scenario), scenario_group=scenario_group,
                                     variant=variant, policy_bias=bias))
    body = dict(format_version=2, seed=seed, track_length_m=length, reference_digest=reference_digest,
                seconds=seconds, control_dt=control_dt,warmup_seconds=warmup_seconds,
                warmup_note="Both cars physically drive the shared reference while their own real LiDAR histories initialize motion estimates. Prelude time and contacts remain part of the scored episode.",
                train_cases_per_stage=train_cases,
                training_rollouts_per_stage=train_cases * 3,
                validation_cases_per_stage=validation_cases, test_cases_per_stage=test_cases,
                split_sectors={k:list(v) for k,v in SECTORS.items()}, jobs=jobs,
                stage3_gap_ranges_m={key:list(value) for key,value in STAGE3_GAP_RANGES_M.items()},
                stratification_note="Speed-advantage strata (stage 2) and close/approach gap strata (stage 3) cycle independently within every sector, avoiding sector/encounter-difficulty confounding.",
                split_note="Paired training variants share one encounter and sensor seed. No encounter or sensor seed is shared across splits. Fit also excludes frames outside each split's sectors.")
    body["manifest_digest"] = digest(body)
    validate_manifest(body)
    return body


def validate_manifest(manifest):
    body = {k:v for k,v in manifest.items() if k != "manifest_digest"}
    if digest(body) != manifest.get("manifest_digest"):
        raise ValueError("Manifest was changed after its partitions were sealed")
    warmup_seconds = manifest.get("warmup_seconds",0.)
    validate_rollout_timing(manifest["seconds"],manifest["control_dt"],warmup_seconds)
    gap_ranges = manifest.get("stage3_gap_ranges_m")
    if gap_ranges is not None:
        close,approach = gap_ranges.get("close",[]),gap_ranges.get("approach",[])
        if (len(close)!=2 or len(approach)!=2 or not all(np.isfinite(v) for v in (*close,*approach))
                or not .63<close[0]<close[1]<=approach[0]<approach[1]):
            raise ValueError("Stage 3 gap ranges must preserve more than 0.63m starting separation and distinct close/approach strata")
    owners = {}
    ids = set()
    for job in manifest["jobs"]:
        if job["trial_id"] in ids:
            raise ValueError("Duplicate trial ID")
        ids.add(job["trial_id"])
        scene, split = job["scenario"], job["split"]
        if digest(scene) != job["scenario_digest"] or scene["stage"] != job["stage"]:
            raise ValueError("Scenario metadata mismatch")
        if scene.get("warmup_seconds",0.) != warmup_seconds:
            raise ValueError("Scenario warmup differs from the sealed round setting")
        validate_rollout_timing(scene["seconds"],manifest["control_dt"],scene.get("warmup_seconds",0.))
        if scene["stage"]==3 and gap_ranges is not None:
            low,high = gap_ranges["close" if job["scenario_group"] else "approach"]
            if not low<=scene["gap"]<=high:
                raise ValueError("Stage 3 encounter is outside its sealed gap stratum")
        if sector_at(scene["s"], manifest["track_length_m"]) not in SECTORS[split]:
            raise ValueError("Scene is outside its reserved partition")
        for key in (("scene", job["scenario_digest"]), ("sensor_seed", scene["seed"]), ("sensor_seed", scene["seed"] + 1)):
            if key in owners and owners[key] != (split, job["case_id"]):
                raise ValueError("Split or scenario sensor-seed overlap")
            owners[key] = (split, job["case_id"])
    return True


def select_jobs(manifest, stage, split, *, policy="teacher"):
    jobs = [j for j in manifest["jobs"] if j["stage"] == stage and j["split"] == split]
    if split == "test" and policy != "neural":
        raise ValueError("Test encounters are for frozen neural evaluation only")
    if policy == "neural":
        seen = set()
        jobs = [j for j in jobs if not (j["case_id"] in seen or seen.add(j["case_id"]))]
    return jobs


_WORKER = {}


def model_artifacts(stage, policy, model):
    paths = []
    if policy == "neural":
        paths.append(Path(model))
    if stage == 3:
        paths.append(MODELS / "stage2.npz")
    files = {}
    for path in paths:
        files[str(path.resolve())] = file_hash(path)
        value = path.with_name(path.stem + ".value.npz")
        files[str(value.resolve())] = file_hash(value) if value.exists() else None
    return files


def _worker_init(expected_sources, expected_reference, expected_models):
    from world import make_env
    from track_geometry import TrackGeometry
    if sources() != expected_sources:
        raise RuntimeError("Controller source changed during rollout collection")
    env = make_env(seed=0)
    geometry = TrackGeometry(env.unwrapped.track)
    if geometry.reference_digest != expected_reference:
        raise RuntimeError("The current racing reference differs from the sealed manifest")
    _WORKER.update(env=env, geometry=geometry, sources=expected_sources, model_artifacts=expected_models)


def _run_job(payload):
    from run_experiment import Scenario, episode
    from benchmark import offline_only
    job, manifest, outdir, policy, model = payload
    if sources() != _WORKER["sources"]:
        raise RuntimeError("Controller source changed after worker startup")
    if model_artifacts(job["stage"], policy, model) != _WORKER["model_artifacts"]:
        raise RuntimeError("A frozen ego/opponent checkpoint changed during collection")
    with offline_only():
        result, frames, samples, _ = episode(Scenario(**job["scenario"]), teacher=policy == "teacher",
            model_path=model, collect=policy == "teacher", env=_WORKER["env"], geometry=_WORKER["geometry"],
            control_dt=manifest["control_dt"], policy_bias=job["policy_bias"] if policy == "teacher" else None)
    if sources() != _WORKER["sources"]:
        raise RuntimeError("Source changed while a rollout was in progress")
    outdir = Path(outdir)
    trace_path = outdir / (job["trial_id"] + ".trace.json")
    samples_path = outdir / (job["trial_id"] + ".samples.jsonl")
    result.update(training_split=job["split"], training_case_id=job["case_id"],
                  training_variant=job["variant"] if policy == "teacher" else "learned", reference_digest=manifest["reference_digest"],
                  manifest_digest=manifest["manifest_digest"], network_access="blocked")
    trace_path.write_text(json.dumps(dict(result=result, frames=frames)) + "\n")
    if policy == "teacher":
        samples_path.write_text("".join(json.dumps(r) + "\n" for r in samples))
    summary = dict(job=job, result=result, trace=str(trace_path.resolve()),
                   samples=str(samples_path.resolve()) if policy == "teacher" else None,
                   completed=True, source_sha256=_WORKER["sources"],
                   reference_digest=manifest["reference_digest"], manifest_digest=manifest["manifest_digest"],
                   policy=policy, model_file=model, model_artifacts=_WORKER["model_artifacts"])
    (outdir / (job["trial_id"] + ".json")).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def collect(root, manifest, stage, split, *, workers=3, policy="teacher", model=None, tag=None, resume=True):
    """Run actual closed-loop episodes, with process-local simulator state."""
    validate_manifest(manifest)
    if workers < 1:
        raise ValueError("workers must be positive")
    if policy == "neural" and model is None:
        raise ValueError("Neural evaluation needs an explicit frozen checkpoint")
    source_hashes = sources()
    model = str(Path(model).resolve()) if model else None
    tag = tag or policy
    if Path(tag).name != tag:
        raise ValueError("Output tag must be one path component")
    outdir = Path(root) / "trials" / f"stage{stage}" / split / tag
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = select_jobs(manifest, stage, split, policy=policy)
    completed, pending = [], []
    expected_model_hash = file_hash(model) if model else None
    expected_models = model_artifacts(stage, policy, model)
    for job in jobs:
        path = outdir / (job["trial_id"] + ".json")
        if resume and path.exists():
            record = json.loads(path.read_text())
            if (not record.get("completed") or record["manifest_digest"] != manifest["manifest_digest"]
                    or record["source_sha256"] != source_hashes
                    or record.get("model_artifacts") != expected_models
                    or record["result"].get("model_sha256") != expected_model_hash):
                raise RuntimeError(f"Stale or incompatible saved rollout: {path}")
            completed.append(record)
        else:
            pending.append((job, manifest, str(outdir), policy, model))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=_worker_init,
                             initargs=(source_hashes, manifest["reference_digest"], expected_models)) as pool:
        futures = {pool.submit(_run_job, payload):payload[0]["trial_id"] for payload in pending}
        for future in as_completed(futures):
            record = future.result()
            completed.append(record)
            print(json.dumps(dict(stage=stage, split=split, policy=policy, completed=len(completed),
                                  requested=len(jobs), trial=futures[future], outcome=record["result"]["outcome"],
                                  passed_at=record["result"]["passed_at"], elapsed_seconds=round(time.monotonic()-started, 1))), flush=True)
    completed.sort(key=lambda r:r["job"]["trial_id"])
    report = dict(stage=stage, split=split, policy=policy, requested_trials=len(jobs),
                  completed_trials=len(completed), completed=len(completed) == len(jobs),
                  unique_encounters=len({r["job"]["case_id"] for r in completed}),
                  outcomes=outcome_summary(completed), manifest_digest=manifest["manifest_digest"],
                  reference_digest=manifest["reference_digest"], source_sha256=source_hashes,
                  model_file=model, model_sha256=expected_model_hash, model_artifacts=expected_models,
                  training_requirement_met=split == "train" and len(completed) >= 100,
                  trials=completed)
    (outdir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def outcome_summary(records):
    runs = [r["result"] for r in records]
    passes = [r["passed_at"] for r in runs if r["success"]]
    return dict(trials=len(runs), passes=len(passes), collisions=sum(r["collision"] is not None for r in runs),
                success_rate=len(passes)/len(runs) if runs else 0.,
                median_success_time_s=float(np.median(passes)) if passes else None,
                mean_success_time_s=float(np.mean(passes)) if passes else None,
                failures_censored=True)


def successful_winners(records):
    """The earliest actually successful policy wins its paired encounter."""
    winners = {}
    for record in records:
        result = record["result"]
        if not result["success"] or result["collision"] is not None or result.get("passed_at") is None:
            continue
        key = record["job"]["case_id"]
        if key not in winners or result["passed_at"] < winners[key]["result"]["passed_at"]:
            winners[key] = record
    return list(winners.values())


def eligible_samples(record, manifest, *, expected_split, cap=48):
    """Avoid no-target/post-pass and wrong-sector samples, even within an episode."""
    job, result = record["job"], record["result"]
    if job["split"] != expected_split or expected_split == "test":
        raise ValueError("Only the declared train/validation partition may enter fit")
    if record["manifest_digest"] != manifest["manifest_digest"] or record["reference_digest"] != manifest["reference_digest"]:
        raise ValueError("Sample provenance mismatch")
    nclasses = 4 if job["stage"] == 3 else 3
    candidates = []
    for line in Path(record["samples"]).read_text().splitlines():
        row = json.loads(line)
        x = np.asarray(row.get("features", []), dtype=np.float32)
        if (x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all()
                or row.get("action") not in ACTIONS[:nclasses]):
            continue
        if sector_at(row["ego_s"], manifest["track_length_m"]) not in SECTORS[expected_split]:
            continue
        if x[FEATURE_NAMES.index("target_present")] < .5:
            continue
        # Value means progress towards passing a currently relevant car, not
        # cruising after passing. Negative target distance is in 5m units.
        if x[FEATURE_NAMES.index("target_distance")] < -.04:
            continue
        if result.get("passed_at") is not None and row["t"] >= result["passed_at"]:
            continue
        if row["t"] < .15:
            continue
        candidates.append(row)
    if len(candidates) > cap:
        candidates = [candidates[i] for i in np.linspace(0, len(candidates)-1, cap, dtype=int)]
    return candidates


def supervision_exclusion(record, row):
    """A prelude collision cannot label an action first applied afterward.

    The complete episode remains a contact failure in every rollout summary.
    Contacts after policy control begins remain ordinary failure supervision.
    """
    collision = record["result"].get("collision")
    policy_start = float(record["result"].get("warmup_seconds",0.))
    if (collision is not None and policy_start > 0
            and float(collision["time"]) <= policy_start + 1e-9
            and float(row["t"]) >= float(collision["time"]) - 1e-9):
        return dict(trial_id=record["job"]["trial_id"],t=row["t"],
                    collision_at_seconds=collision["time"],policy_start_seconds=policy_start,
                    reason="Contact was recorded during the physical prelude before maneuver-policy control")
    return None


def build_arrays(records, manifest, stage, split):
    """All failures produce binary outcomes; only verified winners imitate actions."""
    if split not in ("train", "validation"):
        raise ValueError("Test data is forbidden in fitting")
    imitation_X, imitation_y, value_X, value_y, provenance, exclusions = [], [], [], [], [], []
    winners = {r["job"]["trial_id"] for r in successful_winners(records)}
    for record in records:
        if record["job"]["stage"] != stage or record["job"]["split"] != split:
            raise ValueError("Mixed stages or data partitions")
        rows = eligible_samples(record, manifest, expected_split=split)
        for row in rows:
            exclusion = supervision_exclusion(record,row)
            if exclusion is not None:
                exclusions.append(exclusion)
                continue
            action = ACTIONS.index(row["action"])
            x = np.asarray(row["features"], np.float32)
            onehot = np.eye(4, dtype=np.float32)[action]
            value_X.append(np.r_[x, onehot])
            value_y.append(int(record["result"]["success"] and record["result"]["collision"] is None))
            provenance.append(dict(trial_id=record["job"]["trial_id"], t=row["t"], ego_s=row["ego_s"],
                                   split=split, action=row["action"], episode_outcome=record["result"]["outcome"]))
            if record["job"]["trial_id"] in winners:
                imitation_X.append(x)
                imitation_y.append(action)
    return dict(imitation_X=np.asarray(imitation_X, np.float32).reshape(-1,len(FEATURE_NAMES)),
                imitation_y=np.asarray(imitation_y, np.int64),
                value_X=np.asarray(value_X, np.float32).reshape(-1,len(VALUE_FEATURE_NAMES)),
                value_y=np.asarray(value_y, np.int64), provenance=provenance,
                excluded_supervision_rows=exclusions,
                successful_winning_encounters=len(winners))


def load_collection(root, stage, split, manifest):
    path = Path(root) / "trials" / f"stage{stage}" / split / "teacher" / "summary.json"
    report = json.loads(path.read_text())
    if not report["completed"] or report["manifest_digest"] != manifest["manifest_digest"]:
        raise RuntimeError("Rollout collection is incomplete or belongs to another manifest")
    if report["source_sha256"] != sources():
        raise RuntimeError("Controller changed after rollout collection; do not fit stale trajectories")
    return report


def fit_stage(root, manifest, stage, *, epochs=100, warmstart=None, allow_small=False, reviews=None):
    validate_manifest(manifest)
    training = load_collection(root, stage, "train", manifest)
    validation = load_collection(root, stage, "validation", manifest)
    if training["completed_trials"] < 100 and not allow_small:
        raise RuntimeError("At least 100 completed training episodes are required per stage")
    train = build_arrays(training["trials"], manifest, stage, "train")
    val = build_arrays(validation["trials"], manifest, stage, "validation")
    if len(train["imitation_X"]) < 2:
        raise RuntimeError("No successful training trajectory: do not invent a winning imitation label")
    if len(np.unique(train["value_y"])) < 2:
        raise RuntimeError("Value learning needs both completed passes and failed episodes")
    rollout_imitation_rows = len(train["imitation_X"])
    advice = None
    advice_metadata = dict(rows_consumed=0, value_labels_modified=0, validation_rows_used=0, test_rows_used=0)
    if reviews is not None:
        # Offline cache verification only: this function never makes API calls.
        # The outcome arrays remain untouched; advice teaches imitation alone.
        from review_rollouts_llm import load_advisory_rows
        advice = load_advisory_rows(reviews, manifest, stage, training["trials"])
        train["imitation_X"] = np.concatenate((train["imitation_X"], advice["X"]))
        train["imitation_y"] = np.concatenate((train["imitation_y"], advice["y"]))
        advice_metadata = advice["metadata"]
    outdir = Path(root) / "models"
    outdir.mkdir(parents=True, exist_ok=True)
    common = dict(stage=stage, reference_digest=manifest["reference_digest"],
                  manifest_digest=manifest["manifest_digest"], source_sha256=sources(),
                  training_pipeline_sha256=file_hash(__file__),
                  completed_training_episodes=training["completed_trials"],
                  unique_training_encounters=training["unique_encounters"],
                  supervision_exclusion_counts={"train":len(train["excluded_supervision_rows"]),
                                                "validation":len(val["excluded_supervision_rows"])},
                  supervision_exclusion_rule="Exclude observation supervision at/after a collision already recorded before maneuver-policy control; retain every trial/contact in aggregate outcomes. Post-policy contacts remain failure labels.",
                  test_data_used=False, legacy_llm_feature_rows_used=0,
                  legacy_llm_note="Prior LLM reviews remain archived. Their old-reference feature vectors are not mixed into these rollout datasets.")
    im_val = (val["imitation_X"], val["imitation_y"]) if len(val["imitation_X"]) else None
    imitation = fit_network(train["imitation_X"], train["imitation_y"], outdir/f"stage{stage}.npz",
        validation=im_val, classes=ACTIONS[:4 if stage==3 else 3], feature_names=FEATURE_NAMES,
        epochs=epochs, seed=manifest["seed"]+stage, warmstart=warmstart,
        metadata={**common, "label_source":"actual successful rollout actions; fastest successful paired variant; optional separately identified feasible LLM advice",
                  "label_sources":{"successful_rollout_action":rollout_imitation_rows,
                                   "llm_advisory":advice_metadata["rows_consumed"]},
                  "fresh_llm_advice":advice_metadata,
                  "winning_encounters":train["successful_winning_encounters"]})
    value = fit_network(train["value_X"], train["value_y"], outdir/f"stage{stage}.value.npz",
        validation=(val["value_X"],val["value_y"]) if len(val["value_X"]) else None,
        classes=("FAIL","PASS"), feature_names=VALUE_FEATURE_NAMES, epochs=epochs,
        seed=manifest["seed"]+100+stage,
        metadata={**common, "label_source":"observed complete episode outcome",
                  "outcome_semantics":"Episode success under the applied action and continuation policy; not causal action ground truth or a safety certificate.",
                  "action_encoding":"one-hot LEFT RIGHT WAIT PROBE", "episode_deadline_s":manifest["seconds"]})
    value_calibration = (probability_metrics(TinyNetwork.load(outdir/f"stage{stage}.value.npz").predict_proba(val["value_X"])[:,1],val["value_y"])
                         if len(val["value_X"]) else None)
    for checkpoint in ("early", "mid"):
        shutil.copy2(outdir/f"stage{stage}.value.{checkpoint}.npz", outdir/f"stage{stage}.{checkpoint}.value.npz")
    dataset = Path(root)/"datasets"
    dataset.mkdir(exist_ok=True)
    for split, arrays in (("train", train),("validation",val)):
        np.savez_compressed(dataset/f"stage{stage}.{split}.npz", **{k:v for k,v in arrays.items() if isinstance(v,np.ndarray)})
        (dataset/f"stage{stage}.{split}.provenance.jsonl").write_text("".join(json.dumps(r)+"\n" for r in arrays["provenance"]))
        (dataset/f"stage{stage}.{split}.excluded_supervision.jsonl").write_text("".join(json.dumps(r)+"\n" for r in arrays["excluded_supervision_rows"]))
    if advice is not None:
        (dataset/f"stage{stage}.train.llm_advisory.provenance.jsonl").write_text("".join(json.dumps(r)+"\n" for r in advice["provenance"]))
    report = dict(stage=stage, metadata=common, training_rollouts=training["outcomes"],
                  validation_rollouts=validation["outcomes"], successful_winning_encounters=train["successful_winning_encounters"],
                  imitation_rows=len(train["imitation_X"]), value_rows=len(train["value_X"]),
                  successful_rollout_imitation_rows=rollout_imitation_rows,
                  excluded_supervision_rows=common["supervision_exclusion_counts"],
                  fresh_llm_advice=advice_metadata,
                  value_pass_rows=int(train["value_y"].sum()), value_failure_rows=int((train["value_y"]==0).sum()),
                  imitation=imitation, value=value, value_validation_calibration=value_calibration,
                  deployment_selection="Must choose a checkpoint by closed-loop validation success first and time second; label accuracy alone does not qualify the model.")
    (Path(root)/f"stage{stage}.training.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def probability_metrics(probabilities, labels):
    """Descriptive held-out-validation calibration, not a success guarantee."""
    p, y = np.asarray(probabilities,float),np.asarray(labels,float)
    if p.shape != y.shape or p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Matching finite probability/label vectors are required")
    ece = 0.
    bins = []
    for i in range(10):
        mask = (p >= i/10) & ((p < (i+1)/10) if i<9 else (p <= 1.))
        if mask.any():
            confidence, frequency = float(p[mask].mean()),float(y[mask].mean())
            ece += float(mask.mean()) * abs(confidence-frequency)
            bins.append(dict(low=i/10,high=(i+1)/10,count=int(mask.sum()),mean_probability=confidence,observed_pass_fraction=frequency))
    return dict(samples=len(p),pass_samples=int(y.sum()),failure_samples=int((y==0).sum()),
                brier_score=float(np.mean((p-y)**2)),expected_calibration_error_10_bins=ece,
                accuracy=float(np.mean((p>=.5)==y)),bins=bins,
                scope="Validation frames from separate encounters; the same split calibrates temperature. Correlated episode frames and continuation-policy dependence limit probability interpretation.")


def policy_rank(report):
    score = report["outcomes"]
    return (score["passes"], -score["collisions"], -(score["mean_success_time_s"] or float("inf")))


def select_checkpoint(root, manifest, stage, *, workers=3, deploy=None, include_imitation_only=True):
    reports = {}
    for checkpoint in ("early", "mid", "final"):
        model = Path(root)/"models"/(f"stage{stage}.npz" if checkpoint == "final" else f"stage{stage}.{checkpoint}.npz")
        reports[checkpoint] = collect(root, manifest, stage, "validation", workers=workers,
                                      policy="neural", model=model, tag=f"selection_{checkpoint}")
    if include_imitation_only:
        model = Path(root)/"models"/f"stage{stage}.imitation_only.npz"
        shutil.copy2(Path(root)/"models"/f"stage{stage}.npz",model)
        if model.with_name(model.stem+".value.npz").exists():
            raise RuntimeError("The imitation-only comparison must not have a value head")
        reports["imitation_only"] = collect(root,manifest,stage,"validation",workers=workers,
                                            policy="neural",model=model,tag="selection_imitation_only")
    qualified = [name for name,report in reports.items() if report["outcomes"]["collisions"]==0]
    if not qualified:
        failure = dict(stage=stage,qualified=False,reason="Every validation checkpoint made contact; none deployed",
                       reports={k:{kk:vv for kk,vv in v.items() if kk != "trials"} for k,v in reports.items()})
        (Path(root)/f"stage{stage}.selection.json").write_text(json.dumps(failure,indent=2)+"\n")
        raise RuntimeError(f"Stage {stage}: no zero-contact validation checkpoint qualifies for deployment")
    best = max(qualified, key=lambda name:policy_rank(reports[name]))
    source = Path(reports[best]["model_file"])
    record = dict(stage=stage, qualified=True, selected=best, selected_model=str(source), criterion="Require zero validation contacts, then maximize completed passes, then earliest mean successful pass",
                  validation_only=True, reports={k:{kk:vv for kk,vv in v.items() if kk != "trials"} for k,v in reports.items()})
    if deploy:
        dest = Path(deploy)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        source_value=source.with_name(source.stem+".value.npz")
        dest_value=dest.with_name(dest.stem+".value.npz")
        if source_value.exists():
            shutil.copy2(source_value,dest_value)
        elif dest_value.exists():
            dest_value.replace(dest_value.with_name(dest_value.stem+".disabled-"+file_hash(dest_value)[:10]+".npz"))
        record["deployed_model"] = str(dest.resolve())
        record["deployed_value_head"] = source_value.exists()
    (Path(root)/f"stage{stage}.selection.json").write_text(json.dumps(record,indent=2)+"\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command",required=True)
    init = sub.add_parser("init")
    init.add_argument("--seed",type=int,default=20261119)
    init.add_argument("--train-cases",type=int,default=36)
    init.add_argument("--validation-cases",type=int,default=10)
    init.add_argument("--test-cases",type=int,default=30)
    init.add_argument("--seconds",type=float,default=18.)
    init.add_argument("--control-dt",type=float,default=.05)
    init.add_argument("--warmup-seconds",type=float,default=.4,
                      help="Scored physical sensor warmup for both moving cars; 0..0.4 in whole control periods")
    run = sub.add_parser("collect")
    run.add_argument("--stage",type=int,choices=(2,3),required=True)
    run.add_argument("--split",choices=tuple(SECTORS),required=True)
    run.add_argument("--workers",type=int,default=3)
    run.add_argument("--policy",choices=("teacher","neural"),default="teacher")
    run.add_argument("--model",type=Path)
    run.add_argument("--tag")
    fit = sub.add_parser("fit")
    fit.add_argument("--stage",type=int,choices=(2,3),required=True)
    fit.add_argument("--epochs",type=int,default=100)
    fit.add_argument("--warmstart",type=Path)
    fit.add_argument("--allow-small",action="store_true")
    fit.add_argument("--reviews",type=Path,help="Accepted training-only LLM advice JSONL; supplements imitation only")
    select = sub.add_parser("select")
    select.add_argument("--stage",type=int,choices=(2,3),required=True)
    select.add_argument("--workers",type=int,default=3)
    select.add_argument("--deploy",type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    manifest_path = args.root/"manifest.json"
    if args.command == "init":
        from world import make_env
        from track_geometry import TrackGeometry
        if manifest_path.exists():
            raise FileExistsError("Use a new round directory; a sealed manifest is not overwritten")
        env = make_env()
        try:
            g=TrackGeometry(env.unwrapped.track)
            manifest=make_manifest(g.length,g.reference_digest,seed=args.seed,train_cases=args.train_cases,
                                   validation_cases=args.validation_cases,test_cases=args.test_cases,
                                   seconds=args.seconds,control_dt=args.control_dt,warmup_seconds=args.warmup_seconds)
        finally:
            env.close()
        manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
        print(json.dumps({k:v for k,v in manifest.items() if k != "jobs"},indent=2))
        return
    manifest=json.loads(manifest_path.read_text())
    if args.command == "collect":
        result=collect(args.root,manifest,args.stage,args.split,workers=args.workers,policy=args.policy,model=args.model,tag=args.tag)
        print(json.dumps({k:v for k,v in result.items() if k != "trials"},indent=2))
    elif args.command == "fit":
        result=fit_stage(args.root,manifest,args.stage,epochs=args.epochs,warmstart=args.warmstart,allow_small=args.allow_small,reviews=args.reviews)
        print(json.dumps({k:v for k,v in result.items() if k not in ("imitation","value")},indent=2))
    else:
        print(json.dumps(select_checkpoint(args.root,manifest,args.stage,workers=args.workers,deploy=args.deploy),indent=2))


if __name__ == "__main__":
    main()
