"""Collect, train, select, test and publish one complete overtaking round.

Default training is offline. --llm-reviews 1..10 adds that many training-only
reviews per stage using the existing cumulative $1 API reservation ledger.
Stage 1 is unchanged. Failed/interrupted rounds restore the original Stage 2/3
checkpoints together, while retaining every collected trial and training file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

from bootstrap import ARTIFACTS, HERE, MODELS

MODEL_NAMES = ("stage2.npz", "stage2.value.npz", "stage3.npz", "stage3.value.npz")
PIPELINE_FILES = ("maneuver_core.py", "perception.py", "track_geometry.py", "world.py",
                  "run_experiment.py", "tiny_network.py", "rollout_training.py",
                  "review_rollouts_llm.py", "llm_teacher.py", "v3_report.py", "train_rollout_round.py")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_copy(source, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + destination.name + ".", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def capture_files(paths, directory):
    """Immutable backup, including the intentional absence of optional heads."""
    directory = Path(directory)
    if directory.exists():
        records = json.loads((directory/"manifest.json").read_text())
        if [r["destination"] for r in records] != [str(Path(p).resolve()) for p in paths]:
            raise RuntimeError("Saved checkpoint backup belongs to different deployment paths")
        verify_backup(records, directory)
        return records
    temporary = Path(tempfile.mkdtemp(prefix=".backup-",dir=directory.parent))
    try:
        records = []
        for path in map(Path,paths):
            record = dict(name=path.name, destination=str(path.resolve()),present=path.exists())
            if record["present"]:
                shutil.copy2(path,temporary/path.name)
                record["sha256"] = file_hash(temporary/path.name)
            records.append(record)
        write_json(temporary/"manifest.json", records)
        os.replace(temporary,directory)
        return records
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_backup(records, directory):
    for record in records:
        if record["present"] and file_hash(Path(directory)/record["name"]) != record["sha256"]:
            raise RuntimeError("Original checkpoint backup changed; automatic restoration is unsafe")


def restore_files(records, directory):
    verify_backup(records,directory)
    for record in records:
        destination = Path(record["destination"])
        if record["present"]:
            atomic_copy(Path(directory)/record["name"],destination)
        elif destination.exists():
            displaced = Path(directory)/"displaced"
            displaced.mkdir(exist_ok=True)
            destination.replace(displaced/(destination.name+"."+stamp()))


def run_command(argv, *, log_path, label):
    """Run one argument-vector command, stream progress and retain its full log."""
    print(f"\n{label}",flush=True)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True,exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("MPLCONFIGDIR",str(HERE.parent/"gym-two-car-reactive"/".matplotlib"))
    with log_path.open("w") as logfile:
        process = subprocess.Popen(list(map(str,argv)), cwd=HERE, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                print(line,end="",flush=True)
                logfile.write(line)
                logfile.flush()
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code,argv)
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            raise
        finally:
            process.stdout.close()


def validate_manifest_file(path, seed, warmup_seconds=.4,train_cases=36):
    from rollout_training import validate_manifest
    manifest = json.loads(Path(path).read_text())
    validate_manifest(manifest)
    expected = dict(format_version=2,seed=seed,train_cases_per_stage=train_cases,training_rollouts_per_stage=train_cases*3,
                    validation_cases_per_stage=10,test_cases_per_stage=30,control_dt=.05,seconds=18.,
                    warmup_seconds=warmup_seconds)
    if any(manifest.get(key) != value for key,value in expected.items()):
        raise RuntimeError("Existing manifest differs from this round's seed or required trial counts")
    return manifest


def run_round(root, *, seed=20261120, workers=3, epochs=100, llm_reviews=0,
              warmup_seconds=.4,train_cases=36,runner=run_command, models_dir=MODELS, artifacts=ARTIFACTS):
    if workers<1 or epochs<1 or not 0<=llm_reviews<=10:
        raise ValueError("Positive workers/epochs and zero to ten LLM reviews are required")
    if not isinstance(train_cases,int) or train_cases<34:
        raise ValueError("At least 34 distinct training encounters are required to supply 100 full variant rollouts")
    from rollout_training import validate_rollout_timing
    validate_rollout_timing(18.,.05,warmup_seconds)
    root,models_dir,artifacts = Path(root).resolve(),Path(models_dir).resolve(),Path(artifacts).resolve()
    artifacts.mkdir(parents=True,exist_ok=True)
    root.mkdir(parents=True,exist_ok=True)
    source_hashes = {name:file_hash(HERE/name) for name in PIPELINE_FILES}
    config = dict(seed=seed,epochs=epochs,llm_reviews_per_stage=llm_reviews,warmup_seconds=warmup_seconds,
                  models_directory=str(models_dir),source_sha256=source_hashes,
                  train_rollouts_per_stage=train_cases*3,training_encounters_per_stage=train_cases,
                  teacher_validation_rollouts_per_stage=30,final_tests_per_stage=30)
    state_path,config_path = root/"round_run.json",root/"round_config.json"
    with (artifacts/".training-round.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another round is training; wait before changing the shared deployed models") from None
        old_state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if old_state.get("status")=="complete":
            raise RuntimeError("This round is complete. Choose a new directory for new training")
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise RuntimeError("Round settings or code changed. Keep this round archived and use a new directory")
        if not config_path.exists() and any((root/"trials").glob("*/test/*/*.json")):
            raise RuntimeError("This directory already contains final tests. Use a new round to preserve their independence")
        for stage in (2,3):
            if not (models_dir/f"stage{stage}.npz").exists():
                raise FileNotFoundError(f"The current Stage {stage} checkpoint is required for the pre-round backup")
        priors = root/"priors"
        backup = capture_files([models_dir/name for name in MODEL_NAMES],priors)
        write_json(config_path,config)
        attempt = root/"attempts"/stamp()
        attempt.mkdir(parents=True)
        state = dict(status="running",round_directory=str(root),started_at=stamp(),workers=workers,
                     config=config,attempt_directory=str(attempt),
                     completed_steps=old_state.get("completed_steps",{}),commands=[])
        publication_backup = None

        def persist():
            write_json(state_path,state)
            write_json(attempt/"run.json",state)

        def step(name,label,argv,*,reusable_files=None):
            if any(file_hash(HERE/n)!=h for n,h in source_hashes.items()):
                raise RuntimeError("Code changed during the round; partial trials are retained but cannot be mixed")
            previous = state["completed_steps"].get(name)
            if reusable_files is not None and previous is not None:
                if (previous["argv"] != list(map(str,argv)) or not previous.get("artifacts")
                        or any(not Path(p).exists() or file_hash(p)!=h for p,h in previous["artifacts"].items())):
                    raise RuntimeError("A completed fit changed on disk; start a new round instead of reusing final tests")
                print(f"\n{label}: reusing the unchanged completed fit.",flush=True)
                return
            entry = dict(name=name,label=label,argv=list(map(str,argv)),started_at=stamp(),status="running")
            state["commands"].append(entry)
            persist()
            runner(argv,log_path=attempt/f"{len(state['commands']):02d}_{name}.log",label=label)
            entry.update(status="complete",finished_at=stamp())
            if reusable_files is not None:
                paths = list(reusable_files())
                if not paths:
                    raise RuntimeError("Fit completed without saved artifacts")
                entry["artifacts"] = {str(p.resolve()):file_hash(p) for p in paths}
            state["completed_steps"][name] = dict(entry)
            persist()

        python = sys.executable
        base = [python,str(HERE/"rollout_training.py"),"--root",str(root)]
        try:
            persist()
            # Also recover a round whose previous process was killed before it
            # could catch an interruption. Both stages begin from the same pair.
            if old_state:
                restore_files(backup,priors)
            if not (root/"manifest.json").exists():
                step("init","Preparing independent training, validation and test encounters",
                     base+["init","--seed",str(seed),"--warmup-seconds",str(warmup_seconds),
                           "--train-cases",str(train_cases)])
            manifest = validate_manifest_file(root/"manifest.json",seed,warmup_seconds,train_cases)
            state["manifest_digest"] = manifest["manifest_digest"]
            for stage in (2,3):
                for split,count in (("train",manifest["training_rollouts_per_stage"]),("validation",30)):
                    step(f"stage{stage}_{split}",f"Stage {stage}: collecting {count} complete {split} rollouts",
                         base+["collect","--stage",str(stage),"--split",split,"--workers",str(workers)])
                fit = base+["fit","--stage",str(stage),"--epochs",str(epochs),
                            "--warmstart",str(priors/f"stage{stage}.npz")]
                if llm_reviews:
                    step(f"stage{stage}_review",f"Stage {stage}: reviewing at most {llm_reviews} training encounters with the LLM",
                         [python,str(HERE/"review_rollouts_llm.py"),"--root",str(root),"--stage",str(stage),
                          "--limit",str(llm_reviews),"--send"])
                    reviews = root/"reviews"/f"stage{stage}.llm.jsonl"
                    if not reviews.exists() or not reviews.read_text().strip():
                        raise RuntimeError("LLM reviews were requested but none were accepted; see the review report")
                    fit += ["--reviews",str(reviews)]
                def fit_files(stage=stage):
                    return ([root/f"stage{stage}.training.json"] +
                            sorted((root/"models").glob(f"stage{stage}*.npz")) +
                            sorted((root/"models").glob(f"stage{stage}*.history.json")) +
                            sorted((root/"datasets").glob(f"stage{stage}.*")))
                step(f"stage{stage}_fit",f"Stage {stage}: training maneuver and observed-outcome networks",
                     fit,reusable_files=fit_files)
                step(f"stage{stage}_select",f"Stage {stage}: selecting a collision-free validation checkpoint",
                     base+["select","--stage",str(stage),"--workers",str(workers),
                           "--deploy",str(models_dir/f"stage{stage}.npz")])
                # Stage 2 is deployed here before ANY Stage 3 collection, so its
                # saved network really controls the Stage 3 opponent.
            frozen = {name:file_hash(models_dir/name) if (models_dir/name).exists() else None for name in MODEL_NAMES}
            state["frozen_test_models"] = frozen
            persist()
            for stage in (2,3):
                step(f"stage{stage}_test",f"Stage {stage}: running 30 untouched final tests with API access blocked",
                     base+["collect","--stage",str(stage),"--split","test","--workers",str(workers),
                           "--policy","neural","--model",str(models_dir/f"stage{stage}.npz")])
                if any((file_hash(models_dir/n) if (models_dir/n).exists() else None)!=h for n,h in frozen.items()):
                    raise RuntimeError("A deployed network changed during final testing")
            step("report","Preparing representative replays and results for all final tests",
                 [python,str(HERE/"v3_report.py"),"--root",str(root),"--output",str(root/"replay.html"),
                  "--report",str(root/"v3_report.json"),"--markdown",str(root/"RESULTS.md")])
            publications = [artifacts/name for name in ("replay.html","v3_report.json","RESULTS.md")]
            publication_backup = capture_files(publications,attempt/"publication_backup")
            for destination in publications:
                atomic_copy(root/destination.name,destination)
            state.update(status="complete",finished_at=stamp(),website=str(artifacts/"replay.html"),
                         original_checkpoints=str(priors),all_trials_retained=True)
            persist()
            print(f"\nRound complete. Results: {artifacts/'replay.html'}\nAll trials and original checkpoints: {root}",flush=True)
            return state
        except BaseException as exc:
            state.update(status="failed",failed_at=stamp(),error_type=type(exc).__name__,error=str(exc))
            try:
                restore_files(backup,priors)
                if publication_backup is not None:
                    restore_files(publication_backup,attempt/"publication_backup")
                state["original_checkpoints_restored"] = True
            except BaseException as restoration_error:
                state["restoration_error"] = str(restoration_error)
                persist()
                raise RuntimeError(f"Round failed and checkpoint restoration also failed: {restoration_error}. Original files remain in {priors}") from exc
            persist()
            print(f"\nTraining stopped. The original Stage 2/3 checkpoints were restored.\nAll partial results remain in {root}",flush=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,help="Round directory; default is a new timestamped artifacts folder")
    parser.add_argument("--seed",type=int,default=20261120)
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--train-cases",type=int,default=36,
                        help="Distinct random training encounters per stage, each with three full rollouts (default 36)")
    parser.add_argument("--warmup-seconds",type=float,default=.4,
                        help="Scored physical LiDAR prelude, 0..0.4 seconds in 0.05-second increments")
    parser.add_argument("--llm-reviews",type=int,choices=range(11),default=0,metavar="0..10",
                        help="Training-only API reviews per stage; default 0 makes no API requests")
    args = parser.parse_args()
    root = args.root or ARTIFACTS/("rollout_round_"+stamp())
    def interrupted(_signum,_frame):
        raise KeyboardInterrupt("Training interrupted")
    signal.signal(signal.SIGTERM,interrupted)
    try:
        run_round(root,seed=args.seed,workers=args.workers,epochs=args.epochs,llm_reviews=args.llm_reviews,
                  warmup_seconds=args.warmup_seconds,train_cases=args.train_cases)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"Unable to complete the round: {exc}",file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
