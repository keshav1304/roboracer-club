"""Optional, bounded LLM advice on completed TRAINING rollouts only.

Run without --send to inspect the selected questions without making requests.
With --send, at most ten requests per stage/manifest use the existing cumulative
$1 reservation log. Advice may supplement imitation training; observed PASS/FAIL
labels always come exclusively from the simulator. Runtime never imports this.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import json
from pathlib import Path

import numpy as np

from bootstrap import ARTIFACTS
from llm_teacher import LLMTeacher, TeacherError
from maneuver_core import ACTIONS, FEATURE_NAMES
from rollout_training import (DEFAULT_ROOT, digest, eligible_samples, file_hash,
                              load_collection, sources, successful_winners, validate_manifest)

REVIEW_TYPE = "closed_loop_training_v1"
SHARED_BUDGET_LOG = ARTIFACTS / "llm_reviews_20261106.jsonl"
MAX_COST_USD = 1.
MAX_REVIEWS_PER_STAGE = 10


def _positive(record):
    result = record["result"]
    return bool(result["success"] and result["collision"] is None)


def review_examples(records, manifest, stage, *, limit=10):
    """Select diverse real training states, paired with actual policy outcomes.

    Policy variants have the same initial encounter, but diverge subsequently.
    Their results are not falsely described as causal labels for one action at
    the later anchor observation. No validation/test encounter is inspected.
    """
    validate_manifest(manifest)
    if stage not in (2, 3) or not 1 <= limit <= MAX_REVIEWS_PER_STAGE:
        raise ValueError("Choose stage 2/3 and one to ten training examples")
    jobs = {j["trial_id"]: j for j in manifest["jobs"] if j["stage"] == stage and j["split"] == "train"}
    grouped = defaultdict(list)
    for record in records:
        job = record["job"]
        if job["split"] != "train" or job["stage"] != stage or jobs.get(job["trial_id"]) != job:
            raise ValueError("LLM review accepts only the sealed training partition")
        if (not record.get("completed") or record["source_sha256"] != sources()
                or record["manifest_digest"] != manifest["manifest_digest"]
                or record["reference_digest"] != manifest["reference_digest"]):
            raise ValueError("Training rollout provenance is stale or incomplete")
        grouped[job["case_id"]].append(record)
    buckets = defaultdict(list)
    for case_id, paired in sorted(grouped.items()):
        expected = {j["trial_id"] for j in jobs.values() if j["case_id"] == case_id}
        if {r["job"]["trial_id"] for r in paired} != expected:
            raise ValueError("Review requires all controlled variants of the training encounter")
        winners = successful_winners(paired)
        ordered = winners + [r for r in paired if r not in winners]
        anchors = []
        for preference, record in enumerate(ordered):
            rows = eligible_samples(record, manifest, expected_split="train", cap=100000)
            for row in rows:
                features = row["features"]
                if (sum(c["feasible"] for c in row["candidates"]) < 2
                        or features[FEATURE_NAMES.index("target_confidence")] < .6
                        or features[FEATURE_NAMES.index("target_speed")] < .1):
                    continue
                anchors.append(((preference, row["t"]), record, row))
        if not anchors:
            continue
        _, anchor_record, anchor = min(anchors, key=lambda a:a[0])
        outcomes = [{"trial_id":r["job"]["trial_id"], "preference_variant":r["job"]["variant"],
                     "policy_cost_bias":r["job"]["policy_bias"], "success":_positive(r),
                     "outcome":r["result"]["outcome"], "passed_at_seconds":r["result"].get("passed_at"),
                     "contact":r["result"]["collision"] is not None,
                     "minimum_body_gap_m":r["result"].get("minimum_body_gap"),
                     "minimum_wall_clearance_m":r["result"].get("minimum_wall_clearance"),
                     "simulated_seconds":r["result"].get("sim_seconds")}
                    for r in sorted(paired, key=lambda r:r["job"]["trial_id"])]
        request = dict(review_type=REVIEW_TYPE, stage=stage, split="train", case_id=case_id,
                       manifest_digest=manifest["manifest_digest"], reference_digest=manifest["reference_digest"],
                       anchor_trial_id=anchor_record["job"]["trial_id"], observed_at_seconds=anchor["t"],
                       scenario=anchor_record["job"]["scenario"], state=anchor.get("state"),
                       observation_features=dict(zip(FEATURE_NAMES,anchor["features"])),
                       candidates=anchor["candidates"], max_speed=anchor["max_speed"],
                       actual_counterfactual_policy_outcomes=outcomes,
                       interpretation="The variants share an initial encounter and sensor seed, but use different feasible-maneuver preferences for the whole episode. Their later trajectories diverge. These are actual complete-policy outcomes, not causal outcomes of choosing an alternative action at this anchor state. Recommend a feasible action at this supplied state, prioritizing collision-free completion, success frequency, then early completion. Your answer is advisory imitation supervision, never a replacement for observed success/failure labels.")
        item = dict(format_version=1, review_type=REVIEW_TYPE, stage=stage, split="train", case_id=case_id,
                    anchor_trial_id=anchor_record["job"]["trial_id"], sample=anchor,
                    sample_sha256=digest(anchor), manifest_digest=manifest["manifest_digest"],
                    reference_digest=manifest["reference_digest"], source_sha256=sources(), request=request,
                    request_sha256=digest(request), advisory_not_outcome_ground_truth=True,
                    value_label_eligible=False)
        count = sum(_positive(r) for r in paired)
        category = "mixed" if 0 < count < len(paired) else "passes" if count else "failures"
        buckets[(anchor_record["job"]["start_sector"], category)].append(item)
    # Round-robin the track sectors/outcomes so the first ten are not adjacent.
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for key in sorted(buckets,key=lambda key:({"mixed":0,"passes":1,"failures":2}[key[1]],key[0])):
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                if len(selected) == limit:
                    break
    return selected


def _log_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def run_reviews(root, stage, *, limit=10, send=False, model="gpt-4.1-mini", cache_path=SHARED_BUDGET_LOG):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    report = load_collection(root, stage, "train", manifest)
    examples = review_examples(report["trials"], manifest, stage, limit=limit)
    directory = root / "reviews"
    directory.mkdir(parents=True, exist_ok=True)
    plan = directory / f"stage{stage}.questions.json"
    plan.write_text(json.dumps(examples, indent=2) + "\n")
    output = directory / f"stage{stage}.llm.jsonl"
    result = dict(stage=stage, questions=len(examples), questions_path=str(plan.resolve()),
                  reviews_path=str(output.resolve()), requests_sent=0, accepted=0,
                  used_for_fit=False, consumption_note="Fit consumes advice only when passed --reviews; its training report records the exact review file hash and rows.",
                  cumulative_cost_ceiling_usd=MAX_COST_USD)
    if not send:
        return {**result, "dry_run":True}
    if not examples:
        raise ValueError("No eligible training observations to review")
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize stage 2/3 callers around the shared reservation ledger; otherwise
    # independent processes could both assume the same remaining dollar budget.
    with cache_path.with_suffix(cache_path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        prior = _log_records(cache_path)
        attempts = sum(r.get("scenario", {}).get("review_type") == REVIEW_TYPE
                       and r["scenario"].get("manifest_digest") == manifest["manifest_digest"]
                       and r["scenario"].get("stage") == stage for r in prior)
        teacher = LLMTeacher(model, cache_path, max_calls=max(0, MAX_REVIEWS_PER_STAGE-attempts),
                             max_cost_usd=MAX_COST_USD)
        accepted = {}
        if output.exists():
            accepted = {r["request_sha256"]:r for r in _log_records(output)}
        for example in examples:
            if example["request_sha256"] in accepted:
                continue
            try:
                label = teacher.label(example["request"])
            except TeacherError as exc:
                result["stopped_reason"] = str(exc)
                break
            record = {**example, "label":label, "cache_path":str(cache_path.resolve()),
                      "source":"openai_responses", "sample_weight":1.}
            accepted[example["request_sha256"]] = record
            output.write_text("".join(json.dumps(r, allow_nan=False)+"\n" for _,r in sorted(accepted.items())))
        result.update(requests_sent=teacher.calls, accepted=len(accepted),
                      cumulative_reserved_cost_usd=teacher.reserved_cost_usd,
                      cumulative_estimated_usage_cost_usd=teacher.usage_cost_usd)
    (directory/f"stage{stage}.review_report.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def load_advisory_rows(path, manifest, stage, training_records):
    """Recheck every accepted advice row against actual training data and cache.

    This produces imitation-only labels at one ordinary sample's weight each.
    It never supplies observed PASS/FAIL values or modifies validation examples.
    """
    validate_manifest(manifest)
    rows = _log_records(path)
    if not rows or len(rows) > MAX_REVIEWS_PER_STAGE:
        raise ValueError("Supply one to ten accepted advisory rows per stage")
    jobs = {j["trial_id"]:j for j in manifest["jobs"]}
    records = {r["job"]["trial_id"]:r for r in training_records}
    seen, cache_logs, X, y, provenance = set(), {}, [], [], []
    for row in rows:
        trial_id = row["anchor_trial_id"]
        job = jobs.get(trial_id)
        if (row.get("split") != "train" or row.get("stage") != stage or job is None
                or job["split"] != "train" or job["stage"] != stage or trial_id not in records):
            raise ValueError("LLM advice from validation/test or another stage is forbidden")
        if (row["case_id"] != job["case_id"] or row["case_id"] in seen
                or row["manifest_digest"] != manifest["manifest_digest"]
                or row["reference_digest"] != manifest["reference_digest"]
                or row["source_sha256"] != sources()):
            raise ValueError("Duplicate or stale LLM advice provenance")
        seen.add(row["case_id"])
        sample = row["sample"]
        eligible = eligible_samples(records[trial_id], manifest, expected_split="train",cap=100000)
        if digest(sample) != row["sample_sha256"] or not any(digest(r)==row["sample_sha256"] for r in eligible):
            raise ValueError("LLM observation does not match an actual eligible training sample")
        request = row["request"]
        if (digest(request) != row["request_sha256"] or request.get("split") != "train"
                or request.get("manifest_digest") != manifest["manifest_digest"]
                or request.get("reference_digest") != manifest["reference_digest"]
                or request.get("anchor_trial_id") != trial_id
                or request.get("candidates") != sample["candidates"]
                or request.get("observation_features") != dict(zip(FEATURE_NAMES,sample["features"]))):
            raise ValueError("LLM request differs from its training observation")
        label = row["label"]
        logpath = row["cache_path"]
        if logpath not in cache_logs:
            cache_logs[logpath] = {r["cache_key"]:r for r in _log_records(logpath) if r.get("status")=="accepted"}
        cached = cache_logs[logpath].get(label.get("cache_key"))
        if (cached is None or cached.get("scenario") != request
                or any(cached["label"].get(k) != label.get(k) for k in ("action","confidence","steering","target_speed","response_id","model"))
                or label.get("llm_used") is not True or label.get("source") != "openai_responses"):
            raise ValueError("LLM advice lacks a matching accepted API cache record")
        action = label["action"]
        candidates = {c["action"]:c for c in sample["candidates"]}
        if (action not in ACTIONS[:4 if stage==3 else 3] or action not in candidates
                or candidates[action]["feasible"] is not True
                or not 0 <= label["confidence"] <= 1 or row.get("sample_weight") != 1.):
            raise ValueError("LLM advisory action must be feasible with one ordinary sample's weight")
        X.append(sample["features"])
        y.append(ACTIONS.index(action))
        provenance.append(dict(source="llm_advisory",trial_id=trial_id,case_id=row["case_id"],
                               t=sample["t"],split="train",action=action,confidence=label["confidence"],
                               cache_key=label["cache_key"],response_id=label["response_id"],sample_weight=1.,
                               sample_sha256=row["sample_sha256"],rollout_verified=False,
                               note="Feasible LLM recommendation; no claim that the suggested action caused the paired episode's outcome."))
    return dict(X=np.asarray(X,np.float32),y=np.asarray(y,np.int64),provenance=provenance,
                metadata=dict(rows_consumed=len(X),file=str(Path(path).resolve()),sha256=file_hash(path),
                              sample_weight=1.,value_labels_modified=0,validation_rows_used=0,test_rows_used=0,
                              label_semantics="Training-only feasible LLM advice supplements imitation; binary outcomes remain independently observed simulator labels."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--stage",type=int,choices=(2,3),required=True)
    parser.add_argument("--limit",type=int,default=10)
    parser.add_argument("--send",action="store_true",help="Send the bounded training-only API requests; otherwise write the questions only")
    args = parser.parse_args()
    print(json.dumps(run_reviews(args.root,args.stage,limit=args.limit,send=args.send),indent=2))


if __name__ == "__main__":
    main()
