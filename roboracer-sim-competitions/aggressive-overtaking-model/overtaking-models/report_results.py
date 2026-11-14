"""Report the current revision from completed evaluations and recorded API usage."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from bootstrap import ARTIFACTS, MODELS
from evaluation_metrics import aggregate_metrics


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def api_usage(paths):
    """Include rejected billed responses; cached labels are not additional calls."""
    records = []
    for path in paths:
        if path.exists():
            records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    input_tokens = sum((r.get("usage") or {}).get("input_tokens", 0) for r in records)
    output_tokens = sum((r.get("usage") or {}).get("output_tokens", 0) for r in records)
    return {"files": [str(p.name) for p in paths if p.exists()], "recorded_requests": len(records),
            "accepted_responses": sum(r.get("status") == "accepted" for r in records),
            "status_counts": dict(Counter(r.get("status", "unknown") for r in records)),
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "estimated_usage_cost_usd": sum(r.get("estimated_usage_cost_usd",
                 (((r.get("usage") or {}).get("input_tokens", 0) * .40
                 + (r.get("usage") or {}).get("output_tokens", 0) * 1.60) / 1_000_000))
                 for r in records),
            "conservative_reserved_cost_usd": sum(r.get("reserved_cost_usd", 0.) for r in records),
            "cost_note": "Estimate from recorded response usage at configured model pricing; API billing dashboard is authoritative. Local training and testing add no API cost."}


def condition_row(label, runs):
    counts = Counter(r["outcome"] for r in runs)
    return {"condition": label, "trials": len(runs), "passes": counts["pass"],
            "safe_stops": counts["safe_stop"], "no_pass": counts["no_pass"],
            "collisions": counts["collision"],
            "min_body_gap": min((r["minimum_body_gap"] for r in runs), default=None),
            **aggregate_metrics(runs)}


def validate_evidence(evaluations, benchmark, model_hashes):
    """Refuse to combine measurements from stale source or model versions."""
    checked_sources = {}
    for stage, report in evaluations.items():
        runs = report["results"]
        if not runs or not report.get("evaluation_complete", False) or len(runs) != report.get("requested_trials"):
            raise RuntimeError(f"Stage {stage} evaluation is incomplete or lacks completion metadata")
        for run in runs:
            if run.get("model_sha256") != model_hashes[stage] or "sensor_seeds" not in run:
                raise RuntimeError(f"Stage {stage} evaluation model/seed mismatch")
            for name, recorded in run["source_sha256"].items():
                actual = sha(Path(__file__).with_name(name))
                if actual != recorded:
                    raise RuntimeError(f"Stale evaluation source: {name} in stage {stage}")
                checked_sources[name] = actual
            if run["scenario"]["stage"] == 3 and run["scenario"]["opponent"] == "reactive":
                if run.get("opponent_model_sha256") != model_hashes[2]:
                    raise RuntimeError("Stage 3 rival is not the frozen deployed stage 2 model")
    for name, recorded in benchmark.get("source_sha256", {}).items():
        if sha(Path(__file__).with_name(name)) != recorded:
            raise RuntimeError(f"Stale benchmark source: {name}")
    for stage in (1, 2, 3):
        entry = benchmark["results"][f"stage_{stage}"]
        if entry.get("checkpoint_sha256") != model_hashes[stage]:
            raise RuntimeError(f"Stale benchmark checkpoint for stage {stage}")
    return checked_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="revised_stage")
    parser.add_argument("--benchmark", default="benchmark_revised.json")
    parser.add_argument("--output", default="revised_report.json")
    args = parser.parse_args()
    evaluations = {s: read_json(ARTIFACTS / f"{args.prefix}{s}.json") for s in (1, 2, 3)}
    benchmark = read_json(ARTIFACTS / args.benchmark)
    training = {s: read_json(ARTIFACTS / f"stage{s}.training.json") for s in (1, 2, 3)}
    datasets = {s: read_json(ARTIFACTS / f"datasets/stage{s}.summary.json") for s in (1, 2, 3)}
    hashes = {s: sha(MODELS / f"stage{s}.npz") for s in (1, 2, 3)}
    sources = validate_evidence(evaluations, benchmark, hashes)
    usage = api_usage([ARTIFACTS / "credit_check_20261106.jsonl", ARTIFACTS / "llm_reviews_20261106.jsonl"])
    accepted_labels = sum(d["sources"].get("openai_responses", 0) for d in datasets.values())
    audit = read_json(ARTIFACTS / "opponent_audit_revised.json")
    influence = read_json(ARTIFACTS / "opponent_influence_revised.json")
    for evidence in (audit, influence):
        if any(sha(Path(__file__).with_name(n)) != h for n,h in evidence["source_sha256"].items()):
            raise RuntimeError("Opponent audit sources are stale")
    if not audit.get("completed") or any(r["opponent_model_sha256"] != hashes[2] for r in audit["no_threat_results"]):
        raise RuntimeError("Opponent audit incomplete or wrong checkpoint")
    audit_summary = {"all_no_threat_audits_passed": audit["all_no_threat_audits_passed"],
                     "trials": [{k:v for k,v in r.items() if k != "frames"} for r in audit["no_threat_results"]]}
    influence_summary = {k:influence[k]["result"] for k in ("full_lidar", "wall_only")}
    tuning = read_json(ARTIFACTS / "tuning_summary_20261106.json")
    success_ablation = read_json(ARTIFACTS / "opponent_success_ablation_revised.json")
    if any(sha(Path(__file__).with_name(n)) != h for n,h in success_ablation["source_sha256"].items()):
        raise RuntimeError("Successful-case ablation is stale")
    success_summary = {k:success_ablation[k]["result"] for k in ("full_lidar", "wall_only")}
    success_summary["selection_note"] = success_ablation["selection_note"]
    rows = []
    for stage, report in evaluations.items():
        runs = report["results"]
        if stage == 1:
            for blocked in (False, True):
                subset = [r for r in runs if r["scenario"]["blocked"] == blocked]
                if subset:
                    rows.append(condition_row("Static full blockage" if blocked else "Static passable obstacle", subset))
        else:
            rows.append(condition_row("Slower line-following opponent" if stage == 2 else "Equal-cap stage 2 neural opponent", runs))
    for group in ("close", "approach"):
        subset = [r for r in evaluations[3]["results"] if r.get("interaction_group") == group]
        if subset:
            rows.append(condition_row(f"Stage 3 subgroup: {group} start", subset))
    all_runs = [r for report in evaluations.values() for r in report["results"]]
    per_sector = {str(stage): {str(sector): condition_row(f"Start sector {sector}", [r for r in report["results"] if r["start_sector"] == sector])
                              for sector in sorted({r["start_sector"] for r in report["results"]})}
                  for stage, report in evaluations.items()}
    combined = {"status": "simulation prototype with API-reviewed teacher labels" if accepted_labels else "simulation prototype with geometric teacher labels only",
                "evaluations": rows, "per_start_sector": per_sector,
                "evaluation_files": [f"{args.prefix}{s}.json" for s in (1, 2, 3)],
                "evaluation_roles": {str(s): r.get("evaluation_role") for s, r in evaluations.items()},
                "benchmark": benchmark, "training": training,
                "accepted_api_labels_in_current_datasets": accepted_labels,
                "api_usage": usage, "checkpoint_sha256": hashes, "source_sha256": sources,
                "pressure_hypothesis_verified": False,
                "opponent_validity_audit": audit_summary,
                "opponent_influence_ablation": influence_summary,
                "successful_case_influence_ablation": success_summary,
                "development_tuning": tuning,
                "limitations": ["Neural maneuver probabilities with geometric trajectory and safety control; not end-to-end neural steering/throttle",
                                "API suggestions are not ground truth; rollout results determine success",
                                "Same track with reserved start sectors and new seeds, not unseen-track generalization",
                                "Historical tests in the same sectors informed development",
                                "Equal speed caps do not imply equal instantaneous speed; contact-free overtaking alone does not prove induced yielding",
                                "No proven globally optimal safety margin or competition reliability"]}
    (ARTIFACTS / args.output).write_text(json.dumps(combined, indent=2) + "\n")
    lines = ["# Revised overtaking measurements", "",
             f"The current datasets contain **{accepted_labels} accepted LLM-reviewed labels**. Remaining labels are recorded geometric bootstrap labels. The LLM is used only in training; evaluations block outgoing network connections.", "",
             f"Recorded API requests: {usage['recorded_requests']}; accepted responses: {usage['accepted_responses']}. Estimated API usage: **${usage['estimated_usage_cost_usd']:.4f}**, based on {usage['input_tokens']:,} input and {usage['output_tokens']:,} output tokens. This includes the credit check. Local neural training and simulation use no API credits.", "",
             "## Closed-loop behavior", "",
             f"All {len(all_runs)} trials are retained, including failures. Each completed pass requires >1 metre lead, reference offset <0.28 metres for 1 second, retained lead and continued collision checks. Early means completing within half a lap from the starting position. Pass-time statistics include successful runs only; no-pass trials are censored, never counted as fast passes.", "",
             "| Condition | Trials | Pass | Early | Safe stop | No pass | Collision | Median pass (s) | Median lap fraction |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def num(value, digits=2):
        return "—" if value is None else f"{value:.{digits}f}"
    for r in rows:
        lines.append(f"| {r['condition']} | {r['trials']} | {r['passes']} | {r['early_passes']} | {r['safe_stops']} | {r['no_pass']} | {r['collisions']} | {num(r['median_time_to_pass_seconds_successes_only'])} | {num(r['median_track_fraction_to_pass_successes_only'])} |")
    lines += ["", "The report JSON contains results per starting sector, time and distance spent behind, stopped-behind duration, opponent diagnostics, and exact model/source hashes. These reserved-start tests use the same track; earlier results from these sectors informed development.", "",
              "## Timing", "",
              "Controller timing excludes physics and rendering. Full-controller latency includes perception, planning, neural inference and control; neural-only timings are not compared against full reactive controllers.", "",
              "| Controller | Median (ms) | p95 (ms) | p99 (ms) | Neural-only median (ms) |", "|---|---:|---:|---:|---:|"]
    for name, r in benchmark["results"].items():
        f, n = r["full_controller"], r.get("network_only", {})
        lines.append(f"| {name} | {num(f['p50_ms'],3)} | {num(f['p95_ms'],3)} | {num(f['p99_ms'],3)} | {num(n.get('p50_ms'),4)} |")
    lines += ["", "The user's JEV 70–500 ms range is context, not a controlled benchmark on this simulator. Any claim of being faster must use the same complete processing scope and hardware.", "",
              "## Training and opponent validity", "",
              f"The final stage 2 opponent passed all no-threat audits: {audit['all_no_threat_audits_passed']}. Three 20-second runs check for unexplained stops, departures and collisions. The matched influence diagnostic produced {influence_summary['full_lidar']['outcome']} with full rival LiDAR and {influence_summary['wall_only']['outcome']} when the rival could not see the ego; exact trajectories are retained.", "",
              "Stage 3 uses the saved stage 2 neural model as its opponent, with the same speed cap and conservative collision margins. Stage 2 retains a line-following leader. Opponent controls and target detections are recorded in the traces; there is no scripted yielding command. Contact-free passing by itself does not establish that the ego forced a yield. That causal hypothesis remains unverified without matched opponent-perception ablations.", "",
              f"A separate post-hoc paired test of the successful qualification scenario reproduced a pass at {success_summary['full_lidar']['passed_at']:.2f} seconds with full rival LiDAR; with ego returns hidden the outcome was {success_summary['wall_only']['outcome']}. This supports opponent-response dependence in that one case. It does not establish reliable passing or forcing the rival laterally off its line. The earlier 8.55-second result was invalidated by the rear-target following-rule bug and is archived under superseded_follow_rule/.", "",
              "| Stage | Dataset rows | API-reviewed labels | Validation label accuracy | Selected epoch |", "|---|---:|---:|---:|---:|"]
    for stage, r in training.items():
        lines.append(f"| {stage} | {datasets[stage]['rows']} | {datasets[stage]['sources'].get('openai_responses',0)} | {r['validation']['accuracy']:.3f} | {r['selected_epoch']} |")
    lines += ["", "Validation label agreement measures imitation of teacher choices, not driving success. The three networks select maneuver probabilities; geometric control produces steering and target speed. An optimal racing line or globally optimal safety margin has not been established.", "",
              "Open the revised replay for simulations; use revised_report.json and the linked evaluation files for complete measurements. Previous results are preserved under before_llm_20261106/."]
    (ARTIFACTS / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"evaluations": rows, "api_usage": usage}, indent=2))


if __name__ == "__main__":
    main()
