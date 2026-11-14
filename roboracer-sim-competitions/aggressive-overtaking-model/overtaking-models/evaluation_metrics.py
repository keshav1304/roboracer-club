"""Trace-derived passing metrics; unsuccessful runs never receive a pass time."""
from __future__ import annotations

import numpy as np


TRAIN_START_SECTORS = (0, 3, 4, 7, 8, 11)
RESERVED_START_SECTORS = (1, 5, 9)


def passing_metrics(result, frames, geometry, *, early_fraction=.5):
    """Measure completion, not just being momentarily alongside the opponent.

    Distances follow the car's observed reference-line progress. The trace is
    sampled at 20Hz, so these derived distances/times have that resolution.
    Completion time comes from the original 100Hz episode criterion.
    """
    if not frames:
        return {"time_to_pass_seconds": None, "early_pass": False,
                "metrics_available": False}
    length = float(geometry.length)
    initial_s = float(result["scenario"]["s"])
    previous_s, progress = initial_s % length, 0.
    distances, followed_distance = [], 0.
    followed_seconds, stopped_behind_seconds, opponent_stationary = 0., 0., 0.
    first_move, first_alongside = None, None
    previous_t = 0.
    opponent_offsets = []
    for frame in frames:
        s, _ = geometry.project(*frame["ego"][:2])
        ds = (float(s) - previous_s + length / 2) % length - length / 2
        progress += ds
        distances.append(progress)
        dt = max(0., float(frame["t"]) - previous_t)
        behind = frame["relative_progress"] < 0.
        if behind:
            followed_distance += max(0., ds)
            followed_seconds += dt
            if frame["ego"][3] < .10:
                stopped_behind_seconds += dt
        if frame["opponent"][3] < .10:
            opponent_stationary += dt
        _, offset = geometry.project(*frame["opponent"][:2])
        opponent_offsets.append(abs(float(offset)))
        if first_move is None and frame["action"] in ("LEFT", "RIGHT", "PROBE"):
            first_move = float(frame["t"])
        if first_alongside is None and frame["relative_progress"] >= 0.:
            first_alongside = float(frame["t"])
        previous_t, previous_s = float(frame["t"]), float(s)
    passed = float(result["passed_at"]) if result["success"] and result.get("passed_at") is not None else None
    distance_at_pass = None if passed is None else float(np.interp(passed, [f["t"] for f in frames], distances))
    fraction = None if distance_at_pass is None else max(0., distance_at_pass) / length
    return {
        "metrics_available": True,
        "time_to_first_pass_action_seconds": first_move,
        "time_first_alongside_seconds": first_alongside,
        "time_to_pass_seconds": passed,
        "ego_progress_at_pass_m": distance_at_pass,
        "track_fraction_to_pass": fraction,
        "distance_travelled_while_behind_m": followed_distance,
        "time_spent_behind_seconds": followed_seconds,
        "stopped_behind_seconds": stopped_behind_seconds,
        "opponent_stationary_seconds": opponent_stationary,
        "opponent_max_reference_offset_m": max(opponent_offsets),
        "early_pass": bool(fraction is not None and fraction <= early_fraction),
        "early_pass_max_track_fraction": early_fraction,
        "distance_sampling_seconds": .05,
        "completion_rule": "100Hz: >1m lead and <0.28m reference offset for 1s; retain lead and avoid collisions through follow-through",
        "opponent_audit_note": "Stationary time and reference offset are diagnostics, not proof of induced yielding; use paired perception ablations for causality.",
    }


def aggregate_metrics(results):
    successful = [r["passing_metrics"] for r in results if r.get("success") and r.get("passing_metrics", {}).get("time_to_pass_seconds") is not None]
    times = [m["time_to_pass_seconds"] for m in successful]
    fractions = [m["track_fraction_to_pass"] for m in successful]
    return {
        "early_passes": sum(bool(r.get("passing_metrics", {}).get("early_pass")) for r in results),
        "median_time_to_pass_seconds_successes_only": float(np.median(times)) if times else None,
        "max_time_to_pass_seconds_successes_only": max(times) if times else None,
        "median_track_fraction_to_pass_successes_only": float(np.median(fractions)) if fractions else None,
        "failed_or_censored_trials": sum(not r["success"] and not r.get("safe_blocked_stop", False) for r in results),
        "total_stopped_behind_seconds": sum(r.get("passing_metrics", {}).get("stopped_behind_seconds", 0.) for r in results),
    }
