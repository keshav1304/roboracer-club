# Complete-rollout training workflow

The moving-car models now learn from completed simulator encounters. The
previous snapshot-based workflow and its results remain archived. Current
published results belong in `artifacts/RESULTS.md`; intermediate development
rounds must not be described as final qualified models.

The map is the imported AutoDRIVE default track. Both cars normally follow the
same locally lap-time-optimized reference and speed profile. The optimizer is
finite; this is not a claim of global optimality. Passing paths temporarily
depart from that reference while remaining inside the track. Stage 2's leading
car follows the reference without avoidance. Stage 3's rival independently
runs the saved Stage 2 network, using its own LiDAR and controller.
Both have a 2.4 m/s speed cap in Stage 3. Normal reference following uses a
2.0 m/s² lateral-acceleration allowance. Active passing allows 2.4 m/s² in Stage 2 and 3.2 m/s² in Stage 3. The Stage 3
rival runs the same Stage 2 controller, including its 2.4 m/s² allowance.
These explicit controller settings must not be attributed to learning. Equal
top-speed caps do not imply identical maneuver limits.

Stage 3 can recover from a close side-by-side position when its checked path
restores preferred padding without reducing the current physical body gap.
Only the extra padding brake can be released, and only for a fresh tracked car
beside the ego car. Wall, front, unknown-object and physical-body checks remain.
The 0.5-second restoration condition is a forecast, not a guarantee of the
actual restoration time. Stage 2 uses the same recovery only while completing
a pass: it must have observed the target ahead, committed to a passing side,
and now observe that target behind. Return-to-line checks retain nearby rear
cars until the predicted merge is clear.

## Run a complete new round

Use the interpreter at `../gym-two-car-reactive/.venv/bin/python`.

```bash
# Offline by default: no API requests.
../gym-two-car-reactive/.venv/bin/python train_rollout_round.py

# At least 100 distinct training encounters per stage; three variants each.
../gym-two-car-reactive/.venv/bin/python train_rollout_round.py --train-cases 102 --llm-reviews 10
```

The default round contains 108 training episodes per stage: 36 randomized
encounters, each driven with three maneuver preferences. These are 108 actual
physical rollouts, but only 36 independent initial encounters. Separate sets
contain ten validation encounters and thirty final test encounters per stage.
The expanded November 6 round uses 102 distinct encounters and 306 rollouts per
stage. Set `--train-cases 102` to reproduce that size.
No scenario or sensor seed is shared across partitions. Starting track sectors
are also separated, and fitting checks the actual locations of sampled frames.
This evaluates new encounters on the same known map, not a new track.

Each track sector receives several opponent-speed and starting-gap groups.
Paired policy variants share an initial scene so their actual outcomes can be
compared. Their later trajectories diverge; those comparisons are not causal
labels for arbitrary later actions.

The first 0.4 seconds establish motion from real LiDAR while both cars physically
drive the reference. Stage 3 starts both cars at a common speed limited by the
curvature at both positions and their 1.2 m previews. During this setup only,
both receive the same speed command and each uses its own reference steering.
This prevents a corner-start collision before the overtaking policies take
control. After setup, each controller receives only its own observations.
That time and any contacts count in the episode. No true
opponent speed or commands are given to either policy. Physics and footprint
checks run at 100 Hz; decisions run at 20 Hz. Compute time is measured separately
from the 50 ms decision period.

The runner preserves both deployed models and their optional outcome heads
before starting. It deploys the selected Stage 2 model before collecting Stage
3 data, freezes both models for final testing, and restores the original pair
if the round fails. Collected data and logs remain available for diagnosis.

## Preserving a trained opponent across a setup correction

The November 7 correction preserves Stage 2's trained checkpoint and all its
original training artifacts byte for byte. The Stage 3 startup change is in a
separate conditional path. Forty validation replays verify that Stage 2's
poses, controls, probabilities and outcomes match exactly under the corrected
runtime; no new Stage 2 selection is performed. `stage2_predecessor.py` seals
and verifies this explicitly scoped compatibility record. The report keeps
predecessor training hashes separate from current evaluation hashes. Every
final test and the recollected Stage 3 data use the corrected runtime.

## What the networks learn

- The maneuver network predicts LEFT, RIGHT, WAIT, and, for Stage 3, PROBE.
- Imitation labels come from the fastest actually successful policy in each
  training encounter, supplemented by separately identified feasible LLM advice.
- An additional outcome network learns observed PASS/FAIL episode results from
  both successful and unsuccessful trajectories.
- Validation chooses among early, middle, final, and imitation-only checkpoints.
  Selection first requires no contacts, then maximizes completed passes, then
  prefers earlier completion. Zero-contact selection alone does not establish
  useful passing performance.

The outcome values estimate continuation success under the observed policy.
They are not guaranteed success probabilities or geometric safety certificates.
The local planner still checks physical clearance and produces steering and
target-speed commands. This is a hybrid decision network and trajectory
controller, not end-to-end learned throttle and steering.

Training refuses to invent successful demonstrations when all exploratory runs
fail. Such a round needs a better feasible maneuver or training policy before
it can qualify as a learned passing model. A good classification score cannot
replace a completed physical pass.

## Evidence and playback

Every full trajectory, scenario, sensor seed, source hash, model hash, measured
outcome, and training provenance record remains in the round folder. Final
testing blocks outbound network connections. A completed pass requires a
sustained clear lead, return near the shared reference, and two more seconds of
contact-free driving. A pass must be confirmed within the opportunity deadline;
only its follow-through may extend beyond it. Either car contacting a wall or
the other car makes the trial fail.

The website shows at most four representative final-test clips per moving
stage, including failures when present. Aggregate statistics include every test,
not only the selected clips. `Watch Overtaking Models.command` opens that page;
`Run Neural Overtaking.command` reruns selected encounters in the simulator.
The live launcher checks that the models and controller match the tested files.

The LLM review client reads `OPENAI_API_KEY` locally and never includes it in
datasets or results. Reviews share the existing conservative cumulative $1
reservation ledger. Fitting, validation, driving, and timing are local. API
acceptance and estimated token cost do not establish whether promotional
credits covered the account's actual bill.
