# Aggressive Overtaking Model

**Team release — November 14, 2026.** Includes source, models, saved track, simulator and RViz configuration. The complete training/test archive is supplied as a versioned release asset with checksums.

A small racing car learns which passing maneuver to attempt, while a geometric planner constructs and checks the path. A newer network predicts how a reactive opponent might respond to that path. Everything shown here runs in **F1TENTH Gym**, with **ROS 2 and RViz** displaying the cars, candidate paths, and predictions.

The final controller, **HybridCorridorStage4**, passed **23 of 24 fresh test encounters with zero contacts**, compared with 21 of 24 for the previous wider planner. One timeout remains. These results do not guarantee collision-free behavior elsewhere or establish consistent forced sideways yielding. This update changes planning, recovery and RViz displays; **the trained neural checkpoints are unchanged and no new GPT/API calls were made**.

## What the demonstration shows

`bash run-demo.sh` now runs four selected successful encounters with the final Stage 4 controller, recomputed live in F1TENTH Gym and displayed in RViz:

| Run | Confirmed pass | Previous wider planner |
| --- | ---: | --- |
| `fixed-case-014-response` | 18.44 simulated seconds | Body contact |
| `fixed-case-007-response` | 10.53 seconds | Timeout |
| `random-case-001-response` | 8.15 seconds | Timeout |
| `random-case-008-response` | 8.34 seconds | Timeout |

The first two are development regressions; the latter two are from the final held-out test. All four complete contact-free follow-through. They are selected highlights, not the overall success rate. Ego speed cap: 2.4 m/s; opponent cap: 2.1 or 2.2 m/s. Actual speeds vary through corners and avoidance.

## Final planner update and results

The planner searches sideways as far as the actual walls permit, while still checking the full vehicle footprint, curvature, steering and opponent clearance. It tries different transition distances and speed profiles. It preserves the existing wider approach while the opponent is ahead, then adds guarded checks during close encounters. A recently hidden opponent remains in the checks for up to 0.65 seconds. Predicted alternatives include observed motion, braking and return to the reference, so success does not require assuming the other car will yield.

When blocked at low speed, the car can pre-position its steering while stopped and try short, checked recovery arcs. Releasing a known side-padding LiDAR stop requires a motion certificate; recovery around a stopped opponent is capped at 0.25 m/s and measured nearby clearance must remain positive and nondecreasing. Front/unknown returns and wall checks remain active. See `hybrid_corridor_stage4.py`, `recovery_corridor_stage4.py`, and `robust_corridor_stage4.py` in `overtaking-models/`.

The final test used 24 new starting positions (seed 981114), 48 paired Gym runs, the same reactive opponent and sensor seeds for each pair, and a 40-second passing deadline plus follow-through. This set was not used to tune the final controller.

| Outcome | Previous wider planner | Final hybrid planner |
| --- | ---: | ---: |
| Passes | 21/24 | 23/24 |
| Contacts | 0 | 0 |
| Timeouts | 3 | 1 |

The remaining case 014 keeps moving behind the opponent; neither version passes in time. It is distinct from **development** case 014, whose collision was fixed. All eight original regression cases now pass without contact. On the broader 17-case development suite, passes improved from 11 to 15, contacts from one to zero, and timeouts from five to two. Earlier exploratory failures informed development and are not counted as independent final evidence.

A separate serial benchmark used the same 336 recorded ego/LiDAR inputs for both complete controllers: median latency fell from **26.55 to 14.82 ms (44.2%)**, and p95 from **154.83 to 121.40 ms (21.6%)**. Batching geometry/forecasts, sharing LiDAR calculations across speeds, removing duplicate motions and caching exact calculations reduced work. Timing-replay commands were not applied to physics; driving performance comes from the closed-loop tests. Tail latency still exceeds a 50 ms control budget. The 44 planner tests passed.

Full outcomes, frozen source/checkpoints, failed cases and verification records are included:

- [Final test report](overtaking-models/artifacts/hybrid_corridor_validation_20261114/REPORT.txt)
- [All final paired outcomes](overtaking-models/artifacts/hybrid_corridor_validation_20261114/summary.json)
- [Serial timing results](overtaking-models/artifacts/hybrid_corridor_speed_20261114/results.json)
- [Development outcomes](overtaking-models/artifacts/hybrid_corridor_dev_r1/summary.json)
- [Live RViz verification](overtaking-models/artifacts/hybrid_corridor_demo_20261114/ros_verification.json)

The opponent independently runs the saved Stage 2 model, its own LiDAR tracker, and its own avoidance controller. It is not commanded to slow down, move aside, or leave its racing line for the demonstration.

The track is the **imported AutoDRIVE default track**, simulated in F1TENTH Gym. This package should not be described as a verified implementation of the IROS `track-20260731` layout. Both cars normally follow the same locally lap-time-refined reference. That reference is a numerical approximation, not a proven globally fastest line.

## How the project develops through the stages

| Stage | Situation | Main idea |
| --- | --- | --- |
| 1: static obstacle | A parked obstacle blocks or approaches the reference line. | Choose LEFT, RIGHT, or WAIT. Passing left treats the obstacle as a right-side boundary; passing right does the reverse. Stop if no passage is available. |
| 2: moving obstacle | A slower lead car tracks the reference without avoidance. | Estimate its movement, plan around its future position, and merge back after clearing it. WAIT can mean follow, rather than stop. |
| 3: interactive opponent | A reactive Stage 2 opponent has the same top-speed cap. | Add PROBE and more aggressive passing settings; consider an inside approach while the other car can react. Equal top-speed caps do not mean identical cornering limits. |
| 4: faster interactive ego | The aggressive car is slightly faster than a reactive Stage 2 opponent. | Reuse Stage 3 maneuver weights, expand the path search, and experiment with predictions of the opponent's response to inside pressure. |

Stage 4 originally reused the trained Stage 3 network; it did not create an entirely new maneuver network. Earlier work added a **separately trained opponent-motion predictor**; the November 14 update keeps those weights and improves the planner. Historical Stage 2 data includes nonreactive opponents because that was its training task. The current Stage 4 comparison and selected demonstrations use reactive opponents exclusively during racing. The 0.4-second flying-start setup drives both cars along the reference to build actual sensor history; setup contacts still count.

This is not follow-the-gap with a new name. It is a hybrid system: a map-based path planner, learned maneuver preferences, a learned outcome estimate, an optional opponent-response forecast, and a local steering/speed controller.

## How a pass is planned

### 1. Estimate where the other car is

The ego uses its own pose, velocity, and 1,080-beam LiDAR scan. Returns consistent with the known track walls are removed. Remaining clusters are fitted to a vehicle-sized rectangle and tracked over time to estimate position, speed, lateral motion, and confidence.

The controller does not receive the opponent's true simulator pose, future actions, or commands. Simulator truth is available to the evaluator for training labels and scoring.

### 2. Construct several smooth paths

Describe a position using distance along the reference, `s`, and sideways displacement, `d`. A candidate path moves from the current offset to a passing offset, holds or adjusts that offset, then returns to the reference.

A basic transition uses the smooth polynomial

$$d(u)=d_0+(d_1-d_0)(10u^3-15u^4+6u^5),\qquad 0\leq u\leq1.$$

Here, `u` measures progress through the lane change. This avoids an abrupt lateral jump. The implementation also adjusts for heading, available track width, and the return to the racing line.

The planner varies **where the move begins, how quickly it moves sideways, its clearance, and when it merges back**. It searches a structured set of curves rather than randomly scattering points or asking GPT to draw a new racing line every frame. The earlier expanded base planner evaluates four nominal candidates and up to 28 alternatives when blocked. The final wall-corridor extension adds a structured lateral search with 0.25 m offset spacing and entry distances of 0.65, 1.2, 2.4 and 3.6 m, plus speed variants and checked recovery arcs. The number of candidates depends on available width and pruning; it is not capped at the earlier 28 alternatives.

### 3. Reject paths that fail the checks

For each path, the planner checks the car's footprint against walls, steering limits, speed-dependent turning limits, and predicted opponent footprints over time. A center point clearing the obstacle is insufficient: the entire rectangular car must fit.

Two useful relationships are

$$|\kappa|\leq\frac{\tan\delta_{\max}}{L},\qquad v^2|\kappa|\leq a_{\mathrm{lat,max}},$$

where `κ` is path curvature, `δ` is steering angle, `L` is wheelbase, and `v` is speed. A tighter bend requires more steering or less speed.

A feasible path passes the implemented checks **under the current estimates**. It is not guaranteed safe: perception, prediction, sampling, and execution can be wrong. If passing paths are rejected, the car can follow, hold its corridor, slow down, or stop.

### 4. Rank the remaining choices

Path cost roughly combines travel time, lateral deviation, and proximity to walls:

$$C\approx\sum_i\frac{\Delta\ell_i}{v_i}+\lambda_d|d|+\frac{\lambda_w}{\max(c_{\mathrm{wall}},\epsilon)}.$$

The planner reduces its search to representative maneuvers. A small neural network supplies LEFT/RIGHT/WAIT/PROBE preferences; a second learned head estimates observed continuation success. Feasibility masks and commitment rules can override the network to avoid impossible moves and repeated side switching.

This is not one universal reward function. It combines supervised learning, geometric costs, and explicit control rules. The outcome head is not a calibrated guarantee of success.

### 5. Execute a short part and replan

A pure-pursuit controller steers toward a point ahead on the selected path. Its basic steering relationship is

$$\delta=\arctan\left(\frac{2L\sin\alpha}{\ell_d}\right),$$

where `α` is the angle to the look-ahead point and `ℓ_d` is its distance. Curvature limits, acceleration limits, and a LiDAR braking check adjust the speed command.

Gym receives **steering angle and target speed**, not direct throttle. Physics advances every 0.01 seconds; decisions are scheduled every 0.05 simulated seconds. Only the immediate command is executed before planning again. A displayed path is an intention, not a prerecorded route the car must finish.

## What Stage 4 adds: predicting a reaction

Instead of only assuming that the other car keeps moving as before, the new predictor asks: **“If we take this candidate path, where might the opponent move?”**

Three small networks each use 16 inputs and hidden layers of 48 and 32 units. Inputs include the estimated opponent state, ego state, nearby track geometry, prediction horizon, and points on the proposed ego trajectory. Outputs correct constant-speed longitudinal progress and lateral offset:

$$(\hat r_s,\hat r_d)=f_\theta(\text{observed situation},\text{proposed ego motion},t).$$

The planner rolls the ego forward with steering/acceleration limits, forecasts the opponent over 1.5 seconds, and checks their predicted body separation. It compares the predicted outward response against a waiting proposal. The three networks' disagreement is a rough uncertainty indicator, not a certified safety bound.

For accepted response candidates, the implemented heuristic includes

$$J=C-0.8D_{\mathrm{outward}}+0.8U,$$

where `D_outward` is predicted additional outward displacement and `U` is ensemble spread. These coefficients are hand-tuned planning weights, not a learned physical law. A response candidate can override the ordinary maneuver choice. A forecast mismatch can trigger a temporary pressure cooldown.

Earlier experiments tested **early pressure** and **pressure only after closer overlap**. Those results remain below as history. The final HybridCorridorStage4 builds on that work with wall-bounded search, multiple motion hypotheses and checked close-encounter recovery. The current demo loads the final frozen controller and unchanged forecast checkpoint; it does not run the historical `learned_stage4.py` entry point directly.

A reactive opponent is not obliged to yield sideways. It may brake, hold position, misperceive the ego, or choose an unexpected maneuver. That is the main unresolved limitation of this experiment.

## Where GPT participates in training

GPT is an **offline adviser**, not the live driver or a source of guaranteed truth.

1. The simulator and planner produce a scenario summary: tracked motion, feasible maneuvers, estimated cost, clearance, and proposed controls.
2. `llm_teacher.py` can send that numerical summary to the GPT API. GPT chooses among the supplied maneuvers and returns structured advice: action, confidence, short rationale, steering hint, and target-speed hint.
3. The code rejects infeasible choices or out-of-bounds suggestions and saves accepted responses with their provenance. GPT does not generate the candidate curves or bypass collision checks.
4. Local networks learn maneuver preferences from training examples. The maneuver loss is cross-entropy, approximately `−Σ y(a) log p(a)`. GPT control suggestions are recorded advice; this does not mean the network directly learns raw throttle and steering.
5. Full simulator rollouts provide stronger evidence: imitation uses successful behavior, while outcome training includes observed PASS and FAIL results. GPT advice does not turn a failed rollout into a successful label.

The saved historical review log contains 179 accepted GPT-4.1-mini responses across earlier experiments; that is **not** 179 new reviews for each current model. The current Stage 2 and Stage 3 imitation datasets each include ten separately identified LLM-advice rows alongside rollout examples. Other labels come from geometric bootstrapping or simulator outcomes.

The latest opponent forecast was trained from recorded simulated motion using regression error, approximately `mean(||predicted displacement − observed displacement||²)`. **It used no new GPT calls.** All live demonstrations and final driving tests run locally without API calls. Forecast training on an executed ego trajectory does not guarantee accurate predictions for an untried proposal.

## Historical predictor results, including failures

The comparison used 24 random starting positions distributed around the same track, matching initial conditions between planners. Each trial allowed 40 seconds to confirm a pass, plus two seconds of contact-free follow-through.

| Planner | Passes | Trials with contact | Verified sideways-yield passes |
| --- | ---: | ---: | ---: |
| Previous Stage 4 response planner | 15/24 | 1 | 0 |
| Learned early-pressure variant | 14/24 | 0 | 0 |
| Learned overlap-gated variant | 15/24 | 1 | 0 |

A pass requires more than one metre of lead, return near the reference, and a sustained lead for one second before follow-through. Either car contacting a wall or the other car fails the trial. A sideways yield has a stricter behavioral definition: observed avoidance away from the ego, sustained opposite-side displacement, and an ongoing passing encounter. Braking or ordinary corner-tracking error is not counted as a sideways yield.

New predictor data collection proposed 160 initial states and executed 325 short maneuvers. Of 99 states with usable labels, 82 were assigned to training and 17 to validation, producing 903 training rows and 173 validation rows. Exploration had 91 contacts, including 18 during warmup; those failures are retained. An additional diagnostic had 40 post-warmup pressure maneuvers, 14 contacts, and four further warmup contacts. These probes were not used to fit the final predictor or counted in the 24-position race score.

Longitudinal prediction RMSE on branch validation improved from about 0.209 m to 0.116 m. That validation set also selected checkpoints; it is not an untouched final test. Live planning did **not** establish consistent forced yielding. The two policy variants reused the same 24 evaluation positions, and observations from development informed design choices. These results test new encounters on a known track, not unseen-track generalization.

The forecast ensemble alone took about 0.079 ms median and 0.097 ms at the 95th percentile for a batch of 31 horizon queries. That excludes perception, path generation, geometry checks, and control. The full controller is not guaranteed to meet a 50 ms wall-clock deadline. No real-vehicle qualification is claimed.

## Visual demonstration: Gym and RViz only

The shared visual entry point is the live RViz/Gym demonstration. Historical HTML pages, GIFs, screenshots, and website replays are not included in this presentation.

RViz colors: **white** reference, **green** ego and selected path, **orange** opponent, **blue** feasible candidates, **red** rejected candidates, **cyan** opponent forecasts when used, **amber** checked recovery motion. Status labels distinguish wide approach from guarded close encounters and identify steering-at-rest recovery. A lateral goal is a proposed target, not measured displacement. The selected path may still receive a reduced speed command from the braking checks.

One Gym instance owns physics and sensor observations. A local snapshot bridge sends display data to a separate ROS process; RViz visualizes it. RViz is not a second simulator and does not command the opponent.

On the existing configured workspace, run from `overtaking-models/`:

```bash
../gym-two-car-reactive/.venv/bin/python ros_visualization/launch.py \
  --stage 4 --speed .5 --demo-count 4 \
  --response-experiment artifacts/hybrid_corridor_demo_20261114
```

This loops the four selected cases at half playback speed. Close RViz to stop. The portable package launcher is `run-demo.sh`.

The current tested ROS environment targets Apple Silicon macOS, using the checked-in Pixi configuration and lockfile. The Gym interpreter is separate. A fresh teammate installation still needs dependency installation, the imported map, and the retained sibling-folder layout; the command above is not a claim of a tested clean installation on every platform. No API key is needed to run the demo.

## Models and data to retain in the team release

Paths below refer to the current `overtaking-models/` directory unless noted. Preserve the model/data/source manifests rather than copying only a neural checkpoint.

| Component | Existing source |
| --- | --- |
| Maneuver networks, normalization, early/mid/final snapshots and history | `artifacts/models/stage1*`, `stage2*`, `stage3*` |
| Outcome heads | `artifacts/models/stage2.value.npz`, `stage3.value.npz` |
| Forecast models and exact regression arrays | `artifacts/learned_reaction_v1/`, `artifacts/learned_reaction_v2/` |
| Original snapshot datasets and provenance | `artifacts/datasets/` |
| Current Stage 2 rollout training lineage | `artifacts/rollout_final_20261106_r2/` |
| Current Stage 3 rollout training and forecast-pretraining trajectories | `artifacts/rollout_final_20261107/` |
| GPT advice and provenance | `artifacts/llm_reviews_20261106.jsonl` and round-specific `reviews/` folders |
| New valid forecast collection | `artifacts/reactive_branches_v2/` |
| Pressure diagnostics, not final training data | `artifacts/reactive_pressure_probes/` |
| Development results, including rejected variants | `artifacts/learned_pressure_dev_v1/` through `learned_pressure_dev_v5/` |
| All 24-position tests and frozen code/checkpoints | `artifacts/learned_pressure_random_r1/`, `artifacts/learned_pressure_random_r2/` |
| Existing training handoff and historical archive | `exports/roboracer_training_data_20261107.zip`, `exports/roboracer_all_experiment_data_20261107.zip` |
| Controller, perception, geometry and training source | Project Python files, including `maneuver_core.py`, `expanded_planner.py`, `learned_stage4.py`, `learned_opponent.py`, `llm_teacher.py` and training scripts |
| ROS display and launch configuration | `ros_visualization/`, excluding installed environments and machine-specific runtime files |
| Actual simulator, adapter, map and dependency files | Sibling `../gym-two-car-reactive/`, including `f1tenth_gym/`, `run_two_cars.py`, and `imported_tracks/` |

The November 7 historical archive predates the newest experiments; use the pinned assets in `DATA_RELEASE.json` for the complete retained data. Retain superseded records for provenance but label them as historical. `reactive_branches_v1` was abandoned because of a sensor-update bug and must not be used for training.

The earlier ready-to-train export contains 1,334 Stage 1 imitation rows, 1,165 Stage 2 imitation rows and 13,360 Stage 2 outcome rows, plus 125 Stage 3 imitation rows and 14,572 Stage 3 outcome rows. Stage 2 and Stage 3 each collected 306 training rollouts from 102 initial encounters. These frame counts are correlated samples, not independent trials. Stage 3's exported imitation-validation set contains no successful rows; its failure-heavy outcome data cannot establish well-calibrated passing probabilities.

Most inputs are engineered features and saved trajectories, **not complete raw LiDAR recordings**. Include every retained training/evaluation artifact, but do not claim missing raw scans can be recovered. Keep train, validation, and evaluation-only data separate; split by encounter/episode, not random neighboring frames. Do not mix historical references or opponent versions without checking their manifests. The historical Stage 1 data uses an older reference geometry.

For reproduction, preserve each experiment's seeds, source/model hashes, reference identity, opponent settings, and split metadata. A changed controller requires new closed-loop Gym runs; replaying old trajectories cannot establish its performance.

## Repository and data delivery

The existing repository branch is **`Decision-Models`**, with exact capitalization. The requested parent location is `roboracer-sim-competitions/`. The standalone repository is named **`aggressive-overtaking-model`**.

The sibling-folder layout is preserved inside this package. F1TENTH Gym source and its MIT license are included; installed environments and the original Unity application are not. The package uses the saved map directly. No HTML/GIF/website visualization is shipped. Historical descriptions of those older tools may remain in archival documents.

The standalone repository is [AUnknownZ/aggressive-overtaking-model](https://github.com/AUnknownZ/aggressive-overtaking-model). The same code package is published under `roboracer-sim-competitions/aggressive-overtaking-model/` on the club repository's `Decision-Models` branch.

### Install and run

From the root of this package, with native Python 3.12 and Pixi available:

```bash
bash setup.sh
bash run-demo.sh
```

`setup.sh` installs the separate Gym environment and, when Pixi is available on PATH, the locked ROS environment. If Pixi was installed later, run `pixi install --locked` inside `overtaking-models/ros_visualization/`. The existing ROS lockfile supports Apple Silicon macOS; other platforms need a separate ROS dependency setup. `python verify_package.py` must use the installed Gym interpreter: `gym-two-car-reactive/.venv/bin/python verify_package.py`.

### Download all training and test data

The repository includes the final 24-position paired evaluation, 17-case development suite, four demos, timing inputs, frozen controller sources, unchanged demo weights and reference files. The historical **9,222-file archive** supplies all retained structured experiment data, historical versions, failed trials, exact training arrays, source snapshots, GPT advice, and the ready-to-train handoff. The November 14 supplement adds every new or changed retained structured artifact, including unsuccessful and interrupted development rounds. Both versioned assets together form the complete retained data delivery; the November 7 export names above document lineage.

[Download the historical training/data archive](https://github.com/AUnknownZ/aggressive-overtaking-model/releases/download/v2026.11.13/all-models-training-test-data-20261113.zip) (approximately 496 MB).

```bash
gym-two-car-reactive/.venv/bin/python download-data.py
```

This downloads both the historical archive and the [November 14 supplement](https://github.com/AUnknownZ/aggressive-overtaking-model/releases/download/v2026.11.14/final-planner-development-data-20261114.zip), verifies each SHA-256, and extracts them in that order into this package. No API key is needed. `DATA_RELEASE.json` pins the URL, size and checksum. `DATA_ARCHIVE_INDEX.json` lists original and exported hashes for every data file. Text metadata replaces local home paths with placeholders; executable source and NPZ checkpoint bytes are retained. The archive includes historical and invalidated records for provenance: do not train on evaluation data or the abandoned `reactive_branches_v1` collection.

`SOURCE_EXPORT_INDEX.json` records copied source origins, and `PACKAGE_MANIFEST.json` verifies the final packaged files, including portability changes. Runtime source files verified by the frozen demo manifest remain byte-identical. The portability changes only select the bundled saved map, discover Pixi on PATH, and avoid loading an unrelated historical scene list.

Credentials, `.env`, billing/key-check logs, installed environments, and unrelated workspace files are excluded. Refer to `THIRD_PARTY_NOTICES.md` for upstream component attribution and `VALIDATION.json` for the checks performed on the shared package.
