# Decision model (Decision Models pod)

Picks one of six behaviour states every tick, then hands control to the
matching executor (raceline pursuit, follow-the-gap or recovery). Built in
f1tenth_gym on Spielberg, reusing `../gym-two-car-reactive` (gym install,
venv, env factory, `FollowTheGap`).

| Step | What | Files |
|---|---|---|
| 1 | **Teacher**: rule-based state machine. The labeler, the baseline and the fallback | `teacher.py`, `common.py`, `run_teacher.py`, `compare.py` |
| 2 | **Training data**: the teacher drives randomized scenarios, every decision logged | `features.py`, `generate.py`, `dataset_stats.py` |
| 3 | **LightGBM** baseline: features → state | `train_lgbm.py`, `eval_closed_loop.py` |
| 4 | **RaceDecider-XS** neural net | `race_decider.py`, `train_nn.py` |
| 5 | **DAgger + counterfactual labels**: the net drives, the teacher relabels, sim branching finds better choices | `dagger.py`, `train_nn.py --extra` |
| – | **Our track** (`track-20260731`) in the gym | `tools/port_track.py`, `teacher.MAP_PRESETS` |

## Setup
```bash
git clone -b Decision-Models https://github.com/keshav1304/roboracer-club.git
cd roboracer-club/roboracer-sim-competitions/gym-two-car-reactive && ./setup.sh
cd ../decision-model
PY=../gym-two-car-reactive/.venv/bin/python
$PY -m pip install -r requirements.txt     # pandas, pyarrow, lightgbm, scikit-learn, torch
brew install libomp                        # LightGBM on macOS needs OpenMP
```

## Run
```bash
# teacher
$PY run_teacher.py --render                                # teacher vs a 3.5 m/s FTG car, live
$PY run_teacher.py                                         # headless, 125 s → teacher.png (path coloured by state)
$PY run_teacher.py --solo --seconds 300                    # lap times alone
$PY run_teacher.py --opp-vmax 0 --gap 30 --opp-on-raceline # parked car on our line → OBSTACLE
$PY compare.py                                             # teacher vs plain FTG, 12 scenarios × 60 s

# data + models
$PY generate.py --episodes 1000 --workers 9 --out data/v2  # ~8 min → data/v2 (git-ignored)
$PY dataset_stats.py data/v2
$PY train_lgbm.py data/v2                                  # ~20 s → models/lgbm_state.txt
$PY train_nn.py data/v2 --epochs 12 --batch 4096 --lr 3e-3 # ~16 min CPU → models/race_decider_xs.pt
$PY eval_closed_loop.py --model models/race_decider_xs.pt --model models/lgbm_state.txt   # driving test

# step 5
$PY dagger.py --model models/race_decider_xs.pt --episodes 400 --out data/dagger1   # ~24 min
$PY train_nn.py data/v2 --extra data/dagger1 --init models/race_decider_xs.pt --epochs 4 --batch 4096 --lr 1e-3 \
    --out models/race_decider_xs_dagger.pt
$PY eval_closed_loop.py --model models/race_decider_xs_dagger.pt --model models/race_decider_xs.pt

# our competition track (iros-2026 map + raceline → gym map folder; rerun after setup.sh)
$PY tools/port_track.py track-20260731
$PY run_teacher.py --map track-20260731 --solo --seconds 60
$PY run_teacher.py --map track-20260731 --opp-vmax 2 --gap 4 --seconds 40
```
The trained models are committed in `models/`, so `eval_closed_loop.py` runs
without retraining.

## States (teacher rules)
| State | Fires when | Executor |
|---|---|---|
| `CAR_STOPPED` | stationary > 0.5 s, < 0.3 m of progress in 2 s, or > 0.6 rad off the raceline heading below 2 m/s | three-point turn (forward first if there's room) until within 0.5 rad, reverse nudge out of wall deadlocks, then FTG relaunch |
| `OBSTACLE` | an object on our line within 6.9 m that's static | FTG around it until the pass is done |
| `OVERTAKE` | a car on our line within 6.9 m, we're ≥ 0.3 m/s faster, ≥ 0.61 m free beside it | FTG past it, committed until the pass is done |
| `SPEED_DOWN` | trailing (car ahead, can't pass) or above the speed profile | raceline pursuit + ForzaETH trailing law (1.5 m gap) |
| `CORNERING` | \|κ\| > 0.15 m⁻¹ within 0.8 s of travel | raceline pursuit at the profile speed |
| `SPEED_UP` | otherwise | raceline pursuit up to the profile (90% of it, capped at 6 m/s) |

- **Holds:** racing states hold for at least 0.3 s. CAR_STOPPED and OBSTACLE preempt immediately.
- **Braking cap:** every raceline command is capped by braking distance to anything on the steering arc.
- **Direction-aware speeds:** closing speed and time-to-collision use each car's speed along the track direction (speed × cos of its heading vs the raceline), so a car that's spun or turning around doesn't count as closing at full speed.
- **Opponent sensor** (`common.OpponentSensor`): ground truth plus ForzaETH-level tracking noise (0.17 m position, 0.49 m/s speed). The car is only seen inside the 270° LiDAR fan, within 10 m, and not behind a wall. It stands in for LiDAR detection, which isn't built yet.
- **Pass done:** the opponent is 0.6 m behind, or it was last seen alongside or behind and has dropped out of view (the blind wedge behind us). Then the passing line is held 0.2 s longer to open the gap. If it's lost while still ahead, the state is held up to 0.5 s.
- **Static:** moved < 0.5 m since first seen, judged after 0.6 s of sightings (the LiDAR only sees 1.6 s ahead at 6 m/s). Once static, it has to move 1.0 m to count as moving again.

## Training data
Each episode samples a scenario at a random point on the track. The teacher
drives at 100 Hz and every 5th tick becomes a row: **90 features → teacher
state (label) → action → outcomes**. Episode id = seed, so any episode can be
replayed.

| Scenario | Share | What |
|---|---|---|
| `race` | 30% | FTG opponent at 1.5–5.5 m/s, 2–15 m ahead |
| `close_race` | 10% | opponent at 5–6 m/s, 1.5–4 m ahead |
| `parked` | 20% | stopped car on the raceline 9–40 m ahead |
| `stopgo` | 15% | opponent braking to 0 for 1–3 s every 3–8 s |
| `solo` | 10% | no opponent |
| `hard_start` | 15% | ego starts 35–115° off the track heading |

- **Features** (all things the car can measure; never the teacher's internals):
  - `ego_*` (7): speed, raceline offset and heading error, profile speed, free width.
  - `trk_*` (10): curvature and profile speed 1–8 m ahead.
  - `scan_*` (54): LiDAR in 5° sectors.
  - `opp0_*` (10): the sensed opponent.
  - `hist_*` (3): stopped time, recent progress, opponent displacement.
  - `prev_*` (6): the previous decision.
- **Outcomes:** `y_collide_1s`, `y_progress_2s`, `y_min_gap_2s`, `y_pass_3s`. Filter on `y_horizon_ok`.
- **Split:** by episode (700 / 150 / 150 via `dataset_stats.split`), never by row.

**Dataset v2:** 1000 episodes, 727k rows (10 h of driving).

| State | Share |
|---|---|
| SPEED_UP | 66.5% |
| OVERTAKE | 19.6% |
| CORNERING | 11.3% |
| CAR_STOPPED | 1.2% |
| OBSTACLE | 1.0% |
| SPEED_DOWN | 0.4% |

The rare states are weighted up in training.

## Models
**RaceDecider-XS** (128k parameters, 0.085 ms per decision on 1 CPU thread):
- `ego/trk/hist/prev` → MLP
- LiDAR sectors → 1-D conv
- opponent(s) → DeepSets (masked sum, so more opponents need no retraining)
- → trunk → 4 heads: state, P(collision in 1 s), progress over 2 s, P(pass in 3 s)
- Trained with 15% of the `prev_*` inputs randomized, so it can't just copy its last decision.

**LightGBM**: 119 trees, 0.02 ms. Pass `--no-prev` for a memoryless variant.

**Offline** (v2 test split, 111k rows from 150 unseen episodes):
| | LightGBM | RaceDecider-XS |
|---|---|---|
| Accuracy | 99.91% | 99.55% |
| Macro-F1 | 0.996 | 0.973 |

LightGBM's score is inflated by `prev_*` (about 80% of its gain is "keep the
previous state"), so the driving test is the one that counts.

**Driving test** (`eval_closed_loop.py`): the model picks the state and the
teacher's executors drive, on the 150 held-out test scenarios:
| | Teacher rules | LightGBM | **RaceDecider-XS** |
|---|---|---|---|
| Crashes | 3 | 1 | **1** |
| Stuck (< 50 m) | 5 | 4 | **3** |
| Passes | 113 | 113 | **118** |
| Mean distance | 202.9 m | 202.0 m | **205.0 m** |
| Agrees with the rules | 100% | 60% | 94% |

- The differences are small: 1 vs 3 crashes over 150 episodes is suggestive, not significant.
- LightGBM drives safely but holds states longer than the rules would (60% agreement).
- Without `prev_*`, LightGBM had 5 crashes. Memory matters.

**Teacher on the step-1 grid** (vs plain FTG, 12 scenarios):
| | FTG | Teacher |
|---|---|---|
| Clean passes | 12/12 | 12/12 |
| Mean first pass | 7.64 s | **3.98 s** |
| Closest car gap | 0.14 m | **0.22 m** |
| Solo laps | 59.3–59.7 s | **58.6–59.0 s** |

## Step 5: DAgger + counterfactual labels
- **DAgger:** the neural net drives 400 fresh scenarios (seeds 1000–1399, none from the test split). Every row is labelled with what the teacher's rules, including their dwell, would pick *in the state the net got itself into*. The net learns to recover from its own mistakes.
- **Counterfactuals:** at moments with the opponent in view (whenever the net disagrees with the rules, plus a random 15%), the whole sim is snapshotted (gym sim, controllers, sensor RNG; a restore reproduces the run exactly). Each candidate state is forced for 1.5 s, then the rules drive for 1.5 s.
  - **Score:** raceline progress, −50 for a collision, −20 per metre of contact under 0.10 m.
  - **Override:** if a candidate beats the teacher's pick by > 0.3 m, it becomes the label, weighted ×2 in training.
- **Run 1:** 290,779 rows, 8,030 branch points, **2,957 labels overridden** (median gain 1.3 m per 3 s). 94.7% agreement with the rules while the net drove; 9 of 400 episodes crashed.
- **What the sim says the teacher gets wrong:**
  - OVERTAKE → SPEED_UP 1,238×: the opponent is off our line, so staying on the raceline is faster than FTG around it.
  - OVERTAKE → SPEED_DOWN 506×: wait for a better moment.
  - SPEED_UP → OVERTAKE 438×: go for it sooner.
  - OVERTAKE → OBSTACLE 211×.

**Driving test after one DAgger round** (150 test scenarios; teacher and step-4 net re-run with heading-aware closing speed):
| | Teacher rules | RaceDecider-XS + DAgger | RaceDecider-XS (step 4) |
|---|---|---|---|
| Crashes | 3 | 2 | **1** |
| Stuck (< 50 m) | 5 | 3 | **2** |
| Passes | 116 | 116 | 115 |
| Mean distance | 203.5 m | 204.1 m | **205.8 m** |
| Agrees with the rules | 100% | 97.3% | 94.1% |

**One round didn't improve on step 4.** Both nets beat the teacher, but the
gap between them is within noise, so `race_decider_xs.pt` stays the main
model and `race_decider_xs_dagger.pt` is kept for comparison. Next:
- more rounds (the DAgger model drives the next batch)
- longer branches (3 s + 3 s)
- evaluating on more than 150 episodes so differences this small can be told apart

## ICRA 2026 track (gym), in progress
`tools/port_track.py icra2026-comp` converts `icra-2026/autodrive_devkit/comp_track` + `comp_waypoints.csv`:
- The raceline file has 7 negative speeds; the port interpolates over them.
- Maps are padded to a square because the gym renderer swaps width/height.
- The track is 60.9 m, 1.6–5 m wide.

`tools/icra_pipeline.sh` runs the whole thing (data → LightGBM + net → 2 DAgger rounds → 1000-race test). It was stopped during DAgger round 1, so these stages are done:

| Stage | Result |
|---|---|
| Teacher data, 1000 races | 726k rows; **23 crash episodes (2.3%)**, down from 126 before the reverse-into-wall fix; 793 passes |
| LightGBM | `models/lgbm_state_icra.txt`, test accuracy 99.89% |
| Net (ICRA + Spielberg, fine-tuned from the Spielberg net) | `models/race_decider_xs_icra.pt`, test accuracy 98.68%, collision AUC 0.93 |
| DAgger round 1 | 296 / 300 episodes saved in `data/icra_dag1` (local only) |

**Resume** (it skips finished episodes): rerun `tools/icra_pipeline.sh` from stage 3a, or run its remaining commands by hand.

## Our track: track-20260731 (gym)
`tools/port_track.py` converts the iros-2026 SLAM map and raceline into a gym map:
- Unknown cells become walls.
- SLAM specks under 0.5 m² are removed.
- The centerline is the midpoint between the walls along the raceline normals.
- Width 1.1–2.5 m, median 1.4 m.

`teacher.params_for("track-20260731")` scales the teacher for a 30.5 m loop: v_max 5, horizon 3 m, lookahead 0.6–2.5 m, 0.10 m wall margin. The Spielberg defaults are unchanged.

| Test (40–60 s) | Result |
|---|---|
| Solo | laps 10.40 / 10.19 / 10.19 / 10.18 / 10.19 s, no collisions (raceline estimate 9.77 s) |
| Opponent 1.5–2.5 m/s | 1–2 passes, no collisions, closest gap 0.10–0.16 m |
| Opponent 3 m/s | no pass in 40 s (stuck in OVERTAKE) |
| Parked car 12 m ahead | clean, 4 passes |
| Parked car 8 m ahead | **crash** at 6 s |

- **Slow passes:** passing is slow on 1.4 m corridors (FTG still has Spielberg tuning).
- **Models not retrained:** the learned models are trained on Spielberg only. Next is generating data on this track (`generate.py` still hard-codes Spielberg).
- **Not in AutoDRIVE yet:** that needs a ROS node like `iros-2026/.../pp_backup_node.py` calling `Teacher.plan`.

## Design notes (what broke and why it's built this way)
- **The shipped raceline hugs walls** (about 5 cm clearance). `TrackModel` pulls it to 0.3 m and keeps the speed profile.
- **Pure pursuit understeers at 6 m/s.** Added lateral + heading feedback, capped at 0.2 rad. Narrow gain window: re-tune after any dynamics change.
- **Profile scaling:** `aggression` 0.9 is applied before the 6 m/s cap, otherwise straights slow down too.
- **Off-line rejoin:** the lookahead grows with the lateral offset, so rejoining isn't a swerve into the wall.
- **Static by position, not speed:** sensed speed noise is bigger than any speed threshold.
- **Recovery:** spawning nose-at-wall and blind reversing caused most crashes in the first data run (fixed: misaligned → CAR_STOPPED, forward-first turn, rear-beam guard, reverse nudge).

## Known limitations
- About 2% of episodes crash (hard starts, FTG side-swipes while passing). About 2% get stuck: a car wedged on a wall recovers, then the gentle raceline rejoin drives it back.
- Learned deciders touch the other car at least once in 150 episodes (closest gap 0.00 m).
- **Parked-car squeeze loop:** in ~0.4% of episodes the car noses into a parked car while squeezing past. It reverses, picks the same gap, and repeats for up to 30 s (the "reversing" seen in runs).
- **Scope:** one opponent; perception is ground truth plus noise (no LiDAR detection yet); learned models are Spielberg-only; nothing runs in AutoDRIVE yet.
