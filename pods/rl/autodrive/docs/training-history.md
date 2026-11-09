# Training history

Each run's name is its start time in UTC: `ppo_YYYYMMDD_HHMMSS`. Continuing from a saved model
starts a new run, so one line of training can span several runs.

Only the three models in [models/](../models/README.md) are in the repository. Everything else is in
`archive/` on the development PC:

- `archive/models/`: the other runs' models and checkpoints
- `archive/runs/`: their TensorBoard logs
- `archive/logs/`: simulator logs and crash diagnosis CSVs

Steps are the run's final step count. Laps are the most laps completed in one training episode.

## 1. ICRA track, first reward (5–6 Nov)

Reward: progress, lap, collision, speed, a centre-line term and a steer-rate term, with 75% random
starts.

| Run | Continued from | Steps | Laps | Notes |
| --- | --- | --- | --- | --- |
| `ppo_20261105_014456` | new | 233k | 0 | 3 simulators |
| `ppo_20261105_051707` | `014456` | 779k | 0 | 5 simulators |
| `ppo_20261105_070810` | `051707` | 1.62M | 0 | Overnight run. Stopped without a final model, so its last checkpoint was used |
| `ppo_20261105_132647` | `070810` | 1.68M | 0 | |
| `ppo_20261105_135745` | `132647` | 1.80M | 0 | Reward flat at about 360. Crashed at the 33 m and 39 m hairpins |

## 2. ICRA track, no centre-line term (5–6 Nov)

The centre-line term was removed, leaving progress, lap, collision, speed and steer rate.

| Run | Continued from | Steps | Laps | Notes |
| --- | --- | --- | --- | --- |
| `ppo_20261105_193415` | new | 25k | 0 | |
| `ppo_20261105_194211` | `193415` | 137k | 0 | |
| `ppo_20261105_203603` | `194211` checkpoint at 126k | 648k | 0 | |

## 3. ICRA track, basic reward, stopping problem (6 Nov)

The steer-rate term was removed, which left progress, lap, collision and speed.

| Run | Continued from | Steps | Laps | Notes |
| --- | --- | --- | --- | --- |
| `ppo_20261106_013220` | new | 823k | 0 | Learned to stop and avoid crashing: long episodes, no progress |

## 4. ICRA track, stall rule and random starts (6 Nov)

The stall rule was added: if the car stops making progress, the episode ends with the collision
penalty.

| Run | Continued from | Steps | Laps | Notes |
| --- | --- | --- | --- | --- |
| `ppo_20261106_062208` | new | 897k | **1** | 6 simulators. **Kept in `models/`** |

## 5. ICRA track, wall and wrong-way terms (7 Nov, abandoned)

| Run | Continued from | Steps | Laps | Notes |
| --- | --- | --- | --- | --- |
| `ppo_20261107_162218` | new | 8k | 0 | Wall-distance and wrong-way terms. Stopped after one update; the terms were removed again |

## 6. Oct 24 build, track `none` (7 Nov)

This build has a different, narrow-corridor track with no centre line, so progress is the forward
distance driven. The reward is basic: progress, lap, collision and speed.

| Run | Continued from | Steps | Laps | Notes |
| --- | --- | --- | --- | --- |
| `ppo_20261107_170651` | `ppo_20261105_135745` | 2.08M | 8 | Started from the ICRA model |
| `ppo_20261107_182816` | `170651` | 2.34M | **8** | 6 simulators. Best lap 10.8 s, average 11.3 s. **Kept in `models/`** |

## 7. ICRA track, race-line reward (7–8 Nov)

The reward follows the minimum-curvature race line
[tracks/icra/track3_minimum_curvature_world.csv](../tracks/icra/track3_minimum_curvature_world.csv).
See the `race_line` preset in [reward.md](reward.md).

Two first attempts used random starts and ran at about 10 steps per second. They were stopped
early, and their models were not kept.

| Run | Continued from | Steps | Laps | Notes |
| --- | --- | --- | --- | --- |
| `ppo_20261107_194616` | new | 20k | 0 | No random starts, an update every 512 steps per simulator, 57 steps/s with 4 simulators |
| `ppo_20261108_054349` | `194616` | 1.04M | 0 | 5 simulators, 83 steps/s. Episode length 74 → 146 steps; reward 104 → 498. **Kept in `models/`** |

At 1.04M steps every episode still ended in a crash, but each step of training was getting further
around the track.
