# Trained models

Stable-Baselines3 PPO models (`.zip`). Only these final models are committed. New training saves to
`models/<run>/`, with a checkpoint every 50,000 steps; git ignores those.

| Model | Simulator and track | Reward | Steps | Result |
| --- | --- | --- | --- | --- |
| `ppo_20261107_182816/final.zip` | Oct 24 build, track `none` | basic | 2.34M | 7–8 laps per episode, best lap 10.8 s |
| `ppo_20261108_054349/final.zip` | ICRA build, track `icra` with the race line | race_line | 1.04M | No full lap yet; average episode about 7 s |
| `ppo_20261106_062208/final.zip` | ICRA build, track `icra` | basic, with the stall rule | 0.90M | 1 lap at best |

## Running a model

- **Test:** run `./launch.bash`, set the track (and race line) to match the table, then choose
  option 4 or 5 and type the model's path.
- **Keep training:** choose option 3 with the same settings.

For the race-line model, set the race line to
`tracks/icra/track3_minimum_curvature_world.csv` (menu option `r`). For the Oct 24 model, set
`SIM_EXE_NONE` in `launch.local.bash` and choose track `none`.

For how each model was trained, see [docs/training-history.md](../docs/training-history.md).
