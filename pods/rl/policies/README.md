# Other policies

`rl_policy_R.npz` is R, the one end to end policy that learned in our overnight sweep ([../STRATEGIES.md](../STRATEGIES.md)). The policy sets steering and throttle itself from LiDAR and speed, with no pure pursuit underneath. It is here so anyone can run it or compare against it without retraining. It is not a working driver for AutoDRIVE.

| | |
|---|---|
| Training | PPO, reward v3, 5M steps, seed 0 |
| Env it was trained on | an earlier version with a grip cap of 8.5 m/s² and none of the slide physics in [../env/AUTODRIVE_PHYSICS.md](../env/AUTODRIVE_PHYSICS.md) |
| On that env | 60 of 60 clean laps, 10.94 s average, best 10.65 s |
| On today's env | crashes in its first lap on nominal physics, and in 27 of 30 randomised episodes |
| In AutoDRIVE | never run |

On today's env it asks for more than 18° of steering in 17 of its first 100 steps and sometimes uses full lock. The AutoDRIVE car barely turns past 18°, and the env now models that, so it runs wide and hits the wall about 23 m into its first lap.

The file has the same format as `../deploy/rl_policy_P5.npz`, so `../train/evaluate.py` and the deploy node can run it. Its `residual` is 0, so the node runs it end to end.

```bash
cd pods/rl && python train/evaluate.py policies/rl_policy_R.npz
```

To try it in AutoDRIVE, copy it into `../deploy/` and run `python3 rl_deploy_node.py rl_policy_R.npz --laps 1 --tag R` as in [../deploy/README.md](../deploy/README.md). Add `--no-grip-shield` to run it exactly as trained. Expect it to hit the wall.

The file holds the actor network only (110 inputs, two hidden layers of 256, 2 outputs) as numpy arrays. That is enough to run it. Continuing its PPO training also needs the stable-baselines3 checkpoint, which is not in the repo. Ask Rubi for it.

## rl_policy_E2E_mu.npz

End to end PPO trained on Rishabh's env (branch `rl-pod-env`, commit 401e771). That env sets the gym's vehicle parameters to AutoDRIVE's values (mu 1.0, h 0.07354, steering limit 30°, a_max 5.2) and draws mu from 0.8 to 1.2 every step. It keeps that branch's 6 m/s² grip cap and the whiteboard reward.

| | |
|---|---|
| Training | PPO, 6 workers, whiteboard reward, checkpoint at 1.5M steps |
| In that env | 5 clean laps from each of 3 starts, flying laps 11.5 to 11.8 s |
| In AutoDRIVE | not run yet |

The same setup on the env from before Rishabh's two commits was still crashing at 1.8M steps.

Two things from training that the deploy node now repeats, read from this file.
- The env limits steering at speed to atan(6.0 × wheelbase / v²), so cornering stays under 6 m/s². The file carries `steer_lat_cap` 6.0, and the node clips steering the same way.
- The env's `s_min` is -0.4189 while `s_max` is 0.5236, so the policy never steered right past 24°. The file carries `steer_min` -0.8 and `steer_max` 1.0 in command units, and the node clips to them. This looks like a typo in the env and should be fixed before the next training.

Files without these fields (P5, R) drive exactly as before. Copy it into `../deploy/` (the folder mounted in the container) and run it like P5, with AutoDRIVE in the background.

```bash
python3 rl_deploy_node.py rl_policy_E2E_mu.npz --laps 3 --tag e2e_mu
```

To make it run on f1tenth_gym dev-humble (bdaec14), the env needed two local changes, not pushed. That gym takes vehicle parameters as a `VehicleParameters` inside the env config rather than a `params=` argument to `gym.make`, and `update_params` takes a `VehicleParameters`, not a dict.

## rl_policy_v2_s0_8M.npz

End to end PPO on env_v2 ([../env_v2/README.md](../env_v2/README.md)). LiDAR, speed and the last action go in, steering and throttle come out. There is no pure pursuit underneath and no race line in the input. It is the first end to end policy that has finished clean laps in AutoDRIVE.

| | |
|---|---|
| Training | `train_v2/train.py --preset e2e_v2 --seed 0 --gen 200 --holdout none --cfg '{"focus_track": "icra", "focus_share": 0.5}'`, 15 workers on a RunPod CPU pod, checkpoint at 8M steps |
| ICRA in training | half of all episodes run on ICRA as raced, the other half on 200 generated tracks plus practice, mirrored and reversed |
| Gates (`train_v2/evaluate.py`) | ICRA clean on both models, 11.11 s twin and 11.23 s empirical. Randomised physics 6/20 crashed with timing off and 4/20 with timing on (P5 6/20 and 13/20). Delay tolerance 70 ms. Unseen tracks 9/10 clean |
| In AutoDRIVE (Rubi's Mac) | 9 laps tried in three runs, 7 clean, fastest 10.871 s (P5 best 11.455). One touch in the hairpin. Once the car missed the hairpin exit and drove into the open area below it |

The Mac was heavily loaded during those runs (load average 130 to 218, simulator at 0.36 x real time, 11 to 13 control steps per run more than 100 ms late). P5's clean runs had 0 to 5. So the next test should be on a faster machine.

Checkpoints next to this one were much worse (6M crashed on ICRA, 10M and later failed the randomised gates). Training longer made it faster on ICRA and less robust, which is why 8M was picked.

To run it in AutoDRIVE, copy it into `../deploy/` and use the v2 node, with AutoDRIVE in the background. Grip shield is off by default because the policy was trained without it.

```bash
python3 tools/reset.py
python3 rl_deploy_node_v2.py rl_policy_v2_s0_8M.npz --laps 3 --tag v2_s0_8M
python3 rl_deploy_node_v2.py rl_policy_v2_s0_8M.npz --laps 10 --tag v2_s0_8M_10
```

The node prints late steps at the end. If there are more than a handful, the result says more about the machine than the policy. To watch it in the gym first, run `python3 train_v2/view_icra.py policies/rl_policy_v2_s0_8M.npz` from `pods/rl`.
