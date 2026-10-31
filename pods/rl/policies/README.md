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
