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
