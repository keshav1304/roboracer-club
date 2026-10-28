# Train, the setup behind P5

Owner Nathaniel. These are the scripts from the runs that produced the deployed policy, put here as a reference to build on.

| File | What it does |
|---|---|
| `train.py` | PPO (stable-baselines3) on `env/race_env.py`, 8 parallel envs, checkpoints every 100k steps. `--preset p5` is the P5 setup |
| `evaluate.py` | drives a checkpoint or an exported policy the way it will run in AutoDRIVE, on nominal physics (best fit, no randomisation), tight grip (the slippery end of the training range) and randomised physics |
| `export_policy.py` | turns a checkpoint into the `.npz` the deploy node reads, and checks the numpy copy against SB3 |

## Setup

```bash
git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym
pip install -e f1tenth_gym
pip install "stable-baselines3>=2.3" gymnasium tensorboard
cd pods/rl/env && python test_race_env.py         # should end with "all checks passed"
```

We used Python 3.13, f1tenth_gym dev-humble at commit bdaec14, gymnasium 0.29, stable-baselines3 2.9 and torch 2.14 on CPU.

## The P5 setup

```bash
cd pods/rl
python train/train.py --preset p5 --steps 8000000 --seed 0 --name p5_s0
python train/evaluate.py train/runs/p5_s0/ckpt_4499928_steps.zip
python train/export_policy.py train/runs/p5_s0/ckpt_4499928_steps.zip deploy/rl_policy_mine.npz
```

This uses the P5 settings but does not rebuild P5 exactly (see below), and `ckpt_4499928` is only an example name. Pick your own checkpoint with the rules further down.

`--preset p5` means
- residual mode. Pure pursuit on the club line drives, and the policy adds a correction scaled by 0.4.
- `steer_residual 0`, so the policy only touches the throttle. Steering stays pure pursuit's.
- base speeds x1.15, so an untrained policy already drives pure pursuit's 12.67 s laps.
- reward v3, 10 per metre of progress, minus 1 per step, plus 50 per lap, minus 100 on a crash, minus 0.1 times the change in steering. The time penalty makes faster laps pay, and the larger crash penalty makes risky driving cost more. The env's default is the pod's whiteboard reward instead (10 per metre, minus 0.05 times the distance from the centre line each step, plus 50 per lap, minus 10 on a crash, minus 0.1 times the change in steering), and the preset swaps v3 in.

The deployed P5 was trained in two stages. 4.5M steps on an earlier version of this env (one slide threshold instead of the wheel-angle limit, not in the repo), then `--resume` for 4M more on this one. Two seeds trained from scratch on this env for 8M steps reached 11.26 s and 11.33 s in the gym, faster than P5's 11.42. In AutoDRIVE the faster one crashed twice in lap 1. It turned out less tolerant of late control steps. The other one slides at nominal grip, so we never deployed it. The gym's lap time is not enough to pick a policy.

## Picking a checkpoint

- In our residual runs, reward and eval lap time were flat from about 3M steps. More steps did not help.
- Run `evaluate.py`. As a reference, P5 itself gives a clean 11.42 s on nominal physics without sliding, crashes on tight grip after export, and crashes in 8 of 30 randomised episodes (pure pursuit x1.15 crashes in 4 of 30 and never crashed in AutoDRIVE). So the randomised band is harsher than AutoDRIVE, and tight grip is harsher still. Prefer a checkpoint that stays clean without sliding on nominal physics. Sliding there means the policy is living on the edge of the grip model, and AutoDRIVE's edge is a little lower than the gym's (see `../env/AUTODRIVE_PHYSICS.md`).
- The gym was 0.04 to 0.12 s faster than AutoDRIVE for every policy we deployed.
- Then test in AutoDRIVE, 3 laps first, AutoDRIVE in the background (`../deploy/README.md`).

## What did not work

- End to end, with the policy setting both steering and throttle. It learned on one seed out of three, and that one collapsed after about 5.4M steps when we trained it longer.
- Letting the policy add steering on top of pure pursuit, even limited to ±3°. In AutoDRIVE the total wheel angle passed 18° in the hairpin and the front slid.
- A fixed cap on cornering in the gym. RL drives right up to any cap, and AutoDRIVE's real limit depends on wheel angle and time, so a cap that looked safe still crashed.
