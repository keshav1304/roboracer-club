# RL pod

On the ICRA 2026 track in AutoDRIVE, our RL policy runs 10 laps with no wall contact at 11.52 s average. That is 1.15 s (9%) a lap faster than pure pursuit on the same race line. Results, video and how we got there are in [RESULTS.md](RESULTS.md).

![Top-down replay of the 10-lap runs, RL P5 in red, RL P4 in blue, pure pursuit in green](media/autodrive_P5_vs_P4_vs_pp.png)

## Read this first

The gym and AutoDRIVE disagree most at the grip limit, and RL drives at the limit. Before changing the env, the reward or the deploy node, read [env/AUTODRIVE_PHYSICS.md](env/AUTODRIVE_PHYSICS.md). It lists every AutoDRIVE limit that crashed one of our policies, with the evidence.

## What is where

| Folder | Owner | What is in it |
|---|---|---|
| [env/](env/) | Rubi | `race_env.py`, the gym env with AutoDRIVE's physics, its tests, and `validate_vs_autodrive.py`, which checks the gym against real AutoDRIVE runs |
| [train/](train/) | Nathaniel | `train.py`, `evaluate.py` and `export_policy.py` from the runs behind P5, as a starting point |
| [deploy/](deploy/) | Rishabh | the P5 policy file and the ROS 2 node that ran it in AutoDRIVE, with the pitfalls we hit |
| [media/](media/) | | replay of the AutoDRIVE runs |

## Quick start

```bash
cd pods/rl/env && python test_race_env.py          # the env works and matches its contract
python validate_vs_autodrive.py                    # gym vs AutoDRIVE, side by side
cd .. && python train/evaluate.py deploy/rl_policy_P5.npz
```

Setup for the gym and training is in [train/README.md](train/README.md). Running the policy in AutoDRIVE is in [deploy/README.md](deploy/README.md).
