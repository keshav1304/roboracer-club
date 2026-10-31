# Two-car reactive baseline: f1tenth_gym, Spielberg (Decision Models pod)

Sim-only check that a **reactive** stack can drive laps with a second car on
track and **overtake** it. Two cars run follow-the-gap (`reactive.py`) on the
Spielberg map (343 m) in [f1tenth_gym](https://github.com/f1tenth/f1tenth_gym)
(branch `v1.0.0`). The ego (agent_0) is capped at 6 m/s and the opponent
(agent_1, starting ahead) is slower. This is the baseline the decision models
have to beat.

## Get the code
```bash
git clone -b Decision-Models https://github.com/keshav1304/roboracer-club.git
cd roboracer-club/roboracer-sim-competitions/gym-two-car-reactive
```
Already cloned? Run `git pull` and `cd` into this folder. Every command
below is run from here.

## Setup (once, safe to rerun)
Needs git and Python 3.12. On Apple Silicon use a **native arm64** Python
(python.org or Homebrew); an x86_64/Rosetta Python breaks numba.
```bash
./setup.sh
```
This clones f1tenth_gym into `./f1tenth_gym`, creates `./.venv`, installs
everything and downloads Spielberg. Both folders are git-ignored. The first
`import cv2` / numba compile takes a while, so don't Ctrl-C it.

## Run
```bash
.venv/bin/python run_two_cars.py --render          # watch it live (60 s)
.venv/bin/python run_two_cars.py --seconds 125     # headless, writes two_cars.png
.venv/bin/python run_two_cars.py --opp-vmax 3 --gap 4 --seconds 90
```

## Results
- **125 s race** (opponent 3.5 m/s, starting 5 m ahead): **1 pass at t ≈ 3.7 s**, no collisions. Ego laps 60.46 / 59.40 s.
- **Scenario grid** (opponent 2/3/4/5 m/s × gap 3/5/8 m, `../decision-model/compare.py`): **12/12 clean passes**, closest body-to-body gap **0.14 m**.
- About 27× real time with two cars, one process.

## Files
| File | What it does |
|---|---|
| `reactive.py` | `FollowTheGap`: smoothing → disparity extender → safety bubble → widest gap (±75°) → pure pursuit to the deepest point → side-clearance push → speed capped by braking distance along the steering arc → if blocked, try 15 steering angles → hold, then reverse only after 0.3 s with the rear sides clear |
| `run_two_cars.py` | Two-car race: lead changes (0.6 m margin vs flicker), collisions, laps, trajectory plot |
| `setup.sh` | One-time install |

## How the controller got here
1. **Post-pass wall hit:** FTG steered away from a parallel wall far too gently → added side-clearance push.
2. **Nose-in stall behind a very slow car:** it aimed at a gap about 90° to the side → limited targets to ±75°.
3. **Rubbing side-by-side (4–8 cm):** braking looked at a ±10° cone and missed a car slightly off-centre → braking now checks a car-width lane, the margin was doubled, and the side push is stronger.
4. **Deadlock in the hairpin:** a straight lane saw the outer wall as "in the way" → the lane follows the steering arc, a blocked arc tries other steering angles, and reversing is gated on a 0.3 s wait and clear rear sides (the LiDAR is blind straight back).

## f1tenth_gym v1.0.0 quirks handled here
- The downloaded map ships `Spielberg.yaml` but the loader wants `Spielberg_map.yaml`; `setup.sh` copies it.
- `observation_config={"type": "features"}` needs an explicit feature list.
- The episode only ends on an ego crash, and the env lap counter stops at 2 laps.
- Action per agent = `[steering_angle, speed]`.
