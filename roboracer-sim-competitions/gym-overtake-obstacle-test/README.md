# Gym overtake + obstacle test (Decision Models pod)

A sim-only test harness that checks whether a **reactive** driving stack can
**overtake** a slower car and **avoid static obstacles**. It uses
[f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) (branch `v1.0.0`) on
**PortoWide**: the AutoDRIVE Porto layout scaled 1.8× (55.5 m lap) and widened
to a constant 2.4 m, with three 0.40 × 0.35 m boxes on the track (left, right
and centre).

Both cars run the same follow-the-gap controller (`reactive.py`). The ego is
capped at 6 m/s and the opponent at 1.5–2.5 m/s, so the ego laps it repeatedly.
This is the baseline the decision models have to beat. It is not a racing
controller: there is no raceline and no planning.

## Setup (once)
Needs git and Python 3.12. On Apple Silicon the Python must be a **native arm64**
build (python.org or Homebrew); an x86_64/Rosetta Python breaks numba.
```bash
./setup.sh
```
This clones f1tenth_gym into `./f1tenth_gym`, creates `./.venv`, installs
everything, and builds the PortoWide map. Both folders are git-ignored.

## Run
```bash
.venv/bin/python check.py            # pass/fail: 3 opponent speeds × 10 laps (~1 min)
.venv/bin/python race.py --render    # watch a 10-lap race live
.venv/bin/python race.py --opp-vmax 2.5 --laps 10 --plot race.png
```

`check.py` passes a run only if, over 10 laps:
- neither car collides (car–car, wall or box)
- the ego completes every lap
- the ego makes **≥ 6 overtakes**
- neither car touches a box (body-to-box distance > 0)

Clearances under 0.10 m are reported as `WARN`.

## Current results (reactive baseline)
| Opp v_max | Laps | Overtakes | Closest car gap | Ego–box | Opp–box | Result |
|---|---|---|---|---|---|---|
| 1.5 m/s | 10 | 7 | 0.20 m | 0.05 m | 0.32 m | PASS (warn) |
| 2.0 m/s | 10 | 7 | 0.14 m | 0.08 m | 0.29 m | PASS (warn) |
| 2.5 m/s | 10 | 6 | 0.27 m | 0.09 m | 0.17 m | PASS (warn) |

Ego laps average about 11.9 s (4.6 m/s mean). The sim runs at about 24× real time.

**Known limitation:** the ego clears the boxes with only 5–9 cm. Follow-the-gap
looks only ±75° ahead, so once a box is beside the car it steers back toward the
racing direction while its rear corner is still alongside. Bigger clearance
margins and hard "don't steer toward a close side" rules made it worse (crashes
and stalls). Fixing this properly is a job for the decision layer: an OBSTACLE
state that tracks the box until it's behind, then rejoins.

## Files
| File | What it does |
|---|---|
| `reactive.py` | `FollowTheGap`: smoothing → disparity extender → safety bubble → widest gap (±75°) → pure pursuit to the deepest point → side-clearance push → speed capped by braking distance along the steering arc → if blocked, try 15 steering angles → hold / reverse recovery (checks the rear sides first) |
| `race.py` | Two-car race: lap times, each overtake, collisions, car–car and box clearance, trajectory plot |
| `check.py` | Pass/fail suite (exit code 0 = pass) |
| `build_track.py` | Builds PortoWide from `tracks/porto_centerline.csv`: `--scale`, `--width`, `--no-obstacles` |
| `tracks/porto_centerline.csv` | Porto centerline extracted from the AutoDRIVE IROS 2026 build (Unity x/z → gym x/y) |
| `setup.sh` | One-time install |

## f1tenth_gym v1.0.0 quirks handled here
- The map loader wants `<name>_map.yaml`; `build_track.py` writes it.
- `observation_config={"type": "features"}` needs an explicit feature list.
- The env lap counter stops at 2 laps and the episode only ends on an ego crash, so laps and overtakes are counted here from centerline progress.
- The gym only flags car–car overlap, so clearances are measured with shapely on the 0.58 × 0.31 m car rectangles.
