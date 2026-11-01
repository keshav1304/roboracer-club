# env_v2, the end to end env

Owner Rubi. A gym env for the fully RL (end to end) policy, rebuilt around the two things that broke our policies in AutoDRIVE, **friction** and **lag**. The P5 env (`../env/`) and deploy node are untouched.

```bash
cd pods/rl/env_v2
python test_v2.py          # physics vs AutoDRIVE, 19 checks, ~5 s. Must say "all checks passed"
python validate_v2.py      # the full table below, ~1 min
python calibrate.py        # open-loop manoeuvres for both vehicle models
```

Training is in `../train_v2/`, deployment in `../deploy/rl_deploy_node_v2.py`, RunPod in `../runpod/`.

## Why a new env

E2E_mu (end to end, trained on the `rl-pod-env` branch) learned in 1.5M steps and drove strangely in AutoDRIVE. The env it trained on started from an older copy of ours (commit e053c94) and had none of the AutoDRIVE physics. That made the task easy and the lessons wrong.

- **No sliding.** A 6 m/s² steering cap meant the car could never lose grip.
- **No lag.** Commands acted instantly.
- **Friction that did nothing useful.** `mu` is redrawn every step from Python's unseeded `random`. In the gym's ST model mu only scales a linear tyre, which has no grip limit. In the KS model mu is not used at all.
- **Asymmetric steering.** `s_min` stayed at -24° while `s_max` was 30°.
- **Whiteboard reward.** No time pressure, and a crash cost only 10.

Fed with synthetic straight corridors, E2E_mu steers almost bang-bang (full lock at 0.4 m off centre) at full throttle, and leaves the steering cap to do the rest. Add AutoDRIVE's 30 to 40 ms delay to that and you get oscillation.

## The physics, from AutoDRIVE's own source

We read AutoDRIVE's Unity source (`Tinker-Twins/AutoDRIVE`, branch `AutoDRIVE-Simulator`): `VehicleController.cs`, `TireFriction.cs`, the F1TENTH prefab and the `F1TENTH - Sim Racing` scene. Every tyre force there is (friction curve at the current slip) × (curve stiffness) × (load on the wheel). `TireFriction` scales the forward and the sideways stiffness by **one** coefficient. So one number decides how hard the car can accelerate, brake and turn, and when it slides. In `vehicle.py` that number is `mu`.

| What | Model in `vehicle.py` (twin) | Where the number comes from |
|---|---|---|
| Sideways tyre | per axle F_y = mu · sy · N · curve(slip angle); Unity spline, peak 1.0 at slip 0.01, falls to 0.5 by 0.1 | prefab `m_SidewaysFriction` |
| Front peak by wheel angle | × 1 up to 14°, falling to × 0.75 at 18° | our logs (AUTODRIVE_PHYSICS.md 1) |
| Drive | AWD (`driveType` CAWD). Wheels spin toward v* = 428/4 · 0.059 / 0.25 · throttle = **25.25 · throttle**. a = clip(8 · (v* − v), −mu · 2.9, +mu · 5.4) | scene MotorTorque 428, prefab wheel radius and damping |
| Throttle exactly 0 | **brakes all four wheels** (`brakeTorque = MotorTorque`). a = −mu · 4.1 | `VehicleController.Drive()` |
| Drag | −0.273 · v on everything | prefab Rigidbody `m_Drag` |
| Load transfer | N_front = m (g l_r − a_x h) / L, h 0.065 m | prefab COM |
| Geometry | l_f 0.1747, l_r 0.1553 (L = 0.33), LiDAR 0.268 m ahead of the rear axle | prefab wheel colliders, COM and LIDAR transforms |
| Steering | 30 ms delay, then `Steer()` line by line. Grows at 183°/s, but a smaller command than the current angle is reached in **one physics step** | `VehicleController.Steer()`, our logs |
| Coupling | F_y × sqrt(1 − k · u²), u = share of forward grip in use; nominal k 0, randomised 0 to 0.2 | unknown, the probe measures it |
| Balance | front peak × 0.95, rear × 1.25, yaw inertia 0.10 | fitted, see below |

The forward constants are fitted so the totals with drag match the logs. Balance needed a fit of its own. Every slide in our logs was the front running wide. AutoDRIVE holds 10.4 m/s² of demand for a moment, but slides when about 9.5 is held for half a second. Braking in corners (the grip shield) never spun the car. A grid search on `validate_v2.py` found that front peak 0.95, rear 1.25 and yaw inertia 0.10 reproduce all of these together. The whole neighbourhood (inertia 0.07 to 0.14, rear 1.2 to 1.35) also matches, so this is not a lucky point.

The longitudinal model is a servo toward v* that friction limits. That is what you get when the wheels are damped toward a target speed and the tyres can only push so hard. It explains why throttle 0.20 and 0.26 gave the same 4.7 m/s² ramp (the ramp is traction-limited, not throttle-limited). It also explains why `24.6 × throttle` is a speed and not a force.

### Does it match AutoDRIVE

`validate_v2.py`, nominal physics, from the spawn, flying laps.

| Driver | twin | empirical (`../env` model) | AutoDRIVE |
|---|---|---|---|
| pure pursuit x1.10 | clean, 13.23 s | clean, 13.18 s | clean, 13.21 s |
| pure pursuit x1.15 | clean, 12.69 s | clean, 12.60 s | clean 10/10, 12.67 s |
| pure pursuit x1.20 | **slides (front)** | clean | slid into the wall |
| pure pursuit x1.25 | slides | slides | slid into the wall |
| P5 + grip shield | clean, 11.51 s | clean, 11.42 s | clean 10/10, 11.52 s |

The twin is the first env that gets x1.20 right. The old env was optimistic right at the edge.

## Lag and timing

These are modelled in four layers. The randomisation is wide on purpose.

| Layer | Range per episode | Why |
|---|---|---|
| actuation delay | steering 20 to 50 ms, throttle 20 to 60 ms (nominal 30 / 40), at 0.25 ms resolution | logs, simulator clock |
| observation age | 0 to 20 ms | sensor and bridge age |
| control step | 50 ms ± 5 ms | the node's loop on the simulator clock |
| **late steps** | up to 2 % of steps last 100 to 200 ms, last command held | AutoDRIVE with its window in front |

Late steps are the one that matters. On ICRA with nominal physics, **2 % late steps crash pure pursuit x1.15 in 14 of 20 episodes**. 20 ms of observation age, or the ±5 ms jitter, crash none. That is why policies died with the AutoDRIVE window in front, and why the policy has to train with them.

The observation carries what is still in flight, so the problem stays Markov under delay. It includes the last steering command and the last throttle action, plus yaw rate and two accelerations from odometry, so the policy can feel grip and slides.

## Two physics models, mixed

70 % of episodes use the twin and 30 % use `MODEL_EMPIRICAL`, a port of `../env/race_env.py`'s KS + slide model. A policy has to drive both, so it cannot exploit a mistake only one of them makes. `train_v2/evaluate.py` gates on both.

## Tracks, so the policy learns driving and not ICRA

- **Real tracks.** `practice`, plus `icra`, which is held out and is the test. IROS 2026's track-20260731 is a second SLAM map of the practice loop, so it is not listed.
- **200 generated tracks** (`tracks.py`). Random hull, then edge midpoints pushed in and out, which gives S-bends, hairpins and chicanes. 28 to 75 m long, 1.3 to 2.4 m wide, radii down to about 0.5 m. Built from a fixed seed per track, so every machine gets the same ones.
- **Variants.** Every track is also mirrored and reversed. That doubles the corners in each direction and evens out left and right, which E2E_mu never had.
- **Starts.** Random start point, ±60 % of the half width off centre, ±0.2 rad of heading, 0 to 5 m/s.

## Action, observation, reward

- **Action.** `action[0]` is the share of the steering the envelope allows at this speed. The envelope keeps demand ≤ 0.95 × the nominal limit, and allows the full 30° below 2 m/s. It is the same function in the deploy node, so AutoDRIVE never receives a command the env did not allow. `action[1]` below −0.9 sends throttle 0 (brake). Above that, throttle goes from 0.005 to 0.32, so coasting and braking are two different choices, as in AutoDRIVE.
- **Observation (114).** 108 LiDAR bins, speed, last steer, last throttle action, yaw rate, two accelerations. LiDAR has 1 to 3 cm of noise and up to 1 % dropout, speed has 2 % noise.
- **Reward v4.** Per step: 10 × metres along the centre line, −1 per 50 ms of time, +50 per lap, −100 and the episode ends for a wall, a slide (> 50 ms past 4 × the peak slip), a spin (> 15° beyond kinematic body slip) or a stall (< 0.3 m/s for 1 s). There are also −3.5 × max(0, demand / limit − 0.85)², −0.1 × |steering change| and −0.02 × |throttle change|. **No centre-line term**, because that is what pins a car to the middle.
  - **Slide counts as a crash.** The gym is a little optimistic at the edge, so a slide in the gym is a wall in AutoDRIVE.
  - **Stall is a crash.** With gamma 0.995, standing still is otherwise as cheap as crashing, which is how preset S learned to stand still.

## Outside-inside-outside

Three pieces make the racing line something the policy can find. None of them caps it.

1. **Gamma 0.995** (10 s of look-ahead). A wide entry costs a little now and pays on the next straight.
2. **A teacher** (`teacher.py`) for every track. The line is the minimum-curvature line inside the walls, which is outside-inside-outside (a bounded least squares on second differences, 0.35 m from the walls). Speed is planned with the friction ellipse. The teacher drives it with pure pursuit and is clean on every track and both models (ICRA 12.49 s, faster than the club line at x1.15).
   - **BC warm start.** Training starts with behaviour cloning on the teacher, with action noise so recoveries are seen. The critic learns the teacher's returns.
   - **Potential shaping.** For the first 30 % of training a shaping term γΦ(s′) − Φ(s), Φ = −k (ey − ey_teacher)², pulls toward the teacher's line. It is potential based (Ng et al. 1999), so it does not change which policy is best, and it is turned down to 0.
3. **The reward is pure lap time at the end.** Where the real physics prefers a later apex or a wider arc than minimum curvature, PPO is free to find it. To see whether it did, compare flying laps against `teacher_lap` in `info`.

## Still to measure in AutoDRIVE (`../deploy/probe/`)

`autodrive_probe_node.py` runs short scripted tests in about 30 to 60 minutes. `analyze_probe.py` turns the logs into numbers for `vehicle.py`.

- **coupling.** Does full throttle or braking in a corner cost sideways grip? This sets `k_couple`.
- **brake vs coast.** Deceleration at throttle 0 vs throttle 0.01. This sets `a_brake` and `a_eb`.
- **steering unwind.** Is it instant, as `Steer()` reads? The yaw-rate response to a 0.15 → 0.03 step answers it.
- **acceleration ramp.** Checks `a_drive` and the servo gain.

Also TBC: the collision box (the gym's 0.58 × 0.31 m, kept for consistency) and that the compete build matches the public scene. The 25.25 vs 24.6 m/s per unit throttle agreement suggests it does.
