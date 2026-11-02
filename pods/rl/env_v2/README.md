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
| Balance | front peak × 0.921, rear × 1.289, yaw inertia 0.115, centre of mass 2 cm forward (l_f 0.156) | refit 2026-11-01, see Refit below |

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

## Refit, 2026-11-01

`rl_policy_v2_s0_8M` spun in AutoDRIVE in the corner before the start straight. The logs show why. The policy steered right at its envelope there (95 % of the nominal limit, 9.3 m/s² of demand). With the wheel snapped to that angle, AutoDRIVE gave only 72 to 84 % of the demand, so the front was past the peak of Unity's sideways curve and the car ran wide. When the policy then eased off, the front gripped again and the car rotated more than the wheel asked for (111 %), lateral acceleration jumped to 10 to 16 m/s² and the rear let go. The gym gave 87 to 93 % at the envelope and about 95 % after, so the policy had learned that riding the envelope is safe. Smooth pure pursuit steering never saturates the front, which is why the pure pursuit probe (`../deploy/probe/oversteer_probe.py`, 18 runs through that corner) never slid.

| Achieved ÷ demanded lateral acceleration | At the envelope | Just after easing off |
|---|---|---|
| AutoDRIVE, s 28 to 36 | 0.83 | 1.11 |
| AutoDRIVE, rest of the lap | 0.72 | 0.91 |
| twin before the refit, s 28 to 36 | 0.91 | 0.95 |

Two changes came out of it.

1. **Envelope 0.95 → 0.82** (`contract.ENVELOPE_FRACTION`). The policy can now ask for about 8.0 m/s², which is what AutoDRIVE holds when the wheel is snapped there. The envelope is exported in every policy file, so a policy trained with 0.95 still deploys with 0.95.
2. **Twin refit by open-loop replay.** 16,194 half-second windows from AutoDRIVE logs (the v2 policy, P5 and P4 10-lap runs, pure pursuit and the probe, pure pursuit x1.20 and x1.25): start the twin in the logged state, replay the logged commands with the real delays, compare the heading after 0.5 s. A random search kept every `validate_v2.py` outcome and lowered the median heading error from 1.45° to 1.15° (v2 policy 2.91° → 2.47°, P5/P4 1.23° → 1.07°, pure pursuit 0.46° → 0.32°). The top candidates agree on understeer gain about 1.0 (was 0.955) and the centre of mass about 2 cm further forward. `test_v2.py` passes with the new values (pure pursuit x1.10 13.13 s, x1.15 12.59 s, x1.20 slides, P5 11.40 s).

The policy that came out of this is `../policies/rl_policy_v2_refit_s0_12M.npz`.

## Power rule and wall margin, 2026-11-02

The refit policy was clean on only about 57 % of AutoDRIVE laps (16 of 28). Two more crash rules, both optional `make_env` arguments and both off by default, fixed that.

1. **`power_lat`, no full throttle while turning hard.** Lateral acceleration (speed x yaw rate over the control step) above `power_lat` with throttle at or above `power_thr` (0.25) for 2 steps is a crash. In AutoDRIVE pure pursuit, P4 and P5 never hold more than 7.3 to 9.2 m/s² for 0.1 s, the refit policy reached 13 to 18 exactly where it touched. The gym never gets that high, so 8.0 is used as a proxy. At 8.0 pure pursuit and P5 trigger it 0 times per lap in the gym, the refit policy 8.4 times. Trained alone (`1102_power`, 19M) it removed the start straight touches, but the policy then ran 0 to 5 cm from the wall in the bottom left hairpin and was clean on 6 of 12 AutoDRIVE laps.
2. **`wall_margin`, room around the car.** The collision box is already the car's 0.58 x 0.31 m outline, and every AutoDRIVE contact happened when that box reached the wall on our map, so the size is right. What was missing is room. With `wall_margin` a wall within that distance of the box is a crash. In AutoDRIVE pure pursuit and P5 never come closer than 0.10 m. At 0.08 m the gym ranks drivers the way AutoDRIVE does (pure pursuit and P5 clean on both models, the refit policy crashes on twin, the power policy on both); at 0.10 m P5 crashes on twin.

| AutoDRIVE, ICRA | Clean laps | Wall contacts | Clean flying laps |
|---|---|---|---|
| refit 12M | 16 / 28 (57 %) | 15 | mean 11.54 s |
| + power rule (19M) | 6 / 12 (50 %) | 9 | mean 12.38 s |
| + power rule + 0.08 m margin (27M) | **31 / 31** | **0** | mean 11.24 s |

`../train_v2/margin_gate.py CKPT.zip` runs a checkpoint with both rules on (2 nominal and 10 randomised episodes on ICRA). The policy is `../policies/rl_policy_v2_margin_s1_27M.npz`.

### What the gym still gets wrong (2026-11-02)

Pure pursuit drives the same line and speeds in the gym and in AutoDRIVE (min speeds within 0.15 m/s, line within 0.03 m). The end to end policies do not, because they drive nearer the limit. Replaying 5,794 half-second AutoDRIVE windows in the twin (logged state and commands, real delays) shows where.

| What the car is doing | Gym vs AutoDRIVE |
|---|---|
| 3 to 5.5 m/s, 6 to 15° of steering, part throttle | same (turn ratio 1.00) |
| throttle 0 while turning | twin turns 10 % more and loses 0.5 m/s less in 0.5 s. AutoDRIVE brakes about 1 m/s² harder and braking takes cornering grip (`k_couple` about 0.7 fits, nominal is 0) |
| full throttle while turning | twin turns 10 % less and gains 0.3 m/s less. AutoDRIVE accelerates harder and rotates more under power, whatever `k_couple` is |

None of these are in `vehicle.py` yet. The wall margin covers the scatter they cause, not the cause.

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
