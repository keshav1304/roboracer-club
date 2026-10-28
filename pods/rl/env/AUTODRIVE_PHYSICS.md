# What the AutoDRIVE car can and cannot do

Every number in `race_env.py` comes from the AutoDRIVE source or from our own runs in the ICRA 2026 build (19 logged runs, timed on the simulator clock). This page lists the limits that broke our early policies, with the evidence, so nobody has to find them again by crashing.

The short version. RL drives right at whatever limit the gym gives it. Away from the limit, gym and AutoDRIVE can disagree a lot and nothing bad happens. At the limit, every small difference turns into a crash. So the job of the env is to get the limit right.

## 1. Grip, and why the plain gym is far too optimistic

The plain f1tenth_gym tyre model holds about 20 m/s² of cornering at 4 m/s (we measured it with a full-lock circle), and our first policy learned to take hairpins at up to 50 m/s² in the gym. Even after we capped the gym at 8.5 m/s², the next policy crashed in AutoDRIVE in lap 1.

The AutoDRIVE source is public ([Tinker-Twins/AutoDRIVE](https://github.com/Tinker-Twins/AutoDRIVE), branch `AutoDRIVE-Simulator`, `Assets/Prefabs/F1TENTH/F1TENTH.prefab`). The sideways tyre friction peaks at a slip of 0.01 and falls to half of that peak by 0.1. Once the front tyres pass the peak they slide, and they keep sliding until the corner eases. The same file gives mass 3.47 kg, wheelbase 324 mm, steering limit 30° and steering rate 183°/s.

Lateral demand is `v² tan(wheel angle) / wheelbase`, what the car would need to follow its wheels. In our logs the limit depends on the wheel angle.

| Wheel angle | What AutoDRIVE held | Seen in |
|---|---|---|
| up to about 14° | 10.4 to 10.6 m/s² for a moment, no slide | P4 at (-4.05, -8.0), every lap |
| 16° to 18° and more | slid at 8 to 9.6 m/s² | pure pursuit x1.20 and x1.25, P4 lap 1, P4 with steering |

It also depends on time. Demand of 9.0 to 9.4 held for 0.3 s stayed on the road. Demand of 9.4 to 9.9 held for 0.5 s slid into the wall at the same corner. Once sliding, the car only turns at about 4.5 m/s² (4.2 to 5.6 measured), so it runs wide into the outside wall.

The env models this as a limit of 10.4 m/s² up to 14° of wheel that falls to 7.0 m/s² at 18° and beyond. Above the limit for 0.08 s the front slides and holds 4.5 m/s² until demand drops below that.

## 2. Past about 18° of steering the car barely turns

This is the practical form of section 1, and it is the one that hurts RL most. Full lock in a hairpin at speed does not work in AutoDRIVE. Our policies that controlled steering added a few degrees on top of pure pursuit in the hairpin, reached 21° to 24° in total, and slid off. Pure pursuit at its planned speeds stays clear of that range.

That is why the deployed policy (P5) leaves steering to pure pursuit and only changes the throttle.

## 3. Slight understeer everywhere else

In normal driving the car turns 0.955 times what the wheel asks (median over all logged normal driving, 0.95 to 0.96 at every wheel angle).

## 4. Throttle is not speed, and there is no brake

- `throttle_command` settles at 24.6 m/s per unit of throttle. 0.12 is already 3 m/s.
- From standstill the car accelerates at about 4.7 m/s² (throttle 0.20 and 0.26 gave the same ramp) until it nears that speed.
- Throttle 0 decelerates at 5.7 m/s² on the straight. That is the strongest braking there is, since negative throttle does nothing. Slowing from 8 to 4 m/s takes about 4.2 m.
- Steering costs speed, about 3 m/s² more deceleration per radian of wheel angle.

## 5. Delays

- About 30 ms from steering command to the wheel starting to move (simulator clock), then the wheel turns at 183°/s. Throttle takes about 40 ms.
- The `/steering` and `/throttle` feedback topics lag the car by about 300 ms. Do not put them in the observation. The env and the deploy node use the last steering command sent instead.
- Time everything on the simulator clock. AutoDRIVE runs at 0.4 to 1.0 times real time and it changes during a run. Our first delay estimate (150 ms) was wrong because it was timed on the wall clock.

## 6. The grip shield

The deploy node and the env share one safety rule. When the commanded lateral demand stays above 9.2 m/s² for two control steps, throttle goes to 0 until demand drops below 8.8. It costs almost nothing on a good lap (pure pursuit x1.15 would trigger it on 2 of 2547 steps) and it caught the slow slides that broke P4 without it. Training with the shield inside the env means the policy learns to drive with it rather than fight it.

## 7. How close the gym is now

`validate_vs_autodrive.py` drives the same drivers in the gym the way they ran in AutoDRIVE.

| Driver | Gym (nominal) | AutoDRIVE |
|---|---|---|
| pure pursuit x1.10 | clean, 13.20 s | clean, 13.21 s |
| pure pursuit x1.15 | clean, 12.63 s | clean 10/10, 12.67 s |
| pure pursuit x1.20 | clean, 12.10 s | slid into the wall |
| pure pursuit x1.25 | slid into the wall | slid into the wall |
| P5 + grip shield | clean, 11.42 s | clean 10/10, 11.52 s |

Earlier policies checked the same way (not in the repo) also matched. The ones that crashed in AutoDRIVE crashed here, and the throttle-only P4 seeds landed within 0.04 to 0.12 s of their AutoDRIVE laps.

What is still uncertain.
- Pure pursuit x1.20 stays on in the gym and slid in AutoDRIVE, so the gym is still a little optimistic right at the edge. A policy that slides in the gym is a bad sign even if it does not crash.
- No single setting reproduced every run. P4 itself slid once and stayed on once at the same corner. Training randomises the grip numbers over a band that covers the settings that came close (`randomize=True`), which is wider than AutoDRIVE on purpose.
- The recovery from a slide is the least measured part. We have few logs of a car sliding and then gripping again.
