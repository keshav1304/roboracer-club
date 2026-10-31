# RL on the ICRA 2026 track, AutoDRIVE results

Terms like flying lap, x1.15 and grip shield are explained in [README.md](README.md#words-we-use).

| Driver | Laps | Wall contacts | Flying lap mean | Best |
|---|---|---|---|---|
| **RL P5** + grip shield | 10 | 0 | **11.52 s** | 11.46 s |
| RL P4 + grip shield | 10 | 0 | 11.59 s | 11.58 s |
| Pure pursuit on the club line, x1.15 speed (no RL) | 10 | 0 | 12.67 s | 12.66 s |
| Pure pursuit, x1.20 speed | 0 | slid into the wall in lap 1 | | |

P5 is 1.15 s (9%) a lap faster than the fastest pure pursuit we could run without crashing, on the same race line. (That 1.15 s and the x1.15 speed factor match by coincidence.) Eight of its nine flying laps were between 11.46 and 11.51 s and one was 11.74 s, with no contact.

![Top-down replay of the three 10-lap runs](media/autodrive_P5_vs_P4_vs_pp.png)

[Replay video, 38 s](media/autodrive_P5_vs_P4_vs_pp.mp4). All three cars are replayed from their deploy logs at their real AutoDRIVE positions, synced on the simulator clock. Red is P5, blue P4, green pure pursuit.

How these were measured. The ICRA 2026 competition build of AutoDRIVE (`autodrive_simulator_compete`, release 2026-icra), 2026-10-26, simulator lap clock, flying laps only, AutoDRIVE window in the background, deploy node from `deploy/`.

## What the policy does

Pure pursuit on Keshav's club line does all the steering, with a speed plan 1.15 times the base plan. The policy sees the LiDAR, the speed, the last steering command and what pure pursuit is about to do, and adds a throttle correction of up to ±0.064 throttle, which is about ±1.6 m/s of target speed and a fifth of the throttle range. A grip shield coasts when a corner asks for more grip than the car has. The policy was trained with that shield inside the gym.

## How we got here

Each step below fixed what the one before it crashed on. The physics behind them is in `env/AUTODRIVE_PHYSICS.md`, and the end to end approaches we also tried are in [STRATEGIES.md](STRATEGIES.md).

1. Our first PPO policies drove 10 s laps in the gym by cornering far harder than the AutoDRIVE car can. On the plain f1tenth_gym they cornered at up to 50 m/s². Even under a first grip cap they still cornered at 12 m/s².
2. With AutoDRIVE's throttle response measured and grip capped at 8.5 m/s², a residual policy on pure pursuit did 60/60 clean gym laps at 10.93 s. In AutoDRIVE it crashed. The AutoDRIVE car barely turns past about 18° of steering, and the policy used full lock in the hairpin.
3. Adding that understeer plus randomised physics (P2, P3) still crashed at the hairpin. The policy's own steering pushed the wheel to about 23°.
4. Reading the AutoDRIVE source and 11 deploy logs gave a slide model. Above a certain lateral demand the front tyres slide, and then the car only turns at about 4.5 m/s². Throttle only, steering left to pure pursuit (P4). With the grip shield added in the deploy node it ran 10/10 clean, first at 11.87 s and then 11.59 s with a better seed.
5. Refitting on all 19 logs showed the slide limit falls with wheel angle and depends on how long the demand lasts. With that and the shield inside the gym, P5 ran 10/10 clean at 11.52 s.

Two things outside the gym also caused crashes, both in deployment. Timing the control loop on the wall clock (AutoDRIVE runs slower than real time), and letting AutoDRIVE's window come to the front, which starved the control loop. Both are written up in `deploy/README.md`.

## Limits and next steps

- On this race line P5 is close to what the grip allows. A car holding 8.5 m/s² of cornering everywhere on the club line would lap in about 11.47 s (11.71 s at 8.0, 11.25 s at 9.0), and P5 averages 11.52 s. AutoDRIVE holds a little more than that in gentle corners, so this is a guide rather than a hard limit, but most of the remaining time is in the line itself. Optimising the race line is the next big step.
- Gym lap time alone does not pick a good policy. A faster seed from the same training crashed twice in AutoDRIVE (see `train/README.md`).
- The control loop timing on a laptop is the biggest risk in a live demo. Run with AutoDRIVE in the background and nothing heavy open.
