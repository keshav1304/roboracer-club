# Deploy, running a policy in AutoDRIVE

Owner Rishabh. This folder is a working starting point, the exact node and policy that ran 10 clean laps on the ICRA 2026 track at 11.52 s average. Change anything you like, but check the pitfalls below first. Each one cost us a crash.

| File | What it is |
|---|---|
| `rl_policy_P5.npz` | the P5 policy. Network weights, the race line it drives on, and every constant the node needs. numpy only, no torch |
| `rl_deploy_node.py` | ROS 2 node that runs it. LiDAR and odometry in, steering and throttle commands out, at 20 Hz on the simulator clock |
| `tools/status.py` | prints the car's pose, speed, laps and collision count once. Quick check that the bridge is talking |
| `tools/reset.py` | zeroes both commands and puts the car back at the spawn (`--stop-only` just stops it) |

## First run

Start the bridge (devkit `autodrive_bridge`), open the ICRA 2026 simulator build, click Connect and pick Autonomous. Then, in the same ROS 2 environment as the bridge

```bash
python3 tools/status.py                       # should print pos=(0.80,3.16) ... at the spawn
python3 rl_deploy_node.py rl_policy_P5.npz --base-only --laps 1 --tag base
python3 tools/reset.py
python3 rl_deploy_node.py rl_policy_P5.npz --laps 3 --tag p5
```

`--base-only` drives pure pursuit on the club line with the policy switched off, which is the easiest way to check topics, units and timing. The speed plan inside the P5 file is already 1.15 times the base plan, so expect flying laps of about 12.7 s (that is the pure pursuit result below). With the policy expect about 11.5 s, no wall contacts, and the grip shield coasting about 15 steps a lap. Lap 1 starts from standstill and is slower.

The node stops on its own after `--laps` laps or 2 wall contacts, and on Ctrl-C it sends zero throttle and steering before exiting. Every control step is saved to `rl_run_TAG.npz` (`--log-dir` to put it elsewhere).

## Topics

| Direction | Topic (`/autodrive/roboracer_1/...`) | Type | Used for |
|---|---|---|---|
| in | `lidar` | LaserScan, best effort | 1081 ranges, binned into 108 |
| in | `odom` | Odometry, best effort | x, y, yaw, forward speed |
| in | `lap_time` | Float32 | the simulator clock |
| in | `lap_count`, `collision_count`, `last_lap_time` | Int32 / Float32 | stopping and printing |
| out | `throttle_command` | Float32 | 0 to 0.32 |
| out | `steering_command` | Float32 | -1 to 1, +1 = full left (30°) |

## What happens every control step

```text
P = dict(np.load('rl_policy_P5.npz'))
obs = [108 LiDAR bins / 10 m] + [speed / 8, last steering command sent] + [base steering, base throttle]
act = two tanh layers (w1, w2) then w3, clipped to [-1, 1]      # actor(obs) in the node
sent = clip(base + [P['steer_residual'], P['residual']] * act, -1, 1)    # steer_residual is 0 for P5
throttle_command = (sent[1] + 1) / 2 * P['max_throttle']
steering_command = sent[0]
then the grip shield may set throttle_command to 0
```

`base` is pure pursuit on `P['ref_xy']` at the planned speeds `P['ref_v']` (`base_action` in the node). It must match the training env exactly, so if you rewrite the node, compare your observations with the env's `build_observation` on the same inputs.

## Pitfalls we hit

1. Time the loop on the simulator clock. AutoDRIVE runs at 0.4 to 1.0 times real time and it changes during a run, so a wall-clock 20 Hz loop acts at a different rate from training. The node counts `lap_time` and acts every 50 ms of simulator time.
2. Keep AutoDRIVE in the background while a policy drives, and do not screen record. With AutoDRIVE as the front window, control steps arrived 100 to 250 ms late and good policies hit the wall. In the background we saw 0 late steps in 10 laps. To check a run, look at `sim_t` in the log. Steps should be 0.05 s apart, anything over 0.1 s is late.
3. The bridge keeps the last command forever. A node that dies without sending zeros leaves the car driving. Stop with Ctrl-C (the node sends zeros) or run `tools/reset.py`.
4. Use the last steering command in the observation, not the `/steering` topic. That feedback lags the car by about 300 ms.
5. If the node prints its first line and then nothing, the bridge has stopped forwarding data even though AutoDRIVE still says connected (this happened after a long session). `tools/status.py` will fail with `KeyError: 'odom'`. Quitting AutoDRIVE, restarting the bridge, reopening AutoDRIVE and clicking Connect fixed it. For long runs we wrap the node in `timeout -s INT 600` so a hang still saves the log.
6. Right after AutoDRIVE starts, the first second can have late commands. If lap 1 touches the wall at the start line, reset and run again before blaming the policy.
7. Several AutoDRIVE builds spawn at the same (0.8, 3.16). Check that the ICRA 2026 build is the one open.
8. The simulator's own lap 1 time counts from the reset, before the first command. Compare flying laps (lap 2 on).

## Log columns

`rl_run_TAG.npz` has `rows` and `cols`. Columns are `t` (wall s), `sim_t` (simulator s), `x`, `y`, `yaw`, `v`, `base_steer`, `base_thr`, `act_steer`, `act_thr` (raw policy output), `steer_cmd`, `thr_cmd` (what was sent) and `col` (collision counter, -1 before the first message).

To make your own policy file, see `../train/export_policy.py`. Why the shield and the throttle-only setup exist is in `../env/AUTODRIVE_PHYSICS.md`.
