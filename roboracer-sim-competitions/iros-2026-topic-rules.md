# IROS 2026 AutoDRIVE topic rules

Official access flags for RoboRacer Sim Racing League @ IROS 2026.

IROS 2026 does not publish a separate topic list. The event page points at the shared Sim Racing League **Technical Guide** and **Competition Rules**. Those are the documents below.

Sources:

- [RoboRacer Sim Racing League @ IROS 2026](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-iros-2026/)
- [Technical Guide §2.3 Data Streams](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-guide-2025/)
- [Competition Rules](https://autodrive-ecosystem.github.io/competitions/roboracer-sim-racing-rules-2025/)

## Access flags

| Flag | Meaning |
|------|---------|
| **Input** | Legal at race time. Raw sensor / actuator-feedback data the stack may subscribe to. |
| **Output** | Legal at race time. The only commands the stack may publish to drive the car. |
| **Restricted** | Debug / training only. Must **not** be used while autonomously racing (deployment / inference). |

From the Technical Guide:

> You may use the restricted topics for debugging, training AI models, etc. However, these topics should not be used while autonomously racing at run-time (e.g., during the deployment/inference stage of the algorithm).

From the Competition Rules:

> Any kind of autonomous racing algorithm that makes use of **raw perception data** to generate control commands is permissible.
>
> **Utilizing simulation ground truth data** or controlling aspects other than the vehicle actuators is **not allowed**.

## Allowed at race time — inputs

| Topic | Type | Message |
|-------|------|---------|
| `/autodrive/roboracer_1/front_camera` | Sensor | `sensor_msgs/msg/Image` |
| `/autodrive/roboracer_1/imu` | Sensor | `sensor_msgs/msg/Imu` |
| `/autodrive/roboracer_1/left_encoder` | Sensor | `sensor_msgs/msg/JointState` |
| `/autodrive/roboracer_1/lidar` | Sensor | `sensor_msgs/msg/LaserScan` |
| `/autodrive/roboracer_1/right_encoder` | Sensor | `sensor_msgs/msg/JointState` |
| `/autodrive/roboracer_1/steering` | Sensor | `std_msgs/msg/Float32` |
| `/autodrive/roboracer_1/throttle` | Sensor | `std_msgs/msg/Float32` |

Steering and throttle **without** `_command` are measured actuator feedback, not ground-truth pose.

## Allowed at race time — outputs

| Topic | Type | Message |
|-------|------|---------|
| `/autodrive/roboracer_1/steering_command` | Actuator | `std_msgs/msg/Float32` |
| `/autodrive/roboracer_1/throttle_command` | Actuator | `std_msgs/msg/Float32` |

Range for both commands is `[-1, 1]`.

## Restricted — do not use while racing

| Topic | Type | Message | Why |
|-------|------|---------|-----|
| `/autodrive/roboracer_1/ips` | Sensor | `geometry_msgs/msg/Point` | Indoor positioning = ground-truth `{x, y, z}` |
| `/autodrive/roboracer_1/speed` | Debugging | `std_msgs/msg/Float32` | Ground-truth speedometer |
| `/autodrive/roboracer_1/lap_count` | Debugging | `std_msgs/msg/Int32` | Race scoring |
| `/autodrive/roboracer_1/lap_time` | Debugging | `std_msgs/msg/Float32` | Race scoring |
| `/autodrive/roboracer_1/last_lap_time` | Debugging | `std_msgs/msg/Float32` | Race scoring |
| `/autodrive/roboracer_1/best_lap_time` | Debugging | `std_msgs/msg/Float32` | Race scoring |
| `/autodrive/roboracer_1/collision_count` | Debugging | `std_msgs/msg/Int32` | Race scoring |
| `/autodrive/reset_command` | Debugging | `std_msgs/msg/Bool` | Simulator control, not an actuator |
| `/tf` | Debugging | `tf2_msgs/msg/TFMessage` | Official TF stream is restricted |

OK to subscribe to these while mapping, tuning, or training. The submitted race container must not consume them for control or localization.

## `/odom` is not on the official list

`/autodrive/roboracer_1/odom` is **not** in the Technical Guide table.

Our local `autodrive_bridge.py` still publishes it. That message is built from IPS position + IMU orientation + linear velocity — the same ground truth as `/ips`. Treat it as **restricted** for race-time use, same as IPS.

Do not seed AMCL, drive pure pursuit, or estimate speed from `/odom` or `/ips` in the submission image.

Legal substitutes for motion: `left_encoder` / `right_encoder` and `/imu` (gyro / accel). Legal substitute for pose: LiDAR (+ camera) against a map you built yourself.

## What this stack currently uses (for audit)

This is the **old AMCL + odom** stack (encoder/gyro wheel-odom was reverted). Official `/odom` and `world`→`roboracer_1` TF are **restricted** at race time; they are used here because they make AMCL accurate.

| Use | Topic | Legal? |
|-----|-------|--------|
| AMCL scans | `/autodrive/roboracer_1/lidar` (`lidar`) | Yes |
| AMCL motion TF | `/tf` `world`→`roboracer_1` (same pose as `/odom`) | Restricted |
| AMCL `/initialpose` seed | `/autodrive/roboracer_1/odom` via `map_to_world` | Restricted |
| PP pose | `/amcl_pose` | Yes (filter output) |
| PP speed | `/autodrive/roboracer_1/odom` twist | Restricted |
| MPC pose | `/amcl_pose` | Yes (filter output) |
| MPC speed | `/autodrive/roboracer_1/odom` twist | Restricted |
| Drive commands | `throttle_command`, `steering_command` | Yes |
