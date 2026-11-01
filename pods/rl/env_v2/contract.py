"""One place for every number that the env, the exporter and the deploy node must agree on.

The deploy node copies these values out of the exported .npz, so a policy file always carries the
contract it was trained with. Change a value here only together with deploy/rl_deploy_node_v2.py.

Units at the boundary are AutoDRIVE's own. steering_command is -1..1 with +1 = full left = 30 deg,
throttle_command is 0..MAX_THROTTLE. Inside the physics everything is SI (m, s, rad).

Sources. AutoDRIVE source, branch AutoDRIVE-Simulator, Assets/Prefabs/F1TENTH/F1TENTH.prefab,
Assets/Scripts/VehicleController.cs and Assets/Scenes/F1TENTH - Sim Racing.unity (read 2026-11-01),
plus the club's AutoDRIVE logs summarised in ../env/AUTODRIVE_PHYSICS.md.
"""
from __future__ import annotations

import math

import numpy as np

OBS_VERSION = 2

# ---- vehicle geometry (prefab) ----------------------------------------------------------------
# m, between the prefab's wheel colliders (+0.17 / -0.16). VehicleController's 324 mm is only its Ackermann
# parameter. All our grip fits used 0.3302, the same within 0.1 %
WHEELBASE = 0.33
TRACK_WIDTH = 0.236          # m, VehicleController.TrackWidth 236 mm
MASS = 3.47                  # kg, Rigidbody m_Mass
# m, collision box. The gym's F1TENTH box, kept from the validated env (TBC vs the AutoDRIVE mesh)
CAR_LENGTH, CAR_WIDTH = 0.58, 0.31
# m, LiDAR ahead of the rear axle (prefab: LIDAR z 0.1133, rear wheels z -0.16, COM 4.7 mm behind the origin).
# Poses everywhere are the rear axle, F1TENTH's base_link
LIDAR_X = 0.268

# ---- steering (VehicleController) --------------------------------------------------------------
MAX_STEER_DEG = 30.0         # SteeringLimit, symmetric
MAX_STEER = math.radians(MAX_STEER_DEG)
STEER_RATE_DEG = 183.346     # SteeringRate, deg/s of simulator time, applied when the angle grows (see vehicle.py)

# ---- throttle ------------------------------------------------------------------------------------
MAX_THROTTLE = 0.32          # throttle_command at action[1] = +1 (same cap as P5, AutoDRIVE goes to 1.0)
BRAKE_ZONE = -0.9            # action[1] below this sends throttle_command exactly 0, which brakes all 4 wheels
MIN_DRIVE_THROTTLE = 0.005   # smallest non-zero throttle the policy can send (coasting on engine braking)

# ---- control and sensing -----------------------------------------------------------------------
CONTROL_HZ = 20
DT_CONTROL = 1.0 / CONTROL_HZ
NUM_BEAMS = 1081
FOV = 4.712389               # 270 deg
LIDAR_BINS = 108
MAX_RANGE = 10.0
SPEED_NORM = 8.0             # m/s
YAW_RATE_NORM = 4.0          # rad/s
ACCEL_NORM = 10.0            # m/s^2

# ---- steering envelope (shared by env and deploy node) -----------------------------------------
# The nominal front grip limit by wheel angle, from the AutoDRIVE logs: ~9.8 m/s^2 of lateral demand
# (v^2 tan(wheel) / L) up to 14 deg of wheel, falling to ~7.4 m/s^2 at 18 deg and beyond.
LIM_HI, LIM_D1 = 9.81, 14.0
LIM_LO, LIM_D2 = 7.4, 18.0
ENVELOPE_FRACTION = 0.95     # the policy can ask for at most this fraction of the nominal limit
ENVELOPE_FREE_SPEED = 2.0    # m/s, below this the full 30 deg is allowed (hairpins, recovery)


def nominal_limit(wheel_rad: float) -> float:
    """Lateral demand (m/s^2) the front holds at this wheel angle, nominal AutoDRIVE."""
    w = math.degrees(abs(wheel_rad))
    if w <= LIM_D1:
        return LIM_HI
    if w >= LIM_D2:
        return LIM_LO
    return LIM_HI + (LIM_LO - LIM_HI) * (w - LIM_D1) / (LIM_D2 - LIM_D1)


def demand(speed: float, wheel_rad: float) -> float:
    """Lateral acceleration the car needs to follow its wheels, m/s^2."""
    return speed * speed * math.tan(abs(wheel_rad)) / WHEELBASE


def _envelope_table(fraction: float = ENVELOPE_FRACTION, free_speed: float = ENVELOPE_FREE_SPEED):
    speeds = np.arange(0.0, 12.01, 0.05)
    out = np.empty_like(speeds)
    angles = np.radians(np.linspace(0.0, MAX_STEER_DEG, 601))
    for i, v in enumerate(speeds):
        if v <= free_speed:
            out[i] = MAX_STEER
            continue
        ok = [a for a in angles if demand(v, a) <= fraction * nominal_limit(a)]
        out[i] = max(ok) if ok else angles[1]
    return speeds, out


ENV_SPEEDS, ENV_WHEEL = _envelope_table()


def envelope(speed: float) -> float:
    """Largest wheel angle (rad) the policy may ask for at this speed."""
    return float(np.interp(abs(speed), ENV_SPEEDS, ENV_WHEEL))


def action_to_commands(action, speed: float, use_envelope: bool = True) -> tuple[float, float]:
    """Policy action (2 floats in [-1, 1]) -> (steering_command, throttle_command) for AutoDRIVE.

    action[0] is the fraction of the steering the envelope allows right now, so the whole action range
    stays useful at every speed (no dead zone where every action means full lock).
    action[1] below BRAKE_ZONE is a brake (throttle exactly 0). Above it, throttle grows linearly from
    MIN_DRIVE_THROTTLE to MAX_THROTTLE.
    """
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    limit = envelope(speed) / MAX_STEER if use_envelope else 1.0
    steer = float(a[0] * limit)
    if a[1] < BRAKE_ZONE:
        thr = 0.0
    else:
        thr = max(MIN_DRIVE_THROTTLE, MAX_THROTTLE * (a[1] - BRAKE_ZONE) / (1.0 - BRAKE_ZONE))
    return steer, float(thr)


def commands_to_action(steer: float, thr: float, speed: float, use_envelope: bool = True) -> np.ndarray:
    """Inverse of action_to_commands, for behaviour cloning from a classic driver."""
    limit = envelope(speed) / MAX_STEER if use_envelope else 1.0
    a0 = np.clip(steer / max(limit, 1e-6), -1.0, 1.0)
    if thr <= 0.0:
        a1 = -1.0
    else:
        a1 = BRAKE_ZONE + (1.0 - BRAKE_ZONE) * np.clip(thr / MAX_THROTTLE, 0.0, 1.0)
    return np.array([a0, a1], dtype=np.float32)


# ---- observation v2 ------------------------------------------------------------------------------
OBS_FIELDS = (['lidar'] * LIDAR_BINS
              + ['speed', 'last_steer', 'last_throttle_action', 'yaw_rate', 'accel_now', 'accel_prev'])
OBS_DIM = len(OBS_FIELDS)   # 114


def preprocess_scan(ranges) -> np.ndarray:
    """1081 (or any number of) LiDAR ranges -> 108 values in 0..1. Same as the v1 contract."""
    scan = np.asarray(ranges, dtype=np.float32)
    scan = np.clip(np.nan_to_num(scan, nan=MAX_RANGE, posinf=MAX_RANGE, neginf=0.0), 0.0, MAX_RANGE)
    bins = np.array([c.mean() for c in np.array_split(scan, LIDAR_BINS)], dtype=np.float32)
    return bins / MAX_RANGE


def build_observation(ranges, speed: float, last_steer: float, last_thr_action: float,
                      yaw_rate: float, speed_hist) -> np.ndarray:
    """The v2 observation. speed_hist = (v_now, v_prev, v_prev2), all from odometry, one control step apart.
    yaw_rate is the heading change over the last control step divided by its length (not /odom twist,
    so the deploy node can build the same number from poses)."""
    v0, v1, v2 = speed_hist
    acc_now = (v0 - v1) / DT_CONTROL / ACCEL_NORM
    acc_prev = (v1 - v2) / DT_CONTROL / ACCEL_NORM
    tail = [np.clip(speed / SPEED_NORM, 0.0, 1.0), np.clip(last_steer, -1, 1), np.clip(last_thr_action, -1, 1),
            np.clip(yaw_rate / YAW_RATE_NORM, -1, 1), np.clip(acc_now, -1, 1), np.clip(acc_prev, -1, 1)]
    return np.concatenate([preprocess_scan(ranges), np.asarray(tail, np.float32)]).astype(np.float32)


def metadata() -> dict:
    """Everything the deploy node needs, written into the exported .npz."""
    return dict(obs_version=OBS_VERSION, wheelbase=WHEELBASE, max_steer=MAX_STEER, max_throttle=MAX_THROTTLE,
                brake_zone=BRAKE_ZONE, min_drive_throttle=MIN_DRIVE_THROTTLE, control_hz=CONTROL_HZ,
                lidar_bins=LIDAR_BINS, max_range=MAX_RANGE, speed_norm=SPEED_NORM, yaw_rate_norm=YAW_RATE_NORM,
                accel_norm=ACCEL_NORM, env_speeds=ENV_SPEEDS, env_wheel=ENV_WHEEL)
