"""Gymnasium environment for PPO: f1tenth_gym (dev-humble API) made to drive like the AutoDRIVE car.

    from race_env import make_env
    env = make_env("icra", seed=0)            # end to end, the policy sets steering and throttle
    env = make_env("icra", seed=0, residual=0.4, steer_residual=0.0, ref_speed_scale=1.15)
                                              # the P5 setup, the policy corrects pure pursuit's throttle
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Why the physics looks like this is in AUTODRIVE_PHYSICS.md next to this file, with the AutoDRIVE evidence
behind every number. In short, the gym's own tyre model lets the car corner far harder than the AutoDRIVE
car can, and RL learns to drive right at whatever limit it is given, so a policy trained on the plain gym
crashes in AutoDRIVE. On top of the gym this env adds
1. a slide limit that falls with the wheel angle (10.4 m/s^2 of lateral demand up to 14 deg of wheel,
   7.0 from 18 deg). Above it for 0.08 s the front slides, and then it only holds 4.5 m/s^2 until the
   corner eases. Demand = v^2 tan(wheel) / wheelbase.
2. slight understeer in normal driving, the car turns 0.955 x what the wheel asks
3. AutoDRIVE's throttle response, engine braking at throttle 0 and extra drag while steering (below)
4. 30 ms steering delay plus the 183 deg/s steering rate, 40 ms throttle delay, 2 cm LiDAR noise
5. the deploy node's grip shield (grip_shield=True), so training sees what deployment does
6. per-episode randomisation of all of the above (randomize=True). randomize=False uses the nominal
   values; evaluate with that.
validate_vs_autodrive.py drives pure pursuit and the deployed policy here and prints the result next to
what the same drivers did in AutoDRIVE.

Tracks: "Spielberg" (bundled with the gym), and the AutoDRIVE tracks built from the club's maps by
tracks/build_tracks.py, "icra" (ICRA 2026 compete track, 58 m) and "practice" (30.7 m). On those
two, s = 0 is the AutoDRIVE spawn (0.8, 3.16), facing the way the car starts.

Contract shared with train/ and deploy/. Change these numbers only together.

Action, 2 floats in [-1, 1], in AutoDRIVE's own terms
    action[0]  steering_command, +1 = full left = 0.5236 rad
    action[1]  throttle: -1 = throttle_command 0, +1 = throttle_command MAX_THROTTLE (0.32)
    action_to_autodrive(action) -> (throttle_command, steering_command), published as is.

Residual mode (residual > 0). The env runs a base driver, pure pursuit on the club race line
(tracks/NAME/NAME_line.csv) at a planned speed x ref_speed_scale, and sends
    base + [steer_residual, residual] * action
steer_residual=0 means the policy only changes the throttle. That is what held up in AutoDRIVE: when the
policy also steered, the total wheel angle reached 21 to 24 deg and the front slid (see RESULTS.md).

Throttle is not speed. The gym runs in acceleration mode and the env turns throttle into an
acceleration with what was measured in AutoDRIVE (step tests on the ICRA straight plus the deploy logs):
    throttle > 0   accel = min(A_UP, K_SPEED * (24.6 * throttle - speed)), and no lower than -COAST
    throttle = 0   accel = -COAST_ZERO (5.7 m/s^2, engine braking; there is no harder brake)
    while steering, TURN_DRAG (3 m/s^2 per rad of wheel) more
A_UP, COAST and COAST_ZERO get a random +/-10% per episode (throttle_jitter).

Observation, 110 float32 (112 in residual mode)
    [0:108]    LiDAR: 1081 beams over 270 deg, split in order (right to left) into 108 groups,
               mean of each group, clipped at 10 m, divided by 10 -> 0..1
    [108]      current speed / 8 m/s, clipped to 0..1          (AutoDRIVE /odom twist.linear.x)
    [109]      last steering command sent, -1..1               (not /steering, which lags about 300 ms)
    [110:112]  residual mode only: the base driver's action, -1..1
    build_observation() has no gym or ROS imports, so the deploy node builds the same vector.

Reward per control step (20 Hz), weights in REWARD_WEIGHTS
    progress * ds - time + lap_bonus (per lap) - collision (and the episode ends)
    - steer_rate * |steer_now - steer_prev| - steer_res * |policy steering correction|
    - center * |ey| + tal * closeness to the base driver (off) + speed * v (off)
    ey is the sideways distance from the centre line, ds the metres gained along it this step.
The defaults are the pod's whiteboard reward (progress 10, center 0.05, lap 50, collision 10, steer_rate
0.1). P5 was trained with reward v3 instead (time 1 per step, collision 100, center 0), passed in by
train/train.py --preset p5. info carries each term (r_progress, r_time, ...) for TensorBoard, plus
laps, lap_times, sliding and shield. A contact, a spin (15 deg body slip) or the episode limit ends it.
"""
from __future__ import annotations

import dataclasses
import math
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

LIDAR_BINS = 108
MAX_RANGE = 10.0
NUM_BEAMS = 1081                 # AutoDRIVE RoboRacer LiDAR
FOV = 4.712389                   # 270 deg
MAX_STEER = 0.5236               # rad at steering command 1.0, measured in AutoDRIVE (source: 30 deg)
SPEED_PER_THROTTLE = 24.6        # m/s per unit throttle command, settled speed, measured in AutoDRIVE
MAX_THROTTLE = 0.32              # throttle_command at action[1] = +1
SPEED_NORM = 8.0                 # m/s, speed scale in the observation
A_UP = 4.7                       # m/s^2, constant acceleration ramp, measured (throttle 0.20 and 0.26)
K_SPEED = 8.0                    # 1/s, how sharply it settles onto 24.6 x throttle
COAST_DECEL = 4.5                # m/s^2 when the throttle asks for less than the speed (4.3 over whole laps);
                                 # also the base driver's braking plan
COAST_ZERO = 5.7                 # m/s^2 at throttle 0 (straight-line coast; 6.5 measured coasting in corners)
A_LAT_MAX = 8.5                  # m/s^2: the base driver's speed plan (x REF_GRIP), and the 'cap' grip model
A_HI, LIM_D1 = 10.4, 14.0       # m/s^2 demand the front holds briefly, up to this wheel angle (deg)
A_LO, LIM_D2 = 7.0, 18.0        # ... falling to this from this wheel angle on
SLIDE_TRIG = 0.08               # s above the limit before the front slides
A_SLIDE = 4.5                    # m/s^2 while sliding (logs: 4.2 to 5.6; source curve falls to 0.5 of peak)
K_US = 0.955                     # the car turns 0.955 x the wheel's kinematic yaw rate in normal driving
TURN_DRAG = 3.0                  # m/s^2 per rad of wheel angle, extra deceleration (refit on the logs)
SHIELD_ON, SHIELD_OFF = 9.2, 8.8  # grip shield: coast after 2 steps above SHIELD_ON, until under SHIELD_OFF
STEER_DELAY_MS = 30              # command to wheel on the simulator clock, best fit to the deploy logs
THROTTLE_DELAY_MS = 40
STEER_RATE = math.radians(183.346)  # rad/s, AutoDRIVE's SteeringRate (the gym's own sv_max is 3.2, the same)
GRIP_JITTER = 0.15               # 'cap' model only: per-episode grip factor in [1 - GRIP_JITTER, 1]
SPIN_SLIP = 0.2618               # rad (15 deg) body slip that counts as a crash
REF_GRIP = 0.6                   # reference speeds use 0.6 x A_LAT_MAX, then x ref_speed_scale
REF_BRAKE = 0.5                  # and plan braking at 0.5 x COAST_DECEL
LAT_CRASH = 11.0                 # m/s^2 (speed x yaw rate, 0.3 s average): off in practice, the slide rule does it
LIDAR_NOISE = 0.02               # m, Gaussian noise on every beam
THROTTLE_JITTER = 0.10           # per-episode random factor on A_UP and COAST_DECEL
CONTROL_HZ = 20
SUBSTEP = 0.01                   # s, gym integration step
MAX_LAPS = 10
MAX_STEPS = 3000                 # 150 s at 20 Hz
OBS_DIM = LIDAR_BINS + 2
TRACKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tracks')
REWARD_WEIGHTS = {'progress': 10.0, 'time': 0.0, 'lap_bonus': 50.0, 'collision': 10.0, 'steer_rate': 0.1,
                  'steer_res': 0.0, 'tal': 0.0, 'center': 0.05, 'speed': 0.0}


# ---- shared with deploy/ ------------------------------------------------------------------
def preprocess_scan(ranges) -> np.ndarray:
    """1081 (or any number of) LiDAR ranges -> 108 values in 0..1."""
    scan = np.asarray(ranges, dtype=np.float32)
    scan = np.clip(np.nan_to_num(scan, nan=MAX_RANGE, posinf=MAX_RANGE, neginf=0.0), 0.0, MAX_RANGE)
    bins = np.array([c.mean() for c in np.array_split(scan, LIDAR_BINS)], dtype=np.float32)
    return bins / MAX_RANGE


def build_observation(ranges, speed_mps: float, steering_cmd: float, base_action=None) -> np.ndarray:
    extra = [] if base_action is None else list(np.clip(np.asarray(base_action, np.float64), -1.0, 1.0))
    return np.concatenate([preprocess_scan(ranges),
                           [np.clip(speed_mps / SPEED_NORM, 0.0, 1.0), np.clip(steering_cmd, -1.0, 1.0)],
                           extra]).astype(np.float32)


def throttle_to_accel(throttle: float, speed_mps: float, a_up: float = A_UP, coast: float = COAST_DECEL,
                      coast_zero: float = COAST_ZERO) -> float:
    """AutoDRIVE's longitudinal response, measured: m/s^2 for a throttle_command at a given speed."""
    if throttle > 0.0:
        return max(-coast, min(a_up, K_SPEED * (SPEED_PER_THROTTLE * throttle - speed_mps)))
    return -coast_zero if speed_mps > 0.05 else -speed_mps / 0.01    # throttle 0: motor braking, then stop


def action_to_autodrive(action) -> tuple[float, float]:
    """(throttle_command, steering_command) for /autodrive/roboracer_1/*_command."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return float((a[1] + 1.0) * 0.5 * MAX_THROTTLE), float(a[0])


def _speed_profile(xy, a_lat, a_brake=COAST_DECEL, v_max=0.95 * MAX_THROTTLE * SPEED_PER_THROTTLE):
    """Min-time speed along a closed line: v <= sqrt(a_lat / |kappa|), the fitted throttle response
    forward, coasting backward (AutoDRIVE cannot brake harder than it coasts)."""
    a, m, z = np.roll(xy, 3, 0), xy, np.roll(xy, -3, 0)
    cross = (m - a)[:, 0] * (z - a)[:, 1] - (m - a)[:, 1] * (z - a)[:, 0]
    k = 2 * np.abs(cross) / np.maximum(1e-9, np.linalg.norm(m - a, axis=1) * np.linalg.norm(z - m, axis=1)
                                        * np.linalg.norm(z - a, axis=1))
    ds = np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1)
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(k, 1e-6)))
    n = len(xy)
    for _ in range(2):
        for i in range(n):
            j = (i + 1) % n
            acc = max(0.0, throttle_to_accel(MAX_THROTTLE, v[i]))
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * acc * ds[i]))
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * a_brake * ds[i]))
    return v, ds


def action_to_gym(action, speed_mps: float) -> tuple[float, float]:
    """(steering angle rad, acceleration m/s^2) for f1tenth_gym in acceleration mode."""
    throttle, steering = action_to_autodrive(action)
    return steering * MAX_STEER, throttle_to_accel(throttle, speed_mps)


# ---- environment --------------------------------------------------------------------------
class RaceEnv(gym.Env):
    metadata = {'render_modes': ['human', 'human_fast', 'rgb_array'], 'render_fps': CONTROL_HZ}

    def __init__(self, track_name: str = 'Spielberg', seed: int = 0, render_mode: str | None = None,
                 random_start: bool = True, reward_weights: dict | None = None,
                 max_laps: int = MAX_LAPS, max_steps: int = MAX_STEPS, a_lat_max: float = A_LAT_MAX,
                 grip_jitter: float = GRIP_JITTER, lat_crash: float = LAT_CRASH,
                 steer_delay: int | None = None, throttle_delay: int | None = None,
                 lidar_noise: float = LIDAR_NOISE, throttle_jitter: float = THROTTLE_JITTER,
                 residual: float = 0.0, steer_residual: float | None = None, ref_speed_scale: float = 1.0,
                 grip_model: str = 'slide', grip_shield: bool = True, model: str = 'ks', randomize: bool = True):
        super().__init__()
        unknown = set(reward_weights or {}) - set(REWARD_WEIGHTS)
        if unknown:
            raise ValueError(f'unknown reward weights {sorted(unknown)}; have {sorted(REWARD_WEIGHTS)}')
        self.w = dict(REWARD_WEIGHTS, **(reward_weights or {}))
        self.max_laps, self.max_steps = max_laps, max_steps
        self.a_lat_max, self.grip_jitter = a_lat_max, grip_jitter
        self.a_lat = a_lat_max
        self.lat_crash = lat_crash
        self.lidar_noise, self.throttle_jitter, self.residual = lidar_noise, throttle_jitter, residual
        self.steer_residual = residual if steer_residual is None else steer_residual
        self.res_vec = np.array([self.steer_residual, residual], np.float32)
        self.grip_model, self.model, self.randomize = grip_model, model, randomize
        self.grip_shield = grip_shield
        # delays in 10 ms substeps; overrides (in 50 ms control steps) win and switch off their randomisation
        self.steer_sub_fixed = None if steer_delay is None else int(round(steer_delay / (CONTROL_HZ * SUBSTEP)))
        self.thr_sub_fixed = None if throttle_delay is None else int(round(throttle_delay / (CONTROL_HZ * SUBSTEP)))
        self.steer_sub = self.steer_sub_fixed if self.steer_sub_fixed is not None else round(STEER_DELAY_MS / 10)
        self.thr_sub = self.thr_sub_fixed if self.thr_sub_fixed is not None else round(THROTTLE_DELAY_MS / 10)
        self.steer_delay = self.steer_sub * SUBSTEP * CONTROL_HZ          # in control steps, for exporters
        self._nominal_grip()
        self.line_shift = 0.0
        self.a_up, self.coast, self.coast_zero = A_UP, COAST_DECEL, COAST_ZERO
        from f1tenth_gym.envs.action import LongitudinalActionType
        from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS
        from f1tenth_gym.envs.env_config import (ControlConfig, DynamicModel, EnvConfig, ObservationConfig,
                                                 ResetConfig, SimulationConfig)
        from f1tenth_gym.envs.lidar.config import LiDARConfig
        from f1tenth_gym.envs.observation import ObservationType

        # sv_max stays the gym's 3.2 rad/s: AutoDRIVE's SteeringRate is 183.346 deg/s, the same
        params = dataclasses.replace(F1TENTH_VEHICLE_PARAMETERS, s_min=-MAX_STEER, s_max=MAX_STEER,
                                     v_max=12.0)
        local = os.path.join(TRACKS_DIR, track_name)
        if os.path.isfile(os.path.join(local, f'{track_name}_map.yaml')):     # our AutoDRIVE tracks
            from f1tenth_gym.envs.track import Track
            map_name = Track.from_track_path(os.path.join(local, track_name))
        else:                                                                # a track bundled with the gym
            map_name = track_name
        cfg = EnvConfig(
            seed=seed,
            map_name=map_name,
            params=params,
            num_agents=1,
            control_config=ControlConfig(longitudinal_mode=LongitudinalActionType.ACCL),
            simulation_config=SimulationConfig(timestep=SUBSTEP, max_laps=10_000,   # laps counted here
                                               dynamics_model=DynamicModel.KS if model == 'ks' else DynamicModel.ST),
            observation_config=ObservationConfig(type=ObservationType.DIRECT),
            reset_config=ResetConfig(),
            lidar_config=LiDARConfig(num_beams=NUM_BEAMS, field_of_view=FOV, range_max=MAX_RANGE),
            render_enabled=render_mode is not None,
        )
        self.sim = gym.make('f1tenth_gym:f1tenth-v0', config=cfg, render_mode=render_mode)
        self.wheelbase = params.lf + params.lr
        self.track = self.sim.unwrapped.track
        self.track_len = float(self.track.centerline.length)
        # TAL reference: the club race line if the track has one, else the centre line
        line_csv = os.path.join(local, f'{track_name}_line.csv')
        if os.path.isfile(line_csv):
            self.ref_xy = np.loadtxt(line_csv, delimiter=',', comments='#')[:, :2]
        else:
            self.ref_xy = np.stack([self.track.centerline.xs, self.track.centerline.ys], 1)
        self.ref_v, ds = _speed_profile(self.ref_xy, REF_GRIP * a_lat_max, a_brake=REF_BRAKE * COAST_DECEL)
        self.ref_speed_scale = ref_speed_scale
        self.ref_v = self.ref_v * ref_speed_scale                         # same as the deploy node's --speed-scale
        tang = np.roll(self.ref_xy, -1, 0) - np.roll(self.ref_xy, 1, 0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        self._ref_normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
        self._ref_xy_ep = self.ref_xy
        self._ref_ds = ds
        self.ref_s = np.concatenate([[0.0], np.cumsum(ds)[:-1]])
        self.ref_len = float(ds.sum())
        self.ref_lap_s = float(np.sum(ds / np.maximum(self.ref_v, 0.1)))    # reference lap time
        self.ref_la = (0.5, 0.25)                                         # pure pursuit lookahead m, s
        self.ref_lead = 0.0                                               # fraction of delay x speed added
        self._ref_i = None
        self.render_mode = render_mode
        self.random_start = random_start
        self.substeps = int(round(1.0 / (CONTROL_HZ * SUBSTEP)))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        n_extra = 2 if residual else 0
        low = np.concatenate([np.zeros(LIDAR_BINS + 1), -np.ones(1 + n_extra)]).astype(np.float32)
        self.observation_space = spaces.Box(low, np.ones(OBS_DIM + n_extra, np.float32), dtype=np.float32)
        self._seed = seed

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed if seed is not None else self._seed)
        self._seed = None                                           # seed once, then keep the stream
        s0 = float(self.np_random.uniform(0.0, self.track_len)) if self.random_start else 0.0
        x, y, yaw = self.track.frenet_to_cartesian(s0, 0.0, 0.0)
        self.sim.reset(options={'poses': np.array([[x, y, yaw]])})
        obs, *_ = self.sim.step(np.array([[0.0, 0.0]]))            # the scan is all zeros until one step
        a = obs['agent_0']
        st = a['std_state']
        self.prev_s, _, _ = self.track.cartesian_to_frenet(st[0], st[1], st[4], s_guess=s0)
        self.prev_action = np.zeros(2, np.float32)
        self.progress, self.t, self.steps, self.laps = 0.0, 0.0, 0, 0
        self.lap_start_t, self.lap_times = 0.0, []
        self.speed = float(st[3])
        self.state = np.array(st, dtype=np.float64)                  # x, y, steer, v, yaw, yaw rate, slip
        self.a_lat = self.a_lat_max * float(self.np_random.uniform(1 - self.grip_jitter, 1.0))
        self._ref_i = None
        r = self.np_random
        j = self.throttle_jitter
        self.a_up = A_UP * float(r.uniform(1 - j, 1 + j))
        f = float(r.uniform(1 - j, 1 + j))
        self.coast, self.coast_zero = COAST_DECEL * f, COAST_ZERO * f
        if self.randomize:                                           # domain randomisation, see the top
            self.a_hi, self.a_lo = float(r.uniform(10.1, 10.7)), float(r.uniform(7.0, 8.5))
            self.d1 = math.radians(float(r.uniform(13.0, 15.0)))
            self.d2 = self.d1 + math.radians(float(r.uniform(3.5, 4.5)))
            self.slide_trig = float(r.uniform(0.06, 0.10))
            self.a_slide = float(r.uniform(4.0, 5.5))
            self.k_us = float(r.uniform(0.93, 0.98))
            self.turn_drag = float(r.uniform(2.0, 4.0))
            self.line_shift = float(r.uniform(-0.10, 0.10))
            self.steer_sub = self.steer_sub_fixed if self.steer_sub_fixed is not None else int(r.integers(2, 7))
            self.thr_sub = self.thr_sub_fixed if self.thr_sub_fixed is not None else int(r.integers(2, 7))
        else:
            self._nominal_grip()
            self.line_shift = 0.0
            self.steer_sub = self.steer_sub_fixed if self.steer_sub_fixed is not None else round(STEER_DELAY_MS / 10)
            self.thr_sub = self.thr_sub_fixed if self.thr_sub_fixed is not None else round(THROTTLE_DELAY_MS / 10)
        self._ref_xy_ep = self.ref_xy + self.line_shift * self._ref_normal
        self._steer_q = [0.0] * self.steer_sub                        # commands still on their way, per 10 ms
        self._thr_q = [0.0] * self.thr_sub
        self.sliding, self._over_t = False, 0.0
        self._shield_over, self._coasting = 0, False
        self.wheel = 0.0                                             # the physical wheel angle, rad
        self.steer_cmd = 0.0
        self._prev_steer_sent = 0.0
        self._lat_hist = []                                          # 10 ms samples of speed x yaw rate
        return (build_observation(self._noisy(a['scan']), st[3], 0.0, self._base()),
                {'s': float(self.prev_s), 'a_lat': self.a_lat, 'a_up': self.a_up, 'coast': self.coast,
                 'a_hi': self.a_hi, 'a_lo': self.a_lo, 'a_slide': self.a_slide})

    def _nominal_grip(self):
        self.a_hi, self.a_lo, self.d1, self.d2 = A_HI, A_LO, math.radians(LIM_D1), math.radians(LIM_D2)
        self.slide_trig, self.a_slide, self.k_us, self.turn_drag = SLIDE_TRIG, A_SLIDE, K_US, TURN_DRAG

    def _base(self):
        """Base driver's action in residual mode, None otherwise."""
        return self.expert_action() if self.residual else None

    def _noisy(self, scan):
        if not self.lidar_noise:
            return scan
        return np.asarray(scan) + self.np_random.normal(0.0, self.lidar_noise, size=np.shape(scan))

    def _grip(self, wheel: float) -> float:
        """Wheel angle the car turns with, given the angle it has (see the module docstring)."""
        v = self.speed
        if v < 0.5:
            self.sliding = False
            return wheel
        if self.grip_model == 'cap':
            lim = math.atan(self.a_lat * self.wheelbase / v ** 2)
            return float(np.clip(wheel, -lim, lim))
        w = abs(wheel)
        demand = v * v * math.tan(w) / self.wheelbase
        if self.sliding:
            if demand < self.a_slide:
                self.sliding = False                                  # the tyres grip again
        else:
            lim = (self.a_hi if w <= self.d1 else self.a_lo if w >= self.d2 else
                   self.a_hi + (self.a_lo - self.a_hi) * (w - self.d1) / (self.d2 - self.d1))
            self._over_t = self._over_t + SUBSTEP if demand > lim else 0.0
            if self._over_t >= self.slide_trig:
                self.sliding, self._over_t = True, 0.0
        if self.sliding and demand > self.a_slide:
            return math.copysign(math.atan(self.a_slide * self.wheelbase / (v * v)), wheel)
        return math.copysign(math.atan(self.k_us * math.tan(w)), wheel)      # normal driving: slight understeer

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32).reshape(2), -1.0, 1.0)
        if self.residual:                                            # the policy corrects the base driver
            sent = np.clip(self.expert_action() + self.res_vec * action, -1.0, 1.0)
        else:
            sent = action
        throttle_cmd, steer_now = action_to_autodrive(sent)
        self.steer_cmd = steer_now
        shielded = False
        if self.grip_shield:                                         # same rule as the deploy node's --grip-shield
            demand = self.speed ** 2 * math.tan(abs(steer_now) * MAX_STEER) / self.wheelbase
            self._shield_over = self._shield_over + 1 if demand > SHIELD_ON else 0
            self._coasting = (self._coasting and demand > SHIELD_OFF) or self._shield_over >= 2
            if self._coasting:
                throttle_cmd, shielded = 0.0, True
        collision, slide_steps = False, 0
        for _ in range(self.substeps):
            self._steer_q.append(steer_now)                          # latency, in 10 ms steps
            self._thr_q.append(throttle_cmd)
            steer_target, thr_exec = self._steer_q.pop(0), self._thr_q.pop(0)
            step = STEER_RATE * SUBSTEP                              # the wheel turns at 183 deg/s
            self.wheel += float(np.clip(steer_target * MAX_STEER - self.wheel, -step, step))
            steer = self._grip(self.wheel)                           # what the tyres make of it
            slide_steps += int(self.sliding)
            accel = throttle_to_accel(thr_exec, self.speed, self.a_up, self.coast, self.coast_zero)
            if self.speed > 0.5:                                     # tyre scrub while steering
                accel -= self.turn_drag * abs(self.wheel)
            obs, *_ = self.sim.step(np.array([[steer, accel]]))
            self.t += SUBSTEP
            a = obs['agent_0']
            self.state = np.array(a['std_state'], dtype=np.float64)
            self.speed = float(self.state[3])
            yaw_rate = float(self.state[5]) if self.model == 'st' else self.speed * np.tan(self.state[2]) / self.wheelbase
            self._lat_hist = (self._lat_hist + [abs(self.speed * yaw_rate)])[-30:]
            if a['collision']:
                collision = True
                break
        st = self.state                                             # x, y, steer, v, yaw, yaw rate, slip
        spin_out = bool(st[3] > 1.0 and abs(st[6]) > SPIN_SLIP)
        lat_accel = float(np.mean(self._lat_hist))                   # 0.3 s average, as measured in AutoDRIVE
        lat_out = bool(self.lat_crash and st[3] > 1.0 and lat_accel > self.lat_crash)
        collision = collision or spin_out or lat_out
        s, ey, _ = self.track.cartesian_to_frenet(st[0], st[1], st[4], s_guess=self.prev_s)
        ds = s - self.prev_s
        if ds < -self.track_len / 2:                                # crossed the line forwards, s wrapped to 0
            ds += self.track_len
        elif ds > self.track_len / 2:                               # crossed it backwards
            ds -= self.track_len
        self.prev_s = s
        self.progress += ds
        self.steps += 1

        lap_completed = self.progress >= (self.laps + 1) * self.track_len
        if lap_completed:
            self.laps += 1
            self.lap_times.append(round(self.t - self.lap_start_t, 3))
            self.lap_start_t = self.t
        terms = {'r_progress': self.w['progress'] * ds,
                 'r_time': -self.w['time'],
                 'r_lap': self.w['lap_bonus'] if lap_completed else 0.0,
                 'r_collision': -self.w['collision'] if collision else 0.0,
                 'r_steer_rate': -self.w['steer_rate'] * abs(self.steer_cmd - self._prev_steer_sent),
                 'r_steer_res': -self.w['steer_res'] * abs(float(self.res_vec[0] * action[0])),
                 'r_tal': self.w['tal'] * self._tal_closeness(action, throttle_cmd, st) if self.w['tal'] else 0.0,
                 'r_center': -self.w['center'] * abs(ey),
                 'r_speed': self.w['speed'] * float(st[3])}
        reward = sum(terms.values())

        terminated = collision
        truncated = self.laps >= self.max_laps or self.steps >= self.max_steps
        self.prev_action = action
        self._prev_steer_sent = self.steer_cmd
        info = {'collision': collision, 'laps': self.laps, 'lap_times': list(self.lap_times),
                's': float(s), 'ey': float(ey), 'speed': float(st[3]), 'yaw_rate': float(st[5]),
                'progress_m': float(self.progress), 'sim_time': round(self.t, 3), 'throttle_cmd': throttle_cmd,
                'slip': float(st[6]), 'spin_out': spin_out, 'a_lat': self.a_lat, 'lat_accel': lat_accel,
                'lat_out': lat_out, 'sliding': slide_steps / self.substeps, 'shield': shielded,
                **{k: float(v) for k, v in terms.items()}}
        if self.render_mode == 'human':
            self.render()
        obs_out = build_observation(self._noisy(a['scan']), st[3], self.steer_cmd, self._base())
        return obs_out, float(reward), terminated, truncated, info

    def expert_action(self) -> np.ndarray:
        """The base driver (pure pursuit on the club line at the planned speed) as an action in [-1, 1]."""
        st = self.state
        steer, thr = self.reference_action(st[0], st[1], st[4], st[3])
        return np.array([steer, thr / MAX_THROTTLE * 2.0 - 1.0], dtype=np.float32)

    def reference_action(self, x, y, yaw, v):
        """What pure pursuit on the reference line would command here: (steering -1..1, throttle)."""
        n = len(self.ref_xy)
        if self._ref_i is None:                                      # first call: nearest point running our way,
            tang = np.roll(self.ref_xy, -1, 0) - self.ref_xy          # so a corridor across a thin wall can't win
            ok = np.where(tang[:, 0] * np.cos(yaw) + tang[:, 1] * np.sin(yaw) > 0)[0]
            idx = ok if len(ok) else np.arange(n)
        else:
            idx = np.arange(self._ref_i - 40, self._ref_i + 41) % n
        line = self._ref_xy_ep                                       # the line, shifted this episode
        i = int(idx[np.argmin((line[idx, 0] - x) ** 2 + (line[idx, 1] - y) ** 2)])
        self._ref_i = i
        lead = self.ref_lead * v * self.steer_delay / CONTROL_HZ      # optionally aim past the steering delay
        goal = (self.ref_s[i] + self.ref_la[0] + self.ref_la[1] * max(0.0, v) + lead) % self.ref_len
        tx, ty = line[int(np.searchsorted(self.ref_s, goal) % n)]
        dx, dy = tx - x, ty - y
        xc, yc = np.cos(yaw) * dx + np.sin(yaw) * dy, -np.sin(yaw) * dx + np.cos(yaw) * dy
        steer = np.arctan(2.0 * yc / max(1e-6, xc * xc + yc * yc) * self.wheelbase) / MAX_STEER
        throttle = min(MAX_THROTTLE, self.ref_v[i] / SPEED_PER_THROTTLE)          # settles at ref speed
        return float(np.clip(steer, -1, 1)), float(throttle)

    def _tal_closeness(self, action, throttle_cmd, st):
        steer_ref, thr_ref = self.reference_action(st[0], st[1], st[4], st[3])
        return max(0.0, 1.0 - abs(float(action[0]) - steer_ref) - abs(throttle_cmd - thr_ref) / MAX_THROTTLE)

    def render(self):
        return self.sim.render()

    def close(self):
        self.sim.close()


def make_env(track_name: str = 'Spielberg', seed: int = 0, **kwargs) -> RaceEnv:
    """The interface for train/: make_env("icra", seed=i), like gym.make("Pendulum-v1").
    Optional overrides, e.g. make_env("icra", seed=0, residual=0.4, steer_residual=0.0,
    ref_speed_scale=1.15) for P5, randomize=False for evaluation, or max_laps=50 for a long run."""
    return RaceEnv(track_name=track_name, seed=seed, **kwargs)
