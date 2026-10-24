"""Gymnasium environment for PPO: f1tenth_gym (dev-humble API) wrapped for stable-baselines3.

    from race_env import make_env
    env = make_env("Spielberg", seed=0)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Contract shared with train/ (Nathaniel) and deploy/ (Rishabh). Change these numbers only together.

Action, 2 floats in [-1, 1]
    action[0]  steering, +1 = full left = 0.5236 rad (AutoDRIVE steering_command 1.0)
    action[1]  speed, -1 = 0 m/s, +1 = V_MAX m/s
    action_to_autodrive(action) -> (throttle_command, steering_command) for the AutoDRIVE topics.
    Normalised to [-1, 1] because SB3's check_env warns on any other Box range.

Observation, 110 float32
    [0:108]    LiDAR: 1081 beams over 270 deg, split in order (right to left) into 108 groups,
               mean of each group, clipped at 10 m, divided by 10 -> 0..1
    [108:110]  previous action, as sent (each -1..1); zeros after reset
    build_observation(ranges, prev_action) does all of this and has no gym or ROS imports, so the
    deploy node can import it and feed AutoDRIVE's /lidar ranges straight in.

Reward per control step (20 Hz), from the pod doc and the whiteboard, weights in REWARD_WEIGHTS
    progress * ds        metres gained along the centre line, ds = s_now - s_prev (10)
    - center * |ey|      distance from the centre line, m (0.05)
    + lap_bonus          when a lap is completed (50); a contact ends the episode, so every lap
                         that completes is a clean one
    - collision          on a wall contact, which also ends the episode, terminated (10)
    + speed * v          speed in m/s (0, off). ds already pays for speed: at a fixed time step,
                         ds per step is speed along the track
s and ey come from the car's pose (x, y), not from the LiDAR: cartesian_to_frenet projects the
pose onto the centre line, s = arc length from the start line, ey = sideways offset (+ left).
info carries each term (r_progress, r_center, r_lap, r_collision, r_speed) for TensorBoard.
Lap time is deliberately not a reward term: "last lap minus this lap" depends on history the
policy cannot see (breaks the Markov property, see the pod doc). Eval ranks policies by
consecutive clean laps, then lap time; info['laps'] and info['lap_times'] give both.
Episodes are truncated after max_laps laps or max_steps control steps.
"""
from __future__ import annotations

import dataclasses

import gymnasium as gym
import numpy as np
from gymnasium import spaces

LIDAR_BINS = 108
MAX_RANGE = 10.0
NUM_BEAMS = 1081                 # AutoDRIVE RoboRacer LiDAR
FOV = 4.712389                   # 270 deg
MAX_STEER = 0.5236               # rad at steering command 1.0, measured in AutoDRIVE
SPEED_PER_THROTTLE = 24.8        # m/s per unit throttle command, measured in AutoDRIVE
V_MAX = 8.0                      # m/s at action[1] = +1
CONTROL_HZ = 20
MAX_LAPS = 10
MAX_STEPS = 3000                 # 150 s at 20 Hz
OBS_DIM = LIDAR_BINS + 2
REWARD_WEIGHTS = {'progress': 10.0, 'center': 0.05, 'lap_bonus': 50.0, 'collision': 10.0, 'speed': 0.0}


# ---- shared with deploy/ ------------------------------------------------------------------
def preprocess_scan(ranges) -> np.ndarray:
    """1081 (or any number of) LiDAR ranges -> 108 values in 0..1."""
    scan = np.asarray(ranges, dtype=np.float32)
    scan = np.clip(np.nan_to_num(scan, nan=MAX_RANGE, posinf=MAX_RANGE, neginf=0.0), 0.0, MAX_RANGE)
    bins = np.array([c.mean() for c in np.array_split(scan, LIDAR_BINS)], dtype=np.float32)
    return bins / MAX_RANGE


def build_observation(ranges, prev_action) -> np.ndarray:
    return np.concatenate([preprocess_scan(ranges),
                           np.clip(np.asarray(prev_action, np.float32), -1.0, 1.0)]).astype(np.float32)


def action_to_gym(action) -> tuple[float, float]:
    """(steering angle rad, speed m/s) for f1tenth_gym."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return float(a[0] * MAX_STEER), float((a[1] + 1.0) * 0.5 * V_MAX)


def action_to_autodrive(action) -> tuple[float, float]:
    """(throttle_command, steering_command) for /autodrive/roboracer_1/*_command."""
    steer, speed = action_to_gym(action)
    return speed / SPEED_PER_THROTTLE, steer / MAX_STEER


# ---- environment --------------------------------------------------------------------------
class RaceEnv(gym.Env):
    metadata = {'render_modes': ['human', 'human_fast', 'rgb_array'], 'render_fps': CONTROL_HZ}

    def __init__(self, track_name: str = 'Spielberg', seed: int = 0, render_mode: str | None = None,
                 random_start: bool = True, reward_weights: dict | None = None,
                 max_laps: int = MAX_LAPS, max_steps: int = MAX_STEPS):
        super().__init__()
        unknown = set(reward_weights or {}) - set(REWARD_WEIGHTS)
        if unknown:
            raise ValueError(f'unknown reward weights {sorted(unknown)}; have {sorted(REWARD_WEIGHTS)}')
        self.w = dict(REWARD_WEIGHTS, **(reward_weights or {}))
        self.max_laps, self.max_steps = max_laps, max_steps
        from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS
        from f1tenth_gym.envs.env_config import (ControlConfig, EnvConfig, ObservationConfig,
                                                 ResetConfig, SimulationConfig)
        from f1tenth_gym.envs.lidar.config import LiDARConfig
        from f1tenth_gym.envs.observation import ObservationType

        params = dataclasses.replace(F1TENTH_VEHICLE_PARAMETERS, s_min=-MAX_STEER, s_max=MAX_STEER,
                                     v_max=V_MAX)
        cfg = EnvConfig(
            seed=seed,
            map_name=track_name,
            params=params,
            num_agents=1,
            control_config=ControlConfig(),
            simulation_config=SimulationConfig(timestep=0.01, max_laps=10_000),   # laps counted here
            observation_config=ObservationConfig(type=ObservationType.DIRECT),
            reset_config=ResetConfig(),
            lidar_config=LiDARConfig(num_beams=NUM_BEAMS, field_of_view=FOV, range_max=MAX_RANGE),
            render_enabled=render_mode is not None,
        )
        self.sim = gym.make('f1tenth_gym:f1tenth-v0', config=cfg, render_mode=render_mode)
        self.track = self.sim.unwrapped.track
        self.track_len = float(self.track.centerline.length)
        self.render_mode = render_mode
        self.random_start = random_start
        self.substeps = int(round(1.0 / (CONTROL_HZ * 0.01)))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        low = np.concatenate([np.zeros(LIDAR_BINS), -np.ones(2)]).astype(np.float32)
        self.observation_space = spaces.Box(low, np.ones(OBS_DIM, np.float32), dtype=np.float32)
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
        return build_observation(a['scan'], self.prev_action), {'s': float(self.prev_s)}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32).reshape(2), -1.0, 1.0)
        steer, speed = action_to_gym(action)
        collision = False
        for _ in range(self.substeps):
            obs, *_ = self.sim.step(np.array([[steer, speed]]))
            self.t += 0.01
            a = obs['agent_0']
            if a['collision']:
                collision = True
                break
        st = a['std_state']                                         # x, y, steer, v, yaw, yaw rate, slip
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
                 'r_center': -self.w['center'] * abs(ey),
                 'r_lap': self.w['lap_bonus'] if lap_completed else 0.0,
                 'r_collision': -self.w['collision'] if collision else 0.0,
                 'r_speed': self.w['speed'] * float(st[3])}
        reward = sum(terms.values())

        terminated = collision
        truncated = self.laps >= self.max_laps or self.steps >= self.max_steps
        self.prev_action = action
        info = {'collision': collision, 'laps': self.laps, 'lap_times': list(self.lap_times),
                's': float(s), 'ey': float(ey), 'speed': float(st[3]), 'progress_m': float(self.progress),
                'sim_time': round(self.t, 3), **{k: float(v) for k, v in terms.items()}}
        if self.render_mode == 'human':
            self.render()
        return build_observation(a['scan'], action), float(reward), terminated, truncated, info

    def render(self):
        return self.sim.render()

    def close(self):
        self.sim.close()


def make_env(track_name: str = 'Spielberg', seed: int = 0, **kwargs) -> RaceEnv:
    """The interface for train/: make_env("Spielberg", seed=i), like gym.make("Pendulum-v1").
    Optional overrides, e.g. make_env("Spielberg", seed=0, reward_weights={"center": 0.1},
    max_laps=50) for a long eval run."""
    return RaceEnv(track_name=track_name, seed=seed, **kwargs)
