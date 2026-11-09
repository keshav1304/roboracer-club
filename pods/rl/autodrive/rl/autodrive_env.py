"""Gymnasium environment on the real AutoDRIVE Simulator, for stable-baselines3.

Same action and observation as the gym environment (reference/race_env_gym.py), so a policy sees the
same interface in both. The reward weights differ: see REWARD_WEIGHTS.

What is different, because the real simulator provides it:
- Physics: f1tenth_gym, the throttle fit, the grip cap and the random friction are gone.
- Collisions come from the simulator's own counter.
- Time: the simulator runs in real time, so one 20 Hz control step takes 50 ms of simulator time.
  Steps are timed on the simulator's clock, not the wall clock.
- Starts: the simulator can only reset to its spawn and ignores pose commands (tested 2026-11-06).
  For a random start, a scripted driver follows the centre line from the spawn to a random point
  before the episode begins. Those approach steps are not part of the episode.
- Stall and wrong-way rules (off by default): end the episode, with the collision penalty, when
  the car gains less than STALL_MIN_M in STALL_WINDOW_S, or falls WRONG_WAY_M behind its best.
- Track 'none': for a simulator whose track has no centre line file, progress is the distance
  driven forward and laps come from the simulator's lap counter.
"""
from __future__ import annotations

import math
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim_bridge import SimBridge, Telemetry, decode_lidar
from track import Track

# ---- shared contract, values and functions as in race_env.py ----------------------------------
LIDAR_BINS = 108
MAX_RANGE = 10.0
MAX_STEER = 0.5236               # rad at steering command 1.0
MAX_THROTTLE = 0.32              # throttle_command at action[1] = +1
SPEED_NORM = 8.0                 # m/s, speed scale in the observation
CONTROL_HZ = 20
MAX_LAPS = 10
MAX_STEPS = 3000                 # 150 s at 20 Hz
OBS_DIM = LIDAR_BINS + 2
# Changed from race_env.py on 2026-11-05: centre and steer-rate terms removed, progress 10 -> 20,
# collision 10 -> 30, speed 0 -> 0.5. 2026-11-06: stall penalty added, equal to a crash.
# 2026-11-07: back to four terms (now the 'basic' preset); the stall and wrong-way rules are off by
# default and the wall term is gone. 2026-11-07, later: the 'race_line' preset, the default, measures
# progress and laps along a given race line and adds the line-following, steer-rate and hard
# deceleration terms. Every term is listed in both presets so the logs have the same columns.
REWARD_PRESETS = {
    'race_line': {'progress': 20.0, 'lap_bonus': 100.0, 'collision': 30.0, 'speed': 0.0,
                  'race_line': 2.0, 'steer_rate': 0.1, 'decel': 0.05},
    'basic': {'progress': 20.0, 'lap_bonus': 50.0, 'collision': 30.0, 'speed': 0.5,
              'race_line': 0.0, 'steer_rate': 0.0, 'decel': 0.0},
}
REWARD_WEIGHTS = REWARD_PRESETS['race_line']
RACE_LINE_CAP_M = 1.0            # the line-following penalty stops growing beyond this distance
DECEL_FREE_MPS2 = 7.0            # deceleration up to this is free: lifting off the throttle alone gives
                                 # 6-7 m/s^2 (measured 2026-11-07), and the car has no brake
WRONG_WAY_M = 2.0                # with wrong_way_m on: end if the car falls this far behind its furthest point
NO_TRACK = 'none'                # track name for a simulator with no centre line file: see __init__
STALL_WINDOW_S = 2.0             # with stall_window_s on: end if the car gains less than STALL_MIN_M in this time
STALL_MIN_M = 1.0
APPROACH_MAX_SPEED = 3.0         # m/s, scripted driver on the way to a random start
APPROACH_MIN_SPEED = 1.2
APPROACH_LATERAL_ACCEL = 2.0     # m/s^2: the scripted driver slows to sqrt(this * radius) for corners
APPROACH_PREVIEW_M = 3.0         # how far ahead it looks for the tightest corner
THROTTLE_PER_MPS = 1 / 24.86     # throttle command per m/s of steady speed (measured)
HAIRPIN_RADIUS_M = 1.5           # corners tighter than this count as hairpins for random starts
HAIRPIN_LEAD_M = (3.0, 8.0)      # a hairpin start is this far before the corner
WHEELBASE = 0.33


def preprocess_scan(ranges) -> np.ndarray:
    """1081 (or any number of) LiDAR ranges -> 108 values in 0..1."""
    scan = np.asarray(ranges, dtype=np.float32)
    scan = np.clip(np.nan_to_num(scan, nan=MAX_RANGE, posinf=MAX_RANGE, neginf=0.0), 0.0, MAX_RANGE)
    bins = np.array([c.mean() for c in np.array_split(scan, LIDAR_BINS)], dtype=np.float32)
    return bins / MAX_RANGE


def build_observation(ranges, speed_mps: float, steering_rad: float) -> np.ndarray:
    return np.concatenate([preprocess_scan(ranges),
                           [np.clip(speed_mps / SPEED_NORM, 0.0, 1.0),
                            np.clip(steering_rad / MAX_STEER, -1.0, 1.0)]]).astype(np.float32)


def pursuit_steer(track: Track, f: Telemetry, s: float, lookahead: float = 1.0) -> float:
    """Steering action (-1..1) that follows the centre line: pure pursuit, lookahead in metres."""
    target = (s + lookahead) % track.length
    i = int(np.searchsorted(track.s_at, target, side='right')) - 1
    tx, ty = track.points[i]
    alpha = math.atan2(ty - f.y, tx - f.x) - f.yaw
    steer = math.atan2(2 * WHEELBASE * math.sin(alpha), lookahead)
    return float(np.clip(steer / MAX_STEER, -1.0, 1.0))


def action_to_autodrive(action) -> tuple[float, float]:
    """(throttle_command, steering_command)."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return float((a[1] + 1.0) * 0.5 * MAX_THROTTLE), float(a[0])


# ---- environment ------------------------------------------------------------------------------
class AutoDriveEnv(gym.Env):
    metadata = {'render_modes': [], 'render_fps': CONTROL_HZ}

    CONNECT_TIMEOUT_S = 300.0        # a simulator may need restarting
    STEP_TIMEOUT_S = 10.0
    SETTLE_S = 0.6                   # the car drops onto its suspension after a reset
    SETTLE_SPEED = 0.05

    def __init__(self, port: int = 4567, track_name: str = 'icra', reward_weights: dict | None = None,
                 race_line: str | None = None,
                 max_laps: int = MAX_LAPS, max_steps: int = MAX_STEPS, random_start: float = 0.0,
                 stall_window_s: float = 0.0, stall_min_m: float = STALL_MIN_M, wrong_way_m: float = 0.0):
        """random_start: chance (0..1) that an episode starts away from the spawn; half of those start
        just before a hairpin. stall_window_s and wrong_way_m above 0 turn on those episode rules,
        which end the episode with the collision penalty.

        track_name 'none' is for a track with no centre line file: progress is then the distance
        driven forward (displacement along the car's heading), laps come from the simulator's own lap
        counter, and random starts are not possible.

        race_line: path of a race line CSV (x_m, y_m in the simulator's frame, in driving order). With
        it, progress and laps are measured along the race line instead of the centre line, and the
        race_line weight penalises the distance from it. Random starts still follow the centre line."""
        super().__init__()
        unknown = set(reward_weights or {}) - set(REWARD_WEIGHTS)
        if unknown:
            raise ValueError(f'unknown reward weights {sorted(unknown)}; have {sorted(REWARD_WEIGHTS)}')
        self.w = dict(REWARD_WEIGHTS, **(reward_weights or {}))
        self.max_laps, self.max_steps = max_laps, max_steps
        self.random_start = random_start
        self.stall_steps = int(round(stall_window_s * CONTROL_HZ))
        self.stall_min_m = stall_min_m
        self.wrong_way_m = wrong_way_m
        self.has_track = track_name != NO_TRACK
        if not self.has_track:
            if random_start > 0:
                raise ValueError('random starts need a centre line; track none has none')
            self.track, self.track_len, self.hairpin_s = None, float('inf'), np.zeros(0)
        else:
            self._load_track(track_name)
        self.line = Track.from_csv(race_line) if race_line else None
        if self.line is not None and not self.has_track:
            raise ValueError('a race line needs a track with a centre line, not track none')
        self.bridge = SimBridge(port)
        self._init_spaces(port)

    def _load_track(self, track_name: str) -> None:
        self.track = Track.load(track_name)
        self.track_len = self.track.length
        heading = np.unwrap(np.arctan2(self.track.seg[:, 1], self.track.seg[:, 0]))
        turn = np.abs(np.diff(np.append(heading, heading[0] + 2 * np.pi * round((heading[-1] - heading[0]) / (2 * np.pi)))))
        turn = np.minimum(turn, 2 * np.pi - turn)
        # radius at each centre line point, from the heading change over about 1 m
        k = max(1, int(round(1.0 / (self.track_len / len(heading)))))
        turn_1m = np.convolve(np.concatenate([turn, turn[:k]]), np.ones(k), 'valid')[:len(turn)]
        self.radius = np.where(turn_1m > 1e-6, k * self.track.seg_len.mean() / np.maximum(turn_1m, 1e-6), 1e6)
        tight = self.radius < HAIRPIN_RADIUS_M
        self.hairpin_s = self.track.s_at[tight & ~np.roll(tight, 1)]     # where each tight corner begins

    def _init_spaces(self, port: int) -> None:
        self.port = port
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        low = np.concatenate([np.zeros(LIDAR_BINS + 1), -np.ones(1)]).astype(np.float32)
        self.observation_space = spaces.Box(low, np.ones(OBS_DIM, np.float32), dtype=np.float32)
        self._last_lidar = np.full(1081, MAX_RANGE, dtype=np.float32)

    # ---- helpers ------------------------------------------------------------------------------
    def _observe(self, f: Telemetry) -> np.ndarray:
        if f.lidar_b64:
            self._last_lidar = decode_lidar(f.lidar_b64)
        return build_observation(self._last_lidar, f.speed, f.steering)   # V1 Steering is in radians

    def _require_connection(self) -> None:
        if not self.bridge.wait_connected(self.CONNECT_TIMEOUT_S):
            raise TimeoutError(f'no simulator connected on port {self.port} after {self.CONNECT_TIMEOUT_S:.0f} s')

    def hold(self) -> None:
        """Freeze the car at the spawn while the policy updates: the simulator does not pause,
        and a car left on its last command would drive off during a slow update."""
        self.bridge.set_command(0.0, 0.0, reset=True)

    # ---- gym API ------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        while True:
            self._require_connection()
            self.bridge.set_command(0.0, 0.0, reset=True)
            if self.bridge.wait_frames(5, self.STEP_TIMEOUT_S) is None:
                continue
            self.bridge.set_command(0.0, 0.0, reset=False)
            f = self.bridge.wait_sim_time(self.SETTLE_S, self.STEP_TIMEOUT_S)
            while f is not None and abs(f.speed) > self.SETTLE_SPEED:
                f = self.bridge.wait_sim_time(0.05, self.STEP_TIMEOUT_S)
            if f is None or not f.lidar_b64:
                continue
            start_action = np.zeros(2, np.float32)
            if self.random_start > 0 and self.np_random.random() < self.random_start:
                if len(self.hairpin_s) and self.np_random.random() < 0.5:
                    target = (self.np_random.choice(self.hairpin_s)
                              - self.np_random.uniform(*HAIRPIN_LEAD_M))
                else:
                    target = self.np_random.uniform(0.0, self.track_len)
                s0 = self.track.cartesian_to_frenet(f.x, f.y)[0]
                f, start_action = self._approach(f, float((target - s0) % self.track_len))
                if f is None:
                    continue                 # crashed or stalled on the way: start again from the spawn
            break

        if self.line is not None:
            self.prev_s = self.line.cartesian_to_frenet(f.x, f.y)[0]
        else:
            self.prev_s = self.track.cartesian_to_frenet(f.x, f.y)[0] if self.has_track else 0.0
        self.prev_speed, self.prev_t = f.speed, self.bridge.sim_clock
        self._follow_s = None                # scripted test drivers keep their centre-line position here
        self.prev_xy, self.prev_lap_count = (f.x, f.y), f.lap_count
        self.base_collisions = f.collisions
        self.prev_action = start_action
        self.recent_progress = deque([0.0], maxlen=self.stall_steps + 1)
        self.max_progress = 0.0
        self.progress, self.steps, self.laps = 0.0, 0, 0
        self.t0 = self.bridge.sim_clock
        self.next_step_t = self.t0
        self.lap_start_t, self.lap_times = 0.0, []
        return self._observe(f), {'s': float(self.prev_s)}

    def _approach(self, f: Telemetry, distance: float):
        """Drive the scripted line `distance` metres from the spawn. Returns (frame, last action), or
        (None, None) if the car crashed or stopped responding on the way."""
        s = self.track.cartesian_to_frenet(f.x, f.y)[0]
        base_collisions, done = f.collisions, 0.0
        dt = 1.0 / CONTROL_HZ
        n = len(self.radius)
        preview = max(1, int(APPROACH_PREVIEW_M / (self.track_len / n)))
        limit = self.bridge.sim_clock + 2 * distance / APPROACH_MIN_SPEED + 5.0
        while done < distance:
            i = int(np.searchsorted(self.track.s_at, s % self.track_len, side='right')) - 1
            r_min = float(self.radius[(i + np.arange(preview)) % n].min())
            speed = float(np.clip(math.sqrt(APPROACH_LATERAL_ACCEL * r_min), APPROACH_MIN_SPEED, APPROACH_MAX_SPEED))
            throttle_action = float(np.clip(2 * speed * THROTTLE_PER_MPS / MAX_THROTTLE - 1, -1.0, 1.0))
            lookahead = float(np.clip(0.4 * abs(f.speed) + 0.6, 0.8, 2.0))   # longer when faster, or it weaves
            action = np.array([pursuit_steer(self.track, f, s, lookahead), throttle_action], np.float32)
            throttle_cmd, steering_cmd = action_to_autodrive(action)
            self.bridge.set_command(throttle_cmd, steering_cmd)
            f = self.bridge.wait_sim_time(dt, self.STEP_TIMEOUT_S)
            if f is None or f.collisions > base_collisions or self.bridge.sim_clock > limit:
                return None, None
            s_new = self.track.cartesian_to_frenet(f.x, f.y, s_guess=s)[0]
            done += self.track.wrap_ds(s_new, s)
            s = s_new
        return f, action

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32).reshape(2), -1.0, 1.0)
        throttle_cmd, steering_cmd = action_to_autodrive(action)
        self.bridge.set_command(throttle_cmd, steering_cmd)
        # A fixed schedule, so overshooting to the next frame does not build up and slow the rate.
        # If a step ran late by more than a whole period, start the schedule again from now.
        dt = 1.0 / CONTROL_HZ
        now = self.bridge.sim_clock
        self.next_step_t = max(self.next_step_t + dt, now + 0.5 * dt)
        f = self.bridge.wait_sim_clock(self.next_step_t, self.STEP_TIMEOUT_S)
        if f is None:                      # simulator crashed or was closed: end the episode cleanly
            obs = np.zeros(OBS_DIM, np.float32)
            return obs, 0.0, False, True, {'sim_stalled': True, 'collision': False, 'stalled': False,
                                           'wrong_way': False, 'laps': self.laps,
                                           'lap_times': list(self.lap_times)}

        collision = f.collisions > self.base_collisions
        if self.line is not None:            # progress and laps along the race line; ey is the distance from it
            s, ey = self.line.cartesian_to_frenet(f.x, f.y, s_guess=self.prev_s)
            ds = self.line.wrap_ds(s, self.prev_s)
            lap_completed = self.progress + ds >= (self.laps + 1) * self.line.length
        elif self.has_track:
            s, ey = self.track.cartesian_to_frenet(f.x, f.y, s_guess=self.prev_s)
            ds = self.track.wrap_ds(s, self.prev_s)
            lap_completed = self.progress + ds >= (self.laps + 1) * self.track_len
        else:                              # distance driven forward, laps from the simulator's counter
            ds = (f.x - self.prev_xy[0]) * math.cos(f.yaw) + (f.y - self.prev_xy[1]) * math.sin(f.yaw)
            s, ey = self.progress + ds, 0.0
            lap_completed = f.lap_count > self.prev_lap_count
        self.prev_s, self.prev_xy, self.prev_lap_count = s, (f.x, f.y), f.lap_count
        self.progress += ds
        self.steps += 1
        t = self.bridge.sim_clock - self.t0

        if lap_completed:
            self.laps += 1
            self.lap_times.append(round(t - self.lap_start_t, 3))
            self.lap_start_t = t
        obs = self._observe(f)
        self.recent_progress.append(self.progress)
        self.max_progress = max(self.max_progress, self.progress)
        wrong_way = (not collision and self.wrong_way_m > 0
                     and self.progress < self.max_progress - self.wrong_way_m)
        stalled = (not collision and not wrong_way and self.stall_steps > 0
                   and len(self.recent_progress) > self.stall_steps
                   and self.progress - self.recent_progress[0] < self.stall_min_m)
        now_t = self.bridge.sim_clock
        decel = (self.prev_speed - f.speed) / (now_t - self.prev_t) if now_t > self.prev_t else 0.0
        self.prev_speed, self.prev_t = f.speed, now_t
        terms = {'r_progress': self.w['progress'] * ds,
                 'r_lap': self.w['lap_bonus'] if lap_completed else 0.0,
                 'r_collision': -self.w['collision'] if (collision or stalled or wrong_way) else 0.0,
                 'r_speed': self.w['speed'] * f.speed,
                 'r_race_line': -self.w['race_line'] * min(abs(ey), RACE_LINE_CAP_M) if self.line is not None else 0.0,
                 'r_steer_rate': -self.w['steer_rate'] * abs(float(action[0]) - float(self.prev_action[0])),
                 'r_decel': -self.w['decel'] * max(0.0, decel - DECEL_FREE_MPS2) if not collision else 0.0}
        reward = sum(terms.values())

        terminated = collision or stalled or wrong_way
        truncated = self.laps >= self.max_laps or self.steps >= self.max_steps
        self.prev_action = action
        info = {'collision': collision, 'stalled': stalled, 'wrong_way': wrong_way, 'decel_mps2': float(decel),
                'laps': self.laps, 'lap_times': list(self.lap_times),
                's': float(s), 'ey': float(ey), 'speed': f.speed, 'yaw_rate': f.yaw_rate,
                'progress_m': float(self.progress), 'sim_time': round(t, 3), 'throttle_cmd': throttle_cmd,
                'steering_feedback': f.steering, 'sim_lap_count': f.lap_count, 'sim_last_lap_time': f.last_lap_time,
                **{k: float(v) for k, v in terms.items()}}
        return obs, float(reward), terminated, truncated, info

    def close(self):
        self.bridge.set_command(0.0, 0.0)
