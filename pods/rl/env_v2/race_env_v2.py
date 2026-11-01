"""env_v2: end to end PPO on a car that brakes, slides and lags like AutoDRIVE's, on many tracks.

    from race_env_v2 import make_env
    env = make_env(seed=0)                           # training: random track, random physics, random start
    env = make_env(tracks=['icra'], randomize=False, random_start=False, seed=0)   # evaluation on ICRA
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

What the policy does not get to rely on, by design (the reasons are in README.md):
- the physics. Each episode draws one friction value MU that scales accelerating, braking and turning
  together, plus the tyre curve, steering, delays and mass around the values fitted to AutoDRIVE
  (vehicle.DR_RANGES). 30 % of episodes use the older KS + slide model instead (vehicle.MODEL_EMPIRICAL).
- the timing. Steering and throttle reach the car 20 to 50 ms and 20 to 60 ms after they are sent, the
  observation can be up to 20 ms old, the control step is 50 ms +/- 5 ms, and up to 2 % of steps arrive
  100 to 200 ms late (what AutoDRIVE does when its window has the focus). The late steps are what crash
  drivers, so they are what the policy most needs to see.
- the track. Random track from the bank (tracks.training_bank), mirrored and reversed, random start.

Action, 2 floats in [-1, 1] (contract.action_to_commands)
    action[0]  share of the steering the envelope allows at the current speed, +1 = left
    action[1]  below -0.9: throttle 0, which brakes all four wheels. Above: throttle 0.005 to 0.32.
Observation, 114 float32 (contract.build_observation)
    108 LiDAR bins, speed, last steering command, last throttle action, yaw rate, two accelerations.
Reward per control step (REWARD_WEIGHTS), "v4"
    10 * metres along the centre line - 1 per 50 ms + 50 per lap - 100 on a crash (wall, slide, spin or
    stall, all end the episode) - 3.5 * max(0, demand / limit - 0.85)^2 - 0.1 * |steering change|
    - 0.02 * |throttle action change| + shaping (gamma * phi(s') - phi(s), phi = -k (ey - ey_teacher)^2, k
    set by the trainer and turned down to 0).
"""
from __future__ import annotations

import math
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import contract as C                                    # noqa: E402
import tracks as T                                      # noqa: E402
import vehicle as V                                     # noqa: E402
from teacher import Teacher                             # noqa: E402

REWARD_WEIGHTS = {'progress': 10.0, 'time': 1.0, 'lap_bonus': 50.0, 'crash': 100.0, 'margin': 3.5,
                  'steer_rate': 0.1, 'thr_rate': 0.02, 'shaping': 0.0}
# Late steps are the timing fault that matters: 2 % of steps arriving 100 to 200 ms late crash pure pursuit
# x1.15 in 14 of 20 episodes, while 20 ms of observation delay or +/-5 ms of jitter crash none (validate_v2.py).
TIMING = dict(late_prob=(0.0, 0.02), late_extra=(0.05, 0.15), step_jitter=0.005, obs_delay=(0.0, 0.020))
SENSOR = dict(lidar_noise=(0.01, 0.03), lidar_dropout=(0.0, 0.01), speed_noise=0.02)
SPIN_BETA = math.radians(15.0)
SLIDE_TIME = 0.05             # s of tyre sliding in one step that counts as a slide
STALL_SPEED, STALL_TIME, STALL_GRACE = 0.3, 1.0, 2.0


class RaceEnvV2(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, tracks=None, seed: int = 0, randomize: bool = True, random_start: bool = True,
                 empirical_share: float = 0.3, reward_weights: dict | None = None, slide_is_crash: bool = True,
                 use_envelope: bool = True, grip_shield: bool = False, n_generated: int = 200,
                 tracks_per_worker: int = 40, max_steps: int = 3000, max_laps: int = 10, gamma: float = 0.995,
                 timing: bool = True, model: int | None = None, holdout=T.HOLDOUT,
                 focus_track: str | None = None, focus_share: float = 0.0):
        super().__init__()
        unknown = set(reward_weights or {}) - set(REWARD_WEIGHTS)
        if unknown:
            raise ValueError(f'unknown reward weights {sorted(unknown)}; have {sorted(REWARD_WEIGHTS)}')
        self.w = dict(REWARD_WEIGHTS, **(reward_weights or {}))
        self.randomize, self.random_start, self.empirical_share = randomize, random_start, empirical_share
        self.slide_is_crash, self.use_envelope, self.grip_shield = slide_is_crash, use_envelope, grip_shield
        self.max_steps, self.max_laps, self.gamma, self.timing = max_steps, max_laps, gamma, timing
        self.fixed_model = model
        # focus_share of episodes run on focus_track as raced (not mirrored or reversed), the rest on the bank
        self.focus = (T.REAL[focus_track], False, False) if focus_track and focus_share > 0 else None
        self.focus_share = focus_share
        self._rng0 = np.random.default_rng(seed)
        if tracks is None:                                   # training bank, a fixed subset per worker
            bank = T.training_bank(n_generated, holdout=tuple(holdout or ()))
            pick = self._rng0.choice(len(bank), size=min(tracks_per_worker, len(bank)), replace=False)
            real = [b for b in bank if not os.path.basename(b[0]).startswith('gen')]
            self.bank = list({b: None for b in real + [bank[i] for i in pick]})
        else:
            self.bank = []
            for name in tracks:
                folder = T.REAL.get(name) or (T.ensure_generated(int(name[3:])) if name.startswith('gen') else name)
                self.bank.append((folder, name.endswith('_m'), name.endswith('_r')))
        self._cache = {}
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        low = np.concatenate([np.zeros(C.LIDAR_BINS + 1), -np.ones(5)]).astype(np.float32)
        self.observation_space = spaces.Box(low, np.ones(C.OBS_DIM, np.float32), dtype=np.float32)
        self._seed = seed
        self.car = None

    # ---- setup ---------------------------------------------------------------------------------
    def _track(self, entry):
        if entry not in self._cache:
            folder, mirror, reverse = entry
            if os.path.basename(folder).startswith('gen'):
                T.ensure_generated(int(os.path.basename(folder)[3:]))
            tr = T.RaceTrack(folder, mirror=mirror, reverse=reverse)
            tr.teacher = Teacher(tr, cache_dir=os.path.join(T.GEN_DIR, 'teacher'))
            self._cache[entry] = tr
        return self._cache[entry]

    def set_shaping(self, k: float):
        self.w['shaping'] = float(k)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed if seed is not None else self._seed)
        self._seed = None
        r = self.np_random
        if self.focus is not None and r.uniform() < self.focus_share:
            self.track = self._track(self.focus)
        else:
            self.track = self._track(self.bank[int(r.integers(len(self.bank)))])
        model = self.fixed_model if self.fixed_model is not None else (
            V.MODEL_EMPIRICAL if r.uniform() < self.empirical_share else V.MODEL_TWIN)
        self.params = V.sample_params(r, model, self.randomize)
        if not self.timing:
            self.params[V.IDX['delay_s']] = self.params[V.IDX['delay_t']] = 0.0
        self.car = V.Car(self.params)
        # sensors and timing for this episode
        u = (lambda lo_hi: float(r.uniform(*lo_hi))) if self.randomize else (lambda lo_hi: float(lo_hi[0]))
        self.lidar_noise = u(SENSOR['lidar_noise'])
        self.lidar_dropout = u(SENSOR['lidar_dropout'])
        self.late_prob = u(TIMING['late_prob']) if self.timing else 0.0
        self.obs_delay = u(TIMING['obs_delay']) if self.timing else 0.0
        # start
        tr = self.track
        for _ in range(50):
            if self.random_start:
                s0 = float(r.uniform(0, tr.length))
                i = int(round(s0 / 0.1)) % tr.n
                ey = float(r.uniform(-1, 1)) * max(0.0, 0.6 * (min(tr.wr[i], tr.wl[i]) - 0.3))
                x, y, yaw, i = tr.pose_at(s0, ey, float(r.uniform(-0.2, 0.2)))
                v0 = float(r.uniform(0.0, 5.0)) if r.uniform() < 0.7 else 0.0
            else:
                x, y, yaw, i = tr.pose_at(0.0)
                v0 = 0.0
            if all(tr.free(x + dx, y + dy, 0.2) for dx, dy in ((0, 0), (0.25 * math.cos(yaw), 0.25 * math.sin(yaw)),
                                                               (-0.25 * math.cos(yaw), -0.25 * math.sin(yaw)))):
                break
        self.car.reset(x, y, yaw, v0)
        self.car.command(0.0, v0 / 24.4)
        tr.teacher.reset()
        self.prev_s, _, self.prev_i = tr.project(x, y)
        self.progress, self.steps, self.laps, self.lap_start_t, self.lap_times = 0.0, 0, 0, 0.0, []
        self.last_steer, self.last_thr_action = 0.0, -1.0 if v0 == 0 else float(C.commands_to_action(0, v0 / 24.4, v0)[1])
        self.v_hist = [v0, v0, v0]
        self._obs_pose = (x, y, yaw, 0.0)
        self.snapshot = (x, y, yaw, v0)
        self._stall_t, self._shield_over, self._coasting = 0.0, 0, False
        self.prev_phi = self._phi(self.prev_i, self._ey(x, y))
        obs = self._observe(x, y, yaw, v0, 0.0)
        return obs, self._info_base()

    # ---- helpers --------------------------------------------------------------------------------
    def _ey(self, x, y):
        return self.track.project(x, y, self.prev_i)[1]

    def _phi(self, i, ey):
        if not self.w['shaping']:
            return 0.0
        return -self.w['shaping'] * (ey - self.track.teacher.ey_at(i)) ** 2

    def _observe(self, x, y, yaw, v, yaw_rate):
        r = self.np_random
        scan = self.track.scan(x + C.LIDAR_X * math.cos(yaw), y + C.LIDAR_X * math.sin(yaw), yaw)
        if self.lidar_noise:
            scan = scan + r.normal(0.0, self.lidar_noise, size=scan.shape)
        if self.lidar_dropout:
            scan = np.where(r.uniform(size=scan.shape) < self.lidar_dropout, C.MAX_RANGE, scan)
        v_meas = v * (1.0 + r.normal(0.0, SENSOR['speed_noise'])) if self.randomize else v
        self.v_hist = [v_meas] + self.v_hist[:2]
        self.v_obs = v_meas
        return C.build_observation(scan, v_meas, self.last_steer, self.last_thr_action, yaw_rate, self.v_hist)

    def _info_base(self):
        p = self.params
        return {'track': self.track.name, 'model': 'twin' if p[0] == V.MODEL_TWIN else 'empirical',
                'mu': float(p[V.IDX['mu']]), 'k_couple': float(p[V.IDX['k_couple']]),
                'delay_s': float(p[V.IDX['delay_s']]), 'teacher_lap': self.track.teacher.lap_time}

    def expert_action(self) -> np.ndarray:
        """The teacher's action for the current (true) state."""
        c = self.car
        return self.track.teacher.action(c.x, c.y, c.yaw, c.speed, self.use_envelope)

    # ---- step -------------------------------------------------------------------------------------
    def step(self, action):
        action = np.clip(np.asarray(action, np.float32).reshape(2), -1.0, 1.0)
        steer_cmd, thr_cmd = C.action_to_commands(action, self.v_obs, self.use_envelope)
        return self._advance(steer_cmd, thr_cmd, float(action[1]))

    def step_commands(self, steer_cmd: float, thr_cmd: float):
        """Step with raw AutoDRIVE commands (for classic drivers and validation), no envelope."""
        a1 = float(C.commands_to_action(steer_cmd, thr_cmd, self.v_obs, False)[1])
        return self._advance(float(np.clip(steer_cmd, -1, 1)), float(np.clip(thr_cmd, 0, 1)), a1)

    def _advance(self, steer_cmd, thr_cmd, a1):
        shielded = False
        if self.grip_shield:                                     # the P5 deploy node's rule, optional here
            dem = C.demand(self.v_obs, steer_cmd * C.MAX_STEER)
            self._shield_over = self._shield_over + 1 if dem > 9.2 else 0
            self._coasting = (self._coasting and dem > 8.8) or self._shield_over >= 2
            if self._coasting:
                thr_cmd, shielded = 0.0, True
        self.car.command(steer_cmd, thr_cmd)
        r = self.np_random
        dt = C.DT_CONTROL
        if self.timing:
            dt += float(r.uniform(-TIMING['step_jitter'], TIMING['step_jitter']))
            if r.uniform() < self.late_prob:
                dt += float(r.uniform(*TIMING['late_extra']))
        ed, ox, oy, res = self.track.edt, self.track.ox, self.track.oy, self.track.res
        t_obs = max(0.001, dt - self.obs_delay)
        out1 = self.car.run(t_obs, ed, ox, oy, res)
        snap = (self.car.x, self.car.y, self.car.yaw, self.car.speed)
        out2 = self.car.run(dt - t_obs, ed, ox, oy, res) if out1[V.O_COL] == 0 and dt > t_obs else np.zeros(6)
        out = np.maximum(out1, out2)
        out[V.O_TSF], out[V.O_TSR] = out1[V.O_TSF] + out2[V.O_TSF], out1[V.O_TSR] + out2[V.O_TSR]
        c = self.car
        wall = bool(out[V.O_COL] > 0)
        slide = bool(out[V.O_TSF] > SLIDE_TIME or out[V.O_TSR] > SLIDE_TIME)
        spin = bool(out[V.O_BETA] > SPIN_BETA)
        self._stall_t = self._stall_t + dt if (c.speed < STALL_SPEED and c.t > STALL_GRACE) else 0.0
        stall = self._stall_t > STALL_TIME
        crash = wall or spin or stall or (slide and self.slide_is_crash)
        # progress
        s, ey, i = self.track.project(c.x, c.y, self.prev_i)
        ds = s - self.prev_s
        L = self.track.length
        if ds < -L / 2:
            ds += L
        elif ds > L / 2:
            ds -= L
        self.prev_s, self.prev_i = s, i
        self.progress += ds
        self.steps += 1
        lap = self.progress >= (self.laps + 1) * L
        if lap:
            self.laps += 1
            self.lap_times.append(round(c.t - self.lap_start_t, 3))
            self.lap_start_t = c.t
        phi = self._phi(i, ey)
        dem = C.demand(self.v_obs, steer_cmd * C.MAX_STEER)
        ratio = dem / C.nominal_limit(steer_cmd * C.MAX_STEER)
        w = self.w
        terms = {'r_progress': w['progress'] * ds,
                 'r_time': -w['time'] * dt / C.DT_CONTROL,
                 'r_lap': w['lap_bonus'] if lap else 0.0,
                 'r_crash': -w['crash'] if crash else 0.0,
                 'r_margin': -w['margin'] * max(0.0, ratio - 0.85) ** 2,
                 'r_steer_rate': -w['steer_rate'] * abs(steer_cmd - self.last_steer),
                 'r_thr_rate': -w['thr_rate'] * abs(a1 - self.last_thr_action),
                 'r_shaping': (self.gamma * phi - self.prev_phi) if w['shaping'] else 0.0}
        self.prev_phi = phi
        reward = float(sum(terms.values()))
        # observation from the (possibly stale) snapshot
        px, py, pyaw, _ = self._obs_pose
        yaw_rate = math.atan2(math.sin(snap[2] - pyaw), math.cos(snap[2] - pyaw)) / max(1e-3, dt)
        self._obs_pose = (*snap[:3], c.t)
        self.last_steer, self.last_thr_action = steer_cmd, a1
        self.snapshot = snap                                    # what the sensors saw, for validation tools
        obs = self._observe(snap[0], snap[1], snap[2], snap[3], yaw_rate)
        terminated = crash
        truncated = (not crash) and (self.laps >= self.max_laps or self.steps >= self.max_steps)
        info = dict(self._info_base(), wall=wall, slide=slide, spin=spin, stall=stall, crash=crash, laps=self.laps,
                    lap_times=list(self.lap_times), s=float(s), ey=float(ey), speed=c.speed, sim_time=c.t,
                    steer_cmd=steer_cmd, thr_cmd=thr_cmd, shield=shielded, demand_ratio=ratio,
                    slip_front=float(out[V.O_SF]), slip_rear=float(out[V.O_SR]), progress_m=self.progress,
                    **{k: float(v) for k, v in terms.items()})
        return obs, reward, terminated, truncated, info


def make_env(seed: int = 0, **kwargs) -> RaceEnvV2:
    return RaceEnvV2(seed=seed, **kwargs)
