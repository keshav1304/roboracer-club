"""Checks for race_env. Run from pods/rl/env:  python test_race_env.py

1. SB3 check_env passes with no warnings, end to end and in residual mode.
2. The observation matches the contract (110 float32, 112 in residual mode, LiDAR in 0..1, then speed and
   the last steering command).
3. Throttle drives the gym car like AutoDRIVE (ramps at about 4.7 m/s^2 to 24.6 x throttle, engine
   braking about 5.7 m/s^2 at throttle 0, a little less in the first 0.5 s because of the 40 ms delay).
4. Grip: full lock at speed makes the front slide (it cannot keep cornering hard), and the grip shield
   lifts off the throttle when the commanded corner asks for too much.
5. On icra, practice and Spielberg the base driver (pure pursuit, residual mode, zero policy action)
   completes two clean laps: the step that crosses the start/finish line has a normal positive ds (the
   wraparound fix) and pays the lap bonus. On icra at x1.15 its flying lap is close to AutoDRIVE's 12.67 s.
"""
import warnings

import numpy as np
from stable_baselines3.common.env_checker import check_env

from race_env import LIDAR_BINS, MAX_THROTTLE, OBS_DIM, SPEED_PER_THROTTLE, make_env

NOMINAL = dict(randomize=False, throttle_jitter=0.0, grip_jitter=0.0, random_start=False)
RESIDUAL = dict(residual=0.4, steer_residual=0.0)


def no_warnings(env, label):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        check_env(env, warn=True)
    assert not w, (label, [str(x.message) for x in w])


def base_laps(track, scale, laps=2):
    """The base driver alone for `laps` laps; checks the finish-line crossings."""
    env = make_env(track, seed=0, max_laps=laps, ref_speed_scale=scale, **RESIDUAL, **NOMINAL)
    env.reset()
    crossings = []
    for k in range(10000):
        s_before = env.prev_s
        obs, r, term, trunc, info = env.step(np.zeros(2, np.float32))
        if info['s'] < s_before - env.track_len / 2:                # this step crossed the line
            crossings.append(r)
        assert not term, f'{track}: contact at step {k}: {info}'
        if trunc:
            break
    assert info['laps'] == laps, f'{track}: base driver did not finish {laps} laps'
    assert crossings and all(r > 40 for r in crossings), f'{track}: crossing rewards {crossings}'
    print(f'   {track} x{scale}: {laps} clean laps, {", ".join(f"{t:.2f}" for t in info["lap_times"])} s, '
          f'crossings {", ".join(f"{r:.1f}" for r in crossings)}')
    return info['lap_times']


def main():
    no_warnings(make_env('icra', seed=0), 'end to end')
    no_warnings(make_env('icra', seed=0, **RESIDUAL), 'residual')
    print('1. check_env: passed, no warnings, end to end and residual')

    env = make_env('icra', seed=0, **NOMINAL)
    obs, _ = env.reset()
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert 0.0 <= obs[:LIDAR_BINS].min() and obs[:LIDAR_BINS].max() <= 1.0 and abs(obs[LIDAR_BINS]) < 0.01
    obs, *_ = env.step(np.array([0.5, 0.0], np.float32))
    assert abs(obs[LIDAR_BINS + 1] - 0.5) < 1e-6, obs[LIDAR_BINS + 1]
    res = make_env('icra', seed=0, **RESIDUAL, **NOMINAL).reset()[0]
    assert res.shape == (OBS_DIM + 2,)
    print(f'2. observation: {OBS_DIM} float32 ({OBS_DIM + 2} residual), LiDAR {obs[:LIDAR_BINS].min():.2f}..'
          f'{obs[:LIDAR_BINS].max():.2f}, obs[109] = last steering command')

    env = make_env('Spielberg', seed=0, **NOMINAL)                 # long straight at the start
    env.reset()
    v = []
    for _ in range(60):                                            # 3 s at action 0 = throttle 0.16
        env.step(np.array([0.0, 0.0], np.float32))
        v.append(env.speed)
    target = SPEED_PER_THROTTLE * 0.5 * MAX_THROTTLE
    ramp = v[9] / 0.5                                              # first 0.5 s, minus the 40 ms delay
    assert abs(v[-1] - target) < 0.3, (v[-1], target)
    assert 3.8 < ramp < 4.8, ramp
    for _ in range(10):                                            # 0.5 s at throttle 0
        env.step(np.array([0.0, -1.0], np.float32))
    decel = (v[-1] - env.speed) / 0.5
    assert 4.9 < decel < 6.0, decel
    print(f'3. throttle 0.16 -> {v[-1]:.2f} m/s (AutoDRIVE {target:.2f}), ramp {ramp:.1f} m/s^2, '
          f'throttle 0 {decel:.1f} m/s^2')

    for shield in (False, True):
        env = make_env('Spielberg', seed=0, grip_shield=shield, **NOMINAL)
        env.reset()
        for _ in range(30):                                        # to about 4 m/s, straight
            env.step(np.array([0.0, 0.0], np.float32))
        slid, shielded, lat = 0.0, 0, []
        for _ in range(16):                                        # 0.8 s at full left lock
            o, r, term, trunc, info = env.step(np.array([1.0, 0.0], np.float32))
            slid += info['sliding']
            shielded += info['shield']
            lat.append(info['speed'] ** 2 * np.tan(env.state[2]) / env.wheelbase)
            if term:
                break
        late = float(np.mean(lat[-6:]))
        if not shield:
            asked = info['speed'] ** 2 * np.tan(0.5236) / env.wheelbase
            assert slid > 0 and late < 7.5 < asked, (slid, late, asked)
            print(f'4. grip: full lock at 4 m/s slides ({slid:.1f} control steps) and corners at {late:.1f} m/s^2, '
                  f'not the {asked:.0f} the wheel asks for')
        else:
            assert shielded > 0, shielded
            print(f'   grip shield: lifted off the throttle for {shielded} of {len(lat)} steps')

    print('5. base driver laps (finish-line wraparound, lap bonus, no contact)')
    lap = base_laps('icra', 1.15)[1]
    assert 12.3 < lap < 12.9, lap                                  # AutoDRIVE 12.67 s
    base_laps('practice', 1.0)
    base_laps('Spielberg', 1.0)
    print('all checks passed')


if __name__ == '__main__':
    main()
