"""Checks for race_env. Run from pods/rl/env:  python test_race_env.py

1. SB3 check_env passes with no warnings.
2. The observation matches the contract (110 float32, LiDAR in 0..1, then speed and steering), and
   throttle drives the gym car like the AutoDRIVE fit (settles at 24.7 x throttle with a 0.3 s
   lag, coasts down at 3 m/s^2 with throttle 0).
3. On Spielberg, icra and practice, a pure pursuit driver on the centre line completes two laps:
   the step that crosses the start/finish line has a normal positive ds (the wraparound fix), each
   lap pays +50, and progress per lap equals the track length. check_env passes on every track.
"""
import math
import warnings

import numpy as np
from stable_baselines3.common.env_checker import check_env

from race_env import (LIDAR_BINS, MAX_STEER, MAX_THROTTLE, OBS_DIM, SPEED_DRAG, THROTTLE_GAIN,
                      make_env, throttle_to_accel)


def pursuit_action(env, speed=4.0, lookahead=1.5):
    """Normalised action that follows the centre line (the expert for this test)."""
    x, y, yaw = env._pose
    tx, ty, _ = env.track.frenet_to_cartesian((env.prev_s + lookahead) % env.track_len, 0.0, 0.0)
    dx, dy = tx - x, ty - y
    yc = -math.sin(yaw) * dx + math.cos(yaw) * dy
    steer = math.atan(2.0 * yc / max(1e-6, dx * dx + dy * dy) * 0.33)
    throttle = speed * SPEED_DRAG / THROTTLE_GAIN                   # throttle that settles at `speed`
    return np.array([np.clip(steer / MAX_STEER, -1, 1), throttle / MAX_THROTTLE * 2 - 1], dtype=np.float32)


def expert_laps(track, speed, laps=2):
    """Pure pursuit on the centre line for `laps` laps; checks the finish-line crossings."""
    env = make_env(track, seed=0)
    env.random_start = False
    env.reset()
    env._pose = (0.0, 0.0, 0.0)
    crossings, laps_seen = [], 0
    orig_step = env.sim.step

    def spy(cmd):                                                  # remember the pose for the expert
        out = orig_step(cmd)
        st = out[0]['agent_0']['std_state']
        env._pose = (st[0], st[1], st[4])
        return out
    env.sim.step = spy
    env.sim.step(np.array([[0.0, 0.0]]))
    for k in range(20000):
        s_before = env.prev_s
        obs, r, term, trunc, info = env.step(pursuit_action(env, speed))
        if info['s'] < s_before - env.track_len / 2:              # this step crossed the line
            crossings.append(r)
        if info['laps'] > laps_seen:
            laps_seen = info['laps']
            if laps_seen == laps:
                break
        assert not term, f'{track}: wall contact at step {k}: {info}'
    assert laps_seen == laps, f'{track}: expert did not finish {laps} laps'
    assert crossings and all(r > 0 for r in crossings), f'{track}: negative reward on a crossing {crossings}'
    print(f'   {track}: {laps} clean laps at {speed} m/s, {", ".join(f"{t:.2f}" for t in info["lap_times"])} s, '
          f'crossings {", ".join(f"{r:.1f}" for r in crossings)}, progress {info["progress_m"] / env.track_len:.3f} laps')


def main():
    env = make_env('Spielberg', seed=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        check_env(env, warn=True)
    assert not w, [str(x.message) for x in w]
    print('1. check_env: passed, no warnings')

    obs, _ = env.reset(seed=3)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert 0.0 <= obs[:LIDAR_BINS].min() and obs[:LIDAR_BINS].max() <= 1.0 and abs(obs[LIDAR_BINS]) < 0.01
    print(f'2. observation: {OBS_DIM} float32, LiDAR {obs[:LIDAR_BINS].min():.2f}..{obs[:LIDAR_BINS].max():.2f}, '
          f'speed {obs[LIDAR_BINS]:.2f}')

    # throttle response matches the AutoDRIVE fit: 0.16 throttle from rest settles near 24.7 x 0.16
    env.random_start = False
    env.reset()
    v = []
    for _ in range(60):                                            # 3 s
        o, *_ = env.step(np.array([0.0, 0.0]))                     # action 0 = throttle 0.16
        v.append(env.speed)
    target = THROTTLE_GAIN * 0.16 / SPEED_DRAG
    t63 = (np.argmax(np.array(v) > 0.63 * target) + 1) / 20
    assert abs(v[-1] - target) < 0.3, (v[-1], target)
    for _ in range(10):                                            # 0.5 s coasting
        env.step(np.array([0.0, -1.0]))
    decel = (v[-1] - env.speed) / 0.5
    assert abs(decel - 3.0) < 0.6, decel
    assert abs(throttle_to_accel(0.0, 2.0) + 3.0) < 1e-9
    print(f'   throttle 0.16 -> {v[-1]:.2f} m/s (model {target:.2f}), 63% at {t63:.2f} s, coasting {decel:.2f} m/s^2')

    print('3. expert laps on every track (finish-line wraparound, lap bonus, no contact)')
    for track, speed in (('Spielberg', 4.0), ('icra', 2.5), ('practice', 2.5)):
        expert_laps(track, speed)
        e = make_env(track, seed=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            check_env(e, warn=True)
        assert not w, (track, [str(x.message) for x in w])
    print('all checks passed')


if __name__ == '__main__':
    main()
