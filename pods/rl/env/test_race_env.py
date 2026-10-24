"""Checks for race_env. Run from pods/rl/env:  python test_race_env.py

1. SB3 check_env passes with no warnings.
2. The observation matches the contract (110 float32, LiDAR in 0..1, previous action at the end).
3. A pure pursuit driver on the centre line completes laps: the step that crosses the start/finish
   line has a normal positive ds (the wraparound fix), each lap pays +50, and progress per lap
   equals the track length.
"""
import math
import warnings

import numpy as np
from stable_baselines3.common.env_checker import check_env

from race_env import LIDAR_BINS, MAX_STEER, OBS_DIM, V_MAX, make_env


def pursuit_action(env, speed=4.0, lookahead=1.5):
    """Normalised action that follows the centre line (the expert for this test)."""
    x, y, yaw = env._pose
    tx, ty, _ = env.track.frenet_to_cartesian((env.prev_s + lookahead) % env.track_len, 0.0, 0.0)
    dx, dy = tx - x, ty - y
    yc = -math.sin(yaw) * dx + math.cos(yaw) * dy
    steer = math.atan(2.0 * yc / max(1e-6, dx * dx + dy * dy) * 0.33)
    return np.array([np.clip(steer / MAX_STEER, -1, 1), speed / V_MAX * 2 - 1], dtype=np.float32)


def main():
    env = make_env('Spielberg', seed=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        check_env(env, warn=True)
    assert not w, [str(x.message) for x in w]
    print('1. check_env: passed, no warnings')

    obs, _ = env.reset(seed=3)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert 0.0 <= obs[:LIDAR_BINS].min() and obs[:LIDAR_BINS].max() <= 1.0 and not obs[LIDAR_BINS:].any()
    print(f'2. observation: {OBS_DIM} float32, LiDAR {obs[:LIDAR_BINS].min():.2f}..{obs[:LIDAR_BINS].max():.2f}')

    env = make_env('Spielberg', seed=0)
    env.random_start = False
    env.reset()
    env._pose = (0.0, 0.0, 0.0)
    rewards, crossings, laps_seen = [], [], 0
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
        obs, r, term, trunc, info = env.step(pursuit_action(env))
        rewards.append(r)
        if info['s'] < s_before - env.track_len / 2:              # this step crossed the line
            crossings.append((k, r))
        if info['laps'] > laps_seen:
            laps_seen = info['laps']
            print(f'   lap {laps_seen}: {info["lap_times"][-1]:.2f} s, step reward {r:.1f}')
            if laps_seen == 2:
                break
        assert not term, f'wall contact at step {k}: {info}'
    assert laps_seen == 2, 'expert did not finish two laps'
    assert crossings and all(r > 0 for _, r in crossings), f'negative reward on a line crossing: {crossings}'
    print(f'3. two clean laps, {len(crossings)} line crossings all positive '
          f'({", ".join(f"{r:.1f}" for _, r in crossings)}), progress {info["progress_m"]:.1f} m '
          f'= {info["progress_m"] / env.track_len:.3f} laps')
    print('all checks passed')


if __name__ == '__main__':
    main()
