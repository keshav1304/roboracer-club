"""Checks the environment against one running simulator, with no trained policy.

A scripted controller follows the centre line for one lap. Passing shows that the simulator's track
matches the centre line file, and that progress, laps, steering sign, collisions and reset all work.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from autodrive_env import MAX_STEER, MAX_THROTTLE, AutoDriveEnv, pursuit_steer


def follow_action(env: AutoDriveEnv, throttle: float, lookahead: float = 1.0) -> np.ndarray:
    f = env.bridge.latest()
    # Track the centre-line position here: env.prev_s is along the race line when one is set.
    guess = getattr(env, '_follow_s', None) if env.line is not None else env.prev_s
    s, _ = env.track.cartesian_to_frenet(f.x, f.y, s_guess=guess)
    env._follow_s = s
    return np.array([pursuit_steer(env.track, f, s, lookahead), 2 * throttle / MAX_THROTTLE - 1], np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=4567)
    ap.add_argument('--track', default='icra')
    ap.add_argument('--throttle', type=float, default=0.08)
    ap.add_argument('--max-seconds', type=float, default=90.0)
    args = ap.parse_args()

    env = AutoDriveEnv(port=args.port, track_name=args.track, max_laps=1, max_steps=100000)
    print(f'waiting for a simulator on port {args.port} ...', flush=True)
    env.bridge.wait_connected(env.CONNECT_TIMEOUT_S)
    obs, info = env.reset()
    f = env.bridge.latest()
    print(f'reset ok: pos=({f.x:.3f}, {f.y:.3f}) yaw={f.yaw:.3f} s={info["s"]:.2f}/{env.track.length:.2f} '
          f'obs shape={obs.shape} in bounds={env.observation_space.contains(obs)}', flush=True)

    # Steering sign: full left must read positive and turn the car left (yaw increases).
    yaw0 = env.bridge.latest().yaw
    for _ in range(10):
        steer_obs, _, _, _, info = env.step(np.array([1.0, -0.5], np.float32))
    steer_fb, yaw1 = info['steering_feedback'], env.bridge.latest().yaw
    steer_obs = float(steer_obs[109])
    dyaw = (yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi
    print(f'steering +1: feedback={steer_fb:.3f} rad, observation={steer_obs:.3f}, yaw change={dyaw:+.3f} rad', flush=True)
    env.reset()

    # One lap at a low speed.
    t0, ret, max_ey, rows, info, term, trunc = time.time(), 0.0, 0.0, 0, {}, False, False
    while not (term or trunc) and time.time() - t0 < args.max_seconds:
        obs, r, term, trunc, info = env.step(follow_action(env, args.throttle))
        ret += r
        max_ey = max(max_ey, abs(info.get('ey', 0.0)))
        rows += 1
        if rows % 100 == 0:
            print(f"  step {rows:5d} s={info['s']:6.2f} ey={info['ey']:+.2f} v={info['speed']:.2f} "
                  f"progress={info['progress_m']:6.2f} return={ret:8.2f}", flush=True)
    wall = time.time() - t0
    hz = rows / max(wall, 1e-6)
    print(f"lap run: laps={info.get('laps')} lap_times={info.get('lap_times')} sim_lap_count={info.get('sim_lap_count')} "
          f"sim_last_lap={info.get('sim_last_lap_time')} max|ey|={max_ey:.2f} collision={info.get('collision')} "
          f"return={ret:.1f} control_hz={hz:.1f}", flush=True)
    lap_ok = info.get('laps') == 1 and not info.get('collision') and max_ey < 1.0

    # Collision: drive straight off the track.
    env.reset()
    term, trunc, steps, info = False, False, 0, {}
    while not (term or trunc) and steps < 400:
        _, r, term, trunc, info = env.step(np.array([0.0, 0.0], np.float32))
        steps += 1
    print(f"collision run: terminated={term} after {steps} steps, r_collision={info.get('r_collision')}", flush=True)
    collision_ok = term and info.get('r_collision') == -env.w['collision']

    env.reset()
    f = env.bridge.latest()
    reset_ok = f.collisions == 0 and abs(f.x - 0.8) < 0.1 and abs(f.y - 3.16) < 0.1
    print(f'reset after collision: collisions={f.collisions} pos=({f.x:.3f}, {f.y:.3f})', flush=True)

    steer_ok = abs(steer_fb - MAX_STEER) < 0.02 and steer_obs > 0.95 and dyaw > 0
    results = {'steering sign and scale': steer_ok, 'lap on centre line': lap_ok,
               'collision ends episode': collision_ok, 'reset': reset_ok, 'keeps 20 Hz': 19.0 < hz < 21.0}
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}", flush=True)
    env.close()
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
