"""Records every control step of a policy, and of a scripted centre-line driver, to CSV files.

For finding out why and where a policy crashes. It changes nothing in the environment.
Output: logs/diagnose/<label>_ep<n>.csv, one row per 20 Hz step.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from stable_baselines3 import PPO

from autodrive_env import MAX_THROTTLE, AutoDriveEnv
from check_env import follow_action

FIELDS = ['step', 't', 's', 'progress', 'ey', 'x', 'y', 'yaw', 'speed', 'yaw_rate', 'steer_action', 'throttle_cmd',
          'steering_feedback', 'lidar_front', 'lidar_i900', 'lidar_i180', 'lidar_min', 'lidar_min_deg', 'collision']


def lidar_summary(scan: np.ndarray) -> dict:
    """1081 rays over 270 degrees: index 540 is straight ahead, 4 rays per degree."""
    clean = np.nan_to_num(np.asarray(scan, dtype=np.float64), nan=10.0, posinf=10.0, neginf=0.0)
    i = int(np.argmin(clean))
    return {'lidar_front': clean[520:561].min(), 'lidar_i900': clean[880:921].min(), 'lidar_i180': clean[160:201].min(),
            'lidar_min': clean[i], 'lidar_min_deg': (i - 540) / 4.0}


def run_episode(env: AutoDriveEnv, act, path: str, max_steps: int) -> dict:
    obs, _ = env.reset()
    rows, info, done = [], {}, False
    while not done and len(rows) < max_steps:
        action = act(obs)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        if info.get('sim_stalled'):
            break
        f = env.bridge.latest()
        rows.append({'step': len(rows) + 1, 't': info['sim_time'], 's': info['s'], 'progress': info['progress_m'],
                     'ey': info['ey'], 'x': f.x, 'y': f.y, 'yaw': f.yaw, 'speed': info['speed'],
                     'yaw_rate': info['yaw_rate'], 'steer_action': float(np.clip(action[0], -1, 1)),
                     'throttle_cmd': info['throttle_cmd'], 'steering_feedback': info['steering_feedback'],
                     **lidar_summary(env._last_lidar), 'collision': int(info['collision'])})
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return {'steps': len(rows), 'progress': info.get('progress_m', 0.0), 'laps': info.get('laps', 0),
            'collision': info.get('collision'), 'speed': info.get('speed', 0.0), 'ey': info.get('ey', 0.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--port', type=int, default=4567)
    ap.add_argument('--track', default='icra')
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--out', default='logs/diagnose')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = PPO.load(args.model, device=args.device)
    env = AutoDriveEnv(port=args.port, track_name=args.track, max_laps=1, max_steps=100000)
    print(f'waiting for a simulator on port {args.port} ...', flush=True)

    def report(label: str, ep: int, act, max_steps: int = 3000) -> None:
        r = run_episode(env, act, os.path.join(args.out, f'{label}_ep{ep}.csv'), max_steps)
        print(f"{label} ep{ep}: steps={r['steps']} progress={r['progress']:.2f} m laps={r['laps']} "
              f"crashed={r['collision']} v_end={r['speed']:.2f} ey_end={r['ey']:+.2f}", flush=True)

    try:
        for ep in range(args.episodes):
            report('policy', ep, lambda o: model.predict(o, deterministic=True)[0])
        for ep in range(args.episodes):
            report('stochastic', ep, lambda o: model.predict(o, deterministic=False)[0])
        # The same corner driven on the centre line: shows whether the track allows it at each speed.
        for throttle in (0.08, 0.16, 0.24):
            if throttle > MAX_THROTTLE:
                continue
            report(f'scripted_{throttle:.2f}', 0, lambda o, th=throttle: follow_action(env, th))
    finally:
        env.close()


if __name__ == '__main__':
    main()
