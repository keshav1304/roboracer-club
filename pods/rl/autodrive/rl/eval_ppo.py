"""Runs a trained policy on one running simulator and reports laps and lap times.

Policies are ranked as in race_env.py: consecutive clean laps, then lap time.
"""
from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from autodrive_env import AutoDriveEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--port', type=int, default=4567)
    ap.add_argument('--track', default='icra', choices=['icra', 'practice', 'none'])
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--laps', type=int, default=10, help='laps before an episode ends')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--race-line', default=None, help='race line CSV, to count laps and distance along it')
    ap.add_argument('--stochastic', action='store_true', help='sample actions instead of the mean')
    args = ap.parse_args()

    model = PPO.load(args.model, device=args.device)
    env = AutoDriveEnv(port=args.port, track_name=args.track, max_laps=args.laps,
                       max_steps=args.laps * 3000, race_line=args.race_line)
    print(f'waiting for a simulator on port {args.port} ...', flush=True)
    results = []
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            ret, done, info, steps = 0.0, False, {}, 0
            while not done:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, r, term, trunc, info = env.step(action)
                ret += r
                steps += 1
                done = term or trunc
                if steps % 100 == 0:
                    print(f"  ep {ep + 1} step {steps:5d} laps={info.get('laps')} v={info.get('speed', 0):.2f} "
                          f"progress={info.get('progress_m', 0):.1f} m", flush=True)
            times = info.get('lap_times', [])
            ended = ('crash' if info.get('collision') else 'stall' if info.get('stalled')
                     else 'wrong way' if info.get('wrong_way') else 'simulator stopped' if info.get('sim_stalled') else 'lap or step limit')
            results.append((info.get('laps', 0), min(times) if times else float('inf')))
            print(f"episode {ep + 1}: laps={info.get('laps')} lap_times={times} return={ret:.1f} "
                  f"ended by {ended}, distance={info.get('progress_m', 0):.1f} m", flush=True)
    except KeyboardInterrupt:
        print('stopped', flush=True)
    finally:
        env.close()
    if results:
        print(f'best: {max(r[0] for r in results)} clean laps, best lap {min(r[1] for r in results):.3f} s, '
              f'mean laps {np.mean([r[0] for r in results]):.1f}', flush=True)


if __name__ == '__main__':
    main()
