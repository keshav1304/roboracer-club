"""Does this env reproduce what AutoDRIVE did? Run from pods/rl/env:  python validate_vs_autodrive.py

Drives the same drivers that were run in AutoDRIVE (ICRA 2026 build, 2026-10-25/26) the same way (from
the spawn, nominal physics, grip shield as in AutoDRIVE) and prints the gym result next to AutoDRIVE's.
1. pure pursuit on the club line at 1.10 to 1.25 x its planned speeds (deploy node --base-only
   --speed-scale). AutoDRIVE was clean up to 1.15 and slid into the wall at 1.20 and 1.25.
2. the deployed policy, ../deploy/rl_policy_P5.npz, with the grip shield.
Then the randomised physics used in training, 30 episodes x 3 laps, as a crash rate.
Earlier policies (not in the repo) were checked the same way: P2, P3 and the steering P4 crashed here
as they did in AutoDRIVE, and the three throttle P4 seeds landed within 0.04 to 0.12 s of their AutoDRIVE
laps. The one miss is pure pursuit x1.20, clean here and a crash in AutoDRIVE, so treat a policy that
runs right at the edge here with suspicion. The randomised band is wider than AutoDRIVE on purpose (pure
pursuit x1.15 crashes in a few episodes here and never did there), so read its crash rate as a
comparison between drivers, not as a prediction.
"""
import os

import numpy as np

from race_env import make_env

HERE = os.path.dirname(os.path.abspath(__file__))
POLICY = os.path.join(HERE, '..', 'deploy', 'rl_policy_P5.npz')
JUDGE = dict(randomize=False, grip_jitter=0.0, throttle_jitter=0.0, random_start=False, seed=500)
CASES = [   # label, speed scale, policy file or None, grip shield, what AutoDRIVE did
    ('pure pursuit x1.10', 1.10, None, False, 'clean, 13.21 s'),
    ('pure pursuit x1.15', 1.15, None, False, 'clean 10/10, 12.67 s'),
    ('pure pursuit x1.20', 1.20, None, False, 'slid into the wall, lap 1'),
    ('pure pursuit x1.25', 1.25, None, False, 'slid into the wall, lap 1'),
    ('P5 + grip shield', None, POLICY, True, 'clean 10/10, 11.52 s'),
]


def load(path):
    if path is None:
        return None, {}
    P = dict(np.load(path))
    cfg = dict(residual=float(P['residual']), steer_residual=float(P['steer_residual']),
               ref_speed_scale=float(P['ref_speed_scale']))

    def actor(obs):
        h = np.tanh(P['w1'] @ obs + P['b1'])
        h = np.tanh(P['w2'] @ h + P['b2'])
        return np.clip(P['w3'] @ h + P['b3'], -1.0, 1.0).astype(np.float32)
    return actor, cfg


def drive(env, actor, laps):
    obs, _ = env.reset()
    slid = 0
    while True:
        act = np.zeros(2, np.float32) if actor is None else actor(obs)
        obs, _, term, trunc, info = env.step(act)
        slid += info['sliding'] > 0
        if term or trunc or info['laps'] >= laps:
            return info['lap_times'], info['collision'], slid


def main():
    print('gym (nominal physics, from the spawn) vs AutoDRIVE; flying laps, lap 1 starts from standstill')
    runs = []
    for label, scale, path, shield, autodrive in CASES:
        if path is not None and not os.path.exists(path):
            print(f'  {label:20s} skipped, {path} not found')
            continue
        actor, cfg = load(path)
        if actor is None:
            cfg = dict(residual=0.4, steer_residual=0.0, ref_speed_scale=scale)
        env = make_env('icra', max_laps=4, grip_shield=shield, **cfg, **JUDGE)
        lt, hit, slid = drive(env, actor, 4)
        gym = (f'mean {np.mean(lt[1:]):.2f} s' if len(lt) > 1 else '-') + (' then CRASH' if hit else ', clean')
        print(f'  {label:20s} gym {gym:24s} slid {slid:3d} steps | AutoDRIVE {autodrive}', flush=True)
        runs.append((label, actor, cfg, shield))
    print('randomised physics (training settings), 30 episodes x 3 laps from random starts')
    for label, actor, cfg, shield in runs:
        if '1.10' in label or '1.25' in label:
            continue
        env = make_env('icra', seed=901, max_laps=3, grip_shield=shield, **cfg)
        crashes, laps = 0, []
        for _ in range(30):
            lt, hit, _ = drive(env, actor, 3)
            crashes += bool(hit)
            laps += lt[1:]
        print(f'  {label:20s} {crashes:2d}/30 crashed, flying mean {np.mean(laps):.2f} s', flush=True)


if __name__ == '__main__':
    main()
