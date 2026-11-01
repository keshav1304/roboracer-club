"""Does env_v2 reproduce what AutoDRIVE did? Run from pods/rl/env_v2:  python validate_v2.py

The same drivers that ran in AutoDRIVE (ICRA 2026 compete build, 2026-10-25/26), driven the same way:
from the spawn, nominal physics, on both vehicle models.
1. pure pursuit on the club line at 1.10 to 1.25 x its planned speeds (the deploy node's --base-only).
   AutoDRIVE: clean at x1.10 (13.21 s) and x1.15 (12.67 s), slid into the wall at x1.20 and x1.25.
2. P5 with the grip shield, exactly as deployed (../deploy/rl_policy_P5.npz). AutoDRIVE: 10/10, 11.52 s.
The env passes when every row lands on the same side of the line as AutoDRIVE (clean vs slid).
Lap times may differ by ~0.1 s (the old env was 0.04 to 0.12 s fast).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from race_env_v2 import make_env      # noqa: E402
import vehicle as V                   # noqa: E402

P5 = dict(np.load(os.path.join(HERE, '..', 'deploy', 'rl_policy_P5.npz')))
REF_XY, REF_S, REF_LEN = P5['ref_xy'], P5['ref_s'], float(P5['ref_len'])
REF_V_BASE = P5['ref_v'] / float(P5['ref_speed_scale'])
CASES = [('pure pursuit x1.10', 1.10, False, 'clean, 13.21 s'),
         ('pure pursuit x1.15', 1.15, False, 'clean 10/10, 12.67 s'),
         ('pure pursuit x1.20', 1.20, False, 'slid into the wall, lap 1'),
         ('pure pursuit x1.25', 1.25, False, 'slid into the wall, lap 1'),
         ('P5 + grip shield', 1.15, True, 'clean 10/10, 11.52 s')]


class ClubPursuit:
    """The deploy node's base_action: pure pursuit on the club line, planned speed."""

    def __init__(self, scale):
        self.v = REF_V_BASE * scale
        self.i = None

    def __call__(self, x, y, yaw, v):
        n = len(REF_XY)
        if self.i is None:
            tang = np.roll(REF_XY, -1, 0) - REF_XY
            ok = np.where(tang[:, 0] * math.cos(yaw) + tang[:, 1] * math.sin(yaw) > 0)[0]
            idx = ok if len(ok) else np.arange(n)
        else:
            idx = np.arange(self.i - 40, self.i + 41) % n
        i = int(idx[np.argmin((REF_XY[idx, 0] - x) ** 2 + (REF_XY[idx, 1] - y) ** 2)])
        self.i = i
        goal = (REF_S[i] + P5['ref_la'][0] + P5['ref_la'][1] * max(0.0, v)) % REF_LEN
        tx, ty = REF_XY[int(np.searchsorted(REF_S, goal) % n)]
        dx, dy = tx - x, ty - y
        xc, yc = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
        steer = math.atan(2.0 * yc / max(1e-6, xc * xc + yc * yc) * float(P5['wheelbase'])) / float(P5['max_steer'])
        thr = min(float(P5['max_throttle']), self.v[i] / float(P5['speed_per_throttle']))
        return float(np.clip(steer, -1, 1)), thr


def p5_actor(obs):
    h = np.tanh(P5['w1'] @ obs + P5['b1'])
    h = np.tanh(P5['w2'] @ h + P5['b2'])
    return np.clip(P5['w3'] @ h + P5['b3'], -1.0, 1.0)


def drive(model, scale, use_p5, laps=4, seed=500, randomize=False, timing=True):
    sys.path.insert(0, os.path.join(HERE, '..', 'env'))
    from race_env import build_observation as build_v1   # the v1 contract P5 was trained with
    env = make_env(tracks=['icra'], randomize=randomize, random_start=False, seed=seed, model=model,
                   grip_shield=use_p5, empirical_share=0.0, timing=timing)
    env.reset()
    pp = ClubPursuit(scale)
    slid = 0
    while True:
        x, y, yaw, v = env.snapshot
        steer, thr = pp(x, y, yaw, v)
        if use_p5:
            base = np.array([steer, thr / float(P5['max_throttle']) * 2 - 1])
            lx = 0.275                                          # the gym's base_link -> laser, what P5 trained with
            scan = env.track.scan(x + lx * math.cos(yaw), y + lx * math.sin(yaw), yaw)
            act = p5_actor(build_v1(scan, v, env.last_steer, base))
            sent = np.clip(base + np.array([0.0, float(P5['residual'])]) * act, -1, 1)
            steer, thr = float(sent[0]), (sent[1] + 1) * 0.5 * float(P5['max_throttle'])
        _, _, term, trunc, info = env.step_commands(steer, thr)
        slid += info['slide']
        if term or trunc or info['laps'] >= laps:
            return info, slid


def main():
    print('env_v2 (nominal physics, from the spawn) vs AutoDRIVE. Flying laps = laps after lap 1.')
    for model, name in ((V.MODEL_TWIN, 'twin'), (V.MODEL_EMPIRICAL, 'empirical')):
        print(f'---- {name}')
        for label, scale, use_p5, ad in CASES:
            info, slid = drive(model, scale, use_p5)
            fly = info['lap_times'][1:]
            res = (f'{len(fly)} flying laps {np.mean(fly):.2f} s' if fly else 'no flying lap')
            why = ' wall' if info['wall'] else ' slide' if info['slide'] else ' spin' if info['spin'] else ''
            print(f'  {label:20s} {"CRASH" + why if info["crash"] else "clean"}, {res}, slide steps {slid}'
                  f'   | AutoDRIVE: {ad}')
    print('---- randomised twin, 20 episodes x 3 laps, crashes (compare drivers, not a prediction)')
    print('     physics only | physics + timing (the training distribution, with late steps)')
    for label, scale, use_p5, _ in CASES[1:3] + CASES[4:]:
        a = sum(drive(V.MODEL_TWIN, scale, use_p5, 3, 1000 + k, True, False)[0]['crash'] for k in range(20))
        b = sum(drive(V.MODEL_TWIN, scale, use_p5, 3, 1000 + k, True, True)[0]['crash'] for k in range(20))
        print(f'  {label:20s} {a:2d}/20 | {b:2d}/20')


if __name__ == '__main__':
    main()
