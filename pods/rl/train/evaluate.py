"""Check a policy before spending AutoDRIVE time on it.

Usage (from pods/rl):
  python train/evaluate.py train/runs/p5_s0/ckpt_4499928_steps.zip      # a checkpoint, cfg.json next to it
  python train/evaluate.py deploy/rl_policy_P5.npz                       # or an exported policy
Drives it the way it will run in AutoDRIVE (from the spawn, grip shield on) on three settings
1. nominal physics, 6 laps. The gym landed 0.04 to 0.12 s faster than AutoDRIVE for every policy we
   deployed, so add about 0.1 s.
2. tight grip (slide limit at the bottom of the training band, slides sooner). Stricter than AutoDRIVE
   (P4 seed2 crashes here and ran 10/10 clean there), so sliding here is a warning, a crash is not a verdict.
   P5 sits right on its edge: the SB3 checkpoint stays clean and the exported .npz, whose actions differ
   by 1e-7, slides once and crashes.
3. 30 randomised episodes x 3 laps, a crash rate for comparing policies (P5 8/30, pure pursuit x1.15 4/30).
Reference in AutoDRIVE: pure pursuit x1.15 12.67 s, P5 + grip shield 11.52 s.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'env'))
import race_env  # noqa: E402

NOMINAL = {k: getattr(race_env, k) for k in ('A_HI', 'A_LO', 'SLIDE_TRIG')}
TIGHT = {'A_HI': 10.1, 'A_LO': 7.0, 'SLIDE_TRIG': 0.06}


def load(path):
    """(act(obs) -> action, make_env overrides) for a .zip checkpoint or an exported .npz."""
    if path.endswith('.npz'):
        P = dict(np.load(path))

        def act(obs):
            h = np.tanh(P['w1'] @ obs + P['b1'])
            h = np.tanh(P['w2'] @ h + P['b2'])
            return np.clip(P['w3'] @ h + P['b3'], -1.0, 1.0).astype(np.float32)
        return act, dict(residual=float(P['residual']), steer_residual=float(P['steer_residual']),
                         ref_speed_scale=float(P.get('ref_speed_scale', 1.0)))
    from stable_baselines3 import PPO
    model = PPO.load(path, device='cpu')
    cfg = json.load(open(os.path.join(os.path.dirname(path), 'cfg.json')))['cfg']
    return (lambda obs: model.predict(obs, deterministic=True)[0]), {k: v for k, v in cfg.items()
                                                                       if k != 'reward_weights'}


def drive(env, act, laps):
    obs, _ = env.reset()
    slid = shield = 0
    while True:
        obs, _, term, trunc, info = env.step(act(obs))
        slid += info['sliding'] > 0
        shield += bool(info['shield'])
        if term or trunc or info['laps'] >= laps:
            return info['lap_times'], info['collision'], slid, shield


def main():
    act, cfg = load(sys.argv[1])
    print(f'{sys.argv[1]}  {cfg}')
    for name, grip in (('nominal', NOMINAL), ('tight grip', TIGHT)):
        for k, v in grip.items():
            setattr(race_env, k, v)
        env = race_env.make_env('icra', seed=500, randomize=False, grip_jitter=0.0, throttle_jitter=0.0,
                                random_start=False, max_laps=6, grip_shield=True, **cfg)
        lt, hit, slid, sh = drive(env, act, 6)
        fly = lt[1:]
        print(f'  {name:11s} ' + (f'flying laps mean {np.mean(fly):.2f} s (best {min(fly):.2f})' if fly else 'no flying lap')
              + (', CRASH' if hit else ', clean') + f', sliding {slid} steps, shield {sh} steps', flush=True)
    for k, v in NOMINAL.items():
        setattr(race_env, k, v)
    env = race_env.make_env('icra', seed=901, max_laps=3, grip_shield=True, **cfg)
    crashes, laps = 0, []
    for _ in range(30):
        lt, hit, _, _ = drive(env, act, 3)
        crashes += bool(hit)
        laps += lt[1:]
    print(f'  randomised  {crashes}/30 crashed, flying laps mean {np.mean(laps):.2f} s')


if __name__ == '__main__':
    main()
