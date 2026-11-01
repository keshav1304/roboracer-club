"""Gates a v2 policy must pass before it spends AutoDRIVE time. Run from pods/rl:

  python train_v2/evaluate.py train_v2/runs/NAME/best_icra.zip       # a checkpoint
  python train_v2/evaluate.py deploy/rl_policy_v2.npz                 # or the exported file

G1  ICRA from the spawn, nominal twin, 6 laps: clean and no sliding.            (ICRA is never trained on)
G2  the same on the empirical model (the KS + slide env P5 was validated on).
G3  ICRA, randomised physics, 20 episodes x 3 laps, timing off: crashes.         compare: P5 6/20, PP x1.15 0/20
G4  ICRA, randomised physics + timing (late steps up to 2 %), 20 x 3: crashes.  compare: P5 13/20, PP x1.15 7/20
G5  delay sweep: largest steering = throttle delay (ms) with 3 clean laps.       AutoDRIVE is ~30 / 40 ms
G6  10 generated tracks the policy never saw (gen 900+), nominal twin: clean share and lap vs the teacher.
A policy is worth an AutoDRIVE run when G1, G2 pass, G3 <= P5's, G5 >= 90 ms and G6 >= 8/10.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'env_v2'))

from race_env_v2 import make_env              # noqa: E402
import vehicle as V                           # noqa: E402


def load(path):
    if path.endswith('.npz'):
        P = dict(np.load(path))

        def act(o):
            h = np.tanh(P['w1'] @ o + P['b1'])
            h = np.tanh(P['w2'] @ h + P['b2'])
            return np.clip(P['w3'] @ h + P['b3'], -1, 1).astype(np.float32)
        return act
    from stable_baselines3 import PPO
    # the schedules are pickled lambdas, which segfault on another Python version; only the weights are needed
    m = PPO.load(path, device='cpu', custom_objects={'learning_rate': 0.0, 'lr_schedule': lambda _: 0.0, 'clip_range': lambda _: 0.0})
    return lambda o: m.predict(o, deterministic=True)[0]


def run(act, track='icra', laps=3, model=V.MODEL_TWIN, randomize=False, timing=True, seed=7, delay=None):
    env = make_env(tracks=[track], randomize=randomize, random_start=False, seed=seed, model=model, timing=timing)
    obs, _ = env.reset()
    if delay is not None:
        env.car.p[V.IDX['delay_s']] = env.car.p[V.IDX['delay_t']] = delay
    slides = 0
    while True:
        obs, _, term, trunc, info = env.step(act(obs))
        slides += info['slide']
        if term or trunc or info['laps'] >= laps:
            fly = info['lap_times'][1:]
            return dict(crash=bool(info['crash']), laps=info['laps'], slides=slides,
                        flying=float(np.mean(fly)) if fly else None, teacher=info['teacher_lap'],
                        why='wall' if info['wall'] else 'slide' if info['slide'] else 'spin' if info['spin'] else
                        'stall' if info['stall'] else '')


def main(path):
    act = load(path)
    ok = {}
    for g, m in (('G1', V.MODEL_TWIN), ('G2', V.MODEL_EMPIRICAL)):
        r = run(act, laps=6, model=m)
        ok[g] = not r['crash'] and r['slides'] == 0
        print(f'{g} ICRA {"twin" if m == V.MODEL_TWIN else "empirical"}: '
              f'{"CRASH " + r["why"] if r["crash"] else "clean"}, {r["laps"]} laps, flying {r["flying"]}, '
              f'slide steps {r["slides"]}')
    g3 = sum(run(act, randomize=True, timing=False, seed=1000 + k)['crash'] for k in range(20))
    print(f'G3 randomised physics: {g3}/20 crashed   (P5 6/20, PP x1.15 0/20)')
    g4 = sum(run(act, randomize=True, timing=True, seed=1000 + k)['crash'] for k in range(20))
    print(f'G4 randomised physics + timing: {g4}/20 crashed   (P5 13/20, PP x1.15 7/20)')
    best = 0
    for ms in range(30, 160, 20):
        if run(act, delay=ms / 1000.0)['crash']:
            break
        best = ms
    print(f'G5 delay tolerance: clean up to {best} ms of steering and throttle delay')
    g6, rel = 0, []
    for i in range(900, 910):
        r = run(act, track=f'gen{i:04d}')
        g6 += not r['crash']
        if r['flying']:
            rel.append(r['flying'] / r['teacher'])
    print(f'G6 unseen generated tracks: {g6}/10 clean, flying lap {np.mean(rel) if rel else float("nan"):.3f} x the '
          f'teacher plan')
    verdict = ok['G1'] and ok['G2'] and g3 <= 6 and best >= 90 and g6 >= 8
    print('VERDICT:', 'worth an AutoDRIVE run (3 laps first, window in the background)' if verdict else 'not yet')


if __name__ == '__main__':
    main(sys.argv[1])
