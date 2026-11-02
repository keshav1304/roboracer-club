"""PPO on env_v2, started from the teacher. The setup for the end to end (fully RL) policy.

Usage (from pods/rl):
  python train_v2/train.py --name e2e_s0 --seed 0 --steps 40000000              # the default recipe
  python train_v2/train.py --name smoke --steps 200000 --workers 4 --bc-steps 20000 --gen 8   # quick check
  python train_v2/train.py --name e2e_s0_more --resume train_v2/runs/e2e_s0/final.zip --steps 20000000
  tensorboard --logdir train_v2/runs/tb

The recipe, in order (README.md in env_v2 explains why each part is there):
1. Build the generated tracks once (--gen of them, default 200) so the workers do not race to build them.
2. Behaviour cloning. The teacher (pure pursuit on each track's minimum-curvature line) drives the training
   envs with some action noise (so we also see recoveries), the actor learns its actions and the critic
   learns the discounted returns. End to end PPO from scratch learned on 1 seed in 3 in our sweep.
3. PPO, gamma 0.995 (10 s of look-ahead, so a wide corner entry that pays off on the next straight counts),
   with the shaping toward the teacher's line turned down from --shaping to 0 over the first 30 %.
4. Every --eval-every steps: ICRA (never trained on) on both physics models from the spawn. Best-by-ICRA and
   periodic checkpoints are saved; pick the final one with train_v2/evaluate.py, not by reward.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'env_v2'))

import torch                                                            # noqa: E402
from stable_baselines3 import PPO                                       # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor  # noqa: E402

import tracks as T                                                      # noqa: E402
from race_env_v2 import make_env                                        # noqa: E402
import vehicle as V                                                     # noqa: E402

PRESETS = {
    # the default: everything on
    'e2e_v2': {},
    # ablations, to see what each part buys
    'no_timing': {'timing': False},                     # no delays / late steps
    'twin_only': {'empirical_share': 0.0},
    'no_envelope': {'use_envelope': False},
    'slide_ok': {'slide_is_crash': False},
}


def factory(seed, cfg):
    return lambda: make_env(seed=seed, **cfg)


def evaluate_icra(model, laps=3):
    """Deterministic policy on ICRA from the spawn, nominal physics, both models."""
    out = {}
    for m, name in ((V.MODEL_TWIN, 'twin'), (V.MODEL_EMPIRICAL, 'empirical')):
        env = make_env(tracks=['icra'], randomize=False, random_start=False, seed=7, model=m)
        obs, _ = env.reset()
        slides = 0
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            slides += info['slide']
            if term or trunc or info['laps'] >= laps:
                break
        fly = info['lap_times'][1:]
        out[name] = {'laps': info['laps'], 'crash': bool(info['crash']), 'flying': float(np.mean(fly)) if fly else None,
                     'progress_m': float(info['progress_m']), 'slide_steps': int(slides)}
    return out


class Schedule(BaseCallback):
    """Turns the shaping down to 0 and evaluates on ICRA."""

    def __init__(self, total, shaping0, frac, eval_every, out_dir):
        super().__init__()
        self.total, self.k0, self.frac, self.every, self.out = total, shaping0, frac, eval_every, out_dir
        self.next_eval = eval_every
        self.best = (-1, -1e9)
        self.t0 = None

    def _on_rollout_start(self):
        if self.t0 is None:
            self.t0 = self.num_timesteps
        done = (self.num_timesteps - self.t0) / max(1, self.total)
        k = self.k0 * max(0.0, 1.0 - done / self.frac) if self.frac > 0 else 0.0
        self.training_env.env_method('set_shaping', k)
        self.logger.record('train/shaping_k', k)

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.every
            res = evaluate_icra(self.model)
            for name, r in res.items():
                self.logger.record(f'icra/{name}_laps', r['laps'])
                self.logger.record(f'icra/{name}_progress_m', r['progress_m'])
                if r['flying']:
                    self.logger.record(f'icra/{name}_flying_lap', r['flying'])
            with open(os.path.join(self.out, 'icra_eval.jsonl'), 'a') as f:
                f.write(json.dumps({'steps': int(self.num_timesteps), **res}) + '\n')
            score = (min(r['laps'] for r in res.values()),
                     -max((r['flying'] or 99.0) for r in res.values()))
            if score > self.best:
                self.best = score
                self.model.save(os.path.join(self.out, 'best_icra'))
        return True


def behaviour_cloning(model, vec, steps, gamma, noise=0.15, epochs=30, batch=512, lr=1e-3, log=print):
    """Teacher drives (with noise), the actor learns its actions, the critic learns the returns."""
    n = vec.num_envs
    obs = vec.reset()
    O, A, R, D = [], [], [], []
    rng = np.random.default_rng(0)
    for _ in range(steps // n):
        expert = np.array(vec.env_method('expert_action'), dtype=np.float32)
        act = np.clip(expert + rng.normal(0, noise, expert.shape), -1, 1).astype(np.float32)
        O.append(obs.copy())
        A.append(expert)
        obs, rew, done, _ = vec.step(act)
        R.append(rew.copy())
        D.append(done.copy())
    O, A, R, D = np.array(O), np.array(A), np.array(R), np.array(D)          # (T, n, ...)
    G = np.zeros_like(R)
    run = np.zeros(n)
    for t in range(len(R) - 1, -1, -1):                                       # discounted returns per env
        run = R[t] + gamma * run * (1.0 - D[t])
        G[t] = run
    keep = slice(0, max(1, int(len(R) * 0.8)))                                # last 20 %: returns are cut short
    dev = model.device
    obs_t = torch.as_tensor(O[keep].reshape(-1, O.shape[-1]), device=dev)
    act_t = torch.as_tensor(A[keep].reshape(-1, 2), device=dev)
    ret_t = torch.as_tensor(G[keep].reshape(-1), dtype=torch.float32, device=dev)
    pol = model.policy
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    idx = np.arange(len(obs_t))
    for ep in range(epochs):
        rng.shuffle(idx)
        la = lv = 0.0
        for i in range(0, len(idx), batch):
            j = torch.as_tensor(idx[i:i + batch], device=dev)
            dist = pol.get_distribution(obs_t[j])
            mean = dist.distribution.mean
            loss_a = ((mean - act_t[j]) ** 2).mean()
            v = pol.predict_values(obs_t[j]).flatten()
            loss_v = ((v - ret_t[j]) ** 2).mean() / max(1.0, float(ret_t.var()))
            loss = loss_a + 0.5 * loss_v
            opt.zero_grad()
            loss.backward()
            opt.step()
            la, lv = loss_a.item(), loss_v.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            log(f'  BC epoch {ep + 1}/{epochs}: action mse {la:.4f}, value loss {lv:.4f}')
    with torch.no_grad():
        pol.log_std.fill_(-1.0)                                               # start exploring close to the teacher
    crash = float(D.mean() * 20)
    log(f'  teacher data: {len(obs_t)} samples, {crash:.3f} episode ends per second of driving')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default=time.strftime('v2_%m%d_%H%M'))
    ap.add_argument('--steps', type=int, default=40_000_000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--preset', choices=sorted(PRESETS), default='e2e_v2')
    ap.add_argument('--cfg', default='{}', help='JSON of make_env overrides on top of the preset')
    ap.add_argument('--gen', type=int, default=200, help='generated tracks in the training bank')
    ap.add_argument('--holdout', default='icra', help="real tracks kept out of training, comma separated, or 'none'")
    ap.add_argument('--bc-steps', type=int, default=400_000, help='teacher samples for behaviour cloning (0 = off)')
    ap.add_argument('--bc-epochs', type=int, default=30)
    ap.add_argument('--shaping', type=float, default=2.0, help='initial weight of the pull toward the teacher line')
    ap.add_argument('--shaping-frac', type=float, default=0.3, help='share of training over which it goes to 0')
    ap.add_argument('--gamma', type=float, default=0.995)
    ap.add_argument('--n-steps', type=int, default=512, help='rollout length per worker')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval-every', type=int, default=1_000_000)
    ap.add_argument('--ckpt', type=int, default=2_000_000)
    ap.add_argument('--resume', default=None)
    ap.add_argument('--device', default='cpu', help="torch device for the networks: cpu, mps or cuda")
    a = ap.parse_args()

    out = os.path.join(HERE, 'runs', a.name)
    os.makedirs(out, exist_ok=True)
    holdout = [] if a.holdout == 'none' else [h for h in a.holdout.split(',') if h]
    cfg = dict(PRESETS[a.preset], n_generated=a.gen, gamma=a.gamma, holdout=holdout, **json.loads(a.cfg))
    with open(os.path.join(out, 'cfg.json'), 'w') as f:
        json.dump({'preset': a.preset, 'cfg': cfg, 'args': vars(a)}, f, indent=1)
    print(f'building {a.gen} generated tracks in {T.GEN_DIR} (once per machine)', flush=True)
    for i in range(a.gen):
        T.ensure_generated(i)

    vec = VecMonitor(SubprocVecEnv([factory(1000 * a.seed + i, cfg) for i in range(a.workers)]))
    tb = os.path.join(HERE, 'runs', 'tb')
    if a.resume:
        # the schedules are pickled lambdas, which segfault on another Python version; rebuild them here
        lr_fn = lambda f: a.lr * (0.3 + 0.7 * f)                       # noqa: E731
        model = PPO.load(a.resume, env=vec, device=a.device, tensorboard_log=tb,
                         custom_objects={'learning_rate': lr_fn, 'lr_schedule': lr_fn, 'clip_range': lambda _: 0.2})
    else:
        model = PPO('MlpPolicy', vec, n_steps=a.n_steps, batch_size=min(4096, a.n_steps * a.workers),
                    n_epochs=10, gamma=a.gamma,
                    gae_lambda=0.95, learning_rate=lambda f: a.lr * (0.3 + 0.7 * f), clip_range=0.2,
                    ent_coef=0.0, target_kl=0.03, policy_kwargs={'net_arch': [256, 256], 'log_std_init': -0.5},
                    tensorboard_log=tb, verbose=1, device=a.device, seed=a.seed)
        if a.bc_steps:
            print(f'behaviour cloning on {a.bc_steps} teacher steps', flush=True)
            behaviour_cloning(model, vec, a.bc_steps, a.gamma, epochs=a.bc_epochs,
                              log=lambda s: print(s, flush=True))
            model.save(os.path.join(out, 'bc'))
            print('after BC on ICRA:', evaluate_icra(model), flush=True)
    cbs = [CheckpointCallback(save_freq=max(1, a.ckpt // a.workers), save_path=out, name_prefix='ckpt'),
           Schedule(a.steps, a.shaping if not a.resume else 0.0, a.shaping_frac, a.eval_every, out)]
    t = time.time()
    model.learn(total_timesteps=a.steps, callback=cbs, tb_log_name=a.name, reset_num_timesteps=not a.resume)
    model.save(os.path.join(out, 'final'))
    print(f'done: {a.steps} steps in {(time.time() - t) / 60:.1f} min, saved {out}/final.zip', flush=True)
    vec.close()
