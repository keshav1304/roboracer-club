"""PPO on race_env with stable-baselines3. The setup that produced the P5 policy, as a starting point.

Usage (from pods/rl):
  python train/train.py --preset p5 --steps 8000000 --seed 0 --name p5_s0
  python train/train.py --preset p5 --steps 4000000 --resume train/runs/p5_s0/ckpt_4499928_steps.zip --name p5_s0_more
  tensorboard --logdir train/runs/tb
Hyperparameters are the pod doc's (n_steps 2048, batch 256, lr 3e-4, net [256, 256]), 8 workers.
Saves train/runs/NAME/cfg.json (the env settings, read back by evaluate.py and export_policy.py),
ckpt_*.zip every 100k steps and final.zip. train/runs/ is git-ignored.
On an M-series Mac one run does about 1300 steps/s, so 8M steps is under 2 hours.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'env'))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor  # noqa: E402

from race_env import make_env  # noqa: E402

PRESETS = {
    # P5: the policy only corrects pure pursuit's throttle (steering stays pure pursuit), base speeds
    # x1.15, reward v3 = 10 per metre - 1 per step + 50 per lap - 100 on a crash - 0.1 x steering change
    'p5': {'residual': 0.4, 'steer_residual': 0.0, 'ref_speed_scale': 1.15,
           'reward_weights': {'time': 1.0, 'collision': 100.0, 'center': 0.0}},
    'none': {},
}


def factory(track, seed, cfg):
    return lambda: make_env(track, seed=seed, **cfg)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--track', default='icra')
    ap.add_argument('--steps', type=int, default=1_000_000)
    ap.add_argument('--name', default=time.strftime('run_%m%d_%H%M'))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--preset', choices=sorted(PRESETS), default='none')
    ap.add_argument('--cfg', default='{}', help='JSON of make_env overrides on top of the preset, '
                    'e.g. {"reward_weights": {"tal": 1.0}}')
    ap.add_argument('--ckpt', type=int, default=100_000, help='save a checkpoint every this many steps')
    ap.add_argument('--resume', default=None, help='continue this saved model for --steps more steps')
    a = ap.parse_args()
    cfg = dict(PRESETS[a.preset], **json.loads(a.cfg))
    runs = os.path.join(HERE, 'runs')
    out = os.path.join(runs, a.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'cfg.json'), 'w') as f:            # evaluators rebuild the same env from this
        json.dump({'track': a.track, 'cfg': cfg}, f)
    env = VecMonitor(SubprocVecEnv([factory(a.track, 1000 * a.seed + i, cfg) for i in range(a.workers)]))
    tb = os.path.join(runs, 'tb')
    if a.resume:                                                     # same weights and optimiser, step count continues
        model = PPO.load(a.resume, env=env, device='cpu', tensorboard_log=tb, verbose=1)
    else:
        model = PPO('MlpPolicy', env, n_steps=2048, batch_size=256, learning_rate=3e-4,
                    policy_kwargs={'net_arch': [256, 256]}, tensorboard_log=tb, verbose=1, device='cpu', seed=a.seed)
    ckpt = CheckpointCallback(save_freq=max(1, a.ckpt // a.workers), save_path=out, name_prefix='ckpt')
    t = time.time()
    model.learn(total_timesteps=a.steps, callback=ckpt, tb_log_name=a.name, reset_num_timesteps=not a.resume)
    model.save(os.path.join(out, 'final'))
    print(f'done: {a.steps} steps in {(time.time() - t) / 60:.1f} min, saved {out}/final.zip')
    env.close()
