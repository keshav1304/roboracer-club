"""Export a trained PPO policy (actor only) plus everything the deploy node needs into one .npz, so it
runs next to the AutoDRIVE bridge with numpy alone (no torch there).

Usage (from pods/rl):
  python train/export_policy.py train/runs/p5_s0/ckpt_4499928_steps.zip deploy/rl_policy_mine.npz
Reads cfg.json next to the checkpoint (written by train.py) to rebuild the same env. Saves the actor
weights (112 -> 256 -> 256 -> 2, tanh), the residual scales, the reference line with its speed plan and
pure pursuit settings (for residual mode), and the observation/action constants.
Checks that the numpy forward pass matches stable-baselines3 on 1000 random observations.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'env'))
from stable_baselines3 import PPO  # noqa: E402

import race_env  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
cfg_file = os.path.join(os.path.dirname(src), 'cfg.json')
cfg = json.load(open(cfg_file))['cfg'] if os.path.exists(cfg_file) else {}
cfg = {k: v for k, v in cfg.items() if k != 'reward_weights'}
model = PPO.load(src, device='cpu')
W = {n: t.detach().numpy().astype(np.float64) for n, t in model.policy.state_dict().items()}
env = race_env.make_env('icra', seed=0, randomize=False, **cfg)    # ref_v includes ref_speed_scale
arrays = dict(
    w1=W['mlp_extractor.policy_net.0.weight'], b1=W['mlp_extractor.policy_net.0.bias'],
    w2=W['mlp_extractor.policy_net.2.weight'], b2=W['mlp_extractor.policy_net.2.bias'],
    w3=W['action_net.weight'], b3=W['action_net.bias'],
    residual=env.residual, steer_residual=env.steer_residual, ref_speed_scale=env.ref_speed_scale,
    ref_xy=env.ref_xy, ref_v=env.ref_v, ref_s=env.ref_s, ref_len=env.ref_len,
    ref_la=np.array(env.ref_la), ref_lead=env.ref_lead, steer_delay=env.steer_delay,
    wheelbase=env.wheelbase, lidar_bins=race_env.LIDAR_BINS, max_range=race_env.MAX_RANGE,
    speed_norm=race_env.SPEED_NORM, max_steer=race_env.MAX_STEER, max_throttle=race_env.MAX_THROTTLE,
    speed_per_throttle=race_env.SPEED_PER_THROTTLE, control_hz=race_env.CONTROL_HZ,
    source=os.path.join(os.path.basename(os.path.dirname(os.path.abspath(src))), os.path.basename(src)))
np.savez(dst, **arrays)

# the deploy node's forward pass, checked against SB3
rng = np.random.default_rng(0)
obs = rng.uniform(env.observation_space.low, env.observation_space.high, size=(1000,) + env.observation_space.shape)
h = np.tanh(obs @ arrays['w1'].T + arrays['b1'])
h = np.tanh(h @ arrays['w2'].T + arrays['b2'])
ours = np.clip(h @ arrays['w3'].T + arrays['b3'], -1, 1)
theirs = np.array([model.predict(o.astype(np.float32), deterministic=True)[0] for o in obs])
print(f'exported {src} -> {dst}: obs {obs.shape[1]}, residual {env.residual} (steering {env.steer_residual}), '
      f'max |numpy - sb3| = {np.abs(ours - np.clip(theirs, -1, 1)).max():.2e}')
