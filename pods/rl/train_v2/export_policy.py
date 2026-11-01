"""SB3 checkpoint -> .npz the v2 deploy node reads (numpy only, no torch), with the env_v2 contract inside.

Usage (from pods/rl):  python train_v2/export_policy.py train_v2/runs/NAME/best_icra.zip deploy/rl_policy_v2.npz
Checks the numpy actor against SB3 on random observations before writing.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'env_v2'))

import contract as C                         # noqa: E402


def export(ckpt, out):
    from stable_baselines3 import PPO
    # the schedules are pickled lambdas, which segfault on another Python version; only the weights are needed
    model = PPO.load(ckpt, device='cpu', custom_objects={'learning_rate': 0.0, 'lr_schedule': lambda _: 0.0, 'clip_range': lambda _: 0.0})
    pol = model.policy
    lin = [m for m in pol.mlp_extractor.policy_net if hasattr(m, 'weight')]
    assert len(lin) == 2, 'expected net_arch [256, 256]'
    W = {'w1': lin[0].weight, 'b1': lin[0].bias, 'w2': lin[1].weight, 'b2': lin[1].bias,
         'w3': pol.action_net.weight, 'b3': pol.action_net.bias}
    W = {k: v.detach().numpy().astype(np.float64) for k, v in W.items()}

    def actor(o):
        h = np.tanh(W['w1'] @ o + W['b1'])
        h = np.tanh(W['w2'] @ h + W['b2'])
        return np.clip(W['w3'] @ h + W['b3'], -1, 1)
    rng = np.random.default_rng(0)
    for _ in range(200):
        o = rng.uniform(model.observation_space.low, model.observation_space.high).astype(np.float32)
        a_sb3, _ = model.predict(o, deterministic=True)
        assert np.max(np.abs(actor(o) - a_sb3)) < 1e-4, 'numpy actor does not match SB3'
    cfg_path = os.path.join(os.path.dirname(ckpt), 'cfg.json')
    cfg = open(cfg_path).read() if os.path.isfile(cfg_path) else '{}'
    np.savez(out, **W, **C.metadata(), residual=0.0, steer_residual=0.0,
             source=f'env_v2 end to end PPO, {os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}',
             train_cfg=json.dumps(json.loads(cfg)))
    print(f'wrote {out} (obs {C.OBS_DIM}, contract v{C.OBS_VERSION}), numpy actor matches SB3')


if __name__ == '__main__':
    export(sys.argv[1], sys.argv[2])
