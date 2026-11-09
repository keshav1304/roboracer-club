"""Environment process for LightSubprocVecEnv. Must not import torch or stable_baselines3.

Speaks the same pipe protocol as SB3's SubprocVecEnv worker, and records episodes the way SB3's
Monitor does (info['episode'] = {'r', 'l', 't'} plus the info keywords), which SB3 reads for
ep_rew_mean and ep_len_mean.
"""
from __future__ import annotations

import time

from autodrive_env import AutoDriveEnv


def worker(remote, parent_remote, env_kwargs: dict, info_keywords: tuple[str, ...]) -> None:
    parent_remote.close()
    env = AutoDriveEnv(**env_kwargs)
    t_start = time.time()
    ep_return, ep_len = 0.0, 0
    reset_info: dict = {}
    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break
        if cmd == 'step':
            obs, reward, terminated, truncated, info = env.step(data)
            ep_return += reward
            ep_len += 1
            done = terminated or truncated
            info['TimeLimit.truncated'] = truncated and not terminated
            if done:
                info['episode'] = {'r': round(ep_return, 6), 'l': ep_len, 't': round(time.time() - t_start, 6),
                                   **{k: info[k] for k in info_keywords}}
                info['terminal_observation'] = obs
                obs, reset_info = env.reset()
                ep_return, ep_len = 0.0, 0
            remote.send((obs, reward, done, info, reset_info))
        elif cmd == 'reset':
            options = {'options': data[1]} if data[1] else {}
            obs, reset_info = env.reset(seed=data[0], **options)
            ep_return, ep_len = 0.0, 0
            remote.send((obs, reset_info))
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'env_method':
            remote.send(getattr(env, data[0])(*data[1], **data[2]))
        elif cmd == 'get_attr':
            remote.send(getattr(env, data))
        elif cmd == 'set_attr':
            remote.send(setattr(env, data[0], data[1]))
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        else:
            raise NotImplementedError(f'{cmd!r} is not supported by env_worker')
