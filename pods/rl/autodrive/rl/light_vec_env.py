"""SubprocVecEnv whose environment processes do not load PyTorch.

SB3's SubprocVecEnv starts each environment process from a forkserver that first imports the
training script, and runs SB3's own worker, so every process loads PyTorch and SB3: about 445 MB
each, which limits how many simulators fit in memory. The environments here only need numpy,
gymnasium and the simulator link (about 51 MB), so these processes start empty and import
env_worker, which loads nothing heavier. A new process also re-runs the parent's main script unless
it is hidden while the process starts; see _main_script_hidden. The parent side is SB3's
SubprocVecEnv, so AsyncPPO can use its pipes.
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import sys

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import env_worker


class LightSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_kwargs: list[dict], info_keywords: tuple[str, ...] = ()):
        """env_kwargs: one dict of AutoDriveEnv arguments per environment."""
        self.waiting = False
        self.closed = False
        ctx = mp.get_context('forkserver')
        ctx.set_forkserver_preload([])     # the default preloads __main__, which imports torch
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in env_kwargs])
        self.processes = []
        for work_remote, remote, kwargs in zip(self.work_remotes, self.remotes, env_kwargs):
            process = ctx.Process(target=env_worker.worker, args=(work_remote, remote, kwargs, info_keywords),
                                  daemon=True)
            with _main_script_hidden():
                process.start()
            self.processes.append(process)
            work_remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_kwargs), observation_space, action_space)

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        # SB3 asks whether the environments record episodes (Monitor). The workers do, and asking
        # them would send SB3's Monitor class across, which would import PyTorch there after all.
        return [True] * len(self._get_indices(indices))


@contextlib.contextmanager
def _main_script_hidden():
    """multiprocessing tells a new process to run the parent's main script first (as __mp_main__)
    when that script has a __file__. Ours imports torch, so hide it while a process starts."""
    main = sys.modules['__main__']
    path = main.__dict__.pop('__file__', None)
    try:
        yield
    finally:
        if path is not None:
            main.__file__ = path
