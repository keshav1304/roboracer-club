"""PPO whose rollout collection does not wait for all simulators at each step.

SB3 steps every environment together, which suits simulators that wait for you. These do not: they
run in real time. In lockstep, one car's crash reset (about 0.7 s) would stall the others while
their cars carry on with their last command, so every crash corrupts the other cars' transitions
and adding simulators barely adds throughput.

Here each simulator gets a thread that steps its environment process on its own schedule and fills
its own column of the rollout buffer. The PPO update itself is unchanged.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv


class AsyncPPO(PPO):
    """Needs a SubprocVecEnv, for direct access to the pipe of each environment process."""

    def _excluded_save_params(self) -> list[str]:
        # Collection state holds thread locks, which cannot be pickled, and is rebuilt for each rollout.
        return super()._excluded_save_params() + [
            '_collect_lock', '_collect_stop', '_collect_continue', '_last_values', '_last_dones',
            '_step_time_sum', '_step_time_count']

    def _policy_step(self, obs: np.ndarray):
        with th.no_grad():
            actions, values, log_probs = self.policy(obs_as_tensor(obs[None], self.device))
        action = actions.cpu().numpy()[0]
        return action, np.clip(action, self.action_space.low, self.action_space.high), values, log_probs

    def _collect_one(self, env_idx: int, remote, callback: BaseCallback, buffer: RolloutBuffer, n_steps: int) -> None:
        lock, stop = self._collect_lock, self._collect_stop

        # The car was held at the spawn during the update, so every rollout starts a new episode.
        remote.send(('reset', (None, None)))
        obs, _ = remote.recv()
        episode_start, done = True, False
        prev_send = None                 # None after a reset, so a reset is not timed as a control step

        for t in range(n_steps):
            if stop.is_set():
                return
            with lock:
                action, clipped, value, log_prob = self._policy_step(obs)

            now = time.perf_counter()
            period = None if prev_send is None else now - prev_send
            prev_send = now
            remote.send(('step', clipped))
            new_obs, reward, done, info, _ = remote.recv()

            with lock:
                if done and info.get('TimeLimit.truncated') and not info.get('sim_stalled'):
                    terminal = obs_as_tensor(info['terminal_observation'][None], self.device)
                    with th.no_grad():
                        reward += self.gamma * self.policy.predict_values(terminal).item()

                buffer.observations[t, env_idx] = obs
                buffer.actions[t, env_idx] = action
                buffer.rewards[t, env_idx] = reward
                buffer.episode_starts[t, env_idx] = episode_start
                buffer.values[t, env_idx] = value.cpu().numpy().flatten()[0]
                buffer.log_probs[t, env_idx] = log_prob.cpu().numpy().flatten()[0]

                self.num_timesteps += 1
                if period is not None:       # command to command, so policy time counts as well
                    self._step_time_sum += period
                    self._step_time_count += 1
                self._update_info_buffer([info], np.array([done]))
                callback.update_locals({'infos': [info], 'dones': np.array([done])})
                if not callback.on_step():
                    self._collect_continue = False
                    stop.set()
                    return
            obs, episode_start = new_obs, done
            if done:
                prev_send = None

        with lock:
            with th.no_grad():
                self._last_values[env_idx] = self.policy.predict_values(obs_as_tensor(obs[None], self.device)).item()
            self._last_dones[env_idx] = done

        # This car is finished for now: freeze it until the others are done and the update has run.
        remote.send(('env_method', ('hold', [], {})))
        remote.recv()

    def collect_rollouts(self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer,
                         n_rollout_steps: int) -> bool:
        if not isinstance(env, SubprocVecEnv):
            raise TypeError('AsyncPPO needs a SubprocVecEnv')
        if not isinstance(self.action_space, spaces.Box):
            raise TypeError('AsyncPPO supports continuous actions only')

        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()

        self._collect_lock = threading.Lock()
        self._collect_stop = threading.Event()
        self._collect_continue = True
        self._last_values = np.zeros(env.num_envs, dtype=np.float32)
        self._last_dones = np.zeros(env.num_envs, dtype=bool)
        self._step_time_sum, self._step_time_count = 0.0, 0
        errors: list[BaseException] = []

        def run(i: int, remote) -> None:
            try:
                self._collect_one(i, remote, callback, rollout_buffer, n_rollout_steps)
            except BaseException as e:    # noqa: BLE001  raised again on the main thread
                errors.append(e)
                self._collect_stop.set()

        threads = [threading.Thread(target=run, args=(i, r), daemon=True, name=f'collect-{i}')
                   for i, r in enumerate(env.remotes)]
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(0.25)
        except KeyboardInterrupt:         # let each thread finish its step, so the pipes stay in order
            self._collect_stop.set()
            for t in threads:
                t.join(30.0)
            raise
        if errors:
            raise errors[0]
        if not self._collect_continue:
            return False

        rollout_buffer.pos = n_rollout_steps
        rollout_buffer.full = True
        rollout_buffer.compute_returns_and_advantage(last_values=th.as_tensor(self._last_values),
                                                     dones=self._last_dones)
        if self._step_time_count:
            self.logger.record('time/control_hz', self._step_time_count / self._step_time_sum)
        callback.on_rollout_end()
        return True
