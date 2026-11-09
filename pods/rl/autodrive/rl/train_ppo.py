"""PPO training on one or more running AutoDRIVE simulators, one environment per port."""
from __future__ import annotations

import argparse
import datetime
import os
import signal
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from async_ppo import AsyncPPO
from autodrive_env import REWARD_PRESETS
from light_vec_env import LightSubprocVecEnv

BASE_PORT = 4567
REWARD_TERMS = ('r_progress', 'r_lap', 'r_collision', 'r_speed', 'r_race_line', 'r_steer_rate', 'r_decel')


INFO_KEYWORDS = ('laps', 'collision', 'stalled', 'wrong_way')


class RaceLogCallback(BaseCallback):
    """Logs the reward terms and race results to TensorBoard."""

    def __init__(self):
        super().__init__()
        self.sums = {k: 0.0 for k in REWARD_TERMS}
        self.count = 0
        self.best_laps = 0
        self.ends = {'collision': 0, 'stalled': 0, 'wrong_way': 0, 'other': 0}
        self.lap_times: list[float] = []
        self.t_start = time.time()
        self.steps_at_start = None

    def _on_rollout_end(self) -> None:
        if self.count:
            for k, v in self.sums.items():
                self.logger.record(f'reward_terms/{k}', v / self.count)
            self.sums = {k: 0.0 for k in REWARD_TERMS}
            self.count = 0
        self.logger.record('race/best_laps_in_episode', self.best_laps)
        total = sum(self.ends.values())
        if total:
            for k, v in self.ends.items():
                self.logger.record(f'race/episode_end_{k}', v / total)
            self.ends = {k: 0 for k in self.ends}
        if self.lap_times:
            self.logger.record('race/mean_lap_time_s', float(np.mean(self.lap_times)))
            self.logger.record('race/best_lap_time_s', float(np.min(self.lap_times)))
            self.lap_times = []
        self.logger.record('time/env_steps_per_s',
                           (self.num_timesteps - self.steps_at_start) / max(time.time() - self.t_start, 1e-6))

    def _on_step(self) -> bool:
        if self.steps_at_start is None:
            self.steps_at_start = self.num_timesteps - 1
            self.t_start = time.time()
        for info, done in zip(self.locals['infos'], self.locals['dones']):
            if 'r_progress' in info:
                for k in REWARD_TERMS:
                    self.sums[k] += info[k]
                self.count += 1
            if done:
                end = next((k for k in ('collision', 'stalled', 'wrong_way') if info.get(k)), 'other')
                self.ends[end] += 1
                self.best_laps = max(self.best_laps, info.get('laps', 0))
                self.lap_times.extend(info.get('lap_times', []))
            if info.get('sim_stalled'):
                print('warning: a simulator stopped responding; waiting for it to come back', flush=True)
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-envs', type=int, default=1)
    ap.add_argument('--track', default='icra', choices=['icra', 'practice', 'none'],
                    help='none: no centre line, progress is distance driven forward')
    ap.add_argument('--total-steps', type=int, default=1_000_000)
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--resume', default=None, help='path of a saved model .zip to continue from')
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--n-steps', type=int, default=2048, help='steps per simulator between updates')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--n-epochs', type=int, default=10)
    ap.add_argument('--learning-rate', type=float, default=3e-4)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--race-line', default=None,
                    help='race line CSV (x_m, y_m): progress, laps and the line-following term use it')
    ap.add_argument('--reward', default=None, choices=sorted(REWARD_PRESETS),
                    help='reward weights preset (default: race_line with --race-line, otherwise basic)')
    ap.add_argument('--random-start', type=float, default=0.0,
                    help='chance that an episode starts at a random point on the track instead of the spawn '
                         '(default 0: off, because driving to the start point slows training to about '
                         '10 steps/s; track none cannot use it)')
    ap.add_argument('--checkpoint-every', type=int, default=50_000, help='total steps between checkpoints')
    args = ap.parse_args()

    def on_sigterm(signum, frame):      # `docker stop` should save the model like Ctrl+C does
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_sigterm)

    if args.reward is None:
        args.reward = 'race_line' if args.race_line else 'basic'
    if args.reward == 'race_line' and not args.race_line:
        ap.error('--reward race_line needs --race-line')
    print(f'reward preset {args.reward}: {REWARD_PRESETS[args.reward]}'
          + (f', race line {args.race_line}' if args.race_line else ''), flush=True)
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available in this container; using the CPU', flush=True)
        device = 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f'training on {torch.cuda.get_device_name(0)}', flush=True)

    run = args.run_name or datetime.datetime.now().strftime('ppo_%Y%m%d_%H%M%S')
    model_dir = os.path.join('models', run)
    os.makedirs(model_dir, exist_ok=True)

    env = LightSubprocVecEnv([dict(port=BASE_PORT + i, track_name=args.track, random_start=args.random_start,
                                   race_line=args.race_line, reward_weights=REWARD_PRESETS[args.reward])
                              for i in range(args.num_envs)], info_keywords=INFO_KEYWORDS)

    if args.resume:
        print(f'resuming from {args.resume}', flush=True)
        model = AsyncPPO.load(args.resume, env=env, device=device, tensorboard_log='runs')
    else:
        model = AsyncPPO('MlpPolicy', env, device=device, tensorboard_log='runs', verbose=1,
                    n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
                    learning_rate=args.learning_rate, gamma=args.gamma)

    checkpoints = CheckpointCallback(save_freq=args.checkpoint_every,
                                     save_path=model_dir, name_prefix='ppo')
    rate = args.num_envs * 20
    print(f'run {run}: {args.num_envs} simulator(s), about {rate} steps/s, so {args.total_steps:,} steps '
          f'takes about {args.total_steps / rate / 3600:.1f} h plus updates', flush=True)
    try:
        model.learn(total_timesteps=args.total_steps, callback=[RaceLogCallback(), checkpoints],
                    tb_log_name=run, reset_num_timesteps=not args.resume)
    except KeyboardInterrupt:
        print('stopping: saving the model first', flush=True)
    finally:
        final = os.path.join(model_dir, 'final')
        model.save(final)
        print(f'saved {final}.zip after {model.num_timesteps:,} steps', flush=True)
        try:
            env.env_method('close')
            env.close()
        except (EOFError, BrokenPipeError, ConnectionError):
            pass                        # the job is ending and the model is already saved


if __name__ == '__main__':
    main()
