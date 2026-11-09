# AutoDRIVE RoboRacer PPO

Trains a PPO driving policy in the real **AutoDRIVE Simulator** (RoboRacer, ICRA 2026 compete build).
The simulators run on Windows. The trainer runs in a Docker container that uses the GPU. Each
simulator connects to its own port on the container.

- **Policy input:** 108 LIDAR bins, plus speed and steering angle.
- **Policy output:** steering and throttle, each from -1 to 1.

These match the club's gym environment, so policies work in both.
[reference/race_env_gym.py](reference/race_env_gym.py) is the snapshot of the gym that this was
matched to. The current gym is [../env/race_env.py](../env/race_env.py), which has since added
random surface friction.

| Model | Track | Result |
| --- | --- | --- |
| [`ppo_20261107_182816`](models/README.md) | Oct 24 build (`none`) | 8 laps per episode, best lap 10.8 s |
| [`ppo_20261108_054349`](models/README.md) | ICRA, race-line reward | Drives about 7 s on average, no full lap yet |
| [`ppo_20261106_062208`](models/README.md) | ICRA, basic reward | 1 lap at best |

For the earlier runs and what each one tried, see [docs/training-history.md](docs/training-history.md).

## Requirements

- Windows 10 or 11 with an NVIDIA GPU
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) using the WSL 2 backend
- [Git for Windows](https://git-scm.com/download/win), which provides **Git Bash**
- The AutoDRIVE Simulator compete build for Windows (not in this repository)
- About 1.2 GB of free RAM for each simulator

## Setup

1. Copy the settings template and set your simulator path in it:

   ```bash
   cp launch.local.example.bash launch.local.bash
   ```

2. Start Docker Desktop.
3. Open **Git Bash** in this folder and start the menu:

   ```bash
   ./launch.bash
   ```

   The first start builds the Docker image, which takes several minutes.
4. Choose option **6**. It checks the GPU and drives one scripted lap.

In PowerShell, plain `bash` is WSL's bash and won't work. Run Git Bash's bash instead:

```bash
& "C:\Program Files\Git\bin\bash.exe" launch.bash
```

## Using the menu

| Option | What it does |
| --- | --- |
| 1, 2 | Start training a new model: 1 with one simulator window, 2 with several headless simulators |
| 3 | Continue training from a saved `.zip` |
| 4, 5 | Test a model: 4 in the simulator window, 5 headless (prints laps and lap times) |
| 6 | Check the setup |
| 7 | Start TensorBoard at http://localhost:6006 |
| 8, 9 | Status; stop every container and simulator |
| t, r | Change the track (`icra`, `practice`, `none`); set or clear the race line |

- **Stopping:** press Ctrl+C to stop a job. Training saves `final.zip` before it exits.
- **Simulator window:** it doesn't connect by itself. Click the menu icon at its top left, then
  click **Disconnected**. Headless simulators connect on their own.

**Tracks:**

- **`icra`:** the ICRA 2026 compete track.
- **`practice`:** the club's practice track. Neither compete build contains it.
- **`none`:** for a simulator whose track has no centre line in `tracks/`, such as the Oct 24
  build. Progress is then the forward distance driven, and the simulator counts the laps.
  `SIM_EXE_NONE` selects that simulator.

**Outputs:**

| Output | Where |
| --- | --- |
| Models, with a checkpoint every 50,000 steps | `models/<run>/` |
| TensorBoard logs | `runs/<run>_<n>/` |
| Simulator logs | `logs/sim_<port>.log` |

Training settings can be passed through, for example
`EXTRA_TRAIN_ARGS="--learning-rate 1e-4" ./launch.bash`. For all options, see
`python rl/train_ppo.py --help`.

## Repository layout

| Path | Contents |
| --- | --- |
| [launch.bash](launch.bash) | The menu. Starts, watches and stops the simulators and the training container |
| [launch.local.example.bash](launch.local.example.bash) | Template for `launch.local.bash`, which holds your simulator paths and isn't committed |
| [docker/Dockerfile](docker/Dockerfile) | Training image: ROS 2 Humble, the simulator's pinned Socket.IO packages, PyTorch (CUDA), Stable-Baselines3 |
| [rl/sim_bridge.py](rl/sim_bridge.py) | Socket.IO server that one simulator connects to; no ROS needed |
| [rl/autodrive_env.py](rl/autodrive_env.py) | Gymnasium environment: observation, action, reward presets, episode rules |
| [rl/track.py](rl/track.py) | Centre-line and race-line geometry: distance along the line and sideways offset |
| [rl/train_ppo.py](rl/train_ppo.py) | Training entry point |
| [rl/async_ppo.py](rl/async_ppo.py) | PPO that collects from each simulator on its own clock, since simulators run in real time |
| [rl/light_vec_env.py](rl/light_vec_env.py), [rl/env_worker.py](rl/env_worker.py) | Lightweight environment processes (about 50 MB each instead of 445 MB), so more simulators fit in RAM |
| [rl/eval_ppo.py](rl/eval_ppo.py) | Test a model: laps, lap times and why each episode ended |
| [rl/check_env.py](rl/check_env.py) | Scripted lap that checks the environment against a simulator (menu option 6) |
| [rl/diagnose_run.py](rl/diagnose_run.py) | Records every step of a policy to CSV, to find where and why it crashes |
| [tracks/](tracks/README.md) | Centre lines, race line and maps for each track |
| [models/](models/README.md) | The trained models kept in the repository |
| [reference/race_env_gym.py](reference/race_env_gym.py) | Snapshot of the club's gym environment that the observation and action were matched to |
| [docs/reward.md](docs/reward.md) | The reward presets and episode rules |
| [docs/simulator-notes.md](docs/simulator-notes.md) | Simulator behaviour, performance limits and known problems |
| [docs/training-history.md](docs/training-history.md) | Every training run so far: what it tried and what it reached |

Not committed: `launch.local.bash`, checkpoints, `runs/`, `logs/` and `archive/` (old runs).
