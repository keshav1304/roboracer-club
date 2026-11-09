# Simulator notes

Measured on the Windows compete builds, with an RTX 4060 laptop GPU and 16 GB of RAM.

## How the simulator behaves

**Timing:**

- It runs in real time and can't be paused or sped up, so each simulator gives at most 20 steps per
  second.
- It sends the next frame only after it receives a reply. The first frame in batch mode has no
  LIDAR field.

**Control:**

- **Reset:** `V1 Reset` returns the car to the spawn and clears the lap and collision counters. While
  it is `True` the car is held there, which is how cars wait during policy updates.
- **No placing:** pose commands are ignored, so the car can only start at the spawn.
- **Steering:** `V1 Steering` is in radians, with 0.5236 at full lock. The car's minimum turning
  radius is about 0.57 m.
- **Throttle:** it settles at 24.86 m/s per unit of throttle command. The policy's throttle is capped
  at 0.32.
- **Braking:** there is no brake. Lifting off the throttle slows the car at 6–7 m/s².

**Sensors and tracks:**

- **LIDAR:** 1,081 beams over 270°, ordered right to left. They are reduced to 108 bins.
- **Tracks:** each build contains only one track. The `icra` centre line matches the ICRA build.
- **Hairpins on `icra`:**

  | Distance along the centre line | Radius |
  | --- | --- |
  | 14 m | 1.1 m |
  | 33 m | 0.88 m |
  | 39 m | 0.73 m |

  Most crashes happen at the hairpins at 33 m and 39 m.

## Window and headless modes

- **Headless:** `-batchmode` connects by itself to `-ip` and `-port`.
- **Window:** it ignores `-ip` and `-port`. Open the menu at the top left and click **Disconnected**.
- **Never use `-nographics`.** It caused a blue screen (`0x10E`, GPU memory) within seconds on the
  development laptop.

## How many simulators fit

- **CPU:** each headless simulator uses about 1.3 CPU cores. The CPU limit is about 12–14
  simulators.
- **Memory:** each simulator also uses about 400 MB of VRAM and reserves about 1.2 GB of RAM, of
  which 60–145 MB is actively used.
- **In practice:** RAM is the limit. With 16 GB, 5 simulators fit alongside Docker and a browser,
  giving about 80 steps per second. A load test held 6 simulators at 19.9 Hz (115 steps per second),
  using 57% CPU and 37% GPU.
- **Trainer processes:** the trainer's environment processes are kept small
  ([rl/light_vec_env.py](../rl/light_vec_env.py)): about 53 MB each, against 445 MB for
  Stable-Baselines3's default.

## Known problems

- **Closing simulators:** `taskkill` run from Git Bash can fail silently, leaving simulators running.
  Simulators left over from an earlier run share the ports and corrupt training data. Use menu
  option 9, or this command in PowerShell:

  ```powershell
  Get-Process -Name 'AutoDRIVE Simulator','UnityCrashHandler64' -ErrorAction SilentlyContinue | Stop-Process -Force
  ```

- **Container paths:** in Git Bash, set `MSYS_NO_PATHCONV=1` before running `docker` by hand.
  Otherwise paths such as `/workspace` are rewritten. The launcher already sets it.
- **Docker after a Windows restart:** if Docker Desktop won't start, rename the `run` and
  `secrets-engine` folders in its data folder, then start it again.

## Differences from the gym environment

The observation and action are the same as in [reference/race_env_gym.py](../reference/race_env_gym.py).
The reward and the following parts differ:

- **Physics:** the simulator supplies real physics, so the gym's throttle fit, grip cap and random
  friction are not used.
- **Starts:** random starts are driven by a scripted driver instead of placing the car (see
  [reward.md](reward.md)).
- **Collisions:** they come from the simulator's own counter.
