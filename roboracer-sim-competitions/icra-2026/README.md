# ICRA 2026 online sim racing

Penn RoboRacer stack for the AutoDRIVE RoboRacer Sim-Racing League (ICRA 2026, online). We race **Pure Pursuit** on a mapped raceline.

The organizers provide the Unity simulator, the ROS 2 Socket.IO bridge, and the Docker submission pattern. This folder is our racing stack on top of that: algorithms, launch files, maps, and waypoints.

```
Simulator (host)  --Socket.IO :4567-->  Docker (ROS 2 Humble + Pure Pursuit)
```

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose)
- The AutoDRIVE **practice** simulator for your OS (not in this folder)

Node.js is not required to build or run.

## 1. Get the simulator

1. Open the [ICRA 2026 release](https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing/releases/tag/2026-icra).
2. Download `autodrive_simulator_practice_XXX.zip` for your OS.
3. Extract it anywhere (for example next to this folder).

**macOS only** — grant execute permission and drop the quarantine flag:

```bash
cd /path/to/extracted/autodrive_simulator
sudo chmod -R +x "AutoDRIVE Simulator.app/Contents/MacOS"
xattr -d com.apple.quarantine "AutoDRIVE Simulator.app"
```

## 2. Build the container

From this `icra-2026/` directory:

```bash
docker build --platform linux/amd64 \
  --tag penn-roboracer/icra-2026-api:latest \
  -f autodrive_devkit.Dockerfile .
```

Use `--platform linux/amd64` on Apple Silicon. Omit it on native Linux/amd64.

The image is ROS 2 Humble. It already includes `foxglove-bridge` and `slam-toolbox`.

## 3. Run the container

**macOS / Windows:** map ports (`-p 4567:4567`). The official organizer docs use `--network=host`, which is Linux-only.

Do not launch `bringup_headless.launch.py` in the same container as the default entrypoint — the entrypoint already starts the race stack.

### Race (matches the submission entrypoint)

This starts `race.launch.py`: the AutoDRIVE bridge plus Pure Pursuit.

```bash
docker run --platform linux/amd64 --name icra-2026-api --rm -it \
  -p 4567:4567 -p 8765:8765 \
  penn-roboracer/icra-2026-api:latest
```

### Dev (edit code without rebuilding)

Overrides the entrypoint and mounts this folder’s `autodrive_devkit/` into the container:

```bash
docker run --platform linux/amd64 --name icra-2026-api --rm -it \
  -p 4567:4567 -p 8765:8765 \
  -v "$(pwd)/autodrive_devkit":/home/autodrive_devkit/src/autodrive_devkit \
  --entrypoint /bin/bash \
  penn-roboracer/icra-2026-api:latest
```

Inside the container, rebuild if you changed Python/launch files, then start the stack:

```bash
cd /home/autodrive_devkit && colcon build && source install/setup.bash
ros2 launch autodrive_roboracer race.launch.py
```

Bridge only (no racing algorithm):

```bash
ros2 launch autodrive_roboracer bringup_headless.launch.py
```

## 4. Open the simulator

1. Launch the practice app (on Mac, double-click `AutoDRIVE Simulator.app`).
2. Set IP `127.0.0.1`, port `4567`, then **Connect**.
3. Wait for `Connected!` in the Docker terminal.
4. Switch the vehicle to **Autonomous**.

## 5. Optional: Foxglove

1. Open [app.foxglove.dev](https://app.foxglove.dev/).
2. **Open Connection** → Foxglove WebSocket → `ws://localhost:8765`.
3. Import the layout `foxglove/AutoDRIVE RoboRacer Sim Racing League.json`.

If Foxglove is not already running (dev / bringup-only), exec into the container and start it:

```bash
docker exec -it icra-2026-api bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

## What we race

| Piece | Path |
|-------|------|
| Controller | `autodrive_devkit/racing_algo/pure_pursuit_node.py` |
| Raceline | `autodrive_devkit/comp_waypoints.csv` |
| Launch | `autodrive_devkit/launch/race.launch.py` |

Alternatives in the same package: `wall_follow` and `follow_gap` (`autodrive_devkit/racing_algo/`). Swap the executable in `race.launch.py` if you want to try them.

## Submission

Evaluators pull a Docker Hub image and run it with `--network=host` next to **their** simulator container. We submit only the **devkit** image, not the simulator.

Typical organizer run (Linux):

```bash
docker run --name autodrive_roboracer_api --rm -it \
  --network=host --ipc=host \
  --entrypoint /bin/bash \
  <TEAM_USERNAME>/<IMAGE_NAME>:<TAG>
```

Our image’s default entrypoint already launches `race.launch.py`. If they override `--entrypoint /bin/bash`, start the race with `ros2 launch autodrive_roboracer race.launch.py`.

## Appendix: remake the map

Pure Pursuit does **not** run SLAM while racing. Use this only to rebuild a map and raceline.

1. Start the container in **dev** mode and launch `bringup_headless.launch.py` (not `race.launch.py`).
2. Connect the simulator in **Manual** mode.
3. In a second container shell:

```bash
ros2 run slam_toolbox async_slam_toolbox_node --ros-args \
  --params-file /home/autodrive_devkit/src/autodrive_devkit/f1tenth_online_async.yaml \
  -r __node:=slam_toolbox
```

4. Drive one slow lap. Save:

```bash
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap \
  "{name: {data: 'my_map'}}"
```

5. Copy `my_map.yaml` / `my_map.pgm` into the mounted `autodrive_devkit/`, then generate waypoints:

```bash
cd autodrive_devkit
python3 generate_waypoints.py --map comp_track.yaml --out comp_waypoints.csv
```

Or open `raceline_explorer_gui.py` / `waypoint_editor_gui.py`.

## Citation

The bridge, Docker pattern, and simulator are part of the [AutoDRIVE Ecosystem](https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing). If you use that framework, cite:

- Samak et al., “AutoDRIVE: A Comprehensive, Flexible and Integrated Digital Twin Ecosystem…”, *MDPI Robotics*, 2023. [doi:10.3390/robotics12030077](https://doi.org/10.3390/robotics12030077)
- Samak et al., “AutoDRIVE Simulator…”, CCRIS 2021. [doi:10.1145/3483845.3483846](https://doi.org/10.1145/3483845.3483846)
