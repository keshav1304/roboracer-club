# RoboRacer Playground

A self-contained web playground for the AutoDRIVE RoboRacer simulator:
pick a racing algorithm, tune parameters live, and watch LiDAR /
telemetry visualizations — while the Unity simulator runs on your
computer.

It also includes **Map-based** algorithms: a guided three-step workflow that
maps the track with SLAM (driven by a slow wall follow or manually),
generates and hand-edits an optimal raceline, and tracks it with pure
pursuit or MPC — all from the browser. See [§4 Map-based](#4-map-based).

Everything for the playground lives under this folder. It includes its
own copy of the AutoDRIVE Devkit (`gateway/autodrive_devkit/`) and does
**not** depend on `AutoDRIVE-RoboRacer-Sim-Racing/autodrive_devkit/`.

```
Browser (Next.js)  <--WSS-->  Gateway + ROS (Docker)  <--Socket.IO :4567-->  Unity Sim (laptop)
```

| Piece | Path | Runs where |
|-------|------|------------|
| Web UI | `web/` | local (`npm run dev`) or Vercel |
| Gateway + ROS bridge | `gateway/` + `Dockerfile` | local Docker or Fly.io |
| Algorithms | `gateway/autodrive_devkit/racing_algo/` | inside the backend container |
| Simulator | AutoDRIVE practice build (sibling repo) | your computer (GUI) |

```
roboracer-playground/
├── Dockerfile              # backend image (bridge + gateway + slam + AMCL)
├── entrypoint.sh
├── docker-compose.yml      # local backend (mounts ./data for maps/racelines)
├── fly.toml                # Fly.io deploy
├── data/                   # saved maps + racelines (created on first save)
├── gateway/
│   ├── main.py             # FastAPI WebSocket gateway
│   ├── slam.py             # slam_toolbox session manager (Mapping)
│   ├── localize.py         # map_server + AMCL session manager (racing)
│   ├── raceline.py         # raceline engine: spline editing + CMA-ES worker
│   ├── store.py            # map/raceline artifact store (/data)
│   ├── requirements.txt
│   └── autodrive_devkit/   # bundled ROS package (bridge + algos)
└── web/                    # Next.js frontend
```

---

## 1. Run locally

### Prerequisites

- Docker Desktop (or Docker Engine + Compose)
- Node.js 18+
- AutoDRIVE Simulator (practice build) from `../AutoDRIVE-RoboRacer-Sim-Racing/`

### Step A — Backend (gateway + ROS bridge)

From this `roboracer-playground/` directory:

```bash
cd roboracer-playground
docker compose up --build
```

This starts:

| Port | Service |
|------|---------|
| **4567** | AutoDRIVE Socket.IO bridge (simulator connects here) |
| **8000** | Playground gateway (`ws://localhost:8000/ws`, health at `http://localhost:8000/health`) |

Leave this terminal running. To stop: `Ctrl+C`, or `docker compose down`.

### Step B — Web UI

In a second terminal:

```bash
cd roboracer-playground/web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

The UI defaults to `ws://localhost:8000/ws`. You can change the Gateway
URL in the **Connection** tab if needed.

### Step C — Simulator

1. Launch the AutoDRIVE Simulator (practice build) on your machine.
2. Menu Panel → IP `127.0.0.1`, Port `4567` → **Connection**.
3. In the web UI, the Connection tab should show the simulator as connected.
4. Pick an algorithm (Wall Follow or Follow the Gap) → **Start**.
5. In the simulator, switch to **Autonomous** driving mode.
6. Use the **Tuning** tab sliders — parameters apply live (no restart).

---

## 2. Deploy online (Fly.io + Vercel)

### Backend on Fly.io

Install the [Fly CLI](https://fly.io/docs/hands-on/install-flyctl/) and
log in (`fly auth login`). Then from this folder:

```bash
cd roboracer-playground

# First time only — keep the existing fly.toml when prompted
fly launch --no-deploy

# Shared password for the WebSocket gateway (users enter this in the UI)
fly secrets set PLAYGROUND_TOKEN=your-club-password

# Dedicated IPv4 so the Unity sim can dial :4567 over the public internet
fly ips allocate-v4

# Build and deploy
fly deploy

# Note the dedicated IPv4
fly ips list
```

After deploy:

- Health check: `https://<app>.fly.dev/health`
- Browser WebSocket: `wss://<app>.fly.dev/ws`
- Simulator: put `<dedicated-IPv4>` in the sim IP field, port `4567`

**Smoke-test order:** deploy → open `/health` → connect the sim with the
Fly IPv4 → confirm `sim_connected: true` in `/health`.

### Frontend on Vercel

From `roboracer-playground/web/`:

```bash
cd roboracer-playground/web
npx vercel
```

Set these environment variables in the Vercel project settings (or via
`vercel env add`):

| Variable | Value |
|----------|-------|
| `NEXT_PUBLIC_ROS_GATEWAY_URL` | `wss://<app>.fly.dev/ws` |
| `NEXT_PUBLIC_SIM_BRIDGE_ADDRESS` | `<dedicated-IPv4> : 4567` |

Redeploy after setting env vars so the build picks them up. Users still
enter the club token (same as `PLAYGROUND_TOKEN`) in the header field;
it is stored in `localStorage`.

### Plan B — backend local, UI hosted

If campus firewalls block port 4567 or WAN bandwidth is too high for the
sim’s Bridge payload:

1. Run the backend locally (`docker compose up --build`); sim connects to
   `127.0.0.1:4567`.
2. Expose only the gateway to the hosted UI, e.g.
   `cloudflared tunnel --url http://localhost:8000`.
3. Paste the tunnel URL (`wss://…/ws`) into the **Gateway URL** field in
   the web UI (and optionally set `NEXT_PUBLIC_ROS_GATEWAY_URL` on Vercel
   to that tunnel).

---

## 3. Gateway API

`GET /health` → `{ ok, sim_connected, algorithm, slam_active, opt_running }`

`WS /ws?token=...` (token required when `PLAYGROUND_TOKEN` is set)

**REST (artifacts, same token via `?token=...`):**

| Endpoint | Returns |
|----------|---------|
| `GET /maps` | saved map metadata list |
| `GET /maps/{name}/image.png` | map thumbnail (PNG from PGM) |
| `GET /maps/{name}/grid` | Int8 occupancy (`occ_b64`) for canvas rendering |
| `GET /racelines` | saved raceline metadata list |
| `GET /racelines/{name}` | raceline meta + `s/x/y/theta/v` arrays |

**Client → server (WS):**

```json
{"type": "set_params", "params": {"kp": 7.2, "desired_dist": 0.8}}
{"type": "start_algo", "algorithm": "wall_follow"}
{"type": "start_algo", "algorithm": "pure_pursuit", "raceline": "icra-fast"}
{"type": "stop_algo"}
{"type": "reset"}
{"type": "ping", "t": 123}

{"type": "slam_start", "drive": "wall_follow",
 "params": {"throttle": 0.03, "desired_dist": 0.8, "kp": 6.8}}
{"type": "slam_stop"}
{"type": "slam_save_map", "name": "track-20260719"}

{"type": "map_edit", "name": "track-20260719",
 "ops": [{"tool": "erase", "points": [[120, 88], [126, 90]], "radius": 3}]}
{"type": "map_undo", "name": "..."}   {"type": "map_redo", "name": "..."}
{"type": "map_delete", "name": "..."} {"type": "map_rename", "old": "...", "new": "..."}

{"type": "raceline_update", "map": "track-20260719", "req_id": 7,
 "anchors": [[1.2, -3.4], "..."],
 "params": {"vmin": 0.5, "vmax": 6.5, "alat": 3.5, "smooth": 2.0}}
{"type": "raceline_save", "name": "icra-fast", "map": "track-20260719",
 "data": {"s": [], "x": [], "y": [], "theta": [], "v": []},
 "params": {}, "anchors": []}
{"type": "raceline_delete", "name": "..."}

{"type": "opt_start", "map": "track-20260719",
 "params": {"margin": 0.25, "budget": 8000, "n_ctrl": 80}}
{"type": "opt_cancel"}
```

`set_params` targets the currently running algorithm by default; pass
`"algorithm": "follow_gap"` to target one explicitly.

**Server → client (WS):** `telemetry` (~20 Hz, includes `algo_debug` when an
algorithm is running), `lidar` (~10 Hz, strided), `status` (1 Hz, includes
`slam` and `raceline` fields), `map_frame` (≤1 Hz raw Int8 occupancy as
`occ_b64` plus origin/resolution and the `map→world` TF; rendered on the
client with a continuous RViz-style colormap), `map_updated`,
`maps_changed`, `racelines_changed`, `opt_progress` (optimizer events:
`started`/`progress` with a preview polyline/`done`/`error`/`cancelled`),
`raceline_data` (recomputed line for the editor, echoes `req_id`),
`params_ack`, `algo_ack`, `slam_ack`, `map_ack`, `raceline_ack`, `opt_ack`,
`reset_ack`, `error`, `pong`.

REST also exposes `GET /maps/{name}/grid` (same Int8 occupancy payload) for
saved maps; `GET /maps/{name}/image.png` remains for list thumbnails.
### Algorithm debug payloads

- **wall_follow:** `ray_a`, `ray_b`, `alpha`, `car_dist`, `car_dist_future`,
  `error`, `p_term`, `i_term`, `d_term`, `desired_dist`, `theta_rad`,
  `lookahead`
- **follow_gap:** `best_point_angle`, `best_point_range`, `closest_range`,
  `gap_start_angle`, `gap_end_angle`, `fov_deg`
- **pure_pursuit:** `s_curr`, `target_x`, `target_y`, `y_local`,
  `steering_norm`, `target_velocity`, `speed`, `speed_error`, `lookahead`,
  `throttle`

### Tunable parameters

- **wall_follow** (`/wall_follow_node`): `kp`, `kd`, `ki`, `desired_dist`,
  `lookahead`, `throttle`, `max_steering_rad`, `theta_deg`
- **follow_gap** (`/follow_gap_node`): `throttle`, `vehicle_half_width`,
  `free_space_threshold`, `best_point_threshold`, `disparity_threshold`,
  `smoothing_window`, `fov_deg`, `heading_weight`, `max_steering_rad`
- **pure_pursuit** (`/pure_pursuit_node`): `lookahead`, `velocity_scale`,
  `throttle_gain`, `corner_throttle_gain`, `corner_steering_threshold`,
  `kp_speed`, `ki_speed`, `integral_limit`, `max_steering_rad`,
  `steering_direction`, `path_direction`
  (plus `waypoint_file`, set at launch by the gateway)

From a shell inside the container:

```bash
docker exec -it roboracer_playground_backend bash
source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
ros2 param set /wall_follow_node desired_dist 0.9
ros2 param set /follow_gap_node throttle 0.06
```

---

## 4. Map-based

Open the **Map-based** tab in the web UI. It is a three-step workflow;
each step is a collapsible panel with its own controls, and the center
canvas switches to a pan/zoomable track view (wheel = zoom, drag = pan,
Shift-drag = pan while editing).

### Step 1 — Build Map

1. Connect the simulator (Connection tab) and put it in the driving mode
   you chose:
   - **Auto — slow wall follow**: sim in *Autonomous* mode; the backend
     drives a conservative wall-follow lap (throttle slider applies live).
   - **Manual**: drive yourself, slowly and smoothly.
2. Click **Start Mapping**. slam_toolbox starts on the backend and the
   live occupancy grid grows on the canvas (car pose + trail overlaid).
   In wall-follow mode, the same tunable parameters as Playground Wall
   Follow are available before and during mapping (live via SetParameters).
3. Drive several slow laps so loop closures can refine the map, then
   click **Save Map…** and give it a name (e.g. `track-20260719`). Maps
   are stored under `/data/maps/<name>/` (yaml + pgm + meta, including
   the `map→world` transform at save time).
4. Optional cleanup: open a saved map and use the **Erase** / **Draw
   wall** brushes to remove speckle or close gaps — with undo/redo. Edits
   apply to the stored PGM immediately (versioned, last 20 steps kept).

### Step 2 — Generate Raceline

1. Pick a saved map.
2. **Generate raceline** runs the CMA-ES minimum-curvature optimizer
   (same engine as `generate_waypoints.py`) in a background process.
   A progress bar tracks evaluations and the dashed preview line animates
   as the optimizer improves. **Cancel** any time.
3. When it finishes, the line appears color-coded by target velocity
   (blue = slow, red = fast) with ~40 draggable anchor handles:
   - drag an anchor to reshape the line (the backend refits the spline,
     Frenet frame, and velocity profile in real time);
   - click the line between anchors to insert one;
   - right-click an anchor (or select + Delete) to remove it;
   - segments closer to a wall than the margin get a red halo.
4. The velocity sliders (v-min / v-max / lateral accel / smoothing)
   recompute instantly without re-optimizing. The bottom chart shows the
   v(s) profile; stats show length, est. lap time, min clearance.
5. **Save Raceline…** writes `/data/racelines/<name>/waypoints.csv`
   (`s,x,y,theta,velocity` — same format as `comp_waypoints.csv`).

### Step 3 — Controller

1. Pick a raceline and controller (Pure Pursuit or MPC), put the sim in
   **Autonomous** mode, click **Start**. The gateway starts a Nav2
   particle filter (AMCL) on the raceline's parent map (seeded from
   current odom via the saved `map→world` transform), then launches the
   selected controller against that raceline's CSV.
2. Tune controller parameters live (lookahead / velocity scale for Pure
   Pursuit; cost weights and throttle gains for MPC).
3. The canvas shows the raceline, car, trail, and controller overlay
   (lookahead target or MPC horizon); the side charts show v(s) with the
   car's live position and speed, plus pose error; the left panel keeps
   a lap-time board. Status includes `localize.localized` once
   particle-filter covariance has converged.

Notes:

- Maps and racelines persist in `roboracer-playground/data/` (mounted as
  `/data` in the container) across rebuilds.
- Mapping and controllers are mutually exclusive; the gateway rejects
  map controllers while SLAM is active (and stops the particle filter
  when mapping starts).
- **Controller pose** uses the particle-filter estimate in the `map`
  frame (same frame as the raceline CSV). **UI / telemetry** still use
  simulator odometry in `world` (ground truth for overlays and
  reporting). If the particle-filter pose goes stale, the controller
  holds zero throttle/steer rather than falling back to odom.

---

## 5. Notes

- Edit algorithms under `gateway/autodrive_devkit/racing_algo/`, then
  rebuild the backend image (`docker compose up --build`). Changes in
  `AutoDRIVE-RoboRacer-Sim-Racing/autodrive_devkit/` are **not** used by
  the playground.
- **Do not upgrade** the pinned Socket.IO/Flask stack in `Dockerfile` —
  the Unity sim speaks that exact old engine.io protocol version.
- One simulator session per backend instance. Multiple browsers can
  watch; they share one car.
- Camera is not forwarded to the browser (bandwidth); LiDAR + pose are
  the primary visualizations.
- Use the **practice** simulator build, not compete.
- Heavy optional trees (`autodrive_rl`, `slam_toolbox_panel`) were left
  out of the bundled Devkit to keep the image small; add them under
  `gateway/autodrive_devkit/` if you need them later. (`slam_toolbox`
  and Nav2 AMCL / map_server are installed via apt for Mapping
  and racing localization.)
