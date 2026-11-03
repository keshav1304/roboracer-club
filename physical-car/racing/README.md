# Racing on the physical car: map → raceline → race

Lives on the Jetson at `~/racing/` (mirrored here in `physical-car/racing/`).
`~/bin/race` links to the `race` script, so `race` works from any terminal on the car.

| Step | Command | Result |
|---|---|---|
| 0. Connect | `ssh nvidia@<car-ip>` (USB-C: `192.168.55.1`) | |
| 1. Map | `race map`, drive the track under RC | live `/map` |
| 1b. Save | `race save track1` (second terminal) | `maps/track1.yaml` + `.pgm` |
| 2. Raceline | `race editor`, open `http://<car-ip>:8090` | `maps/track1_raceline.csv` |
| 3. Race | `race run track1` | AMCL + pure pursuit on `/drive` |

`maps/` = `~/osracer_ws/src/osracer/osracer_slam/maps/` (where `racecar mapping save` writes).
`race list` shows the maps and which ones have racelines. `race stop` kills everything and sends a zero `/drive`.

## Prerequisites

- `neoracer-teleop` service running (`racecar service status`). It provides `/scan`, `/odom` and the `/drive` mux.
- Foxglove for watching: the bridge already runs at boot (`neoracer-foxglove`, a systemd user service).
  Connect Foxglove to `ws://<car-ip>:8765`. `race foxglove` prints the URL, and starts the bridge only if it isn't running.

## 1. Map

```bash
race map                 # slam_toolbox (also: race map gmapping | cartographer)
# drive slowly under RC, cover the whole track, come back to the start to close the loop
race save track1         # in a second terminal, while SLAM is still running
# then Ctrl-C the mapping terminal
```

## 2. Raceline (browser GUI)

```bash
race editor              # prints the URL(s); Ctrl-C to stop
```

- Pick the map. **Click** to add waypoints in driving order around the track, **click on the line** to insert one,
  **drag** to move, **right-click** to delete. Scroll to zoom, drag empty space to pan, Ctrl/⌘-Z to undo.
- Waypoint 1 (blue) is the start; arrows show the driving direction. **Reverse dir** flips it.
- A closed spline is fitted through the waypoints and resampled every 5 cm.
  Green = fine, orange = tighter than the min turn radius (default 0.50 m; the car's minimum is about 0.49 m),
  red = closer to a wall or unknown space than the clearance limit (default 0.25 m).
- **Save** writes `maps/<name>_raceline.csv` (`x,y` per line, map frame, driving order) and
  `maps/<name>_waypoints.json` (so you can reload and edit it later).

## 3. Localize and race

```bash
race run track1                                  # speed 0.13..0.36, lookahead 0.45..1.45 m
race run track1 --max-speed 0.40 --full-speed-radius 2.14 --lookahead-max 1.62   # faster straights, same turns
race run track1 --max-speed 0.08 --min-speed 0.08 --lookahead 0.5   # old constant-speed behavior
```

1. In Foxglove (fixed frame `map`), show `/map`, `/scan`, `/pursuit/path`, `/pursuit/markers`.
2. Use the 2D pose estimate tool to publish the car's pose on `/initialpose`. Check that `/scan` lines up with the walls.
   If it doesn't, publish the pose again or nudge the car by hand until AMCL settles.
3. Flip **SWB down** to let it drive. SWB up or Ctrl-C stops it.

Until AMCL has a pose, the node sends zero speed. The red marker is the lookahead point and the blue one is the nearest point on the line.

### How speed and lookahead are chosen

`/drive` speed is normalized throttle: the throttle node clamps it to [-1, 1] and scales it by `max_speed_forward`
(1.0 in `~/ros2_ws/src/neoracer_ros2_driver/config/throttle.yaml`). So 0.36 is 36% throttle.

- **Speed ∝ 1 / curvature.** The node finds the sharpest bend in the next *lookahead + preview* meters of raceline
  (radius `R`) and commands `max_speed · R / full_speed_radius`, clamped to `[min_speed, max_speed]`.
  Bends looser than `full_speed_radius` run at full speed. Looking ahead means it brakes *before* the corner.
- **Rate limits.** The command rises at most `accel`/s and falls at most `decel`/s, so the throttle doesn't jump.
  It starts at `min_speed` when localization comes up.
- **Lookahead ∝ speed.** `L` goes linearly from `lookahead_min` at `min_speed` to `lookahead_max` at `max_speed`.
  Short lookahead in slow corners tracks tightly, and long lookahead on fast straights avoids wobble.

The node logs `v`, target, `L` and the radius ahead once a second.

### Tuning

| Flag | Default | Effect |
|---|---|---|
| `--max-speed` (`--speed`) | 0.36 | Throttle on straights. Raise in steps of ~0.02–0.04. To leave the turns unchanged, scale `--full-speed-radius` by the same factor (keep max_speed / radius ≈ 0.187) and set `--lookahead-max ≈ 0.45 + 4.33·(max_speed − 0.13)`. |
| `--min-speed` | 0.13 | Throttle in the tightest bends. Must be enough to keep the car rolling. |
| `--full-speed-radius` | 1.93 m | Bends at or above this radius get max speed. Raise it to slow down more in corners. |
| `--lookahead-min` / `--lookahead-max` | 0.45 / 1.45 m | Lookahead at min / max speed. Wobble on straights: raise max. Cuts corners: lower min. |
| `--lookahead` | – | Fixed lookahead (sets both). |
| `--preview` | 0.8 m | Extra distance past the lookahead scanned for bends. Raise it if the car brakes too late. |
| `--accel` / `--decel` | 0.25 / 0.60 per s | Throttle ramp limits. |
| `--wheelbase` | 0.28 m | Chassis constant. |
| `--max-steer-deg` | 30 | Steering is normalized by this before it goes on `/drive`. |

AMCL settings are in `amcl.yaml` (copied from `labs/tests/amcl_circle.yaml`).

## Files

| File | What |
|---|---|
| `race` | The CLI above. |
| `raceline_editor.py` + `editor.html` | Raceline GUI (stdlib HTTP server + Pillow; no ROS). |
| `pure_pursuit.py` | Generalized `labs/tests/circle_pursuit.py`: same steering law, raceline read from CSV. |
| `amcl.yaml` | AMCL parameters. |

Sync from the Mac: `scp -r physical-car/racing/* nvidia@<car-ip>:~/racing/`
