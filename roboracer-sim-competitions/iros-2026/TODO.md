# IROS 2026 TODOs

## NTU DeepSpeed

Features in their public [`race_stack_pp`](https://github.com/NTUDeepSpeed/race_stack_pp) that this stack does not have. Highest leverage for sim time-attack is at the top.

- [x] **Adaptive lookahead** — scale lookahead with speed (`min_lookahead` / `max_lookahead` / `lookahead_ratio`). Toggle in the playground; fixed slider is disabled when adaptive is on. Default remains fixed.
- [ ] **More aggressive raceline speeds** — their PP will command ~6 m/s (or CSV `V` × `velocity_percentage`). Our `track-20260731` line is 1.93–5.68 m/s (mean 3.36); `generate_waypoints.py` defaults `v_max` to 3.0.
- [ ] **Do not hard-gate control on AMCL** — their PP runs on odom. Ours holds zero throttle until `/amcl_pose` is fresh. Consider odom pose (or EKF odom) as fallback in sim.
- [ ] **Forward-half-plane lookahead** — if the chosen waypoint is behind the car, search forward along the raceline. We use Frenet `s`; still worth a behind-car guard.
- [ ] **C++ / odom-callback PP** — their controller is C++ on the odom callback (lower latency). Ours is Python on `/amcl_pose`.
- [ ] **EKF odom + IMU fusion** — `robot_localization` `ekf_node` at 50 Hz (`two_d_mode`), fusing wheel vx/yaw with IMU yaw + yaw rate. We publish IMU from the bridge and do not fuse it. Low value unless AutoDRIVE odom is bad at high speed. Not a replacement for AMCL.
- [ ] **Follow-the-Gap + disparity extender** — LiDAR reactive layer (`gap_finder.py`). Unused LiDAR today except AMCL/SLAM. More relevant for obstacles / H2H than empty-track time-attack.
- [ ] **Safety / AEB node** — dedicated `safety_node` that can override drive. We have none in `race.launch.py`.
- [ ] **Waypoint Studio** — in-practice raceline editor they credited at IFAC 2026. We have CMA-ES + GUIs, not a fast pit-loop editor.
