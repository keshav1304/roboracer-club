FINAL STAGE 4 DEMONSTRATION — NOVEMBER 14, 2026

From the package root: bash run-demo.sh
This starts four HybridCorridorStage4 encounters in actual F1TENTH Gym, with
live ROS/RViz visualization. No commands are replayed and no API is called.
The independently reactive Stage 2 opponent is unchanged.

Selected passes: fixed-case-014 (18.44 s), fixed-case-007 (10.53 s),
random-case-001 (8.15 s), random-case-008 (8.34 s). These are highlights.
The full held-out result is 23/24 passes, zero contacts and one timeout.

White: reference. Green: ego and selected path. Orange: opponent.
Blue: feasible candidates. Red: rejected candidates. Cyan: opponent forecasts.
Amber: checked recovery. Labels show wide approach / guarded close encounter,
steering-at-rest recovery, rejected speeds and wall-bounded lateral goals.
A goal offset is not the opponent's measured displacement or evidence of yield.

The final source hashes and four verified live outcomes are retained under
artifacts/hybrid_corridor_demo_20261114. Frozen controller/shared dependency
checks prevent silently displaying a different controller than the test.
The portable launcher finds Pixi on PATH; its display dependencies are locked
for Apple Silicon macOS. A fresh ROS installation was not repeated for this update.

One Gym owns physics and sensors; snapshots are display-only. ROS publishes
map/walls, clock, poses, LiDAR, reference, candidates, selected path and status.
RViz uses wall-boundary cubes from the Gym map because the occupancy-grid
texture plugin had a macOS shader issue. No alternate environment is used.
ROS discovery is local on domain 47. Visualization overhead is not a speed test.

See the package README and final REPORT.txt for methods, full results and limits.
