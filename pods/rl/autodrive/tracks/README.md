# Tracks

Coordinates are in the simulator's world frame, in metres. `s = 0` is near the spawn at (0.8, 3.16).

| File | Used by | Contents |
| --- | --- | --- |
| `icra/icra_centerline.csv` | Training (track `icra`) | Centre line with track widths: `x_m, y_m, w_tr_right_m, w_tr_left_m`. 58 m long |
| `icra/track3_minimum_curvature_world.csv` | Training with `--race-line` | Minimum-curvature race line: `x_m, y_m`. Points every 0.1 m, 65.4 m, closed loop, same direction as the track |
| `icra/icra_line.csv` | Not used | Older line from the club's gym, in a different frame |
| `icra/icra_map.png`, `icra/icra_map.yaml` | Reference | Occupancy map (0.05 m per pixel) and its origin |
| `icra/icra_view.png` | Reference | Picture of the track |
| `practice/…` | Track `practice` | The same set for the club's practice track (30.7 m). Neither compete build contains it |

## Adding a race line

The format is a CSV with a `# x_m,y_m` header line, in world coordinates. The points should run in
the driving direction and form a closed loop. Spacing of about 0.1 m works well.

The ICRA race line comes within 0.12 m of the wall on the far-left straight, and 41 of its points
are closer than 0.25 m.
