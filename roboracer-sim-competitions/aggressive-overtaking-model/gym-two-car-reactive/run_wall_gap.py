"""Drive a slower wall follower and a slightly faster follow-the-gap car."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from shapely.geometry import Polygon

from run_two_cars import GYM_MAPS, make_env, start_poses
from gap_follower import GapFollower
from wall_follower import WallFollower, WallParams
from direction_guard import ForwardRoute, CheckpointMonitor
from reactive_safety import ReactiveDriver

LABELS = {"agent_0": "Follow the gap", "agent_1": "Wall follower"}
COLORS = {"agent_0": "#7c3aed", "agent_1": "#e63946"}


def plot_result(map_name, traces, passes, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import yaml
    from PIL import Image
    directory = GYM_MAPS / map_name
    spec = yaml.safe_load((directory / f"{map_name}_map.yaml").read_text())
    image = np.asarray(Image.open(directory / spec["image"]))
    height, width = image.shape
    ox, oy, _ = spec["origin"]
    res = spec["resolution"]
    fig, ax = plt.subplots(figsize=(12, 6), dpi=140)
    ax.imshow(image, cmap="gray", origin="upper", extent=[ox, ox + width * res, oy, oy + height * res])
    for agent, points in traces.items():
        xy = np.asarray(points)
        if len(xy):
            ax.plot(*xy.T, color=COLORS[agent], lw=1.2, label=LABELS[agent])
            ax.scatter(*xy[0], color=COLORS[agent], marker="o", s=25)
    for event in passes:
        ax.scatter(*event["position"], marker="^", s=55, color="gold", edgecolors="black", zorder=5)
    ax.set_aspect("equal")
    ax.set_title(f"{map_name}: wall following and follow the gap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def race(seconds=90.0, fast_speed=3.6, slow_speed=2.0, gap=3., wall_distance=.7,
         render=False, map_name="autodrive-default", verbose=True, gap_margin=.15, seed=0):
    if seconds < .01 or not 0 < slow_speed < fast_speed or gap < 1:
        raise ValueError("Use positive speeds with the gap car faster, seconds >= .01, and a starting gap >= 1 m.")
    if wall_distance <= .155 or gap_margin < 0:
        raise ValueError("Wall distance must exceed the car half-width; gap margin must be nonnegative.")
    env = make_env(map_name, render, seed=seed)
    u = env.unwrapped
    angles = u.sim.agents[0].scan_angles
    controllers = {"agent_0": GapFollower(angles, v_max=fast_speed, safety_margin=gap_margin),
                   "agent_1": WallFollower(angles, WallParams(v_max=slow_speed, desired_dist=wall_distance))}
    # These drivers accept only LiDAR ranges and measured speed.
    controllers = {
        "agent_0": ReactiveDriver(controllers["agent_0"], angles, fast_speed,
                                  steering_rate=3., acceleration=2.,
                                  lateral_acceleration=2.5, stop_distance=.45, corridor_margin=.04),
        "agent_1": ReactiveDriver(controllers["agent_1"], angles, slow_speed, corridor_margin=.04),
    }
    obs, _ = env.reset(seed=seed, options={"poses": start_poses(u.track, gap)})
    # Static geometry and own pose are used below for reporting only.
    routes = {a: ForwardRoute(u.track) for a in obs}
    monitors = {}
    if map_name == "autodrive-default":
        from import_unity_track import EXPORT
        gates = json.loads((EXPORT / "source.json").read_text())
        monitors = {a: CheckpointMonitor(gates["checkpoint_positions"], gates["checkpoint_headings"],
                                        np.array([o["pose_x"], o["pose_y"]], float)) for a, o in obs.items()}
    traces = {a: [] for a in obs}
    travelled = {a: 0.0 for a in obs}
    last_xy = {a: np.array([o["pose_x"], o["pose_y"]], float) for a, o in obs.items()}
    speed_log = {a: [] for a in obs}
    from scipy.ndimage import distance_transform_edt
    wall_distance_field = distance_transform_edt(u.track.occupancy_map > 128) * u.track.spec.resolution
    wall_clearance = {a: float("inf") for a in obs}
    reverse_motion = Counter()
    stalled = Counter()
    stop_streak, longest_stop = Counter(), Counter()
    collisions, passes = [], []
    contacts = []
    touching_before = False
    minimum_gap = float("inf")
    seen_behind = True
    t = 0.0
    wall_start = time.monotonic()
    if render:
        import pygame
        pygame.display.set_caption(f"Purple: gap {fast_speed:g} m/s | Red: wall {slow_speed:g} m/s")
        env.render()
    try:
        from f1tenth_gym.envs.collision_models import get_vertices
        for k in range(round(seconds / .01)):
            if render and pygame.event.get(pygame.QUIT):
                break
            actions = [controllers[a].plan(obs[a]["scan"], float(obs[a]["linear_vel_x"]))
                       for a in ("agent_0", "agent_1")]
            obs, *_ = env.step(np.asarray(actions))
            t = (k + 1) * .01
            if render:
                env.render()
            bodies = []
            for agent, o in obs.items():
                routes[agent].observe(o, t)
                point = [float(o["pose_x"]), float(o["pose_y"])]
                travelled[agent] += float(np.linalg.norm(np.asarray(point) - last_xy[agent]))
                last_xy[agent] = np.asarray(point)
                traces[agent].append(point)
                velocity = float(o["linear_vel_x"])
                if a_monitor := monitors.get(agent):
                    a_monitor.observe(o)
                if velocity < -.05:
                    reverse_motion[agent] += 1
                speed_log[agent].append(velocity)
                if t > 2 and abs(velocity) < .1:
                    stalled[agent] += 1
                    stop_streak[agent] += 1
                    longest_stop[agent] = max(longest_stop[agent], stop_streak[agent])
                else:
                    stop_streak[agent] = 0
                if o["collision"]:
                    collisions.append({"agent": agent, "time": round(t, 2)})
                pose = np.array([*point, float(o["pose_theta"])])
                vertices = get_vertices(pose, .58, .31)
                bodies.append(Polygon(vertices))
                perimeter = np.vstack([vertices + fraction * (np.roll(vertices, -1, axis=0) - vertices)
                                       for fraction in (0., .25, .5, .75)])
                cells = np.floor((perimeter - np.array(u.track.spec.origin[:2])) / u.track.spec.resolution).astype(int)
                inside = (cells[:, 0] >= 0) & (cells[:, 0] < wall_distance_field.shape[1]) & (cells[:, 1] >= 0) & (cells[:, 1] < wall_distance_field.shape[0])
                clearance = float(np.min(wall_distance_field[cells[:, 1], cells[:, 0]])) if inside.all() else 0.
                wall_clearance[agent] = min(wall_clearance[agent], clearance)
            body_gap = bodies[0].distance(bodies[1])
            minimum_gap = min(minimum_gap, body_gap)
            touching = body_gap < .001
            if touching and not touching_before:
                contacts.append(round(t, 2))
            touching_before = touching
            # Reporting only: these measurements are never passed to a controller.
            o0, o1 = obs["agent_0"], obs["agent_1"]
            displacement = np.asarray(traces["agent_0"][-1]) - traces["agent_1"][-1]
            heading = float(o1["pose_theta"])
            same_direction = np.cos(float(o0["pose_theta"]) - heading) > .5
            longitudinal = float(displacement @ [np.cos(heading), np.sin(heading)])
            both_moving = min(float(o0["linear_vel_x"]), float(o1["linear_vel_x"])) > .2
            nearby = np.linalg.norm(displacement) < 3
            if nearby and same_direction and longitudinal < -.6:
                seen_behind = True
            if seen_behind and not collisions and same_direction and longitudinal > .6 and both_moving and nearby:
                event = {"time": round(t, 2), "position": traces["agent_0"][-1],
                         "body_gap": round(float(body_gap), 3)}
                passes.append(event)
                seen_behind = False
                if verbose:
                    print(f"[t={t:.2f}s] Follow-the-gap car moves ahead; body gap {body_gap:.3f} m", flush=True)
            if collisions:
                if verbose:
                    print("Collision:", collisions, flush=True)
                break
    finally:
        env.close()
    result = {"map": map_name, "seconds": round(t, 2), "wall_seconds": round(time.monotonic() - wall_start, 2),
              "speed_caps": {"follow_the_gap": fast_speed, "wall_follower": slow_speed},
              "starting_gap": gap, "wall_distance": wall_distance,
              "gap_margin": gap_margin, "collisions": collisions, "body_contacts": contacts, "passes": passes,
              "direction": "forward LiDAR targets; no route steering",
              "control_inputs": ["scan", "linear_vel_x"], "racing_line_control": False,
              "controller_features": {"wall_corner_preview": True, "gap_side_proximity_braking": True},
              "preview_config": {a: {"horizon_metres": c.controller.preview.depth * c.controller.preview.segment_length,
                                     "grid_resolution": c.controller.preview.resolution,
                                     "body_circles": len(c.controller.preview.body_offsets),
                                     "body_radius": c.controller.preview.body_radius}
                                 for a, c in controllers.items()}, "seed": seed,
              "minimum_body_gap": None if not np.isfinite(minimum_gap) else round(float(minimum_gap), 6),
              "cars": {a: {"controller": LABELS[a], "distance": round(travelled[a], 2),
                            "mean_speed": round(float(np.mean(speed_log[a])), 3) if speed_log[a] else 0,
                            "peak_speed": round(float(max(speed_log[a])), 3) if speed_log[a] else 0,
                            "forward_progress": round(routes[a].progress, 2),
                            "lap_times": routes[a].lap_times,
                            "ordered_checkpoint_crossings": monitors[a].crossings if a in monitors else None,
                            "checkpoint_laps": monitors[a].laps if a in monitors else None,
                            "reverse_motion_seconds": round(reverse_motion[a] * .01, 3),
                            "wrong_way_seconds": round(routes[a].wrong_way_seconds, 3),
                            "max_heading_error_degrees": round(np.rad2deg(routes[a].max_heading_error), 2),
                            "max_steering_rate": round(controllers[a].max_steering_rate, 3),
                            "minimum_wall_clearance": round(wall_clearance[a], 3) if speed_log[a] else None,
                            "longest_stop_seconds": round(longest_stop[a] * .01, 2),
                            "stalled_seconds": round(stalled[a] * .01, 2)} for a in obs}}
    return result, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="autodrive-default")
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--fast-speed", type=float, default=3.6)
    parser.add_argument("--slow-speed", type=float, default=2.0)
    parser.add_argument("--gap", type=float, default=3.)
    parser.add_argument("--wall-distance", type=float, default=.7)
    parser.add_argument("--gap-margin", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--plot", type=Path, default=Path("wall_gap.png"))
    parser.add_argument("--metrics", type=Path, default=Path("wall_gap.json"))
    args = parser.parse_args()
    result, traces = race(args.seconds, args.fast_speed, args.slow_speed, args.gap,
                          args.wall_distance, args.render, args.map, gap_margin=args.gap_margin, seed=args.seed)
    args.metrics.write_text(json.dumps(result, indent=2) + "\n")
    if traces["agent_0"]:
        plot_result(args.map, traces, result["passes"], args.plot)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
