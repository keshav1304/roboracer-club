"""Experiment-only adapters for the existing F1TENTH simulator.

Environment truth is confined to scenario setup, rendering, and scoring. The
learned ego policy receives only its own observation and the original map.
Custom static blockers use a copied occupancy map: this gym version does not
consistently support different car body dimensions for individual agents.
"""
from copy import copy
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import Polygon

BASELINE = Path(__file__).resolve().parent.parent / "gym-two-car-reactive"
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

import run_two_cars as baseline
from f1tenth_gym.envs.collision_models import get_vertices

MAP_NAME = "autodrive-default"
DT = .01
CAR_LENGTH = .58
CAR_WIDTH = .31


def make_stage2_opponent(geometry, angles, speed_limit, observation, steering=0., model_path=None):
    """Run the saved stage-2 policy using only the opponent's own observations.

    The opponent has the same maximum speed as configured for the ego. Its
    stage-2 clearance margins stay conservative; no scripted yielding, speed
    reduction, or external ego position is supplied. Initialize command state
    from its physical state so a running spawn does not cause a false brake.
    """
    from moving_model import MovingModel
    opponent = MovingModel(geometry, angles, vmax=float(speed_limit), model_path=model_path)
    opponent.command_speed = max(0., float(observation["linear_vel_x"]))
    opponent.steering = float(steering)
    opponent.planner.wall_margin = .07
    opponent.planner.car_margin = .12
    opponent.planner.detour_clearance = .20
    return opponent


def make_env(render=False, seed=0):
    """Two standard cars on the same imported track; optionally RGB capture.

    Do not use simultaneous environments with different maps in one process:
    this simulator's LiDAR map is class-shared. Sequential episodes are safe.
    """
    if render == "rgb_array":
        import gymnasium as gym
        from import_unity_track import ensure_unity_track
        from import_iros_track import configure_iros_renderer
        ensure_unity_track()
        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config={"map": MAP_NAME, "num_agents": 2, "seed": int(seed),
                    "timestep": DT, "integrator": "rk4", "model": "st",
                    "control_input": ["speed", "steering_angle"],
                    "observation_config": {"type": "features", "features": baseline.FEATURES}},
            render_mode="rgb_array",
        )
        configure_iros_renderer(env)
    else:
        env = baseline.make_env(MAP_NAME, bool(render), seed=int(seed))
    env.unwrapped._overtaking_base_track = env.unwrapped.track
    env.unwrapped._overtaking_barrier = None
    return env


def _set_track(env, track):
    env.unwrapped.track = track
    env.unwrapped.sim.set_map(track)
    if env.unwrapped.renderer is not None:
        from import_iros_track import configure_iros_renderer
        configure_iros_renderer(env)


def restore_base_map(env):
    """Remove the temporary blocker without modifying the imported files."""
    u = env.unwrapped
    base = getattr(u, "_overtaking_base_track", u.track)
    u._overtaking_base_track = base
    # Also reset the class-shared LiDAR map after a different environment.
    _set_track(env, base)
    u._overtaking_barrier = None


def install_barrier(env, geometry, s, width):
    """Place a .58 m long, specified-width static rectangle at reference s.

    Its pixels are conservatively rounded outward by at most one map cell.
    The returned polygon is the nominal physical rectangle for exact scoring.
    The controller's geometry and base map remain unchanged, allowing LiDAR
    subtraction to identify this new obstacle.
    """
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Barrier width must be finite and positive.")
    u = env.unwrapped
    base = getattr(u, "_overtaking_base_track", u.track)
    u._overtaking_base_track = base
    track = copy(base)
    track.occupancy_map = base.occupancy_map.copy()
    xy, yaw, _ = geometry.sample(float(s))
    xy, yaw = np.asarray(xy, float), float(yaw)
    vertices = get_vertices(np.r_[xy, yaw], CAR_LENGTH, float(width))
    polygon = Polygon(vertices)

    resolution = float(base.spec.resolution)
    origin = np.asarray(base.spec.origin[:2], float)
    lower = np.floor((vertices.min(axis=0) - origin) / resolution).astype(int) - 1
    upper = np.ceil((vertices.max(axis=0) - origin) / resolution).astype(int) + 1
    height, image_width = track.occupancy_map.shape
    lower = np.maximum(lower, [0, 0])
    upper = np.minimum(upper, [image_width - 1, height - 1])
    if np.any(upper < lower):
        raise ValueError("Barrier lies outside the map.")
    col, row = np.meshgrid(np.arange(lower[0], upper[0] + 1),
                           np.arange(lower[1], upper[1] + 1))
    delta_x = origin[0] + (col + .5) * resolution - xy[0]
    delta_y = origin[1] + (row + .5) * resolution - xy[1]
    along = np.cos(yaw) * delta_x + np.sin(yaw) * delta_y
    across = -np.sin(yaw) * delta_x + np.cos(yaw) * delta_y
    radius = resolution / np.sqrt(2.)
    occupied = (np.abs(along) <= CAR_LENGTH / 2 + radius) & (np.abs(across) <= width / 2 + radius)
    track.occupancy_map[row[occupied], col[occupied]] = 0.
    _set_track(env, track)
    u._overtaking_barrier = polygon
    return polygon


def _pose(geometry, s, lateral=0.):
    xy, yaw, _ = geometry.sample(s, lateral)
    return np.r_[xy, float(yaw)]


def reset_scenario(env, geometry, s, gap, opponent_speed, ego_speed=0., lateral=0., blocked=False):
    """Set up a static obstacle, a moving lead car, or a full-width blocker.

    s/gap are metres along the reference; lateral is the lead car's offset.
    Returned metadata is scoring-only and must not be passed to the policy.
    Initial speeds are genuine simulated velocities, not merely speed caps.
    """
    numbers = np.asarray([s, gap, opponent_speed, ego_speed, lateral], float)
    if not np.all(np.isfinite(numbers)) or gap <= CAR_LENGTH + .05 or min(opponent_speed, ego_speed) < 0:
        raise ValueError("Use finite values, a gap greater than .63 m, and nonnegative speeds.")
    if blocked and opponent_speed != 0:
        raise ValueError("A full-width static blocker must have zero opponent speed.")
    restore_base_map(env)
    obstacle_s = float((s + gap) % geometry.length)
    poses = np.stack([_pose(geometry, s), _pose(geometry, obstacle_s, lateral)])
    barrier = None
    if blocked:
        left, right = geometry.widths(obstacle_s)
        width = 2 * max(float(left), float(right)) + .2
        barrier = install_barrier(env, geometry, obstacle_s, width)
        # The second simulated car is irrelevant to barrier episodes. Park it
        # at the safest distant reference point, not on an adjacent close lane.
        barrier_xy, _, _ = geometry.sample(obstacle_s)
        distance_ego = np.linalg.norm(geometry.xy - poses[0, :2], axis=1)
        distance_obstacle = np.linalg.norm(geometry.xy - barrier_xy, axis=1)
        parking_index = int(np.argmax(np.minimum(distance_ego, distance_obstacle)))
        poses[1] = _pose(geometry, float(geometry.s[parking_index]))
    for pose in poses:
        if float(np.min(geometry.body_clearance(pose[:2], pose[2]))) <= .01:
            raise ValueError("Requested initial pose does not have enough wall clearance.")
    if Polygon(get_vertices(poses[0], CAR_LENGTH, CAR_WIDTH)).distance(
            Polygon(get_vertices(poses[1], CAR_LENGTH, CAR_WIDTH))) < .05:
        raise ValueError("Initial car bodies are too close.")
    if barrier is not None and Polygon(get_vertices(poses[0], CAR_LENGTH, CAR_WIDTH)).distance(barrier) < .1:
        raise ValueError("Ego starts too close to the blocker.")

    obs, _ = env.reset(options={"poses": poses})
    speeds = [float(ego_speed), 0. if blocked else float(opponent_speed)]
    for agent, speed in zip(env.unwrapped.sim.agents, speeds):
        agent.state[3] = speed
    # Refresh scan and kinematic observations after changing initial velocity.
    # The setup step is excluded from the caller's timed/scored episode.
    obs, *_ = env.step(np.asarray([[0., v] for v in speeds]))
    meta = {"initial_poses": poses.tolist(), "initial_speeds": speeds,
            "obstacle_s": obstacle_s, "obstacle_lateral": float(lateral),
            "blocked": bool(blocked), "obstacle_polygon": barrier,
            "initial_gap": float(gap), "setup_seconds": 2 * DT,
            "blocker_raster_resolution": float(geometry.resolution) if blocked else None}
    return obs, meta


def body_polygon(observation, length=CAR_LENGTH, width=CAR_WIDTH):
    """Exact rectangular body for evaluation, not a policy input."""
    pose = np.asarray([observation[k] for k in ("pose_x", "pose_y", "pose_theta")], float)
    return Polygon(get_vertices(pose, float(length), float(width)))


def timing_summary(milliseconds):
    """Percentiles of actual measured calls; input units are milliseconds."""
    values = np.asarray(milliseconds, float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Timings must be a finite, nonnegative one-dimensional sequence.")
    if not len(values):
        return {"count": 0, "mean_ms": None, "p50_ms": None,
                "p95_ms": None, "p99_ms": None, "max_ms": None}
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return {"count": int(len(values)), "mean_ms": float(values.mean()),
            "p50_ms": float(p50), "p95_ms": float(p95),
            "p99_ms": float(p99), "max_ms": float(values.max())}
