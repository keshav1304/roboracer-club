#!/usr/bin/env python3
"""
RoboRacer Playground Gateway.

FastAPI WebSocket service that sits next to the AutoDRIVE ROS 2 bridge and
exposes a browser-friendly, real-time JSON API:

  Client -> server : set_params | start_algo | stop_algo | reset | ping
                     | slam_start | slam_stop | slam_save_map
                     | map_edit | map_undo | map_redo | map_delete | map_rename
                     | raceline_update | raceline_save | raceline_delete
                     | opt_start | opt_cancel
  Server -> client : status | telemetry | lidar | map_frame | map_updated
                     | opt_progress | raceline_data | *_ack | error | pong

The control loops (wall follow, follow gap, pure pursuit, MPC) stay in ROS; the
browser tunes parameters, drives the Mapping workflow (SLAM mapping →
raceline optimization/editing → AMCL + raceline tracking) and visualizes
state.

Auth: a shared token via ?token=... query param (PLAYGROUND_TOKEN env var).
"""

import asyncio
import base64
import json
import math
import os
import signal
import subprocess
import threading
import time
from typing import Dict, List, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import uvicorn

import raceline as raceline_engine
import localize as localize_module
import slam as slam_module
import store

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8000"))
PLAYGROUND_TOKEN = os.environ.get("PLAYGROUND_TOKEN", "")  # empty = no auth (dev)
TELEMETRY_HZ = float(os.environ.get("TELEMETRY_HZ", "20"))
LIDAR_HZ = float(os.environ.get("LIDAR_HZ", "10"))
LIDAR_STRIDE = int(os.environ.get("LIDAR_STRIDE", "4"))  # 1080 -> 270 beams
SIM_TIMEOUT_S = 1.5  # data older than this => sim considered disconnected

# Defaults applied when mapping starts with wall_follow and the UI
# did not send an explicit param set. Prefer slow, stable mapping.
MAPPING_PRESET = {"throttle": 0.03, "desired_dist": 0.8, "lookahead": 0.5}


def _jsonable(obj):
    """Coerce ROS/numpy values into JSON (no NaN/Inf — browsers reject those)."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            jv = _jsonable(v)
            # Drop non-finite numbers rather than emitting null (the UI
            # treats Number(null) as 0, which would draw overlays at origin).
            if jv is None and v is not None and not isinstance(v, (dict, list, tuple)):
                continue
            out[str(k)] = jv
        return out
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    return obj


def dumps_ws(obj) -> str:
    return json.dumps(_jsonable(obj), allow_nan=False)

# Named layouts for /playground/<algo>/debug Float32MultiArray topics.
_RRT_DEBUG_KEYS = (
    "plan_ms", "tree_size", "path_len", "reached_goal", "steering",
    "throttle", "lookahead_dist", "goal_x", "target_x", "target_y",
    "goal_tolerance", "grid_resolution",
)
_RRT_PARAMS = {
    "max_rrt_iters", "max_planning_ms", "plan_hz", "expand_dist",
    "goal_x", "goal_y", "goal_tolerance", "goal_sample_rate",
    "inflate_radius", "near_radius", "lookahead_dist", "throttle",
    "max_steering_rad", "scan_angle_min_deg", "scan_angle_max_deg",
}
_MPC_PARAMS = {
    "throttle_gain", "corner_throttle_gain", "corner_steering_threshold",
    "kp_speed", "ki_speed", "integral_limit", "max_steering_rad",
    "velocity_scale", "steering_direction", "path_direction",
    "max_speed", "q_pos", "q_yaw", "q_vel", "r_accel", "r_steer",
}

# Map-frame raceline trackers (AMCL + waypoint_file bringup).
MAP_CONTROLLER_ALGOS = frozenset({"pure_pursuit", "mpc"})

DEBUG_LAYOUTS = {
    "wall_follow": (
        "ray_a", "ray_b", "alpha", "car_dist", "car_dist_future", "error",
        "p_term", "i_term", "d_term", "desired_dist", "theta_rad", "lookahead",
    ),
    "follow_gap": (
        "best_point_angle", "best_point_range", "closest_range",
        "gap_start_angle", "gap_end_angle", "fov_deg",
    ),
    "pure_pursuit": (
        "s_curr", "target_x", "target_y", "y_local", "steering_norm",
        "target_velocity", "speed", "speed_error", "lookahead", "throttle",
    ),
    "mpc": (
        "s_curr", "target_x", "target_y", "steering_norm",
        "target_velocity", "speed", "speed_error", "throttle", "solve_ms",
    ),
    "rrt": _RRT_DEBUG_KEYS,
    "rrt_star": _RRT_DEBUG_KEYS,
}

ALGORITHMS = {
    "wall_follow": {
        "cmd": ["ros2", "run", "autodrive_roboracer", "wall_follow"],
        "node": "/wall_follow_node",
        "params": {
            "kp", "kd", "ki", "desired_dist", "lookahead",
            "throttle", "max_steering_rad", "theta_deg",
        },
    },
    "follow_gap": {
        "cmd": ["ros2", "run", "autodrive_roboracer", "follow_gap"],
        "node": "/follow_gap_node",
        "params": {
            "vehicle_half_width", "disparity_threshold", "smoothing_window",
            "free_space_threshold", "best_point_threshold", "max_steering_rad",
            "fov_deg", "heading_weight", "throttle",
        },
    },
    "pure_pursuit": {
        "cmd": ["ros2", "run", "autodrive_roboracer", "pure_pursuit_node"],
        "node": "/pure_pursuit_node",
        "params": {
            "lookahead", "throttle_gain", "corner_throttle_gain",
            "corner_steering_threshold", "kp_speed", "ki_speed",
            "integral_limit", "max_steering_rad",
            "velocity_scale", "steering_direction", "path_direction",
        },
    },
    "mpc": {
        "cmd": ["ros2", "run", "autodrive_roboracer", "mpc_node"],
        "node": "/mpc_node",
        "params": set(_MPC_PARAMS),
    },
    "rrt": {
        "cmd": ["ros2", "run", "autodrive_roboracer", "rrt"],
        "node": "/rrt_node",
        "params": set(_RRT_PARAMS),
    },
    "rrt_star": {
        "cmd": ["ros2", "run", "autodrive_roboracer", "rrt_star"],
        "node": "/rrt_star_node",
        "params": set(_RRT_PARAMS),
    },
}

# ---------------------------------------------------------------------------
# ROS node (runs in a background thread)
# ---------------------------------------------------------------------------


class GatewayNode(Node):
    """Caches the latest state from AutoDRIVE topics for WS streaming."""

    def __init__(self):
        super().__init__("playground_gateway")
        qos = QoSProfile(
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.lock = threading.Lock()
        self.state: Dict = self._empty_state()
        self.algo_debug: Optional[Dict] = None
        self.lidar: Optional[Dict] = None
        self.last_data_time: float = 0.0
        self._sim_was_connected = False
        self.session_epoch: int = 0

        # Live SLAM map (PNG-encoded, seq-stamped for WS fan-out).
        self.map_frame: Optional[Dict] = None
        self.map_seq: int = 0
        self._last_map_encode: float = 0.0

        ns = "/autodrive/roboracer_1"
        self.create_subscription(LaserScan, f"{ns}/lidar", self._on_lidar, qos)
        self.create_subscription(Odometry, f"{ns}/odom", self._on_odom, qos)
        self.create_subscription(Float32, f"{ns}/throttle", self._on_throttle, qos)
        self.create_subscription(Float32, f"{ns}/steering", self._on_steering, qos)
        self.create_subscription(Int32, f"{ns}/lap_count", self._on_lap_count, qos)
        self.create_subscription(Float32, f"{ns}/lap_time", self._on_lap_time, qos)
        self.create_subscription(Float32, f"{ns}/best_lap_time", self._on_best_lap, qos)
        self.create_subscription(Int32, f"{ns}/collision_count", self._on_collisions, qos)
        self.create_subscription(Point, f"{ns}/ips", self._on_ips, qos)
        self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._on_amcl_pose, qos)
        for algo_name in DEBUG_LAYOUTS:
            self.create_subscription(
                Float32MultiArray, f"/playground/{algo_name}/debug",
                (lambda name: lambda msg: self._on_algo_debug(name, msg))(algo_name),
                qos)

        # slam_toolbox / map_server publish /map latched (transient local).
        map_qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(OccupancyGrid, "/map", self._on_map, map_qos)

        # TF listener for the map -> world correction.
        try:
            from tf2_ros import Buffer, TransformListener
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        except ImportError:
            self.tf_buffer = None
            self.tf_listener = None

        self.reset_pub = self.create_publisher(Bool, "/autodrive/reset_command", qos)
        self.throttle_cmd_pub = self.create_publisher(
            Float32, f"{ns}/throttle_command", qos)
        self.steering_cmd_pub = self.create_publisher(
            Float32, f"{ns}/steering_command", qos)
        self._param_clients: Dict[str, object] = {}

        # Latest AMCL estimate (for Mapping status; PP uses the topic directly).
        self._amcl_pose: Optional[Dict] = None
        self._amcl_pose_time: float = 0.0

    @staticmethod
    def _empty_state() -> Dict:
        return {
            "position": [0.0, 0.0, 0.0],
            "yaw": 0.0,
            "speed": 0.0,
            "throttle": 0.0,
            "steering": 0.0,
            "lap_count": 0,
            "lap_time": 0.0,
            "best_lap_time": 0.0,
            "collision_count": 0,
        }

    def _param_client(self, node_name: str):
        if node_name not in self._param_clients:
            self._param_clients[node_name] = self.create_client(
                SetParameters, f"{node_name}/set_parameters")
        return self._param_clients[node_name]

    # --- callbacks -------------------------------------------------------

    def _touch(self):
        self.last_data_time = time.monotonic()

    def _on_lidar(self, msg: LaserScan):
        ranges = list(msg.ranges[::LIDAR_STRIDE])
        cleaned = [
            (r if math.isfinite(r) else 0.0) for r in ranges
        ]
        with self.lock:
            self.lidar = {
                "angle_min": msg.angle_min,
                "angle_increment": msg.angle_increment * LIDAR_STRIDE,
                "range_max": msg.range_max,
                "ranges": [round(r, 3) for r in cleaned],
            }
        self._touch()

    def _on_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        v = msg.twist.twist.linear
        with self.lock:
            # Odom is authoritative for UI / reporting (world frame).
            # Pure Pursuit control uses /amcl_pose (map frame) instead.
            self.state["position"] = [
                round(msg.pose.pose.position.x, 3),
                round(msg.pose.pose.position.y, 3),
                round(msg.pose.pose.position.z, 3),
            ]
            self.state["yaw"] = round(yaw, 4)
            self.state["speed"] = round(math.sqrt(v.x ** 2 + v.y ** 2), 3)
        self._touch()

    def _on_ips(self, msg: Point):
        # Keep IPS available for debugging but do not drive the UI pose.
        with self.lock:
            self.state["ips"] = [round(msg.x, 3), round(msg.y, 3), round(msg.z, 3)]
        self._touch()

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        cov = msg.pose.covariance
        with self.lock:
            self._amcl_pose = {
                "x": round(msg.pose.pose.position.x, 3),
                "y": round(msg.pose.pose.position.y, 3),
                "yaw": round(yaw, 4),
                "var_x": float(cov[0]),
                "var_y": float(cov[7]),
                "var_yaw": float(cov[35]),
            }
            self._amcl_pose_time = time.monotonic()

    def amcl_pose_fresh(self, max_age: float = 2.5) -> bool:
        with self.lock:
            if self._amcl_pose is None:
                return False
            return (time.monotonic() - self._amcl_pose_time) <= max_age

    def localization_status(self) -> Dict:
        """Mapping status: whether AMCL is producing a convergent pose."""
        with self.lock:
            pose = dict(self._amcl_pose) if self._amcl_pose else None
            age = (time.monotonic() - self._amcl_pose_time
                   if self._amcl_pose is not None else None)
        if pose is None or age is None or age > 2.5:
            return {"pose_fresh": False, "localized": False, "pose": None}
        localized = (pose["var_x"] < 0.25 and pose["var_y"] < 0.25
                     and pose["var_yaw"] < 0.07)
        return {
            "pose_fresh": True,
            "localized": localized,
            "pose": pose,
            "age_s": round(age, 3),
        }

    def _on_throttle(self, msg: Float32):
        with self.lock:
            self.state["throttle"] = round(msg.data, 3)

    def _on_steering(self, msg: Float32):
        with self.lock:
            self.state["steering"] = round(msg.data, 3)

    def _on_lap_count(self, msg: Int32):
        with self.lock:
            self.state["lap_count"] = msg.data

    def _on_lap_time(self, msg: Float32):
        with self.lock:
            self.state["lap_time"] = round(msg.data, 2)

    def _on_best_lap(self, msg: Float32):
        with self.lock:
            self.state["best_lap_time"] = round(msg.data, 2)

    def _on_collisions(self, msg: Int32):
        with self.lock:
            self.state["collision_count"] = msg.data

    def _on_algo_debug(self, algorithm: str, msg: Float32MultiArray):
        keys = DEBUG_LAYOUTS.get(algorithm)
        if not keys:
            return
        values = list(msg.data)
        debug = {"algorithm": algorithm}
        for i, key in enumerate(keys):
            if i < len(values):
                v = float(values[i])
                # inf/NaN would serialize as invalid JSON and make the
                # browser drop the whole telemetry frame — omit instead.
                if math.isfinite(v):
                    debug[key] = round(v, 4)
        # Trailing polyline after scalars: MPC horizon, or RRT path (+ tree/occ).
        if algorithm in ("mpc", "rrt", "rrt_star") and len(values) > len(keys):
            i = len(keys)
            try:
                n_path = int(values[i])
                i += 1
                path = []
                for _ in range(max(0, n_path)):
                    if i + 1 >= len(values):
                        break
                    x, y = float(values[i]), float(values[i + 1])
                    i += 2
                    if math.isfinite(x) and math.isfinite(y):
                        path.append([round(x, 3), round(y, 3)])
                debug["path"] = path
                if algorithm in ("rrt", "rrt_star"):
                    if i < len(values):
                        n_edges = int(values[i])
                        i += 1
                        tree = []
                        for _ in range(max(0, n_edges)):
                            if i + 4 >= len(values):
                                break
                            seg = [float(values[j]) for j in range(i, i + 5)]
                            i += 5
                            if all(math.isfinite(v) for v in seg):
                                tree.append([round(v, 3) for v in seg])
                        debug["tree"] = tree
                    if i < len(values):
                        n_hits = int(values[i])
                        i += 1
                        hits = []
                        for _ in range(max(0, n_hits)):
                            if i + 1 >= len(values):
                                break
                            x, y = float(values[i]), float(values[i + 1])
                            i += 2
                            if math.isfinite(x) and math.isfinite(y):
                                hits.append([round(x, 3), round(y, 3)])
                        debug["occ_hits"] = hits
                    if i < len(values):
                        n_inf = int(values[i])
                        i += 1
                        inflated = []
                        for _ in range(max(0, n_inf)):
                            if i + 1 >= len(values):
                                break
                            x, y = float(values[i]), float(values[i + 1])
                            i += 2
                            if math.isfinite(x) and math.isfinite(y):
                                inflated.append([round(x, 3), round(y, 3)])
                        debug["occ_inflated"] = inflated
            except (TypeError, ValueError, IndexError):
                pass
        with self.lock:
            self.algo_debug = debug

    def _on_map(self, msg: OccupancyGrid):
        """Stream the live SLAM grid as raw Int8 occupancy (throttled ~1 Hz).

        Values follow ROS OccupancyGrid: -1 unknown, 0 free, 1–100 occupied.
        Rows are flipped so row 0 is the top of the image (max world-y),
        matching canvas ImageData / typical image conventions.
        """
        now = time.monotonic()
        if now - self._last_map_encode < 0.8:
            return
        self._last_map_encode = now

        w = msg.info.width
        h = msg.info.height
        if w == 0 or h == 0:
            return
        # OccupancyGrid is int8-valued; clip anything outside int8 just in case.
        data = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
        data = np.clip(data, -1, 100).astype(np.int8)
        data = np.flipud(data)

        frame = {
            "occ_b64": base64.b64encode(data.tobytes()).decode("ascii"),
            "width": int(w),
            "height": int(h),
            "resolution": float(msg.info.resolution),
            "origin": [
                float(msg.info.origin.position.x),
                float(msg.info.origin.position.y),
            ],
            "map_to_world": self.map_to_world_transform(),
        }
        with self.lock:
            self.map_frame = frame
            self.map_seq += 1

    def map_to_world_transform(self) -> Optional[Dict]:
        """Return map → world as {tx, ty, yaw, frame}, or None if unavailable.

        Used to place world-frame odom on a map-frame canvas:
        p_map = R^T * (p_world - t).
        """
        if self.tf_buffer is None:
            return None
        try:
            from rclpy.time import Time
            # Transform that takes a point in `map` into `world`.
            t = self.tf_buffer.lookup_transform("world", "map", Time())
            d = slam_module.transform_to_dict(t.transform)
            d["frame"] = "map_to_world"
            return d
        except Exception:
            return None

    def map_snapshot(self):
        with self.lock:
            return self.map_seq, (dict(self.map_frame) if self.map_frame else None)

    def clear_map_frame(self):
        with self.lock:
            self.map_frame = None
            self.map_seq += 1

    def clear_algo_debug(self):
        with self.lock:
            self.algo_debug = None

    def clear_amcl_pose(self):
        """Drop cached particle-filter pose so reseeding cannot see a stale hit."""
        with self.lock:
            self._amcl_pose = None
            self._amcl_pose_time = 0.0

    def clear_session_state(self):
        """Drop cached sim/algo state so the UI does not keep stale frames."""
        with self.lock:
            self.state = self._empty_state()
            self.lidar = None
            self.algo_debug = None
            self._amcl_pose = None
            self._amcl_pose_time = 0.0
            self.session_epoch += 1

    def get_session_epoch(self) -> int:
        with self.lock:
            return self.session_epoch

    # --- commands ---------------------------------------------------------

    def sim_connected(self) -> bool:
        return (time.monotonic() - self.last_data_time) < SIM_TIMEOUT_S

    def poll_sim_disconnect(self) -> bool:
        """Return True once when the simulator transitions connected → disconnected."""
        connected = self.sim_connected()
        edged = self._sim_was_connected and not connected
        self._sim_was_connected = connected
        return edged

    def snapshot(self) -> Dict:
        with self.lock:
            out = dict(self.state)
            out["algo_debug"] = dict(self.algo_debug) if self.algo_debug else None
            return out

    def compute_pose_error(self, map_name: Optional[str]) -> Optional[Dict]:
        """True odom (world→map) vs AMCL pose — for Mapping charting."""
        if not map_name or not self.amcl_pose_fresh(max_age=1.0):
            return None
        with self.lock:
            amcl = dict(self._amcl_pose) if self._amcl_pose else None
            pos = list(self.state.get("position") or [0.0, 0.0, 0.0])
            yaw_w = float(self.state.get("yaw", 0.0))
        if amcl is None:
            return None
        m2w = None
        try:
            m2w = store.map_meta(map_name).get("map_to_world")
        except store.StoreError:
            pass
        mx, my, myaw = localize_module.world_to_map(
            float(pos[0]), float(pos[1]), yaw_w, m2w)
        dx = mx - float(amcl["x"])
        dy = my - float(amcl["y"])
        xy = math.hypot(dx, dy)
        dyaw = math.atan2(math.sin(myaw - float(amcl["yaw"])),
                          math.cos(myaw - float(amcl["yaw"])))
        return {
            "xy_m": round(xy, 4),
            "yaw_rad": round(dyaw, 4),
            "yaw_deg": round(math.degrees(dyaw), 2),
            "true_map": {
                "x": round(mx, 3), "y": round(my, 3), "yaw": round(myaw, 4),
            },
            "amcl": {
                "x": float(amcl["x"]), "y": float(amcl["y"]),
                "yaw": float(amcl["yaw"]),
            },
        }

    def lidar_snapshot(self) -> Optional[Dict]:
        with self.lock:
            return dict(self.lidar) if self.lidar else None

    def lap_count(self) -> int:
        with self.lock:
            return int(self.state.get("lap_count", 0))

    def clear_drive_commands(self):
        """Publish zero throttle/steer so the sim has no residual drive cmds."""
        self.throttle_cmd_pub.publish(Float32(data=0.0))
        self.steering_cmd_pub.publish(Float32(data=0.0))
        with self.lock:
            self.state["throttle"] = 0.0
            self.state["steering"] = 0.0

    def publish_reset(self):
        """Zero drive commands, then pulse reset (True → False)."""
        self.clear_drive_commands()

        msg = Bool()
        msg.data = True
        self.reset_pub.publish(msg)

        def finish():
            # Keep commanding zeros while the reset pulse settles so a late
            # algo message (or latched bridge state) cannot leave residual
            # throttle/steer on the car.
            for _ in range(12):
                self.clear_drive_commands()
                time.sleep(0.05)
            off = Bool()
            off.data = False
            self.reset_pub.publish(off)
            self.clear_drive_commands()

        threading.Thread(target=finish, daemon=True).start()

    def set_algo_params(self, algorithm: str, params: Dict[str, float],
                        timeout: float = 2.0):
        """Set parameters on an algorithm node via its SetParameters service."""
        spec = ALGORITHMS.get(algorithm)
        if spec is None:
            return {"ok": False, "reason": f"Unknown algorithm: {algorithm}"}

        client = self._param_client(spec["node"])
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=0.5):
                return {"ok": False, "reason": f"{algorithm} node is not running"}

        req = SetParameters.Request()
        for name, value in params.items():
            if name not in spec["params"]:
                return {"ok": False, "reason": f"Unknown parameter: {name}"}
            p = Parameter()
            p.name = name
            pv = ParameterValue()
            pv.type = ParameterType.PARAMETER_DOUBLE
            pv.double_value = float(value)
            p.value = pv
            req.parameters.append(p)

        future = client.call_async(req)
        deadline = time.monotonic() + timeout
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            return {"ok": False, "reason": "Parameter update timed out"}

        results = future.result().results
        failed = [
            f"{name}: {r.reason}"
            for name, r in zip(params.keys(), results)
            if not r.successful
        ]
        if failed:
            return {"ok": False, "reason": "; ".join(failed)}
        return {"ok": True, "applied": params}


# ---------------------------------------------------------------------------
# Algorithm process manager
# ---------------------------------------------------------------------------


class AlgoManager:
    """Starts/stops racing algorithm nodes as subprocesses."""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._name: Optional[str] = None
        self._raceline: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def running(self) -> Optional[str]:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return self._name
            return None

    @property
    def raceline(self) -> Optional[str]:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return self._raceline
            return None

    def start(self, name: str, extra_ros_params: Optional[Dict] = None,
              raceline: Optional[str] = None) -> Dict:
        if name not in ALGORITHMS:
            return {"ok": False, "reason": f"Unknown algorithm: {name}"}
        cmd = list(ALGORITHMS[name]["cmd"])
        if extra_ros_params:
            cmd.append("--ros-args")
            for k, v in extra_ros_params.items():
                cmd += ["-p", f"{k}:={v}"]
        with self._lock:
            self._stop_locked()
            if ros_node is not None:
                ros_node.clear_algo_debug()
                ros_node.clear_drive_commands()
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid,
                )
                self._name = name
                self._raceline = raceline
            except OSError as e:
                return {"ok": False, "reason": str(e)}
        return {"ok": True, "algorithm": name}

    def stop(self) -> Dict:
        with self._lock:
            self._stop_locked()
        if ros_node is not None:
            ros_node.clear_algo_debug()
            # Algo nodes stop publishing; clear any latched drive commands.
            ros_node.clear_drive_commands()
        return {"ok": True}

    def _stop_locked(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)
                self._proc.wait(timeout=3)
            except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError):
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        self._proc = None
        self._name = None
        self._raceline = None


# ---------------------------------------------------------------------------
# Broadcast helper (map edits etc. fan out to every WS client)
# ---------------------------------------------------------------------------


class Broadcast:
    """Seq-stamped event ring polled by each connection's sender loop."""

    def __init__(self):
        self._lock = threading.Lock()
        self.seq = 0
        self.events: List[Dict] = []

    def push(self, event: Dict):
        with self._lock:
            self.seq += 1
            event = dict(event)
            event["seq"] = self.seq
            self.events.append(event)
            if len(self.events) > 50:
                del self.events[:-50]

    def since(self, seq: int) -> List[Dict]:
        with self._lock:
            return [e for e in self.events if e["seq"] > seq]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="RoboRacer Playground Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ros_node: Optional[GatewayNode] = None
algo = AlgoManager()
slam_mgr = slam_module.SlamManager()
localize_mgr = localize_module.LocalizeManager()
opt_mgr = raceline_engine.OptManager()
broadcast = Broadcast()


def _authorized(token: Optional[str]) -> bool:
    return not PLAYGROUND_TOKEN or token == PLAYGROUND_TOKEN


@app.get("/health")
def health():
    return {
        "ok": True,
        "sim_connected": ros_node.sim_connected() if ros_node else False,
        "algorithm": algo.running,
        "slam_active": slam_mgr.active,
        "localize_active": localize_mgr.active,
        "opt_running": opt_mgr.running,
    }


# --- REST: artifact browsing (maps + racelines) ----------------------------


@app.get("/maps")
def rest_list_maps(token: Optional[str] = None):
    if not _authorized(token):
        return JSONResponse({"error": "invalid token"}, status_code=401)
    return {"maps": store.list_maps()}


@app.get("/maps/{name}/image.png")
def rest_map_image(name: str, token: Optional[str] = None, v: Optional[str] = None):
    if not _authorized(token):
        return JSONResponse({"error": "invalid token"}, status_code=401)
    try:
        png = store.map_image_png(name)
    except store.StoreError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/maps/{name}/grid")
def rest_map_grid(name: str, token: Optional[str] = None, v: Optional[str] = None):
    """Int8 occupancy grid for canvas rendering (same format as live map_frame)."""
    if not _authorized(token):
        return JSONResponse({"error": "invalid token"}, status_code=401)
    try:
        return store.map_occupancy(name)
    except store.StoreError as e:
        return JSONResponse({"error": str(e)}, status_code=404)


@app.get("/racelines")
def rest_list_racelines(token: Optional[str] = None):
    if not _authorized(token):
        return JSONResponse({"error": "invalid token"}, status_code=401)
    return {"racelines": store.list_racelines()}


@app.get("/racelines/{name}")
def rest_raceline(name: str, token: Optional[str] = None):
    if not _authorized(token):
        return JSONResponse({"error": "invalid token"}, status_code=401)
    try:
        meta = store.raceline_meta(name)
        rows = store.load_raceline(name)
    except store.StoreError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    return {
        "meta": meta,
        "data": {
            "s": rows[:, 0].round(4).tolist(),
            "x": rows[:, 1].round(4).tolist(),
            "y": rows[:, 2].round(4).tolist(),
            "theta": rows[:, 3].round(4).tolist(),
            "v": rows[:, 4].round(3).tolist(),
        },
    }


# --- WS command handlers ----------------------------------------------------


def _handle_slam_start(msg: Dict) -> Dict:
    """Blocking; run in executor."""
    drive = msg.get("drive", "manual")
    if drive not in ("manual", "wall_follow"):
        return {"ok": False, "reason": f"Unknown drive mode: {drive}"}
    if algo.running in MAP_CONTROLLER_ALGOS:
        algo.stop()
    if localize_mgr.active:
        localize_mgr.stop()

    result = slam_mgr.start(
        drive=drive,
        auto_stop=False,
        start_lap=ros_node.lap_count(),
    )
    if not result["ok"]:
        return result

    if drive == "wall_follow":
        started = algo.start("wall_follow")
        if not started["ok"]:
            slam_mgr.stop()
            return {"ok": False,
                    "reason": f"wall_follow failed: {started['reason']}"}
        preset = dict(MAPPING_PRESET)
        for k, v in (msg.get("params") or {}).items():
            if k in ALGORITHMS["wall_follow"]["params"]:
                preset[k] = float(v)

        def apply_preset():
            deadline = time.monotonic() + 6.0
            while time.monotonic() < deadline:
                r = ros_node.set_algo_params("wall_follow", preset)
                if r.get("ok"):
                    return
                time.sleep(0.5)

        threading.Thread(target=apply_preset, daemon=True).start()
    return {"ok": True, "drive": drive}


def _handle_slam_stop() -> Dict:
    if slam_mgr.drive == "wall_follow":
        algo.stop()
    result = slam_mgr.stop()
    if ros_node is not None:
        ros_node.clear_map_frame()
    return result


def _handle_slam_save(msg: Dict) -> Dict:
    """Blocking; run in executor."""
    name = store.slugify(str(msg.get("name", "")))
    if not name:
        return {"ok": False, "reason": "Map name required"}
    result = slam_mgr.save_map(ros_node, name)
    if result.get("ok"):
        broadcast.push({"type": "maps_changed"})
    return result


def _handle_map_edit(msg: Dict) -> Dict:
    try:
        meta = store.apply_map_edit(str(msg.get("name", "")),
                                    msg.get("ops") or [])
    except store.StoreError as e:
        return {"ok": False, "reason": str(e)}
    hist = store.map_history_counts(meta["name"])
    broadcast.push({"type": "map_updated", "name": meta["name"],
                    "version": meta["version"], **hist})
    return {"ok": True, "version": meta["version"], **hist}


def _handle_map_undo_redo(msg: Dict, redo: bool) -> Dict:
    try:
        name = str(msg.get("name", ""))
        meta = (store.redo_map_edit(name) if redo
                else store.undo_map_edit(name))
    except store.StoreError as e:
        return {"ok": False, "reason": str(e)}
    hist = store.map_history_counts(name)
    broadcast.push({"type": "map_updated", "name": name,
                    "version": meta["version"], **hist})
    return {"ok": True, "version": meta["version"], **hist}


def _handle_raceline_update(msg: Dict) -> Dict:
    """Blocking (scipy); run in executor."""
    result = raceline_engine.compute_from_anchors(
        str(msg.get("map", "")),
        msg.get("anchors") or [],
        msg.get("params") or {},
    )
    result["req_id"] = msg.get("req_id")
    return result


def _handle_raceline_save(msg: Dict) -> Dict:
    name = store.slugify(str(msg.get("name", "")))
    if not name:
        return {"ok": False, "reason": "Raceline name required"}
    data = msg.get("data") or {}
    required = ("s", "x", "y", "theta", "v")
    if not all(isinstance(data.get(k), list) and data.get(k) for k in required):
        return {"ok": False, "reason": "Missing raceline data arrays"}
    lengths = {len(data[k]) for k in required}
    if len(lengths) != 1:
        return {"ok": False, "reason": "Raceline data arrays are inconsistent"}
    try:
        frenet = raceline_engine.frenet_from_payload_arrays(data)
    except (ValueError, TypeError):
        return {"ok": False, "reason": "Invalid raceline data"}
    try:
        meta = store.save_raceline(
            name, frenet,
            map_name=str(msg.get("map", "")),
            params=msg.get("params") or {},
            anchors=msg.get("anchors"),
        )
    except store.StoreError as e:
        return {"ok": False, "reason": str(e)}
    broadcast.push({"type": "racelines_changed"})
    return {"ok": True, "raceline": meta}


def _handle_start_algo(msg: Dict) -> Dict:
    """Blocking when starting map-race algos (AMCL bringup); run in executor."""
    name = msg.get("algorithm", "wall_follow")
    if name in MAP_CONTROLLER_ALGOS:
        if slam_mgr.active:
            return {"ok": False,
                    "reason": "Stop SLAM mapping before racing"}
        rl_name = str(msg.get("raceline", ""))
        if not rl_name or not store.raceline_exists(rl_name):
            return {"ok": False,
                    "reason": "Select a saved raceline before starting"}
        try:
            rl_meta = store.raceline_meta(rl_name)
        except store.StoreError as e:
            return {"ok": False, "reason": str(e)}
        map_name = str(rl_meta.get("map") or "")
        if not map_name or not store.map_exists(map_name):
            return {"ok": False,
                    "reason": "Raceline has no parent map — "
                              "re-save it from Mapping step 2"}

        loc = localize_mgr.start(map_name)
        if not loc.get("ok"):
            return {"ok": False,
                    "reason": f"Localization failed: {loc.get('reason')}"}

        seed = localize_mgr.seed_initial_pose(ros_node)
        if not seed.get("ok"):
            localize_mgr.stop()
            return {"ok": False,
                    "reason": f"Initial pose seed failed: {seed.get('reason')}"}

        started = algo.start(
            name,
            extra_ros_params={
                "waypoint_file": store.raceline_csv_path(rl_name)},
            raceline=rl_name,
        )
        if not started.get("ok"):
            localize_mgr.stop()
            return started
        return {"ok": True, "algorithm": name,
                "raceline": rl_name, "map": map_name,
                "localized": True}
    return algo.start(name)


def _handle_stop_algo() -> Dict:
    result = algo.stop()
    localize_mgr.stop()
    if ros_node is not None:
        ros_node.clear_amcl_pose()
    return result


def _handle_reset() -> Dict:
    """Soft-reset the sim; if a map controller is running, re-seed AMCL."""
    if ros_node is None:
        return {"ok": False, "reason": "ROS node unavailable"}
    ros_node.publish_reset()
    if not localize_mgr.active:
        return {"ok": True, "reseeded": False}

    # Wait for odom to settle at the spawn pose after the reset pulse.
    time.sleep(0.85)
    seed = localize_mgr.seed_initial_pose(ros_node, timeout=8.0)
    if not seed.get("ok"):
        return {
            "ok": True,
            "reseeded": False,
            "localize_warning": seed.get("reason"),
        }
    return {"ok": True, "reseeded": True}


# --- WebSocket endpoint -----------------------------------------------------


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    token = ws.query_params.get("token")
    if not _authorized(token):
        await ws.close(code=4401, reason="invalid token")
        return

    await ws.accept()

    loop = asyncio.get_running_loop()

    def status_payload() -> Dict:
        loc = localize_mgr.status()
        if ros_node is not None:
            loc = {**loc, **ros_node.localization_status()}
        return {
            "type": "status",
            "sim_connected": ros_node.sim_connected(),
            "algorithm": algo.running,
            "raceline": algo.raceline,
            "slam": slam_mgr.status(),
            "localize": loc,
            "opt_running": opt_mgr.running,
        }

    async def send(obj):
        await ws.send_text(dumps_ws(obj))

    async def sender():
        telemetry_period = 1.0 / TELEMETRY_HZ
        lidar_period = 1.0 / LIDAR_HZ
        last_lidar = 0.0
        last_status = 0.0
        last_epoch = ros_node.get_session_epoch()
        last_map_seq, first_map = ros_node.map_snapshot()
        # Send the latched map immediately on (re)connect.
        if first_map is not None:
            await send({"type": "map_frame", **first_map})
        last_opt_seq = opt_mgr.seq
        last_bcast_seq = broadcast.seq
        # Status before the first telemetry so the UI can accept odom immediately.
        await send(status_payload())
        last_status = time.monotonic()
        while True:
            try:
                now = time.monotonic()

                # Watchdog may have wiped state after a sim drop — tell the UI.
                epoch = ros_node.get_session_epoch()
                if epoch != last_epoch:
                    last_epoch = epoch
                    await send({**status_payload(), "session_reset": True})

                payload = {
                    "type": "telemetry",
                    "t": round(now, 3),
                    "sim_connected": ros_node.sim_connected(),
                    **ros_node.snapshot(),
                }
                pe = ros_node.compute_pose_error(localize_mgr.map_name)
                if pe is not None:
                    payload["pose_error"] = pe
                await send(payload)

                if now - last_lidar >= lidar_period:
                    lidar = ros_node.lidar_snapshot()
                    if lidar is not None:
                        await send({"type": "lidar", **lidar})
                    last_lidar = now

                map_seq, map_frame = ros_node.map_snapshot()
                if map_seq != last_map_seq:
                    last_map_seq = map_seq
                    if map_frame is not None:
                        await send({"type": "map_frame", **map_frame})
                    else:
                        await send({"type": "map_frame", "cleared": True})

                for event in opt_mgr.events_since(last_opt_seq):
                    last_opt_seq = event["seq"]
                    # Keep envelope type "opt_progress"; the phase lives in "event"
                    # so it does not clobber the WS message type.
                    ev = {k: v for k, v in event.items() if k != "type"}
                    ev["type"] = "opt_progress"
                    ev["event"] = event.get("type", "progress")
                    await send(ev)

                for event in broadcast.since(last_bcast_seq):
                    last_bcast_seq = event["seq"]
                    await send(event)

                if now - last_status >= 1.0:
                    await send(status_payload())
                    last_status = now
            except asyncio.CancelledError:
                raise
            except Exception:
                # A single bad frame (NaN/Inf, oversized debug, …) must not
                # kill the sender — otherwise the car keeps driving in sim
                # while the UI never gets algorithm overlays.
                pass

            await asyncio.sleep(telemetry_period)

    send_task = asyncio.create_task(sender())
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps(
                    {"type": "error", "reason": "Invalid JSON"}))
                continue

            mtype = msg.get("type")
            if mtype == "ping":
                await ws.send_text(json.dumps(
                    {"type": "pong", "t": msg.get("t")}))
            elif mtype == "set_params":
                params = msg.get("params", {})
                algorithm = msg.get("algorithm") or algo.running
                if algorithm is None:
                    await ws.send_text(json.dumps({
                        "type": "params_ack", "ok": False,
                        "reason": "No algorithm running"}))
                    continue
                result = await loop.run_in_executor(
                    None, ros_node.set_algo_params, algorithm, params)
                await ws.send_text(json.dumps({"type": "params_ack", **result}))
            elif mtype == "start_algo":
                result = await loop.run_in_executor(
                    None, _handle_start_algo, msg)
                await ws.send_text(json.dumps({"type": "algo_ack", **result}))
            elif mtype == "stop_algo":
                result = _handle_stop_algo()
                await ws.send_text(json.dumps({"type": "algo_ack", **result}))
            elif mtype == "reset":
                result = await loop.run_in_executor(None, _handle_reset)
                await ws.send_text(json.dumps({"type": "reset_ack", **result}))
                await ws.send_text(json.dumps(status_payload()))
            elif mtype == "slam_start":
                result = await loop.run_in_executor(
                    None, _handle_slam_start, msg)
                await ws.send_text(json.dumps(
                    {"type": "slam_ack", "op": "start", **result}))
            elif mtype == "slam_stop":
                result = _handle_slam_stop()
                await ws.send_text(json.dumps(
                    {"type": "slam_ack", "op": "stop", **result}))
            elif mtype == "slam_save_map":
                result = await loop.run_in_executor(
                    None, _handle_slam_save, msg)
                await ws.send_text(json.dumps(
                    {"type": "slam_ack", "op": "save", **result}))
            elif mtype == "map_edit":
                result = await loop.run_in_executor(
                    None, _handle_map_edit, msg)
                await ws.send_text(json.dumps(
                    {"type": "map_ack", "op": "edit", **result}))
            elif mtype == "map_undo":
                result = await loop.run_in_executor(
                    None, _handle_map_undo_redo, msg, False)
                await ws.send_text(json.dumps(
                    {"type": "map_ack", "op": "undo", **result}))
            elif mtype == "map_redo":
                result = await loop.run_in_executor(
                    None, _handle_map_undo_redo, msg, True)
                await ws.send_text(json.dumps(
                    {"type": "map_ack", "op": "redo", **result}))
            elif mtype == "map_delete":
                try:
                    store.delete_map(str(msg.get("name", "")))
                    result = {"ok": True}
                    broadcast.push({"type": "maps_changed"})
                except store.StoreError as e:
                    result = {"ok": False, "reason": str(e)}
                await ws.send_text(json.dumps(
                    {"type": "map_ack", "op": "delete", **result}))
            elif mtype == "map_rename":
                try:
                    store.rename_map(str(msg.get("old", "")),
                                     store.slugify(str(msg.get("new", ""))))
                    result = {"ok": True}
                    broadcast.push({"type": "maps_changed"})
                except store.StoreError as e:
                    result = {"ok": False, "reason": str(e)}
                await ws.send_text(json.dumps(
                    {"type": "map_ack", "op": "rename", **result}))
            elif mtype == "raceline_update":
                result = await loop.run_in_executor(
                    None, _handle_raceline_update, msg)
                await ws.send_text(json.dumps(
                    {"type": "raceline_data", **result}))
            elif mtype == "raceline_save":
                result = await loop.run_in_executor(
                    None, _handle_raceline_save, msg)
                await ws.send_text(json.dumps(
                    {"type": "raceline_ack", "op": "save", **result}))
            elif mtype == "raceline_delete":
                try:
                    store.delete_raceline(str(msg.get("name", "")))
                    result = {"ok": True}
                    broadcast.push({"type": "racelines_changed"})
                except store.StoreError as e:
                    result = {"ok": False, "reason": str(e)}
                await ws.send_text(json.dumps(
                    {"type": "raceline_ack", "op": "delete", **result}))
            elif mtype == "opt_start":
                result = opt_mgr.start(str(msg.get("map", "")),
                                       msg.get("params") or {})
                await ws.send_text(json.dumps(
                    {"type": "opt_ack", "op": "start", **result}))
            elif mtype == "opt_cancel":
                result = opt_mgr.cancel()
                await ws.send_text(json.dumps(
                    {"type": "opt_ack", "op": "cancel", **result}))
            else:
                await ws.send_text(json.dumps(
                    {"type": "error", "reason": f"Unknown message type: {mtype}"}))
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _spin_ros():
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(ros_node)
    executor.spin()


def _sim_watchdog():
    """Stop algorithms and clear caches whenever the simulator drops.
    Also handles the 'auto-stop mapping drive after one lap' feature."""
    while True:
        node = ros_node
        if node is not None:
            if node.poll_sim_disconnect():
                node.get_logger().info(
                    "Simulator disconnected — stopping algorithm and clearing state")
                algo.stop()
                if localize_mgr.active:
                    localize_mgr.stop()
                if slam_mgr.active:
                    slam_mgr.stop()
                    node.clear_map_frame()
                node.clear_session_state()
        time.sleep(0.25)


def main():
    global ros_node
    try:
        store.ensure_dirs()
    except OSError as e:
        print(f"[gateway] WARNING: data dir unavailable ({e}); "
              "maps/racelines will not persist")
    rclpy.init()
    ros_node = GatewayNode()
    threading.Thread(target=_spin_ros, daemon=True).start()
    threading.Thread(target=_sim_watchdog, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT, log_level="info")

    algo.stop()
    slam_mgr.stop()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
