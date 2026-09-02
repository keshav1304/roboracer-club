"""
Localization session manager for Mapping racing.

Starts Nav2 map_server + AMCL against a saved Mapping map, seeds
/initialpose from current odometry transformed by the map's saved
map→world transform, and tears everything down when racing stops.

Pure Pursuit consumes /amcl_pose (map frame). Gateway telemetry still
uses ground-truth odom for UI / reporting.
"""

import math
import os
import signal
import subprocess
import threading
import time
from typing import Dict, Optional, Tuple

import store

_PARAM_CANDIDATES = (
    os.environ.get("AMCL_PARAMS_FILE", ""),
    "/home/autodrive_devkit/install/autodrive_roboracer/share/"
    "autodrive_roboracer/config/amcl_params.yaml",
    "/home/autodrive_devkit/src/autodrive_devkit/amcl_params.yaml",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "autodrive_devkit", "amcl_params.yaml"),
)


def _params_file() -> Optional[str]:
    for p in _PARAM_CANDIDATES:
        if p and os.path.isfile(p):
            return p
    return None


def world_to_map(x: float, y: float, yaw: float,
                 m2w: Optional[Dict]) -> Tuple[float, float, float]:
    """
    Transform a world-frame pose into the map frame.

    `m2w` is the saved map→world SE(2) {tx, ty, yaw} (p_w = R p_m + t).
    If missing, identity is assumed (map ≈ world).
    """
    if not m2w:
        return x, y, yaw
    tx = float(m2w.get("tx", 0.0))
    ty = float(m2w.get("ty", 0.0))
    yaw_t = float(m2w.get("yaw", 0.0))
    dx, dy = x - tx, y - ty
    c, s = math.cos(yaw_t), math.sin(yaw_t)
    mx = c * dx + s * dy
    my = -s * dx + c * dy
    return mx, my, yaw - yaw_t


def yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    """Return (x, y, z, w) for a pure Z rotation."""
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


class LocalizeManager:
    """Owns map_server + AMCL + lifecycle_manager subprocesses."""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self.map_name: Optional[str] = None
        self.started_at: Optional[float] = None
        self._seeded: bool = False

    @property
    def active(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def status(self) -> Dict:
        return {
            "active": self.active,
            "map": self.map_name,
            "seeded": self._seeded,
            "started_at": self.started_at,
        }

    def start(self, map_name: str) -> Dict:
        """Start localization against a saved map. Blocking briefly."""
        try:
            store.validate_name(map_name)
        except store.StoreError as e:
            return {"ok": False, "reason": str(e)}
        if not store.map_exists(map_name):
            return {"ok": False, "reason": f"Map not found: {map_name}"}

        params = _params_file()
        if params is None:
            return {"ok": False,
                    "reason": "AMCL params file not found on backend"}

        map_yaml = store.map_yaml_path(map_name)
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                # Already localizing the same map — reuse.
                if self.map_name == map_name:
                    return {"ok": True, "reused": True}
                self._stop_locked()

            cmd = [
                "ros2", "launch", "autodrive_roboracer", "localize.launch.py",
                f"map:={map_yaml}",
                f"params_file:={params}",
                "use_sim_time:=false",
            ]
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,
                )
            except OSError as e:
                self._proc = None
                return {"ok": False, "reason": str(e)}
            self.map_name = map_name
            self.started_at = time.time()
            self._seeded = False

        # Outside the lock: wait for lifecycle configure/activate + map load.
        time.sleep(3.0)
        if not self.active:
            return {"ok": False,
                    "reason": "Localization launch exited immediately "
                              "(is nav2-amcl installed?)"}
        return {"ok": True}

    def seed_initial_pose(self, node, *, timeout: float = 8.0) -> Dict:
        """
        Publish /initialpose from current odom, mapped via saved map_to_world.
        Blocking — call from a worker thread.
        """
        if not self.active:
            return {"ok": False, "reason": "Localization is not running"}
        if node is None:
            return {"ok": False, "reason": "Gateway ROS node unavailable"}

        # Ignore any pose published before this seed attempt (e.g. left over
        # from a prior run or pre-reset location).
        if hasattr(node, "clear_amcl_pose"):
            node.clear_amcl_pose()

        from geometry_msgs.msg import PoseWithCovarianceStamped
        # Reuse a latched-style publisher on the gateway node when possible.
        if not hasattr(node, "_initialpose_pub") or node._initialpose_pub is None:
            node._initialpose_pub = node.create_publisher(
                PoseWithCovarianceStamped, "/initialpose", 10)
            # Allow DDS discovery before the first publish.
            time.sleep(0.5)
        pub = node._initialpose_pub

        deadline = time.monotonic() + timeout
        published = 0
        last_seed = None
        while time.monotonic() < deadline:
            snap = node.snapshot()
            pos = snap.get("position") or [0.0, 0.0, 0.0]
            yaw_w = float(snap.get("yaw", 0.0))
            x_w, y_w = float(pos[0]), float(pos[1])

            m2w = None
            try:
                meta = store.map_meta(self.map_name)
                m2w = meta.get("map_to_world")
            except store.StoreError:
                pass

            mx, my, myaw = world_to_map(x_w, y_w, yaw_w, m2w)
            qx, qy, qz, qw = yaw_to_quat(myaw)
            last_seed = {"x": mx, "y": my, "yaw": myaw}

            msg = PoseWithCovarianceStamped()
            # Stamp 0 → AMCL uses latest TF (avoids extrapolation warnings).
            msg.header.stamp.sec = 0
            msg.header.stamp.nanosec = 0
            msg.header.frame_id = "map"
            msg.pose.pose.position.x = float(mx)
            msg.pose.pose.position.y = float(my)
            msg.pose.pose.position.z = 0.0
            msg.pose.pose.orientation.x = qx
            msg.pose.pose.orientation.y = qy
            msg.pose.pose.orientation.z = qz
            msg.pose.pose.orientation.w = qw
            cov = [0.0] * 36
            cov[0] = 0.25
            cov[7] = 0.25
            cov[35] = 0.068
            msg.pose.covariance = cov
            pub.publish(msg)
            published += 1
            time.sleep(0.3)
            if node.amcl_pose_fresh(max_age=2.5):
                break

        self._seeded = published > 0
        if not node.amcl_pose_fresh(max_age=2.5):
            return {
                "ok": False,
                "reason": "AMCL did not publish a pose after seeding "
                          f"(published {published} initialpose msgs). "
                          "Is the simulator connected and Autonomous mode on?",
            }
        return {"ok": True, "pose": last_seed or {"x": 0.0, "y": 0.0, "yaw": 0.0}}

    def stop(self) -> Dict:
        with self._lock:
            self._stop_locked()
        return {"ok": True}

    def _stop_locked(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)
                self._proc.wait(timeout=4)
            except (subprocess.TimeoutExpired, ProcessLookupError,
                    PermissionError):
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        self._proc = None
        self.map_name = None
        self.started_at = None
        self._seeded = False
