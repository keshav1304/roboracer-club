"""
SLAM session manager for the Mapping.

Runs slam_toolbox (async mapping) as a subprocess, saves named maps into the
artifact store via the /slam_toolbox/save_map service, and implements the
"auto-stop after one lap" convenience for mapping drives.

The OccupancyGrid subscription/streaming lives on the gateway ROS node in
main.py; this module only owns the slam_toolbox process lifecycle + saving.
"""

import math
import os
import signal
import subprocess
import threading
import time
from typing import Dict, Optional

import store

# slam_toolbox interfaces are only available where the apt package is
# installed (i.e. inside the backend container).
try:
    from slam_toolbox.srv import SaveMap  # type: ignore
    HAVE_SLAM_TOOLBOX = True
except ImportError:
    SaveMap = None
    HAVE_SLAM_TOOLBOX = False

# Bundled slam_toolbox params (wired to AutoDRIVE topics/frames).
_PARAM_CANDIDATES = (
    os.environ.get("SLAM_PARAMS_FILE", ""),
    "/home/autodrive_devkit/src/autodrive_devkit/f1tenth_online_async.yaml",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "autodrive_devkit", "f1tenth_online_async.yaml"),
)

SAVE_TMP_DIR = "/tmp/playground_slam"


def _params_file() -> Optional[str]:
    for p in _PARAM_CANDIDATES:
        if p and os.path.isfile(p):
            return p
    return None


class SlamManager:
    """Owns the slam_toolbox subprocess and mapping-session state."""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._save_client = None

        # Mapping-session state (reported in the status stream).
        self.drive: Optional[str] = None       # "wall_follow" | "manual"
        self.auto_stop: bool = False
        self.lap_done: bool = False
        self._start_lap: Optional[int] = None
        self.started_at: Optional[float] = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def status(self) -> Dict:
        return {
            "active": self.active,
            "drive": self.drive,
            "auto_stop": self.auto_stop,
            "lap_done": self.lap_done,
            "started_at": self.started_at,
        }

    # --- lifecycle ---------------------------------------------------------

    def start(self, *, drive: str, auto_stop: bool, start_lap: int) -> Dict:
        params = _params_file()
        if params is None:
            return {"ok": False,
                    "reason": "slam_toolbox params file not found on backend"}
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return {"ok": False, "reason": "SLAM is already running"}
            try:
                self._proc = subprocess.Popen(
                    [
                        "ros2", "run", "slam_toolbox",
                        "async_slam_toolbox_node",
                        "--ros-args",
                        "--params-file", params,
                        "-r", "__node:=slam_toolbox",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid,
                )
            except OSError as e:
                self._proc = None
                return {"ok": False, "reason": str(e)}
            self.drive = drive
            self.auto_stop = bool(auto_stop)
            self.lap_done = False
            self._start_lap = int(start_lap)
            self.started_at = time.time()
        return {"ok": True}

    def stop(self) -> Dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGINT)
                    self._proc.wait(timeout=3)
                except (subprocess.TimeoutExpired, ProcessLookupError,
                        PermissionError):
                    try:
                        os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
            self._proc = None
            self.drive = None
            self.auto_stop = False
            self.lap_done = False
            self._start_lap = None
            self.started_at = None
        return {"ok": True}

    # --- auto-stop ---------------------------------------------------------

    def poll_lap_complete(self, lap_count: int) -> bool:
        """
        Called from the watchdog loop. Returns True exactly once when the
        mapping drive has completed a lap and should be stopped.
        """
        if (not self.active or not self.auto_stop or self.lap_done
                or self._start_lap is None):
            return False
        if lap_count > self._start_lap:
            self.lap_done = True
            return True
        return False

    # --- saving ------------------------------------------------------------

    def save_map(self, node, name: str) -> Dict:
        """
        Save the live SLAM map into the store under `name`.
        Blocking — call from a worker thread, never the event loop.
        """
        if not self.active:
            return {"ok": False, "reason": "SLAM is not running"}
        if not HAVE_SLAM_TOOLBOX:
            return {"ok": False,
                    "reason": "slam_toolbox interfaces not installed"}
        try:
            store.validate_name(name)
        except store.StoreError as e:
            return {"ok": False, "reason": str(e)}

        os.makedirs(SAVE_TMP_DIR, exist_ok=True)
        base = os.path.join(SAVE_TMP_DIR, name)
        for ext in (".yaml", ".pgm"):
            try:
                os.remove(base + ext)
            except FileNotFoundError:
                pass

        if self._save_client is None:
            self._save_client = node.create_client(
                SaveMap, "/slam_toolbox/save_map")
        client = self._save_client
        if not client.wait_for_service(timeout_sec=3.0):
            return {"ok": False, "reason": "save_map service unavailable"}

        req = SaveMap.Request()
        req.name.data = base
        future = client.call_async(req)
        deadline = time.monotonic() + 15.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not future.done():
            return {"ok": False, "reason": "save_map service timed out"}

        yaml_path, pgm_path = base + ".yaml", base + ".pgm"
        # slam_toolbox writes asynchronously; give the files a moment.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not (
                os.path.isfile(yaml_path) and os.path.isfile(pgm_path)):
            time.sleep(0.1)
        if not (os.path.isfile(yaml_path) and os.path.isfile(pgm_path)):
            return {"ok": False,
                    "reason": "slam_toolbox did not produce map files"}

        map_to_world = node.map_to_world_transform()
        try:
            meta = store.save_map(name, yaml_path, pgm_path,
                                  map_to_world=map_to_world)
        except store.StoreError as e:
            return {"ok": False, "reason": str(e)}
        return {"ok": True, "map": meta}


def transform_to_dict(t) -> Dict:
    """geometry_msgs/Transform → {tx, ty, yaw} (2-D)."""
    q = t.rotation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                     1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return {
        "tx": round(t.translation.x, 4),
        "ty": round(t.translation.y, 4),
        "yaw": round(yaw, 5),
    }
