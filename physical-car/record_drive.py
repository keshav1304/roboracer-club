"""Record camera frames and the human drive command for behavioral cloning.

Copy to the car and run it there:

    cd ~/jupyter_ws/neoracer-os/labs/tests
    python3 record_drive.py

Leave the FlySky SWB switch up so the sticks drive the car. This script
publishes zeros on /drive, so flipping SWB down stops the car. SWA up is the
transmitter's slow speed; collect at the speed you want the model to drive.
Ctrl+C finishes the run and flushes the log.

Frames land in ~/data/bc_runs/<timestamp>/:

    meta.json
    frames.csv
    images/000000.jpg

steer and throttle in frames.csv are what rc.drive.set_speed_angle expects:
+steer is right, +throttle is forward, both in [-1, 1]. They come from
/gamepad_drive. Train on rows where manual is 1, which means /mux_out matched
the sticks and SWB was in manual.
"""

import csv
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Float32

_LIBRARY = Path.home() / "jupyter_ws" / "neoracer-os" / "library"
if _LIBRARY.is_dir() and str(_LIBRARY) not in sys.path:
    sys.path.insert(0, str(_LIBRARY))

import racecar_core

# /mux_out must sit on top of /gamepad_drive for the row to count as manual.
MANUAL_MATCH = 0.05
# Drop a row when the stick command is older than this. The mux itself zeros
# a stick that has been quiet for half a second.
COMMAND_TIMEOUT_SEC = 0.3
JPEG_QUALITY = 85
PRINT_PERIOD_SEC = 0.5

_SENSOR_QOS = QoSProfile(depth=1)
_SENSOR_QOS.history = QoSHistoryPolicy.KEEP_LAST
_SENSOR_QOS.reliability = QoSReliabilityPolicy.BEST_EFFORT
_SENSOR_QOS.durability = QoSDurabilityPolicy.VOLATILE

_COLUMNS = [
    "index",
    "file",
    "t_sec",
    "steer",
    "throttle",
    "manual",
    "gamepad_speed",
    "gamepad_steer",
    "mux_speed",
    "mux_steer",
    "stick_right_x",
    "stick_left_y",
    "encoder_speed",
]


class _Commands:
    """Latest stick command, the mux output, and the encoder speed."""

    def __init__(self):
        self._lock = threading.Lock()
        self._gamepad_speed = 0.0
        self._gamepad_steer = 0.0
        self._gamepad_at = None
        self._mux_speed = 0.0
        self._mux_steer = 0.0
        self._mux_at = None
        self._encoder = None

    def on_gamepad(self, msg):
        with self._lock:
            self._gamepad_speed = float(msg.drive.speed)
            self._gamepad_steer = float(msg.drive.steering_angle)
            self._gamepad_at = time.monotonic()

    def on_mux(self, msg):
        with self._lock:
            self._mux_speed = float(msg.drive.speed)
            self._mux_steer = float(msg.drive.steering_angle)
            self._mux_at = time.monotonic()

    def on_encoder(self, msg):
        with self._lock:
            self._encoder = float(msg.data)

    def snapshot(self):
        now = time.monotonic()
        with self._lock:
            gamepad_age = None if self._gamepad_at is None else now - self._gamepad_at
            mux_age = None if self._mux_at is None else now - self._mux_at
            fresh = (
                gamepad_age is not None
                and gamepad_age <= COMMAND_TIMEOUT_SEC
                and mux_age is not None
                and mux_age <= COMMAND_TIMEOUT_SEC
            )
            manual = fresh and (
                abs(self._mux_speed - self._gamepad_speed) <= MANUAL_MATCH
                and abs(self._mux_steer - self._gamepad_steer) <= MANUAL_MATCH
            )
            return {
                "fresh": fresh,
                "manual": manual,
                "gamepad_speed": self._gamepad_speed,
                "gamepad_steer": self._gamepad_steer,
                "mux_speed": self._mux_speed,
                "mux_steer": self._mux_steer,
                "encoder_speed": self._encoder,
            }


class _Writer:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.image_dir = self.run_dir / "images"
        self.image_dir.mkdir(parents=True)
        self._csv = open(self.run_dir / "frames.csv", "w", newline="", buffering=1)
        self._rows = csv.DictWriter(self._csv, fieldnames=_COLUMNS)
        self._rows.writeheader()
        self._queue = queue.Queue(maxsize=90)
        self.saved = 0
        self.dropped = 0
        self._thread = threading.Thread(target=self._loop, name="frame-writer", daemon=True)
        self._thread.start()

    def submit(self, image, row):
        try:
            self._queue.put_nowait((image, row))
        except queue.Full:
            self.dropped += 1

    def close(self):
        self._queue.put(None)
        self._thread.join()
        self._csv.close()

    def _loop(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            image, row = item
            path = self.image_dir / row["file"]
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                print(f"failed to write {path}")
                continue
            self._rows.writerow(row)
            self.saved += 1


rc = racecar_core.create_racecar()
_commands = _Commands()
_node = rclpy.create_node("record_drive")
_node.create_subscription(
    AckermannDriveStamped, "/gamepad_drive", _commands.on_gamepad, _SENSOR_QOS
)
_node.create_subscription(
    AckermannDriveStamped, "/mux_out", _commands.on_mux, _SENSOR_QOS
)
_node.create_subscription(Float32, "/encoder/speed", _commands.on_encoder, _SENSOR_QOS)
rclpy.get_global_executor().add_node(_node)

_writer = None
_t0 = 0.0
_next_index = 0
_last_image_id = None
_print_at = 0.0
_waiting_notice = False


def _run_dir():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path.home() / "data" / "bc_runs"
    return root / stamp


def _write_meta(run_dir, image_shape):
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "image_encoding": "bgr jpeg",
        "image_shape_hwc": list(image_shape) if image_shape is not None else None,
        "steer": "+1 is full right. Pass this number to rc.drive.set_speed_angle as angle.",
        "throttle": "+1 is full forward. Pass this number to rc.drive.set_speed_angle as speed.",
        "label_source": "/gamepad_drive. steer is the negation of steering_angle, because that topic uses +left and set_speed_angle uses +right.",
        "manual": "1 when /mux_out matched /gamepad_drive, so the FlySky sticks were in control.",
        "stick_right_x": "rc.controller right stick x, after the library's 0.2 deadzone. Same sign as steer.",
        "stick_left_y": "rc.controller left stick y, after the library's 0.2 deadzone. Same sign as throttle.",
        "encoder_speed": "latest /encoder/speed sample, or blank when none has arrived.",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def start():
    global _writer, _t0, _next_index, _last_image_id, _print_at, _waiting_notice
    if _writer is not None:
        return
    run_dir = _run_dir()
    _writer = _Writer(run_dir)
    _write_meta(run_dir, None)
    _t0 = time.monotonic()
    _next_index = 0
    _last_image_id = None
    _print_at = 0.0
    _waiting_notice = False
    rc.drive.stop()
    print(f"recording to {run_dir}")
    print("SWB up. Drive the lap. Ctrl+C to stop and flush.")


def update():
    global _next_index, _last_image_id, _print_at, _waiting_notice
    rc.drive.stop()
    image = rc.camera.get_color_image_no_copy()
    command = _commands.snapshot()
    now = time.monotonic()

    if image is None or not command["fresh"]:
        if not _waiting_notice or now >= _print_at:
            missing = "camera" if image is None else "transmitter"
            print(f"waiting for {missing}")
            _waiting_notice = True
            _print_at = now + PRINT_PERIOD_SEC
        return

    image_id = id(image)
    if image_id == _last_image_id:
        return
    _last_image_id = image_id
    _waiting_notice = False

    # +steer is right. /gamepad_drive steering_angle is +left.
    steer = -command["gamepad_steer"]
    throttle = command["gamepad_speed"]
    right_x, _right_y = rc.controller.get_joystick(rc.controller.Joystick.RIGHT)
    _left_x, left_y = rc.controller.get_joystick(rc.controller.Joystick.LEFT)
    encoder = command["encoder_speed"]
    row = {
        "index": _next_index,
        "file": f"{_next_index:06d}.jpg",
        "t_sec": f"{now - _t0:.4f}",
        "steer": f"{steer:.6f}",
        "throttle": f"{throttle:.6f}",
        "manual": "1" if command["manual"] else "0",
        "gamepad_speed": f"{command['gamepad_speed']:.6f}",
        "gamepad_steer": f"{command['gamepad_steer']:.6f}",
        "mux_speed": f"{command['mux_speed']:.6f}",
        "mux_steer": f"{command['mux_steer']:.6f}",
        "stick_right_x": f"{right_x:.6f}",
        "stick_left_y": f"{left_y:.6f}",
        "encoder_speed": "" if encoder is None else f"{encoder:.6f}",
    }
    _writer.submit(image.copy(), row)
    if _next_index == 0:
        _write_meta(_writer.run_dir, image.shape)
    _next_index += 1

    if now >= _print_at:
        _print_at = now + PRINT_PERIOD_SEC
        mode = "manual" if command["manual"] else "NOT manual, SWB is down"
        encoder_text = "none" if encoder is None else f"{encoder:.2f}"
        print(
            f"{mode}  saved {_writer.saved}  dropped {_writer.dropped}  "
            f"steer {steer:+.2f}  throttle {throttle:+.2f}  encoder {encoder_text}"
        )


rc.set_start_update(start, update)

try:
    rc.go()
finally:
    if _writer is not None:
        _writer.close()
        print(
            f"wrote {_writer.saved} frames to {_writer.run_dir} "
            f"({_writer.dropped} dropped)"
        )
