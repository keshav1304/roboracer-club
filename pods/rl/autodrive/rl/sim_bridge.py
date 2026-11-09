"""Socket.IO link to one AutoDRIVE Simulator instance, without ROS.

The simulator is the client and connects to this server. It sends a 'Bridge' event and only sends
the next one after it gets a reply, so every reply carries the latest command.

Everything gevent runs on one background thread, created there, with no monkey patching: gevent
objects belong to the thread that made them, and the training code must keep real threads.
"""
from __future__ import annotations

import base64
import gzip
import threading
from dataclasses import dataclass

import numpy as np
import socketio
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler


@dataclass
class Telemetry:
    x: float
    y: float
    yaw: float
    speed: float
    yaw_rate: float
    steering: float
    throttle: float
    lap_count: int
    lap_time: float
    last_lap_time: float
    best_lap_time: float
    collisions: int
    lidar_b64: str | None


def _vec(text: str) -> list[float]:
    return [float(v) for v in text.split()]


def _parse(d: dict) -> Telemetry:
    pos = _vec(d['V1 Position'])
    euler = _vec(d['V1 Orientation Euler Angles'])
    return Telemetry(
        x=pos[0], y=pos[1], yaw=euler[2],
        speed=_vec(d['V1 Linear Velocity'])[0],
        yaw_rate=_vec(d['V1 Angular Velocity'])[2],
        steering=float(d['V1 Steering']),
        throttle=float(d['V1 Throttle']),
        lap_count=int(float(d['V1 Lap Count'])),
        lap_time=float(d['V1 Lap Time']),
        last_lap_time=float(d['V1 Last Lap Time']),
        best_lap_time=float(d['V1 Best Lap Time']),
        collisions=int(float(d['V1 Collisions'])),
        lidar_b64=d.get('V1 LIDAR Range Array'),     # missing from the first frame in batch mode
    )


def decode_lidar(b64: str) -> np.ndarray:
    """Costly compared to the rest of a frame, so only done once per control step."""
    return np.array(gzip.decompress(base64.b64decode(b64)).decode().split(), dtype=np.float32)


class SimBridge:
    def __init__(self, port: int = 4567):
        self.port = port
        self._cond = threading.Condition()
        self._latest: Telemetry | None = None
        self._frames = 0
        self._sim_clock = 0.0             # seconds of simulator time, survives lap and scene resets
        self._prev_lap_time: float | None = None
        self._connected = False
        self._cmd = {'V1 Throttle': '0.0', 'V1 Steering': '0.0', 'V1 Reset': 'False'}
        threading.Thread(target=self._serve, daemon=True, name=f'bridge-{port}').start()

    def _serve(self) -> None:
        sio = socketio.Server(async_mode='gevent', ping_timeout=60, logger=False, engineio_logger=False)

        @sio.on('connect')
        def on_connect(sid, environ):
            # A plain HTTP probe also lands here, so "connected" waits for the first telemetry frame.
            pass

        @sio.on('disconnect')
        def on_disconnect(sid):
            with self._cond:
                self._connected = False
                self._latest = None
                self._prev_lap_time = None
                self._cond.notify_all()

        @sio.on('Bridge')
        def on_bridge(sid, data):
            if data:
                frame = _parse(data)
                with self._cond:
                    if self._prev_lap_time is not None and frame.lap_time > self._prev_lap_time:
                        self._sim_clock += frame.lap_time - self._prev_lap_time
                    self._prev_lap_time = frame.lap_time
                    self._latest = frame
                    self._frames += 1
                    self._connected = True
                    self._cond.notify_all()
            with self._cond:
                cmd = dict(self._cmd)
            sio.emit('Bridge', cmd, room=sid)

        pywsgi.WSGIServer(('0.0.0.0', self.port), socketio.WSGIApp(sio),
                          handler_class=WebSocketHandler, log=None, error_log=None).serve_forever()

    # ---- commands -----------------------------------------------------------------------------
    def set_command(self, throttle: float, steering: float, reset: bool = False) -> None:
        with self._cond:
            self._cmd = {'V1 Throttle': f'{throttle:.4f}', 'V1 Steering': f'{steering:.4f}',
                         'V1 Reset': 'True' if reset else 'False'}

    # ---- telemetry ----------------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        with self._cond:
            return self._connected

    @property
    def sim_clock(self) -> float:
        with self._cond:
            return self._sim_clock

    def latest(self) -> Telemetry | None:
        with self._cond:
            return self._latest

    def wait_connected(self, timeout: float) -> bool:
        with self._cond:
            return self._cond.wait_for(lambda: self._connected, timeout)

    def wait_frames(self, count: int = 1, timeout: float = 5.0) -> Telemetry | None:
        """The frame `count` frames from now, or None if the simulator stops talking."""
        with self._cond:
            target = self._frames + count
            ok = self._cond.wait_for(lambda: self._frames >= target or not self._connected, timeout)
            return self._latest if ok and self._connected else None

    def wait_sim_clock(self, target: float, timeout: float = 5.0) -> Telemetry | None:
        """The first frame at or after simulator time `target`, or None on a stall."""
        with self._cond:
            ok = self._cond.wait_for(lambda: self._sim_clock >= target or not self._connected, timeout)
            return self._latest if ok and self._connected else None

    def wait_sim_time(self, seconds: float, timeout: float = 5.0) -> Telemetry | None:
        """The first frame at least `seconds` of simulator time from now, or None on a stall."""
        return self.wait_sim_clock(self.sim_clock + seconds, timeout)


