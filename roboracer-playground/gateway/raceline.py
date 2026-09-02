"""
Raceline engine for the Mapping.

Two paths, both built on the devkit's generate_waypoints.py:

  * Fast path  — anchors + params → periodic spline → Frenet + velocity
                 profile + wall-clearance audit. Milliseconds; used for
                 real-time editing in the browser.
  * Heavy path — CMA-ES minimum-curvature optimization in a separate
                 process, emitting progress (best cost + preview polyline)
                 so the UI can animate the line improving live.
"""

import multiprocessing as mp
import os
import sys
import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.ndimage import distance_transform_edt

import store

# generate_waypoints.py lives in the bundled devkit.
_DEVKIT_CANDIDATES = (
    os.environ.get("DEVKIT_DIR", ""),
    "/home/autodrive_devkit/src/autodrive_devkit",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "autodrive_devkit"),
)
for _d in _DEVKIT_CANDIDATES:
    if _d and os.path.isfile(os.path.join(_d, "generate_waypoints.py")):
        if _d not in sys.path:
            sys.path.insert(0, _d)
        break

import generate_waypoints as gw  # noqa: E402


DEFAULT_PARAMS = {
    "vmin": 0.5,
    "vmax": 6.5,
    "alat": 3.5,
    "smooth": 2.0,        # splprep s for the anchor spline (fast path)
    "n_out": 800,
    "margin": 0.25,       # metres, wall clearance target
    # heavy path only:
    "budget": 8000,
    "n_ctrl": 80,
    "sigma0": 2.0,
    "erode": 2,
    "spl_smooth": 0.0,
    "frenet_sigma": 0.0,
}


def _merged_params(params: Optional[Dict]) -> Dict:
    out = dict(DEFAULT_PARAMS)
    for k, v in (params or {}).items():
        if k in out:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out


# ---------------------------------------------------------------------------
# Map cache (free mask + distance map per map version)
# ---------------------------------------------------------------------------

_map_cache: Dict[str, Dict] = {}
_map_cache_lock = threading.Lock()


def _load_map_ctx(map_name: str) -> Dict:
    meta = store.map_meta(map_name)
    key = f"{map_name}:{meta.get('version', 0)}"
    with _map_cache_lock:
        if key in _map_cache:
            return _map_cache[key]
    img, res, origin, occ_t, free_t, neg = gw.load_map(
        store.map_yaml_path(map_name))
    free_mask = gw.build_free_mask(img, occ_t, free_t, neg)
    dist_map = distance_transform_edt(free_mask.astype(np.uint8))
    ctx = {
        "res": float(res),
        "origin": list(origin),
        "img_h": int(img.shape[0]),
        "img_w": int(img.shape[1]),
        "dist_map": dist_map,
        "meta": meta,
    }
    with _map_cache_lock:
        _map_cache.clear()  # keep at most one map resident (they are large)
        _map_cache[key] = ctx
    return ctx


def _clearance_m(world_pts: np.ndarray, ctx: Dict) -> np.ndarray:
    """Distance to nearest wall (metres) for world-frame points."""
    res, origin = ctx["res"], ctx["origin"]
    img_h, img_w = ctx["img_h"], ctx["img_w"]
    cols = np.clip(((world_pts[:, 0] - origin[0]) / res).astype(int),
                   0, img_w - 1)
    rows = np.clip((img_h - 1 - (world_pts[:, 1] - origin[1]) / res)
                   .astype(int), 0, img_h - 1)
    return ctx["dist_map"][rows, cols] * res


# ---------------------------------------------------------------------------
# Fast path: anchors → raceline
# ---------------------------------------------------------------------------

def compute_from_anchors(map_name: str, anchors: List[List[float]],
                         params: Optional[Dict]) -> Dict:
    """
    anchors: [[x,y], ...] world coordinates, ≥4 points (closed loop).
    Returns a raceline_data payload dict (JSON-safe).
    """
    p = _merged_params(params)
    try:
        pts = np.asarray(anchors, dtype=float)
    except (ValueError, TypeError):
        return {"ok": False, "reason": "Invalid anchor points"}
    if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] < 2:
        return {"ok": False,
                "reason": "Need at least 4 anchor points for a closed loop"}
    pts = pts[:, :2]

    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]],
                         s=max(0.0, p["smooth"]), per=True, k=3)
    except Exception:
        return {"ok": False,
                "reason": "Spline fit failed — adjust anchors or smoothing"}
    n_out = int(max(100, min(2000, p["n_out"])))
    u = np.linspace(0, 1, n_out, endpoint=False)
    sx, sy = splev(u, tck)
    world_pts = np.stack([sx, sy], axis=1)

    frenet = gw.compute_frenet(
        world_pts, p["vmin"], p["vmax"], p["alat"],
        smooth_sigma=max(0.0, p["frenet_sigma"]))

    try:
        ctx = _load_map_ctx(map_name)
        clearance = _clearance_m(world_pts, ctx)
    except (store.StoreError, RuntimeError, FileNotFoundError):
        clearance = None

    return _payload(frenet, clearance, p)


def _payload(frenet: np.ndarray, clearance: Optional[np.ndarray],
             p: Dict) -> Dict:
    ds = np.diff(frenet[:, 0])
    v = np.maximum(frenet[:-1, 4], 0.1)
    lap_time = float(np.sum(ds / v)) if len(frenet) > 1 else 0.0
    out = {
        "ok": True,
        "s": [round(float(x), 4) for x in frenet[:, 0]],
        "x": [round(float(x), 4) for x in frenet[:, 1]],
        "y": [round(float(x), 4) for x in frenet[:, 2]],
        "theta": [round(float(x), 4) for x in frenet[:, 3]],
        "v": [round(float(x), 3) for x in frenet[:, 4]],
        "length_m": round(float(frenet[-1, 0]), 2) if len(frenet) else 0.0,
        "lap_time_est": round(lap_time, 2),
        "margin": p["margin"],
    }
    if clearance is not None:
        out["clearance"] = [round(float(c), 3) for c in clearance]
        out["min_clearance_m"] = round(float(clearance.min()), 3)
        out["violations"] = int(np.sum(clearance < p["margin"]))
    return out


def frenet_from_payload_arrays(data: Dict) -> np.ndarray:
    """Rebuild the (N,5) frenet table from a raceline_data payload."""
    return np.stack([
        np.asarray(data["s"], dtype=float),
        np.asarray(data["x"], dtype=float),
        np.asarray(data["y"], dtype=float),
        np.asarray(data["theta"], dtype=float),
        np.asarray(data["v"], dtype=float),
    ], axis=1)


def anchors_from_raceline(x: List[float], y: List[float],
                          n_anchors: int = 40) -> List[List[float]]:
    """Subsample a dense raceline into editable anchors."""
    n = len(x)
    if n == 0:
        return []
    idx = np.round(np.linspace(0, n - 1, min(n_anchors, n),
                               endpoint=False)).astype(int)
    return [[round(float(x[i]), 4), round(float(y[i]), 4)] for i in idx]


# ---------------------------------------------------------------------------
# Heavy path: CMA-ES optimization worker (separate process)
# ---------------------------------------------------------------------------

def _opt_worker(map_yaml: str, params: Dict, q):
    """Runs in a spawned process. Streams progress dicts into q."""
    try:
        import cma  # local import: only needed in the worker

        p = params
        ctx = gw.load_track_and_ctrl(
            map_yaml, erode=int(p["erode"]), n_ctrl=int(p["n_ctrl"]))
        ctrl_pts = ctx["ctrl_pts"]
        dist_map = ctx["dist_map"]
        res, origin, img_h = ctx["res"], ctx["origin"], ctx["img_h"]
        margin_px = p["margin"] / res
        spl_s = max(0.0, p["spl_smooth"])

        n_ctrl = int(ctrl_pts.shape[0])
        bound = gw.build_offset_bounds(ctrl_pts, dist_map, margin_px)
        opts = cma.CMAOptions()
        opts["maxfevals"] = int(p["budget"])
        opts["verbose"] = -9
        opts["tolx"] = 1e-4
        opts["tolfun"] = 1e-5
        opts["bounds"] = [[-b for b in bound], bound]
        opts["popsize"] = max(24, n_ctrl)

        es = cma.CMAEvolutionStrategy(np.zeros(n_ctrl), float(p["sigma0"]),
                                      opts)
        evals = 0
        last_emit = 0.0
        while not es.stop():
            candidates = es.ask()
            fitnesses = [
                gw.objective(np.array(c), ctrl_pts, dist_map, margin_px,
                             n_eval=200, spl_smooth=spl_s)
                for c in candidates
            ]
            es.tell(candidates, fitnesses)
            evals += len(candidates)

            now = time.monotonic()
            # Emit immediately after the first generation, then ~1.5 s cadence.
            if ((last_emit == 0.0 or now - last_emit >= 1.5)
                    and es.result.xbest is not None):
                last_emit = now
                preview_px = gw.perturb_and_sample(
                    ctrl_pts, np.array(es.result.xbest), n_out=200,
                    spl_smooth=spl_s)
                preview = None
                if preview_px is not None:
                    w = gw.pixels_to_world(preview_px, img_h, res, origin)
                    preview = {
                        "x": [round(float(v), 3) for v in w[:, 0]],
                        "y": [round(float(v), 3) for v in w[:, 1]],
                    }
                q.put({
                    "type": "progress",
                    "evals": evals,
                    "budget": int(p["budget"]),
                    "best_cost": round(float(es.result.fbest), 5),
                    "preview": preview,
                })

        best = np.array(es.result.xbest)
        frenet = gw.frenet_from_offsets(
            ctrl_pts, best, img_h, res, origin,
            spl_smooth=spl_s,
            n_out=int(p["n_out"]),
            frenet_sigma=max(0.0, p["frenet_sigma"]),
            v_min=p["vmin"], v_max=p["vmax"], a_lat=p["alat"],
        )
        if frenet is None:
            q.put({"type": "error",
                   "reason": "Optimized spline invalid — try a smaller "
                             "margin or a larger budget"})
            return
        q.put({
            "type": "done",
            "evals": evals,
            "best_cost": round(float(es.result.fbest), 5),
            "frenet": frenet.tolist(),
        })
    except Exception as e:  # surfaced to the UI
        q.put({"type": "error", "reason": f"{type(e).__name__}: {e}"})


class OptManager:
    """
    One optimization at a time. Progress is exposed via (seq, message) so
    every websocket sender loop can forward new events to its client.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._proc: Optional[mp.Process] = None
        self._queue = None
        self._reader: Optional[threading.Thread] = None
        self._map: Optional[str] = None
        self._params: Optional[Dict] = None
        self.seq = 0
        self.events: List[Dict] = []   # ring of recent events (seq-stamped)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.is_alive()

    def start(self, map_name: str, params: Optional[Dict]) -> Dict:
        try:
            store.validate_name(map_name)
            if not store.map_exists(map_name):
                return {"ok": False, "reason": f"Map not found: {map_name}"}
        except store.StoreError as e:
            return {"ok": False, "reason": str(e)}

        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                return {"ok": False,
                        "reason": "An optimization is already running"}
            p = _merged_params(params)
            ctx = mp.get_context("spawn")
            self._queue = ctx.Queue()
            self._proc = ctx.Process(
                target=_opt_worker,
                args=(store.map_yaml_path(map_name), p, self._queue),
                daemon=True,
            )
            self._map = map_name
            self._params = p
            self._proc.start()
            self._push({"type": "started", "map": map_name, "params": p})

        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()
        return {"ok": True, "map": map_name, "params": p}

    def cancel(self) -> Dict:
        with self._lock:
            if self._proc is None or not self._proc.is_alive():
                return {"ok": False, "reason": "No optimization running"}
            self._proc.terminate()
            self._push({"type": "cancelled"})
        return {"ok": True}

    def _push(self, event: Dict):
        # Called with or without the lock held; append is atomic enough
        # for our single-writer usage.
        self.seq += 1
        event = dict(event)
        event["seq"] = self.seq
        event["map"] = self._map
        self.events.append(event)
        if len(self.events) > 50:
            del self.events[:-50]

    def events_since(self, seq: int) -> List[Dict]:
        return [e for e in self.events if e["seq"] > seq]

    def _drain(self):
        """Forward worker queue events; convert 'done' into raceline_data."""
        proc, q = self._proc, self._queue
        while True:
            try:
                msg = q.get(timeout=0.5)
            except Exception:
                if proc is None or not proc.is_alive():
                    break
                continue
            if msg["type"] == "progress":
                self._push({"type": "progress", **{
                    k: msg[k] for k in ("evals", "budget", "best_cost",
                                        "preview")}})
            elif msg["type"] == "error":
                self._push({"type": "error", "reason": msg["reason"]})
                break
            elif msg["type"] == "done":
                frenet = np.asarray(msg["frenet"], dtype=float)
                try:
                    ctx2 = _load_map_ctx(self._map)
                    clearance = _clearance_m(frenet[:, 1:3], ctx2)
                except (store.StoreError, RuntimeError, FileNotFoundError):
                    clearance = None
                data = _payload(frenet, clearance, self._params)
                anchors = anchors_from_raceline(data["x"], data["y"])
                self._push({
                    "type": "done",
                    "evals": msg["evals"],
                    "best_cost": msg["best_cost"],
                    "raceline": data,
                    "anchors": anchors,
                    "params": self._params,
                })
                break
        with self._lock:
            self._proc = None
