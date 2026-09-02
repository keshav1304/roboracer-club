"""
Artifact store for the Mapping: saved SLAM maps and racelines.

Layout (root = $PLAYGROUND_DATA_DIR, default /data):

    maps/<name>/map.yaml        ROS map metadata (image: map.pgm)
    maps/<name>/map.pgm         occupancy image (editable)
    maps/<name>/meta.json       {created, resolution, origin, version,
                                 map_to_world, ...}
    maps/<name>/history/<n>.pgm undo snapshots (bounded ring)
    racelines/<name>/waypoints.csv   s,x,y,theta,velocity (comp format)
    racelines/<name>/meta.json       {map, params, anchors, stats, created}

All names are validated slugs so they are safe as path components.
"""

import base64
import json
import os
import re
import shutil
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

DATA_DIR = os.environ.get("PLAYGROUND_DATA_DIR", "/data")
MAPS_DIR = os.path.join(DATA_DIR, "maps")
RACELINES_DIR = os.path.join(DATA_DIR, "racelines")

MAX_HISTORY = 20
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class StoreError(Exception):
    """User-facing store failure (bad name, missing artifact, ...)."""


def slugify(name: str) -> str:
    """Best-effort conversion of a display name to a valid slug."""
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("-_")
    return slug[:64]


def validate_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise StoreError(
            "Invalid name: use lowercase letters, digits, '-' or '_' "
            "(max 64 chars)")
    return name


def ensure_dirs():
    os.makedirs(MAPS_DIR, exist_ok=True)
    os.makedirs(RACELINES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

def map_dir(name: str) -> str:
    return os.path.join(MAPS_DIR, validate_name(name))


def map_yaml_path(name: str) -> str:
    return os.path.join(map_dir(name), "map.yaml")


def map_pgm_path(name: str) -> str:
    return os.path.join(map_dir(name), "map.pgm")


def map_exists(name: str) -> bool:
    try:
        return os.path.isfile(map_yaml_path(name))
    except StoreError:
        return False


def _read_json(path: str) -> Dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: str, data: Dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def save_map(name: str, src_yaml: str, src_pgm: str,
             map_to_world: Optional[Dict] = None) -> Dict:
    """Import a slam_toolbox-saved map (yaml+pgm) into the store."""
    validate_name(name)
    ensure_dirs()
    d = map_dir(name)
    os.makedirs(d, exist_ok=True)

    with open(src_yaml) as f:
        cfg = yaml.safe_load(f)

    shutil.copyfile(src_pgm, map_pgm_path(name))
    # Rewrite the yaml so `image:` always points at our canonical file name.
    cfg["image"] = "map.pgm"
    with open(map_yaml_path(name), "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=None)

    meta = {
        "name": name,
        "created": time.time(),
        "resolution": float(cfg.get("resolution", 0.05)),
        "origin": list(cfg.get("origin", [0.0, 0.0, 0.0])),
        "version": 0,
        "map_to_world": map_to_world,
    }
    _write_json(os.path.join(d, "meta.json"), meta)
    # Clear stale undo history from a previous map of the same name.
    shutil.rmtree(os.path.join(d, "history"), ignore_errors=True)
    return map_meta(name)


def map_meta(name: str) -> Dict:
    d = map_dir(name)
    if not os.path.isfile(map_yaml_path(name)):
        raise StoreError(f"Map not found: {name}")
    meta = _read_json(os.path.join(d, "meta.json"))
    img = cv2.imread(map_pgm_path(name), cv2.IMREAD_GRAYSCALE)
    h, w = (img.shape if img is not None else (0, 0))
    meta.setdefault("name", name)
    meta["width"] = int(w)
    meta["height"] = int(h)
    meta.setdefault("version", 0)
    return meta


def list_maps() -> List[Dict]:
    ensure_dirs()
    out = []
    for name in sorted(os.listdir(MAPS_DIR)):
        if _NAME_RE.match(name) and os.path.isfile(map_yaml_path(name)):
            try:
                out.append(map_meta(name))
            except StoreError:
                continue
    out.sort(key=lambda m: m.get("created", 0), reverse=True)
    return out


def delete_map(name: str):
    d = map_dir(name)
    if not os.path.isdir(d):
        raise StoreError(f"Map not found: {name}")
    shutil.rmtree(d)


def rename_map(old: str, new: str):
    validate_name(new)
    if map_exists(new):
        raise StoreError(f"Map already exists: {new}")
    src = map_dir(old)
    if not os.path.isdir(src):
        raise StoreError(f"Map not found: {old}")
    shutil.move(src, map_dir(new))
    meta_path = os.path.join(map_dir(new), "meta.json")
    meta = _read_json(meta_path)
    meta["name"] = new
    _write_json(meta_path, meta)


def map_image_png(name: str) -> bytes:
    img = cv2.imread(map_pgm_path(name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise StoreError(f"Map image unreadable: {name}")
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise StoreError("PNG encode failed")
    return buf.tobytes()


def map_occupancy(name: str) -> Dict:
    """
    PGM → Int8 occupancy (ROS-style), row 0 = top of image.

    map_saver / slam_toolbox PGMs typically use:
      0   occupied, 254 free, 205 unknown
    """
    img = cv2.imread(map_pgm_path(name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise StoreError(f"Map image unreadable: {name}")
    h, w = img.shape
    occ = np.full((h, w), -1, dtype=np.int8)
    occ[img >= 250] = 0
    occ[img <= 50] = 100
    # Soft mid-grays (if any) → approximate occupancy 0–100
    mid = (img > 50) & (img < 250)
    if np.any(mid):
        # 254→0, 0→100 linear over the free/occ convention
        occ[mid] = np.clip(
            ((254 - img[mid].astype(np.int16)) * (100 / 254.0)).astype(np.int16),
            0, 100,
        ).astype(np.int8)
    meta = map_meta(name)
    return {
        "name": name,
        "width": int(w),
        "height": int(h),
        "resolution": float(meta.get("resolution", 0.05)),
        "origin": list(meta.get("origin", [0.0, 0.0, 0.0]))[:2],
        "version": int(meta.get("version", 0)),
        "occ_b64": base64.b64encode(occ.tobytes()).decode("ascii"),
        "map_to_world": meta.get("map_to_world"),
    }


# --- map editing with bounded undo history --------------------------------

def _history_dir(name: str) -> str:
    return os.path.join(map_dir(name), "history")


def _bump_version(name: str, delta_stack: Optional[List[int]] = None) -> Dict:
    meta_path = os.path.join(map_dir(name), "meta.json")
    meta = _read_json(meta_path)
    meta["version"] = int(meta.get("version", 0)) + 1
    if delta_stack is not None:
        meta["undo_stack"] = delta_stack
    _write_json(meta_path, meta)
    return meta


def _push_history(name: str):
    """Snapshot the current PGM before an edit; keep the last MAX_HISTORY."""
    hdir = _history_dir(name)
    os.makedirs(hdir, exist_ok=True)
    stamp = f"{time.time():.6f}"
    shutil.copyfile(map_pgm_path(name), os.path.join(hdir, stamp + ".pgm"))
    snaps = sorted(os.listdir(hdir))
    for old in snaps[:-MAX_HISTORY]:
        os.remove(os.path.join(hdir, old))
    # A new edit invalidates any redo snapshots.
    rdir = os.path.join(map_dir(name), "redo")
    shutil.rmtree(rdir, ignore_errors=True)


def apply_map_edit(name: str, ops: List[Dict]) -> Dict:
    """
    Apply brush strokes to the map PGM.

    op = {tool: "erase"|"wall", points: [[col,row],...], radius: px}
      erase -> paint free (254), wall -> paint occupied (0)
    """
    pgm = map_pgm_path(name)
    img = cv2.imread(pgm, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise StoreError(f"Map not found: {name}")

    _push_history(name)

    for op in ops:
        tool = op.get("tool")
        value = 254 if tool == "erase" else 0 if tool == "wall" else None
        if value is None:
            raise StoreError(f"Unknown edit tool: {tool}")
        radius = max(1, int(op.get("radius", 3)))
        pts = op.get("points") or []
        prev = None
        for p in pts:
            c, r = int(round(p[0])), int(round(p[1]))
            if prev is not None:
                cv2.line(img, prev, (c, r), value, thickness=radius * 2)
            cv2.circle(img, (c, r), radius, value, thickness=-1)
            prev = (c, r)

    cv2.imwrite(pgm, img)
    return _bump_version(name)


def undo_map_edit(name: str) -> Dict:
    hdir = _history_dir(name)
    snaps = sorted(os.listdir(hdir)) if os.path.isdir(hdir) else []
    if not snaps:
        raise StoreError("Nothing to undo")
    # Move current state to the redo stack.
    rdir = os.path.join(map_dir(name), "redo")
    os.makedirs(rdir, exist_ok=True)
    shutil.copyfile(map_pgm_path(name),
                    os.path.join(rdir, f"{time.time():.6f}.pgm"))
    latest = os.path.join(hdir, snaps[-1])
    shutil.move(latest, map_pgm_path(name))
    return _bump_version(name)


def redo_map_edit(name: str) -> Dict:
    rdir = os.path.join(map_dir(name), "redo")
    snaps = sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []
    if not snaps:
        raise StoreError("Nothing to redo")
    hdir = _history_dir(name)
    os.makedirs(hdir, exist_ok=True)
    shutil.copyfile(map_pgm_path(name),
                    os.path.join(hdir, f"{time.time():.6f}.pgm"))
    latest = os.path.join(rdir, snaps[-1])
    shutil.move(latest, map_pgm_path(name))
    return _bump_version(name)


def map_history_counts(name: str) -> Dict:
    hdir = _history_dir(name)
    rdir = os.path.join(map_dir(name), "redo")
    return {
        "undo": len(os.listdir(hdir)) if os.path.isdir(hdir) else 0,
        "redo": len(os.listdir(rdir)) if os.path.isdir(rdir) else 0,
    }


# ---------------------------------------------------------------------------
# Racelines
# ---------------------------------------------------------------------------

def raceline_dir(name: str) -> str:
    return os.path.join(RACELINES_DIR, validate_name(name))


def raceline_csv_path(name: str) -> str:
    return os.path.join(raceline_dir(name), "waypoints.csv")


def raceline_exists(name: str) -> bool:
    try:
        return os.path.isfile(raceline_csv_path(name))
    except StoreError:
        return False


def save_raceline(name: str, frenet: np.ndarray, *, map_name: str,
                  params: Dict, anchors: Optional[List] = None) -> Dict:
    """frenet: (N,5) array of [s, x, y, theta, velocity]."""
    validate_name(name)
    ensure_dirs()
    d = raceline_dir(name)
    os.makedirs(d, exist_ok=True)

    rows = np.asarray(frenet, dtype=float)
    with open(raceline_csv_path(name), "w", newline="") as f:
        f.write("s,x,y,theta,velocity\n")
        for r in rows:
            f.write(",".join(f"{v:.6f}" for v in r) + "\n")

    meta = {
        "name": name,
        "map": map_name,
        "created": time.time(),
        "params": params,
        "anchors": anchors,
        "stats": {
            "length_m": float(rows[-1, 0]) if len(rows) else 0.0,
            "n_points": int(len(rows)),
            "v_min": float(rows[:, 4].min()) if len(rows) else 0.0,
            "v_max": float(rows[:, 4].max()) if len(rows) else 0.0,
            "lap_time_est": _lap_time_estimate(rows),
        },
    }
    _write_json(os.path.join(d, "meta.json"), meta)
    return meta


def _lap_time_estimate(rows: np.ndarray) -> float:
    if len(rows) < 2:
        return 0.0
    ds = np.diff(rows[:, 0])
    v = np.maximum(rows[:-1, 4], 0.1)
    return float(np.sum(ds / v))


def raceline_meta(name: str) -> Dict:
    d = raceline_dir(name)
    if not os.path.isfile(raceline_csv_path(name)):
        raise StoreError(f"Raceline not found: {name}")
    meta = _read_json(os.path.join(d, "meta.json"))
    meta.setdefault("name", name)
    return meta


def load_raceline(name: str) -> np.ndarray:
    path = raceline_csv_path(name)
    if not os.path.isfile(path):
        raise StoreError(f"Raceline not found: {name}")
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            try:
                rows.append([float(x) for x in parts[:5]])
            except ValueError:
                continue  # header
    return np.array(rows, dtype=float)


def list_racelines() -> List[Dict]:
    ensure_dirs()
    out = []
    for name in sorted(os.listdir(RACELINES_DIR)):
        if _NAME_RE.match(name) and raceline_exists(name):
            try:
                out.append(raceline_meta(name))
            except StoreError:
                continue
    out.sort(key=lambda m: m.get("created", 0), reverse=True)
    return out


def delete_raceline(name: str):
    d = raceline_dir(name)
    if not os.path.isdir(d):
        raise StoreError(f"Raceline not found: {name}")
    shutil.rmtree(d)
