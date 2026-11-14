"""Read the default AutoDRIVE Unity track without modifying the application.

Rasterize the collision mesh's ground footprint at 2 cm, keep the connected
driving area at Spawn 1, and build a periodic reference through the scene's
ordered checkpoint centres and forward directions. Unity (X,Z) becomes gym
(x,y), with metres preserved. This imports geometry, not Unity vehicle physics.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial.transform import Rotation
import yaml

from import_iros_track import HERE, geometry, import_saved_track

NAME = "autodrive-default"
APP = Path("AutoDRIVE Simulator.app")
EXPORT = HERE / "imported_tracks" / NAME
VERSION = 1
RESOLUTION = 0.02


def world_matrix(transform_id, transforms):
    """Unity local-to-world hierarchy: translation × rotation × scale."""
    t = transforms[transform_id]
    local = np.eye(4)
    local[:3, :3] = Rotation.from_quat([t["m_LocalRotation"][k] for k in "xyzw"]).as_matrix()
    local[:3, :3] @= np.diag([t["m_LocalScale"][k] for k in "xyz"])
    local[:3, 3] = [t["m_LocalPosition"][k] for k in "xyz"]
    parent = t["m_Father"]["m_PathID"]
    return world_matrix(parent, transforms) @ local if parent else local


def read_scene(app):
    import UnityPy
    from UnityPy.helpers.MeshHelper import MeshHandler
    data = app / "Contents" / "Resources" / "Data"
    env = UnityPy.load(str(data / "level0"), str(data / "sharedassets0.assets"))
    scene = next(o.assets_file for o in env.objects if o.assets_file.name == "level0")
    transforms, objects = {}, {}
    for ident, obj in scene.objects.items():
        if obj.type.name == "GameObject":
            objects[ident] = obj.read_typetree()
        elif obj.type.name == "Transform":
            transforms[ident] = obj.read_typetree()
    object_transform = {t["m_GameObject"]["m_PathID"]: i for i, t in transforms.items()}

    def active(ident):
        t = transforms[object_transform[ident]]
        parent = t["m_Father"]["m_PathID"]
        return objects[ident]["m_IsActive"] and (not parent or active(transforms[parent]["m_GameObject"]["m_PathID"]))

    candidates = []
    for ident, obj in objects.items():
        if not obj["m_Name"].endswith(" Track") or not active(ident):
            continue
        for component in obj["m_Component"]:
            reader = scene.objects[component["component"]["m_PathID"]]
            if reader.type.name == "MeshCollider":
                collider = reader.read()
                if collider.m_Enabled and not collider.m_IsTrigger:
                    candidates.append((ident, collider))
    if len(candidates) != 1:
        raise ValueError(f"Expected one default active track collider, found {len(candidates)}.")
    ident, collider = candidates[0]
    track_name = objects[ident]["m_Name"]
    matrix = world_matrix(object_transform[ident], transforms)
    mesh = collider.m_Mesh.read()
    handler = MeshHandler(mesh)
    handler.process()
    vertices = np.asarray(handler.m_Vertices, float)[:, :3]
    vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    triangles = np.vstack([np.asarray(t, int) + (sub.baseVertex or 0)
                           for sub, t in zip(mesh.m_SubMeshes, handler.get_triangles())])

    def group(suffix):
        group_name = track_name.removesuffix(" Track") + suffix
        ids = [i for i, obj in objects.items() if obj["m_Name"] == group_name and active(i)]
        if len(ids) != 1:
            raise ValueError(f"Cannot identify active scene group {group_name}.")
        return transforms[object_transform[ids[0]]]["m_Children"]

    checkpoints = {}
    for child in group(" Checkpoints"):
        tid = child["m_PathID"]
        name = objects[transforms[tid]["m_GameObject"]["m_PathID"]]["m_Name"]
        if name.isdigit():
            w = world_matrix(tid, transforms)
            heading = (w[:3, :3] @ [0, 0, 1])[[0, 2]]
            checkpoints[int(name)] = (w[[0, 2], 3], heading / np.linalg.norm(heading))
    if sorted(checkpoints) != list(range(len(checkpoints))) or len(checkpoints) < 4:
        raise ValueError("The scene checkpoint sequence is incomplete.")
    spawn = None
    for child in group(" Spawn Points"):
        tid = child["m_PathID"]
        if objects[transforms[tid]["m_GameObject"]["m_PathID"]]["m_Name"] == "Spawn 1":
            w = world_matrix(tid, transforms)
            forward = (w[:3, :3] @ [0, 0, 1])[[0, 2]]
            spawn = [*w[[0, 2], 3], float(np.arctan2(forward[1], forward[0]))]
    if spawn is None:
        raise ValueError("Cannot locate the default track's Spawn 1.")
    return vertices, triangles, np.array([checkpoints[i][0] for i in sorted(checkpoints)]), np.array([checkpoints[i][1] for i in sorted(checkpoints)]), {
        "track_name": track_name, "mesh_name": mesh.m_Name, "unity_to_world": matrix.tolist(),
        "spawn": spawn, "unity_version": scene.unity_version, "triangles": len(triangles),
    }


def rasterize(vertices, triangles, spawn, resolution=RESOLUTION):
    """Project the real barrier footprint; no outline traced from a picture."""
    ground = vertices[:, [0, 2]]
    origin = np.floor((ground.min(axis=0) - 1.0) / resolution) * resolution
    size = np.ceil((ground.max(axis=0) + 1.0 - origin) / resolution).astype(int)
    pixels = (ground - origin) / resolution
    pixels[:, 1] = size[1] - 1 - pixels[:, 1]
    wall_image = Image.new("1", tuple(size))
    draw = ImageDraw.Draw(wall_image)
    for triangle in triangles:
        draw.polygon([tuple(p) for p in pixels[triangle]], fill=1)
    regions, _ = label(~np.asarray(wall_image))
    cell = np.floor((np.asarray(spawn[:2]) - origin) / resolution).astype(int)
    ident = regions[size[1] - 1 - cell[1], cell[0]]
    if ident == 0:
        raise ValueError("Spawn 1 intersects the extracted barriers.")
    free = regions == ident
    if free[0].any() or free[-1].any() or free[:, 0].any() or free[:, -1].any():
        raise ValueError("The default driving area is not enclosed by the barriers.")
    return free, origin


def reference_line(anchors, headings, free, origin):
    closed = np.vstack([anchors, anchors[:1]])
    station = np.r_[0.0, np.linalg.norm(np.diff(closed, axis=0), axis=1).cumsum()]
    curve = CubicHermiteSpline(station, closed, np.vstack([headings, headings[:1]]))
    dense = curve(np.linspace(0, station[-1], 5000))
    distance = np.r_[0.0, np.linalg.norm(np.diff(dense, axis=0), axis=1).cumsum()]
    samples = np.linspace(0, distance[-1], int(np.ceil(distance[-1] / 0.05)), endpoint=False)
    xy = CubicSpline(distance, dense)(samples)
    yaw, ss, curvature = geometry(xy)
    clearance = distance_transform_edt(free) * RESOLUTION
    cells = np.floor((xy - origin) / RESOLUTION).astype(int)
    if clearance[len(free) - 1 - cells[:, 1], cells[:, 0]].min() < 0.45:
        raise ValueError("Generated reference has insufficient clearance from the real walls.")
    # Conservative curvature speed profile, with braking before each corner.
    speed = np.minimum(4.5, np.sqrt(2.0 / np.maximum(np.abs(curvature), 0.02)))
    ds = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    for _ in range(4):
        for i in range(len(speed) - 1, -1, -1):
            speed[i] = min(speed[i], np.sqrt(speed[(i + 1) % len(speed)] ** 2 + 2 * 2.5 * ds[i]))
        for i in range(len(speed)):
            speed[(i + 1) % len(speed)] = min(speed[(i + 1) % len(speed)], np.sqrt(speed[i] ** 2 + 2 * 2.0 * ds[i]))
    return np.column_stack([ss, xy, yaw, speed])


def ensure_unity_track(app=None, force=False):
    if app is None and not force and (EXPORT / "maps" / NAME / "map.yaml").exists():
        return import_saved_track(EXPORT, NAME, center_mode="reference")
    metadata_file = EXPORT / "source.json"
    if app is None:
        app = json.loads(metadata_file.read_text())["app"] if metadata_file.exists() else APP
    app = Path(app)
    data = app / "Contents" / "Resources" / "Data"
    fingerprint = {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                   for name in ("level0", "sharedassets0.assets", "sharedassets0.assets.resS")}
    fingerprint["import_version"] = VERSION
    if not force and metadata_file.exists():
        if json.loads(metadata_file.read_text())["fingerprint"] == fingerprint:
            return import_saved_track(EXPORT, NAME, center_mode="reference")
    vertices, triangles, anchors, headings, metadata = read_scene(app)
    free, origin = rasterize(vertices, triangles, metadata["spawn"])
    line = reference_line(anchors, headings, free, origin)
    map_dir = EXPORT / "maps" / NAME
    race_dir = EXPORT / "racelines" / f"{NAME}-line"
    map_dir.mkdir(parents=True, exist_ok=True)
    race_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((free * 255).astype(np.uint8)).save(map_dir / "map.pgm")
    spec = {"image": "map.pgm", "resolution": RESOLUTION, "origin": [*origin.tolist(), 0],
            "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.25}
    (map_dir / "map.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    np.savetxt(race_dir / "waypoints.csv", line, delimiter=",", header="s,x,y,theta,velocity", comments="", fmt="%.8f")
    metadata.update({"app": str(app), "fingerprint": fingerprint, "coordinate_mapping": "gym x=Unity X, gym y=Unity Z; metres",
                     "resolution": RESOLUTION, "checkpoint_positions": anchors.tolist(), "checkpoint_headings": headings.tolist(),
                     "reference_length": float(line[-1, 0]), "reference_points": len(line)})
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")
    return import_saved_track(EXPORT, NAME, center_mode="reference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, help="app path; defaults to the last imported app")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print("AutoDRIVE default track ready:", ensure_unity_track(args.app, args.force))
