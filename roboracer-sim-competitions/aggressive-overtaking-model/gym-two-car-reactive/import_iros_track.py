"""Import the saved IROS occupancy grid and raceline into f1tenth_gym.

Run with the baseline's Python. No ROS, Unity, or map download is needed.
The source map frame, pixels, track direction and waypoint speeds are
preserved. Only the file formats are adapted.
"""
import hashlib
import json
import pathlib

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter1d

HERE = pathlib.Path(__file__).resolve().parent
NAME = "track-20260731"
SOURCE = HERE.parent / "iros-2026" / "autodrive_devkit"
DEST = HERE / "f1tenth_gym" / "maps" / NAME
VERSION = 1


def geometry(xy):
    """Periodic tangent, arc length, and curvature; no seam discontinuity."""
    delta = np.roll(xy, -1, axis=0) - xy
    ds = np.linalg.norm(delta, axis=1)
    tangent = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    yaw = np.arctan2(tangent[:, 1], tangent[:, 0])
    dyaw = (np.roll(yaw, -1) - np.roll(yaw, 1) + np.pi) % (2 * np.pi) - np.pi
    curvature = dyaw / (ds + np.roll(ds, 1))
    return yaw, np.r_[0.0, np.cumsum(ds)[:-1]], curvature


def ensure_iros_track():
    return import_saved_track(SOURCE, NAME)


def import_saved_track(source, name, center_mode="midpoint"):
    """Convert a local ROS map plus waypoint CSV to the gym's file formats."""
    source = pathlib.Path(source)
    dest = HERE / "f1tenth_gym" / "maps" / name
    inputs = [source / "maps" / name / f for f in ("map.yaml", "map.pgm")]
    inputs.append(source / "racelines" / f"{name}-line" / "waypoints.csv")
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    manifest = {"import_version": VERSION, "source_sha256": hashes, "center_mode": center_mode}
    stamp = dest / "import.json"
    outputs = [dest / f"{name}{suffix}" for suffix in (".png", "_map.yaml", "_centerline.csv", "_raceline.csv")]
    if stamp.exists() and all(p.exists() for p in outputs):
        if json.loads(stamp.read_text()) == manifest:
            return dest

    spec = yaml.safe_load(inputs[0].read_text())
    if spec["origin"][2] != 0 or spec.get("negate", 0):
        raise ValueError("This importer expects an unrotated, non-negated ROS map.")
    image = np.asarray(Image.open(inputs[1]).convert("L"))
    # Match the gym's occupancy threshold exactly; retain the original pixels.
    # In this ROS map 0 is a wall, 254 is free, and 205 is unobserved.
    free = image > 128
    resolution = float(spec["resolution"])
    origin = np.asarray(spec["origin"][:2], float)
    height, width = free.shape

    def is_free(points):
        pixels = np.floor((points - origin) / resolution).astype(int)
        cols, rows = pixels[..., 0], height - 1 - pixels[..., 1]
        valid = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
        return valid & free[np.clip(rows, 0, height - 1), np.clip(cols, 0, width - 1)]

    def widths(xy, yaw):
        normals = np.column_stack([-np.sin(yaw), np.cos(yaw)])
        distances = np.arange(0, max(height, width) * resolution, resolution / 4)
        out = []
        for sign in (-1, 1):
            samples = xy[:, None, :] + sign * normals[:, None, :] * distances[None, :, None]
            blocked = ~is_free(samples)
            if not blocked.any(axis=1).all():
                raise ValueError("Unbounded track cross-section in source map.")
            out.append(distances[blocked.argmax(axis=1)])
        return out  # right, left

    waypoints = np.genfromtxt(inputs[2], delimiter=",", names=True)
    race_xy = np.column_stack([waypoints["x"], waypoints["y"]])
    if not is_free(race_xy).all():
        raise ValueError("The source raceline intersects a mapped wall.")
    yaw, ss, curvature = geometry(race_xy)
    wr, wl = widths(race_xy, yaw)
    # Use the cross-section midpoints as a centre reference for spawns and
    # lateral widths, keeping the original raceline separately for driving.
    if center_mode == "reference":
        # Checkpoint-guided routes can include open-ended internal barriers.
        # Keep their valid reference instead of averaging across another lane.
        center_xy = race_xy.copy()
    elif center_mode == "midpoint":
        offset = gaussian_filter1d((wl - wr) / 2, sigma=5, mode="wrap")
        normals = np.column_stack([-np.sin(yaw), np.cos(yaw)])
        center_xy = race_xy + offset[:, None] * normals
        center_xy = gaussian_filter1d(center_xy, sigma=2, axis=0, mode="wrap")
    else:
        raise ValueError(f"Unknown centre-reference mode: {center_mode}")
    center_yaw, _, _ = geometry(center_xy)
    wr, wl = widths(center_xy, center_yaw)
    if not is_free(center_xy).all() or np.minimum(wr, wl).min() < 0.31 / 2 + 0.05:
        raise ValueError("Imported centre reference has insufficient car clearance.")
    velocity = waypoints["velocity"]
    acceleration = np.gradient(velocity, ss) * velocity
    dest.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(outputs[0])
    gym_spec = {k: spec[k] for k in ("resolution", "origin", "negate", "occupied_thresh", "free_thresh")}
    gym_spec["image"] = f"{name}.png"
    outputs[1].write_text(yaml.safe_dump(gym_spec, sort_keys=False))
    # This repo's TrackModel reads right then left (as in Spielberg).
    np.savetxt(outputs[2], np.column_stack([center_xy, wr, wl]), delimiter=",",
               header="x,y,width_right,width_left", fmt="%.8f")
    np.savetxt(outputs[3], np.column_stack([ss, race_xy, yaw, curvature, velocity, acceleration]),
               delimiter=";", header="s;x;y;yaw;curvature;speed;acceleration", fmt="%.8f")
    stamp.write_text(json.dumps(manifest, indent=2) + "\n")
    return dest


def configure_iros_renderer(env):
    """Correct the gym renderer's width/height swap for this rectangular map.

    Keep this adaptation local to the imported track; other maps and the
    installed engine are unchanged. Start with the entire circuit in view.
    """
    import cv2
    import pygame
    renderer = env.unwrapped.renderer
    if renderer is None:
        return env
    image = env.unwrapped.track.occupancy_map
    height, width = image.shape
    overview_zoom = (renderer.render_spec.window_size - 100) / max(height, width)
    for key, zoom in (("map", overview_zoom), ("car", overview_zoom * 2)):
        resized = cv2.resize(image, (round(width * zoom), round(height * zoom)),
                             interpolation=cv2.INTER_NEAREST)
        # Pygame surfaces take (x, y); occupancy_map is already bottom-up.
        rgb = np.repeat(resized.T[:, :, None], 3, axis=2)
        obj = renderer.map_renderers[key]
        obj.track_map = rgb
        obj.map_surface = pygame.surfarray.make_surface(rgb)
        renderer.map_canvases[key] = pygame.Surface(rgb.shape[:2])
        renderer.ppus[key] = height / resized.shape[0]
    renderer.active_map_renderer = "map"
    renderer.follow_agent_flag = False
    renderer.agent_to_follow = None
    return env


if __name__ == "__main__":
    print("IROS track ready:", ensure_iros_track())
