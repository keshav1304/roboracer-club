"""Raceline editor: a small web GUI for drawing racelines on saved SLAM maps.

    python3 raceline_editor.py [--port 8090]

Open http://<car-ip>:8090 from any browser on the same network. Pick a map,
click waypoints around the track, and save. The editor fits a closed spline
through the waypoints and writes, next to the map:

    <maps>/<name>_raceline.csv     x,y in meters, map frame, 5 cm spacing,
                                   in driving order (read by pure_pursuit.py)
    <maps>/<name>_waypoints.json   the clicked waypoints, so you can re-edit

No ROS needed; it only reads the map files.
"""

import argparse
import io
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
MAPS_DIR = Path(
    os.environ.get(
        "RACING_MAPS_DIR",
        Path.home() / "osracer_ws/src/osracer/osracer_slam/maps",
    )
)
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


def list_maps():
    out = []
    for y in sorted(MAPS_DIR.glob("*.yaml")):
        name = y.stem
        out.append({
            "name": name,
            "has_raceline": (MAPS_DIR / f"{name}_raceline.csv").exists(),
            "mtime": y.stat().st_mtime,
        })
    return out


def map_meta(name):
    meta = yaml.safe_load((MAPS_DIR / f"{name}.yaml").read_text())
    img = Image.open(MAPS_DIR / meta["image"])
    origin = meta.get("origin", [0, 0, 0])
    return {
        "name": name,
        "resolution": float(meta["resolution"]),
        "origin": [float(origin[0]), float(origin[1]), float(origin[2]) if len(origin) > 2 else 0.0],
        "width": img.width,
        "height": img.height,
        "negate": int(meta.get("negate", 0)),
    }


def map_png(name):
    meta = yaml.safe_load((MAPS_DIR / f"{name}.yaml").read_text())
    img = Image.open(MAPS_DIR / meta["image"]).convert("L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_raceline(name):
    wp = MAPS_DIR / f"{name}_waypoints.json"
    if wp.exists():
        return json.loads(wp.read_text())
    return {"waypoints": []}


def save_raceline(name, body):
    waypoints = body.get("waypoints") or []
    raceline = body.get("raceline") or []
    if len(waypoints) < 3 or len(raceline) < 3:
        raise ValueError("need at least 3 waypoints")
    for p in waypoints + raceline:
        if len(p) != 2 or not all(isinstance(v, (int, float)) for v in p):
            raise ValueError("points must be [x, y] numbers")
    lines = [f"# raceline for map '{name}': x_m, y_m (map frame), driving order"]
    lines += [f"{x:.4f},{y:.4f}" for x, y in raceline]
    (MAPS_DIR / f"{name}_raceline.csv").write_text("\n".join(lines) + "\n")
    (MAPS_DIR / f"{name}_waypoints.json").write_text(
        json.dumps({"waypoints": waypoints, "params": body.get("params", {})}, indent=1)
    )
    return {"ok": True, "csv": str(MAPS_DIR / f"{name}_raceline.csv"), "points": len(raceline)}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _name(self, part):
        if not NAME_RE.match(part) or not (MAPS_DIR / f"{part}.yaml").exists():
            raise FileNotFoundError(part)
        return part

    def do_GET(self):
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        try:
            if not parts:
                return self._send(200, (HERE / "editor.html").read_bytes(), "text/html; charset=utf-8")
            if parts == ["api", "maps"]:
                return self._send(200, list_maps())
            if len(parts) >= 3 and parts[:2] == ["api", "maps"]:
                name = self._name(parts[2])
                if len(parts) == 3:
                    return self._send(200, map_meta(name))
                if parts[3] == "image.png":
                    return self._send(200, map_png(name), "image/png")
                if parts[3] == "raceline":
                    return self._send(200, load_raceline(name))
            self._send(404, {"error": "not found"})
        except FileNotFoundError as e:
            self._send(404, {"error": f"no map {e}"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": str(e)})

    def do_POST(self):
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        try:
            if len(parts) == 4 and parts[:2] == ["api", "maps"] and parts[3] == "raceline":
                name = self._name(parts[2])
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                return self._send(200, save_raceline(name, body))
            self._send(404, {"error": "not found"})
        except FileNotFoundError as e:
            self._send(404, {"error": f"no map {e}"})
        except ValueError as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8090)
    args = p.parse_args()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Raceline editor on http://0.0.0.0:{args.port}  (maps: {MAPS_DIR})")
    print("Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
