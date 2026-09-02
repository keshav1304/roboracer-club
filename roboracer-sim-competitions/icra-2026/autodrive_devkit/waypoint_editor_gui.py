#!/usr/bin/env python3
"""
Interactive waypoint editor for AutoDRIVE racelines.

Loads a ROS map (YAML + PGM), click to place anchor points, fits a periodic
cubic spline (same family as generate_waypoints.py), then builds
[s, x, y, theta, velocity] using the same compute_frenet / velocity_profile
logic as generate_waypoints.py.

Run from autodrive_devkit (or pass absolute --map / --out):

    python3 waypoint_editor_gui.py --map comp_track.yaml --out comp_waypoints.csv

Loads ``comp_waypoints.csv`` by default as editable anchors; use ``--empty`` to start blank.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider
from scipy.interpolate import splprep, splev

_DEVKIT = os.path.dirname(os.path.abspath(__file__))
if _DEVKIT not in sys.path:
    sys.path.insert(0, _DEVKIT)

import generate_waypoints as gw  # noqa: E402


def load_xy_from_waypoints_csv(path: str) -> tuple[list[float], list[float]]:
    """Read comp_waypoints-style CSV (s, x, y, theta, velocity); return x,y lists. Skips header."""
    wx: list[float] = []
    wy: list[float] = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                float(row[0])
                x = float(row[1])
                y = float(row[2])
            except ValueError:
                continue
            wx.append(x)
            wy.append(y)
    return wx, wy


def nearest_free_pixel(col: float, row: float, free_mask: np.ndarray) -> tuple[int, int]:
    """Snap (col, row) to nearest free cell."""
    H, W = free_mask.shape
    c0, r0 = int(round(col)), int(round(row))
    if 0 <= r0 < H and 0 <= c0 < W and free_mask[r0, c0]:
        return c0, r0
    yy, xx = np.ogrid[:H, :W]
    d2 = (xx.astype(np.float64) - col) ** 2 + (yy.astype(np.float64) - row) ** 2
    d2 = d2.copy()
    d2[~free_mask] = np.inf
    if not np.any(np.isfinite(d2)):
        return max(0, min(W - 1, c0)), max(0, min(H - 1, r0))
    idx = int(np.argmin(d2))
    r, c = np.unravel_index(idx, d2.shape)
    return int(c), int(r)


def world_xy_to_col_row(wx: float, wy: float, img_h: int, res: float, origin: list) -> tuple[float, float]:
    col = (wx - origin[0]) / res
    row = (img_h - 1) - (wy - origin[1]) / res
    return col, row


def sample_closed_spline(
    world_pts: np.ndarray,
    n_out: int,
    smooth_s: float,
) -> np.ndarray | None:
    """Periodic cubic spline through world (x,y); same idea as generate_waypoints fit_closed_spline."""
    if world_pts.shape[0] < 4:
        return None
    x, y = world_pts[:, 0], world_pts[:, 1]
    try:
        tck, _ = splprep([x, y], s=smooth_s, per=True, k=3)
    except Exception:
        return None
    u = np.linspace(0, 1, n_out, endpoint=False)
    sx, sy = splev(u, tck)
    return np.stack([sx, sy], axis=1)


def sample_open_spline(world_pts: np.ndarray, n_out: int, smooth_s: float) -> np.ndarray | None:
    if world_pts.shape[0] < 2:
        return None
    x, y = world_pts[:, 0], world_pts[:, 1]
    k = min(3, max(1, len(x) - 1))
    try:
        tck, _ = splprep([x, y], s=smooth_s, k=k)
    except Exception:
        return None
    u = np.linspace(0, 1, n_out)
    sx, sy = splev(u, tck)
    return np.stack([sx, sy], axis=1)


class WaypointEditor:
    def __init__(
        self,
        map_yaml: str,
        out_csv: str,
        *,
        vmin: float,
        vmax: float,
        alat: float,
        n_out: int,
        snap_free: bool,
        closed: bool,
        align_x: float,
        align_y: float,
        do_align: bool,
        initial_csv: str | None,
    ) -> None:
        self.out_csv = os.path.abspath(out_csv)
        self.map_yaml = os.path.abspath(map_yaml)
        self.img, self.res, self.origin, occ_t, free_t, neg = gw.load_map(self.map_yaml)
        self.img_h, self.img_w = self.img.shape
        self.free_mask = gw.build_free_mask(self.img, occ_t, free_t, neg)
        self.vmin, self.vmax, self.alat = vmin, vmax, alat
        self.n_out = n_out
        self.snap_free = snap_free
        self.closed = closed
        self.align_x, self.align_y = align_x, align_y
        self.do_align = do_align

        self._wx: list[float] = []
        self._wy: list[float] = []
        self._initial_loaded_from: str | None = None
        if initial_csv and os.path.isfile(initial_csv):
            try:
                self._wx, self._wy = load_xy_from_waypoints_csv(initial_csv)
                self._initial_loaded_from = os.path.abspath(initial_csv)
            except OSError as e:
                print(f"Warning: could not read initial CSV {initial_csv}: {e}", file=sys.stderr)

        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(left=0.08, bottom=0.18, right=0.72, top=0.94)

        ox, oy = self.origin[0], self.origin[1]
        extent = (
            ox,
            ox + (self.img_w - 1) * self.res,
            oy,
            oy + (self.img_h - 1) * self.res,
        )
        self.ax.imshow(self.img, cmap="gray", extent=extent, origin="upper", aspect="equal")
        self.ax.set_xlabel("world x (m)")
        self.ax.set_ylabel("world y (m)")
        self.ax.set_title(
            "Click: add waypoint  |  Closed: ≥4 anchors  |  "
            "Spline + Frenet/velocity same as generate_waypoints.py"
        )

        self._scatter = self.ax.scatter([], [], s=80, c="lime", zorder=5, edgecolors="black")
        self._line, = self.ax.plot([], [], "c-", lw=2, alpha=0.85, zorder=4)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        ax_clr = self.fig.add_axes([0.76, 0.82, 0.2, 0.05])
        ax_undo = self.fig.add_axes([0.76, 0.75, 0.2, 0.05])
        ax_save = self.fig.add_axes([0.76, 0.68, 0.2, 0.05])
        self._btn_clear = Button(ax_clr, "Clear")
        self._btn_undo = Button(ax_undo, "Undo")
        self._btn_save = Button(ax_save, "Save CSV")
        self._btn_clear.on_clicked(self._clear)
        self._btn_undo.on_clicked(self._undo)
        self._btn_save.on_clicked(self._save)

        rax = self.fig.add_axes([0.76, 0.52, 0.2, 0.12])
        self._chk = CheckButtons(
            rax,
            ["Closed loop", "Snap to drivable", "Align s=0 to ref"],
            [closed, snap_free, do_align],
        )
        self._chk.on_clicked(self._on_check)

        ax_smooth = self.fig.add_axes([0.76, 0.38, 0.2, 0.04])
        self._slider_smooth = Slider(ax_smooth, "spline s", 0.0, max(50.0, n_out * 0.5), valinit=2.0, valfmt="%.2f")
        self._slider_smooth.on_changed(lambda _v: self._redraw_preview())

        ax_n = self.fig.add_axes([0.76, 0.30, 0.2, 0.04])
        self._slider_n = Slider(ax_n, "n samples", 50, 2000, valinit=float(n_out), valfmt="%d")
        self._slider_n.on_changed(lambda _v: self._redraw_preview())

        self._status = self.fig.text(0.08, 0.06, "", fontsize=10)
        if self._initial_loaded_from:
            self._set_status(
                f"Loaded {len(self._wx)} points from {os.path.basename(self._initial_loaded_from)} — "
                "click to add, Undo/Clear to edit."
            )
        else:
            self._set_status("Click map to add anchors. Adjust spline s if fit is stiff/loose.")
        self._redraw_preview()

    def _on_check(self, _label: str) -> None:
        s = self._chk.get_status()
        self.closed, self.snap_free, self.do_align = bool(s[0]), bool(s[1]), bool(s[2])
        self._redraw_preview()

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        wx, wy = float(event.xdata), float(event.ydata)
        col, row = world_xy_to_col_row(wx, wy, self.img_h, self.res, self.origin)
        if self.snap_free:
            col, row = nearest_free_pixel(col, row, self.free_mask)
            wxy = gw.pixels_to_world(np.array([[col, row]]), self.img_h, self.res, self.origin)[0]
            wx, wy = float(wxy[0]), float(wxy[1])
        self._wx.append(wx)
        self._wy.append(wy)
        self._redraw_preview()
        self._set_status(f"{len(self._wx)} waypoints")

    def _clear(self, _event=None) -> None:
        self._wx.clear()
        self._wy.clear()
        self._redraw_preview()
        self._set_status("Cleared")

    def _undo(self, _event=None) -> None:
        if self._wx:
            self._wx.pop()
            self._wy.pop()
        self._redraw_preview()
        self._set_status(f"{len(self._wx)} waypoints")

    def _set_status(self, msg: str) -> None:
        self._status.set_text(msg)
        self.fig.canvas.draw_idle()

    def _redraw_preview(self) -> None:
        if not self._wx:
            self._scatter.set_offsets(np.empty((0, 2)))
            self._line.set_data([], [])
            self.fig.canvas.draw_idle()
            return
        pts = np.column_stack([self._wx, self._wy])
        self._scatter.set_offsets(pts)

        smooth_s = float(self._slider_smooth.val)
        n_out = int(self._slider_n.val)

        if self.closed and len(self._wx) >= 4:
            sampled = sample_closed_spline(pts, n_out, smooth_s)
        elif not self.closed and len(self._wx) >= 2:
            sampled = sample_open_spline(pts, n_out, smooth_s)
        else:
            sampled = None

        if sampled is not None:
            self._line.set_data(sampled[:, 0], sampled[:, 1])
        else:
            self._line.set_data([], [])
        self.fig.canvas.draw_idle()

    def _save(self, _event=None) -> None:
        if len(self._wx) < 2:
            self._set_status("Need at least 2 points (4 for closed loop).")
            return
        pts = np.column_stack([self._wx, self._wy])
        smooth_s = float(self._slider_smooth.val)
        n_out = int(self._slider_n.val)

        if self.closed:
            if len(self._wx) < 4:
                self._set_status("Closed loop needs ≥ 4 anchors.")
                return
            sampled = sample_closed_spline(pts, n_out, smooth_s)
        else:
            sampled = sample_open_spline(pts, n_out, smooth_s)

        if sampled is None:
            self._set_status("Spline fit failed — adjust smoothing or points.")
            return

        world_pts = np.asarray(sampled, dtype=np.float64)
        if self.do_align and self.closed:
            world_pts = gw.roll_closed_polyline_to_reference(world_pts, self.align_x, self.align_y)

        frenet = gw.compute_frenet(world_pts, self.vmin, self.vmax, self.alat)

        out_dir = os.path.dirname(self.out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["s", "x", "y", "theta", "velocity"])
            w.writerows(frenet.tolist())

        self._set_status(f"Saved {len(frenet)} rows → {self._path_tail(self.out_csv)}")

    @staticmethod
    def _path_tail(p: str, n: int = 48) -> str:
        return p if len(p) <= n else "…" + p[-n:]

    def show(self) -> None:
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description="Click-to-edit raceline → comp_waypoints-style CSV")
    p.add_argument("--map", default="comp_track.yaml", help="ROS map YAML (PGM path from yaml)")
    p.add_argument("--out", default="comp_waypoints.csv", help="Output CSV path")
    p.add_argument("--vmin", type=float, default=0.2)
    p.add_argument("--vmax", type=float, default=6.5)
    p.add_argument("--alat", type=float, default=3.5)
    p.add_argument("--n-out", type=int, default=800, dest="n_out")
    p.add_argument("--no-snap", action="store_true", help="Start with snap-to-drivable off")
    p.add_argument("--open", action="store_true", help="Start as open polyline (not closed loop)")
    p.add_argument("--no-align", action="store_true", help="Start with s=0 alignment off")
    p.add_argument("--align-x", type=float, default=0.0, help="World x for s=0 alignment when enabled")
    p.add_argument("--align-y", type=float, default=0.0, help="World y for s=0 alignment when enabled")
    p.add_argument(
        "--initial",
        default="comp_waypoints.csv",
        metavar="CSV",
        help="Existing waypoints CSV (s,x,y,...) to load as anchors (default: comp_waypoints.csv)",
    )
    p.add_argument("--empty", action="store_true", help="Do not load --initial; start with no anchors")
    args = p.parse_args()

    map_path = args.map
    if not os.path.isabs(map_path):
        map_path = os.path.join(_DEVKIT, map_path)
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(_DEVKIT, out_path)

    initial_csv: str | None = None
    if not args.empty and args.initial:
        initial_csv = args.initial
        if not os.path.isabs(initial_csv):
            initial_csv = os.path.join(_DEVKIT, initial_csv)

    editor = WaypointEditor(
        map_path,
        out_path,
        vmin=args.vmin,
        vmax=args.vmax,
        alat=args.alat,
        n_out=args.n_out,
        snap_free=not args.no_snap,
        closed=not args.open,
        align_x=args.align_x,
        align_y=args.align_y,
        do_align=not args.no_align,
        initial_csv=initial_csv,
    )
    editor.show()


if __name__ == "__main__":
    main()
