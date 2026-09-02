#!/usr/bin/env python3
"""
Interactive explorer for ``generate_waypoints.py`` racelines.

Loads the competition map, shows the optimised raceline and velocity profile.
Sliders tune parameters; changes that only affect sampling / Frenet / velocity
update instantly. Changes that affect CMA-ES (margin, budget, σ₀, control
count, erosion, or spline smooth used inside the optimiser) debounce and
re-run optimisation in a background thread.

Run from ``autodrive_devkit``:

    python3 raceline_explorer_gui.py --map comp_track.yaml

Requires: matplotlib, numpy, scipy, cma, opencv-python-headless, scikit-image,
pyyaml (same stack as ``generate_waypoints.py``).
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

_DEVKIT = os.path.dirname(os.path.abspath(__file__))
if _DEVKIT not in sys.path:
    sys.path.insert(0, _DEVKIT)

import generate_waypoints as gw  # noqa: E402


def _slider_axes(fig, y: float) -> Any:
    return fig.add_axes([0.64, y, 0.33, 0.026])


class RacelineExplorer:
    def __init__(
        self,
        map_yaml: str,
        *,
        debounce_ms: float,
        skip_initial_opt: bool,
    ) -> None:
        self.map_yaml = os.path.abspath(map_yaml)
        self.debounce_ms = float(debounce_ms)
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._heavy_gen = 0
        self._track_key: tuple[int, int] | None = None
        self._ctx: dict | None = None
        self._best: np.ndarray | None = None
        self._last_fbest: float | None = None
        self._busy = False

        self.fig = plt.figure(figsize=(14, 9))
        try:
            self.fig.canvas.manager.set_window_title("Raceline parameter explorer")
        except Exception:
            pass
        self.ax_map = self.fig.add_axes([0.06, 0.24, 0.54, 0.70])
        self.ax_vel = self.fig.add_axes([0.06, 0.06, 0.54, 0.14])

        y = 0.02
        dy = 0.068
        self.sl_erode = Slider(
            _slider_axes(self.fig, y),
            "erode",
            1.0,
            3.0,
            valinit=2.0,
            valstep=1.0,
        )
        y += dy
        self.sl_nctrl = Slider(
            _slider_axes(self.fig, y),
            "n_ctrl",
            30.0,
            120.0,
            valinit=80.0,
            valstep=5.0,
        )
        y += dy
        self.sl_margin = Slider(
            _slider_axes(self.fig, y),
            "margin (m)",
            0.08,
            0.42,
            valinit=0.25,
        )
        y += dy
        self.sl_budget = Slider(
            _slider_axes(self.fig, y),
            "CMA budget",
            3000.0,
            35000.0,
            valinit=12000.0,
            valstep=500.0,
        )
        y += dy
        self.sl_sigma0 = Slider(
            _slider_axes(self.fig, y),
            "sigma0 (px)",
            0.4,
            4.5,
            valinit=2.0,
        )
        y += dy
        self.sl_spl_cma = Slider(
            _slider_axes(self.fig, y),
            "spl (CMA obj)",
            0.0,
            100.0,
            valinit=0.0,
        )
        y += dy
        self.sl_spl_out = Slider(
            _slider_axes(self.fig, y),
            "spl (output)",
            0.0,
            120.0,
            valinit=0.0,
        )
        y += dy
        self.sl_frenet = Slider(
            _slider_axes(self.fig, y),
            "frenet σ",
            0.0,
            5.0,
            valinit=0.0,
        )
        y += dy
        self.sl_vmin = Slider(
            _slider_axes(self.fig, y),
            "v_min",
            0.15,
            1.2,
            valinit=0.5,
        )
        y += dy
        self.sl_vmax = Slider(
            _slider_axes(self.fig, y),
            "v_max",
            2.0,
            8.0,
            valinit=6.5,
        )
        y += dy
        self.sl_alat = Slider(
            _slider_axes(self.fig, y),
            "a_lat",
            1.5,
            7.0,
            valinit=3.5,
        )
        y += dy
        self.sl_nout = Slider(
            _slider_axes(self.fig, y),
            "n_out",
            300.0,
            1400.0,
            valinit=800.0,
            valstep=50.0,
        )

        print("Loading map and centreline (may take a few seconds) …")
        self._reload_track(
            int(round(self.sl_erode.val)),
            int(round(self.sl_nctrl.val)),
        )

        ctx = self._ctx
        assert ctx is not None
        img_h, img_w = ctx["img_h"], ctx["img_w"]
        res, origin = ctx["res"], ctx["origin"]
        ox, oy = origin[0], origin[1]
        extent = (
            ox,
            ox + (img_w - 1) * res,
            oy,
            oy + (img_h - 1) * res,
        )
        self.ax_map.imshow(
            ctx["img"], cmap="gray", extent=extent, origin="upper", aspect="equal"
        )
        self.ax_map.set_xlabel("x (m)")
        self.ax_map.set_ylabel("y (m)")
        (self._line_xy,) = self.ax_map.plot([], [], "c-", lw=2.2, alpha=0.9, zorder=5)
        self._status = self.fig.text(0.06, 0.97, "", fontsize=10, va="top")
        self.ax_vel.set_xlabel("s (m)")
        self.ax_vel.set_ylabel("v (m/s)")
        (self._line_v,) = self.ax_vel.plot([], [], "m-", lw=1.5)

        ax_btn = self.fig.add_axes([0.64, 0.955, 0.33, 0.035])
        self._btn = Button(ax_btn, "Re-run optimisation now")
        self._btn.on_clicked(self._on_rerun_clicked)

        self._heavy_timer = self.fig.canvas.new_timer(interval=int(self.debounce_ms))
        self._heavy_timer.single_shot = True
        self._heavy_timer.add_callback(self._fire_heavy_optimisation)

        self._pump = self.fig.canvas.new_timer(interval=120)
        self._pump.add_callback(self._pump_queue)
        self._pump.start()

        for sl in (
            self.sl_erode,
            self.sl_nctrl,
            self.sl_margin,
            self.sl_budget,
            self.sl_sigma0,
            self.sl_spl_cma,
        ):
            sl.on_changed(self._on_heavy_slider)

        for sl in (
            self.sl_spl_out,
            self.sl_frenet,
            self.sl_vmin,
            self.sl_vmax,
            self.sl_alat,
            self.sl_nout,
        ):
            sl.on_changed(lambda _v: self._refresh_fast())

        self._refresh_fast()
        self._set_status("Adjust sliders. CMA-ES re-runs after you stop moving geometry sliders.")

        if not skip_initial_opt:
            self._schedule_heavy(immediate=False)

    def _reload_track(self, erode: int, n_ctrl: int) -> None:
        ctx = gw.load_track_and_ctrl(self.map_yaml, int(erode), int(n_ctrl))
        with self._lock:
            self._ctx = ctx
            self._best = np.zeros(int(n_ctrl), dtype=np.float64)
            self._track_key = (int(erode), int(n_ctrl))

    def _set_status(self, msg: str) -> None:
        self._status.set_text(msg)
        self.fig.canvas.draw_idle()

    def _on_heavy_slider(self, _val: float) -> None:
        self._schedule_heavy(immediate=False)

    def _on_rerun_clicked(self, _event=None) -> None:
        self._schedule_heavy(immediate=True)

    def _schedule_heavy(self, immediate: bool) -> None:
        if immediate:
            self._heavy_timer.stop()
            self._fire_heavy_optimisation()
        else:
            self._heavy_timer.stop()
            self._heavy_timer.interval = int(self.debounce_ms)
            self._heavy_timer.start()

    def _fire_heavy_optimisation(self) -> None:
        self._heavy_gen += 1
        ticket = self._heavy_gen
        erode = int(round(self.sl_erode.val))
        n_ctrl = int(round(self.sl_nctrl.val))
        margin_m = float(self.sl_margin.val)
        budget = int(round(self.sl_budget.val))
        sigma0 = float(self.sl_sigma0.val)
        spl_cma = max(0.0, float(self.sl_spl_cma.val))

        self._queue.put(("busy", ticket))
        t = threading.Thread(
            target=self._cma_worker,
            args=(ticket, erode, n_ctrl, margin_m, budget, sigma0, spl_cma),
            daemon=True,
        )
        t.start()

    def _cma_worker(
        self,
        ticket: int,
        erode: int,
        n_ctrl: int,
        margin_m: float,
        budget: int,
        sigma0: float,
        spl_cma: float,
    ) -> None:
        try:
            with self._lock:
                key = self._track_key
                need_reload = key is None or key != (erode, n_ctrl)

            if need_reload:
                ctx = gw.load_track_and_ctrl(self.map_yaml, erode, n_ctrl)
            else:
                with self._lock:
                    ctx = self._ctx
                assert ctx is not None

            margin_px = margin_m / float(ctx["res"])
            best, fbest = gw.run_cma_optimize(
                ctx["ctrl_pts"],
                ctx["dist_map"],
                margin_px,
                spl_cma,
                budget,
                sigma0,
            )
            self._queue.put(("done", ticket, ctx, best, fbest))
        except Exception as e:  # noqa: BLE001
            self._queue.put(("err", ticket, str(e)))

    def _pump_queue(self) -> None:
        try:
            while True:
                msg = self._queue.get_nowait()
                kind = msg[0]
                if kind == "busy":
                    self._busy = True
                    self._set_status(f"Running CMA-ES (ticket {msg[1]}) …")
                elif kind == "err":
                    self._busy = False
                    self._set_status(f"Error: {msg[2]}")
                elif kind == "done":
                    _, ticket, ctx, best, fbest = msg
                    if ticket != self._heavy_gen:
                        continue
                    with self._lock:
                        self._ctx = ctx
                        self._best = best
                        self._track_key = (
                            int(round(self.sl_erode.val)),
                            int(round(self.sl_nctrl.val)),
                        )
                        self._last_fbest = fbest
                    self._busy = False
                    self._set_status(f"CMA done. f_best={fbest:.4f}  (ticket {ticket})")
                    self._refresh_fast()
        except queue.Empty:
            pass
        return True

    def _refresh_fast(self) -> None:
        with self._lock:
            ctx = self._ctx
            best = None if self._best is None else self._best.copy()

        if ctx is None or best is None:
            return

        frenet = gw.frenet_from_offsets(
            ctx["ctrl_pts"],
            best,
            ctx["img_h"],
            ctx["res"],
            ctx["origin"],
            spl_smooth=max(0.0, float(self.sl_spl_out.val)),
            n_out=int(round(self.sl_nout.val)),
            frenet_sigma=max(0.0, float(self.sl_frenet.val)),
            v_min=float(self.sl_vmin.val),
            v_max=float(self.sl_vmax.val),
            a_lat=float(self.sl_alat.val),
        )
        if frenet is None:
            self._set_status("Spline failed — try lowering output spl_smooth or CMA knobs.")
            return

        self._line_xy.set_data(frenet[:, 1], frenet[:, 2])
        self.ax_map.relim()
        self.ax_map.autoscale_view()
        self._line_v.set_data(frenet[:, 0], frenet[:, 4])
        self.ax_vel.relim()
        self.ax_vel.autoscale_view()
        if not self._busy:
            fb = self._last_fbest
            extra = f"  |  f_best={fb:.4f}" if fb is not None else ""
            self._set_status(
                f"Track L≈{frenet[-1, 0]:.1f} m  |  v∈[{frenet[:, 4].min():.2f}, "
                f"{frenet[:, 4].max():.2f}] m/s{extra}"
            )
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def main() -> None:
    p = argparse.ArgumentParser(description="Slider GUI for generate_waypoints racelines")
    p.add_argument("--map", default="comp_track.yaml", help="ROS map YAML (under devkit if relative)")
    p.add_argument(
        "--debounce-ms",
        type=float,
        default=1400.0,
        help="Delay after moving a CMA-related slider before re-optimising (ms)",
    )
    p.add_argument(
        "--skip-initial-opt",
        action="store_true",
        help="Open quickly with zero lateral offsets (no CMA until you move sliders / click button)",
    )
    args = p.parse_args()

    map_path = args.map
    if not os.path.isabs(map_path):
        map_path = os.path.join(_DEVKIT, map_path)

    RacelineExplorer(
        map_path,
        debounce_ms=args.debounce_ms,
        skip_initial_opt=args.skip_initial_opt,
    ).show()


if __name__ == "__main__":
    main()
