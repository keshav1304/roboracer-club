"""Feature extractor: one fixed-length vector per tick, from what the car can
actually know at decision time (LiDAR, odometry, the track map, the sensed
opponent and its own recent history). It never reads the teacher's internals,
so the learned model sees exactly the same inputs on the car.

Groups (column prefixes):
  ego_*    speed, Frenet offset/heading vs the raceline, profile speed, free width
  trk_*    raceline curvature + profile speed previewed 1..8 m ahead
  scan_*   LiDAR min-pooled into SCAN_BINS sectors, clipped to SCAN_MAX
  opp0_*   the sensed opponent (present flag + Frenet relative state); a fixed
           slot per opponent so the DeepSets model can take a set later
  hist_*   stopped time, recent progress, opponent displacement (static-ness)
  prev_*   one-hot of the previous decision (the model feeds its own back)
"""
import numpy as np

from common import CAR_L, DT, TrackModel

SCAN_BINS = 54          # 1080 beams / 20 → 5° sectors
SCAN_MAX = 10.0         # m
PREVIEW = (1.0, 2.0, 4.0, 6.0, 8.0)   # m ahead along the raceline
OPP_RANGE = 15.0        # m; opponents further than this along s are "absent"
STATES = ["CAR_STOPPED", "OBSTACLE", "OVERTAKE", "CORNERING", "SPEED_UP", "SPEED_DOWN"]


class FeatureExtractor:
    def __init__(self, tm: TrackModel, v_max=6.0, aggression=0.9, static_window=1.5, stuck_window=2.0,
                 static_min=0.6):
        self.tm, self.v_max, self.aggr = tm, v_max, aggression
        self.static_window, self.stuck_window, self.static_min = static_window, stuck_window, static_min
        self.reset()

    def reset(self):
        self.t, self.stopped_for = 0.0, 0.0
        self.s_un, self.s_last = 0.0, None
        self.prog, self.opp = [], []       # (t, s_un), (t, x, y)
        self.prev_state = "SPEED_UP"

    @staticmethod
    def names():
        n = ["ego_v", "ego_d", "ego_e_head", "ego_v_ref", "ego_v_err", "ego_free_left", "ego_free_right"]
        n += [f"trk_k_{m:g}" for m in PREVIEW] + [f"trk_v_{m:g}" for m in PREVIEW]
        n += [f"scan_{i:02d}" for i in range(SCAN_BINS)]
        n += ["opp0_present", "opp0_rel", "opp0_gap", "opp0_d", "opp0_dd", "opp0_v", "opp0_dv",
              "opp0_ttc", "opp0_free_left", "opp0_free_right"]
        n += ["hist_stopped_for", "hist_progress", "hist_opp_disp"]
        n += [f"prev_{s}" for s in STATES]
        return n

    def _vprof(self, s):
        line = self.tm.race
        return min(float(line.v[line.index_at(s)]) * self.aggr, self.v_max)

    def __call__(self, ego_obs, opp_sensed):
        line, tm = self.tm.race, self.tm
        x, y = float(ego_obs["pose_x"]), float(ego_obs["pose_y"])
        yaw, v = float(ego_obs["pose_theta"]), float(ego_obs["linear_vel_x"])
        self.t += DT
        self.stopped_for = self.stopped_for + DT if abs(v) < 0.1 else 0.0

        s, d, i = line.project(x, y)
        e_head = (yaw - line.yaw[i] + np.pi) % (2 * np.pi) - np.pi
        if self.s_last is not None:
            self.s_un += line.ds(self.s_last, s)
        self.s_last = s
        self.prog.append((self.t, self.s_un))
        while self.prog[0][0] < self.t - self.stuck_window:
            self.prog.pop(0)
        v_ref = self._vprof(s + 0.25 * v)
        fl, fr = tm.free_width(x, y)
        f = [v, d, e_head, v_ref, v - v_ref, fl, fr]
        f += [float(line.k[line.index_at(s + m)]) for m in PREVIEW]
        f += [self._vprof(s + m) for m in PREVIEW]

        scan = np.minimum(np.asarray(ego_obs["scan"], float), SCAN_MAX)
        f += list(scan[: SCAN_BINS * (len(scan) // SCAN_BINS)].reshape(SCAN_BINS, -1).min(1))

        opp = None
        if opp_sensed is not None:
            s_o, d_o, i_o = line.project(opp_sensed["x"], opp_sensed["y"])
            rel = line.ds(s, s_o)
            if abs(rel) < OPP_RANGE:
                opp = (opp_sensed, rel, d_o, i_o)
        if opp is None:
            self.opp.clear()
            f += [0.0, OPP_RANGE, OPP_RANGE, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0]
            disp = 0.0
        else:
            o, rel, d_o, i_o = opp
            gap = rel - CAR_L if rel > 0 else rel + CAR_L       # bumper-to-bumper, signed
            # speeds along the track direction (heading-aware), as in Teacher._situation
            v_os = o["v"] * np.cos(o["yaw"] - line.yaw[i_o])
            dv = v * np.cos(e_head) - v_os
            ttc = gap / dv if (rel > 0 and dv > 0.05) else 10.0
            ofl, ofr = tm.free_width(o["x"], o["y"])
            f += [1.0, rel, gap, d_o, d_o - d, v_os, dv, min(ttc, 10.0), ofl, ofr]
            self.opp.append((self.t, o["x"], o["y"]))
            while self.opp[0][0] < self.t - self.static_window:
                self.opp.pop(0)
            full = self.opp[-1][0] - self.opp[0][0] >= self.static_min - 2 * DT
            disp = float(np.hypot(self.opp[-1][1] - self.opp[0][1], self.opp[-1][2] - self.opp[0][2])) \
                if full else 9.9                                  # unknown yet = "moving"
        f += [self.stopped_for, self.prog[-1][1] - self.prog[0][1], disp]
        f += [float(self.prev_state == st) for st in STATES]
        return np.asarray(f, np.float32), {"s_un": self.s_un}

    def observe_decision(self, state):
        self.prev_state = state
