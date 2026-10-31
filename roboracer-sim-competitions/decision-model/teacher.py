"""Teacher state machine: the rule-based decision layer.

It has three jobs: (1) labeler for training data, (2) the baseline the learned
model must beat, (3) the fallback when the model is unsure.

Six states, chosen every tick by priority, with a minimum dwell time for the
racing states (safety states preempt immediately):

  CAR_STOPPED  stuck or stationary          → FTG (hold / reverse / relaunch)
  OBSTACLE     static object on our line    → FTG around it
  OVERTAKE     slower car ahead, gap exists → FTG past it (ForzaETH-style)
  SPEED_DOWN   must brake / trailing        → raceline pursuit, trailing speed law
  CORNERING    corner within braking horizon→ raceline pursuit at the profile speed
  SPEED_UP     clear track                  → raceline pursuit, accelerate

Thresholds follow the Decision Models spec §3.3 (ForzaETH values where cited).
"""
from dataclasses import dataclass, field

import numpy as np

from common import CAR_L, CAR_W, DT, FollowTheGap, FTGParams, TrackModel

STATES = ["CAR_STOPPED", "OBSTACLE", "OVERTAKE", "CORNERING", "SPEED_UP", "SPEED_DOWN"]
SAFETY = {"CAR_STOPPED", "OBSTACLE"}


@dataclass
class TeacherParams:
    v_max: float = 6.0            # cap on the raceline speed profile (fair vs FTG at 6)
    aggression: float = 0.9       # α_k scale on the raceline speed profile (1.0 slides wide in the hairpin)
    horizon: float = 6.9          # m, ForzaETH gb/overtaking horizon
    lane: float = 0.45            # m, |d_obj - d_line| below this = "on our line"
    static_window: float = 1.5    # s; static = moved < static_move over this window. By position, like
    static_move: float = 0.5      # m   ForzaETH: sensed speed noise (σ 0.49 m/s) is too big to threshold
    static_release: float = 1.0   # m; once static, it must move this much to count as moving again
    static_min: float = 0.6       # s of sightings before static can be judged (noise: ~2% false static)
    closing_v: float = 0.3        # m/s, must be this much faster to overtake
    min_gap: float = CAR_W + 2 * 0.15  # m, free width needed beside the opponent
    pass_margin: float = 0.6      # m, opponent this far behind = pass complete (or out of view, see _choose)
    block_arc: float = 1.0        # rad of beams clipped on the narrow side of a parked car
    squeeze_gap: float = 3.0      # m; within this of a parked car in OBSTACLE, cap speed at squeeze_v
    squeeze_v: float = 2.5        # m/s
    pass_hold: float = 0.2        # s to stay on the passing line after the pass is done
    lost_grace: float = 0.5       # s to hold OVERTAKE/OBSTACLE when the opponent is lost while still ahead
    trail_gap: float = 1.5        # m, ForzaETH trailing reference gap
    trail_kp: float = 1.0
    trail_kd: float = 0.2
    trail_floor: float = 1.5      # m/s
    kappa_corner: float = 0.15    # 1/m (R < 6.7 m) counts as a corner
    corner_time: float = 0.8      # s of look-ahead for "corner within reach"
    brake_margin: float = 0.3     # m/s over target before SPEED_DOWN
    stop_v: float = 0.1
    stop_time: float = 0.5        # s stationary → CAR_STOPPED
    relaunch_v: float = 1.0       # m/s to leave CAR_STOPPED
    dwell: float = 0.3            # s minimum time in a racing state
    k_lat: float = 0.4            # rad/m lateral-error feedback (gym tyres understeer at 6 m/s)
    k_head: float = 0.5           # rad/rad heading-error damping (stops weaving at speed)
    rejoin_k: float = 2.5         # lookahead ≥ rejoin_k·|d| when off the raceline
    k_lat_max: float = 0.2        # rad cap on the combined feedback
    stuck_window: float = 2.0     # s; also stuck if raceline progress < stuck_progress over this window
    stuck_progress: float = 0.3   # m   (creeping and stopping never reads as v < stop_v for long)
    turn_back: float = 1.0        # s per reverse leg of the three-point turn
    turn_fwd: float = 0.7         # s per forward leg
    turn_speed: float = 1.0       # m/s during the turn
    turn_front_ok: float = 0.8    # m of free space ahead to start the turn going forward
    turn_rear_min: float = 0.3    # m on the rear-most beams: stop reversing
    hold_time: float = 0.5        # s of FTG holding at 0 in CAR_STOPPED before a reverse nudge
    nudge_time: float = 0.6       # s of reversing per nudge
    nudge_tail_min: float = 0.2   # m on the rear quadrant the tail swings into
    heading_ok: float = 0.5       # rad; done turning when within this of the raceline heading
    misaligned: float = 0.6       # rad; this far off the raceline heading below misaligned_v → CAR_STOPPED
    misaligned_v: float = 2.0     # m/s  (spawned or spun facing a wall: launching drives into it)
    ld_min: float = 0.9           # m, pure-pursuit lookahead clip
    ld_max: float = 5.0
    wheelbase: float = 0.3302
    max_steer: float = 0.4189
    ftg: FTGParams = field(default_factory=lambda: FTGParams(v_max=6.0))


# Per-map overrides of the Spielberg-tuned defaults. track-20260731 is our
# mapped competition track (iros-2026): a 30.5 m loop, 0.7–2.6 m wide.
MAP_PRESETS = {
    "icra2026-comp": dict(misaligned=1.0),   # tight hairpins: heading legitimately >0.6 rad off the line
    "track-20260731": dict(v_max=5.0, aggression=1.0, horizon=3.0, lane=0.35, trail_gap=0.8, ld_min=0.6, ld_max=2.5,
                           kappa_corner=0.6, ftg=FTGParams(v_max=5.0)),
}


def params_for(map_name: str) -> TeacherParams:
    return TeacherParams(**MAP_PRESETS.get(map_name, {}))


class RacelinePursuit:
    """Pure pursuit on the raceline (ForzaETH lookahead L_d = 0.6·v − 0.18 m) + lateral-error feedback."""

    def __init__(self, tm: TrackModel, p: TeacherParams):
        self.tm, self.p = tm, p

    def steer(self, x, y, yaw, v):
        line = self.tm.race
        s, d0, _ = line.project(x, y)
        # far off the line (race start, after an FTG pass): look further ahead so
        # the rejoin is a gradual merge, not a swerve into the wall the line hugs
        ld = float(np.clip(max(0.6 * v - 0.18, self.p.rejoin_k * abs(d0)), self.p.ld_min, self.p.ld_max))
        tx, ty = line.xy[line.index_at(s + ld)]
        alpha = np.arctan2(ty - y, tx - x) - yaw
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        steer = np.arctan2(2 * self.p.wheelbase * np.sin(alpha), ld)
        # Pure pursuit alone lets the car slide 0.6 m wide in fast sweepers
        # (tyre understeer in the gym's single-track model) → add lateral-error
        # feedback, capped so it can't yank the car into a wall.
        _, d, i = line.project(x, y)
        e_head = (yaw - line.yaw[i] + np.pi) % (2 * np.pi) - np.pi
        fb = self.p.k_lat * d + self.p.k_head * e_head
        steer -= float(np.clip(fb, -self.p.k_lat_max, self.p.k_lat_max))
        return float(np.clip(steer, -self.p.max_steer, self.p.max_steer))

    def v_ref(self, s, v):
        """Profile speed a short time ahead (delay compensation), capped."""
        line = self.tm.race
        # scale the profile (corner speeds), then cap: scaling after the cap
        # would also slow the straights
        return min(float(line.v[line.index_at(s + 0.25 * v)]) * self.p.aggression, self.p.v_max)


class Teacher:
    def __init__(self, tm: TrackModel, scan_angles, params: TeacherParams | None = None):
        self.p = params or TeacherParams()
        self.tm = tm
        self.pp = RacelinePursuit(tm, self.p)
        self.ftg = FollowTheGap(scan_angles, self.p.ftg)
        # clearance check for raceline driving: the min-curvature line passes
        # ~0.3 m from walls, so FTG's 0.305 m lane would brake for the wall
        # alongside; use the car's half-width + 5 cm instead
        self.clear = FollowTheGap(scan_angles, FTGParams(corridor_margin=0.05))
        self.state, self.t_in_state, self.t = "SPEED_UP", 0.0, 0.0
        self.stopped_for = 0.0
        self.prog_hist = []         # (t, unwrapped raceline s) for the no-progress check
        self.s_un, self.s_last = 0.0, None
        self.turn_phase, self.turn_left = None, 0.0
        self.held_for, self.nudge_left = 0.0, 0.0
        self.opp_hist = []          # (t, x, y) of the sensed opponent over static_window
        self.last_seen = None       # (t, rel) at the last tick the opponent was visible
        self.static_latched = False
        self.pass_done_at = None    # t when the current OVERTAKE/OBSTACLE pass completed
        self.obs_side = None        # +1 / −1: side chosen to pass the current parked car

    # ---------------------------------------------------------------- features
    def _situation(self, ego, opp):
        """Everything the rules need, in the raceline Frenet frame."""
        line = self.tm.race
        s_e, d_e, i_e = line.project(ego["x"], ego["y"])
        out = {"s": s_e, "d": d_e, "v": ego["v"], "opp": None}
        if opp is not None:
            s_o, d_o, i_o = line.project(opp["x"], opp["y"])
            rel = line.ds(s_e, s_o)                    # centre-to-centre along s, + = opponent ahead
            fl, fr = self.tm.free_width(opp["x"], opp["y"])
            # speeds along the track direction: a car turned sideways or backwards
            # (spin, three-point turn) isn't closing / opening the gap at its full speed
            v_es = ego["v"] * np.cos(ego["yaw"] - line.yaw[i_e])
            v_os = opp["v"] * np.cos(opp["yaw"] - line.yaw[i_o])
            out["opp"] = {
                "rel": rel, "gap": rel - CAR_L,       # gap = bumper-to-bumper when ahead
                "d": d_o, "v": v_os, "dv": v_es - v_os,
                "on_line": abs(d_o) < self.p.lane or abs(d_o - d_e) < self.p.lane,
                "static": self._opp_static(),
                "free_left": fl, "free_right": fr,
            }
        return out

    def _no_progress(self):
        h = self.prog_hist
        return (self.t > self.p.stuck_window and h[-1][0] - h[0][0] >= self.p.stuck_window - 2 * DT
                and h[-1][1] - h[0][1] < self.p.stuck_progress)

    def _opp_static(self):
        """Static if the opponent's sensed position moved < static_move over a
        full static_window (also covers a car waiting at the start: it moves
        off within the window and is reclassified)."""
        h = self.opp_hist
        # needs static_min of sightings, not the full window: the LiDAR only
        # sees 10 m (1.6 s at 6 m/s), so waiting 1.5 s finds a parked car too late
        if not h or h[-1][0] - h[0][0] < self.p.static_min - 2 * DT:
            return False
        disp = float(np.hypot(h[-1][1] - h[0][1], h[-1][2] - h[0][2]))
        # hysteresis: once judged static, only a clear move (static_release) undoes
        # it; position noise alone flips a 0.5 m threshold now and then
        self.static_latched = disp < (self.p.static_release if self.static_latched else self.p.static_move)
        return self.static_latched

    def _corner_ahead(self, s, v):
        line = self.tm.race
        reach = max(1.0, v * self.p.corner_time)
        i0, i1 = line.index_at(s), line.index_at(s + reach)
        idx = np.arange(i0, i1 + (line.n if i1 < i0 else 0) + 1) % line.n
        return bool(np.any(np.abs(line.k[idx]) > self.p.kappa_corner))

    # ------------------------------------------------------------------ rules
    def _choose(self, sit):
        p, o = self.p, sit["opp"]
        if self.state == "CAR_STOPPED" and (sit["v"] < p.relaunch_v or abs(sit["e_head"]) > p.heading_ok
                                             or self.turn_phase is not None):
            return "CAR_STOPPED"
        if self.stopped_for >= p.stop_time or self._no_progress():
            return "CAR_STOPPED"
        if abs(sit["e_head"]) > p.misaligned and sit["v"] < p.misaligned_v:
            return "CAR_STOPPED"            # pointing at a wall at low speed: turn first, don't launch into it
        ahead = o is not None and 0.0 < o["gap"] < p.horizon and o["on_line"]
        if ahead and o["static"]:
            return "OBSTACLE"
        if self.state == "OBSTACLE" and o is not None and o["static"] and o["rel"] > -(CAR_L + p.pass_margin):
            return "OBSTACLE"                                   # until it is behind us
        gap_ok = o is not None and max(o["free_left"], o["free_right"]) >= p.min_gap
        if self.state in ("OVERTAKE", "OBSTACLE"):
            if o is None and self.last_seen is not None and self.last_seen[1] > CAR_L / 2 \
                    and self.t - self.last_seen[0] < p.lost_grace:
                return self.state     # lost while still ahead (corner, flicker): hold briefly
            # pass done: it is pass_margin behind us, or it was last seen alongside /
            # behind and has gone into the blind wedge behind the LiDAR (on the real
            # car that's the only signal on a tight pass: we never see it reach 0.6 m)
            lost_behind = o is None and self.last_seen is not None and self.last_seen[1] <= CAR_L / 2
            behind = o is not None and o["rel"] <= -(CAR_L + p.pass_margin)
            if (lost_behind or behind) and self.pass_done_at is None:
                self.pass_done_at = self.t
            if self.pass_done_at is not None and self.t - self.pass_done_at < p.pass_hold:
                return self.state     # keep the passing line a moment longer: opens the gap before rejoining
        if self.state == "OVERTAKE" and o is not None and o["rel"] > -(CAR_L + p.pass_margin):
            # committed: continue until the opponent is behind; abort only while
            # still fully behind it and the gap has closed
            if o["rel"] < CAR_L or gap_ok:
                return "OVERTAKE"
            return "SPEED_DOWN"
        if ahead:
            return "OVERTAKE" if (o["dv"] > p.closing_v and gap_ok) else "SPEED_DOWN"  # SPEED_DOWN = trailing
        if self._corner_ahead(sit["s"], sit["v"]):
            return "CORNERING"
        if sit["v"] > self.pp.v_ref(sit["s"], sit["v"]) + p.brake_margin:
            return "SPEED_DOWN"
        return "SPEED_UP"

    def _three_point_turn(self, sit, scan):
        """CAR_STOPPED recovery. If pointing away from the raceline heading,
        alternate reverse (wheels toward the error, so reversing rotates us
        back) and forward (opposite lock) legs until |e_head| < heading_ok,
        then let FTG relaunch. Returns (steer, v) or None to use FTG."""
        p, e = self.p, sit["e_head"]
        front = float(np.min(scan[np.abs(self.ftg.angles) < 0.35]))
        # the LiDAR can't see straight back; the rear-most beams (±110–135°) are the best proxy
        rear = float(np.min(scan[np.abs(self.ftg.angles) > 1.9]))
        if self.turn_phase is None:
            if abs(e) <= p.heading_ok:
                return None
            # go forward first when there is room: reversing blind is what hits walls
            fwd = front > p.turn_front_ok or rear < p.turn_rear_min
            self.turn_phase = "fwd" if fwd else "back"
            self.turn_left = p.turn_fwd if fwd else p.turn_back
        if abs(e) <= p.heading_ok and self.turn_phase == "fwd":
            self.turn_phase = None
            return None
        self.turn_left -= DT
        if self.turn_phase == "fwd" and front < 0.35:      # about to touch: switch to reverse early
            self.turn_left = 0.0
        if self.turn_phase == "back" and rear < p.turn_rear_min:
            self.turn_left = 0.0
        if self.turn_left <= 0.0:
            self.turn_phase = "fwd" if self.turn_phase == "back" else "back"
            self.turn_left = p.turn_fwd if self.turn_phase == "fwd" else p.turn_back
        lock = np.sign(e) * p.max_steer
        if self.turn_phase == "back":
            return float(lock), -p.turn_speed
        return float(-lock), p.turn_speed

    def _rear_clear(self, ego):
        """Map check behind the rear bumper (the LiDAR is blind straight back):
        reversing into walls caused most CAR_STOPPED crashes on tight tracks."""
        c, s_ = np.cos(ego["yaw"]), np.sin(ego["yaw"])
        for back in (0.10, 0.25, 0.40):
            for side in (-0.18, 0.0, 0.18):
                bx, by = -CAR_L / 2 - back, side
                if self.tm.occupied(ego["x"] + bx * c - by * s_, ego["y"] + bx * s_ + by * c):
                    return False
        return True

    def _update_obstacle_side(self, st, o):
        """Pick the side to pass a parked car once, when OBSTACLE starts, from its
        averaged position (static, so averaging removes the sensor noise), and
        keep it until the car is passed or lost. Re-deciding every tick from
        noisy positions made FTG flip sides."""
        near = o is not None and o["static"] and o["rel"] > -(CAR_L + self.p.pass_margin)
        if not near:
            self.obs_side = None
            return
        if self.obs_side is None and st == "OBSTACLE" and self.opp_hist:
            mx = float(np.mean([h[1] for h in self.opp_hist]))
            my = float(np.mean([h[2] for h in self.opp_hist]))
            fl, fr = self.tm.free_width(mx, my)
            self.obs_side = 1.0 if fl >= fr else -1.0          # +1: pass on its left

    def _block_narrow_side(self, scan, ego):
        """Make the narrow side of the parked car look blocked to FTG: beams from
        its bearing toward that side are clipped at its distance."""
        mx = float(np.mean([h[1] for h in self.opp_hist]))
        my = float(np.mean([h[2] for h in self.opp_hist]))
        dist = float(np.hypot(mx - ego["x"], my - ego["y"]))
        b = (np.arctan2(my - ego["y"], mx - ego["x"]) - ego["yaw"] + np.pi) % (2 * np.pi) - np.pi
        rel = (self.ftg.angles - b) * -self.obs_side      # > 0 on the narrow side of its bearing
        block = (rel > 0.0) & (rel < self.p.block_arc)
        scan = scan.copy()
        scan[block] = np.minimum(scan[block], dist)
        return scan

    def _nudge(self, ego, scan, v_ftg):
        """FTG can deadlock nose-close to a wall with the heading already 'ok'
        (no three-point turn): it holds at 0 forever. After hold_time of that,
        reverse for nudge_time with the wheels toward the nearer wall, which
        swings the nose away from it, then let FTG try again."""
        p, a = self.p, self.ftg.angles
        fl, fr = self.tm.free_width(ego["x"], ego["y"])
        lock = 1.0 if fl < fr else -1.0       # wheels toward the nearer wall: nose swings away, tail toward it
        tail = float(np.min(scan[lock * a > 1.9]))     # rear quadrant the tail swings into
        other = float(np.min(scan[lock * a < -1.9]))
        clear = tail >= p.nudge_tail_min and other >= p.turn_rear_min
        if self.nudge_left > 0.0:
            self.nudge_left -= DT
            if not clear:
                self.nudge_left = 0.0
            else:
                return float(lock * p.max_steer), -p.turn_speed
        self.held_for = self.held_for + DT if (v_ftg < 0.2 and abs(ego["v"]) < p.stop_v) else 0.0
        if self.held_for >= p.hold_time and clear:
            self.held_for, self.nudge_left = 0.0, p.nudge_time
        return None

    # ------------------------------------------------------------------- tick
    def plan(self, ego_obs, opp_sensed, decide=None):
        steer, v, st, sit = self._plan(ego_obs, opp_sensed, decide)
        if v < 0.0:
            ego = {"x": float(ego_obs["pose_x"]), "y": float(ego_obs["pose_y"]), "yaw": float(ego_obs["pose_theta"])}
            if not self._rear_clear(ego):
                v = 0.0
                self.nudge_left = 0.0
                if self.turn_phase == "back":
                    self.turn_left = 0.0          # switch to the forward leg next tick
        return steer, v, st, sit

    def _plan(self, ego_obs, opp_sensed, decide=None):
        """ego_obs: gym features dict for the ego; opp_sensed: dict from
        OpponentSensor (or None). decide: optional callable(situation) → state
        that replaces the rule-based choice (a learned decider driving the
        teacher's executors). Returns (steer, speed, state, situation)."""
        p = self.p
        ego = {"x": float(ego_obs["pose_x"]), "y": float(ego_obs["pose_y"]),
               "yaw": float(ego_obs["pose_theta"]), "v": float(ego_obs["linear_vel_x"])}
        self.t += DT
        self.stopped_for = self.stopped_for + DT if (abs(ego["v"]) < p.stop_v and self.t > 1.0) else 0.0
        if opp_sensed is not None:
            self.opp_hist.append((self.t, opp_sensed["x"], opp_sensed["y"]))
            while self.opp_hist and self.opp_hist[0][0] < self.t - p.static_window:
                self.opp_hist.pop(0)
        else:
            self.opp_hist.clear()       # lost track: static-ness has to be re-established
            self.static_latched = False
        sit = self._situation(ego, opp_sensed)
        if sit["opp"] is not None:
            self.last_seen = (self.t, sit["opp"]["rel"])
        i = self.tm.race.project(ego["x"], ego["y"])[2]
        sit["e_head"] = (ego["yaw"] - self.tm.race.yaw[i] + np.pi) % (2 * np.pi) - np.pi
        if self.s_last is not None:
            self.s_un += self.tm.race.ds(self.s_last, sit["s"])
        self.s_last = sit["s"]
        self.prog_hist.append((self.t, self.s_un))
        while self.prog_hist[0][0] < self.t - p.stuck_window:
            self.prog_hist.pop(0)

        want = self._choose(sit) if decide is None else decide(sit)
        if want != self.state and (want in SAFETY or self.state in SAFETY or self.t_in_state >= p.dwell):
            if self.state == "CAR_STOPPED":
                self.prog_hist = [(self.t, self.s_un)]   # don't re-trigger on the stuck history
                self.stopped_for = 0.0
            self.state, self.t_in_state = want, 0.0
            self.pass_done_at = None
        else:
            self.t_in_state += DT
        st = self.state

        scan = np.asarray(ego_obs["scan"], float)
        if st == "CAR_STOPPED":
            out = self._three_point_turn(sit, scan)
            if out is not None:
                return out[0], out[1], st, sit
        elif self.turn_phase is not None:
            self.turn_phase = None
        if st in ("CAR_STOPPED", "OBSTACLE", "OVERTAKE"):
            o = sit["opp"]
            self._update_obstacle_side(st, o)
            if self.obs_side is not None and st in ("OBSTACLE", "CAR_STOPPED") and o is not None and self.opp_hist:
                scan = self._block_narrow_side(scan, ego)
            steer, v = self.ftg.plan(scan, ego["v"])
            if st == "OBSTACLE" and o is not None and 0.0 < o["gap"] < p.squeeze_gap:
                v = min(v, p.squeeze_v)       # tight turn around a parked car (FTG at 3.6 m/s wedged in hairpins)
            if st == "CAR_STOPPED":
                nudge = self._nudge(ego, scan, v)
                if nudge is not None:
                    return nudge[0], nudge[1], st, sit
            return steer, v, st, sit

        steer = self.pp.steer(ego["x"], ego["y"], ego["yaw"], ego["v"])
        v = self.pp.v_ref(sit["s"], ego["v"])
        o = sit["opp"]
        if st == "SPEED_DOWN" and o is not None and 0 <= o["gap"] < p.horizon:
            e_gap = p.trail_gap - o["gap"]
            v = min(v, max(o["v"] - (p.trail_kp * e_gap + p.trail_kd * o["dv"]), p.trail_floor))
        # never drive into something on the steering arc (walls, cars)
        clear = self.clear._arc_clearance(scan, steer)
        v = min(v, np.sqrt(max(2.0 * p.ftg.a_brake * (clear - p.ftg.stop_dist), 0.0)))
        return steer, float(max(v, 0.0)), st, sit
