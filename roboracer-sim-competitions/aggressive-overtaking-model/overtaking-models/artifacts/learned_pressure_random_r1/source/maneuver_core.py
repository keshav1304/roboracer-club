"""Local learned decisions with explicit geometric feasibility checks.

No API client is imported here. Inputs are the ego's own LiDAR, localized pose,
velocity, the fixed reference line, and a track estimated from previous scans.
Opponent simulator poses are deliberately absent from this interface.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
import numpy as np

from bootstrap import MODELS
from perception import LidarTracker
from track_geometry import PathTracker

ACTIONS = ("LEFT", "RIGHT", "WAIT", "PROBE")
FEATURE_NAMES = (
    [f"lidar_sector_{i:02d}" for i in range(18)]
    + ["ego_speed", "line_offset", "heading_error", "curvature_now", "curvature_1m", "curvature_3m",
       "target_present", "target_distance", "target_offset", "target_speed", "target_width", "target_confidence"]
    + [f"{a.lower()}_{f}" for a in ACTIONS for f in ("feasible", "cost", "clearance", "steering", "speed")]
)


def delta_s(a, b, length):
    return (np.asarray(b) - a + length / 2) % length - length / 2


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * t * (10 + t * (-15 + 6 * t))


@dataclass
class Candidate:
    action: str
    feasible: bool
    cost: float
    clearance: float
    steering: float
    speed: float
    offset: float
    path: np.ndarray
    recovery_certified: bool = False

    def record(self):
        return {"action": self.action, "feasible": bool(self.feasible),
                "estimated_time": round(self.cost, 4), "clearance": round(self.clearance, 4),
                "steering": round(self.steering, 4), "target_speed": round(self.speed, 4), "padding_recovery_certified":self.recovery_certified}


class CandidatePlanner:
    def __init__(self, track, vmax=2.4, stage=1):
        self.track, self.vmax, self.stage = track, vmax, stage
        self.tracker = PathTracker(track)
        self.distances = np.linspace(0, 6., 41)
        # Explicit margins permit controlled simulator sweeps; body dimensions
        # and the independent LiDAR collision brake are never reduced.
        self.wall_margin = .04 if stage == 3 else .07
        self.car_margin = .08 if stage == 3 else .12
        self.detour_clearance = .15 if stage == 3 else .20
        self.lateral_acceleration = 2.
        self.passing_lateral_acceleration = 3.2 if stage == 3 else 2.4 if stage == 2 else 2.
        self.probe_distance = 2. if stage == 3 else 6.
        self.probe_offset = .41 if stage == 3 else .30
        self.probe_entry = 3. if stage == 3 else 1.7
        self.padding_recovery_enabled = stage == 3
        self.pass_completion_recovery = False
        self.control_dt = .01

    def planning_profile(self):
        return {
            "wall_margin_m": self.wall_margin,
            "car_margin_m": self.car_margin,
            "detour_clearance_m": self.detour_clearance,
            "lateral_acceleration_m_s2": self.lateral_acceleration,
            "passing_lateral_acceleration_m_s2": self.passing_lateral_acceleration,
            "passing_allowance_requires_active_target": True,
            "preview_distance_m": float(self.distances[-1]),
            "probe_preview_distance_m": self.probe_distance,
            "probe_offset_m": self.probe_offset,
            "probe_entry_distance_m": self.probe_entry,
            "padding_recovery_enabled": self.padding_recovery_enabled,
            "pass_completion_recovery_enabled": self.stage == 2,
            "padding_recovery_mode": "passing" if self.stage == 3 else "pass_completion_only" if self.stage == 2 else "disabled",
            "pass_completion_requires_observed_target_ahead": True,
            "pass_completion_requires_committed_side": True,
            "pass_completion_target_relative_range_m": [-2.2, -.25],
            "padding_recovery_minimum_predicted_gap_m": .025,
            "padding_recovery_maximum_predicted_gap_loss_m": .005,
            "padding_recovery_predicted_restore_seconds": .5,
            "padding_recovery_timing_is_prediction_only": True,
            "padding_recovery_lidar_halfwidth_m": .170,
            "physical_vehicle_halfwidth_m": .155,
        }

    def motion_clearance(self, xy, yaws, target, times):
        """Conservative separation of oriented bodies along a measured motion tube.

        Frenet lateral distance alone misses front-corner contact in tight bends.
        The separating axes of both rectangles also account for their rotation.
        """
        if target is None:
            return 10.
        drift = target.lateral_speed * .35 * (1 - np.exp(-times / .35))
        other, other_yaw, _ = self.track.sample(target.s + max(0., target.speed)*times,
                                               target.d + drift)
        forward = np.column_stack([np.cos(yaws), np.sin(yaws)])
        normal = np.column_stack([-np.sin(yaws), np.cos(yaws)])
        other_forward = np.column_stack([np.cos(other_yaw), np.sin(other_yaw)])
        other_normal = np.column_stack([-np.sin(other_yaw), np.cos(other_yaw)])
        axes = np.stack([forward, normal, other_forward, other_normal], axis=1)
        separation = np.abs(np.einsum('nij,nj->ni', axes, other-xy))
        separation -= .29*np.abs(np.einsum('nij,nj->ni', axes, forward))
        separation -= .155*np.abs(np.einsum('nij,nj->ni', axes, normal))
        separation -= target.length/2*np.abs(np.einsum('nij,nj->ni', axes, other_forward))
        separation -= target.width/2*np.abs(np.einsum('nij,nj->ni', axes, other_normal))
        return float(np.max(separation, axis=1).min())

    def certify_car_departure(self, xy, yaws, target, times, obs):
        """Check a predicted departure from depleted comfort padding.

        The .5s threshold uses the same constant-speed preview as the candidate;
        it is not a guarantee of actual restoration time. Only the initial own
        footprint is replaced with measured pose. No opponent truth is used.
        """
        if (not (self.padding_recovery_enabled or (self.stage == 2 and self.pass_completion_recovery)) or target is None or target.confidence < .6 or target.time_since_seen > .10
                or len(xy) < 3 or not np.isfinite(times).all() or times[-1] < .5):
            return False
        xy=np.asarray(xy,dtype=float).copy();yaws=np.asarray(yaws,dtype=float).copy()
        xy[0]=[obs["pose_x"],obs["pose_y"]];yaws[0]=float(obs["pose_theta"])
        drift=target.lateral_speed*.35*(1-np.exp(-times/.35))
        other,oyaw,_=self.track.sample(target.s+max(0.,target.speed)*times,target.d+drift)
        forward=np.column_stack([np.cos(yaws),np.sin(yaws)])
        normal=np.column_stack([-np.sin(yaws),np.cos(yaws)])
        other_forward=np.column_stack([np.cos(oyaw),np.sin(oyaw)])
        other_normal=np.column_stack([-np.sin(oyaw),np.cos(oyaw)])
        axes=np.stack([forward,normal,other_forward,other_normal],axis=1)
        sep=np.abs(np.einsum('nij,nj->ni',axes,other-xy))
        sep-=.29*np.abs(np.einsum('nij,nj->ni',axes,forward))
        sep-=.155*np.abs(np.einsum('nij,nj->ni',axes,normal))
        sep-=target.length/2*np.abs(np.einsum('nij,nj->ni',axes,other_forward))
        sep-=target.width/2*np.abs(np.einsum('nij,nj->ni',axes,other_normal))
        gaps=np.max(sep,axis=1)
        initial=float(gaps[0]);floor=max(.025,initial-.005)
        after=times>=.5
        return bool(.025 <= initial < self.car_margin and after.any()
                    and gaps.min() >= floor and gaps[after].min() >= self.car_margin
                    and np.max(gaps[times<=.5]) >= self.car_margin)

    def recovery(self, obs, target, s, d):
        """Find a finite, heading-matched return when the nominal preview fails."""
        g = self.track
        yaw = float(obs['pose_theta'])
        _, reference_yaw, bend = g.sample(s)
        error = np.arctan2(np.sin(yaw-reference_yaw), np.cos(yaw-reference_yaw))
        slope = float(np.clip(np.tan(error)*(1-float(bend)*d), -.8, .8))
        present = float(g.body_clearance(np.array([[obs['pose_x'],obs['pose_y']]]),np.array([yaw]))[0])
        # An already slightly depleted buffer must permit a path that restores it.
        required_wall = min(self.wall_margin, max(.02, present-.015))
        best = None
        for distance in (.85, 1.2, 1.7, 2.4, 3.2, 4.5):
            def offsets(ss, distance=distance):
                t = np.clip(delta_s(s,ss,g.length)/distance,0.,1.)
                return d*(1-smoothstep(t)) + slope*distance*(t-6*t**3+8*t**4-3*t**5)
            ss = s+self.distances
            xy,_,_=g.sample(ss,offsets(ss))
            tangents=np.gradient(xy,axis=0)
            yaws=np.arctan2(tangents[:,1],tangents[:,0]); yaws[0]=yaw
            wall=float(g.body_clearance(xy,yaws).min())
            if wall < required_wall:
                continue
            steer,speed=self.tracker.control(obs,offset=offsets,vmax=self.vmax,smooth=False)
            curvature=np.abs(np.gradient(np.unwrap(yaws),self.distances))
            limits=np.minimum(self.vmax,np.sqrt(self.lateral_acceleration/np.maximum(curvature,.05)))
            speed=min(speed,float(np.sqrt(limits**2+6*self.distances).min()))
            speed=min(speed,float(g.speed_at(s,self.vmax,self.lateral_acceleration,2.,3.)))
            travel=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]/max(.5,speed)
            car=self.motion_clearance(xy,yaws,target,travel)
            recovery_certified=self.certify_car_departure(xy,yaws,target,travel,obs)
            if target is not None and car < self.car_margin and not recovery_certified:
                continue
            score=speed-.06*distance
            if best is None or score>best[0]:
                best=(score,steer,speed,min(wall,car),xy,recovery_certified)
        return best

    def candidates(self, obs, target, committed=None):
        g = self.track
        s, d = g.project(float(obs["pose_x"]), float(obs["pose_y"]))
        s, d = float(s), float(d)
        v = float(obs["linear_vel_x"])
        rel = 20. if target is None else float(delta_s(s, target.s, g.length))
        active = target is not None and -2.2 < rel < 7.
        _, _, upcoming_bends = g.sample(s + np.linspace(1.,5.,9))
        corner_bend = float(upcoming_bends[int(np.argmax(np.abs(upcoming_bends)))])
        corner_near = abs(corner_bend) >= .18
        # Each candidate is a smooth, temporary departure from the reference.
        # Passing left makes the obstacle a right-hand boundary and vice versa.
        result = []
        for action in ACTIONS:
            recovery_certified=False
            lateral_acceleration=self.passing_lateral_acceleration if self.stage > 1 and active and action != "WAIT" else self.lateral_acceleration
            distances=np.linspace(0.,self.probe_distance,21) if self.stage == 3 and action == "PROBE" else self.distances
            side = 1 if action == "LEFT" else -1
            if active and action in ("LEFT", "RIGHT"):
                desired = float(target.d + side * (.155 + target.width / 2 + self.detour_clearance))
            elif active and action == "PROBE":
                side = 1 if corner_bend >= 0 else -1
                desired = float(target.d + side * self.probe_offset)
            else:
                desired = 0.
            if committed and action in ("LEFT", "RIGHT") and active:
                if committed[0] == action:
                    # Keep the chosen boundary stable through intermittent returns.
                    desired = committed[1]
            transition = max(.85, min(2.0, rel - .95)) if active else max(3.0, 3.0 * abs(d))
            if active and self.stage > 1 and target.speed > .4:
                transition = self.probe_entry if action == "PROBE" else 1.7
            # A moving pass also needs a planned exit. A constant-offset path
            # extending six metres can hit a future inside wall after there is
            # already room to rejoin, falsely invalidating a nearly complete pass.
            rejoin = None
            if active and self.stage > 1 and target.speed >= .3:
                closing = max(.05, self.vmax - target.speed)
                rejoin = max(0., (rel + .90) * self.vmax / closing)
            def corridor_offset(ss, desired):
                # Reference-to-wall widths change through corners. A detour
                # is a corridor, not a constant parallel racing line: taper it
                # towards the available road while retaining collision checks.
                left, right = g.widths(ss)
                inset = .24 + self.wall_margin
                return np.clip(desired, -np.asarray(right) + inset,
                               np.asarray(left) - inset)
            def offsets(ss, desired=desired, transition=transition):
                ahead = delta_s(s, ss, g.length)
                lane = corridor_offset(ss, desired) if self.stage > 1 else desired
                value = d + (lane - d) * smoothstep(ahead / transition)
                if active and target.speed < .3:
                    value *= 1 - smoothstep((ahead - rel - 1.05) / 2.0)
                elif rejoin is not None:
                    value *= 1 - smoothstep((ahead - rejoin) / 2.0)
                return value
            path_s = s + distances
            ds = offsets(path_s)
            xy, _, _ = g.sample(path_s, ds)
            tangents = np.gradient(xy, axis=0)
            yaws = np.arctan2(tangents[:, 1], tangents[:, 0])
            wall_clear = float(np.min(g.body_clearance(xy, yaws)))
            self.tracker.lateral_acceleration = lateral_acceleration
            steer, speed = self.tracker.control(obs, offset=offsets, vmax=self.vmax, smooth=False)
            curvature = np.abs(np.gradient(np.unwrap(yaws), distances))
            limits = np.minimum(self.vmax, np.sqrt(lateral_acceleration / np.maximum(curvature, .05)))
            speed = min(speed, float(np.min(np.sqrt(limits**2 + (6. if self.stage > 1 else 3.6) * distances))))
            if action == "WAIT" and hasattr(g, "speed_at"):
                speed = min(speed, float(g.speed_at(s, self.vmax, lateral_acceleration, 2., 3.)))
            # Conservative constant-speed motion tube, inflated for uncertainty.
            collision_clear = 10.
            if active:
                physical_distances = np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
                ts = physical_distances / max(.7, min(self.vmax, max(v, speed)))
                moving_s = target.s + max(0., target.speed) * ts
                longitudinal = np.abs(delta_s(path_s, moving_s, g.length))
                lateral_drift = target.lateral_speed * .35 * (1 - np.exp(-ts / .35)) if self.stage > 1 else 0.
                lateral_gap = np.abs(ds - (target.d + lateral_drift)) - (.155 + target.width / 2)
                relevant = longitudinal < (.29 + target.length / 2 + .16)
                if relevant.any():
                    collision_clear = float(np.min(lateral_gap[relevant]))
                if self.stage > 1:
                    collision_clear = self.motion_clearance(xy,yaws,target,ts)
                    recovery_certified=self.certify_car_departure(xy,yaws,target,ts,obs)
            clear = min(wall_clear, collision_clear)
            feasible = wall_clear >= self.wall_margin and (collision_clear >= self.car_margin or recovery_certified)
            length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
            cost = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.maximum(.4, limits[:-1])))
            cost *= 6. / distances[-1]
            cost += .15 * abs(desired) + .1 / max(.1, wall_clear)
            if action == "WAIT":
                # WAIT means follow the reference when clear, otherwise follow/stop.
                feasible = True
                hold_clear = None
                # A rear car can intersect a return-to-line path too. Do not
                # bypass the rolling hold solely because its centre is >.7m back.
                if active and (rel > -.7 or (self.stage > 1 and collision_clear < self.car_margin)):
                    # A nearby target is not necessarily in this path. In
                    # particular, a car behind and laterally clear must not
                    # trigger a zero follow-speed command on the reference.
                    # Only a predicted intersection warrants target braking;
                    # the independent LiDAR brake still checks all obstacles.
                    if collision_clear < self.car_margin and not recovery_certified:
                        buffer = .20 if self.stage > 1 else .65
                        remaining = max(0., rel - target.length / 2 - .29 - buffer)
                        closing_limit = np.sqrt(max(0., target.speed)**2 + (6. if self.stage > 1 else 3.6) * remaining)
                        follow_limit = max(0., target.speed + (1.6 * (rel - .95) if self.stage > 1 else .9 * (rel - 1.35)))
                        if self.stage > 1 and target.observations * self.control_dt < .18:
                            # A fresh target's zero velocity is unknown, not an
                            # observed stop. Respect stopping distance while the
                            # temporal velocity estimate settles.
                            follow_limit = closing_limit
                        speed = min(speed, closing_limit, follow_limit)
                    # Stay on the fast reference while closing from a distance;
                    # spending every corner in a detour lane can lose the catch.
                    cost += (1.0 + max(0., self.vmax - target.speed) * 2.
                             if target.speed < .4 or rel < 3.5 else 0.)
                    if rel < .9 and collision_clear < self.car_margin and not recovery_certified:
                        # An unsafe merge is not necessarily an unsafe lane.
                        # Keep moving in a collision-checked holding corridor
                        # until sufficient longitudinal separation permits exit.
                        steer, _ = self.tracker.control(obs, offset=d, vmax=self.vmax, smooth=False)
                        speed = 0.
                        if self.stage > 1 and abs(d - target.d) > .155 + target.width / 2 + self.car_margin:
                            hold_ss = s + np.linspace(0., 3., 25)
                            def hold_offsets(ss):
                                a = delta_s(s, ss, g.length)
                                return d + (corridor_offset(ss, d) - d) * smoothstep(a / 1.7)
                            hold_d = hold_offsets(hold_ss)
                            hold_xy, _, _ = g.sample(hold_ss, hold_d)
                            hold_v = np.gradient(hold_xy, axis=0)
                            hold_yaw = np.arctan2(hold_v[:,1], hold_v[:,0])
                            hold_wall = float(np.min(g.body_clearance(hold_xy, hold_yaw)))
                            travel = np.r_[0., np.cumsum(np.linalg.norm(np.diff(hold_xy, axis=0),axis=1))] / max(.7, self.vmax)
                            near = np.abs(delta_s(hold_ss, target.s + max(0.,target.speed)*travel,g.length)) < .29 + target.length/2 + .16
                            drift = target.lateral_speed * .35 * (1 - np.exp(-travel/.35))
                            sep = np.abs(hold_d - target.d - drift) - .155 - target.width/2
                            body_separation = self.motion_clearance(hold_xy,hold_yaw,target,travel)
                            hold_certified=self.certify_car_departure(hold_xy,hold_yaw,target,travel,obs)
                            if hold_wall >= self.wall_margin and (body_separation >= self.car_margin or hold_certified):
                                recovery_certified=hold_certified
                                steer, speed = self.tracker.control(obs, offset=hold_offsets, vmax=self.vmax, smooth=False)
                                wall_clear = hold_wall
                                hold_clear = body_separation
                                clear = min(hold_wall, hold_clear)
                                xy = hold_xy
                                desired = d
                if wall_clear < self.wall_margin:
                    speed = 0.
                clear = wall_clear if hold_clear is None else min(wall_clear, hold_clear)
                if self.stage > 1 and speed < .05:
                    recovery = self.recovery(obs,target if active else None,s,d)
                    if recovery is not None:
                        _,steer,speed,clear,xy,recovery_certified = recovery
                        desired = 0.
            if action == "PROBE":
                # Probe only occupies already-free room; it cannot assume a yield.
                feasible = feasible and self.stage == 3 and active and rel > -1.2 and corner_near
                if self.stage == 3:
                    speed = min(speed, (6*(self.probe_distance-.3))**.5)
                else:
                    speed = min(speed, max(0., target.speed if active else 0.))
                cost += .1
            if (self.stage == 3 and active and action != "WAIT"
                    and abs(d) < .16 and not (committed and committed[0] in ("LEFT","RIGHT"))
                    and not corner_near and self.vmax - target.speed < .25):
                # Equal-speed cars stay on the shared reference until there is
                # a corner to attack or a genuine longitudinal speed advantage.
                feasible = False
            if not active and action != "WAIT":
                feasible = False
            if active and self.stage > 1 and target.observations * self.control_dt < .18 and action != "WAIT":
                # Establish motion before committing to a boundary.
                feasible = False
            if not feasible:
                cost = 30.
            result.append(Candidate(action, feasible, cost, clear, float(steer),
                                    float(max(0., speed)), desired, xy,recovery_certified))
        return result

    def features(self, obs, target, candidates):
        scan = np.nan_to_num(np.asarray(obs["scan"], float), nan=0., posinf=20., neginf=0.)
        scan = np.clip(scan, 0, 20)
        sectors = ((np.percentile(scan.reshape(18,-1),15,axis=1)/10).tolist() if len(scan)%18==0
                   else [float(np.percentile(a,15))/10 for a in np.array_split(scan,18)])
        s, d = self.track.project(float(obs["pose_x"]), float(obs["pose_y"]))
        _, yaw, k = self.track.sample(np.array([s, s+1, s+3]))
        error = np.arctan2(np.sin(float(obs["pose_theta"])-yaw[0]), np.cos(float(obs["pose_theta"])-yaw[0]))
        values = sectors + [float(obs["linear_vel_x"])/3, float(d), float(error), *np.clip(k, -2, 2).tolist(),
                            float(target is not None),
                            2. if target is None else float(delta_s(s, target.s, self.track.length))/5,
                            0. if target is None else target.d,
                            0. if target is None else target.speed/3,
                            0. if target is None else target.width,
                            0. if target is None else target.confidence]
        for c in candidates:
            values.extend([float(c.feasible), c.cost/10, np.clip(c.clearance, -1, 2), c.steering, c.speed/3])
        return np.asarray(values, dtype=np.float32)


class NeuralDriver:
    """Network chooses a maneuver; bounded trajectory tracking produces controls."""
    def __init__(self, track, angles, *, stage, vmax=2.4, model_path=None, teacher=False):
        self.stage, self.vmax, self.teacher = stage, vmax, teacher
        self.planner = CandidatePlanner(track, vmax, stage)
        self.sensor = LidarTracker(track, angles)
        self.angles = angles
        self.model = None
        self.value_model = None
        self.value_model_path, self.value_model_sha256 = None, None
        self.teacher_bias = np.zeros(4 if stage == 3 else 3, dtype=float)
        self.model_path, self.model_sha256 = None, None
        if not teacher:
            from tiny_network import TinyNetwork
            self.model_path = Path(model_path or MODELS / f"stage{stage}.npz")
            self.model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
            self.model = TinyNetwork.load(self.model_path, expected_feature_names=FEATURE_NAMES)
            if self.model.classes != ACTIONS[:4 if stage==3 else 3]:
                raise ValueError("Checkpoint action classes do not match this stage")
            value_path = self.model_path.with_name(self.model_path.stem + ".value.npz")
            if stage > 1 and value_path.exists():
                names = FEATURE_NAMES + [f"action_{a.lower()}" for a in ACTIONS]
                value = TinyNetwork.load(value_path, expected_feature_names=names)
                if (value.classes != ("FAIL", "PASS") or value.metadata.get("stage") != stage
                        or not getattr(track, "reference_digest", None)
                        or value.metadata.get("reference_digest") != track.reference_digest):
                    raise ValueError("Outcome checkpoint stage/reference does not match this runtime")
                self.value_model = value
                self.value_model_path = value_path
                self.value_model_sha256 = hashlib.sha256(value_path.read_bytes()).hexdigest()
        self.action = "WAIT"
        self.commit = None
        self.pass_target_was_ahead = False
        self.steering = 0.
        self.command_speed = 0.
        self.telemetry = {}
        self.counter = 0
        self.control_times = []
        self.network_times = []
        self.padding_recovery_steps = 0
        self.padding_recovery_seconds = 0.
        self.padding_lidar_release_steps = 0
        self.padding_lidar_released_beams = 0

    def update_pass_completion_context(self, obs, target):
        """Enable Stage2 departure recovery only after its own observed pass."""
        s_now, _ = self.planner.track.project(float(obs['pose_x']),float(obs['pose_y']))
        rel_now = 20. if target is None else float(delta_s(s_now,target.s,self.planner.track.length))
        if target is None:
            self.pass_target_was_ahead=False
        elif rel_now > .3:
            self.pass_target_was_ahead=True
        self.planner.pass_completion_recovery=bool(self.stage==2 and self.commit is not None
            and self.pass_target_was_ahead and -2.2 < rel_now < -.25)

    def plan(self, obs, dt=.01):
        if not 0 < dt <= .1:
            raise ValueError("Control interval must be in (0, .1] seconds")
        start = time.perf_counter_ns()
        self.planner.control_dt = float(dt)
        target = self.sensor.update(obs, dt=dt)
        if self.stage == 1 and target is not None:
            from dataclasses import replace
            target = replace(target, speed=0., lateral_speed=0.)
        self.update_pass_completion_context(obs, target)
        candidates = self.planner.candidates(obs, target, self.commit)
        features = self.planner.features(obs, target, candidates)
        size = 4 if self.stage == 3 else 3
        if self.teacher:
            choices = np.array([c.cost if c.feasible else 100. for c in candidates[:size]])
            bias = np.asarray(self.teacher_bias, dtype=float)
            if bias.shape not in ((size,), (4,)) or not np.isfinite(bias).all():
                raise ValueError("Teacher bias must contain finite action costs")
            choices += bias[:size]
            p = np.exp(-(choices - choices.min())*2.)
            p /= p.sum()
        else:
            begin = time.perf_counter_ns()
            p = np.asarray(self.model.proba(features), float)
        raw_p = p.copy()
        success_probabilities = None
        if self.value_model is not None:
            outcome_inputs = np.column_stack([np.repeat(features[None,:],size,axis=0), np.eye(4,dtype=np.float32)[:size]])
            success_probabilities = self.value_model.predict_proba(outcome_inputs)[:,1]
        if not self.teacher:
            self.network_times.append((time.perf_counter_ns() - begin) / 1e6)
        feasible = np.array([c.feasible for c in candidates[:size]])
        p = p * feasible
        if p.sum() < 1e-8:
            p[:] = 0.; p[2] = 1.
        p /= p.sum()
        index = int(np.argmax(p))
        if success_probabilities is not None and target is not None:
            # Empirical continuation success is the primary learned objective.
            # Near-equal estimates retain the maneuver network preference.
            best_value = float(np.max(success_probabilities[feasible]))
            competitive = feasible & (success_probabilities >= best_value - .025)
            index = int(np.argmax(np.where(competitive, p, -1.)))
        filtered_index = index
        s, _ = self.planner.track.project(float(obs["pose_x"]), float(obs["pose_y"]))
        rel = 20. if target is None else float(delta_s(s, target.s, self.planner.track.length))
        if self.commit and target is not None and -2.2 < rel < 6.:
            old = ACTIONS.index(self.commit[0])
            if feasible[old] and old in (0,1):
                index = old
            elif abs(rel) < 1.0 and candidates[2].speed > .2:
                # Complete the existing lane before considering a side switch.
                index = 2
        # Optional experimental response planner; baseline drivers have no hook.
        selector = getattr(self, "select_response_maneuver", None)
        if selector is not None:
            index = selector(candidates, index, obs, target)
        selected = candidates[index]
        if selected.action in ("LEFT", "RIGHT") and target is not None:
            self.commit = (selected.action, selected.offset)
        elif target is None or rel < -2.2:
            self.commit = None
        steer, speed = selected.steering, selected.speed
        planned_speed = speed
        # LiDAR failsafe is independent of model probabilities and map fit.
        ranges = np.nan_to_num(np.asarray(obs["scan"], float), nan=0., posinf=20., neginf=0.)
        x, y = ranges*np.cos(self.angles), ranges*np.sin(self.angles)
        curv = np.tan(steer)/.3302
        if abs(curv) < 1e-3:
            lateral, along = np.abs(y), x
        else:
            radius = 1./curv
            lateral = np.abs(np.hypot(x, y-radius)-abs(radius))
            along = abs(radius)*np.arctan2(x,np.sign(radius)*(radius-y))
        forward = (x > 0) & (along > 0) & (along < 2.0) & (lateral < .195)
        released_beams = 0
        if forward.any():
            free = np.maximum(0., along[forward] - .40)
            obstacle_speed = np.zeros(len(free))
            tracked=np.zeros(len(free),dtype=bool)
            if (self.stage > 1 and target is not None and target.confidence > .6
                    and target.time_since_seen < .15):
                # Moving-car braking uses a conservative measured velocity.
                # Wall/unknown beams always retain the stationary-object limit.
                txy, tyaw, _ = self.planner.track.sample(target.s, target.d)
                eyaw = float(obs["pose_theta"])
                c, sn = np.cos(eyaw), np.sin(eyaw)
                hits = np.column_stack([float(obs["pose_x"])+c*x[forward]-sn*y[forward],
                                        float(obs["pose_y"])+sn*x[forward]+c*y[forward]])
                relative_hits = hits-txy
                longitudinal = relative_hits @ np.array([np.cos(tyaw),np.sin(tyaw)])
                transverse = relative_hits @ np.array([-np.sin(tyaw),np.cos(tyaw)])
                tracked = ((np.abs(longitudinal)<target.length/2+.12)
                           & (np.abs(transverse)<target.width/2+.08)
                           & (self.planner.track.clearance(hits)>.12))
                lower_velocity = max(0., target.speed-.3-3.*target.time_since_seen)
                obstacle_speed[tracked] = lower_velocity*max(0.,float(np.cos(tyaw-eyaw)))
            caps=np.sqrt(obstacle_speed**2+6.*free)
            response_caps = getattr(self, "response_speed_caps", None)
            if response_caps is not None:
                caps = response_caps(caps, tracked, selected, obs, target)
            if (self.stage == 3 or self.planner.pass_completion_recovery) and selected.recovery_certified:
                # A currently observed known car's side-padding beams may be
                # released only after the full moving-body departure certificate.
                # .155m physical half-width remains; .015m hard side buffer.
                # Unknown/wall returns and all beams forward of the front axle
                # preserve the original stationary/moving stopping-distance cap.
                padding_only=tracked & (x[forward]<.29) & (lateral[forward]>=.170)
                caps[padding_only]=self.vmax
                released_beams = int(padding_only.sum())
            speed = min(speed, float(caps.min()))
        self.steering += float(np.clip(steer-self.steering, -2.5 * dt, 2.5 * dt))
        actual = max(0., float(obs["linear_vel_x"]))
        self.command_speed = min(speed, max(actual, self.command_speed)+2. * dt)
        self.action = selected.action
        certified = bool((self.stage == 3 or self.planner.pass_completion_recovery) and selected.recovery_certified)
        self.padding_recovery_steps += int(certified)
        self.padding_recovery_seconds += dt if certified else 0.
        self.padding_lidar_release_steps += int(released_beams > 0)
        self.padding_lidar_released_beams += released_beams
        self.telemetry = {"action": self.action, "probabilities": raw_p.tolist(),
                          "raw_action":ACTIONS[int(np.argmax(raw_p))],
                          "action_success_probabilities": None if success_probabilities is None else success_probabilities.tolist(),
                          "safety_mask_override":int(np.argmax(raw_p))!=filtered_index,
                          "commitment_override":index!=filtered_index,
                          "safe_probabilities": p.tolist(), "features": features,
                          "candidates": [c.record() for c in candidates[:size]],
                          "target": None if target is None else {"distance":rel,"d":target.d,"speed":target.speed,
                                                                  "width":target.width,"confidence":target.confidence},
                          "planned_speed": planned_speed,
                          "padding_recovery_certified":certified,
                          "pass_completion_recovery_eligible":self.planner.pass_completion_recovery,
                          "padding_released_beams":released_beams,
                          "lidar_speed_cap": speed,
                          "control_dt": dt,
                          "command": [self.steering, self.command_speed]}
        self.counter += 1
        self.control_times.append((time.perf_counter_ns()-start)/1e6)
        return self.steering, self.command_speed

