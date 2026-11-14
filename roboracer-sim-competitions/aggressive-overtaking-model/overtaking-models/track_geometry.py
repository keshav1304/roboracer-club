"""Track coordinates and a shared lap-time-refined racing reference.

A finite numerical optimizer improves a curvature-reduced starting reference
under explicit speed, acceleration and wall constraints. This is a local
solution, not a globally minimum-time claim. No opponent state enters it.
"""
from pathlib import Path
import hashlib

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import least_squares, minimize
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree

CAR_LENGTH = .58
CAR_WIDTH = .31
WHEELBASE = .3302
MAX_STEER = .4189
REFERENCE_VERSION = "lap_time_v3"


def periodic_speed_profile(xy, curvature, vmax=2.4, lateral_acceleration=2., acceleration=2., braking=3.):
    """Maximal sampled periodic speed envelope under longitudinal/lateral limits.

    This is a quasi-static planning model, not a claim about globally optimal
    dynamic driving. Forward/backward passes propagate braking across lap zero.
    """
    xy, curvature = np.asarray(xy, float), np.asarray(curvature, float)
    if len(xy) < 3 or curvature.shape != (len(xy),) or min(vmax, lateral_acceleration, acceleration, braking) <= 0:
        raise ValueError("A finite closed path and positive dynamic limits are required")
    if not np.isfinite(xy).all() or not np.isfinite(curvature).all():
        raise ValueError("Reference samples must be finite")
    ds = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    v2 = np.minimum(float(vmax)**2, float(lateral_acceleration)/np.maximum(np.abs(curvature), .01))
    for _ in range(3):
        previous = v2.copy()
        for i in range(len(v2)):
            j = (i+1) % len(v2)
            v2[j] = min(v2[j], v2[i] + 2*acceleration*ds[i])
        for i in range(len(v2)-1, -1, -1):
            j = (i+1) % len(v2)
            v2[i] = min(v2[i], v2[j] + 2*braking*ds[i])
        if np.max(np.abs(v2-previous)) < 1e-10:
            break
    speed = np.sqrt(v2)
    lap_seconds = float(np.sum(2*ds/np.maximum(speed+np.roll(speed,-1), 1e-8)))
    return speed, lap_seconds


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


class TrackGeometry:
    """Periodic metres-along-line / left-positive offset coordinates."""

    def __init__(self, track, optimize=True, reference_mode="lap_time"):
        if reference_mode not in ("lap_time", "legacy"):
            raise ValueError("Reference mode must be lap_time or legacy")
        self.reference_mode = reference_mode if optimize else "centerline"
        self.reference_optimization = {}
        self._speed_profiles = {}
        self.resolution = float(track.spec.resolution)
        self.origin = np.asarray(track.spec.origin[:2], float)
        self.occupancy = np.asarray(track.occupancy_map)
        self.distance_field = distance_transform_edt(self.occupancy > 128) * self.resolution
        raw = np.column_stack([track.centerline.xs, track.centerline.ys]).astype(float)
        # Approximately 20 cm knots keep offline optimization small. The final
        # spline is sampled much more finely for nearest-segment projection.
        stride = max(1, round(.2 / np.median(np.linalg.norm(np.diff(raw, axis=0), axis=1))))
        knots = raw[::stride]
        digest = hashlib.sha256(raw.tobytes() + self.occupancy.tobytes()).hexdigest()[:16]
        cache = Path(__file__).parent / "artifacts" / f"reference_{digest}.npz"
        self.reference_description = "Locally curvature-reduced reference; not certified time-optimal"
        if optimize and cache.exists():
            with np.load(cache) as saved:
                knots = saved["knots"]
        elif optimize:
            knots = self._optimize(knots)
            cache.parent.mkdir(exist_ok=True)
            np.savez(cache, knots=knots, description=self.reference_description)
        if optimize and reference_mode == "lap_time":
            refined_cache = cache.with_name(f"reference_{REFERENCE_VERSION}_{digest}.npz")
            if refined_cache.exists():
                with np.load(refined_cache, allow_pickle=False) as saved:
                    knots = saved["knots"]
                    self.reference_optimization = {"predicted_before_seconds": float(saved["before"]),
                        "predicted_after_seconds": float(saved["after"]),
                        "minimum_planned_body_clearance_m": float(saved["minimum_clearance"]),
                        "optimizer_converged": bool(saved["optimizer_converged"]),
                        "accepted": bool(saved["accepted"])}
            else:
                knots, self.reference_optimization = self._lap_time_refine(knots)
                np.savez(refined_cache, knots=knots,
                         before=self.reference_optimization["predicted_before_seconds"],
                         after=self.reference_optimization["predicted_after_seconds"],
                         minimum_clearance=self.reference_optimization["minimum_planned_body_clearance_m"],
                         optimizer_converged=self.reference_optimization["optimizer_converged"],
                         accepted=self.reference_optimization["accepted"])
            self.reference_description = "Shared locally lap-time-refined racing line; finite optimizer, not certified globally optimal"
        self.reference_hash = hashlib.sha256(knots.tobytes()).hexdigest()
        self.reference_digest = self.reference_hash
        lengths = np.linalg.norm(np.roll(knots, -1, axis=0) - knots, axis=1)
        parameter = np.r_[0., np.cumsum(lengths)]
        spline = CubicSpline(parameter, np.vstack([knots, knots[0]]), bc_type="periodic", axis=0)
        dense = spline(np.linspace(0., parameter[-1], round(parameter[-1] / .025), endpoint=False))
        lengths = np.linalg.norm(np.roll(dense, -1, axis=0) - dense, axis=1)
        self.s = np.r_[0., np.cumsum(lengths)[:-1]]
        self.length = float(lengths.sum())
        self.xy = dense
        self._spline = CubicSpline(np.r_[self.s, self.length], np.vstack([dense, dense[0]]), bc_type="periodic", axis=0)
        self._tree = cKDTree(dense)
        self.yaw = np.arctan2(self._spline(self.s, 1)[:, 1], self._spline(self.s, 1)[:, 0])
        self.kappa = self.sample(self.s)[2]
        # Lateral bounds stop at the FIRST wall, not a neighboring track lane.
        lateral = np.arange(0., 3.01, .02)
        n = np.column_stack([-np.sin(self.yaw), np.cos(self.yaw)])
        self._left = self._ray_width(dense[:, None, :] + n[:, None, :] * lateral[None, :, None], lateral)
        self._right = self._ray_width(dense[:, None, :] - n[:, None, :] * lateral[None, :, None], lateral)

    def _ray_width(self, points, distance):
        free = self.clearance(points) > .015
        first = np.argmax(~free, axis=1)
        first[np.all(free, axis=1)] = len(distance) - 1
        return np.maximum(0., distance[first] - self.resolution)

    def _optimize(self, xy):
        npts = len(xy)
        velocity = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
        normals = np.column_stack([-velocity[:, 1], velocity[:, 0]]) / np.linalg.norm(velocity, axis=1)[:, None]
        # Conservative bounded offsets keep the approximation in this lane.
        # Final geometry is checked against the actual occupancy map below.
        bound = np.minimum(.28, np.maximum(.01, self.clearance(xy) - .43))

        def residual(offset):
            points = xy + normals * offset[:, None]
            v = (np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)) / 2
            a = np.roll(points, -1, axis=0) - 2 * points + np.roll(points, 1, axis=0)
            ds = np.maximum(.01, np.linalg.norm(v, axis=1))
            curvature = (v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]) / ds ** 3
            return np.r_[curvature * np.sqrt(ds), .07 * offset]

        sparsity = lil_matrix((2 * npts, npts))
        for i in range(npts):
            sparsity[i, [(i - 1) % npts, i, (i + 1) % npts]] = 1
            sparsity[npts + i, i] = 1
        result = least_squares(residual, np.zeros(npts), bounds=(-bound, bound),
                               jac_sparsity=sparsity.tocsr(), max_nfev=100)
        optimized = xy + normals * result.x[:, None]
        if self.clearance(optimized).min() < .40:
            return xy
        return optimized

    @staticmethod
    def _closed_geometry(xy):
        velocity = (np.roll(xy,-1,axis=0)-np.roll(xy,1,axis=0))/2
        second = np.roll(xy,-1,axis=0)-2*xy+np.roll(xy,1,axis=0)
        yaw = np.arctan2(velocity[:,1],velocity[:,0])
        curvature = (velocity[:,0]*second[:,1]-velocity[:,1]*second[:,0])/np.maximum(np.linalg.norm(velocity,axis=1),1e-6)**3
        return yaw, curvature

    def _lap_time_refine(self, original):
        lengths = np.linalg.norm(np.roll(original,-1,axis=0)-original,axis=1)
        arc = np.r_[0.,np.cumsum(lengths)]
        spline = CubicSpline(arc,np.vstack([original,original[0]]),bc_type="periodic",axis=0)
        ss = np.arange(0.,arc[-1],.20)
        base = spline(ss)
        velocity = spline(ss,1)
        normal = np.column_stack([-velocity[:,1],velocity[:,0]])/np.linalg.norm(velocity,axis=1)[:,None]
        controls = np.linspace(0.,arc[-1],33)
        identity = np.eye(32)
        basis = CubicSpline(controls,np.vstack([identity,identity[0]]),axis=0,bc_type="periodic")(ss)
        def evaluate(offset, details=False):
            xy = base + normal*(basis@offset)[:,None]
            yaw,curvature = self._closed_geometry(xy)
            clearance = self.body_clearance(xy,yaw)
            _,lap = periodic_speed_profile(xy,curvature)
            penalty = 1e4*np.mean(np.maximum(.15-clearance,0.)**2) + .0001*np.sum(offset**2)
            return (xy,lap,clearance) if details else lap+penalty
        before = evaluate(np.zeros(32),True)[1]
        result = minimize(evaluate,np.zeros(32),method="L-BFGS-B",bounds=[(-.25,.25)]*32,
                          options={"maxiter":55,"maxls":15,"ftol":1e-8})
        candidate,after,clearance = evaluate(result.x,True)
        # Check the interpolated path too, not just optimizer knots.
        candidate_lengths = np.linalg.norm(np.roll(candidate,-1,axis=0)-candidate,axis=1)
        candidate_arc = np.r_[0.,np.cumsum(candidate_lengths)]
        fine_spline = CubicSpline(candidate_arc,np.vstack([candidate,candidate[0]]),axis=0,bc_type="periodic")
        fine_s = np.arange(0,candidate_arc[-1],.025)
        fine_xy,fine_velocity = fine_spline(fine_s),fine_spline(fine_s,1)
        fine_yaw = np.arctan2(fine_velocity[:,1],fine_velocity[:,0])
        minimum = float(self.body_clearance(fine_xy,fine_yaw).min())
        accepted = after < before and minimum >= .18
        if not accepted:
            candidate,after = original,before
            yaw,_ = self._closed_geometry(original)
            minimum = float(self.body_clearance(original,yaw).min())
        return candidate,{"predicted_before_seconds":before,"predicted_after_seconds":after,
                          "minimum_planned_body_clearance_m":minimum,
                          "optimizer_converged":bool(result.success),"accepted":bool(accepted),
                          "method":"32 periodic lateral controls, 55 L-BFGS-B iterations; 2.4m/s cap, 2m/s² lateral/accel, 3m/s² brake"}

    def speed_at(self, s, vmax=2.4, lateral_acceleration=2., acceleration=2., braking=3.):
        """Shared periodic reference speed plan, cached once per physical limits."""
        key = tuple(float(v) for v in (vmax,lateral_acceleration,acceleration,braking))
        if key not in self._speed_profiles:
            self._speed_profiles[key] = periodic_speed_profile(self.xy,self.kappa,*key)
        speed,_ = self._speed_profiles[key]
        return np.interp(np.asarray(s)%self.length,self.s,speed,period=self.length)

    def predicted_lap_time(self, vmax=2.4, lateral_acceleration=2., acceleration=2., braking=3.):
        key = tuple(float(v) for v in (vmax,lateral_acceleration,acceleration,braking))
        self.speed_at(0.,*key)
        return self._speed_profiles[key][1]

    def sample(self, s, d=0.):
        """Return (xy, heading, curvature), accepting scalars or broadcast arrays."""
        s, d = np.broadcast_arrays(np.asarray(s, float), np.asarray(d, float))
        position = self._spline(s % self.length)
        velocity = self._spline(s % self.length, 1)
        acceleration = self._spline(s % self.length, 2)
        yaw = np.arctan2(velocity[..., 1], velocity[..., 0])
        normal = np.stack([-np.sin(yaw), np.cos(yaw)], axis=-1)
        curvature = (velocity[..., 0] * acceleration[..., 1] - velocity[..., 1] * acceleration[..., 0]) / np.maximum(np.linalg.norm(velocity, axis=-1), 1e-6) ** 3
        return position + d[..., None] * normal, yaw, curvature

    def project(self, x, y):
        """Continuous nearest-segment projection, not a quantized waypoint index."""
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        points = np.stack([x, y], axis=-1)
        shape = x.shape
        flat = points.reshape(-1, 2)
        _, near = self._tree.query(flat)
        indices = np.stack([(near - 1) % len(self.xy), near], axis=1)
        starts = self.xy[indices]
        ends = self.xy[(indices + 1) % len(self.xy)]
        segment = ends - starts
        frac = np.clip(np.sum((flat[:, None, :] - starts) * segment, axis=-1) / np.sum(segment * segment, axis=-1), 0., 1.)
        foot = starts + frac[..., None] * segment
        best = np.argmin(np.sum((flat[:, None, :] - foot) ** 2, axis=-1), axis=1)
        row = np.arange(len(flat))
        ind = indices[row, best]
        s = (self.s[ind] + frac[row, best] * np.linalg.norm(segment[row, best], axis=1)) % self.length
        pos, yaw, _ = self.sample(s)
        delta = flat - pos
        d = -np.sin(yaw) * delta[:, 0] + np.cos(yaw) * delta[:, 1]
        s, d = s.reshape(shape), d.reshape(shape)
        return (float(s), float(d)) if not shape else (s, d)

    def delta_s(self, from_s, to_s):
        return (np.asarray(to_s) - np.asarray(from_s) + self.length / 2) % self.length - self.length / 2

    def widths(self, s):
        """Distance from reference to first left/right wall, in metres."""
        positions = np.asarray(s) % self.length
        return (np.interp(positions, self.s, self._left, period=self.length),
                np.interp(positions, self.s, self._right, period=self.length))

    def clearance(self, xy):
        points = np.asarray(xy, float)
        shape = points.shape[:-1]
        coords = ((points.reshape(-1, 2) - self.origin) / self.resolution).T[::-1]
        return map_coordinates(self.distance_field, coords, order=1, mode="constant", cval=0.).reshape(shape)

    def body_clearance(self, xy, yaw, length=CAR_LENGTH, width=CAR_WIDTH):
        """Conservative four-circle bound enclosing the rectangular car.

        Negative means the enclosing footprint intersects a map wall. This
        conservative clearance is for planning, separate from exact collision
        reporting in the simulator.
        """
        points = np.asarray(xy, float)
        yaw = np.asarray(yaw, float)
        forward = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)
        offsets = np.linspace(-3 * length / 8, 3 * length / 8, 4)
        radius = np.hypot(length / 8, width / 2)
        centres = points[..., None, :] + forward[..., None, :] * offsets[:, None]
        return np.min(self.clearance(centres), axis=-1) - radius


class PathTracker:
    """Pure pursuit on a reference with an optional local lateral detour."""

    def __init__(self, track, lateral_acceleration=2., acceleration=2., braking=3.):
        self.track = track
        self.lateral_acceleration = lateral_acceleration
        self.acceleration = acceleration
        self.braking = braking
        self.last_steer = 0.

    def control(self, obs, offset=0., vmax=2.5, dt=.01, smooth=True):
        x, y, yaw, speed = (float(obs[k]) for k in ("pose_x", "pose_y", "pose_theta", "linear_vel_x"))
        s, d = self.track.project(x, y)
        lookahead = float(np.clip(.65 + .14 * max(speed, 0.), .65, 1.1))
        distances = np.linspace(0., max(3., speed * 1.5), 25)
        ss = s + distances
        lateral = offset(ss) if callable(offset) else np.full_like(ss, offset)
        points, _, curvature = self.track.sample(ss, lateral)
        # Evaluate the detour's curvature as well as the reference curvature.
        gradients = np.gradient(points, distances, axis=0)
        second = np.gradient(gradients, distances, axis=0)
        curv = (gradients[:, 0] * second[:, 1] - gradients[:, 1] * second[:, 0]) / np.maximum(np.linalg.norm(gradients, axis=1), .1) ** 3
        curve_speed = np.sqrt(self.lateral_acceleration / np.maximum(np.abs(curv), .03))
        allowed = np.sqrt(curve_speed ** 2 + 2 * self.braking * np.maximum(0., distances - .35))
        target_speed = min(float(vmax), float(allowed.min()))
        if not callable(offset) and abs(float(offset)) < 1e-8:
            target_speed = min(target_speed,float(self.track.speed_at(s,vmax,self.lateral_acceleration,self.acceleration,self.braking)))
        target_d = float(offset(s + lookahead)) if callable(offset) else float(offset)
        target_xy, _, _ = self.track.sample(s + lookahead, target_d)
        delta = target_xy - [x, y]
        local_y = -np.sin(yaw) * delta[0] + np.cos(yaw) * delta[1]
        steer = np.arctan2(2 * WHEELBASE * local_y, max(.05, float(delta @ delta)))
        steer = float(np.clip(steer, -MAX_STEER, MAX_STEER))
        if smooth:
            steer = float(np.clip(steer, self.last_steer - 3. * dt, self.last_steer + 3. * dt))
            self.last_steer = steer
            target_speed = min(target_speed, max(0., speed) + self.acceleration * dt)
        return steer, max(0., target_speed)
