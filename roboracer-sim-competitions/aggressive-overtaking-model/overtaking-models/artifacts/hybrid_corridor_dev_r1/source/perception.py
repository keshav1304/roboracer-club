"""Scan-to-map residual clustering and short-horizon opponent tracking.

Inputs are the ego's LiDAR, localization and odometry plus the static map.
No simulator opponent position, speed or future action is used. Vehicle-sized
occluded faces require a size prior, recorded in the Target estimate.
"""
from dataclasses import dataclass, replace
import numpy as np


@dataclass
class Target:
    s: float
    d: float
    speed: float = 0.
    length: float = .58
    width: float = .31
    confidence: float = .2
    lateral_speed: float = 0.
    time_since_seen: float = 0.
    observations: int = 1
    observed_seconds: float = 0.
    longitudinal_observed: bool = True
    lateral_observed: bool = True
    endpoint_only: bool = False


class LidarTracker:
    def __init__(self, track, angles, max_range=9., coast_time=.6):
        self.track = track
        self.angles = np.asarray(angles, float)
        self.max_range = float(max_range)
        self.coast_time = float(coast_time)
        self.target = None
        self.time = 0.
        self.history = []
        self.velocity_initialized = False

    def reset(self):
        self.target = None
        self.history = []
        self.time = 0.
        self.velocity_initialized = False

    def _initial_velocity(self, observed, minimum_points=10):
        """Fit first motion from consistent measured displacement, not ego speed.

        Ten visible longitudinal measurements spanning at least 120 ms reject
        isolated scan jumps. This avoids filtering a plainly moving car up
        from a fictitious stationary prior for several additional tenths.
        """
        if len(observed) < minimum_points or observed[-1][0] - observed[0][0] < .12:
            return None
        times = np.array([h[0] - observed[0][0] for h in observed])
        distance = np.array([self.track.delta_s(observed[0][1], h[1]) for h in observed])
        centred_t = times - times.mean()
        denominator = float(centred_t @ centred_t)
        if denominator <= 1e-10:
            return None
        velocity = float(centred_t @ (distance - distance.mean()) / denominator)
        residual = distance - (distance.mean() + velocity * centred_t)
        if (not -1. <= velocity <= 6. or np.sqrt(np.mean(residual**2)) > .025
                or np.max(np.abs(residual)) > .06):
            return None
        return max(0., velocity)

    def _fit_vehicle(self, points, obs, dt):
        """Fit visible rectangle faces, retaining unobserved axes from motion.

        A side-only scan does not measure the centre along the side. Likewise,
        a cropped rear/front face does not measure lateral centre. Using the
        midpoint of that cropped cloud invents motion as a car leaves the
        LiDAR field of view. The previous state supplies those missing axes.
        """
        prior = self.target
        predicted_s = (prior.s + prior.speed * dt) % self.track.length
        predicted_d = prior.d + prior.lateral_speed * dt
        reference, yaw, _ = self.track.sample(predicted_s)
        forward = np.array([np.cos(yaw), np.sin(yaw)])
        normal = np.array([-np.sin(yaw), np.cos(yaw)])
        cloud = np.column_stack([(points - reference) @ forward, (points - reference) @ normal])
        observer_delta = np.array([float(obs["pose_x"]), float(obs["pose_y"])]) - reference
        observer = np.array([observer_delta @ forward, observer_delta @ normal])
        centre = np.array([0., predicted_d])
        half = np.array([.29, .155])
        observed = [False, False]
        # Alternating face assignment and robust plane fitting is a tiny
        # translation-only ICP. The reference tangent supplies the yaw prior.
        for _ in range(3):
            side = np.where(observer >= centre, 1., -1.)
            visible = np.abs(observer - centre) > half + .015
            residual = np.abs(cloud - (centre + side * half))
            residual[:, ~visible] = np.inf
            assigned = np.argmin(residual, axis=1)
            for axis in (0, 1):
                members = cloud[assigned == axis]
                if not visible[axis] or len(members) < 3:
                    continue
                proposal = float(np.median(members[:, axis]) - side[axis] * half[axis])
                if abs(proposal - centre[axis]) > .18:
                    continue
                centre[axis] = proposal
                observed[axis] = True
        # Full visible edges constrain their midpoint even when the other
        # face is hidden. A cropped edge intentionally does not.
        low, high = np.quantile(cloud, [.02, .98], axis=0)
        # Before velocity is measured, the displacement gate must allow a
        # physical tick of motion (2.4m/s already travels .12m at20Hz).
        # The subsequent multi-frame velocity fit still rejects inconsistent jumps.
        midpoint_gate = max(.12, 6.*dt+.04) if not self.velocity_initialized else .12
        for axis in (0, 1):
            if not observed[axis] and high[axis] - low[axis] > 1.65 * half[axis]:
                proposal = (high[axis] + low[axis]) / 2
                if abs(proposal - centre[axis]) < midpoint_gate:
                    centre[axis] = proposal
                    observed[axis] = True
        position = reference + centre[0] * forward + centre[1] * normal
        s, d = self.track.project(*position)
        return float(s), float(d), observed

    def _detect(self, obs, dt=.01):
        scan = np.asarray(obs["scan"], float)
        if scan.shape != self.angles.shape:
            raise ValueError("LiDAR scan must match scan angles")
        valid = np.isfinite(scan) & (scan > .12) & (scan < self.max_range)
        yaw = float(obs["pose_theta"])
        rays = self.angles + yaw
        ranges = np.where(valid, scan, 0.)
        points = np.column_stack([float(obs["pose_x"]) + ranges * np.cos(rays),
                                  float(obs["pose_y"]) + ranges * np.sin(rays)])
        # Wall range noise and map rasterization errors are below this margin.
        valid &= self.track.clearance(np.nan_to_num(points)) > .12
        indices = np.flatnonzero(valid)
        if not len(indices):
            return []
        breaks = np.flatnonzero((np.diff(indices) > 3) |
                               (np.linalg.norm(np.diff(points[indices], axis=0), axis=1) > .25)) + 1
        clusters = np.split(indices, breaks)
        ego_s, ego_d = self.track.project(float(obs["pose_x"]), float(obs["pose_y"]))
        detections = []
        for cluster in clusters:
            if len(cluster) < 4:
                continue
            s, d = self.track.project(points[cluster, 0], points[cluster, 1])
            relative = self.track.delta_s(ego_s, s)
            low_s, high_s = np.quantile(relative, [.05, .95])
            low_d, high_d = np.quantile(d, [.05, .95])
            if high_s - low_s > 2.5 or high_d - low_d > 3.0:
                continue
            length = max(.58, float(high_s - low_s))
            width = max(.31, float(high_d - low_d))
            centre_s = (low_s + high_s) / 2
            centre_d = (low_d + high_d) / 2
            # A distant leading car exposes its rear face, not its centre.
            # Beside the ego, use the visible side face instead.
            # At a close start the rear bumper is only .35-.60 m ahead. A
            # broad transverse face still measures a bumper, not the centre.
            # Missing that half-length correction leaves a .29 m error which
            # the later ICP jump gate deliberately cannot repair. Require face
            # shape evidence here so cropped side returns are not shifted.
            close_end_face = high_s - low_s < .12 and high_d - low_d > .20
            if (centre_s > .6 or (centre_s > .05 and close_end_face)) and high_s - low_s < .48:
                centre_s = low_s + length / 2
            elif centre_s < -.05 and high_s - low_s < .48:
                centre_s = high_s - length / 2
            if high_d - low_d < .20 and abs(centre_s) < 2.:
                centre_d = high_d - width / 2 if centre_d < ego_d else low_d + width / 2
            # Ignore residuals from adjacent lanes unless a track is already
            # being followed nearby. This is a single-target tracker.
            if abs(centre_s) > self.max_range:
                continue
            target_s = float((ego_s + centre_s) % self.track.length)
            observed = [True, True]
            # A prior is needed to resolve centres from partly visible faces.
            # Wide blockers keep their measured extent instead of being forced
            # into the vehicle-size model.
            if (self.target is not None and self.target.width <= .45 and width <= .45
                    and abs(self.track.delta_s(self.target.s, target_s)) < 1.0):
                target_s, centre_d, observed = self._fit_vehicle(points[cluster], obs, dt)
                width, length = .31, .58
            # Rear FOV cropping hides the rear endpoint, but the front
            # endpoint is visible at the other end of the same side cluster.
            endpoint_only = ((cluster[0] <= 2 or cluster[-1] >= len(scan)-3)
                             and high_d-low_d < .20 and .08 < high_s-low_s < .58)
            if endpoint_only:
                target_s = float((ego_s + high_s - length/2) % self.track.length)
                observed[0] = True
            detections.append(Target(target_s, float(centre_d), length=length, width=width, endpoint_only=endpoint_only,
                                     confidence=min(.45, .15 + len(cluster) / 100),
                                     longitudinal_observed=observed[0], lateral_observed=observed[1]))
        return detections

    def update(self, obs, dt=.01):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.time += dt
        detections = self._detect(obs, dt)
        ego_s, _ = self.track.project(float(obs["pose_x"]), float(obs["pose_y"]))
        selected = None
        if self.target is not None and detections:
            predicted_s = (self.target.s + self.target.speed * dt) % self.track.length
            distances = [np.hypot(self.track.delta_s(predicted_s, d.s), d.d - self.target.d) for d in detections]
            nearest = int(np.argmin(distances))
            if distances[nearest] < 1.:
                selected = detections[nearest]
        if selected is None and self.target is None and detections:
            ahead = [d for d in detections if -.8 < self.track.delta_s(ego_s, d.s) < self.max_range]
            if ahead:
                selected = min(ahead, key=lambda d: max(0., self.track.delta_s(ego_s, d.s)))
        if selected is None:
            if self.target is None:
                return None
            age = self.target.time_since_seen + dt
            if age > self.coast_time:
                self.target = None
                self.history.clear()
                self.velocity_initialized = False
                return None
            # Do not extrapolate a spurious lateral velocity indefinitely into
            # the rear blind region. Longitudinal motion is better observed.
            lateral_decay = float(np.exp(-dt / .20))
            self.target = replace(self.target,
                                  s=float((self.target.s + self.target.speed * dt) % self.track.length),
                                  d=self.target.d + self.target.lateral_speed * lateral_decay * dt,
                                  lateral_speed=self.target.lateral_speed * lateral_decay,
                                  time_since_seen=age,
                                  confidence=max(0., self.target.confidence - dt))
            return self.target
        old = self.target
        if old is None:
            self.velocity_initialized = False
            self.history.clear()
        observations = 1 if old is None else old.observations + 1
        # Use a 150–350 ms displacement baseline, avoiding differentiation of
        # adjacent noisy scans. Clamp physically implausible estimates.
        self.history.append((self.time, selected.s, selected.d,
                             selected.longitudinal_observed, selected.lateral_observed))
        # A longer initial window also supports lower-rate sensor recordings
        # while still requiring ten actual visible measurements.
        window = .35 if self.velocity_initialized else .6
        self.history = [h for h in self.history if self.time - h[0] <= window]
        speed, lateral_speed = (0., 0.) if old is None else (old.speed, old.lateral_speed)
        observed_s = [h for h in self.history if h[3]]
        observed_d = [h for h in self.history if h[4]]
        alpha = min(1., dt / .16)
        if selected.longitudinal_observed and observed_s and self.time - observed_s[0][0] >= .12:
            if not self.velocity_initialized:
                initial = self._initial_velocity(observed_s, minimum_points=max(4, round(.10 / dt)))
                if initial is not None:
                    speed = initial
                    self.velocity_initialized = True
            else:
                past_time, past_s = observed_s[0][:2]
                measured_v = np.clip(self.track.delta_s(past_s, selected.s) / (self.time - past_time), -1., 6.)
                speed += alpha * (measured_v - speed)
        if selected.lateral_observed and observed_d and self.time - observed_d[0][0] >= .12:
            past_time, _, past_d = observed_d[0][:3]
            measured_dv = np.clip((selected.d - past_d) / (self.time - past_time), -.6, .6)
            lateral_speed += alpha * (measured_dv - lateral_speed)
        elif not selected.lateral_observed:
            lateral_speed *= np.exp(-dt / .20)
        # Position filtering is mild so approaching stationary objects does
        # not lag into a collision. Tracker uncertainty is explicit.
        if old is not None:
            position_alpha = 1. - .6 ** (dt / .01)
            selected.s = float((old.s + old.speed * dt + position_alpha * self.track.delta_s(old.s + old.speed * dt, selected.s)) % self.track.length)
            selected.d = float(old.d + old.lateral_speed * dt + position_alpha * (selected.d - old.d - old.lateral_speed * dt))
        selected.speed = float(max(0., speed))
        selected.lateral_speed = float(lateral_speed)
        selected.observations = observations
        selected.observed_seconds = dt + (old.observed_seconds if old is not None else 0.)
        certainty_cap = .98 if selected.longitudinal_observed and selected.lateral_observed and not selected.endpoint_only else .75
        selected.confidence = float(min(certainty_cap, .2 + selected.observed_seconds * 2.5))
        selected.time_since_seen = 0.
        self.target = selected
        return selected
