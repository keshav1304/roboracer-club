"""Smooth and brake local controllers using LiDAR and speed only."""
import numpy as np
from reactive import FollowTheGap, FTGParams


class ReactiveDriver:
    """Apply actuator limits and scan-based braking to a reactive controller."""
    def __init__(self, controller, angles, v_max, *, steering_rate=1.6,
                 acceleration=1.2, lateral_acceleration=1.5, stop_distance=.55, corridor_margin=.10):
        self.controller = controller
        self.clearance = FollowTheGap(angles, FTGParams(corridor_margin=corridor_margin))
        self.v_max = v_max
        self.steering_step = steering_rate * .01
        self.acceleration_step = acceleration * .01
        self.lateral_acceleration = lateral_acceleration
        self.stop_distance = stop_distance
        self.steering = 0.
        self.speed = 0.
        self.max_steering_rate = 0.

    def plan(self, scan, velocity):
        scan = np.clip(np.nan_to_num(np.asarray(scan, float), nan=10., posinf=10., neginf=0.), 0, 30)
        raw_steer, raw_speed = self.controller.plan(scan, velocity)
        desired_steer = float(np.clip(np.nan_to_num(raw_steer), -.4189, .4189))
        previous = self.steering
        self.steering += float(np.clip(desired_steer - self.steering, -self.steering_step, self.steering_step))
        self.max_steering_rate = max(self.max_steering_rate, abs(self.steering - previous) / .01)
        clearance = min(self.clearance._arc_clearance(scan, self.steering),
                        self.clearance._arc_clearance(scan, desired_steer))
        braking_cap = np.sqrt(max(6 * (clearance - self.stop_distance), 0.))
        curvature = abs(np.tan(self.steering) / .3302)
        corner_cap = np.sqrt(self.lateral_acceleration / max(curvature, .01))
        target_speed = min(self.v_max, max(0., float(np.nan_to_num(raw_speed))),
                           braking_cap, corner_cap)
        self.speed = max(0., self.speed + float(np.clip(target_speed - self.speed, -.03, self.acceleration_step)))
        # A lateral proximity limit is a fresh scan safety constraint and
        # must not wait for the normal command-deceleration ramp.
        self.speed = min(self.speed, getattr(self.controller, "proximity_speed_cap", self.v_max))
        if braking_cap < .05 or raw_speed <= 0 or np.max(scan) < .35:
            self.speed = 0.
        return self.steering, self.speed


