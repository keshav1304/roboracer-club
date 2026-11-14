"""Measured motion startup, stationary targets, and occlusion reset."""
import unittest
from unittest.mock import patch

import numpy as np

from perception import LidarTracker, Target


class StraightTrack:
    length = 100.

    def project(self, x, y):
        return x, y

    def delta_s(self, start, end):
        return (np.asarray(end) - start + self.length / 2) % self.length - self.length / 2


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.tracker = LidarTracker(StraightTrack(), np.linspace(-2., 2., 1080))
        # Deliberately misleading ego speed must never become target speed.
        self.obs = {"pose_x": 0., "pose_y": 0., "linear_vel_x": 19.}

    def observe(self, position, visible=True, dt=.01):
        detection = Target(position % 100, 0., longitudinal_observed=visible)
        with patch.object(self.tracker, "_detect", return_value=[detection]):
            return self.tracker.update(self.obs, dt=dt)

    def test_consistent_moving_measurements_initialize_without_zero_prior_lag(self):
        for i in range(15):
            target = self.observe(3. + 2.4 * i * .01 + .002 * np.sin(i))
            if i < 10:
                self.assertFalse(self.tracker.velocity_initialized)
        self.assertTrue(self.tracker.velocity_initialized)
        self.assertAlmostEqual(target.speed, 2.4, delta=.04)

    def test_stationary_measurements_remain_stationary(self):
        for i in range(20):
            target = self.observe(3. + .001 * np.sin(i))
        self.assertTrue(self.tracker.velocity_initialized)
        self.assertLess(target.speed, .03)

    def test_inconsistent_measurements_do_not_create_a_motion_prior(self):
        for i in range(24):
            target = self.observe(3. + 2.4 * i * .01 + (.12 if i % 2 else -.12))
        self.assertFalse(self.tracker.velocity_initialized)
        self.assertEqual(target.speed, 0.)

    def test_unobserved_longitudinal_axis_does_not_supply_startup_samples(self):
        for i in range(24):
            self.observe(3. + 2.4 * i * .01, visible=i % 3 == 0)
        self.assertFalse(self.tracker.velocity_initialized)

    def test_coast_expiry_resets_motion_and_stationary_reacquisition(self):
        for i in range(20):
            self.observe(3. + 2.4 * i * .01)
        with patch.object(self.tracker, "_detect", return_value=[]):
            for _ in range(61):
                self.tracker.update(self.obs)
        self.assertIsNone(self.tracker.target)
        self.assertFalse(self.tracker.velocity_initialized)
        for _ in range(20):
            target = self.observe(4.)
        self.assertLess(target.speed, .01)

    def test_confidence_and_velocity_use_time_at_twenty_hz(self):
        for i in range(4):
            target = self.observe(3. + 2.4*i*.05, dt=.05)
        self.assertTrue(self.tracker.velocity_initialized)
        self.assertAlmostEqual(target.speed, 2.4, delta=.02)
        self.assertGreater(target.confidence, .6)
        self.assertAlmostEqual(target.observed_seconds, .2)

    def test_startup_handles_reference_wrap_and_lower_sensor_rate(self):
        self.obs["pose_x"] = 97.
        for i in range(11):
            target = self.observe(99.4 + 2.4 * i * .05, dt=.05)
        self.assertTrue(self.tracker.velocity_initialized)
        self.assertAlmostEqual(target.speed, 2.4, delta=.01)


if __name__ == "__main__":
    unittest.main()
