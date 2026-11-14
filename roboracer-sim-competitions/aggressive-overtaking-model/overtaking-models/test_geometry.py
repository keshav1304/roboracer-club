"""Geometry and perception checks on the actual imported map."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gym-two-car-reactive"))
from run_two_cars import make_env
from track_geometry import TrackGeometry, PathTracker
from perception import LidarTracker


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_env("autodrive-default", False)
        cls.track = TrackGeometry(cls.env.unwrapped.track)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def reset_poses(self, ss):
        poses = np.array([[*self.track.sample(s)[0], self.track.sample(s)[1]] for s in ss])
        return self.env.reset(options={"poses": poses})[0]

    def test_periodic_reference_and_projection(self):
        g = self.track
        ss = np.linspace(-1., g.length + 1., 141)
        xy, _, _ = g.sample(ss, .1)
        projected, lateral = g.project(xy[:, 0], xy[:, 1])
        np.testing.assert_allclose(g.delta_s(ss, projected), 0., atol=.015)
        np.testing.assert_allclose(lateral, .1, atol=.005)
        np.testing.assert_allclose(g.sample(0.)[0], g.sample(g.length)[0], atol=1e-8)

    def test_reference_has_body_room(self):
        g = self.track
        self.assertGreater(float(g.body_clearance(g.xy, g.yaw).min()), .08)
        self.assertGreater(float(np.min(g.widths(g.s))), .3)
        self.assertEqual(float(g.clearance(np.array([-1000., -1000.]))), 0.)
        self.assertLess(float(g.body_clearance(np.array([-1000., -1000.]), 0.)), 0.)

    def test_tracker_does_not_invent_wall_obstacles(self):
        # Opponent is half a lap away and initially occluded by static walls.
        obs = self.reset_poses([0., self.track.length / 2])
        tracker = LidarTracker(self.track, self.env.unwrapped.sim.agents[0].scan_angles)
        for _ in range(30):
            self.assertIsNone(tracker.update(obs["agent_0"]))
            obs, *_ = self.env.step(np.zeros((2, 2)))

    def test_stationary_car_detected_from_lidar(self):
        obs = self.reset_poses([0., 3.])
        tracker = LidarTracker(self.track, self.env.unwrapped.sim.agents[0].scan_angles)
        for _ in range(30):
            target = tracker.update(obs["agent_0"])
            obs, *_ = self.env.step(np.zeros((2, 2)))
        self.assertIsNotNone(target)
        self.assertAlmostEqual(float(self.track.delta_s(3., target.s)), 0., delta=.20)
        self.assertAlmostEqual(target.d, 0., delta=.18)
        self.assertLess(target.speed, .10)
        self.assertGreater(target.confidence, .8)

    def test_invalid_scans_and_occlusion_expiry(self):
        obs = self.reset_poses([0., 3.])
        tracker = LidarTracker(self.track, self.env.unwrapped.sim.agents[0].scan_angles)
        self.assertIsNotNone(tracker.update(obs["agent_0"]))
        obs["agent_0"]["scan"] = np.full_like(obs["agent_0"]["scan"], np.nan)
        self.assertIsNotNone(tracker.update(obs["agent_0"]))
        for _ in range(61):
            target = tracker.update(obs["agent_0"])
        self.assertIsNone(target)

    def test_side_to_rear_visibility_does_not_invent_motion(self):
        """Rescan an overtaking geometry including the rear LiDAR cutoff.

        True poses are used only to generate/evaluate observations. The
        tracker receives the same single-car observation as during a race.
        """
        tracker = LidarTracker(self.track, self.env.unwrapped.sim.agents[0].scan_angles)
        errors = []
        lower_confidence = False
        for t in np.arange(0., 4.05, .05):
            ego_s, target_s = 1.5 + 2.0 * t, 3.0 + 1.2 * t
            ego_xy, ego_yaw, _ = self.track.sample(ego_s, -.55)
            target_xy, target_yaw, _ = self.track.sample(target_s)
            obs, _ = self.env.reset(options={"poses": np.array([[*ego_xy, ego_yaw], [*target_xy, target_yaw]])})
            target = tracker.update(obs["agent_0"], dt=.05)
            if target is not None and t > .8:
                errors.append([abs(float(self.track.delta_s(target_s, target.s))), abs(target.d), abs(target.speed - 1.2)])
                if t > 2.0 and target.confidence <= .75:
                    lower_confidence = True
        self.assertGreater(len(errors), 30)
        max_errors = np.max(errors, axis=0)
        self.assertLess(max_errors[0], .10, f"Longitudinal centre error: {max_errors}")
        self.assertLess(max_errors[1], .05, f"Lateral centre error: {max_errors}")
        self.assertLess(max_errors[2], .16, f"Speed error: {max_errors}")
        self.assertTrue(lower_confidence)

    def test_pursuit_limited_forward_commands(self):
        obs = self.reset_poses([0., self.track.length / 2])
        controller = PathTracker(self.track)
        steer, speed = controller.control(obs["agent_0"], offset=lambda s: np.zeros_like(s), vmax=2.)
        self.assertLessEqual(abs(steer), .4189)
        self.assertGreaterEqual(speed, 0.)
        self.assertLessEqual(speed, 2.)


if __name__ == "__main__":
    unittest.main()
