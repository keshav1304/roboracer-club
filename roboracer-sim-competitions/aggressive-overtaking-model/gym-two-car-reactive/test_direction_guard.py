"""Direction is progress around the circuit, including its legitimate bends."""
from types import SimpleNamespace
import unittest
import numpy as np
from direction_guard import ForwardRoute, CheckpointMonitor
from reactive_safety import ReactiveDriver


class DirectionTests(unittest.TestCase):
    def setUp(self):
        theta = np.linspace(0, 2 * np.pi, 600, endpoint=False)
        line = SimpleNamespace(xs=3 * np.cos(theta), ys=3 * np.sin(theta), yaws=theta + np.pi / 2,
                               ks=np.full(600, 1 / 3), vxs=np.full(600, 2.))
        self.track = SimpleNamespace(raceline=line)
        self.angles = np.linspace(-3 * np.pi / 4, 3 * np.pi / 4, 1080)

    def observation(self, theta, velocity=1.):
        return {"pose_x": 3 * np.cos(theta), "pose_y": 3 * np.sin(theta),
                "pose_theta": theta + np.pi / 2, "linear_vel_x": velocity,
                "scan": np.full(1080, 10.)}

    def test_lap_seam_preserves_positive_progress(self):
        route = ForwardRoute(self.track)
        for i, theta in enumerate(np.linspace(2 * np.pi - .1, 2 * np.pi + .1, 40)):
            route.observe(self.observation(theta), i * .01)
        self.assertAlmostEqual(route.progress, .6, delta=.01)
        self.assertEqual(route.wrong_way_seconds, 0)

    def test_half_circle_is_forward_not_a_global_heading_reversal(self):
        route = ForwardRoute(self.track)
        for i, theta in enumerate(np.linspace(0, np.pi, 400)):
            route.observe(self.observation(theta), i * .01)
        self.assertGreater(route.progress, 9.4)
        self.assertEqual(route.wrong_way_seconds, 0)

    def test_reverse_proposal_is_rejected_and_steering_slew_is_bounded(self):
        controller = SimpleNamespace(plan=lambda scan, speed: (-.4189, -1.))
        driver = ReactiveDriver(controller, self.angles, 2.)
        previous = 0.
        for i, theta in enumerate(np.linspace(0, .1, 60)):
            steering, speed = driver.plan(self.observation(theta)["scan"], 1.)
            self.assertGreaterEqual(speed, 0)
            self.assertLessEqual(speed, 2.)
            self.assertLessEqual(abs(steering - previous), .01600001)
            previous = steering

    def test_no_route_correction_changes_local_steering(self):
        for raw_steer in (-.25, .25):
            controller = SimpleNamespace(plan=lambda scan, speed: (raw_steer, 1.))
            driver = ReactiveDriver(controller, self.angles, 2.)
            for _ in range(40):
                steering, _ = driver.plan(np.full(1080, 10.), 1.)
            self.assertAlmostEqual(steering, raw_steer)

    def test_controller_speed_is_respected(self):
        controller = SimpleNamespace(plan=lambda scan, speed: (0., .45))
        driver = ReactiveDriver(controller, self.angles, 2.)
        for _ in range(100):
            _, speed = driver.plan(np.full(1080, 10.), 1.)
        self.assertAlmostEqual(speed, .45)

    def test_completely_blocked_scan_stops(self):
        controller = SimpleNamespace(plan=lambda scan, speed: (0., 2.))
        driver = ReactiveDriver(controller, self.angles, 2.)
        observation = self.observation(0)
        observation["scan"][:] = 0
        self.assertEqual(driver.plan(observation["scan"], 0)[1], 0)

    def test_ordered_checkpoint_monitor_ignores_skipped_gate(self):
        monitor = CheckpointMonitor([[0, 0], [1, 0], [2, 0]], [[1, 0]] * 3, np.array([0., 0.]))
        monitor.observe({"pose_x": 1.1, "pose_y": 3.})
        self.assertEqual(monitor.crossings, 0)
        monitor.observe({"pose_x": .8, "pose_y": 0.})
        monitor.observe({"pose_x": 1.1, "pose_y": 0.})
        self.assertEqual(monitor.crossings, 1)


if __name__ == "__main__":
    unittest.main()
