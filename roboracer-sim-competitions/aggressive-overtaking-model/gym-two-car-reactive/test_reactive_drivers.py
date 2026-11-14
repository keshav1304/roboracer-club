"""Check steering conventions, stopping, and malformed scan handling."""
import unittest
import numpy as np
from gap_follower import GapFollower
from wall_follower import WallFollower, WallParams


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.angles = np.linspace(-3 * np.pi / 4, 3 * np.pi / 4, 1080)

    def left_wall(self, distance):
        scan = np.full(len(self.angles), 30.)
        left = self.angles > .01
        scan[left] = np.minimum(30, distance / np.sin(self.angles[left]))
        return scan

    def test_parallel_wall_keeps_heading(self):
        steer, speed = WallFollower(self.angles).plan(self.left_wall(.7))
        self.assertAlmostEqual(steer, 0, delta=.01)
        self.assertAlmostEqual(speed, 2.)

    def test_wall_error_turns_towards_far_wall_and_away_from_close_wall(self):
        self.assertGreater(WallFollower(self.angles).plan(self.left_wall(1.1))[0], 0)
        self.assertLess(WallFollower(self.angles).plan(self.left_wall(.4))[0], 0)

    def test_wall_driver_stops_for_front_obstacle(self):
        scan = self.left_wall(.7)
        scan[np.abs(self.angles) < .3] = .3
        self.assertEqual(WallFollower(self.angles).plan(scan)[1], 0)

    def test_gap_driver_stops_when_scan_is_blocked(self):
        self.assertEqual(GapFollower(self.angles).plan(np.zeros(1080)), (0., 0.))

    def test_gap_driver_never_requests_reverse(self):
        controller = GapFollower(self.angles)
        scan = self.left_wall(.5)
        scan[np.abs(self.angles) < .4] = .3
        for _ in range(100):
            self.assertGreaterEqual(controller.plan(scan, 0.)[1], 0.)

    def corner_scan(self, mirror=1, front_shift=0.):
        segments = np.array([
            [[-5., -.8], [2., -.8]], [[2., -.8], [2., 6.]],
            [[.6, .8], [.6, 6.]], [[-5., .8], [.6, .8]],
        ])
        segments[:, :, 0] += front_shift
        segments[:, :, 1] *= mirror
        directions = np.column_stack((np.cos(self.angles), np.sin(self.angles)))
        scan = np.full(len(self.angles), 30.)
        cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        for start, end in segments:
            edge = end - start
            denominator = cross(directions, edge)
            valid = abs(denominator) > 1e-8
            distance = np.divide(cross(start, edge), denominator, out=np.full(len(scan), 30.), where=valid)
            fraction = np.divide(cross(start, directions), denominator, out=np.full(len(scan), -1.), where=valid)
            hit = valid & (distance > 0) & (fraction >= 0) & (fraction <= 1)
            scan[hit] = np.minimum(scan[hit], distance[hit])
        return scan

    def test_wall_preview_turns_both_ways_before_the_front_wall(self):
        for sign in (1, -1):
            steering, speed = WallFollower(self.angles).plan(self.corner_scan(sign, front_shift=-.6))
            self.assertGreater(sign * steering, .05)
            self.assertGreater(speed, 0.)
            self.assertLessEqual(speed, 2.)

    def test_wall_standoff_remains_a_control_input(self):
        scan = self.left_wall(.7)
        near = WallFollower(self.angles, WallParams(desired_dist=.5))
        far = WallFollower(self.angles, WallParams(desired_dist=.9))
        self.assertGreater(near.plan(scan)[0], 0.)
        self.assertLess(far.plan(scan)[0], 0.)

    def test_wall_driver_does_not_call_the_gap_algorithm(self):
        from unittest.mock import patch
        with patch("reactive.FollowTheGap.plan", side_effect=AssertionError("gap control called")):
            steering, speed = WallFollower(self.angles).plan(self.left_wall(.7))
        self.assertAlmostEqual(steering, 0., delta=.01)
        self.assertGreater(speed, 0.)

    def test_wall_driver_stops_between_preview_updates(self):
        controller = WallFollower(self.angles)
        scan = self.left_wall(.7)
        controller.plan(scan)
        scan[np.abs(self.angles) < .6] = .2
        self.assertEqual(controller.plan(scan)[1], 0.)
        self.assertEqual(controller.plan(np.zeros(len(scan))), (0., 0.))

    def test_wall_car_can_continue_with_a_close_parallel_surface(self):
        scan = np.minimum(self.left_wall(.7), self.left_wall(.25)[::-1])
        steering, speed = WallFollower(self.angles).plan(scan)
        self.assertAlmostEqual(steering, 0., delta=.01)
        self.assertGreater(speed, 0.)

    def test_gap_car_slows_immediately_when_side_clearance_closes(self):
        from reactive_safety import ReactiveDriver
        controller = GapFollower(self.angles)
        driver = ReactiveDriver(controller, self.angles, 3.6, steering_rate=3.,
                                acceleration=2., lateral_acceleration=2.5,
                                stop_distance=.45, corridor_margin=.04)
        scan = np.minimum(self.left_wall(.9), self.left_wall(.9)[::-1])
        for _ in range(200):
            driver.plan(scan, 2.)
        close_scan = np.minimum(self.left_wall(.9), self.left_wall(.45)[::-1])
        self.assertLess(driver.plan(close_scan, 2.)[1], 1.1)

    def test_gap_preview_turns_before_the_front_wall(self):
        for sign in (1, -1):
            steering, speed = GapFollower(self.angles).plan(self.corner_scan(sign))
            self.assertGreater(sign * steering, .05)
            self.assertGreater(speed, 0.)

    def test_gap_car_is_faster_on_an_open_straight_and_slows_for_a_bend(self):
        open_scan = np.minimum(self.left_wall(.9), self.left_wall(.9)[::-1])
        straight_steer, straight_speed = GapFollower(self.angles).plan(open_scan)
        _, corner_speed = GapFollower(self.angles).plan(self.corner_scan())
        self.assertAlmostEqual(straight_steer, 0., delta=.01)
        self.assertGreater(straight_speed, 2.4)
        self.assertLess(corner_speed, straight_speed)

    def test_new_obstacle_stops_the_car_between_preview_updates(self):
        from reactive_safety import ReactiveDriver
        driver = ReactiveDriver(GapFollower(self.angles), self.angles, 3.6,
                                steering_rate=3., acceleration=2., lateral_acceleration=2.5,
                                stop_distance=.45)
        scan = np.full(len(self.angles), 10.)
        for _ in range(21):
            driver.plan(scan, .5)
        scan[np.abs(self.angles) < .6] = .2
        self.assertEqual(driver.plan(scan, .5)[1], 0.)

    def test_drivers_return_finite_bounded_commands(self):
        scan = self.left_wall(.9)
        scan[100:110] = np.nan
        scan[300:310] = np.inf
        for driver, cap in [(WallFollower(self.angles), 2), (GapFollower(self.angles, v_max=2.4), 2.4)]:
            steer, speed = driver.plan(scan)
            self.assertTrue(np.isfinite([steer, speed]).all())
            self.assertLessEqual(abs(steer), .4189)
            self.assertGreaterEqual(speed, 0)
            self.assertLessEqual(speed, cap)


if __name__ == "__main__":
    unittest.main()
