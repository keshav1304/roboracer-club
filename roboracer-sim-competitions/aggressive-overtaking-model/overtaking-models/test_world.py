"""Setup checks which protect simulator fidelity and timing claims."""
import unittest

import numpy as np

from world import make_env, install_barrier, restore_base_map, reset_scenario, body_polygon, timing_summary, make_stage2_opponent
from track_geometry import TrackGeometry


class WorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_env(seed=314)
        cls.geometry = TrackGeometry(cls.env.unwrapped.track, optimize=False)
        cls.original = cls.env.unwrapped.track.occupancy_map.copy()

    @classmethod
    def tearDownClass(cls):
        restore_base_map(cls.env)
        cls.env.close()

    def tearDown(self):
        restore_base_map(self.env)

    def test_custom_barrier_changes_only_simulator_copy(self):
        track = self.env.unwrapped.track
        polygon = install_barrier(self.env, self.geometry, 4., 1.3)
        self.assertAlmostEqual(polygon.area, .58 * 1.3, places=5)
        self.assertIsNot(self.env.unwrapped.track, track)
        np.testing.assert_array_equal(track.occupancy_map, self.original)
        np.testing.assert_array_equal(self.geometry.occupancy, self.original)
        self.assertGreater(np.count_nonzero(self.env.unwrapped.track.occupancy_map != self.original), 30)
        restore_base_map(self.env)
        self.assertIs(self.env.unwrapped.track, track)
        np.testing.assert_array_equal(self.env.unwrapped.sim.agents[0].scan_simulator.map_img, self.original)

    def test_reset_has_real_initial_speed_and_independent_contact_geometry(self):
        obs, meta = reset_scenario(self.env, self.geometry, 0., 3., 1.3, ego_speed=1.3)
        self.assertAlmostEqual(float(obs['agent_0']['linear_vel_x']), 1.3, places=4)
        self.assertAlmostEqual(float(obs['agent_1']['linear_vel_x']), 1.3, places=4)
        self.assertGreater(body_polygon(obs['agent_0']).distance(body_polygon(obs['agent_1'])), .5)
        self.assertIsNone(meta['obstacle_polygon'])

    def test_full_blockage_is_visible_and_base_map_remains_clean(self):
        normal, _ = reset_scenario(self.env, self.geometry, 0., 3., 0.)
        blocked, meta = reset_scenario(self.env, self.geometry, 0., 3., 0., blocked=True)
        self.assertIsNotNone(meta['obstacle_polygon'])
        np.testing.assert_array_equal(self.geometry.occupancy, self.original)
        self.assertGreater(np.count_nonzero(np.asarray(blocked['agent_0']['scan']) < np.asarray(normal['agent_0']['scan']) - .1), 10)
        self.assertGreater(body_polygon(blocked['agent_0']).distance(meta['obstacle_polygon']), .1)

    def test_invalid_spawn_rejected(self):
        with self.assertRaises(ValueError):
            reset_scenario(self.env, self.geometry, 0., .2, 0.)
        with self.assertRaises(ValueError):
            reset_scenario(self.env, self.geometry, 0., 3., 1., blocked=True)

    def test_stage3_opponent_uses_saved_stage2_and_its_own_running_state(self):
        from moving_model import MovingModel
        # A deployed outcome head is tied to its optimized reference, unlike
        # the deliberately unoptimized geometry used by setup-only fixtures.
        geometry = TrackGeometry(self.env.unwrapped.track)
        obs, _ = reset_scenario(self.env, geometry, 0., 3., 1.3, ego_speed=1.3)
        rival_obs = obs["agent_1"]
        angles = self.env.unwrapped.sim.agents[1].scan_angles
        rival = make_stage2_opponent(geometry, angles, 2.4, rival_obs, steering=.05)
        self.assertIsInstance(rival, MovingModel)
        self.assertEqual(rival.stage, 2)
        self.assertFalse(rival.teacher)
        self.assertEqual(rival.model_path.name, "stage2.npz")
        self.assertEqual(len(rival.model_sha256), 64)
        self.assertAlmostEqual(rival.command_speed, float(rival_obs["linear_vel_x"]))
        self.assertAlmostEqual(rival.steering, .05)
        self.assertIsNone(rival.sensor.target)
        self.assertEqual(rival.planner.car_margin, .12)
        # There is no leader in the rival's driving corridor. It must not
        # invent a yield or request a stop simply because an ego exists.
        _, speed = rival.plan(rival_obs)
        self.assertGreater(speed, 1.)
        self.assertEqual(rival.action, "WAIT")

    def test_timing_is_in_milliseconds_and_empty_is_not_fake_zero(self):
        summary = timing_summary([1., 2., 3.])
        self.assertEqual(summary['count'], 3)
        self.assertEqual(summary['p50_ms'], 2.)
        self.assertEqual(summary['max_ms'], 3.)
        self.assertIsNone(timing_summary([])['p50_ms'])
        with self.assertRaises(ValueError):
            timing_summary([float('nan')])


if __name__ == '__main__':
    unittest.main()
