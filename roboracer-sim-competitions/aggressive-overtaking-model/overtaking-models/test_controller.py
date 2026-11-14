"""Meaningful controller contracts: isolation, feasible actions and safe holds."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from bootstrap import MODELS
from world import make_env, reset_scenario
from track_geometry import TrackGeometry
from perception import Target
from maneuver_core import ACTIONS, FEATURE_NAMES, Candidate, CandidatePlanner, NeuralDriver
from benchmark import offline_only


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_env(seed=205)
        cls.geometry = TrackGeometry(cls.env.unwrapped.track)
        cls.angles = cls.env.unwrapped.sim.agents[0].scan_angles.copy()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        obs, _ = reset_scenario(self.env, self.geometry, 0., 3., 0.)
        self.obs = obs['agent_0']

    def test_runtime_imports_no_teacher_client(self):
        script = """import socket
def forbidden(*args, **kwargs):
    raise AssertionError('runtime attempted network')
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden
import static_model, moving_model, interactive_model, tiny_network, sys
assert 'llm_teacher' not in sys.modules
assert 'urllib.request' not in sys.modules
"""
        result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).parent,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_candidate_evaluation_does_not_advance_actuator_state(self):
        planner = CandidatePlanner(self.geometry)
        planner.tracker.last_steer = .12
        target = Target(3., 0., confidence=.9)
        normal = planner.candidates(self.obs, target)
        self.assertEqual(planner.tracker.last_steer, .12)
        with patch('maneuver_core.ACTIONS', tuple(reversed(ACTIONS))):
            reversed_order = planner.candidates(self.obs, target)
        for candidate in normal:
            other = next(c for c in reversed_order if c.action == candidate.action)
            np.testing.assert_allclose([candidate.steering, candidate.speed, candidate.cost],
                                       [other.steering, other.speed, other.cost], rtol=1e-7, atol=1e-7)

    def test_wait_stops_for_a_close_full_width_blocker(self):
        planner = CandidatePlanner(self.geometry)
        s, _ = self.geometry.project(self.obs['pose_x'], self.obs['pose_y'])
        target = Target((s + 1.) % self.geometry.length, 0., width=3., confidence=.9)
        candidates = planner.candidates(self.obs, target)
        self.assertFalse(candidates[0].feasible)
        self.assertFalse(candidates[1].feasible)
        self.assertTrue(candidates[2].feasible)
        self.assertEqual(candidates[2].speed, 0.)

    def test_rear_laterally_clear_car_does_not_brake_the_reference_path(self):
        for stage in (2, 3):
            with self.subTest(stage=stage):
                planner = CandidatePlanner(self.geometry, stage=stage)
                s, _ = self.geometry.project(self.obs['pose_x'], self.obs['pose_y'])
                observation = dict(self.obs, linear_vel_x=2.4)
                target = Target(s - .684, -.713, speed=2.4,
                                confidence=.98, observations=100)
                unoccupied = planner.candidates(observation, None)[2]
                adjacent = planner.candidates(observation, target)[2]
                self.assertGreater(unoccupied.speed, 1.)
                self.assertAlmostEqual(adjacent.speed, unoccupied.speed, places=6)
                self.assertAlmostEqual(adjacent.steering, unoccupied.steering, places=6)

    def test_front_blocker_stops_but_clear_parallel_corridor_can_roll(self):
        planner = CandidatePlanner(self.geometry, stage=2)
        s, _ = self.geometry.project(self.obs['pose_x'], self.obs['pose_y'])
        observation = dict(self.obs, linear_vel_x=2.4)
        ahead = Target(s + .8, 0., speed=0., confidence=.98, observations=100)
        self.assertEqual(planner.candidates(observation, ahead)[2].speed, 0.)
        # An ego already in a detour must not merge across a neighboring car.
        xy, yaw, _ = self.geometry.sample(s, -.51)
        detour = dict(observation, pose_x=float(xy[0]), pose_y=float(xy[1]),
                      pose_theta=float(yaw))
        neighbor = Target(s - .1, 0., speed=2.4, confidence=.98, observations=100)
        hold = planner.candidates(detour, neighbor)[2]
        self.assertGreater(hold.speed, 0.)
        parallel_steer, _ = planner.tracker.control(detour, offset=-.51, vmax=2.4, smooth=False)
        self.assertAlmostEqual(hold.steering, parallel_steer, delta=.03)

    def test_safety_filter_rejects_network_preference_for_infeasible_pass(self):
        driver = NeuralDriver(self.geometry, self.angles, stage=1, teacher=True)
        driver.teacher = False
        class InfeasiblePreference:
            def proba(self, x):
                return np.asarray([.999, .0005, .0005])
        driver.model = InfeasiblePreference()
        candidates = [Candidate(action, action == 'WAIT', 1., .2, 0., 0., 0., np.zeros((2, 2)))
                      for action in ACTIONS]
        with patch.object(driver.sensor, 'update', return_value=None), \
             patch.object(driver.planner, 'candidates', return_value=candidates), \
             patch.object(driver.planner, 'features', return_value=np.zeros(len(FEATURE_NAMES))), offline_only():
            command = driver.plan(self.obs)
        self.assertEqual(driver.action, 'WAIT')
        np.testing.assert_allclose(driver.telemetry['safe_probabilities'], [0., 0., 1.])
        self.assertEqual(command[1], 0.)

    def test_trained_probability_shapes_and_offline_control(self):
        available = [stage for stage in (1, 2, 3) if (MODELS / f'stage{stage}.npz').exists()]
        if not available:
            self.skipTest('Train checkpoints before the learned runtime test.')
        for stage in available:
            with self.subTest(stage=stage), offline_only():
                driver = NeuralDriver(self.geometry, self.angles, stage=stage)
                command = driver.plan(self.obs)
                probabilities = np.asarray(driver.telemetry['probabilities'])
                self.assertEqual(probabilities.shape, (4 if stage == 3 else 3,))
                self.assertAlmostEqual(float(probabilities.sum()), 1., places=5)
                self.assertTrue(np.all(probabilities >= 0))
                self.assertTrue(np.all(np.isfinite(command)))
                self.assertGreaterEqual(command[1], 0.)

    def test_moving_target_braking_uses_closing_speed_but_stale_tracks_do_not(self):
        s, _ = self.geometry.project(self.obs['pose_x'], self.obs['pose_y'])
        target = Target((s + 1.) % self.geometry.length, 0., speed=2., confidence=.9,
                        observations=30)
        moving = self._isolated_braking_command(target)
        stopped = self._isolated_braking_command(replace(target, speed=0.))
        stale = self._isolated_braking_command(replace(target, time_since_seen=.2))
        self.assertGreater(moving, stopped + .4)
        self.assertAlmostEqual(stale, stopped, places=6)

    def test_a_wall_hit_cannot_inherit_a_nearby_targets_velocity(self):
        s, _ = self.geometry.project(self.obs['pose_x'], self.obs['pose_y'])
        target = Target((s + 1.) % self.geometry.length, 0., speed=2., confidence=.9,
                        observations=30)
        stationary_cap = self._isolated_braking_command(replace(target, speed=0.))
        with patch.object(self.geometry, 'clearance', side_effect=lambda hits: np.zeros(np.shape(hits)[:-1])):
            wall_cap = self._isolated_braking_command(target)
        self.assertAlmostEqual(wall_cap, stationary_cap, places=6)

    def test_a_new_motion_track_cannot_trigger_a_pass_before_velocity_estimation(self):
        planner = CandidatePlanner(self.geometry, stage=2)
        target = Target(3., 0., speed=0., confidence=.2, observations=1)
        candidates = planner.candidates(self.obs, target)
        self.assertFalse(candidates[0].feasible)
        self.assertFalse(candidates[1].feasible)
        self.assertFalse(candidates[3].feasible)
        self.assertTrue(candidates[2].feasible)

    def test_unknown_motion_uses_stopping_distance_before_a_follow_speed(self):
        planner = CandidatePlanner(self.geometry, stage=2)
        s, _ = self.geometry.project(self.obs['pose_x'], self.obs['pose_y'])
        target = Target(s + 1.6, 0., speed=0., observations=1)
        fresh = planner.candidates(self.obs, target)[2]
        stopped = planner.candidates(self.obs, replace(target, observations=30))[2]
        self.assertGreater(fresh.speed, stopped.speed)
        stopping_cap = np.sqrt(6. * (1.6 - .29 - .29 - .20))
        self.assertLessEqual(fresh.speed, stopping_cap + 1e-5)

    def test_moving_pass_plans_rejoining_after_lateral_clearance(self):
        planner = CandidatePlanner(self.geometry, stage=2)
        xy, yaw, _ = self.geometry.sample(8.2, -.51)
        observation = dict(self.obs, pose_x=float(xy[0]), pose_y=float(xy[1]),
                           pose_theta=float(yaw), linear_vel_x=2.4)
        target = Target(8.0, 0., speed=2., confidence=.98, observations=100)
        candidate = planner.candidates(observation, target, ('RIGHT', -.51))[1]
        _, ending_offset = self.geometry.project(*candidate.path[-1])
        self.assertLess(abs(ending_offset), .20)
        # The preview must still preserve physical width at the current overlap.
        _, initial_offset = self.geometry.project(*candidate.path[0])
        self.assertAlmostEqual(initial_offset, -.51, places=2)

    def test_close_inside_probe_never_assumes_an_occupied_gap_will_open(self):
        planner = CandidatePlanner(self.geometry, stage=3)
        xy, yaw, _ = self.geometry.sample(6.5)
        observation = dict(self.obs, pose_x=float(xy[0]), pose_y=float(xy[1]),
                           pose_theta=float(yaw), linear_vel_x=2.4)
        target = Target(7.4, 0., speed=2.4, confidence=.98, observations=100)
        moving = planner.candidates(observation, target)
        stopped = planner.candidates(observation, replace(target, speed=0.))
        self.assertTrue(moving[3].feasible)
        self.assertFalse(stopped[3].feasible)
        self.assertTrue(stopped[2].feasible)
        self.assertLess(stopped[3].clearance, planner.car_margin)

    def test_equal_speed_opponent_does_not_start_with_a_zero_speed_command(self):
        from run_experiment import Scenario, episode
        from moving_model import MovingModel
        first_internal_speeds = []
        class CaptureInitialSpeed(MovingModel):
            def plan(self, observation, dt=.01):
                first_internal_speeds.append((self.command_speed, observation['linear_vel_x']))
                return super().plan(observation, dt=dt)
        with patch('moving_model.MovingModel', CaptureInitialSpeed), offline_only():
            episode(Scenario(stage=3, s=0., gap=2., ego_speed=2.4, opponent_speed=2.4,
                             seconds=.01, opponent='reactive'),
                    teacher=True, env=self.env, geometry=self.geometry)
        self.assertEqual(len(first_internal_speeds), 1)
        command_state, actual_speed = first_internal_speeds[0]
        self.assertAlmostEqual(actual_speed, 2.4, places=4)
        self.assertAlmostEqual(command_state, actual_speed, places=6)

    def _isolated_braking_command(self, target):
        """Test the safety constraint independently of the chosen maneuver."""
        driver = NeuralDriver(self.geometry, self.angles, stage=2, teacher=True)
        observation = dict(self.obs)
        observation['linear_vel_x'] = 2.4
        scan = np.full(len(self.angles), 20.)
        scan[np.argmin(np.abs(self.angles))] = .71
        observation['scan'] = scan
        candidates = [Candidate(action, action == 'WAIT', 0. if action == 'WAIT' else 30.,
                                .2, 0., 3., 0., np.zeros((2, 2))) for action in ACTIONS]
        with patch.object(driver.sensor, 'update', return_value=target), \
             patch.object(driver.planner, 'candidates', return_value=candidates), \
             patch.object(driver.planner, 'features', return_value=np.zeros(len(FEATURE_NAMES))), offline_only():
            return driver.plan(observation)[1]


if __name__ == '__main__':
    unittest.main()
