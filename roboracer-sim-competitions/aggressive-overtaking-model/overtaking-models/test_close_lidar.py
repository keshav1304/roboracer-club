"""Real LiDAR regression for a nearby bumper mistaken for a vehicle centre."""
import unittest
import numpy as np
from perception import LidarTracker
from track_geometry import TrackGeometry, PathTracker
from world import make_env, reset_scenario


class CloseLidarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=make_env(seed=5521)
        cls.g=TrackGeometry(cls.env.unwrapped.track)
        cls.angles=cls.env.unwrapped.sim.agents[0].scan_angles
    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_near_rear_face_initializes_physical_center(self):
        for gap in (.7,.85,1.0):
            with self.subTest(gap=gap):
                obs,_=reset_scenario(self.env,self.g,2.,gap,0.)
                tracker=LidarTracker(self.g,self.angles)
                target=tracker.update(obs['agent_0'],dt=.05)
                ego_s,_=self.g.project(obs['agent_0']['pose_x'],obs['agent_0']['pose_y'])
                self.assertIsNotNone(target)
                self.assertAlmostEqual(float(self.g.delta_s(ego_s,target.s)),gap,delta=.04)

    def test_actual_close_moving_scans_establish_motion_without_freezing(self):
        obs,_=reset_scenario(self.env,self.g,2.,.7,2.4,ego_speed=2.4)
        tracker=LidarTracker(self.g,self.angles)
        drivers=[PathTracker(self.g),PathTracker(self.g)]
        for tick in range(25):
            if tick%5==0:
                target=tracker.update(obs['agent_0'],dt=.05)
                controls=[drivers[i].control(obs[f'agent_{i}'],vmax=2.4,dt=.05) for i in (0,1)]
            obs,*_=self.env.step(np.array(controls))
        self.assertTrue(tracker.velocity_initialized)
        self.assertAlmostEqual(target.speed,2.4,delta=.15)
        ego_s,_=self.g.project(obs['agent_0']['pose_x'],obs['agent_0']['pose_y'])
        self.assertGreater(float(self.g.delta_s(ego_s,target.s)),.5)

    def test_close_stationary_target_does_not_inherit_ego_speed(self):
        obs,_=reset_scenario(self.env,self.g,2.,.7,0.)
        tracker=LidarTracker(self.g,self.angles)
        for tick in range(25):
            if tick%5==0:
                target=tracker.update(obs['agent_0'],dt=.05)
            obs,*_=self.env.step(np.zeros((2,2)))
        self.assertTrue(tracker.velocity_initialized)
        self.assertLess(target.speed,.08)


if __name__=='__main__':
    unittest.main()
