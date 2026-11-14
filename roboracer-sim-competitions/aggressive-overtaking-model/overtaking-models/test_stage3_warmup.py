"""A close corner encounter must not collide before either race policy starts."""
import unittest
import numpy as np
from run_experiment import Scenario, episode, stage3_setup_speed
from track_geometry import TrackGeometry
from world import make_env


class Stage3WarmupTests(unittest.TestCase):
    def test_shared_limit_includes_both_cars_and_their_preview(self):
        class Geometry:
            def speed_at(self,s,vmax,*limits):
                # A tight corner begins at position3; both cars must respect it.
                return np.where((s>=3.) & (s<=4.),min(1.3,vmax),vmax)
        g=Geometry()
        self.assertEqual(stage3_setup_speed(g,[0.,3.],2.4),1.3)
        self.assertEqual(stage3_setup_speed(g,[3.,0.],2.4),1.3)
        self.assertEqual(stage3_setup_speed(g,[0.,1.8],2.4),1.3)
        self.assertEqual(stage3_setup_speed(g,[0.,1.8],2.4,preview_distance=.2),2.4)
        self.assertEqual(stage3_setup_speed(g,[0.,3.],.8),.8)

    def test_physical_close_corner_setups_preserve_gap_and_own_sensor_history(self):
        env=make_env()
        try:
            g=TrackGeometry(env.unwrapped.track)
            # These archived training/validation setups contacted at.35–.44s
            # when initialized at2.4m/s, before a meaningful race response.
            scenes=[(335715471,38.14166589401987,.8286213928897382),
                    (169528006,37.23299122367434,.7336618281792833),
                    (257834833,13.09511405302019,.7303447276394524)]
            for seed,s,gap in scenes:
                with self.subTest(seed=seed):
                    scenario=Scenario(stage=3,seed=seed,s=s,gap=gap,
                        ego_speed=2.4,opponent_speed=2.4,seconds=.5,
                        opponent='reactive',warmup_seconds=.4)
                    r,frames,*_=episode(scenario,teacher=True,control_dt=.05,
                        policy_bias=[0.,0.,0.,-4.],env=env,geometry=g)
                    self.assertIsNone(r['collision'])
                    self.assertGreater(r['warmup_minimum_body_gap_m'],.09)
                    self.assertEqual(r['warmup_seconds'],.4)
                    self.assertEqual(r['warmup_lidar_samples_per_car'],8)
                    self.assertTrue(r['opponent_uses_own_observation_only'])
                    self.assertTrue(r['coordinated_stage3_warmup'])
                    self.assertEqual(r['warmup_initial_speed_m_s'][0],r['warmup_initial_speed_m_s'][1])
                    self.assertLess(r['warmup_initial_speed_m_s'][0],2.4)
                    self.assertEqual(len(r['warmup_command_history']),8)
                    for step in r['warmup_command_history']:
                        self.assertLess(step['t'],.4)
                        self.assertEqual(step['commands'][0][1],step['commands'][1][1])
                        self.assertLessEqual(step['commands'][0][1],step['speed_cap'])
                    self.assertTrue(any(abs(step['commands'][0][0]-step['commands'][1][0])>1e-4
                        for step in r['warmup_command_history']))
                    self.assertAlmostEqual(frames[0]['t'],.41)
        finally:
            env.close()


if __name__=='__main__':
    unittest.main()
