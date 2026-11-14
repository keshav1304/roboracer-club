"""Decision cadence does not change physical contact sampling."""
import unittest
from unittest.mock import patch
import numpy as np
from run_experiment import episode, Scenario
from world import make_env
from track_geometry import TrackGeometry

class EpisodeCadenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=make_env(seed=448)
        cls.geometry=TrackGeometry(cls.env.unwrapped.track)
    @classmethod
    def tearDownClass(cls):
        cls.env.close()
    def test_twenty_hz_control_keeps_hundred_hz_physics(self):
        with patch.object(self.env,'step',wraps=self.env.step) as step:
            result,_,_,_=episode(Scenario(stage=2,s=0.,gap=3.,opponent_speed=2.,seconds=.2),
                                 env=self.env,geometry=self.geometry,control_dt=.05)
        self.assertEqual(step.call_count,21)  # one setup scan refresh, then 20 physics ticks
        self.assertEqual(result['network_latency']['count'],4)
        self.assertEqual(result['physics_dt_seconds'],.01)
        self.assertEqual(result['control_dt_seconds'],.05)
        commands=[c.args[0] for c in step.call_args_list][1:]
        for i in range(0,20,5):
            for j in range(1,5):np.testing.assert_array_equal(commands[i],commands[i+j])
    def test_invalid_cadence_and_test_exploration_are_rejected(self):
        with self.assertRaises(ValueError):episode(Scenario(),control_dt=.035)
        with self.assertRaisesRegex(ValueError,'training-teacher'):
            episode(Scenario(),policy_bias=[0,0,0])
        with self.assertRaisesRegex(ValueError,'warmup'):
            episode(Scenario(stage=1,warmup_seconds=.2))
    def test_flying_start_uses_real_scans_and_counts_setup_time(self):
        from interactive_model import InteractiveModel
        from run_experiment import MODELS
        measurements=[]
        class Capture(InteractiveModel):
            def plan(self,observation,dt=.01):
                result=super().plan(observation,dt=dt)
                measurements.append(dict(self.telemetry['target']))
                return result
        with patch.dict(MODELS,{3:Capture}),patch.object(self.env,'step',wraps=self.env.step) as step:
            result,frames,_,_=episode(Scenario(stage=3,s=2.,gap=.7,ego_speed=2.4,opponent_speed=2.4,
                opponent='reactive',seconds=.3,warmup_seconds=.2),teacher=True,
                env=self.env,geometry=self.geometry,control_dt=.05)
        self.assertEqual(step.call_count,31)
        self.assertEqual(result['warmup_lidar_samples_per_car'],4)
        self.assertAlmostEqual(frames[0]['t'],.21)
        self.assertAlmostEqual(result['sim_seconds'],.3)
        self.assertAlmostEqual(measurements[0]['distance'],.7,delta=.05)
        self.assertAlmostEqual(measurements[0]['speed'],2.4,delta=.15)
        self.assertIsNone(result['collision'])

if __name__=='__main__':unittest.main()
