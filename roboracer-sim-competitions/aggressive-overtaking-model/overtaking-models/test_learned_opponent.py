import json
import unittest
from pathlib import Path
import numpy as np
from learned_opponent import OpponentForecast
from learned_stage4 import LearnedPlanner
from expanded_planner import ExpandedPlanner
from perception import Target
from world import make_env
from track_geometry import TrackGeometry

class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=make_env(seed=184);cls.g=TrackGeometry(cls.env.unwrapped.track)
    @classmethod
    def tearDownClass(cls):cls.env.close()
    def obs(self,s):
        xy,yaw,_=self.g.sample(s)
        return dict(pose_x=xy[0],pose_y=xy[1],pose_theta=yaw,linear_vel_x=2.,scan=np.full(1080,20.))
    def test_forecast_starts_at_measured_pose_and_is_finite(self):
        f=OpponentForecast();g=self.g;times=np.arange(21)*.05
        for s in (1.,17.,35.,58.8):
            target=Target(s+.8,.05,speed=1.8,confidence=.99,observations=100)
            path=g.sample(s+times*2.)[0]
            xy,yaw,spread,d=f.trajectory(g,target,self.obs(s),path,times,times)
            np.testing.assert_allclose(xy[0],g.sample(target.s,target.d)[0],atol=1e-6)
            self.assertTrue(np.isfinite(xy).all() and np.isfinite(yaw).all())
            self.assertTrue((spread>=0).all());self.assertEqual(xy.shape,(21,2))
    def test_clear_track_preserves_original_plans(self):
        a=ExpandedPlanner(self.g,stage=3);b=LearnedPlanner(self.g,stage=3)
        for s in (2.,18.,32.,48.):
            for x,y in zip(a.candidates(self.obs(s),None),b.candidates(self.obs(s),None)):
                np.testing.assert_array_equal(x.path,y.path)
                self.assertEqual((x.speed,x.steering,x.feasible),(y.speed,y.steering,y.feasible))
    def test_branch_splits_are_disjoint_and_reactive(self):
        cases=json.loads(Path('artifacts/reactive_branches_v2/manifest.json').read_text())['cases']
        train={c['seed'] for c in cases if c['split']=='train'}
        validation={c['seed'] for c in cases if c['split']=='validation'}
        self.assertTrue(train and validation);self.assertFalse(train&validation)
        self.assertTrue(all(c['opponent']=='reactive' for c in cases))
    def test_collector_refuses_existing_record(self):
        import collect_reactive_branches as collect
        with self.assertRaises(FileExistsError):collect.trial({'id':0})
    def test_experiment_rejects_nonreactive_opponent(self):
        from reactive_pressure_experiment import run
        with self.assertRaises(AssertionError):run({'scenario':{'opponent':'line'},'mode':'reactive'},'/tmp')

if __name__=='__main__':unittest.main()
