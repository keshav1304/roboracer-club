import unittest
import numpy as np
from world import make_env
from track_geometry import TrackGeometry,MAX_STEER,WHEELBASE
from perception import Target
from maneuver_core import CandidatePlanner
from expanded_planner import ExpandedPlanner

class ExpandedPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=make_env(seed=12801);cls.g=TrackGeometry(cls.env.unwrapped.track)
    @classmethod
    def tearDownClass(cls):cls.env.close()
    def obs(self,s,d=0.):
        xy,yaw,_=self.g.sample(s,d)
        return dict(pose_x=float(xy[0]),pose_y=float(xy[1]),pose_theta=float(yaw),linear_vel_x=2.1,scan=np.full(1080,20.))
    def test_clear_track_is_unchanged(self):
        for stage in (2,3):
            a=CandidatePlanner(self.g,stage=stage);b=ExpandedPlanner(self.g,stage=stage)
            for s in (4.,19.,34.,49.):
                for x,y in zip(a.candidates(self.obs(s),None),b.candidates(self.obs(s),None)):
                    np.testing.assert_array_equal(x.path,y.path)
                    self.assertEqual((x.speed,x.steering,x.feasible),(y.speed,y.steering,y.feasible))
    def test_extra_paths_keep_clearance_and_add_options(self):
        added=0;rescued=0
        for stage in (2,3):
            a=CandidatePlanner(self.g,stage=stage);b=ExpandedPlanner(self.g,stage=stage)
            a.control_dt=b.control_dt=.05
            for s in np.linspace(0,self.g.length,24,endpoint=False):
                obs=self.obs(s)
                target=Target((s+2.)%self.g.length,0.,speed=1.7,confidence=.98,observations=100)
                old=a.candidates(obs,target);new=b.candidates(obs,target)
                self.assertEqual([x.action for x in old],[x.action for x in new])
                added+=b.search_stats['additional_feasible']
                for before,after in zip(old[:2],new[:2]):
                    limit=.95*np.tan(MAX_STEER)/WHEELBASE
                    if before.feasible and (stage==2 or b.path_curvature(before.path).max()<=limit):
                        self.assertTrue(after.feasible)
                    if after.feasible and not before.feasible:rescued+=1
                    if hasattr(after,'path_parameters'):
                        self.assertTrue(np.isfinite(after.path).all())
                        self.assertGreaterEqual(after.clearance,min(b.wall_margin,b.car_margin))
                        self.assertLessEqual(abs(after.steering),.4189)
                        self.assertLessEqual(after.speed,b.vmax)
                        self.assertLessEqual(b.path_curvature(after.path).max(),limit)
        self.assertGreater(added,0)
        self.assertGreater(rescued,0)
    def test_full_width_blocker_does_not_create_false_gap(self):
        planner=ExpandedPlanner(self.g,stage=2);planner.control_dt=.05
        target=Target(5.,0.,speed=.5,width=5.,confidence=.99,observations=100)
        c=planner.candidates(self.obs(4.),target)
        self.assertFalse(c[0].feasible);self.assertFalse(c[1].feasible)
    def test_fresh_track_does_not_bypass_motion_history(self):
        planner=ExpandedPlanner(self.g,stage=2);planner.control_dt=.05
        planner.candidates(self.obs(4.),Target(6.,0.,speed=1.7,observations=1))
        self.assertEqual(planner.search_stats['evaluated'],4)
    def test_inside_arc_curvature_uses_its_own_length(self):
        angle=np.linspace(0.,.7,61)
        xy=.5*np.column_stack([np.cos(angle),np.sin(angle)])
        # A half-metre radius requires 2 inverse-metres of curvature, regardless
        # of how far apart its projections are along a different reference.
        self.assertAlmostEqual(float(np.median(ExpandedPlanner.path_curvature(xy)[2:-2])),2.,places=3)

if __name__=='__main__':unittest.main()
