import unittest
import numpy as np
from world import make_env
from track_geometry import TrackGeometry,MAX_STEER,WHEELBASE
from expanded_planner import ExpandedPlanner
from response_stage4 import ResponsePlanner,ResponseStage4,separation
from perception import Target

class ResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=make_env(seed=181);cls.g=TrackGeometry(cls.env.unwrapped.track)
    @classmethod
    def tearDownClass(cls):cls.env.close()
    def obs(self,s,d=0.):
        xy,yaw,_=self.g.sample(s,d)
        return dict(pose_x=xy[0],pose_y=xy[1],pose_theta=yaw,linear_vel_x=2.4,scan=np.full(1080,20.))
    def test_body_overlap_is_not_a_center_distance_test(self):
        ego=np.array([[0.,0.]]);yaw=np.zeros(1)
        self.assertAlmostEqual(float(separation(ego,yaw,np.array([[.58,0.]]),yaw)[0]),0.)
        self.assertLess(separation(ego,yaw,np.array([[.4,0.]]),yaw)[0],0.)
    def test_clear_track_has_no_response_override(self):
        a=ExpandedPlanner(self.g,stage=3);b=ResponsePlanner(self.g,stage=3)
        for s in (4.,19.,34.,49.):
            for x,y in zip(a.candidates(self.obs(s),None),b.candidates(self.obs(s),None)):
                np.testing.assert_array_equal(x.path,y.path)
                self.assertEqual((x.speed,x.steering,x.feasible),(y.speed,y.steering,y.feasible))
            self.assertIsNone(b.response_choice)
    def test_challenge_requires_real_counterfactual_conflict(self):
        p=ResponsePlanner(self.g,stage=3);p.control_dt=.05
        s=self.g.length*8/60
        p.candidates(self.obs(s,-.42),Target(s-.4,0.,speed=2.2,confidence=.99,observations=100))
        c=p.response_choice;self.assertIsNotNone(c)
        self.assertLess(c.response['no_response_gap'],0.)
        self.assertGreaterEqual(c.response['minimum_response_gap'],p.response_margin)
        for pred in c.response['predictions']:
            xy=np.array(pred['path']);self.assertLessEqual(p.path_curvature(xy).max(),np.tan(MAX_STEER)/WHEELBASE)
        self.assertGreater(c.response['no_response_contact_time'],0.)
    def test_wall_and_unknown_lidar_returns_are_not_released(self):
        angles=self.env.unwrapped.sim.agents[0].scan_angles
        d=ResponseStage4(self.g,angles);d.planner.control_dt=.05;s=self.g.length*8/60
        target=Target(s-.4,0.,speed=2.2,confidence=.99,observations=100)
        d.planner.candidates(self.obs(s,-.42),target);c=d.planner.response_choice
        values=np.array([.1,.2,.3]);tracked=np.array([False,True,False])
        changed=d.response_speed_caps(values,tracked,c,self.obs(s),target)
        np.testing.assert_array_equal(changed[[0,2]],values[[0,2]])
        c.response['no_response_contact_time']=.3
        np.testing.assert_array_equal(d.response_speed_caps(values,tracked,c,self.obs(s),target),values)
    def test_unknown_opponent_does_not_enable_bet(self):
        p=ResponsePlanner(self.g,stage=3);p.control_dt=.05;s=self.g.length*8/60
        p.candidates(self.obs(s),Target(s+1.25,0.,speed=2.2,confidence=.2,observations=100))
        self.assertIsNone(p.response_choice)

if __name__=='__main__':unittest.main()
