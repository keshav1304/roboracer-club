"""Completion-oriented planning without privileged opponent state."""
import unittest
from unittest.mock import patch
import numpy as np
from world import make_env, reset_scenario
from track_geometry import TrackGeometry
from perception import Target
from maneuver_core import CandidatePlanner, NeuralDriver, Candidate, ACTIONS, FEATURE_NAMES


class MovingCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = make_env(seed=8401)
        cls.g = TrackGeometry(cls.env.unwrapped.track)
        cls.angles = cls.env.unwrapped.sim.agents[0].scan_angles

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def observation(self, s, d=0., v=2.4):
        xy, yaw, _ = self.g.sample(s, d)
        return dict(pose_x=float(xy[0]), pose_y=float(xy[1]), pose_theta=float(yaw),
                    linear_vel_x=v, scan=np.full(len(self.angles),20.))

    def test_blocked_merge_can_continue_in_clear_existing_lane(self):
        ss=np.linspace(0,self.g.length,200,endpoint=False)
        left,right=self.g.widths(ss)
        s=float(ss[np.argmax(np.minimum(left,right))])
        planner=CandidatePlanner(self.g,stage=2)
        obs=self.observation(s,-.51)
        target=Target((s-.1)%self.g.length,0.,speed=2.4,confidence=.98,observations=100)
        candidate=planner.candidates(obs,target)[2]
        self.assertGreater(candidate.speed,.5)
        _,lateral=self.g.project(candidate.path[:,0],candidate.path[:,1])
        self.assertLess(float(lateral.max()),-.4)
        tangents=np.gradient(candidate.path,axis=0)
        yaw=np.arctan2(tangents[:,1],tangents[:,0])
        self.assertGreater(float(self.g.body_clearance(candidate.path,yaw).min()),.06)

    def test_same_clear_track_policy_limits_for_both_moving_stages(self):
        for s in (0.,self.g.length*.27,self.g.length*.64):
            observation=self.observation(s)
            a=CandidatePlanner(self.g,stage=2).candidates(observation,None)[2]
            b=CandidatePlanner(self.g,stage=3).candidates(observation,None)[2]
            np.testing.assert_allclose([a.speed,a.steering],[b.speed,b.steering],atol=1e-7)

    def test_postpass_corner_recovery_does_not_latch_into_zero_speed(self):
        planner=CandidatePlanner(self.g,stage=2)
        # Recorded no-target states from train018: a nearly recovered car was
        # stopped by a 6.8cm preview clearance against a 7cm nominal buffer.
        for x,y,yaw,v in ((9.95533466,-.98022699,5.19981909,1.91859746),
                          (10.02145386,-1.16494739,5.04728174,.17128402)):
            obs=dict(pose_x=x,pose_y=y,pose_theta=yaw,linear_vel_x=v)
            candidate=planner.candidates(obs,None)[2]
            self.assertGreater(candidate.speed,.2)
            self.assertGreater(candidate.clearance,.02)

    def test_rotated_front_corner_is_not_mistaken_for_clear_lateral_lane(self):
        planner=CandidatePlanner(self.g,stage=2)
        target=Target(.2,.46,speed=0.,observations=100)
        # Lateral centre separation .46 exceeds width+margin .43, yet the
        # rotated body reaches into the ego rectangle at this tight corner.
        with patch.object(self.g,'sample',return_value=(np.array([[.2,.46]]),np.array([np.pi/4]),None)):
            clearance=planner.motion_clearance(np.array([[0.,0.]]),np.array([0.]),target,np.array([0.]))
        self.assertLess(clearance,0.)

    def test_actuator_rates_and_scan_interval_scale_with_control_period(self):
        driver=NeuralDriver(self.g,self.angles,stage=2,teacher=True)
        obs=self.observation(0.,v=1.)
        fixture=[Candidate(a,a=='WAIT',0.,1.,.4,2.4,0.,np.zeros((2,2))) for a in ACTIONS]
        with patch.object(driver.sensor,'update',return_value=None) as sensor, \
             patch.object(driver.planner,'candidates',return_value=fixture), \
             patch.object(driver.planner,'features',return_value=np.zeros(len(FEATURE_NAMES))):
            steering,speed=driver.plan(obs,dt=.05)
        sensor.assert_called_once_with(obs,dt=.05)
        self.assertAlmostEqual(steering,.125)
        self.assertAlmostEqual(speed,1.1)

    def test_outcome_head_prioritizes_success_over_imitation_preference(self):
        driver=NeuralDriver(self.g,self.angles,stage=2,teacher=True)
        class Imitation:
            def proba(self, features):return np.array([.90,.05,.05])
        class Outcome:
            def predict_proba(self, features):
                np.testing.assert_equal(features.shape,(3,54))
                np.testing.assert_allclose(features[:,-4:],np.eye(4)[:3])
                return np.array([[.8,.2],[.2,.8],[.9,.1]])
        driver.teacher=False
        driver.model=Imitation()
        driver.value_model=Outcome()
        fixture=[Candidate(a,True,1.,1.,0.,2.,0.,np.zeros((2,2))) for a in ACTIONS]
        target=Target(3.,0.,speed=1.7,observations=100,confidence=.98)
        with patch.object(driver.sensor,'update',return_value=target), \
             patch.object(driver.planner,'candidates',return_value=fixture), \
             patch.object(driver.planner,'features',return_value=np.zeros(len(FEATURE_NAMES))):
            driver.plan(self.observation(0.))
        self.assertEqual(driver.action,'RIGHT')
        self.assertEqual(driver.telemetry['action_success_probabilities'],[.2,.8,.1])

    def test_teacher_bias_cannot_make_infeasible_action_safe(self):
        driver=NeuralDriver(self.g,self.angles,stage=2,teacher=True)
        driver.teacher_bias=np.array([-10.,0.,0.])
        fixture=[Candidate(a,a=='WAIT',0.,1.,0.,0.,0.,np.zeros((2,2))) for a in ACTIONS]
        with patch.object(driver.sensor,'update',return_value=None), \
             patch.object(driver.planner,'candidates',return_value=fixture), \
             patch.object(driver.planner,'features',return_value=np.zeros(len(FEATURE_NAMES))):
            driver.plan(self.observation(0.))
        self.assertEqual(driver.action,'WAIT')


if __name__=='__main__':
    unittest.main()
