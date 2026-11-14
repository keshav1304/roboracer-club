"""Pass completion recovery requires observed history, not an ego/rival role."""
import unittest
from unittest.mock import patch
import numpy as np
from maneuver_core import ACTIONS, FEATURE_NAMES, Candidate, CandidatePlanner, NeuralDriver
from perception import Target
from test_padding_recovery_v3 import StraightGeometry

class PassCompletionTests(unittest.TestCase):
    def driver(self,stage=2):
        return NeuralDriver(StraightGeometry(),np.array([0.]),stage=stage,teacher=True)

    def context(self,driver,distance):
        obs=dict(pose_x=0.,pose_y=0.,pose_theta=0.)
        target=None if distance is None else Target(distance,0.,speed=2.,confidence=.98,observations=100)
        driver.update_pass_completion_context(obs,target)
        return driver.planner.pass_completion_recovery

    def test_only_observed_committed_pass_can_restore_padding(self):
        driver=self.driver();driver.commit=('LEFT',.5)
        self.assertFalse(self.context(driver,-.4))
        self.assertFalse(self.context(driver,1.))
        self.assertTrue(self.context(driver,-.4))
        self.assertFalse(self.context(driver,-.2))
        self.assertFalse(self.context(driver,-2.3))
        driver.commit=None
        self.assertFalse(self.context(driver,-.4))

    def test_lost_target_resets_observed_encounter(self):
        driver=self.driver();driver.commit=('LEFT',.5)
        self.context(driver,1.)
        self.assertTrue(self.context(driver,-.4))
        self.assertFalse(self.context(driver,None))
        self.assertFalse(self.context(driver,-.4))

    def test_completion_permission_is_stage2_only(self):
        for stage in(1,3):
            driver=self.driver(stage);driver.commit=('LEFT',.5)
            self.context(driver,1.)
            self.assertFalse(self.context(driver,-.4))

    def test_stage2_certificate_retains_normal_padding_and_hard_floor(self):
        driver=self.driver();planner=driver.planner;driver.commit=('LEFT',.5)
        self.context(driver,1.);self.context(driver,-.4)
        xy=np.array([[0.,.35],[.2,.45],[.4,.50],[.6,.55]])
        times=np.array([0.,.25,.5,.75]);yaws=np.zeros(4)
        obs=dict(pose_x=0.,pose_y=.35,pose_theta=0.)
        target=Target(0.,0.,speed=0.,confidence=.98,observations=100)
        self.assertTrue(planner.certify_car_departure(xy,yaws,target,times,obs))
        self.assertFalse(planner.certify_car_departure(xy,yaws,target,times,dict(obs,pose_y=.33)))
        planner.pass_completion_recovery=False
        self.assertFalse(planner.certify_car_departure(xy,yaws,target,times,obs))
        self.assertEqual(planner.car_margin,.12)
        self.assertFalse(planner.padding_recovery_enabled)
        self.assertTrue(planner.planning_profile()['pass_completion_recovery_enabled'])

    def test_rear_target_cannot_bypass_unsafe_merge_holding_lane(self):
        class WideStraight(StraightGeometry):
            def widths(self,s):
                return np.full_like(np.asarray(s,float),3.),np.full_like(np.asarray(s,float),3.)
            def body_clearance(self,xy,yaws):return np.ones(len(xy))
            def speed_at(self,*a,**kw):return 2.4
        planner=CandidatePlanner(WideStraight(),stage=2)
        obs=dict(pose_x=0.,pose_y=.5,pose_theta=0.,linear_vel_x=1.5)
        rear=Target(-.85,0.,speed=2.4,confidence=.98,observations=100)
        with patch.object(planner.tracker,'control',return_value=(0.,1.5)):
            wait=planner.candidates(obs,rear)[2]
        self.assertGreater(wait.speed,.2)
        self.assertTrue(np.allclose(wait.path[:,1],.5))

    def test_side_lidar_release_still_requires_current_completion_permission(self):
        for prior in(False,True):
            point=np.array([[.05,.19]]);angles=np.arctan2(point[:,1],point[:,0])
            driver=NeuralDriver(StraightGeometry(),angles,stage=2,teacher=True)
            driver.pass_target_was_ahead=prior;driver.commit=('RIGHT',-.5)
            target=Target(-.3,.35,speed=0.,confidence=.98,observations=100)
            obs=dict(pose_x=0.,pose_y=0.,pose_theta=0.,linear_vel_x=1.,scan=np.linalg.norm(point,axis=1))
            candidates=[Candidate(action,action=='WAIT',0.,.04,0.,2.,0.,np.zeros((2,2)),action=='WAIT') for action in ACTIONS]
            with patch.object(driver.sensor,'update',return_value=target),patch.object(driver.planner,'candidates',return_value=candidates),patch.object(driver.planner,'features',return_value=np.zeros(len(FEATURE_NAMES))):
                command=driver.plan(obs,dt=.05)
            self.assertEqual(driver.padding_lidar_released_beams,int(prior))
            self.assertEqual(driver.telemetry['pass_completion_recovery_eligible'],prior)
            self.assertGreater(command[1],1.) if prior else self.assertEqual(command[1],0.)

if __name__=='__main__':unittest.main()
