"""Stage3 padding recovery must retain physical, front, and unknown-object checks."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np

from maneuver_core import ACTIONS, FEATURE_NAMES, Candidate, CandidatePlanner, NeuralDriver
from perception import Target


class StraightGeometry:
    length=100.

    def __init__(self,wall_return=False):
        self.wall_return=wall_return

    def project(self,x,y):
        return x,y

    def sample(self,s,d=0.):
        s,d=np.broadcast_arrays(np.asarray(s,float),np.asarray(d,float))
        return np.stack([s,d],axis=-1),np.zeros_like(s),np.zeros_like(s)

    def clearance(self,points):
        return np.full(len(points),0. if self.wall_return else 1.)


class PaddingRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.geometry=StraightGeometry()
        self.planner=CandidatePlanner(self.geometry,stage=3)
        self.target=Target(0.,0.,speed=0.,confidence=.98,observations=100)
        self.xy=np.array([[0.,.35],[.2,.39],[.4,.41],[.6,.45]])
        self.yaws=np.zeros(4)
        self.times=np.array([0.,.25,.5,.75])
        self.obs=dict(pose_x=0.,pose_y=.35,pose_theta=0.)

    def certify(self,xy=None,times=None,obs=None,target=None,planner=None):
        return (planner or self.planner).certify_car_departure(
            self.xy if xy is None else xy,self.yaws,
            self.target if target is None else target,
            self.times if times is None else times,self.obs if obs is None else obs)

    def test_predicted_departure_restores_padding_without_crossing_hard_floor(self):
        self.assertTrue(self.certify())
        below_floor=self.xy.copy();below_floor[0,1]=.334
        self.assertFalse(self.certify(xy=below_floor,obs=dict(self.obs,pose_y=.334)))
        converging=self.xy.copy();converging[1,1]=.344
        self.assertFalse(self.certify(xy=converging))
        late_loss=self.xy.copy();late_loss[-1,1]=.36
        self.assertFalse(self.certify(xy=late_loss))
        unrestored=self.xy.copy();unrestored[1:,1]=.35
        self.assertFalse(self.certify(xy=unrestored))

    def test_certificate_requires_fresh_observation_and_explicit_permission(self):
        self.assertFalse(self.certify(target=replace(self.target,confidence=.59)))
        self.assertFalse(self.certify(target=replace(self.target,time_since_seen=.11)))
        self.assertFalse(self.certify(times=np.array([0.,.1,.2,.4])))
        self.assertFalse(self.certify(times=np.array([0.,.25,.5,np.inf])))
        for stage in(1,2):
            self.assertFalse(self.certify(planner=CandidatePlanner(self.geometry,stage=stage)))

    def test_actual_initial_pose_is_used_only_inside_certificate(self):
        xy=self.xy.copy();xy[0,1]=.50
        original_xy=xy.copy();original_yaws=self.yaws.copy()
        self.assertTrue(self.certify(xy=xy))
        self.assertFalse(self.certify(xy=xy,obs=dict(self.obs,pose_theta=np.pi/4)))
        np.testing.assert_array_equal(xy,original_xy)
        np.testing.assert_array_equal(self.yaws,original_yaws)

    def run_brake(self,points,*,certified=True,stage=3,target=True,wall=False):
        points=np.asarray(points,float)
        angles=np.arctan2(points[:,1],points[:,0])
        geometry=StraightGeometry(wall_return=wall)
        driver=NeuralDriver(geometry,angles,stage=stage,teacher=True)
        obs=dict(pose_x=0.,pose_y=0.,pose_theta=0.,linear_vel_x=1.,
                 scan=np.linalg.norm(points,axis=1))
        tracked=Target(0.,.35,speed=0.,confidence=.98,observations=100) if target else None
        candidates=[Candidate(action,action=='WAIT',0.,.04,0.,2.,0.,
                              np.zeros((2,2)),certified if action=='WAIT' else False)
                    for action in ACTIONS]
        with patch.object(driver.sensor,'update',return_value=tracked), \
             patch.object(driver.planner,'candidates',return_value=candidates), \
             patch.object(driver.planner,'features',return_value=np.zeros(len(FEATURE_NAMES))):
            command=driver.plan(obs,dt=.05)
        return command,driver

    def test_known_side_padding_requires_certificate_and_stage_permission(self):
        command,driver=self.run_brake([[.1,.19]])
        self.assertGreater(command[1],1.)
        self.assertEqual(driver.padding_recovery_steps,1)
        self.assertEqual(driver.padding_lidar_released_beams,1)
        self.assertEqual(driver.telemetry['padding_released_beams'],1)
        for kwargs in(dict(certified=False),dict(stage=2),dict(target=False)):
            with self.subTest(kwargs=kwargs):
                command,driver=self.run_brake([[.1,.19]],**kwargs)
                self.assertEqual(command[1],0.)
                self.assertEqual(driver.padding_lidar_released_beams,0)

    def test_front_wall_unknown_and_physical_body_beams_keep_original_brake(self):
        for points,kwargs in (([[.35,.18]],{}),([[.1,.19]],dict(wall=True)),
                              ([[.1,-.19]],{}),([[.1,.16]],{})):
            with self.subTest(points=points,kwargs=kwargs):
                command,driver=self.run_brake(points,**kwargs)
                self.assertEqual(command[1],0.)
                self.assertEqual(driver.padding_lidar_released_beams,0)
        # Releasing an identified side return cannot hide another blocking ray.
        command,driver=self.run_brake([[.1,.19],[.35,-.18]])
        self.assertEqual(driver.padding_lidar_released_beams,1)
        self.assertEqual(command[1],0.)

    def test_profile_retains_shared_cruise_and_stage2_padding(self):
        stage2=CandidatePlanner(self.geometry,stage=2).planning_profile()
        stage3=self.planner.planning_profile()
        self.assertEqual(stage2['lateral_acceleration_m_s2'],stage3['lateral_acceleration_m_s2'])
        self.assertEqual(stage2['passing_lateral_acceleration_m_s2'],2.4)
        self.assertEqual(stage3['passing_lateral_acceleration_m_s2'],3.2)
        self.assertFalse(stage2['padding_recovery_enabled'])
        self.assertTrue(stage3['padding_recovery_timing_is_prediction_only'])
        self.assertEqual(stage2['probe_preview_distance_m'],6.)
        self.assertEqual(stage2['probe_offset_m'],.30)


if __name__=='__main__':
    unittest.main()
