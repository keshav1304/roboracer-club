"""The visualization must observe, never change the chosen controls."""
from pathlib import Path
import sys
import unittest
import importlib.util
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from world import make_env,reset_scenario
from track_geometry import TrackGeometry
from expanded_planner import ExpandedStage2,ExpandedStage4

frozen=Path(__file__).resolve().parents[1]/'artifacts/expanded_planner_evaluation_r6_20261107/source/expanded_planner.py'
spec=importlib.util.spec_from_file_location('frozen_expanded_planner',frozen)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

class CaptureParity(unittest.TestCase):
    def test_decision_parity(self):
        env=make_env(seed=6501)
        try:
            g=TrackGeometry(env.unwrapped.track)
            angles=env.unwrapped.sim.agents[0].scan_angles
            for cls,previous in ((ExpandedStage2,old.ExpandedStage2),(ExpandedStage4,old.ExpandedStage4)):
                a,b=previous(g,angles,vmax=2.4),cls(g,angles,vmax=2.4)
                b.planner.capture_paths=True
                obs,_=reset_scenario(env,g,49.,1.25,2.1,ego_speed=2.1)
                for _ in range(35):
                    ca=a.plan(obs['agent_0'],dt=.05);cb=b.plan(obs['agent_0'],dt=.05)
                    np.testing.assert_array_equal(ca,cb)
                    np.testing.assert_array_equal(a.telemetry['features'],b.telemetry['features'])
                    self.assertEqual(a.telemetry['action'],b.telemetry['action'])
                    self.assertEqual(len(b.planner.debug_paths),b.planner.search_stats['evaluated'])
                    selected=next(p for p in b.planner.debug_representatives if p['action']==b.action)
                    self.assertEqual(len(selected['path']),41 if b.action!='PROBE' else 21)
                    obs,*_=env.step(np.array([ca,(0.,2.1)]))
        finally:env.close()

if __name__=='__main__':unittest.main()
