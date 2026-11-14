"""Serial compute comparison on identical recorded sensor streams, offline."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from benchmark import offline_only,record_streams
from world import make_env,timing_summary
from track_geometry import TrackGeometry
from moving_model import MovingModel
from interactive_model import InteractiveModel
from expanded_planner import ExpandedStage2,ExpandedStage4

def main():
    with offline_only():
        env=make_env(seed=12702)
        try:
            g=TrackGeometry(env.unwrapped.track);angles=env.unwrapped.sim.agents[0].scan_angles.copy()
            streams,scenes=record_streams(g,env,measured_per_scene=600,warmup=100)
        finally:env.close()
        result={}
        for name,cls in (('stage2_baseline',MovingModel),('stage2_expanded',ExpandedStage2),
                         ('stage4_baseline',InteractiveModel),('stage4_expanded',ExpandedStage4)):
            full=[];network=[]
            for stream in streams:
                driver=cls(g,angles)
                for i in range(0,len(stream),5):
                    t=time.perf_counter_ns();driver.plan(stream[i],dt=.05);elapsed=(time.perf_counter_ns()-t)/1e6
                    if i>=100:full.append(elapsed);network.append(driver.network_times[-1])
            result[name]={'whole_controller':timing_summary(full),'networks':timing_summary(network)}
        out=Path('artifacts/expanded_planner_evaluation_r6_20261107/serial_benchmark.json')
        out.write_text(json.dumps({'results':result,'scenes':scenes,'network_access':'blocked',
            'expanded_planner_sha256':hashlib.sha256(Path('expanded_planner.py').read_bytes()).hexdigest(),
            'note':'Warmed serial execution on two recorded scenes; excludes physics/rendering. Not a worst-case deadline guarantee or real-car reaction time.'},indent=2)+'\n')
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
