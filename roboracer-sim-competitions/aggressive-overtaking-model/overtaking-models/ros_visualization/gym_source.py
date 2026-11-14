"""One live Gym environment; atomic local snapshots for the ROS display process."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark import offline_only
from world import make_env
from track_geometry import TrackGeometry
from expanded_planner import ExpandedStage2, ExpandedStage4
import run_experiment as experiment


def atomic_json(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,allow_nan=False,separators=(',',':')))
    os.replace(tmp,path)


def choose_scenes(stage):
    folder=ROOT/'artifacts/expanded_planner_evaluation_r6_20261107'
    jobs=json.loads((folder/'manifest.json').read_text())['jobs']
    scenes=[]
    for number in ((2,4) if stage=='all' else (int(stage),)):
        eligible=[]
        for job in jobs:
            if job['stage']!=number or not job['expanded']:continue
            result=json.loads((folder/(job['id']+'.trace.json')).read_text())['result']
            if result['success']:
                previous=json.loads(Path(job['baseline_trace']).read_text())['result'] if job.get('baseline_trace') else None
                priority=0 if previous and not previous['success'] else 1
                eligible.append((priority,result['passed_at'],job['id'],job['scenario']))
        _,_,name,scenario=min(eligible)
        scenes.append((number,name,scenario))
    return scenes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['2','4','all'],default='all')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--show-counterfactual',action='store_true')
    parser.add_argument('--response-experiment',type=Path,help='Show response-aware Stage 4 cases from a completed experiment')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--speed',type=float,default=.5,help='Wall-clock playback rate; physics timestep unchanged')
    parser.add_argument('--directory',type=Path,default=Path(__file__).parent/'live')
    args=parser.parse_args()
    if args.speed<=0:parser.error('--speed must be positive')
    if args.show_counterfactual:parser.error('Nonreactive opponent demos are disabled; all Stage 4 opponents must use avoidance.')
    args.directory.mkdir(parents=True,exist_ok=True)
    state={'driver':None,'time':0.,'decision_time':0.,'stage':2,'episode':0,'seq':0,'clock_offset':0.,'ticks':0,'quit':False}
    def capture(base):
        class Captured(base):
            def plan(self,obs,*a,**kw):
                self.planner.capture_paths=True
                command=super().plan(obs,*a,**kw)
                state['driver']=self
                state['decision_time']=state['time']
                return command
        return Captured
    stage4=ExpandedStage4
    scenes=[] if args.response_experiment else choose_scenes(args.stage)
    if args.response_experiment:
        from response_stage4 import ResponseStage4
        manifest=json.loads((args.response_experiment/"manifest.json").read_text())
        learned=manifest.get("controller")=="LearnedStage4"
        if learned:
            import importlib.util,hashlib,functools
            from learned_opponent import OpponentForecast
            source=args.response_experiment/'source'
            for name in ('learned_opponent.py','expanded_planner.py','maneuver_core.py','world.py','run_experiment.py'):
                expected=manifest['source_sha256'][name]
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
                    raise RuntimeError(f'{name} changed since this experiment; rerun validation before displaying it')
            spec=importlib.util.spec_from_file_location('displayed_learned_stage4',source/'learned_stage4.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            module.OpponentForecast=functools.partial(OpponentForecast,path=source/'opponent_forecast.npz')
            stage4=module.LearnedStage4
        else:stage4=ResponseStage4
        examples=[]
        for path in args.response_experiment.glob('*.json'):
            data=json.loads(path.read_text())
            if 'job' not in data or data['job']['policy']!='response' or data['job']['mode']!='reactive':continue
            if data['result']['success'] and data['result']['collision'] is None and (learned or data['response_counts']['selected']):
                baseline_path=path.with_name(path.name.replace('-response','-baseline'))
                baseline=json.loads(baseline_path.read_text()) if baseline_path.exists() else None
                priority=0 if data.get('yield_metrics',{}).get('lateral_yield_pass') else 1 if baseline and not baseline['result']['success'] else 2 if data['response_counts']['selected'] else 3
                examples.append((priority,data['result']['passed_at'] or 999.,data))
        if not examples:raise RuntimeError('No executed response-aware challenge in this experiment; refusing a misleading demo')
        selected_examples=sorted(examples,key=lambda x:x[:2])[:2]
        scenes=[(4,item[2]['job']['id'],item[2]['job']['scenario']) for item in selected_examples]
    experiment.MODELS.update({2:capture(ExpandedStage2),3:capture(stage4)})
    with offline_only():
        env=make_env(render=not args.headless,seed=12701)
        g=TrackGeometry(env.unwrapped.track)
        track=env.unwrapped.track
        grid=np.where(track.occupancy_map>128,0,100).astype(int)
        reference=g.sample(np.linspace(0,g.length,600))[0]
        atomic_json(args.directory/'map.json',dict(resolution=float(track.spec.resolution),
            origin=list(map(float,track.spec.origin)),width=grid.shape[1],height=grid.shape[0],
            data=grid.ravel().tolist(),reference=reference.tolist()))
        angles=env.unwrapped.sim.agents[0].scan_angles
        original_step=env.step
        start_wall=time.monotonic()
        def step(commands):
            result=original_step(commands)
            state['time']+=.01;state['ticks']+=1
            driver=state['driver']
            if driver is not None and state['ticks']%5==0:
                obs=result[0]
                cars=[[float(obs[f'agent_{i}'][k]) for k in ('pose_x','pose_y','pose_theta','linear_vel_x')] for i in (0,1)]
                te={k:v for k,v in driver.telemetry.items() if k!='features'}
                packet=dict(version=1,sequence=state['seq'],episode=state['episode'],stage=state['stage'],
                    scene=state['scene'],simulation_time=state['time'],clock_time=state['clock_offset']+state['time'],decision_time=state['decision_time'],
                    wall_time=time.time(),cars=cars,commands=np.asarray(commands).tolist(),
                    opponent_offset_m=float(g.project(cars[1][0],cars[1][1])[1]),
                    paths=driver.planner.debug_paths,representatives=driver.planner.debug_representatives,
                    telemetry=te,scan=np.nan_to_num(obs['agent_0']['scan'],nan=30.,posinf=30.,neginf=0.).tolist(),
                    angle_min=float(angles[0]),angle_max=float(angles[-1]),status='running')
                atomic_json(args.directory/'frame.json',packet)
                state['seq']+=1
                if not args.headless:
                    delay=start_wall+state['time']/args.speed-time.monotonic()
                    if delay>0:time.sleep(delay)
            return result
        env.step=step
        if not args.headless:
            import pygame
            original_render=env.render
            original_events=pygame.event.get
            def events(*a,**kw):
                values=original_events(*a,**kw)
                if any(v.type==pygame.QUIT for v in values):state['quit']=True
                return values
            pygame.event.get=events
            env.unwrapped.renderer.render_fps=120
            def draw():
                if state['quit']:raise KeyboardInterrupt
                if state['ticks']%5==0:
                    pygame.display.set_caption(f"F1TENTH Gym + ROS/RViz | Stage {state['stage']} | {state.get('scene','')} | {args.speed:g}x speed")
                    return original_render()
            env.render=draw
        results=[]
        try:
            while True:
                for stage,name,scenario in scenes:
                    if stage==4 and scenario.get('opponent')!='reactive':
                        raise RuntimeError('Stage 4 display requires the reactive opponent')
                    experiment.MODELS[3]=capture(stage4)
                    state['clock_offset']+=state['time']+.01
                    state.update(driver=None,time=0.,ticks=0,stage=stage,scene=name,episode=state['episode']+1)
                    start_wall=time.monotonic()
                    print(f"LIVE Gym Stage {stage}: {name}; ROS display reads local snapshots",flush=True)
                    result,*_=experiment.episode(experiment.Scenario(**scenario),env=env,geometry=g,
                                                   render=not args.headless,control_dt=.05)
                    results.append(dict(stage=stage,scene=name,result=result))
                    results=results[-20:]
                    atomic_json(args.directory/'results.json',results)
                    print(f"Stage {stage}: {result['outcome']}; pass={result['passed_at']}",flush=True)
                    frame_path=args.directory/'frame.json'
                    if frame_path.exists():
                        packet=json.loads(frame_path.read_text());packet['status']=result['outcome']
                        atomic_json(frame_path,packet)
                    if not args.headless:time.sleep(2.)
                if args.once:break
        except KeyboardInterrupt:pass
        finally:
            env.close()
            if not args.headless:pygame.event.get=original_events

if __name__=='__main__':main()
