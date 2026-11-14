"""Live F1TENTH Gym human renderer. No HTML, replay player, or API calls."""
import argparse,json
from pathlib import Path
from benchmark import offline_only
from world import make_env
from track_geometry import TrackGeometry
import run_experiment as experiment
from expanded_planner import ExpandedStage2,ExpandedStage4

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('2','4','all'),default='all')
    p.add_argument('--baseline',action='store_true')
    p.add_argument('--once',action='store_true')
    p.add_argument('--results',type=Path,default=ROOT/'artifacts/expanded_planner_evaluation_r6_20261107')
    args=p.parse_args()
    if not args.baseline:experiment.MODELS.update({2:ExpandedStage2,3:ExpandedStage4})
    scenes=[]
    for stage in ((2,4) if args.stage=='all' else (int(args.stage),)):
        choice=None
        if args.results and (args.results/'manifest.json').exists():
            jobs=json.loads((args.results/'manifest.json').read_text())['jobs']
            candidates=[]
            for job in jobs:
                path=args.results/(job['id']+'.trace.json')
                if job['stage']==stage and job['expanded'] and path.exists():
                    r=json.loads(path.read_text())['result']
                    previous=json.loads(Path(job['baseline_trace']).read_text())['result'] if job.get('baseline_trace') else None
                    priority=0 if r['success'] and previous and not previous['success'] else 1 if r['success'] else 2
                    candidates.append((priority,r['passed_at'] or 999.,job['scenario']))
            if candidates:choice=min(candidates,key=lambda x:x[:2])[2]
        if choice is None:
            if stage==2:
                path=ROOT/'artifacts/rollout_final_20261107/trials/stage2/test/neural/stage2-test-023-learned.trace.json'
                choice=json.loads(path.read_text())['result']['scenario']
            else:
                path=ROOT/'artifacts/stage4_45s_20261107/stage4-reactive-00-0.trace.json'
                choice=json.loads(path.read_text())['result']['harness_scenario']
        scenes.append((stage,choice))
    with offline_only():
        env=make_env(render=True,seed=12701)
        geometry=TrackGeometry(env.unwrapped.track)
        import pygame
        original_render=env.render
        env.unwrapped.renderer.render_fps=20
        state={'draw_calls':0,'stage':2,'quit':False}
        original_events=pygame.event.get
        def events(*a,**kw):
            items=original_events(*a,**kw)
            if any(item.type==pygame.QUIT for item in items):state['quit']=True
            return items
        pygame.event.get=events
        def draw():
            state['draw_calls']+=1
            if state['draw_calls']%5==0:
                pygame.display.set_caption(f"F1TENTH Gym | Stage {state['stage']} | {'baseline' if args.baseline else 'expanded planner'} | click a car to follow; close window to exit")
                frame=original_render()
                if state['draw_calls']==5:
                    print(f"Native display active: {pygame.display.get_driver()}, {pygame.display.get_window_size()}",flush=True)
                if state['quit']:raise KeyboardInterrupt
                return frame
        env.render=draw
        try:
            while True:
                for stage,scenario in scenes:
                    state['stage']=stage
                    print(f"Live F1TENTH Gym: Stage {stage}, {'baseline' if args.baseline else 'expanded'} planner",flush=True)
                    r,_,_,_=experiment.episode(experiment.Scenario(**scenario),env=env,geometry=geometry,render=True,control_dt=.05)
                    if state['quit']:return
                    print(f"Stage {stage}: {r['outcome']}; pass time {r['passed_at']}",flush=True)
                    if not r['success'] and not r['collision'] and r['sim_seconds']<scenario['seconds']-.05:return
                if args.once:break
        except KeyboardInterrupt:
            return
        finally:
            pygame.event.get=original_events
            env.close()

if __name__=='__main__':main()
