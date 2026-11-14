"""Causal Gym check: replay identical ego commands with/without rival reactivity."""
import json
import argparse
from pathlib import Path
import numpy as np
import run_experiment as ex
from expanded_planner import ExpandedStage4
from benchmark import offline_only

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=ROOT/'artifacts/response_stage4_targeted_r3')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cases',type=int,nargs='+',default=[0,2,3]);args=parser.parse_args()
    out=args.output;out.mkdir(exist_ok=False)
    summaries=[]
    for index in args.cases:
        source=args.source/f'target-{index}-response.json'
        recorded=json.loads(source.read_text());commands=[f['command'] for f in recorded['frames']]
        class Replay(ExpandedStage4):
            def __init__(self,*a,**kw):super().__init__(*a,**kw);self.replay_index=0
            def plan(self,*a,**kw):
                super().plan(*a,**kw)
                command=tuple(commands[min(self.replay_index,len(commands)-1)])
                self.replay_index+=1;self.telemetry['command']=list(command)
                return command
        ex.MODELS[3]=Replay
        for mode in ('reactive','line'):
            scenario=dict(recorded['job']['scenario']);scenario['opponent']=mode
            scenario['seconds']=round(recorded['result']['sim_seconds'],2)
            with offline_only():result,frames,_,_=ex.episode(ex.Scenario(**scenario),control_dt=.05)
            common=min(len(frames),len(recorded['frames']))
            ego_error=float(np.max(np.abs(np.array([f['ego'] for f in frames[:common]])-np.array([f['ego'] for f in recorded['frames'][:common]]))))
            summary=dict(case=index,mode=mode,outcome=result['outcome'],collision=result['collision'],pass_time=result['passed_at'],ego_state_max_difference=ego_error)
            summaries.append(summary)
            (out/f'case-{index}-{mode}.json').write_text(json.dumps(dict(summary=summary,result=result,frames=frames),separators=(',',':')))
            print(json.dumps(summary),flush=True)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
