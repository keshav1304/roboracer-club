"""Re-score observed yielding; separate outward avoidance from ordinary drift."""
import argparse,json
from pathlib import Path
import numpy as np
from world import make_env
from track_geometry import TrackGeometry

def metrics(g,frames,result):
    ego=np.array([f['ego'] for f in frames]);opp=np.array([f['opponent'] for f in frames])
    _,ed=g.project(ego[:,0],ego[:,1]);_,od=g.project(opp[:,0],opp[:,1])
    near=np.linalg.norm(ego[:,:2]-opp[:,:2],axis=1)<2.
    outward=-np.sign(ed)*od
    # A lateral displacement alone may be corner tracking error. Require an
    # observed target, an avoidance action AWAY from ego, and an ongoing pass.
    avoiding=np.array([f.get('opponent_target') is not None and f['opponent_action']==('LEFT' if d<0 else 'RIGHT') for f,d in zip(frames,ed)])
    ongoing=np.array([-.9<f['relative_progress']<.8 for f in frames])
    yielding=near & (abs(ed)>.15) & (outward>.12) & avoiding & ongoing
    longest=streak=0
    for flag in yielding:streak=streak+1 if flag else 0;longest=max(longest,streak)
    return dict(definition_version=2,lateral_yield=bool(longest*.05>=.25),longest_lateral_yield_s=longest*.05,
        outward_displacement_m=float(np.max(outward[near])) if near.any() else 0.,
        opponent_max_offset_near_m=float(np.max(abs(od[near]))) if near.any() else 0.,
        lateral_yield_pass=bool(result['success'] and result['collision'] is None and longest*.05>=.25))

def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
    env=make_env();g=TrackGeometry(env.unwrapped.track);groups={k:[] for k in ('baseline','response')}
    for f in sorted(a.folder.glob('case-*.json')):
        data=json.loads(f.read_text());data['yield_metrics']=metrics(g,data['frames'],data['result'])
        f.write_text(json.dumps(data,separators=(',',':')));groups[data['job']['policy']].append(data)
    env.close();summary={}
    for policy,rows in groups.items():
        errors=[e for r in rows for e in r['prediction_errors']]
        summary[policy]=dict(trials=len(rows),passes=sum(r['result']['success'] for r in rows),contacts=sum(r['result']['collision'] is not None for r in rows),
            lateral_yields=sum(r['yield_metrics']['lateral_yield'] for r in rows),lateral_yield_passes=sum(r['yield_metrics']['lateral_yield_pass'] for r in rows),
            learned_selected_decisions=sum(r['response_counts']['selected'] for r in rows),
            median_pass_time_s=float(np.median([r['result']['passed_at'] for r in rows if r['result']['success']])) if any(r['result']['success'] for r in rows) else None,
            forecast_samples=len(errors),forecast_rmse_m=float(np.sqrt(np.mean([e['error_m']**2 for e in errors]))) if errors else None,
            constant_motion_rmse_m=float(np.sqrt(np.mean([e['constant_error_m']**2 for e in errors]))) if errors else None)
    report=dict(definition='Opponent avoids away from ego, target observed, opposite line offset >0.12m for >=0.25s, within 2m, ego relative progress between -0.9m and +0.8m.',summary=summary)
    (a.folder/'yield_audit.json').write_text(json.dumps(report,indent=2));(a.folder/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
