"""All paired results, standstills, and actual versus planned lateral extent."""
import argparse,json,statistics
from pathlib import Path
import numpy as np
from world import make_env
from track_geometry import TrackGeometry
p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();m=json.loads((a.folder/'manifest.json').read_text())
groups={'baseline':{},'response':{}};env=make_env();g=TrackGeometry(env.unwrapped.track)
for job in m['jobs']:
 d=json.loads((a.folder/(job['id']+'.json')).read_text());frames=d['frames'];xy=np.array([f['ego'][:2] for f in frames]);offset=g.project(xy[:,0],xy[:,1])[1]
 longest=streak=0
 for f in frames:
  streak=streak+1 if f['ego'][3]<.1 and f['opponent'][3]<.1 else 0;longest=max(longest,streak)
 goals=[abs(x['interaction']['selected_lateral_goal_m']) for x in d.get('observations',[]) if 'selected_lateral_goal_m' in x['interaction']]
 d['extent']=dict(maximum_actual_abs_offset_m=float(np.max(np.abs(offset))),maximum_selected_abs_goal_m=max(goals,default=0.),longest_mutual_stop_seconds=longest*.05)
 groups[job['policy']][job['id'].split('-')[1]]=d
env.close();summary={}
for policy,items in groups.items():
 ds=list(items.values());r=[d['result'] for d in ds]
 summary[policy]=dict(trials=len(ds),passes=sum(x['success'] for x in r),contacts=sum(x['collision'] is not None for x in r),timeouts=sum(x['outcome']=='no_pass' for x in r),median_trial_p95_ms=statistics.median(x['controller_latency']['p95_ms'] for x in r),max_trial_p95_ms=max(x['controller_latency']['p95_ms'] for x in r),maximum_actual_abs_offset_m=max(d['extent']['maximum_actual_abs_offset_m'] for d in ds),maximum_selected_abs_goal_m=max(d['extent']['maximum_selected_abs_goal_m'] for d in ds),runs_with_mutual_stop_over_3s=sum(d['extent']['longest_mutual_stop_seconds']>3. for d in ds))
paired=[]
for key,b in groups['baseline'].items():
 n=groups['response'][key];paired.append(dict(case=key,baseline=b['result']['outcome'],new=n['result']['outcome'],baseline_seconds=b['result']['passed_at'],new_seconds=n['result']['passed_at'],baseline_extent=b['extent'],new_extent=n['extent']))
analysis=dict(summary=summary,paired=paired,new_only_passes=sum(x['new']=='pass' and x['baseline']!='pass' for x in paired),baseline_only_passes=sum(x['baseline']=='pass' and x['new']!='pass' for x in paired))
(a.folder/'analysis.json').write_text(json.dumps(analysis,indent=2))
lines=['STAGE 4 WALL-BOUNDED CORRIDOR SEARCH',str(a.folder.resolve()),'',m['purpose'],'','Baseline: the previous InteractionStage4, with its corrected opponent predictor and fallback. New: CorridorStage4. Actual F1TENTH Gym on the same imported AutoDRIVE default track. Opponent is unchanged reactive Stage 2. Neural checkpoints and opponent margins unchanged. No API calls.','']
for policy,s in summary.items():lines.append(policy+': '+json.dumps(s))
lines+=['','All paired cases:']+[json.dumps(x) for x in paired]
lines+=['','WHAT CHANGED','The previous fallback sampled fixed nearby offsets and ranked some paths using a speed that its final LiDAR brake would suppress. The new search samples lateral goals at roughly 0.25 m spacing across the available map corridor. When the old 3 m width lookup reaches its range limit, ray tracing continues to the first actual wall or map edge. There is no fixed maximum lane-change offset.', 'Lane-change distances: 0.65, 1.2, 2.4, 3.6 m; trial speed caps: 0.35, 1.2, 2.4 m/s. The normal LiDAR braking limit is included before evaluating new candidate motion. Full candidate paths must fit the walls, respect the turning-radius limit, and retain forward direction. The original body/interaction checks still protect against cars. Lateral search extent is bounded by walls; that does not remove collision or physical steering constraints.', 'Broad maneuvers are initiated before braking into a bend and retained through the encounter. A car already in a slow corner preserves the previous local recovery, with wider search available when its movement is blocked or it stops gaining ground. A clear existing trajectory is retained unless an alternative provides enough extra predicted progress.','','WHY THE TWO OLD DEADLOCKS HAPPENED','One repeatedly planned positive speed while the independent LiDAR brake reduced it to zero. The other ran out of useful nearby offsets. Opening the lateral search and considering executable speed enables earlier departures around those configurations. The diagnostic reruns test prevention from their original starts; they do not prove recovery from every already-interlocked stationary configuration.','','INTERPRETATION','Selected goal offset is not the measured car offset: receding-horizon planning may change its goal before reaching it. Both quantities are reported separately. Best RViz runs are selected successes; all failures remain in this report. No simulated opponent commands or true opponent poses enter the controller; truth is used only for setup and scoring.', 'Success requires passing, returning near the reference, and a two-second follow-through. Timeout is 40 seconds unless the manifest specifies otherwise. Zero contacts in a finite test is not a guarantee for other scenarios. Whole-controller p95 above 50 ms misses a 20 Hz wall-clock budget; timing here was measured during parallel experiments.']
(a.folder/'REPORT.txt').write_text('\n'.join(lines)+'\n');print(json.dumps(analysis,indent=2))
