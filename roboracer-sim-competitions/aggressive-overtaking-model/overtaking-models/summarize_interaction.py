"""Paired outcomes and matched-observation prediction error (truth for scoring only)."""
import argparse,json,statistics
from pathlib import Path
import numpy as np
from world import make_env
from track_geometry import TrackGeometry
p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
m=json.loads((a.folder/'manifest.json').read_text());groups={'baseline':{},'response':{}}
for job in m['jobs']:
 groups[job['policy']][job['id'].split('-')[1]]=json.loads((a.folder/(job['id']+'.json')).read_text())
summary={};paired=[]
for policy,items in groups.items():
 rows=list(items.values());r=[d['result'] for d in rows];times=[x['passed_at'] for x in r if x['success']]
 summary[policy]=dict(trials=len(r),passes=sum(x['success'] for x in r),contacts=sum(x['collision'] is not None for x in r),timeouts=sum(x['outcome']=='no_pass' for x in r),median_successful_pass_seconds=statistics.median(times) if times else None,median_trial_p95_ms=statistics.median(x['controller_latency']['p95_ms'] for x in r),max_trial_p95_ms=max(x['controller_latency']['p95_ms'] for x in r),fallback_decisions=sum(d.get('interaction_counts',{}).get('overrides',0) for d in rows),released_padding_beams=sum(d.get('interaction_counts',{}).get('released_beams',0) for d in rows))
for key,b in groups['baseline'].items():
 n=groups['response'][key];paired.append(dict(case=key,baseline=b['result']['outcome'],new=n['result']['outcome'],baseline_seconds=b['result']['passed_at'],new_seconds=n['result']['passed_at']))
env=make_env();g=TrackGeometry(env.unwrapped.track);errors={str(h):dict(constant=[],trend=[],envelope=[]) for h in (.3,.6,.8)};speed_errors=dict(raw_reference_rate=[],converted_physical=[])
for data in groups['response'].values():
 frames=data['frames'];ts=np.array([f['t'] for f in frames]);truth=np.array([f['opponent'] for f in frames])
 for sample in data['observations']:
  state=sample['interaction'];target=sample['target'];now=sample['time']
  if sample['decision']%5!=1 or not state.get('active') or target is None or target['time_since_seen']>.05 or now<ts[0]:continue
  speed=float(np.interp(now,ts,truth[:,3]));speed_errors['raw_reference_rate'].append(abs(target['speed']-speed));speed_errors['converted_physical'].append(abs(state['physical_opponent_speed']-speed))
  for horizon in (.3,.6,.8):
   if now+horizon>ts[-1]:continue
   actual=np.array([np.interp(now+horizon,ts,truth[:,i]) for i in (0,1)])
   old=g.sample(target['s']+target['speed']*horizon,target['d']+target['lateral_speed']*.35*(1-np.exp(-horizon/.35)))[0]
   index=round(horizon/.05);pred=[np.array(x['path'][index]) for x in state['predictions']]
   errors[str(horizon)]['constant'].append(float(np.linalg.norm(old-actual)))
   errors[str(horizon)]['trend'].append(float(np.linalg.norm(pred[0]-actual)))
   errors[str(horizon)]['envelope'].append(min(float(np.linalg.norm(x-actual)) for x in pred)<=.1)
env.close()
forecast={h:dict(samples=len(d['constant']),constant_motion_mean_error_m=float(np.mean(d['constant'])) if d['constant'] else None,new_trend_mean_error_m=float(np.mean(d['trend'])) if d['trend'] else None,any_hypothesis_within_10cm_fraction=float(np.mean(d['envelope'])) if d['envelope'] else None) for h,d in errors.items()}
speed={k:dict(samples=len(v),mae_m_s=float(np.mean(v)) if v else None) for k,v in speed_errors.items()}
common=[r['new_seconds']-r['baseline_seconds'] for r in paired if r['baseline']=='pass' and r['new']=='pass']
analysis=dict(summary=summary,paired=paired,new_only_passes=sum(x['new']=='pass' and x['baseline']!='pass' for x in paired),baseline_only_passes=sum(x['baseline']=='pass' and x['new']!='pass' for x in paired),median_paired_time_change_s=statistics.median(common) if common else None,forecast=forecast,speed_conversion=speed)
(a.folder/'analysis.json').write_text(json.dumps(analysis,indent=2))
lines=['STAGE 4: OPPONENT PREDICTION AND FALLBACK',str(a.folder.resolve()),'',m['purpose'],'','Baseline is the previous JointStage4 controller. Both cars run in actual F1TENTH Gym on the same imported AutoDRIVE default track. The opponent uses its unchanged Stage 2 reactive controller, own LiDAR, speed cap and margins. Network checkpoints unchanged. API calls: zero.','']
for policy,row in summary.items():lines.append(policy+': '+json.dumps(row))
lines+=['','Paired outcomes (all cases):']
for row in paired:lines.append(json.dumps(row))
lines+=['','WHY THE PREVIOUS CONTROLLER FAILED','The target speed is measured as reference-coordinate progress ds/dt, not physical speed. At lateral offset d on curvature k, physical speed is approximately sqrt(((1-k*d)*ds/dt)^2+(dd/dt)^2). Using the progress rate as physical speed overestimates a car on the inside of a bend. The previous WAIT/follow behavior could brake too late or steer back toward a blocked reference. Two known timeout cases had both cars stopped indefinitely; another remained a rolling chase.','','CHANGES','Forecast physical speed and observed acceleration, then integrate back into track coordinates. Consider continuing the observed lateral trend, braking earlier, and returning toward the racing line. Do not invent motion for a car observed stopped. Check a bicycle rollout of ego motion, including steering-rate, acceleration and braking limits. Preserve a clear neural maneuver; otherwise compare lane holds, reference recovery, side shifts and reduced speeds. If none passes the check, brake in the current lane. Unknown and forward LiDAR obstacles retain their braking caps.','','CHECK PARAMETERS','The new local guard requires 0.06 m predicted body gap and normally 0.025 m conservative map-footprint gap over 0.8 s. Already-depleted body padding has a 0.025 m floor and must recover to 0.08 m by 1.2 s. Existing nominal path margins remain separate. These are explicit simulation parameters, not a formal safety guarantee. Longer predictions rank alternatives; they are not treated as certain opponent actions.','','PREDICTION ACCURACY','Measured from the same LiDAR estimate and future observed opponent trajectory, sampled periodically. Ground truth is used only here for scoring, never by the controller. The constant-motion comparator is the earlier geometric motion forecast, not the learned conditional MLP. Hypothesis coverage is descriptive, not a calibrated confidence bound.',json.dumps(forecast,indent=2),'Physical-speed conversion error:',json.dumps(speed,indent=2),'','LIMITS','Successful runs require a clear lead, return near the reference, and a two-second follow-through. Timeout is 40 s unless the manifest says otherwise. Zero contacts in a finite sample does not establish safety in unseen situations. Controller timing is measured during parallel experiments; p95 above 50 ms misses a 20 Hz wall-clock budget. Published controllers/models are retained; this controller is separately selectable. RViz best runs are selected demonstrations, not aggregate evidence.']
(a.folder/'REPORT.txt').write_text('\n'.join(lines)+'\n');print(json.dumps(analysis,indent=2))
