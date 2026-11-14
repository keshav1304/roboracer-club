"""Summarize paired Gym outcomes without dropping unsuccessful trials."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
manifest=json.loads((a.folder/'manifest.json').read_text());groups={'baseline':{},'response':{}}
for job in manifest['jobs']:
 d=json.loads((a.folder/(job['id']+'.json')).read_text());groups[job['policy']][job['id'].split('-')[1]]=d
paired=[]
for key,b in groups['baseline'].items():
 n=groups['response'][key];paired.append(dict(case=key,baseline=b['result']['outcome'],new=n['result']['outcome'],baseline_seconds=b['result']['passed_at'],new_seconds=n['result']['passed_at'],new_joint_selections=n['response_counts'].get('joint_selected',0)))
summary={}
for policy,items in groups.items():
 ds=list(items.values());results=[d['result'] for d in ds];passes=[r for r in results if r['success']]
 summary[policy]=dict(trials=len(ds),passes=len(passes),contacts=sum(r['collision'] is not None for r in results),median_successful_pass_seconds=statistics.median([r['passed_at'] for r in passes]) if passes else None,median_trial_p95_controller_ms=statistics.median(r['controller_latency']['p95_ms'] for r in results),maximum_trial_p95_controller_ms=max(r['controller_latency']['p95_ms'] for r in results),rescued_sides=sum(d['response_counts']['rescued_sides'] for d in ds),additional_feasible_profiles=sum(d['response_counts']['joint_feasible'] for d in ds),selected_joint_decisions=sum(d['response_counts'].get('joint_selected',0) for d in ds),strict_lateral_yield_passes=sum(d['yield_metrics']['lateral_yield_pass'] for d in ds))
common=[p['new_seconds']-p['baseline_seconds'] for p in paired if p['baseline']=='pass' and p['new']=='pass']
analysis=dict(summary=summary,paired=paired,new_only_passes=sum(p['new']=='pass' and p['baseline']!='pass' for p in paired),baseline_only_passes=sum(p['new']!='pass' and p['baseline']=='pass' for p in paired),median_paired_pass_time_change_seconds=statistics.median(common) if common else None)
(a.folder/'analysis.json').write_text(json.dumps(analysis,indent=2))
lines=['STAGE 4 JOINT PATH AND SPEED SEARCH',str(a.folder.resolve()),'', 'Frozen demonstrated Stage 4 versus joint-search variant. All opponents are unchanged reactive Stage 2 drivers. Actual F1TENTH Gym; no LLM/API calls or network retraining.','']
for name,s in summary.items():
 lines.append(f"{name}: {s['passes']}/{s['trials']} passes; {s['contacts']} contacts; median successful pass {s['median_successful_pass_seconds']} s; median per-trial controller p95 {s['median_trial_p95_controller_ms']:.2f} ms.")
lines += ['',f"New-only passes: {analysis['new_only_passes']}; baseline-only passes: {analysis['baseline_only_passes']}.",f"Median paired time change on common successes: {analysis['median_paired_pass_time_change_seconds']} s (negative is faster).",'', 'All trials (35/40 s timeout as recorded in manifest; successful runs continue for 2 s):']
for p in paired:lines.append(f"{p['case']}: {p['baseline']} ({p['baseline_seconds']}) -> {p['new']} ({p['new_seconds']}); new-path decisions {p['new_joint_selections']}")
lines += ['', 'METHOD', 'The existing candidates remain unchanged. Only blocked/stalled passing sides are searched further. New paths vary lane-change length, side clearance, merge location, and target speed. A quintic lateral transition matches the starting position and heading. Path curvature sets steering and lateral acceleration limits; a forward/backward speed envelope limits acceleration to 2 m/s^2 and braking to 3 m/s^2.', 'Reject wall/body overlap, unsteerable curvature, infeasible braking, and inadequate predicted passing progress. Check rectangular car bodies against measured-motion and map-limited opponent motion predictions. Replan from actual LiDAR/localization; existing maneuver/value networks, commitment and independent LiDAR brake remain active.', '', 'LIMITS', 'This is a planner improvement experiment; the neural-network weights were not retrained. Feasibility uses approximate prediction and does not prove collision avoidance. Zero observed contacts is not a safety guarantee. Timing was measured during paired worker execution; controller p95 above 50 ms misses the nominal 20 Hz decision budget even though simulation can run more slowly than wall time. Best RViz runs are selected demonstrations, not the overall success rate. Strict lateral-yield counts distinguish passing from proved opponent displacement. The track is the previously imported AutoDRIVE default map, not a newly verified IROS map.']
(a.folder/'REPORT.txt').write_text('\n'.join(lines)+'\n');print(json.dumps(analysis,indent=2))
