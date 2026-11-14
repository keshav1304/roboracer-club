"""Read-only audits of sealed manifests, final tests and neural decision influence.

No simulator, network requests or fitting runs here. `manifest` inspects only
the sampling plan. `test` requires completed, frozen final-test collections.
`decision` evaluates a trained network on recorded TRAIN/VALIDATION observations;
it is a decision comparison on those states, not a driving-performance ablation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from maneuver_core import ACTIONS, FEATURE_NAMES
from rollout_training import (eligible_samples, file_hash, load_collection,
                              sources,validate_manifest)
from tiny_network import TinyNetwork


def read_json(path):
    return json.loads(Path(path).read_text())


def distribution(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return dict(count=0,minimum=None,median=None,mean=None,maximum=None)
    return dict(count=len(values),minimum=min(values),median=float(np.median(values)),
                mean=float(np.mean(values)),maximum=max(values))


def wilson_interval(successes, trials):
    if not trials:
        return None
    z=1.959963984540054; p=successes/trials; divisor=1+z*z/trials
    center=(p+z*z/(2*trials))/divisor
    radius=z*np.sqrt(p*(1-p)/trials+z*z/(4*trials*trials))/divisor
    return [float(max(0.,center-radius)),float(min(1.,center+radius))]


def manifest_audit(manifest):
    validate_manifest(manifest)
    owners={}; sensor_owners={}; cases={}; stages={}
    for job in manifest['jobs']:
        owner=(job['stage'],job['split'],job['case_id'])
        if job['scenario_digest'] in owners and owners[job['scenario_digest']]!=owner:
            raise ValueError('Encounter reused across cases/partitions')
        owners[job['scenario_digest']]=owner
        for seed in (job['scenario']['seed'],job['scenario']['seed']+1):
            if seed in sensor_owners and sensor_owners[seed]!=owner:
                raise ValueError('Per-car sensor seed reused across encounters')
            sensor_owners[seed]=owner
        if owner in cases and cases[owner]['scenario_digest']!=job['scenario_digest']:
            raise ValueError('Paired variants do not share the same physical encounter')
        cases[owner]=job
    for stage in (2,3):
        stages[str(stage)]={}
        for split in ('train','validation','test'):
            jobs=[j for key,j in cases.items() if key[:2]==(stage,split)]
            cells=defaultdict(Counter)
            for job in jobs:
                cells[str(job['start_sector'])][str(job['scenario_group'])]+=1
            stages[str(stage)][split]=dict(unique_encounters=len(jobs),
                rollout_jobs=sum(j['stage']==stage and j['split']==split for j in manifest['jobs']),
                sector_by_stratum_counts={k:dict(v) for k,v in cells.items()},
                speed_cap_advantage_m_s=distribution(j['scenario']['ego_speed']-j['scenario']['opponent_speed'] for j in jobs),
                initial_gap_m=distribution(j['scenario']['gap'] for j in jobs),
                start_s_m=distribution(j['scenario']['s'] for j in jobs))
    return dict(manifest_digest=manifest['manifest_digest'],valid=True,
                unique_encounters=len(cases),unique_car_sensor_seeds=len(sensor_owners),stages=stages,
                outcome_data_read=False,
                limitations=['Variants share encounters and are correlated.',
                             'Partitions separate initial scenarios, sensor seeds and start sectors, not entire driving trajectories.',
                             'The physical sensor prelude may cross a start-sector boundary. Fit filters observation rows by their current sector.',
                             'All splits use the same known map; this does not measure unseen-track generalization.'])


def test_trial_metrics(record, frames):
    result,job=record['result'],record['job']
    warmup=float(result.get('warmup_seconds',0.)); end=float(result['sim_seconds'])
    times=np.asarray([f['t'] for f in frames],float)
    if len(times) and (not np.isfinite(times).all() or np.any(np.diff(times)<=0)):
        raise ValueError('Trace times must be finite and increasing')
    stop=stop_behind=covered=0.; first_attempt=None; raw_agreement=0; target_frames=0
    for i,frame in enumerate(frames):
        duration=max(0.,min(end,float(times[i+1]) if i+1<len(times) else end)-max(warmup,float(times[i])))
        covered+=duration
        if abs(frame['ego'][3])<.1:
            stop+=duration
            if frame['relative_progress']<0:
                stop_behind+=duration
        target=frame.get('target')
        if target is not None:
            target_frames+=1
            probabilities=frame.get('probabilities',[])
            if probabilities and ACTIONS[int(np.argmax(probabilities))]==frame['action']:
                raw_agreement+=1
        if (first_attempt is None and target is not None and frame['action'] in ('LEFT','RIGHT','PROBE')
                and frame['relative_progress']<0 and frame.get('command',[0.,0.])[1]>.1):
            first_attempt=float(frame['t'])
    physical_ticks=round(sum(result.get('action_seconds',{}).values())/float(result.get('physics_dt_seconds',.01)))
    walls=result.get('minimum_body_wall_clearances')
    wall=min(walls) if walls else result.get('minimum_wall_clearance')
    return dict(trial_id=job['trial_id'],case_id=job['case_id'],stage=job['stage'],
                initial_sector=job['start_sector'],scenario_group=job['scenario_group'],
                speed_cap_advantage_m_s=job['scenario']['ego_speed']-job['scenario']['opponent_speed'],
                gap_stratum=('approach' if job['scenario_group']==0 else 'close') if job['stage']==3 else 'varied',
                requested_initial_gap_m=job['scenario']['gap'],actual_policy_start_gap_m=result.get('actual_gap_at_policy_start'),
                success=bool(result['success']),collision=result.get('collision') is not None,
                passed_at_seconds=result.get('passed_at') if result['success'] else None,
                nominal_pass_followed_by_contact=bool(result.get('passed_at') is not None and result.get('collision') is not None),
                first_attempt_seconds=first_attempt,
                first_attempt_after_policy_start_seconds=None if first_attempt is None else first_attempt-warmup,
                estimated_stopped_policy_seconds=stop,estimated_stopped_behind_seconds=stop_behind,
                measured_policy_coverage_seconds=covered,policy_phase_seconds=max(0.,end-warmup),
                trace_median_spacing_seconds=float(np.median(np.diff(times))) if len(times)>1 else None,
                minimum_body_gap_m=result.get('minimum_body_gap'),minimum_either_car_wall_clearance_m=wall,
                raw_imitation_matches_final_action_target_frames=raw_agreement,target_present_frames=target_frames,
                raw_preference_changed_by_feasibility_or_value_ticks=result.get('safety_mask_override_ticks',0),
                commitment_override_ticks=result.get('commitment_override_ticks',0),physical_policy_ticks=physical_ticks)


def summarize_trials(rows):
    n=len(rows); passes=sum(r['success'] for r in rows); contacts=sum(r['collision'] for r in rows)
    ticks=sum(r['physical_policy_ticks'] for r in rows)
    target_frames=sum(r['target_present_frames'] for r in rows)
    return dict(trials=n,passes=passes,success_rate=passes/n if n else None,
                approximate_success_wilson_95=wilson_interval(passes,n),contacts=contacts,
                zero_contact_one_sided_95_upper_bound=(1-.05**(1/n)) if n and not contacts else None,
                successful_pass_seconds=distribution(r['passed_at_seconds'] for r in rows),
                first_attempt_seconds_on_attempted_trials=distribution(r['first_attempt_seconds'] for r in rows),
                trials_without_recorded_attempt=sum(r['first_attempt_seconds'] is None for r in rows),
                estimated_stopped_behind_seconds=distribution(r['estimated_stopped_behind_seconds'] for r in rows),
                minimum_body_gap_m=distribution(r['minimum_body_gap_m'] for r in rows),
                minimum_either_car_wall_clearance_m=distribution(r['minimum_either_car_wall_clearance_m'] for r in rows),
                contact_free_trials_below_body_gap_m={str(threshold):sum(not r['collision'] and r['minimum_body_gap_m'] is not None
                    and r['minimum_body_gap_m']<threshold for r in rows) for threshold in (.025,.05,.10)},
                contact_free_trials_below_wall_clearance_m={str(threshold):sum(not r['collision'] and r['minimum_either_car_wall_clearance_m'] is not None
                    and r['minimum_either_car_wall_clearance_m']<threshold for r in rows) for threshold in (.02,.04,.07)},
                nominal_passes_followed_by_contact=sum(r['nominal_pass_followed_by_contact'] for r in rows),
                raw_imitation_matches_final_action_when_target_present=(sum(r['raw_imitation_matches_final_action_target_frames'] for r in rows)/target_frames if target_frames else None),
                raw_preference_changed_by_feasibility_or_value_time_fraction=(sum(r['raw_preference_changed_by_feasibility_or_value_ticks'] for r in rows)/ticks if ticks else None),
                commitment_override_time_fraction=(sum(r['commitment_override_ticks'] for r in rows)/ticks if ticks else None))


def final_test_metrics(root,manifest):
    from v3_report import find_test_report,verify_collection
    stages={}
    for stage in (2,3):
        _,report=find_test_report(root,manifest,stage)
        verify_collection(report,manifest,stage,'test',manifest['test_cases_per_stage'])
        rows=[test_trial_metrics(r,read_json(r['trace'])['frames']) for r in report['trials']]
        groups={}
        for field in ('initial_sector','scenario_group','gap_stratum'):
            groups[field]={str(value):summarize_trials([r for r in rows if r[field]==value]) for value in sorted({r[field] for r in rows})}
        stages[str(stage)]=dict(aggregate=summarize_trials(rows),groups=groups,trials=rows)
    return dict(manifest_digest=manifest['manifest_digest'],stages=stages,
                definitions=dict(first_attempt='First recorded LEFT/RIGHT/PROBE decision while the target is present, ego remains behind, and commanded speed exceeds 0.1m/s. This is a chosen maneuver, not proof of actual lateral departure.',
                    stopped='Absolute measured ego speed below 0.1m/s, integrated by holding each recorded sample to the next. Warmup is excluded because replay frames begin afterward; sample spacing is about0.05s.',
                    times='Attempt/pass times include the scored physical prelude. Pass timing summaries contain successful trials only; failed or unattempted cases remain separately counted.',
                    near_contact='Descriptive thresholds applied to recorded100Hz minimum clearances; these do not replace the simulator contact definition or certify safety.',
                    confidence_intervals='Approximate per-encounter Wilson intervals, not frame-level intervals. Small strata and stratified sampling limit interpretation; repeated selection on validation is not an independent test.',
                    raw_preference_changes='The runtime field called safety_mask_override_ticks includes both feasibility masking and optional value-head preference changes. Commitment overrides are recorded separately.'),
                analysis_only=True,no_fitting_or_simulation=True)


def decision_audit(root,manifest,stage,split,model_path):
    if split not in ('train','validation'):
        raise ValueError('Decision comparisons must not fit or select on final tests')
    report=load_collection(root,stage,split,manifest)
    rows=[row for record in report['trials'] for row in eligible_samples(record,manifest,expected_split=split)]
    if not rows:raise ValueError('No eligible recorded observations')
    model_path=Path(model_path)
    model=TinyNetwork.load(model_path,FEATURE_NAMES)
    if model.metadata.get('reference_digest')!=manifest['reference_digest'] or model.metadata.get('stage')!=stage:
        raise ValueError('Decision audit requires a current-reference trained checkpoint')
    size=4 if stage==3 else 3
    if model.classes!=ACTIONS[:size]:
        raise ValueError('Maneuver classes differ from the driving controller')
    X=np.asarray([r['features'] for r in rows],np.float32)
    probabilities=model.predict_proba(X)
    feasible=np.asarray([[c['feasible'] for c in row['candidates'][:size]] for row in rows],bool)
    if not feasible.any(axis=1).all():raise ValueError('Recorded candidate set lacks a feasible fallback')
    costs=np.asarray([[c['estimated_time'] for c in row['candidates'][:size]] for row in rows],float)
    geometric=np.where(feasible,costs,np.inf).argmin(axis=1)
    masked=probabilities*feasible
    fallback=masked.sum(axis=1)<1e-8
    masked[fallback]=0.;masked[fallback,2]=1.
    masked/=masked.sum(axis=1)[:,None]
    imitation=masked.argmax(axis=1)
    selected=imitation.copy();value_path=model_path.with_name(model_path.stem+'.value.npz')
    if value_path.exists():
        value=TinyNetwork.load(value_path,FEATURE_NAMES+[f'action_{a.lower()}' for a in ACTIONS])
        if value.metadata.get('reference_digest')!=manifest['reference_digest'] or value.metadata.get('stage')!=stage:
            raise ValueError('Outcome checkpoint provenance differs')
        if value.classes!=('FAIL','PASS'):
            raise ValueError('Outcome classes differ from the driving controller')
        vx=np.concatenate((np.repeat(X,size,axis=0),np.tile(np.eye(4,dtype=np.float32)[:size],(len(X),1))),axis=1)
        chances=value.predict_proba(vx)[:,1].reshape(len(X),size)
        best=np.where(feasible,chances,-np.inf).max(axis=1)
        competitive=feasible & (chances>=best[:,None]-.025)
        selected=np.where(competitive,masked,-1.).argmax(axis=1)
    choices=feasible.sum(axis=1)>=2
    def fraction(mask,denominator=choices):
        return float(np.asarray(mask)[denominator].mean()) if denominator.any() else None
    controls_changed=[]
    for row,chosen,baseline in zip(rows,selected,geometric):
        chosen,baseline=row['candidates'][chosen],row['candidates'][baseline]
        controls_changed.append(abs(chosen['steering']-baseline['steering'])>.01 or abs(chosen['target_speed']-baseline['target_speed'])>.05)
    return dict(stage=stage,split=split,model=str(model_path.resolve()),model_sha256=file_hash(model_path),
                value_head_used=value_path.exists(),observation_rows=len(rows),multiple_feasible_action_rows=int(choices.sum()),
                forced_single_candidate_fraction=float((~choices).mean()),
                imitation_vs_geometric_disagreement_on_choice_rows=fraction(imitation!=geometric),
                value_changes_imitation_on_choice_rows=fraction(selected!=imitation),
                final_recommendation_vs_geometric_disagreement_on_choice_rows=fraction(selected!=geometric),
                different_candidate_control_hint_fraction_on_choice_rows=fraction(controls_changed),
                material_control_hint_thresholds=dict(steering_radians=.01,target_speed_m_s=.05),
                scope='Offline recommendations on recorded teacher-policy states, before commitment and LiDAR/rate limits. Correlated rows are not independent encounters. Differences show potential decision influence, not a causal improvement in driving or actual final control commands.',
                architecture='Networks rank feasible maneuver candidates. Geometry supplies trajectories and their steering/speed hints; commitment and independent safety filtering constrain execution. No network directly regresses final steering/throttle here.')


def validation_comparison(root,manifest,stage):
    """Compare actual learned and geometric-policy runs on the same val cases."""
    root=Path(root)
    teachers=load_collection(root,stage,'validation',manifest)
    selection=read_json(root/f'stage{stage}.selection.json')
    if not selection.get('validation_only') or not selection.get('qualified'):
        raise ValueError('An accepted validation-only checkpoint selection is required')
    chosen=selection['selected']
    learned=read_json(root/'trials'/f'stage{stage}'/'validation'/f'selection_{chosen}'/'summary.json')
    if (not learned.get('completed') or learned.get('policy')!='neural'
            or learned['manifest_digest']!=manifest['manifest_digest'] or learned['source_sha256']!=sources()
            or learned['model_sha256']!=selection['reports'][chosen]['model_sha256']):
        raise ValueError('Selected validation collection is incomplete or has changed provenance')
    for path,expected in teachers['model_artifacts'].items():
        if learned['model_artifacts'].get(path)!=expected:
            raise ValueError('Teacher and neural validation did not use the same opponent checkpoint')
    network={r['job']['case_id']:r for r in learned['trials']}
    if len(network)!=manifest['validation_cases_per_stage'] or len(network)!=len(learned['trials']):
        raise ValueError('Neural validation must have one rollout per independent encounter')
    comparisons={}
    for variant in sorted({r['job']['variant'] for r in teachers['trials']}):
        baseline={r['job']['case_id']:r for r in teachers['trials'] if r['job']['variant']==variant}
        if set(baseline)!=set(network):raise ValueError('Paired validation encounter membership differs')
        counts=Counter(); differences=[]
        for case_id,teacher in baseline.items():
            neural=network[case_id]
            if teacher['job']['scenario_digest']!=neural['job']['scenario_digest']:
                raise ValueError('Paired validation initial scenario differs')
            t,n=teacher['result'],neural['result']
            tp,np_=bool(t['success']),bool(n['success'])
            counts['both_pass' if tp and np_ else 'neural_only_pass' if np_ else 'teacher_only_pass' if tp else 'neither_pass']+=1
            counts['neural_passes']+=int(np_);counts['teacher_passes']+=int(tp)
            counts['neural_contacts']+=int(n['collision'] is not None)
            counts['teacher_contacts']+=int(t['collision'] is not None)
            if tp and np_:differences.append(n['passed_at']-t['passed_at'])
        comparisons[variant]={**{key:counts[key] for key in ('both_pass','neural_only_pass','teacher_only_pass','neither_pass',
                              'neural_passes','teacher_passes','neural_contacts','teacher_contacts')},
                              'paired_encounters':len(network),'neural_minus_teacher_pass_seconds_on_joint_successes':distribution(differences)}
    return dict(stage=stage,selected_checkpoint=chosen,comparisons=comparisons,
                scope='Actual full-rollout policy comparisons on the same initial validation encounters and sensor seeds. These cases also selected the checkpoint, so this comparison is development evidence, not an independent test of superiority. Geometric comparators include their named preference bias; Stage3 has no unbiased neutral comparator in the three-variant curriculum.',
                no_final_test_data_used=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('manifest');sub.add_parser('test')
    comparison=sub.add_parser('compare-validation')
    comparison.add_argument('--stage',type=int,choices=(2,3),required=True)
    decision=sub.add_parser('decision')
    decision.add_argument('--stage',type=int,choices=(2,3),required=True)
    decision.add_argument('--split',choices=('train','validation'),default='validation')
    decision.add_argument('--model',type=Path,required=True)
    args=parser.parse_args();manifest=read_json(args.root/'manifest.json');validate_manifest(manifest)
    if args.command=='manifest':result=manifest_audit(manifest)
    elif args.command=='test':result=final_test_metrics(args.root,manifest)
    elif args.command=='compare-validation':result=validation_comparison(args.root,manifest,args.stage)
    else:result=decision_audit(args.root,manifest,args.stage,args.split,args.model)
    output=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(output)
    print(output,end='')


if __name__=='__main__':main()
