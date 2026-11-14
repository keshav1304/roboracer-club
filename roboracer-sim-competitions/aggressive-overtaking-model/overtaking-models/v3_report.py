"""Publish a small representative gallery from a completed, sealed rollout round.

No simulator or policy is changed here. Every new gallery recording must be a
frozen-model TEST trial. Training/validation remain aggregate summaries, with
all full trajectories retained in the round directory for audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np

from bootstrap import ARTIFACTS, MODELS
from rollout_training import validate_manifest, sources

FINAL_ROUND_ROOT = ARTIFACTS / 'rollout_final_20261107'


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup_description(coordinated_stage3):
    if not coordinated_stage3:
        return "The Stage 3 scored setup prelude follows the shared reference using each car's own tracker."
    return ('In Stage 3 only, during the scored setup prelude, both cars start at one common speed limited by the shared reference envelope over both positions and their 1.2 m previews. Each uses its own steering, with a shared lower speed command. After handoff each racing controller uses only its own actual observation history; there is no shared command or opponent truth input. Stage 2 retains its original physical setup.')


def llm_accounting(stages):
    """Separate new requests from advice embedded in a preserved checkpoint."""
    ledgers=[stage['llm_training'] for stage in stages.values()]
    return {
        'new_api_calls':sum(r['new_api_requests_this_round'] for r in ledgers),
        'fresh_llm_advice_rows_consumed':sum(r['new_advice_rows_this_round'] for r in ledgers),
        'imported_llm_advice_rows_consumed':sum(r['imported_advice_rows'] for r in ledgers),
        'total_llm_advice_rows_consumed':sum(r['advice_consumed']['rows_consumed'] for r in ledgers),
        'estimated_api_usage_cost_usd':sum(r['new_estimated_usage_cost_usd'] for r in ledgers),
        'imported_api_usage_cost_usd':sum(r['imported_estimated_usage_cost_usd'] for r in ledgers)}


def verified_serial_benchmark(path,manifest):
    """Optional timing evidence must describe the exact deployed runtime."""
    path=Path(path)
    if not path.exists():
        return None
    try:
        benchmark=read_json(path)
        if (benchmark['source_sha256']!=sources()
                or benchmark['reference_digest']!=manifest['reference_digest']
                or benchmark['benchmark_sha256']!=file_hash(Path(__file__).with_name('benchmark_v3.py'))
                or benchmark['network_access']!='blocked'
                or benchmark['control_period_ms']!=1000*manifest['control_dt']
                or benchmark['physics_period_ms']!=10
                or len(benchmark['scenes'])!=2):
            return None
        expected_count=sum((scene['samples']+4)//5-(scene['warmup_samples']+4)//5
                           for scene in benchmark['scenes'])
        for stage in ('2','3'):
            result=benchmark['results'][stage]
            model=MODELS/f'stage{stage}.npz';value=model.with_name(model.stem+'.value.npz')
            if (result['model_sha256']!=file_hash(model)
                    or result['value_model_sha256']!=(file_hash(value) if value.exists() else None)):
                return None
            for name in ('deployed_network','full_controller'):
                timing=result[name]
                if (timing['count']!=expected_count or expected_count<=0
                        or not np.isfinite(timing['p95_ms']) or timing['p95_ms']<0):
                    return None
        return {**benchmark,'artifact':str(path.resolve()),'artifact_sha256':file_hash(path),
                'interpretation':'Serial warmed computation on two fixed recorded scenes, not an overtaking success test or a physical reaction-time guarantee. Deployed-network time includes the maneuver network and any deployed value head; full-controller time includes perception, planning and safety checks but excludes physics, rendering and model loading. Decisions are scheduled every 50 ms; scheduling can add up to 50 ms beyond computation.'}
    except (OSError,ValueError,KeyError,TypeError):
        return None


def verified_stage2_baseline(root,manifest,current):
    """Recompute the optional fixed comparison from matching saved test records."""
    directory=Path(root)/'baseline_stage2_evaluation';path=directory/'comparison.json'
    if not path.exists():
        return None
    try:
        comparison=read_json(path);plan=read_json(directory/'predeclared_manifest.json')
        verify_collection(current,manifest,2,'test',30)
        if (comparison['provenance']!=plan or plan['harness_sha256']!=sources()['run_experiment.py']
                or plan['wrapper_sha256']!=file_hash(directory/'evaluation_runner.py')
                or plan['control_dt_seconds']!=manifest['control_dt']):
            return None
        artifacts=plan['baseline_artifacts']
        if len(artifacts)!=3 or any(file_hash(p)!=h for p,h in artifacts.items()):
            return None
        core=next(p for p in artifacts if Path(p).name=='maneuver_core.py')
        model=next(p for p in artifacts if Path(p).name=='stage2.npz')
        value=next(p for p in artifacts if Path(p).name=='stage2.value.npz')
        expected_baseline_sources={**sources(),'maneuver_core.py':artifacts[core]}
        expected_jobs={j['trial_id']:j for j in manifest['jobs'] if j['stage']==2 and j['split']=='test'}
        final_records={r['job']['trial_id']:r for r in current['trials']}
        items={r['trial_id']:r for r in plan['items']}
        pairs=comparison['paired_outcomes']
        if (len(plan['items'])!=30 or len(items)!=30 or len(pairs)!=30 or set(items)!=set(expected_jobs)
                or {r['trial_id'] for r in pairs}!=set(expected_jobs)):
            return None
        fields=('outcome','success','passed_at','collision','minimum_body_gap',
                'minimum_wall_clearance','pass_followthrough_complete')
        baseline_runs={};final_runs={}
        for pair in pairs:
            key=pair['trial_id'];item=items[key];record=read_json(pair['baseline_record'])
            old=record['result'];new=final_records[key]['result'];p=old['baseline_provenance']
            if (item['job']!=expected_jobs[key] or record['original_job']!=expected_jobs[key]
                    or item['scenario']!=new['scenario'] or item['scenario_digest']!=expected_jobs[key]['scenario_digest']
                    or not record['completed'] or read_json(record['paired_final_record'])!=final_records[key]
                    or record['paired_final_record']!=item['record_path']
                    or record['trace']!=pair['baseline_trace'] or read_json(record['trace'])['result']!=old
                    or {k:old[k] for k in fields}!=pair['baseline']
                    or {k:new[k] for k in fields}!=pair['final'] or pair['final']!=item['final_result']):
                return None
            if (old['source_sha256']!=expected_baseline_sources or new['source_sha256']!=sources()
                    or old['model_sha256']!=artifacts[model] or old['value_model_sha256']!=artifacts[value]
                    or new['model_sha256']!=file_hash(MODELS/'stage2.npz')
                    or new['value_model_sha256']!=file_hash(MODELS/'stage2.value.npz')
                    or p['controller_core_path']!=core or p['controller_core_sha256']!=artifacts[core]
                    or p['model_path']!=model or p['model_sha256']!=artifacts[model]
                    or p['value_model_path']!=value or p['value_model_sha256']!=artifacts[value]
                    or p['episode_harness_sha256']!=sources()['run_experiment.py']
                    or p['wrapper_sha256']!=plan['wrapper_sha256']
                    or p['harness_imported_current_core_sha256']!=sources()['maneuver_core.py']
                    or p['shared_sources']!={k:v for k,v in sources().items() if k not in ('run_experiment.py','maneuver_core.py')}
                    or p['reference_digest']!=manifest['reference_digest']):
                return None
            if (old['teacher'] or old['llm_calls_in_test']!=0 or old.get('used_for_tuning') is not False
                    or old.get('post_freeze_evaluation') is not True or old.get('held_out_test') is not True
                    or old.get('evaluation_role')!='predeclared_archived_baseline'
                    or old.get('paired_final_trial_id')!=key
                    or old['scenario_digest']!=expected_jobs[key]['scenario_digest']
                    or any(old[k]!=new[k] for k in ('scenario','sensor_seeds','control_dt_seconds',
                           'physics_dt_seconds','warmup_seconds','reference_sha256'))
                    or bool(old['success'])!=(old['outcome']=='pass')
                    or (old['success'] and (old['collision'] is not None or not old['pass_followthrough_complete']))):
                return None
            baseline_runs[key]=old;final_runs[key]=new
        summaries={}
        for name,runs in (('baseline',baseline_runs),('final',final_runs)):
            times=[r['passed_at'] for r in runs.values() if r['success']]
            summary={'runs':len(runs),'passes':len(times),'contacts':sum(r['collision'] is not None for r in runs.values()),
                     'median_successful_pass_seconds':float(np.median(times)) if times else None}
            if any(comparison[name][k]!=v for k,v in summary.items()):
                return None
            summaries[name]=summary
        old_pass={k for k,r in baseline_runs.items() if r['success']};new_pass={k for k,r in final_runs.items() if r['success']}
        if (set(comparison['new_only_passes'])!=new_pass-old_pass
                or set(comparison['baseline_only_passes'])!=old_pass-new_pass):
            return None
        savings={k:baseline_runs[k]['passed_at']-final_runs[k]['passed_at'] for k in sorted(old_pass&new_pass)}
        mean_saving=float(np.mean(list(savings.values()))) if savings else None
        paired=comparison['paired_success_time_change']
        if (paired['same_scenarios_passed']!=len(savings) or paired['individual_seconds_saved']!=savings
                or paired['mean_seconds_saved']!=mean_saving
                or paired['all_common_passes_faster_in_final']!=bool(savings and all(v>0 for v in savings.values()))):
            return None
        return {**summaries,'artifact':str(path.resolve()),'artifact_sha256':file_hash(path),
                'baseline_artifacts':artifacts,'current_source_sha256':sources(),
                'current_model_sha256':file_hash(MODELS/'stage2.npz'),
                'current_value_model_sha256':file_hash(MODELS/'stage2.value.npz'),
                'same_successful_scenarios':old_pass==new_pass,
                'new_only_passes':sorted(new_pass-old_pass),'baseline_only_passes':sorted(old_pass-new_pass),
                'common_successes':len(savings),'common_successes_faster':sum(v>0 for v in savings.values()),
                'mean_paired_seconds_saved':mean_saving,
                'interpretation':('No unseen success-rate improvement demonstrated on these identical 30 scenarios. '
                    if len(new_pass)<=len(old_pass) else 'More passes observed in this fixed comparison. ')
                    +'This compares the previous Stage 2 controller and model with the final controller and model. Timing changes cannot be attributed solely to neural learning because planner settings and safety behavior also changed. No tuning or model selection follows this comparison.'}
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,StopIteration):
        return None


def llm_provenance(root, manifest, stage, fit, *, imported=False):
    """Report the exact training advice consumed, separately from rollout truth."""
    advice=fit.get('fresh_llm_advice',{'rows_consumed':0})
    consumed=int(advice['rows_consumed'])
    if not 0<=consumed<=10:
        raise RuntimeError('Fresh training advice exceeds its per-stage budget')
    if any(advice.get(k,0) for k in ('value_labels_modified','validation_rows_used','test_rows_used')):
        raise RuntimeError('Advisory reviews must not relabel outcomes or use held-out data')
    logpaths={ARTIFACTS/'llm_reviews_20261106.jsonl'}
    planned_requests=set()
    questions_path=root/'reviews'/f'stage{stage}.questions.json'
    if questions_path.exists():
        planned_requests={json.dumps(q['request'],sort_keys=True,separators=(',',':'))
                          for q in read_json(questions_path)}
    if consumed:
        path=Path(advice['file'])
        if file_hash(path)!=advice['sha256']:
            raise RuntimeError('Consumed advice file changed after fitting')
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows)!=consumed or any(r.get('stage')!=stage or r.get('split')!='train'
                or r.get('manifest_digest')!=manifest['manifest_digest']
                or r.get('reference_digest')!=manifest['reference_digest']
                or r.get('source_sha256')!=fit['metadata']['source_sha256'] for r in rows):
            raise RuntimeError('Consumed advice has mixed or incomplete training provenance')
        if not planned_requests or any(json.dumps(r['request'],sort_keys=True,separators=(',',':'))
                                       not in planned_requests for r in rows):
            raise RuntimeError('Consumed advice does not match the preserved training-review questions')
        logpaths.update(Path(r['cache_path']) for r in rows)
    attempts={}
    for path in logpaths:
        if not path.exists():continue
        for line in path.read_text().splitlines():
            if not line.strip():continue
            record=json.loads(line);scenario=record.get('scenario',{})
            if (scenario.get('review_type')=='closed_loop_training_v1'
                    and scenario.get('manifest_digest')==manifest['manifest_digest']
                    and scenario.get('stage')==stage
                    and json.dumps(scenario,sort_keys=True,separators=(',',':')) in planned_requests):
                attempts[(record['cache_key'],record.get('wall_timestamp'))]=record
    records=list(attempts.values())
    review_path=root/'reviews'/f'stage{stage}.review_report.json'
    review=read_json(review_path) if review_path.exists() else None
    if consumed and (not records or consumed>sum(r.get('status')=='accepted' for r in records)):
        raise RuntimeError('Consumed fresh advice lacks accepted API accounting records')
    cost=sum(float(r.get('estimated_usage_cost_usd',0)) for r in records)
    return {'advice_consumed':advice,'review_report':review,'api_requests_recorded':len(records),
            'imported_from_predecessor':bool(imported),
            'new_api_requests_this_round':0 if imported else len(records),
            'new_advice_rows_this_round':0 if imported else consumed,
            'imported_advice_rows':consumed if imported else 0,
            'new_estimated_usage_cost_usd':0. if imported else cost,
            'imported_estimated_usage_cost_usd':cost if imported else 0.,
            'accepted_reviews_recorded':sum(r.get('status')=='accepted' for r in records),
            'estimated_usage_cost_usd':cost,
            'reserved_cost_usd':sum(float(r.get('reserved_cost_usd',0)) for r in records),
            'billing_note':'Usage-based estimate from the request ledger; promotional credits and final billing are not inferred.'}


def metrics(records):
    runs=[r['result'] for r in records]
    times=[float(r['passed_at']) for r in runs if r.get('success') and r.get('passed_at') is not None]
    outcomes=Counter(r['outcome'] for r in runs)
    return {'trials':len(runs),'passes':outcomes['pass'],'no_pass':outcomes['no_pass'],
            'collisions':sum(r.get('collision') is not None for r in runs),
            'success_rate':outcomes['pass']/len(runs) if runs else 0.,
            'median_success_time_s':float(np.median(times)) if times else None,
            'mean_success_time_s':float(np.mean(times)) if times else None,
            'earliest_success_time_s':min(times) if times else None,
            'latest_success_time_s':max(times) if times else None,
            'time_statistics_scope':'Completed passes only; unsuccessful trials remain failures and do not get a zero pass time'}


def passing_status(stage, validation, test):
    """Observed driving outcomes decide the status, never label accuracy."""
    if test['passes']==0:
        return {'status':'not_demonstrated','passing_objective_met':False,
                'headline':f'Stage {stage} passing objective not met',
                'statement':'Overtaking not demonstrated in this evaluation.',
                'detail':f"The selected neural policy completed 0/{test['trials']} test passes and {validation['passes']}/{validation['trials']} validation passes."}
    if validation['passes']==0:
        return {'status':'not_demonstrated_in_validation','passing_objective_met':False,
                'headline':f'Stage {stage} passing objective not met',
                'statement':'Overtaking was not demonstrated during validation.',
                'detail':f"The selected neural policy later completed {test['passes']}/{test['trials']} test passes, but its validation result was 0/{validation['trials']}. A zero-contact selection alone is not successful overtaking validation."}
    return {'status':'passes_observed','passing_objective_met':None,
            'headline':f"Stage {stage}: {test['passes']}/{test['trials']} test passes observed",
            'statement':f"The selected neural policy completed {validation['passes']}/{validation['trials']} validation passes and recorded {test['collisions']} test contacts.",
            'detail':'These observed counts describe this evaluation; they do not establish reliable passing across all encounters.'}


def recovery_usage(records):
    """Activation counts are usage evidence, never an observed recovery deadline."""
    runs=[r['result'] for r in records]
    required=('padding_recovery_decisions','padding_recovery_nominal_seconds',
              'padding_lidar_release_decisions','padding_lidar_released_beams')
    reported=[r for r in runs if all(key in r for key in required)]
    if len(reported)!=len(runs):
        raise RuntimeError('Padding-recovery telemetry is incomplete; do not report missing usage as zero')
    used=[r for r in reported if r['padding_recovery_decisions']>0]
    return {'reported_trials':len(reported),'total_trials':len(runs),
            'trials_using_guard':len(used),'passes_using_guard':sum(bool(r['success']) for r in used),
            'contacts_using_guard':sum(r.get('collision') is not None for r in used),
            'certified_decisions':sum(r['padding_recovery_decisions'] for r in reported),
            'nominal_active_control_seconds':sum(r.get('padding_recovery_nominal_seconds',0.) for r in reported),
            'lidar_release_decisions':sum(r.get('padding_lidar_release_decisions',0) for r in reported),
            'lidar_padding_beams_released':sum(r.get('padding_lidar_released_beams',0) for r in reported),
            'duration_semantics':'Sum of active control intervals across trials; not time to restore padding or a physical timing guarantee.'}


def recorded_profile(records):
    profiles={json.dumps(r['result']['ego_planning_profile'],sort_keys=True) for r in records}
    rival_margins={json.dumps(r['result'].get('opponent_safety_margins'),sort_keys=True) for r in records}
    rival_profiles={json.dumps(r['result'].get('opponent_planning_profile'),sort_keys=True) for r in records}
    if len(profiles)!=1:
        raise RuntimeError('Test runs do not share one recorded planning profile')
    if len(rival_margins)!=1 or len(rival_profiles)!=1:
        raise RuntimeError('Test runs do not share one rival clearance policy')
    runs=[r['result'] for r in records]
    return {'ego':json.loads(profiles.pop()),
            'rival_safety_margins':json.loads(rival_margins.pop()),
            'rival_planning_profile':json.loads(rival_profiles.pop()),
            'ego_top_speed_caps_m_s':sorted({r['scenario']['ego_speed'] for r in runs}),
            'rival_top_speed_caps_m_s':sorted({r['scenario']['opponent_speed'] for r in runs}),
            'equal_top_speed_caps_in_every_trial':all(r['scenario']['ego_speed']==r['scenario']['opponent_speed'] for r in runs)}


def opponent_recovery_usage(records):
    runs=[r['result'] for r in records]
    if any('opponent_padding_recovery_decisions' not in r
           or 'opponent_padding_lidar_release_decisions' not in r for r in runs):
        raise RuntimeError('Opponent recovery telemetry is incomplete')
    return {'trials':len(runs),
            'trials_using_guard':sum(r['opponent_padding_recovery_decisions']>0 for r in runs),
            'certified_decisions':sum(r['opponent_padding_recovery_decisions'] for r in runs),
            'lidar_release_decisions':sum(r['opponent_padding_lidar_release_decisions'] for r in runs)}


def representative_records(records, limit=4):
    """Include failures/contacts when present; never imply this gallery is a rate."""
    if not 1<=limit<=4:
        raise ValueError('Representative gallery limit must be between one and four')
    selected=[]
    def add(record,label):
        if len(selected)<limit and all(r['job']['trial_id']!=record['job']['trial_id'] for r,_ in selected):
            selected.append((record,label))
    passes=sorted((r for r in records if r['result'].get('success')),key=lambda r:r['result']['passed_at'])
    collisions=[r for r in records if r['result'].get('collision') is not None]
    failures=[r for r in records if not r['result'].get('success') and r['result'].get('collision') is None]
    if passes:add(passes[0],'Earliest completed pass')
    if collisions:add(collisions[0],'Contact — unsuccessful test')
    if failures:add(failures[len(failures)//2],'Unfinished pass — unsuccessful test')
    if passes:add(passes[len(passes)//2],'Typical completed pass')
    # Prefer a different starting sector when filling remaining gallery slots.
    while len(selected)<min(limit,len(records)):
        used={r['job']['trial_id'] for r,_ in selected}
        sectors={r['job']['start_sector'] for r,_ in selected}
        remaining=[r for r in records if r['job']['trial_id'] not in used]
        different=[r for r in remaining if r['job']['start_sector'] not in sectors]
        candidate=(different or remaining)[len(different or remaining)//2]
        add(candidate,'Additional test location')
    return selected


def verify_collection(report,manifest,stage,split,expected_count,*,predecessor_sources=None):
    if predecessor_sources is not None and (stage!=2 or split not in ('train','validation')):
        raise RuntimeError('Predecessor sources are allowed only for stage 2 training and teacher validation')
    if not report.get('completed') or report.get('completed_trials')!=expected_count or len(report.get('trials',[]))!=expected_count:
        raise RuntimeError(f'Stage {stage} {split} collection is incomplete')
    if report['manifest_digest']!=manifest['manifest_digest'] or report['reference_digest']!=manifest['reference_digest']:
        raise RuntimeError('Collection belongs to a different sealed round/reference')
    if report['source_sha256']!=(sources() if predecessor_sources is None else predecessor_sources):
        raise RuntimeError('Controller source changed after the recorded collection')
    expected_jobs={j['trial_id']:j for j in manifest['jobs'] if j['stage']==stage and j['split']==split}
    expected=set(expected_jobs)
    observed={r['job']['trial_id'] for r in report['trials']}
    if expected!=observed:
        raise RuntimeError(f'Stage {stage} {split} scenario membership mismatch')
    for record in report['trials']:
        if (record['job']['split']!=split or record['job']['stage']!=stage
                or record['manifest_digest']!=manifest['manifest_digest']
                or record['reference_digest']!=manifest['reference_digest']):
            raise RuntimeError('Mixed trial provenance')
        if record['source_sha256']!=report['source_sha256']:
            raise RuntimeError('Mixed controller versions in one collection')
        if record['job']!=expected_jobs[record['job']['trial_id']]:
            raise RuntimeError('Recorded scenario differs from its sealed manifest')
        if split=='test':
            result=record['result']
            if read_json(record['trace'])['result']!=result:
                raise RuntimeError('Replay result differs from the counted test outcome')
            for path,expected_hash in record['model_artifacts'].items():
                observed_hash=file_hash(path) if Path(path).exists() else None
                if observed_hash!=expected_hash:
                    raise RuntimeError('Ego/rival network or value head changed after testing')
            if bool(result.get('success'))!=(result.get('outcome')=='pass'):
                raise RuntimeError('Success flag and recorded outcome disagree')
            if result.get('teacher') or result.get('network_access')!='blocked':
                raise RuntimeError('Gallery/test results must be offline learned-policy evaluation')
            if result.get('model_sha256')!=file_hash(MODELS/f'stage{stage}.npz'):
                raise RuntimeError('Test checkpoint is not the deployed frozen model')
            if stage==3 and result.get('opponent_model_sha256')!=file_hash(MODELS/'stage2.npz'):
                raise RuntimeError('Stage 3 test rival is not the deployed stage 2 checkpoint')
            if result.get('success') and (result.get('collision') or not result.get('pass_followthrough_complete')):
                raise RuntimeError('Incomplete/contact pass cannot count as success')


def find_test_report(root,manifest,stage,tag=None):
    base=root/'trials'/f'stage{stage}'/'test'
    paths=[base/tag/'summary.json'] if tag else sorted(base.glob('*/summary.json'))
    matches=[]
    for path in paths:
        if not path.exists():continue
        report=read_json(path)
        if report.get('manifest_digest')==manifest['manifest_digest'] and report.get('policy')=='neural':
            matches.append((path,report))
    if len(matches)!=1:
        raise RuntimeError(f'Expected one test report for stage {stage}; use --test-tag to select explicitly')
    return matches[0]


def verify_selected_checkpoint(selection,test,manifest,*,predecessor=None):
    selected=selection['reports'][selection['selected']]
    if predecessor is not None:
        if test.get('stage')!=2 or selection['selected']!=predecessor['selected_checkpoint']:
            raise RuntimeError('Predecessor compatibility cannot change the stage 2 selection')
        original=selected
        old_manifest=predecessor['predecessor_manifest']
        if (original.get('manifest_digest')!=old_manifest['manifest_digest']
                or original.get('reference_digest')!=old_manifest['reference_digest']):
            raise RuntimeError('Preserved selection belongs to another predecessor manifest')
        selected=predecessor['current_validation_reports'][selection['selected']]
        if (original.get('model_sha256')!=selected.get('model_sha256')
                or selected.get('model_sha256')!=predecessor['selected_model_sha256']):
            raise RuntimeError('Compatible validation changed the frozen stage 2 model')
    if (not selected.get('completed') or selected.get('policy')!='neural'
            or selected.get('split')!='validation'
            or selected.get('completed_trials')!=manifest['validation_cases_per_stage']
            or selected.get('manifest_digest')!=manifest['manifest_digest']
            or selected.get('reference_digest')!=manifest['reference_digest']
            or selected.get('source_sha256')!=test['source_sha256']):
        raise RuntimeError('Selected checkpoint lacks complete matching validation provenance')
    if selected.get('model_sha256')!=test['model_sha256']:
        raise RuntimeError('Test maneuver network is not the validation-selected checkpoint')
    source=Path(selected['model_file'])
    value_key=str(source.with_name(source.stem+'.value.npz').resolve())
    if value_key not in selected['model_artifacts']:
        raise RuntimeError('Selected checkpoint value-head provenance is missing')
    expected_value=selected['model_artifacts'][value_key]
    if predecessor is not None and expected_value!=predecessor['selected_value_sha256']:
        raise RuntimeError('Compatible validation changed the frozen stage 2 value head')
    if ({r['result'].get('value_model_sha256') for r in test['trials']}!={expected_value}
            or bool(selection.get('deployed_value_head'))!=(expected_value is not None)):
        raise RuntimeError('Test value head is not the validation-selected value head')
    return selected


def assemble(root,test_tag=None):
    root=Path(root)
    manifest=read_json(root/'manifest.json');validate_manifest(manifest)
    stages={};gallery=[];predecessor=None;test_collections={}
    if (root/'stage2_predecessor'/'COMPATIBILITY.json').exists():
        from stage2_predecessor import verify_stage2_predecessor
        predecessor=verify_stage2_predecessor(root,manifest)
    for stage in (2,3):
        compatible=predecessor if stage==2 else None
        training_root=Path(compatible['imported_root']) if compatible is not None else root
        training_manifest=compatible['predecessor_manifest'] if compatible is not None else manifest
        train=read_json(training_root/'trials'/f'stage{stage}'/'train'/'teacher'/'summary.json')
        validation=read_json(training_root/'trials'/f'stage{stage}'/'validation'/'teacher'/'summary.json')
        predecessor_sources=train['source_sha256'] if compatible is not None else None
        test_path,test=find_test_report(root,manifest,stage,test_tag)
        test_collections[stage]=test
        verify_collection(train,training_manifest,stage,'train',training_manifest['training_rollouts_per_stage'],
                          predecessor_sources=predecessor_sources)
        verify_collection(validation,training_manifest,stage,'validation',training_manifest['validation_cases_per_stage']*3,
                          predecessor_sources=predecessor_sources)
        verify_collection(test,manifest,stage,'test',manifest['test_cases_per_stage'])
        if train['completed_trials']<100:
            raise RuntimeError('At least 100 completed training rollouts per stage are required before publication')
        fit=read_json(training_root/f'stage{stage}.training.json')
        if (fit['metadata']['manifest_digest']!=training_manifest['manifest_digest']
                or fit['metadata']['reference_digest']!=training_manifest['reference_digest']
                or fit['metadata']['source_sha256']!=(sources() if compatible is None else predecessor_sources)):
            raise RuntimeError('Training metadata belongs to a different reference/controller round')
        selection=read_json(training_root/f'stage{stage}.selection.json')
        if not selection.get('qualified') or not selection.get('validation_only'):
            raise RuntimeError('Checkpoint lacks a valid validation-only selection record')
        selected_validation=verify_selected_checkpoint(selection,test,manifest,predecessor=compatible)
        validation_reports=compatible['current_validation_reports'] if compatible is not None else selection['reports']
        setup_modes={bool(r['result'].get('coordinated_stage3_warmup',False)) for r in test['trials']}
        if len(setup_modes)!=1 or (stage==2 and True in setup_modes):
            raise RuntimeError('Mixed or stage-inappropriate setup protocol in tested data')
        representatives=representative_records(test['trials'])
        gallery.extend(representatives)
        winning_encounters={r['job']['case_id'] for r in train['trials']
                            if r['result'].get('success') and r['result'].get('collision') is None}
        if len(winning_encounters)!=fit['successful_winning_encounters']:
            raise RuntimeError('Training winning-encounter count disagrees with the recorded outcomes')
        test_metrics=metrics(test['trials'])
        stages[str(stage)]={'training':metrics(train['trials']),
                           'training_unique_encounters':train['unique_encounters'],
                           'training_unique_winning_encounters':len(winning_encounters),
                           'teacher_validation':metrics(validation['trials']),
                           'selected_neural_validation':selected_validation['outcomes'],
                           'sealed_test':test_metrics,
                           'passing_status':passing_status(stage,selected_validation['outcomes'],test_metrics),
                           'planning_profile':recorded_profile(test['trials']),
                           'padding_recovery':{'training':recovery_usage(train['trials']),
                                               'teacher_validation':recovery_usage(validation['trials']),
                                               'sealed_test':recovery_usage(test['trials'])},
                           'opponent_padding_recovery':opponent_recovery_usage(test['trials']),
                           'test_file':str(test_path.resolve()),
                           'selected_checkpoint':selection['selected'],
                           'value_head_deployed':bool(selection.get('deployed_value_head')),
                           'checkpoint_validation':{name:r['outcomes'] for name,r in validation_reports.items()},
                           'training_rows':{'imitation':fit['imitation_rows'],'value':fit['value_rows'],
                                            'successful_rollout_imitation':fit.get('successful_rollout_imitation_rows',fit['imitation_rows']),
                                            'pass':fit['value_pass_rows'],'failure':fit['value_failure_rows']},
                           'llm_training':llm_provenance(training_root,training_manifest,stage,fit,imported=compatible is not None),
                           'value_validation_calibration':fit.get('value_validation_calibration'),
                           'value_calibration_interpretation':('No positive validation frames were available; accuracy/calibration cannot establish the ability to recognize or complete passes.'
                               if fit.get('value_validation_calibration') is not None and fit['value_validation_calibration'].get('pass_samples')==0
                               else 'Calibration describes correlated continuation labels, not completed neural overtaking success.'),
                           'selected_gallery':[{'trial_id':r['job']['trial_id'],'label':label,'outcome':r['result']['outcome']} for r,label in representatives],
                           'model_sha256':test['model_sha256'],'source_sha256':test['source_sha256'],
                           'coordinated_stage3_warmup':next(iter(setup_modes)),
                           'data_provenance':{
                               'training_imported_unchanged':compatible is not None,
                               'training_directory':str(training_root.resolve()),
                               'original_training_manifest_digest':training_manifest['manifest_digest'],
                               'original_training_source_sha256':train['source_sha256'],
                               'current_evaluation_manifest_digest':manifest['manifest_digest'],
                               'current_evaluation_source_sha256':test['source_sha256'],
                               'selection_rerun':False if compatible is not None else None,
                               'compatibility_certificate':str(compatible['certificate_path']) if compatible is not None else None,
                               'compatibility_certificate_sha256':file_hash(compatible['certificate_path']) if compatible is not None else None,
                               'current_runtime_validation_trials':sum(r['completed_trials'] for r in validation_reports.values()),
                               'validation_semantics':('Preserved selection; all four checkpoints replayed under current runtime and verified identical. No reselection.'
                                                       if compatible is not None else 'Current-round validation-only selection.')}}
    reference=read_json(ARTIFACTS/'reference_validation_v3.json')
    reference_summary={r['mode']:{k:v for k,v in r.items() if k!='frames'} for r in reference['runs']}
    p2=stages['2']['planning_profile']['ego'];p3=stages['3']['planning_profile']['ego']
    rival3=stages['3']['planning_profile']['rival_planning_profile']
    if rival3 is None:
        raise RuntimeError('Stage 3 rival planning profile is missing')
    report={'round_directory':str(root.resolve()),'manifest_digest':manifest['manifest_digest'],
            'reference_digest':manifest['reference_digest'],'reference':reference_summary,
            'training_rollouts_per_stage':manifest['training_rollouts_per_stage'],
            'unique_training_encounters_per_stage':manifest['train_cases_per_stage'],
            'training_variants_per_encounter':3,'validation_encounters_per_stage':manifest['validation_cases_per_stage'],
            'test_encounters_per_stage':manifest['test_cases_per_stage'],
            'episode_deadline_s':manifest['seconds'],'control_dt_s':manifest['control_dt'],
            'warmup_seconds':manifest.get('warmup_seconds',0.),
            'warmup_note':manifest.get('warmup_note','No pre-encounter driving interval configured.'),
            'stage3_gap_ranges_m':manifest.get('stage3_gap_ranges_m'),
            'split_sectors':manifest['split_sectors'],'stages':stages,
            'test_selection_used_for_fitting':False,'all_test_trials_retained':True,
            'website_scope':'At most four representative frozen-policy TEST recordings per stage; aggregate statistics cover every test trial',
            'training_semantics':f"{manifest['training_rollouts_per_stage']} full training rollouts per stage are {manifest['train_cases_per_stage']} independently randomized encounters, each run with three preference policies; they are not {manifest['training_rollouts_per_stage']} independent encounters.",
            'training_policy_semantics':'Training rollouts use the geometric teacher with three maneuver-preference variants. Neural checkpoints are fitted afterward and evaluated separately on validation and sealed tests.',
            'probability_semantics':'The value head estimates success under observed actions and continuation policies within the episode deadline; not causal action ground truth or a safety guarantee.',
            'reference_semantics':'Both cars share a locally lap-time-refined reference and ordinary speed-planning method. Stage 2 uses a slower rival speed cap; stage 3 uses equal top-speed caps. A finite optimizer does not certify the globally fastest possible line.',
            'inside_move_semantics':'The planner searches for temporary inside departures within track bounds and checks car footprints. Opponent reactions emerge from its own stage 2 policy; no scripted yielding or command access. Contact trials remain failures.',
            'stage2_allowance_semantics':f"Stage 2 uses an ordinary turning allowance of {p2['lateral_acceleration_m_s2']:g} m/s² and an active-passing allowance of {p2['passing_lateral_acceleration_m_s2']:g} m/s². The front car follows the same reference with its lower configured speed cap. These are planner allowances; the vehicle physics parameters are unchanged.",
            'stage3_allowance_semantics':f"Stage 3 uses ordinary / active-passing turning allowances of {p3['lateral_acceleration_m_s2']:g} / {p3['passing_lateral_acceleration_m_s2']:g} m/s²; its actual stage 2 neural rival uses {rival3['lateral_acceleration_m_s2']:g} / {rival3['passing_lateral_acceleration_m_s2']:g} m/s². Both have the same 2.4 m/s top-speed cap and shared reference. The larger ego passing allowance and different clearance policies contribute to the comparison; successful passes cannot be attributed to the neural network alone.",
            'stage3_advantage_sources':{'ego_active_passing_allowance_m_s2':p3['passing_lateral_acceleration_m_s2'],
                                        'rival_active_passing_allowance_m_s2':rival3['passing_lateral_acceleration_m_s2'],
                                        'ego_preferred_car_padding_m':p3['car_margin_m'],
                                        'rival_preferred_car_padding_m':rival3['car_margin_m'],
                                        'scope':'Controller allowances, recovery eligibility and learned decisions all differ; this is not a controlled neural-intelligence-only comparison.'},
            'padding_recovery_semantics':'Both moving-car models can restore depleted preferred car padding under a geometric forecast. Stage 2 permits this only to complete an existing committed pass: it must previously have seen the target ahead and now see it behind. Stage 3 permits recovery during passing encounters more generally. The forecast uses the observed starting body and estimated obstacle, requires at least 0.025 m predicted separation, permits no loss more than 0.005 m below the estimated initial gap, and predicts preferred padding restored within 0.5 s. This forecast is not an actual-time guarantee. Only known-car side-padding LiDAR returns may be released; physical body width, unknown/wall returns and front braking remain checked.',
            'network_timing_semantics':'Displayed network latency includes the maneuver network and any deployed value head. Whole-controller latency also includes sensing, candidate planning and safety filtering.',
            'rollout_timing_context':'Per-run latency is computer execution time observed during concurrent collection with five workers, not a physical reaction-time guarantee. A separate serial benchmark is reported separately when available.',
            **llm_accounting(stages),
            'llm_provenance':'Advice uses training observations only and contributes separately identified feasible imitation labels. Pass/fail labels remain simulator outcomes. Preserved stage 2 advice and its historical cost are reported separately from new stage 3 requests. No API calls occur during fitting, validation or testing; old records and hashes are never relabeled as newly collected data.',
            'stage2_predecessor_compatible':predecessor is not None,
            'stage3_setup_semantics':setup_description(stages['3']['coordinated_stage3_warmup']),
            'limitations':['Same known map, with disjoint scenario/seed/start-sector partitions within this round; not unseen-track generalization.',
                           'Paired variants and frames within each episode are correlated.',
                           'A completed inside pass alone does not establish that the rival was forced off its line.',
                           'Active-passing allowances, clearance settings and learned decisions all affect outcomes; the comparison does not isolate neural intelligence.',
                           'Padding-restoration timing is a prediction under the motion estimate. Guard activation duration does not establish actual restoration time.',
                           'Runtime uses learned maneuver/outcome estimates plus geometric trajectories and safety filtering, not end-to-end neural raw steering.']}
    benchmark=verified_serial_benchmark(ARTIFACTS/'benchmark_v3.json',manifest)
    if benchmark is not None:
        report['serial_benchmark']=benchmark
    baseline=verified_stage2_baseline(root,manifest,test_collections[2])
    if baseline is not None:
        report['fixed_stage2_baseline_comparison']=baseline
    return report,manifest,gallery


def fmt(value,digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def profile_section(report):
    record=report['stages']['3'];profile=record['planning_profile'];p=profile['ego']
    rival=profile['rival_planning_profile'];p2=report['stages']['2']['planning_profile']['ego']
    usage=record['padding_recovery']['sealed_test']
    usage2=report['stages']['2']['padding_recovery']['sealed_test'];other_usage=record['opponent_padding_recovery']
    caps=' / '.join(f'{v:g}' for v in profile['ego_top_speed_caps_m_s'])
    rival_caps=' / '.join(f'{v:g}' for v in profile['rival_top_speed_caps_m_s'])
    margins=profile.get('rival_safety_margins')
    clearance_note=('' if margins is None else f" Preferred car / wall padding is {p['car_margin_m']:.2f} / {p['wall_margin_m']:.2f} m for our car and {margins['car']:.2f} / {margins['wall']:.2f} m for the rival.")
    return f'''<section class="card bottom" style="margin:0 0 18px"><h2>Driving limits and recovery conditions</h2>
<p>Stage 2 uses {p2['lateral_acceleration_m_s2']:g} m/s² of ordinary turning allowance and {p2['passing_lateral_acceleration_m_s2']:g} m/s² while actively passing its slower line-following rival. In stage 3, both cars return to the same locally optimized line when unobstructed, with top-speed caps of {caps} / {rival_caps} m/s. Our car uses ordinary / active-passing allowances of {p['lateral_acceleration_m_s2']:g} / {p['passing_lateral_acceleration_m_s2']:g} m/s²; the actual stage 2 rival uses {rival['lateral_acceleration_m_s2']:g} / {rival['passing_lateral_acceleration_m_s2']:g} m/s². These are planner settings, not measured acceleration or changed vehicle physics.{clearance_note} The passing and clearance allowances contribute to the result; success cannot be attributed to the neural network alone.</p>
<p style="margin-top:10px">The rival runs its own stage 2 neural controller and reacts to its own sensors. It receives no instruction to yield. A completed pass alone does not prove that the rival was forced sideways.</p>
<details style="margin-top:14px"><summary>Padding recovery: recorded usage and limits</summary>
<p class="small" style="margin-top:9px">Stage 2 permits padding recovery only while completing an already committed pass, after seeing the target ahead and subsequently behind. Stage 3 can use it more generally during passing encounters. The geometric forecast checks an estimated starting body gap of at least {p['padding_recovery_minimum_predicted_gap_m']:.3f} m, no drop more than {p['padding_recovery_maximum_predicted_gap_loss_m']:.3f} m below that gap, and restoration of preferred padding within {p['padding_recovery_predicted_restore_seconds']:g} s in the forecast. That {p['padding_recovery_predicted_restore_seconds']:g} s is not a guarantee of actual recovery time. Only identified car-side padding returns may receive reduced LiDAR padding; the physical body and unknown, wall and front returns remain checked.</p>
<p class="small" style="margin-top:9px">In sealed stage 3 tests, {usage['trials_using_guard']} / {usage['total_trials']} trials used our guard ({usage['certified_decisions']} control decisions, {fmt(usage['nominal_active_control_seconds'])} s of combined active control intervals). Of those trials, {usage['passes_using_guard']} passed and {usage['contacts_using_guard']} recorded contact. Our LiDAR side-padding release occurred on {usage['lidar_release_decisions']} decisions. The actual rival used completion recovery in {other_usage['trials_using_guard']} / {other_usage['trials']} trials. The stage 2 ego used it in {usage2['trials_using_guard']} / {usage2['total_trials']} stage 2 tests. These are usage counts, not a causal benefit estimate or measured restoration time.</p></details></section>'''


def outcome_status_section(report):
    panels=[]
    for stage,record in report['stages'].items():
        status=record['passing_status'];color='#eab676' if status['passing_objective_met'] is False else '#b9d9da'
        training=record['training']
        evidence=(f"Training teachers completed {training['passes']}/{training['trials']} rollouts successfully, "
                  f"covering {record['training_unique_winning_encounters']}/{record['training_unique_encounters']} independent encounters. "
                  "Those teacher demonstrations are separate from the selected neural policy's validation and test results.")
        panels.append(f'<div style="margin-bottom:14px"><h3 style="color:{color}">{html.escape(status["headline"])}</h3><p>{html.escape(status["statement"])}</p><p class="small">{html.escape(status["detail"])}</p><p class="small" style="margin-top:8px">{html.escape(evidence)}</p></div>')
    return '<section class="card bottom" style="margin:0 0 18px">'+''.join(panels)+'</section>'


def summary_section(report):
    rows=[];selection_rows=[]
    for stage,record in report['stages'].items():
        for label,key in (('Training teacher rollouts','training'),('Selected neural validation','selected_neural_validation'),('Sealed test','sealed_test')):
            if record.get('data_provenance',{}).get('training_imported_unchanged'):
                label={'training':'Preserved training teacher rollouts','selected_neural_validation':'Current-runtime validation replay'}.get(key,label)
            r=record[key]
            rows.append(f'<tr><td>Stage {stage}</td><td>{label}</td><td>{r["trials"]}</td><td>{r["passes"]}</td><td>{100*r["success_rate"]:.1f}%</td><td>{r["collisions"]}</td><td>{fmt(r["median_success_time_s"])} s</td></tr>')
        for checkpoint,metrics_ in record['checkpoint_validation'].items():
            selected=(' · preserved selection' if record.get('data_provenance',{}).get('training_imported_unchanged') else ' · selected') if checkpoint==record['selected_checkpoint'] else ''
            selection_rows.append(f'<tr><td>Stage {stage}</td><td>{html.escape(checkpoint+selected)}</td><td>{metrics_["passes"]}/{metrics_["trials"]}</td><td>{metrics_["collisions"]}</td><td>{fmt(metrics_["mean_success_time_s"])} s</td></tr>')
    return f'''<section class="card bottom" style="margin:0 0 18px"><h2>Success first. Earlier completion second.</h2>
<p>Every sealed test is counted below. The gallery contains a few representative test recordings, including failures when present; its examples are not the success rate.</p>
<div class="tableWrap"><table><thead><tr><th>Model</th><th>Partition</th><th>Trials</th><th>Passes</th><th>Success</th><th>Contacts</th><th>Median completed pass</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="small" style="margin-top:12px">Training: {report['training_rollouts_per_stage']} full teacher rollouts per stage = {report['unique_training_encounters_per_stage']} randomized encounters × three preference policies. Neural checkpoints are fitted afterward. Each stage has {report['validation_encounters_per_stage']} separate validation encounters and {report['test_encounters_per_stage']} sealed test encounters. Tests were never used for fitting or checkpoint selection. Pass times include successful runs only; every unfinished attempt remains a failure. The {report.get('warmup_seconds',0.):g} s of shared-reference driving used to establish real sensor history is included in episode times and contact scoring.</p>
<details style="margin-top:16px"><summary>How the neural checkpoints were selected</summary><p class="small">Selection uses validation only: require no contacts, maximize completed passes, then prefer earlier completion. These are actual driving outcomes, not classification accuracy.</p><div class="tableWrap"><table><thead><tr><th>Model</th><th>Checkpoint</th><th>Validation passes</th><th>Contacts</th><th>Mean completed pass</th></tr></thead><tbody>{''.join(selection_rows)}</tbody></table></div></details></section>'''


def provenance_section(report):
    p=report['stages']['2']['data_provenance']
    retained=''
    if p['training_imported_unchanged']:
        retained=f"<p>Stage 2 retains its original training, advice, selection and model files unchanged. All {p['current_runtime_validation_trials']} validation replays matched their preserved decisions and outcomes under the updated simulator setup. Its selected model was not reselected or retrained. Stage 3 was collected and trained under the new setup. Every final test uses the current runtime.</p>"
    accounting=(f"The models include {report['total_llm_advice_rows_consumed']} training-advice rows: "
                f"{report['fresh_llm_advice_rows_consumed']} from this round and {report['imported_llm_advice_rows_consumed']} preserved with Stage 2. "
                f"This round made {report['new_api_calls']} review requests, with estimated usage ${report['estimated_api_usage_cost_usd']:.6f}. "
                f"Preserved advice has a separate historical usage estimate of ${report['imported_api_usage_cost_usd']:.6f}; it did not trigger new requests. "
                "Promotional credits and actual billing are not inferred. Racing uses no runtime API calls.")
    return f'''<section class="card bottom" style="margin:0 0 18px"><h2>Training provenance and physical setup</h2>{retained}<p style="margin-top:10px">{html.escape(report['stage3_setup_semantics'])} Setup time and contacts count toward the outcome.</p><p class="small" style="margin-top:10px">{html.escape(accounting)}</p></section>'''


def serial_benchmark_section(report):
    benchmark=report.get('serial_benchmark')
    if benchmark is None:
        return ''
    rows=[]
    for stage,result in benchmark['results'].items():
        rows.append(f"<tr><td>Stage {html.escape(stage)}</td><td>{result['full_controller']['count']}</td><td>{result['deployed_network']['p95_ms']:.3f} ms</td><td>{result['full_controller']['p95_ms']:.3f} ms</td></tr>")
    return f'''<section class="card bottom" style="margin:0 0 18px"><h2>Serial compute timing</h2><div class="tableWrap"><table><thead><tr><th>Model</th><th>Measured calls</th><th>Deployed networks · p95</th><th>Full controller · p95</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p class="small" style="margin-top:10px">{html.escape(benchmark['interpretation'])} These results are specific to this computer and these two scenes.</p></section>'''


def baseline_comparison_section(report):
    comparison=report.get('fixed_stage2_baseline_comparison')
    if comparison is None:
        return ''
    rows=[]
    for key,label in (('baseline','Previous Stage 2 controller and model'),('final','Current Stage 2 controller and model')):
        r=comparison[key]
        rows.append(f"<tr><td>{label}</td><td>{r['passes']}/{r['runs']}</td><td>{r['contacts']}</td><td>{fmt(r['median_successful_pass_seconds'],3)} s</td></tr>")
    paired=(f"Both passed exactly the same {comparison['common_successes']} scenarios. " if comparison['same_successful_scenarios'] else '')
    if comparison['common_successes']:
        paired+=f"Of the {comparison['common_successes']} shared successful cases, {comparison['common_successes_faster']} finished sooner; the mean paired time saved was {comparison['mean_paired_seconds_saved']:.3f} s."
    return f'''<section class="card bottom" style="margin:0 0 18px"><h2>Fixed comparison with the previous Stage 2 system</h2><p>{html.escape(comparison['interpretation'])}</p><div class="tableWrap"><table><thead><tr><th>System</th><th>Test passes</th><th>Contacts</th><th>Median completed pass</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p class="small" style="margin-top:10px">{html.escape(paired)}</p></section>'''


def build_page(report,gallery):
    from replay import HTML,load_replay_data
    data=load_replay_data([r['trace'] for r,_ in gallery],history_paths=())
    data['round_summary']=report
    for trace,(record,label) in zip(data['traces'],gallery):
        trace['name']=label+' · '+record['job']['case_id']
        trace['checkpoint']='selected'
        trace['result']['representative_note']=label
    data['initial_index']=0
    page=HTML.replace('<h1>Three stages. One track.</h1>','<h1>Overtaking results on the shared track.</h1>')
    page=page.replace('Watch recorded simulation runs, inspect decisions, and compare training checkpoints.','Inspect representative stage 2 and 3 tests, with complete training and test totals.')
    page=page.replace('<div id="trainingBanner" class="trainingBanner"></div>','<div id="trainingBanner" class="trainingBanner"></div>'+outcome_status_section(report)+summary_section(report)+baseline_comparison_section(report)+provenance_section(report)+serial_benchmark_section(report)+profile_section(report))
    page=page.replace('The reference is a locally curvature-reduced approximation, not a certified fastest line.','Both cars use the same locally lap-time-refined reference when unobstructed. An inside move stays within the track; the finite optimizer does not certify a globally fastest line.')
    page=page.replace('<div id="probabilities"></div>','<div id="probabilities"></div><div class="divider"></div><div class="small">Estimated pass chance by maneuver</div><div id="passChances" class="small"></div>')
    page=page.replace('Raw neural probabilities before safety / commitment filtering','Maneuver preferences before safety / commitment filtering')
    page=page.replace('<div class="small">Measured across this run</div>',f'<div class="small">Compute time during concurrent rollouts (5 workers). Decisions run every {1000*report["control_dt_s"]:g} ms; these timings are not physical reaction guarantees.</div>')
    page=page.replace("$('otherAction').textContent=", "const values=f.maneuver_success_probabilities;$('passChances').textContent=values?values.map((v,i)=>['Left','Right','Wait','Inside'][i]+': '+fmt(v*100,0)+'%').join(' · ')+' — empirical continuation estimates, not guarantees.':'Outcome head not deployed for this selected checkpoint.';$('otherAction').textContent=")
    page=page.replace('Driving checks: actual recorded episodes','Representative sealed tests')
    page=page.replace('These outcomes come from closed-loop simulator runs, separately from classification accuracy. Each row is one episode; a selected demonstration is not an overall success-rate estimate.','These gallery recordings are selected after testing for inspection. All test outcomes count in the aggregate table above, and every full trajectory remains in the round folder.')
    provenance="""function drawProvenance(){const r=DATA.round_summary;$('trainingBanner').textContent=`${2*r.training_rollouts_per_stage} training rollouts supporting stages 2 and 3 · ${r.total_llm_advice_rows_consumed} advice rows (${r.imported_llm_advice_rows_consumed} preserved) · ${2*r.test_encounters_per_stage} sealed tests · no runtime API calls. Both cars share the reference line when unobstructed.`}\nfunction drawDrivingResults"""
    page=re.sub(r'function drawProvenance\(\)\{.*?\}\nfunction drawDrivingResults',provenance,page,flags=re.S)
    encoded=json.dumps(data,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    return page.replace('__DATA__',encoded),data


def markdown_report(report):
    lines=['# Outcome-trained moving-car models','',
           f"Each stage completed {report['training_rollouts_per_stage']} full training rollouts: {report['unique_training_encounters_per_stage']} independent randomized encounters, each with three preference policies. Each also has {report['validation_encounters_per_stage']} separate validation encounters and {report['test_encounters_per_stage']} sealed test encounters. Scenario seeds, encounter records and start-sector partitions are disjoint within this round. Only training data enters fitting; validation selects checkpoints; the test set is used once after selection.",'',
           f"Each moving encounter includes {report.get('warmup_seconds',0.):g} seconds of real shared-reference driving to establish sensor history. This interval is included in the reported episode/pass time, deadline and contact scoring.",'',
           '| Stage | Partition | Trials | Passes | Success | Contacts | Median successful pass |',
           '|---|---|---:|---:|---:|---:|---:|']
    status_lines=[]
    for stage,record in report['stages'].items():
        status=record['passing_status'];training=record['training']
        status_lines.extend([f"**{status['headline']}.** {status['statement']} {status['detail']}",'',
            f"Stage {stage} training teachers passed {training['passes']}/{training['trials']} rollouts, covering {record['training_unique_winning_encounters']}/{record['training_unique_encounters']} independent encounters. Teacher demonstrations are separate from neural validation/test success.",''])
    lines[2:2]=status_lines
    for stage,record in report['stages'].items():
        for label,key in (('Training teacher rollouts','training'),('Selected neural validation','selected_neural_validation'),('Sealed test','sealed_test')):
            if record.get('data_provenance',{}).get('training_imported_unchanged'):
                label={'training':'Preserved training teacher rollouts','selected_neural_validation':'Current-runtime validation replay'}.get(key,label)
            r=record[key]
            lines.append(f"| {stage} | {label} | {r['trials']} | {r['passes']} | {r['success_rate']:.1%} | {r['collisions']} | {fmt(r['median_success_time_s'])} s |")
    lines+=['','Pass-time summaries include successes only. No-pass/contact cases remain failures, not zero-time passes. Training variant rollouts are paired, and episode frames are correlated.','',
            'Both cars share the same optimized reference line and ordinary speed-planning method. The stage 2 rival has a lower top-speed cap; stage 3 uses equal top-speed caps. The planner searches for temporary inside departures within track bounds and checks car footprints; all contact runs remain failures. There is no scripted opponent yielding. The reference is a finite local lap-time optimization, not a certified global optimum.','',
            report['stage2_allowance_semantics'],'',report['stage3_allowance_semantics'],'',report['padding_recovery_semantics'],'',
            report['network_timing_semantics']+' '+report['rollout_timing_context'],'',
            'The imitation network learns from actually successful winning trajectories, supplemented by separately identified feasible LLM advice from training observations. The additional value network uses completed pass/failure outcomes from full rollouts; LLM advice does not modify these binary outcome labels. Its probabilities estimate continuation success under the recorded policy; they are not causal action ground truth or a safety guarantee. Checkpoint selection requires no validation contacts, then prioritizes success count, then successful-pass time.','',
            f"The models consumed {report['total_llm_advice_rows_consumed']} accepted advice rows in total: {report['fresh_llm_advice_rows_consumed']} from this round and {report['imported_llm_advice_rows_consumed']} preserved with Stage 2. This round recorded {report['new_api_calls']} training-review API requests, estimated at ${report['estimated_api_usage_cost_usd']:.6f}. Preserved advice has a separate historical usage estimate of ${report['imported_api_usage_cost_usd']:.6f} and made no new requests. Promotional credits and actual billing are not inferred. No API calls occur during fitting, validation or testing.",'',
            f"The replay page contains at most four representative test clips per stage. Full training, validation and test summaries and recordings remain under `{report['round_directory']}`. Stage 1 is unchanged and is excluded from this gallery to avoid displaying its older reference as the new one.",'',
            'The model remains a hybrid of learned decisions, geometry and safety control. Success on this known track does not establish unseen-track generalization or real-car reliability.']
    lines+=['',report['stage3_setup_semantics']]
    if report['stage2_predecessor_compatible']:
        p=report['stages']['2']['data_provenance']
        lines+=['',f"Stage 2 training, reviews, model files and selection are preserved byte-for-byte. Its {p['current_runtime_validation_trials']} current-runtime validation replays matched their original decisions and outcomes; the checkpoint was not reselected. Original training manifest: `{p['original_training_manifest_digest']}`. Current evaluation manifest: `{p['current_evaluation_manifest_digest']}`. The stage-scoped compatibility certificate is `{p['compatibility_certificate']}` (SHA256 `{p['compatibility_certificate_sha256']}`). Stage 3 and every final test require current runtime provenance."]
    if report.get('serial_benchmark') is not None:
        benchmark=report['serial_benchmark']
        lines+=['','Serial compute benchmark (p95):','',
                '| Model | Measured calls | Deployed networks | Full controller |',
                '|---|---:|---:|---:|']
        for stage,result in benchmark['results'].items():
            lines.append(f"| Stage {stage} | {result['full_controller']['count']} | {result['deployed_network']['p95_ms']:.3f} ms | {result['full_controller']['p95_ms']:.3f} ms |")
        lines+=['',benchmark['interpretation']+' These results are specific to this computer and these two scenes.']
    if report.get('fixed_stage2_baseline_comparison') is not None:
        comparison=report['fixed_stage2_baseline_comparison'];old=comparison['baseline'];new=comparison['final']
        lines+=['','Fixed comparison with the previous Stage 2 controller and model:','',
                comparison['interpretation'],'',
                f"Previous: {old['passes']}/{old['runs']} passes, {old['contacts']} contacts, median completed pass {fmt(old['median_successful_pass_seconds'],3)} s. Current: {new['passes']}/{new['runs']} passes, {new['contacts']} contacts, median completed pass {fmt(new['median_successful_pass_seconds'],3)} s."]
        if comparison['same_successful_scenarios']:
            lines+=['',f"Both passed exactly the same {comparison['common_successes']} scenarios."]
        if comparison['common_successes']:
            lines+=['',f"Among those shared successful cases, {comparison['common_successes_faster']}/{comparison['common_successes']} finished sooner; mean paired time saved was {comparison['mean_paired_seconds_saved']:.3f} s."]
    guard=report['stages']['3']['padding_recovery']['sealed_test']
    lines+=['',f"Stage 3 sealed-test padding recovery: {guard['trials_using_guard']}/{guard['total_trials']} trials used the guard, with {guard['certified_decisions']} active decisions and {guard['nominal_active_control_seconds']:.2f} combined active control seconds. {guard['passes_using_guard']} of those trials passed and {guard['contacts_using_guard']} recorded contact. Side-padding LiDAR release occurred on {guard['lidar_release_decisions']} decisions. Active-control totals are not measured restoration times or causal evidence of benefit."]
    return '\n'.join(lines)+'\n'


def write_gif(report,gallery,output):
    """A short selection of real test frames, including failure clips when present."""
    from replay import load_replay_data
    from render_preview import render_frame,font,MUTED
    from PIL import Image,ImageDraw
    chosen=[]
    for stage in (2,3):
        subset=[(r,label) for r,label in gallery if r['job']['stage']==stage]
        passed=[item for item in subset if item[0]['result'].get('success')]
        failed=[item for item in subset if not item[0]['result'].get('success')]
        if passed:chosen.append(passed[0])
        if failed:chosen.append(failed[0])
        if not passed and not failed:raise RuntimeError('No representative test recordings')
    data=load_replay_data([r['trace'] for r,_ in chosen],history_paths=())
    frames=[];manifest=[]
    for trace,(record,label) in zip(data['traces'],chosen):
        trace['checkpoint']='selected'
        times=np.asarray([f['t'] for f in trace['frames']])
        indices=np.unique(np.r_[np.searchsorted(times,np.arange(times[0],times[-1],.2),side='right')-1,len(times)-1])
        images=[]
        for index in indices:
            image=render_frame(data,trace,int(index),playback_speed=2.)
            draw=ImageDraw.Draw(image)
            draw.rectangle((0,image.height-30,image.width,image.height),fill='#0b1020')
            text=f"{report['training_rollouts_per_stage']} training rollouts / stage | Separate sealed test | {label} | No runtime API"
            draw.text((22,image.height-24),text,fill=MUTED,font=font(11))
            images.append(image)
        frames.extend([images[0]]*3+images+[images[-1]]*7)
        manifest.append({'trial_id':record['job']['trial_id'],'selection_label':label,
                         'outcome':record['result']['outcome'],'frame_indices':indices.tolist(),
                         'model_sha256':record['result']['model_sha256']})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    palette=[im.quantize(colors=128,method=Image.Quantize.FASTOCTREE) for im in frames]
    palette[0].save(output,save_all=True,append_images=palette[1:],duration=100,loop=0,optimize=True,disposal=2)
    frames[0].save(output.with_suffix('.first.png'));frames[-1].save(output.with_suffix('.last.png'))
    output.with_suffix('.manifest.json').write_text(json.dumps({'kind':'recorded test replay','fps':10,'speed':2.,
        'selected_examples_only':True,'frames':len(frames),'sources':manifest},indent=2)+'\n')
    return {'path':str(output.resolve()),'frames':len(frames),'seconds':len(frames)/10,'bytes':output.stat().st_size}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=FINAL_ROUND_ROOT)
    parser.add_argument('--test-tag')
    parser.add_argument('--output',type=Path,default=ARTIFACTS/'replay.html')
    parser.add_argument('--report',type=Path,default=ARTIFACTS/'v3_report.json')
    parser.add_argument('--gif',type=Path,help='Optional short recorded-test GIF, with a failure clip per stage when available')
    parser.add_argument('--markdown',type=Path,default=ARTIFACTS/'RESULTS.md')
    args=parser.parse_args()
    report,manifest,gallery=assemble(args.root,args.test_tag)
    page,data=build_page(report,gallery)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(page)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    args.markdown.write_text(markdown_report(report))
    selected_path=args.root/'representative_tests.json'
    selected_path.write_text(json.dumps([{'trial_id':r['job']['trial_id'],'trace':r['trace'],'label':label} for r,label in gallery],indent=2)+'\n')
    gif=write_gif(report,gallery,args.gif) if args.gif else None
    print(json.dumps({'gif':gif,'website':str(args.output.resolve()),'representative_test_clips':len(data['traces']),
                      'all_test_trials':sum(r['sealed_test']['trials'] for r in report['stages'].values()),
                      'report':str(args.report.resolve())}))


if __name__=='__main__':
    main()
