"""Preserve Stage 2 verbatim and certify a Stage-3-only setup compatibility change.

`prepare` copies an immutable, hash-inventoried predecessor; it never rewrites
its records or models. `revalidate` replays all four existing neural checkpoints
on their same ten validation encounters without training, selection or deployment.
Only exact stable behavior matches can produce a compatibility certificate.
Final tests are never executed or read by this helper.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

from bootstrap import HERE,MODELS
from rollout_training import collect,digest,file_hash,select_jobs,sources,validate_manifest

CHECKPOINTS=('early','mid','final','imitation_only')
EXTRA_COMMON_RUNTIME=('bootstrap.py','tiny_network.py','moving_model.py','static_model.py','interactive_model.py')
STABLE_RESULT_FIELDS=('scenario','sensor_seeds','warmup_seconds','warmup_lidar_samples_per_car',
    'actual_gap_at_policy_start','model_sha256','value_model_sha256','opponent_model_sha256',
    'opponent_value_model_sha256','opponent_controller','opponent_safety_margins',
    'physics_dt_seconds','control_dt_seconds','policy_bias','reference_sha256','ego_planning_profile',
    'success','safe_blocked_stop','outcome','collision','passed_at','sim_seconds',
    'pass_opportunity_deadline_seconds','pass_followthrough_complete','minimum_body_gap',
    'minimum_wall_clearance','minimum_body_wall_clearances','progress','mean_speed','peak_speed',
    'action_seconds','safety_mask_override_ticks','commitment_override_ticks',
    'padding_recovery_decisions','padding_recovery_nominal_seconds',
    'padding_lidar_release_decisions','padding_lidar_released_beams')


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def seal(value,key):
    value=dict(value);value[key]=digest(value);return value


def check_seal(value,key):
    if digest({k:v for k,v in value.items() if k!=key})!=value.get(key):
        raise ValueError(f'{key} integrity check failed')


def stage2_jobs_digest(manifest):
    return digest([j for j in manifest['jobs'] if j['stage']==2])


def runtime_change(previous,current):
    if set(previous)!=set(current):raise ValueError('Runtime file inventory changed')
    changed=sorted(name for name in previous if previous[name]!=current[name])
    if changed!=['run_experiment.py']:
        raise ValueError('Compatibility permits exactly run_experiment.py to change')
    return changed


def verify_import_inventory(preparation):
    imported=Path(preparation['imported_root']).resolve()
    predecessor=Path(preparation['predecessor_root']).resolve()
    seen=set()
    for item in preparation['imported_files']:
        relative=Path(item['relative_path'])
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() in seen:
            raise ValueError('Invalid or duplicate imported artifact path')
        seen.add(relative.as_posix())
        original=(predecessor/relative).resolve();copy=(imported/relative).resolve()
        if (not original.is_relative_to(predecessor) or not copy.is_relative_to(imported)
                or str(original)!=item['original_path']):
            raise ValueError('Imported artifact escaped its declared predecessor namespace')
        if file_hash(original)!=item['sha256'] or file_hash(copy)!=item['sha256']:
            raise ValueError(f'Original/imported Stage2 artifact changed: {relative}')
    return imported


def selected_pair(selection):
    selected=selection['selected'];report=selection['reports'][selected]
    model=Path(selection['selected_model']).resolve()
    value=model.with_name(model.stem+'.value.npz')
    if file_hash(model)!=report['model_sha256']:
        raise ValueError('The originally selected Stage2 maneuver checkpoint changed')
    expected=report['model_artifacts'].get(str(value))
    if (file_hash(value) if value.exists() else None)!=expected:
        raise ValueError('The originally selected Stage2 outcome checkpoint changed')
    return report['model_sha256'],expected


def verify_deployed_pair(model_hash,value_hash):
    model=MODELS/'stage2.npz';value=MODELS/'stage2.value.npz'
    if (file_hash(model)!=model_hash or (file_hash(value) if value.exists() else None)!=value_hash):
        raise ValueError('The frozen deployed Stage2 pair changed; compatibility does not authorize reselection')


def prepare_stage2_predecessor(predecessor_root,root):
    predecessor=Path(predecessor_root).resolve();root=Path(root).resolve()
    if predecessor==root:raise ValueError('Predecessor and current round must differ')
    namespace=root/'stage2_predecessor';imported=namespace/'artifacts'
    path=namespace/'PREPARATION.json'
    if path.exists():
        preparation=read_json(path);check_seal(preparation,'preparation_digest')
        if preparation['predecessor_root']!=str(predecessor) or preparation['current_root']!=str(root):
            raise ValueError('Existing predecessor import belongs to another round')
        verify_import_inventory(preparation)
        verify_deployed_pair(preparation['selected_model_sha256'],preparation['selected_value_sha256'])
        return preparation
    manifest=read_json(predecessor/'manifest.json');validate_manifest(manifest)
    train=read_json(predecessor/'trials/stage2/train/teacher/summary.json')
    validation=read_json(predecessor/'trials/stage2/validation/teacher/summary.json')
    for report,split,count in ((train,'train',manifest['training_rollouts_per_stage']),
                               (validation,'validation',manifest['validation_cases_per_stage']*3)):
        expected={j['trial_id']:j for j in select_jobs(manifest,2,split)}
        if (not report.get('completed') or report['completed_trials']!=count or len(report['trials'])!=count
                or report['manifest_digest']!=manifest['manifest_digest']
                or {r['job']['trial_id']:r['job'] for r in report['trials']}!=expected):
            raise ValueError('Predecessor Stage2 collection is incomplete or mismatched')
    if train['source_sha256']!=validation['source_sha256']:
        raise ValueError('Predecessor Stage2 collections mix runtime versions')
    selection=read_json(predecessor/'stage2.selection.json')
    if not selection.get('qualified') or not selection.get('validation_only'):
        raise ValueError('Predecessor Stage2 checkpoint lacks valid qualification')
    model_hash,value_hash=selected_pair(selection);verify_deployed_pair(model_hash,value_hash)
    required={'manifest.json','stage2.training.json','stage2.selection.json'}
    for folder in ('trials/stage2/train','trials/stage2/validation','frozen_source'):
        required.update(p.relative_to(predecessor).as_posix() for p in (predecessor/folder).rglob('*') if p.is_file())
    for pattern in ('models/stage2*','priors/stage2*','datasets/stage2.*','reviews/stage2.*',
                    'phase_stage2_*.log','stage2.*.json'):
        required.update(p.relative_to(predecessor).as_posix() for p in predecessor.glob(pattern) if p.is_file())
    inventory=[]
    for relative in sorted(required):
        original=predecessor/relative;destination=imported/relative
        checksum=file_hash(original)
        if destination.exists():
            if file_hash(destination)!=checksum:raise ValueError('Refusing to overwrite a differing imported artifact')
        else:
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,destination)
        inventory.append(dict(relative_path=relative,original_path=str(original.resolve()),sha256=checksum))
    for name,checksum in train['source_sha256'].items():
        if file_hash(imported/'frozen_source'/name)!=checksum:
            raise ValueError('Archived predecessor runtime differs from its recorded hashes')
    fit=read_json(imported/'stage2.training.json')
    preparation=seal(dict(format_version=1,kind='stage2_predecessor_import',
        predecessor_root=str(predecessor),current_root=str(root),imported_root=str(imported),
        predecessor_manifest_digest=manifest['manifest_digest'],reference_digest=manifest['reference_digest'],
        stage2_jobs_sha256=stage2_jobs_digest(manifest),predecessor_source_sha256=train['source_sha256'],
        predecessor_training_pipeline_sha256=fit['metadata']['training_pipeline_sha256'],
        extra_common_runtime_sha256={name:file_hash(imported/'frozen_source'/name) for name in EXTRA_COMMON_RUNTIME},
        selected_checkpoint=selection['selected'],selected_model_sha256=model_hash,selected_value_sha256=value_hash,
        imported_files=inventory,prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        no_retraining_reselection_or_test_execution=True),'preparation_digest')
    write_json(path,preparation);verify_import_inventory(preparation)
    return preparation


def behavior_digest(trace):
    result=trace['result']
    missing=set(STABLE_RESULT_FIELDS)-set(result)
    if missing:raise ValueError(f'Missing stable behavior fields: {sorted(missing)}')
    return digest(dict(result={name:result[name] for name in STABLE_RESULT_FIELDS},frames=trace['frames']))


def compare_validation_reports(previous,current,manifest,expected_current_sources):
    old={r['job']['case_id']:r for r in previous['trials']}
    new={r['job']['case_id']:r for r in current['trials']}
    expected={j['case_id']:j for j in select_jobs(manifest,2,'validation',policy='neural')}
    if (len(old)!=10 or len(new)!=10 or len(previous['trials'])!=10 or len(current['trials'])!=10
            or set(old)!=set(new) or set(new)!=set(expected) or not current.get('completed')
            or current.get('policy')!='neural' or current['source_sha256']!=expected_current_sources
            or current['manifest_digest']!=manifest['manifest_digest']):
        raise ValueError('Equivalence requires all ten identical validation encounters under the current runtime')
    comparisons=[]
    for case in sorted(old):
        a,b=old[case],new[case]
        if a['job']!=b['job'] or b['job']!=expected[case]:
            raise ValueError('Stage2 validation initial encounter changed')
        if b['source_sha256']!=expected_current_sources or b['manifest_digest']!=manifest['manifest_digest']:
            raise ValueError('Current Stage2 validation trial provenance differs')
        ta,tb=read_json(a['trace']),read_json(b['trace'])
        if ta['result']!=a['result'] or tb['result']!=b['result']:
            raise ValueError('Validation trace result differs from its summary')
        if (b['result'].get('teacher') or b['result'].get('network_access')!='blocked'
                or a['result']['model_sha256']!=b['result']['model_sha256']
                or a['result']['value_model_sha256']!=b['result']['value_model_sha256']):
            raise ValueError('Equivalence replay changed checkpoint or permitted network access')
        da,db=behavior_digest(ta),behavior_digest(tb)
        if da!=db:raise ValueError(f'Stage2 behavior changed for {case}; no compatibility certificate permitted')
        comparisons.append(dict(case_id=case,predecessor_trial_id=a['job']['trial_id'],current_trial_id=b['job']['trial_id'],
            predecessor_trace=str(Path(a['trace']).resolve()),predecessor_trace_sha256=file_hash(a['trace']),
            current_trace=str(Path(b['trace']).resolve()),current_trace_sha256=file_hash(b['trace']),
            behavior_sha256=da,exact_behavior_match=True,outcome=b['result']['outcome']))
    return comparisons


def check_current_scope(preparation,root,current_manifest):
    check_seal(preparation,'preparation_digest');verify_import_inventory(preparation)
    validate_manifest(current_manifest)
    if (preparation['current_root']!=str(Path(root).resolve())
            or preparation['reference_digest']!=current_manifest['reference_digest']
            or preparation['stage2_jobs_sha256']!=stage2_jobs_digest(current_manifest)):
        raise ValueError('Current Stage2 scenarios/reference differ from the preserved predecessor')
    changed=runtime_change(preparation['predecessor_source_sha256'],sources())
    for name,checksum in preparation['extra_common_runtime_sha256'].items():
        if file_hash(HERE/name)!=checksum:raise ValueError(f'Common Stage2 runtime dependency changed: {name}')
    verify_deployed_pair(preparation['selected_model_sha256'],preparation['selected_value_sha256'])
    return changed


def revalidate_stage2_predecessor(root,*,workers=5):
    root=Path(root).resolve();namespace=root/'stage2_predecessor';preparation=read_json(namespace/'PREPARATION.json')
    manifest=read_json(root/'manifest.json');changed=check_current_scope(preparation,root,manifest)
    imported=Path(preparation['imported_root']);comparisons={}
    for checkpoint in CHECKPOINTS:
        name='stage2.npz' if checkpoint=='final' else f'stage2.{checkpoint}.npz'
        print(f'Stage2 compatibility: replaying {checkpoint} on10 unchanged validation encounters',flush=True)
        current=collect(root,manifest,2,'validation',workers=workers,policy='neural',model=imported/'models'/name,
                        tag=f'compatibility_{checkpoint}')
        previous_path=imported/'trials/stage2/validation'/f'selection_{checkpoint}'/'summary.json'
        current_path=root/'trials/stage2/validation'/f'compatibility_{checkpoint}'/'summary.json'
        previous=read_json(previous_path)
        records=compare_validation_reports(previous,current,manifest,sources())
        comparisons[checkpoint]=dict(predecessor_summary=str(previous_path),predecessor_summary_sha256=file_hash(previous_path),
            current_summary=str(current_path),current_summary_sha256=file_hash(current_path),
            compared_trials=len(records),exact_behavior_matches=len(records),trials=records)
    check_current_scope(preparation,root,manifest)
    certificate=seal(dict(format_version=1,kind='stage2_compatible_predecessor',compatible_stage2=True,
        preparation_digest=preparation['preparation_digest'],predecessor_root=preparation['predecessor_root'],
        imported_root=preparation['imported_root'],current_root=str(root),
        predecessor_manifest_digest=preparation['predecessor_manifest_digest'],current_manifest_digest=manifest['manifest_digest'],
        reference_digest=manifest['reference_digest'],stage2_jobs_sha256=preparation['stage2_jobs_sha256'],
        predecessor_source_sha256=preparation['predecessor_source_sha256'],current_source_sha256=sources(),changed_files=changed,
        selected_checkpoint=preparation['selected_checkpoint'],selected_model_sha256=preparation['selected_model_sha256'],
        selected_value_sha256=preparation['selected_value_sha256'],imported_files=preparation['imported_files'],
        validation_equivalence=dict(expected_trials=40,checkpoints=list(CHECKPOINTS),reports=comparisons),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Stage2 predecessor training/teacher-validation/fit/selection only. Forty current-runtime validation traces match exactly in controls, probabilities, poses and stable outcomes. Original bytes and hashes remain unchanged. Stage3 and all final tests must independently satisfy current-runtime provenance.',
        limitations='Trace equivalence covers these forty validation runs, not every possible state. The setup change is declared Stage3-only; the only changed recorded runtime file is run_experiment.py. No Stage2 refitting, reselection, deployment changes or final tests occur here.'),'certificate_digest')
    write_json(namespace/'COMPATIBILITY.json',certificate)
    return verify_stage2_predecessor(root,manifest)


def verify_stage2_predecessor(root,current_manifest):
    root=Path(root).resolve();namespace=root/'stage2_predecessor'
    preparation=read_json(namespace/'PREPARATION.json');check_current_scope(preparation,root,current_manifest)
    path=namespace/'COMPATIBILITY.json';certificate=read_json(path);check_seal(certificate,'certificate_digest')
    if (certificate.get('kind')!='stage2_compatible_predecessor' or certificate.get('compatible_stage2') is not True
            or certificate.get('preparation_digest')!=preparation['preparation_digest']
            or certificate.get('current_manifest_digest')!=current_manifest['manifest_digest']
            or certificate.get('current_source_sha256')!=sources() or certificate.get('changed_files')!=['run_experiment.py']):
        raise ValueError('Certificate does not establish compatibility with this current round/runtime')
    for name in ('predecessor_root','imported_root','predecessor_manifest_digest','reference_digest','stage2_jobs_sha256',
                 'predecessor_source_sha256','selected_checkpoint','selected_model_sha256','selected_value_sha256','imported_files'):
        if certificate.get(name)!=preparation[name]:raise ValueError(f'Compatibility certificate altered {name}')
    equivalence=certificate['validation_equivalence']
    if equivalence['expected_trials']!=40 or tuple(equivalence['checkpoints'])!=CHECKPOINTS or set(equivalence['reports'])!=set(CHECKPOINTS):
        raise ValueError('All four Stage2 checkpoints and forty validation runs are required')
    current_reports={}
    for checkpoint in CHECKPOINTS:
        record=equivalence['reports'][checkpoint]
        expected_previous=Path(preparation['imported_root'])/'trials/stage2/validation'/f'selection_{checkpoint}'/'summary.json'
        expected_current=root/'trials/stage2/validation'/f'compatibility_{checkpoint}'/'summary.json'
        if (Path(record['predecessor_summary']).resolve()!=expected_previous.resolve()
                or Path(record['current_summary']).resolve()!=expected_current.resolve()
                or file_hash(expected_previous)!=record['predecessor_summary_sha256']
                or file_hash(expected_current)!=record['current_summary_sha256']):
            raise ValueError('A certified validation summary changed or moved outside its declared namespace')
        previous,current=read_json(expected_previous),read_json(expected_current)
        checked=compare_validation_reports(previous,current,current_manifest,sources())
        if checked!=record['trials'] or record['compared_trials']!=10 or record['exact_behavior_matches']!=10:
            raise ValueError('Certified behavior hashes or counts no longer match')
        current_reports[checkpoint]=current
    imported=Path(preparation['imported_root']);predecessor_manifest=read_json(imported/'manifest.json');validate_manifest(predecessor_manifest)
    if predecessor_manifest['manifest_digest']!=preparation['predecessor_manifest_digest']:
        raise ValueError('Predecessor manifest changed')
    return dict(certificate=certificate,certificate_path=path,imported_root=imported,
                predecessor_manifest=predecessor_manifest,current_validation_reports=current_reports,
                selected_checkpoint=certificate['selected_checkpoint'],selected_model_sha256=certificate['selected_model_sha256'],
                selected_value_sha256=certificate['selected_value_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare');prepare.add_argument('--predecessor',type=Path,required=True)
    replay=sub.add_parser('revalidate');replay.add_argument('--workers',type=int,default=5)
    sub.add_parser('verify');args=parser.parse_args()
    if args.command=='prepare':
        p=prepare_stage2_predecessor(args.predecessor,args.root)
        print(json.dumps({'imported_files':len(p['imported_files']),'preparation_digest':p['preparation_digest'],'selected_checkpoint':p['selected_checkpoint']}))
    else:
        verified=(revalidate_stage2_predecessor(args.root,workers=args.workers) if args.command=='revalidate' else
                  verify_stage2_predecessor(args.root,read_json(args.root/'manifest.json')))
        print(json.dumps({'certificate':str(verified['certificate_path']),'selected_checkpoint':verified['selected_checkpoint'],
                          'selected_model_sha256':verified['selected_model_sha256'],'selected_value_sha256':verified['selected_value_sha256'],
                          'exact_validation_matches':40,'no_final_tests_run':True}))


if __name__=='__main__':main()
