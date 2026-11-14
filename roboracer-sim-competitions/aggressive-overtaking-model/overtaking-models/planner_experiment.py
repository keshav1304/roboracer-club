"""Compare expanded ego planning in F1TENTH Gym. Writes JSON only, no HTML."""
import argparse,json,hashlib,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(job,out):
    from benchmark import offline_only
    import run_experiment as experiment
    from expanded_planner import ExpandedStage2,ExpandedStage4
    if job['expanded']:experiment.MODELS.update({2:ExpandedStage2,3:ExpandedStage4})
    else:
        from moving_model import MovingModel
        from interactive_model import InteractiveModel
        experiment.MODELS.update({2:MovingModel,3:InteractiveModel})
    with offline_only():r,frames,_,_=experiment.episode(experiment.Scenario(**job['scenario']),control_dt=.05)
    r.update(expanded_planner=job['expanded'],experiment_stage=job['stage'],
        extra_source_sha256={name:sha(ROOT/name) for name in ('expanded_planner.py','planner_experiment.py')},
        network_access='blocked',baseline_trace=job.get('baseline_trace'))
    Path(out,job['id']+'.trace.json').write_text(json.dumps({'result':r,'frames':frames},separators=(',',':'))+'\n')
    print(json.dumps({'id':job['id'],'outcome':r['outcome'],'pass_time':r['passed_at'],
        'controller_p95_ms':r['controller_latency'].get('p95_ms')}),flush=True)
    return r

def summarize(out):
    manifest=json.loads((out/'manifest.json').read_text());groups={}
    for stage in (2,4):
        pairs=[]
        for job in manifest['jobs']:
            if job['stage']!=stage or not job['expanded']:continue
            new=json.loads((out/(job['id']+'.trace.json')).read_text())['result']
            oldpath=Path(job['baseline_trace']) if job.get('baseline_trace') else out/(job['id'].replace('-expanded','-baseline')+'.trace.json')
            old=json.loads(oldpath.read_text())['result']
            for name in ('model_sha256','value_model_sha256','opponent_model_sha256','opponent_value_model_sha256','source_sha256'):
                assert new[name]==old[name],name
            pairs.append({'id':job['id'],'baseline_outcome':old['outcome'],'expanded_outcome':new['outcome'],
                'baseline_pass_time':old['passed_at'],'expanded_pass_time':new['passed_at'],
                'baseline_controller_p95_ms':old['controller_latency']['p95_ms'],
                'expanded_controller_p95_ms':new['controller_latency']['p95_ms']})
        groups[str(stage)]={'trials':len(pairs),'baseline_passes':sum(p['baseline_outcome']=='pass' for p in pairs),
            'expanded_passes':sum(p['expanded_outcome']=='pass' for p in pairs),
            'baseline_contacts':sum(p['baseline_outcome']=='collision' for p in pairs),
            'expanded_contacts':sum(p['expanded_outcome']=='collision' for p in pairs),'pairs':pairs}
    (out/'summary.json').write_text(json.dumps({'stages':groups,'models_retrained':False,'api_calls':0,
        'note':'Paired regression comparison of existing encounters; not a new independent test set. Timing measured under concurrent collection.'},indent=2)+'\n')
    print(json.dumps({s:{k:v for k,v in d.items() if k!='pairs'} for s,d in groups.items()},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--development-baselines',type=Path)
    p.add_argument('--stage',choices=('2','4','all'),default='all')
    p.add_argument('--case-contains',default='')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise RuntimeError('Refusing to overwrite an existing experiment.')
    jobs=[]
    if a.development:
        manifest=json.loads((ROOT/'artifacts/rollout_final_20261107/manifest.json').read_text())
        for stage,case in ((2,0),(2,12),(4,22),(4,46)):
            internal=2 if stage==2 else 3
            original=next(j for j in manifest['jobs'] if j['stage']==internal and j['split']=='train' and j['trial_id'].split('-')[2]==f'{case:03d}')
            scenario=dict(original['scenario'])
            if stage==4:scenario['opponent_speed']=2.1
            for expanded in (False,True):
                jobs.append(dict(id=f'stage{stage}-dev{case:03d}-'+('expanded' if expanded else 'baseline'),stage=stage,scenario=scenario,expanded=expanded))
    else:
        locations=[(2,ROOT/'artifacts/rollout_final_20261107/trials/stage2/test/neural'),(4,ROOT/'artifacts/stage4_45s_20261107')]
        for stage,directory in locations:
            for path in sorted(directory.glob('*.trace.json')):
                r=json.loads(path.read_text())['result']
                if stage==4 and r['group']!='reactive':continue
                scenario=r.get('harness_scenario',r['scenario'])
                jobs.append(dict(id=f'stage{stage}-'+path.stem.replace('.trace','')+'-expanded',stage=stage,
                    scenario=scenario,expanded=True,baseline_trace=str(path)))
    if a.development_baselines:
        if not a.development:raise ValueError('Baseline reuse requires development mode')
        jobs=[{**j,'baseline_trace':str((a.development_baselines/(j['id'].replace('-expanded','-baseline')+'.trace.json')).resolve())} for j in jobs if j['expanded']]
    jobs=[j for j in jobs if (a.stage=='all' or j['stage']==int(a.stage)) and a.case_contains in j['id']]
    if not jobs:raise ValueError('No matching scenarios')
    frozen=out/'source';frozen.mkdir()
    for name in ('expanded_planner.py','planner_experiment.py'):shutil.copy2(ROOT/name,frozen/name)
    (out/'manifest.json').write_text(json.dumps({'jobs':jobs,'development':a.development,
        'extra_source_sha256':{name:sha(frozen/name) for name in ('expanded_planner.py','planner_experiment.py')}},indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(run,j,str(out)) for j in jobs]):future.result()
    summarize(out)

if __name__=='__main__':main()
