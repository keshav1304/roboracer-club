"""Small, predeclared offline Stage 4 study; preserves earlier models/results."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/stage4_45s_20261107'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(job, output=None):
    from benchmark import offline_only
    import run_experiment as experiment
    from aggressive_moving_model import AggressiveMovingModel
    from world import make_env
    from track_geometry import TrackGeometry
    experiment.MODELS[3]=AggressiveMovingModel
    with offline_only():
        env=make_env(seed=job['scenario']['seed'])
        try:
            geometry=TrackGeometry(env.unwrapped.track)
            scenario=experiment.Scenario(**job['scenario'])
            result,frames,_,_=experiment.episode(scenario,env=env,geometry=geometry,control_dt=.05)
            result['track_length_m']=geometry.length
            for frame in frames:
                for key in ('ego','opponent'):
                    s,d=geometry.project(*frame[key][:2])
                    frame[key+'_s']=s;frame[key+'_line_offset']=d
        finally: env.close()
    result.update(experiment_stage=4,controller_stage=3,harness_scenario=asdict(scenario),
        trial_id=job['id'],group=job['group'],network_access='blocked',new_training=False,
        stage4_source_sha256={name:sha(ROOT/name) for name in ('stage4_experiment.py','aggressive_moving_model.py')})
    # Stage 4 is the experiment label; preserve the exact Stage 3 harness input.
    result['scenario']={**result['scenario'],'stage':4}
    result['maximum_opponent_line_offset_m']=max(abs(f['opponent_line_offset']) for f in frames)
    detected=[f for f in frames if f['opponent_target'] is not None and abs(f['relative_progress'])<1.2]
    result['close_opponent_detection_frames']=len(detected)
    result['maximum_close_detected_opponent_offset_m']=max([abs(f['opponent_line_offset']) for f in detected],default=0.)
    path=Path(output or OUT)/(job['id']+'.trace.json')
    path.write_text(json.dumps({'result':result,'frames':frames},separators=(',',':'))+'\n')
    return {'id':job['id'],'outcome':result['outcome'],'passed_at':result['passed_at'],
            'max_opponent_offset':result['maximum_opponent_line_offset_m']}

def publish():
    from replay import load_replay_data,HTML
    import render_preview as preview
    plan=json.loads((OUT/'manifest.json').read_text())
    traces=[json.loads((OUT/(j['id']+'.trace.json')).read_text()) for j in plan['jobs']]
    reactive=[t for t in traces if t['result']['group']=='reactive']
    controls=[t for t in traces if t['result']['group']=='line_control']
    metrics=[]
    for cap in (2.1,2.2):
        cases=[t['result'] for t in reactive if t['result']['scenario']['opponent_speed']==cap]
        metrics.append({'opponent_cap':cap,'trials':len(cases),'passes':sum(r['success'] for r in cases),
            'contacts':sum(r['collision'] is not None for r in cases),
            'pass_times':[r['passed_at'] for r in cases if r['success']]})
    comparisons=[]
    for baseline in controls:
        r=baseline['result'];key=r['scenario']['s']
        treated=next(t for t in reactive if t['result']['scenario']['s']==key and t['result']['scenario']['opponent_speed']==2.1)
        # Compare at matching track positions, not unmatched elapsed times.
        rows=baseline['frames'];length=r['track_length_m']
        unwrap=lambda ss:np.unwrap(np.asarray(ss)*2*np.pi/length)*length/(2*np.pi)
        bs=unwrap([f['opponent_s'] for f in rows]);bd=np.array([f['opponent_line_offset'] for f in rows])
        fs=unwrap([f['opponent_s'] for f in treated['frames']]);fs+=round((bs[0]-fs[0])/length)*length
        changes=[]
        for f,s in zip(treated['frames'],fs):
            if f['opponent_target'] is None or abs(f['relative_progress'])>=1.2 or not bs[0]<=s<=bs[-1]:continue
            reference_d=float(np.interp(s,bs,bd))
            away=(f['opponent_line_offset']-reference_d)*np.sign(f['opponent_line_offset']-f['ego_line_offset'])
            changes.append(float(away))
        comparisons.append({'start_s':key,'reactive_pass':treated['result']['success'],
            'line_control_pass':r['success'],'matched_close_detected_frames':len(changes),
            'max_additional_away_offset_m':max(changes) if changes else None,
            'note':'Paired observation of displacement; changing the rival also changes the ego trajectory, so this is not isolated causal proof.'})
    summary={'description':'Stage 3 aggressive weights versus slower, independently reactive Stage 2 rival',
        'new_training':False,'api_calls':0,'reactive_trials':len(reactive),'line_control_trials':len(controls),
        'groups':metrics,'paired_comparisons':comparisons,'all_trials_retained':True,
        'passes':sum(t['result']['success'] for t in reactive),'contacts':sum(t['result']['collision'] is not None for t in reactive),
        'line_control_passes':sum(t['result']['success'] for t in controls),
        'opponent_never_tracked_ego_trials':sum(not any(f['opponent_target'] is not None for f in t['frames']) for t in reactive),
        'any_ego_center_ahead_trials':sum(any(f['relative_progress']>0 for f in t['frames']) for t in reactive),
        'report_generator_sha256':sha(__file__),
        'passing_window_seconds':plan['jobs'][0]['scenario']['seconds'],
        'limitation':'Small exploratory study, not independent model qualification or proof of forced yielding.'}
    if plan.get('comparison_directory'):
        earlier=Path(plan['comparison_directory'])
        pairs=[]
        for t in traces:
            current=t['result']
            old=json.loads((earlier/(current['trial_id']+'.trace.json')).read_text())
            prefix=t['frames'][:len(old['frames'])]
            fields=('ego','opponent','command','opponent_command','probabilities','relative_progress')
            same=len(prefix)==len(old['frames']) and all(
                a['t']==b['t'] and a['action']==b['action'] and a['opponent_action']==b['opponent_action']
                and all(np.allclose(a[k],b[k],rtol=0,atol=1e-9) for k in fields)
                for a,b in zip(prefix,old['frames']))
            pairs.append({'trial_id':current['trial_id'],'group':current['group'],
                'old_outcome':old['result']['outcome'],'new_outcome':current['outcome'],
                'old_passed_at':old['result']['passed_at'],'new_passed_at':current['passed_at'],
                'matching_original_trajectory_prefix':same})
        summary['duration_comparison']={'previous_directory':str(earlier),'pairs':pairs,
            'all_original_trajectory_prefixes_match':all(p['matching_original_trajectory_prefix'] for p in pairs)}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Show up to four representative reactive cases, including a failure if any.
    selected=[]
    successes=sorted([t for t in reactive if t['result']['success']],key=lambda t:t['result']['passed_at'])
    failures=[t for t in reactive if not t['result']['success']]
    candidates=successes[:1]+failures[:1]+sorted(reactive,key=lambda t:-t['result']['maximum_close_detected_opponent_offset_m'])
    for t in candidates:
        path=OUT/(t['result']['trial_id']+'.trace.json')
        if path not in selected:selected.append(path)
        if len(selected)==4:break
    data=load_replay_data(selected,[])
    encoded=json.dumps(data,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    duration=summary['passing_window_seconds']
    page=HTML.replace('__DATA__',encoded).replace('Three stages. One track.',f'Stage 4 · {duration:g}-second passing window.')
    page=page.replace('Watch recorded simulation runs, inspect decisions, and compare training checkpoints.',
        'Frozen aggressive model against a slower reactive rival. Eight encounters; four matched line-only comparisons. No new training.')
    page=page.replace("3:'Equal-cap stage 2 neural rival'","3:'Equal-cap stage 2 neural rival',4:'Slower reactive stage 2 rival'")
    page=page.replace('s.stage===3','s.stage>=3')
    rows=''.join(f"<tr><td>{m['opponent_cap']:.1f} m/s</td><td>{m['passes']}/{m['trials']}</td><td>{m['contacts']}</td><td>{', '.join(f'{x:.2f}s' for x in m['pass_times']) or '—'}</td></tr>" for m in metrics)
    intro=f'''<section class="card" style="padding:18px;margin-bottom:18px"><h2>All eight reactive encounters</h2><p>Our cap: 2.4 m/s. Opponent: actual Stage 2 neural controller using its own LiDAR. Both ordinarily follow the same reference. No scripted displacement or contact is required. These are speed caps, not constant speeds.</p><table><tr><th>Opponent cap</th><th>Completed passes</th><th>Contacts</th><th>Pass times</th></tr>{rows}</table><p>Four extra runs use a line-only opponent for comparison. All twelve records are saved. Four representative reactive clips appear below. This is an exploratory configuration of existing weights, not a newly trained Stage 4 network.</p><p><a href="summary.json">Full summary and displacement comparisons</a> · <a href="../replay.html">Previous Stage 2/3 results</a></p></section>'''
    page=page.replace('<div id="trainingBanner" class="trainingBanner"></div>',intro+'<div id="trainingBanner" class="trainingBanner"></div>')
    extra=''
    if plan.get('comparison_directory'):
        old_summary=json.loads((Path(plan['comparison_directory'])/'summary.json').read_text())
        extra=f"<p>Same starting scenarios and frozen models: original 18-second results {old_summary['passes']}/8 passes, {old_summary['contacts']} contacts; extended results {summary['passes']}/8 passes, {summary['contacts']} contacts. These are paired reruns, not eight new independent encounters. <a href='../{Path(plan['comparison_directory']).name}/replay.html'>Original recordings</a>.</p>"
    finding=f'''<section class="card" style="padding:18px;margin-bottom:18px"><h2>What happened</h2><p>The opportunity window is {duration:g} seconds, with up to two further seconds to confirm a completed pass. A run ends earlier on contact or after a confirmed pass. {summary['passes']}/8 completed passes; {summary['contacts']} contacts. {summary['any_ego_center_ahead_trials']}/8 runs briefly got our car's centre ahead.</p><p>{summary['opponent_never_tracked_ego_trials']}/8 reactive encounters had no tracked car in the opponent's perception. The four line-only comparisons completed {summary['line_control_passes']}/4 passes. Departures from the line alone are not evidence of yielding; see the recorded paired displacement comparisons.</p>{extra}</section>'''
    page=page.replace(intro,intro+finding)
    (OUT/'replay.html').write_text(page)
    # Add a discoverable link without replacing historical results.
    old=ROOT/'artifacts/replay.html';existing=old.read_text()
    marker='id="stage4-link"'
    link=f'<p id="stage4-link"><a href="{OUT.name}/replay.html">Stage 4 — {duration:g}-second passing window</a></p>'
    if marker not in existing:existing=existing.replace('</header>','</header>'+link,1)
    else:
        import re
        existing=re.sub(r'<p id="stage4-link">.*?</p>',lambda _:link,existing,count=1)
    old.write_text(existing)
    preview.STAGE_NAMES[4]='Slightly slower reactive rival'
    images=[]
    for trace in data['traces'][:2]:
        times=np.array([f['t'] for f in trace['frames']])
        indices=np.unique(np.r_[np.searchsorted(times,np.arange(times[0],times[-1],.2),side='right')-1,len(times)-1])
        frames=[preview.render_frame(data,trace,int(i)) for i in indices]
        images.extend([frames[0]]*3+frames+[frames[-1]]*7)
    palette=[im.quantize(colors=128) for im in images]
    palette[0].save(OUT/'preview.gif',save_all=True,append_images=palette[1:],duration=100,loop=0,disposal=2)
    images[0].save(OUT/'preview.first.png');images[-1].save(OUT/'preview.last.png')
    print(json.dumps(summary,indent=2),flush=True)

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--publish-only',action='store_true')
    parser.add_argument('--seconds',type=float,default=45.)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--compare-with',type=Path)
    args=parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds<=.4:parser.error('seconds must be finite and greater than the setup interval')
    OUT=args.output.resolve()
    if args.publish_only:publish();return
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'manifest.json').exists():raise RuntimeError('Study already declared; refusing to overwrite trials.')
    jobs=[]
    for i,s in enumerate((4.,19.,34.,49.)):
        for j,cap in enumerate((2.1,2.2)):
            scenario=dict(stage=3,seed=41000+i*10+j*2,s=s,gap=1.25,ego_speed=2.4,
                          opponent_speed=cap,seconds=args.seconds,opponent='reactive',warmup_seconds=.4)
            jobs.append({'id':f'stage4-reactive-{i:02d}-{j}','group':'reactive','scenario':scenario})
            if j==0:jobs.append({'id':f'stage4-line-control-{i:02d}','group':'line_control',
                                'scenario':{**scenario,'opponent':'line'}})
    model_hashes={p.name:sha(p) for p in (ROOT/'artifacts/models').glob('stage[23]*.npz') if p.name in ('stage2.npz','stage2.value.npz','stage3.npz','stage3.value.npz')}
    if args.compare_with:
        previous=json.loads((args.compare_with/'manifest.json').read_text())
        if model_hashes!=previous['models']:raise RuntimeError('Models differ from the earlier study')
        jobs=[{**job,'scenario':{**job['scenario'],'seconds':args.seconds}} for job in previous['jobs']]
    import shutil
    frozen=OUT/'frozen_execution_source';frozen.mkdir(exist_ok=True)
    for name in ('stage4_experiment.py','aggressive_moving_model.py'):shutil.copy2(ROOT/name,frozen/name)
    (OUT/'manifest.json').write_text(json.dumps({'jobs':jobs,'models':model_hashes,'selection':'Four fixed sectors, two speed caps; no outcome-based parameter tuning',
        'new_training':False,'api_calls':0,'source':sha(__file__),
        'comparison_directory':str(args.compare_with.resolve()) if args.compare_with else None},indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,j,str(OUT)) for j in jobs]
        for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
    for name,digest in model_hashes.items():assert sha(ROOT/'artifacts/models'/name)==digest
    publish()

if __name__=='__main__':main()
