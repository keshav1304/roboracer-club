"""Make portable dataset handoffs without credentials or changing source data."""
from pathlib import Path
import hashlib,json,re,shutil,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parent
EXPORT=ROOT/'exports'
READY=EXPORT/'roboracer_training_data_20261107'
ART=ROOT/'artifacts'
SECRET=re.compile(rb'sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}|Bearer\s+[A-Za-z0-9_.-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')

def digest(data):return hashlib.sha256(data).hexdigest()

def sanitize(data):
    if SECRET.search(data):raise RuntimeError('Credential-like content detected; archive creation halted.')
    return data.replace(str(ROOT).encode(),b'source://overtaking-models').replace(b'/Users/aichen',b'<LOCAL_HOME>')

def copy_text(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(sanitize(src.read_bytes()))

def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def save(stage,split,task,X,y,names,classes,provenance):
    assert X.ndim==2 and len(X)==len(y)==len(provenance) and X.shape[1]==len(names)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    dest=READY/split/f'stage{stage}';dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/f'{task}.npz',X=X,y=y,feature_names=np.array(names),classes=np.array(classes))
    (dest/f'{task}.rows.jsonl').write_bytes(sanitize(''.join(json.dumps(p)+'\n' for p in provenance).encode()))
    return dict(stage=stage,split=split,task=task,rows=len(X),features=X.shape[1],classes=list(classes),
                file=str((dest/f'{task}.npz').relative_to(READY)),label_format='soft probabilities' if y.ndim==2 else 'integer class index')

def main():
    from maneuver_core import FEATURE_NAMES,ACTIONS
    from rollout_training import successful_winners,build_arrays
    from train_models import mirror_examples
    READY.mkdir(parents=True,exist_ok=True)
    inventory=[]
    truth={r['id']:r for r in rows(ART/'datasets/stage1.truth.jsonl')}
    for split in ('train','validation'):
        with np.load(ART/f'datasets/stage1.{split}.npz',allow_pickle=False) as a:
            X,y=a['X'],a['y'];provenance=[{'id':str(i),'source':str(s),'scenario':truth[str(i)]['scenario'],
                'split':split,'synthetic_mirror':False} for i,s in zip(a['ids'],a['sources'])]
            if split=='train':
                mx,my=mirror_examples(X,y)
                X=np.vstack([X,mx]);y=np.vstack([y,my])
                provenance=provenance+[{**p,'synthetic_mirror':True} for p in provenance]
            inventory.append(save(1,split,'imitation',X,y,a['feature_names'].tolist(),a['classes'].tolist(),provenance))
    roots={2:ART/'rollout_final_20261106_r2',3:ART/'rollout_final_20261107'}
    for stage,root in roots.items():
        manifest=json.loads((root/'manifest.json').read_text())
        for split in ('train','validation'):
            collection=json.loads((root/f'trials/stage{stage}/{split}/teacher/summary.json').read_text())
            records=collection['trials'];byid={r['job']['trial_id']:r for r in records}
            winners={r['job']['trial_id'] for r in successful_winners(records)}
            provenance=rows(root/f'datasets/stage{stage}.{split}.provenance.jsonl')
            for p in provenance:p['case_id']=byid[p['trial_id']]['job']['case_id']
            ip=[{**p,'source':'successful_rollout_action'} for p in provenance if p['trial_id'] in winners]
            advice=root/f'datasets/stage{stage}.train.llm_advisory.provenance.jsonl'
            if split=='train':ip.extend(rows(advice))
            with np.load(root/f'datasets/stage{stage}.{split}.npz',allow_pickle=False) as a:
                # Reconstruct the non-LLM rows independently to verify ordering.
                rebuilt=build_arrays(records,manifest,stage,split)
                for key in ('value_X','value_y'):np.testing.assert_array_equal(a[key],rebuilt[key])
                n=len(rebuilt['imitation_X'])
                np.testing.assert_array_equal(a['imitation_X'][:n],rebuilt['imitation_X'])
                np.testing.assert_array_equal(a['imitation_y'][:n],rebuilt['imitation_y'])
                inventory.append(save(stage,split,'imitation',a['imitation_X'],a['imitation_y'],FEATURE_NAMES,ACTIONS[:4 if stage==3 else 3],ip))
                inventory.append(save(stage,split,'outcome',a['value_X'],a['value_y'],FEATURE_NAMES+[f'action_{a.lower()}' for a in ACTIONS],('FAIL','PASS'),provenance))
        copy_text(root/'manifest.json',READY/f'metadata/stage{stage}.manifest.json')
        copy_text(root/f'stage{stage}.training.json',READY/f'metadata/stage{stage}.training.json')
    copy_text(ART/'stage1.training.json',READY/'metadata/stage1.training.json')
    copy_text(ART/'datasets/stage1.truth.jsonl',READY/'metadata/stage1.original_snapshots.jsonl')
    for stage,root in roots.items():
        for f in (root/'datasets').glob(f'stage{stage}*'):
            dst=READY/'original_arrays'/f'stage{stage}'/f.name;dst.parent.mkdir(parents=True,exist_ok=True)
            if f.suffix=='.npz':shutil.copy2(f,dst)
            else:copy_text(f,dst)
    for f in (ART/'datasets').glob('stage1*.npz'):
        dst=READY/'original_arrays/stage1'/f.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
    # Holdout records have their own directory and are never opened by train.py.
    for stage in (2,3):
        for f in (ART/f'rollout_final_20261107/trials/stage{stage}/test').rglob('*.json'):
            copy_text(f,READY/'evaluation_only'/f'stage{stage}'/f.relative_to(ART/f'rollout_final_20261107/trials/stage{stage}/test'))
    for f in (ART/'stage4_20261107').glob('*.json'):
        copy_text(f,READY/'evaluation_only/stage4'/f.name)
    for f in (ART/'stage4_20261107').glob('*.trace.json'):
        copy_text(f,READY/'evaluation_only/stage4'/f.name)
    for f in ROOT.glob('*.py'):copy_text(f,READY/'reference_source'/f.name)
    copy_text(ROOT/'tiny_network.py',READY/'tiny_network.py')
    runtime=EXPORT/'stage2_20261107.zip'
    if runtime.exists():shutil.copy2(runtime,READY/'stage2_runtime_reference.zip')
    (READY/'requirements.txt').write_text('numpy==2.5.4\n')
    (READY/'DATASET_INDEX.json').write_text(json.dumps({'datasets':inventory,'notes':[
        'Stage1 train includes 667 originals plus 667 synthetic mirrored rows.',
        'Stage2/3 imitation includes 10 training-only LLM advice rows each.',
        'Stage1 uses an older reference. Do not mix it blindly with Stage2/3.',
        'Stage4 is evaluation only: no new training dataset exists.',
        'Current splits are disjoint within their own protocols; historical rounds overlap.',
        'Absolute local paths are redacted to source://overtaking-models and <LOCAL_HOME>.']},indent=2)+'\n')
    # Every retained structured artifact is included in the historical archive,
    # except billing/key-verification logs unrelated to model training.
    history=EXPORT/'roboracer_all_experiment_data_20261107.zip'
    index=[];excluded=[]
    with zipfile.ZipFile(history,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(ART.rglob('*')):
            if not f.is_file() or f.suffix not in ('.json','.jsonl','.npz'):continue
            if f.name.startswith(('credit_check','llm_smoke')):
                excluded.append(str(f.relative_to(ROOT)));continue
            raw=f.read_bytes();exported=sanitize(raw) if f.suffix!='.npz' else raw
            if f.suffix=='.npz':
                with np.load(f,allow_pickle=False) as arrays:
                    for k in arrays.files:
                        if arrays[k].dtype.kind in 'US':sanitize(str(arrays[k]).encode())
            rel=str(f.relative_to(ROOT));z.writestr(rel,exported)
            index.append({'path':rel,'bytes':len(exported),'original_sha256':digest(raw),'exported_sha256':digest(exported)})
            if len(index)%1000==0:print('Archived',len(index),'structured files',flush=True)
        for f in sorted(READY.glob('reference_source/*.py')):z.write(f,'reference_source/'+f.name)
        z.writestr('ARCHIVE_INDEX.json',json.dumps({'files':index,'excluded_unrelated_logs':excluded,
            'scope':'All retained JSON, JSONL and NPZ experiment artifacts; no media, environments, credentials or virtualenvs.',
            'warning':'Includes overlapping historical train, validation and test rounds. Do not concatenate or fit on everything.'},indent=2))
        z.writestr('READ_ME_FIRST.txt','Start with the companion roboracer_training_data_20261107.zip for ready-to-train arrays and a runnable trainer.\nThis archive preserves all retained structured experiment data, including failed and superseded rounds.\nKeep stages, reference versions, episodes and original train/validation/test partitions separate.\nOld evaluation data cannot become a new independent test after being used for development.\nNo credentials are included. Local text paths are redacted. NPZ arrays are original bytes.\nARCHIVE_INDEX.json maps each file to its original and exported SHA256 digest.\n')
    (READY/'HISTORICAL_ARCHIVE.json').write_text(json.dumps({'file':history.name,'bytes':history.stat().st_size,
        'files':len(index),'sha256':digest(history.read_bytes()),'excluded_logs':excluded},indent=2)+'\n')
    print(json.dumps({'ready_folder':str(READY),'historical_zip':str(history),'historical_files':len(index),'historical_bytes':history.stat().st_size,'datasets':inventory},indent=2),flush=True)

if __name__=='__main__':main()
