"""Adapt the forecasting model using independently specified ego motions."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
from learned_opponent import OpponentForecast,FEATURES
from train_opponent_forecast import fit
ROOT=Path(__file__).resolve().parent

def main():
    folder=ROOT/'artifacts/reactive_branches_v2';out=ROOT/'artifacts/learned_reaction_v2';out.mkdir(exist_ok=False)
    sets={s:[[],[],[]] for s in ('train','validation')};counts={s:0 for s in sets};contact=0;trials=0
    for path in sorted(folder.glob('case-*.json')):
        d=json.loads(path.read_text());split=d['case']['split'];trials+=len(d['trials']);contact+=sum(t['collision'] for t in d['trials'])
        if not d['X']:continue
        counts[split]+=1;sets[split][0].extend(d['X']);sets[split][1].extend(d['y']);sets[split][2].extend([d['case']['id']]*len(d['X']))
    x,y,groups=[np.asarray(z,np.float32) for z in sets['train']];vx,vy,vg=[np.asarray(z,np.float32) for z in sets['validation']]
    assert not set(groups)&set(vg)
    prior=OpponentForecast(ROOT/'artifacts/learned_reaction_v1/model.npz')
    priorp,_=prior.predict(vx)
    with np.load(ROOT/'artifacts/learned_reaction_v1/dataset.npz',allow_pickle=False) as z:
        rng=np.random.default_rng(1015);idx=rng.choice(len(z['X']),4000,replace=False)
        mixed_x=np.concatenate([z['X'][idx],np.repeat(x,5,axis=0)]);mixed_y=np.concatenate([z['y'][idx],np.repeat(y,5,axis=0)])
    arrays=dict(mean=prior.mean,scale=prior.scale,yscale=prior.yscale)
    for m in range(3):
        params,loss=fit((mixed_x-prior.mean)/prior.scale,mixed_y/prior.yscale,(vx-prior.mean)/prior.scale,vy/prior.yscale,
            seed=89111+m,epochs=160,initial=prior.models[m],learning_rate=.0005)
        for name,value in zip(('w0','b0','w1','b1','w2','b2'),params):arrays[f'm{m}_{name}']=value
        print('member',m,'validation MSE',loss,flush=True)
    meta={**prior.meta,'exogenous_motion_finetuning':True,'branch_train_states':counts['train'],'branch_validation_states':counts['validation'],
          'branch_train_rows':len(x),'branch_validation_rows':len(vx),'branch_maneuvers':trials,'branch_contacts':contact,
          'branch_split_by_initial_state':True,'all_branch_opponents_reactive':True}
    arrays['metadata']=np.array(json.dumps(meta));np.savez_compressed(out/'model.npz',**arrays)
    np.savez_compressed(out/'branch_dataset.npz',X=x,y=y,groups=groups,validation_X=vx,validation_y=vy,validation_groups=vg)
    model=OpponentForecast(out/'model.npz');pred,spread=model.predict(vx)
    report=dict(metadata=meta,validation_rmse_m=np.sqrt(np.mean((pred-vy)**2,axis=0)).tolist(),
        prior_model_rmse_m=np.sqrt(np.mean((priorp-vy)**2,axis=0)).tolist(),constant_motion_rmse_m=np.sqrt(np.mean(vy**2,axis=0)).tolist())
    (out/'report.json').write_text(json.dumps(report,indent=2))
    source=out/'source';source.mkdir()
    for name in ('learned_opponent.py','train_opponent_forecast.py','finetune_reactive_forecast.py','collect_reactive_branches.py'):shutil.copy2(ROOT/name,source/name)
    print({k:v for k,v in report.items() if k!='metadata'},flush=True)
if __name__=='__main__':main()
