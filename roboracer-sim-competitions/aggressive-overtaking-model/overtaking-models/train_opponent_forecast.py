"""Fit/validate a conditional residual MLP on saved reactive Gym encounters."""
import json,hashlib
from pathlib import Path
import numpy as np
from learned_opponent import FEATURES,features,OpponentForecast
from maneuver_core import delta_s
from world import make_env
from track_geometry import TrackGeometry
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/learned_reaction_v1'

def dataset(g,split):
    files=sorted((ROOT/f'artifacts/rollout_final_20261107/trials/stage3/{split}/teacher').glob('*.trace.json'))
    rival=hashlib.sha256((ROOT/'artifacts/models/stage2.npz').read_bytes()).hexdigest()
    xs=[];ys=[];groups=[];paths=[]
    for path in files:
        data=json.loads(path.read_text());r=data['result'];f=data['frames']
        if r.get('opponent_model_sha256')!=rival or r['scenario']['opponent']!='reactive' or len(f)<25:continue
        t=np.array([v['t'] for v in f]);ego=np.array([v['ego'] for v in f]);other=np.array([v['opponent'] for v in f])
        es,ed=g.project(ego[:,0],ego[:,1]);os,od=g.project(other[:,0],other[:,1])
        eu=np.unwrap(es*2*np.pi/g.length)*g.length/(2*np.pi)
        ou=np.unwrap(os*2*np.pi/g.length)*g.length/(2*np.pi)
        for i in range(0,len(f)-20,5):
            target=f[i].get('target')
            if not target or target['confidence']<.6 or abs(target['distance'])>4.:continue
            horizons=np.array([.25,.5,.75,1.,1.5]);horizons=horizons[t[i]+horizons<t[-1]-.01]
            if not len(horizons):continue
            so=float(es[i]+target['distance']);do=float(target['d']);vo=float(target['speed'])
            x=features(g,so,do,vo,float(es[i]),float(ed[i]),float(ego[i,3]),float(ego[i,2]),horizons,
                np.interp(t[i]+horizons/2,t,eu),np.interp(t[i]+horizons/2,t,ed),
                np.interp(t[i]+horizons,t,eu),np.interp(t[i]+horizons,t,ed))
            future_s=np.interp(t[i]+horizons,t,ou);future_d=np.interp(t[i]+horizons,t,od)
            y=np.column_stack([delta_s(so,future_s,g.length)-vo*horizons,future_d-do]).astype(np.float32)
            if not np.isfinite(x).all() or not np.isfinite(y).all():continue
            xs.extend(x);ys.extend(y);groups.extend([r['scenario']['seed']]*len(x))
        paths.append(str(path.relative_to(ROOT)))
    return np.asarray(xs),np.asarray(ys),np.asarray(groups),paths

def fit(x,y,vx,vy,seed,epochs=100,initial=None,learning_rate=.0015):
    rng=np.random.default_rng(seed);dims=(x.shape[1],48,32,2)
    params=[]
    for a,b in zip(dims[:-1],dims[1:]):params.extend([(rng.standard_normal((a,b))*np.sqrt(2/a)).astype(np.float32),np.zeros(b,np.float32)])
    if initial is not None:params=[a.copy() for a in initial]
    moment=[np.zeros_like(a) for a in params];variance=[np.zeros_like(a) for a in params]
    best=None;lossbest=np.inf;step=0
    for epoch in range(epochs):
        for idx in np.array_split(rng.permutation(len(x)),max(1,len(x)//256)):
            a=x[idx];w0,b0,w1,b1,w2,b2=params
            h0=np.maximum(a@w0+b0,0);h1=np.maximum(h0@w1+b1,0);pred=h1@w2+b2
            grad=np.clip(pred-y[idx],-2.,2.)/len(idx)
            dw2=h1.T@grad+1e-5*w2;db2=grad.sum(0);g1=(grad@w2.T)*(h1>0)
            dw1=h0.T@g1+1e-5*w1;db1=g1.sum(0);g0=(g1@w1.T)*(h0>0)
            grads=[a.T@g0+1e-5*w0,g0.sum(0),dw1,db1,dw2,db2];step+=1
            for p,g,m,v in zip(params,grads,moment,variance):
                m[:]=.9*m+.1*g;v[:]=.999*v+.001*g*g;p-=learning_rate*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        vpred=np.maximum(np.maximum(vx@params[0]+params[1],0)@params[2]+params[3],0)@params[4]+params[5]
        loss=float(np.mean((vpred-vy)**2))
        if loss<lossbest:lossbest=loss;best=[p.copy() for p in params]
    return best,lossbest

def main():
    OUT.mkdir(exist_ok=False)
    e=make_env(seed=122);g=TrackGeometry(e.unwrapped.track)
    x,y,groups,paths=dataset(g,'train');vx,vy,vg,vpaths=dataset(g,'validation');e.close()
    assert not set(groups)&set(vg),'encounter split overlaps'
    mean=x.mean(0);scale=np.maximum(x.std(0),.05);yscale=np.array([.5,.2],np.float32)
    nx=(x-mean)/scale;nv=(vx-mean)/scale
    arrays=dict(mean=mean,scale=scale,yscale=yscale)
    for m in range(3):
        params,loss=fit(nx,y/yscale,nv,vy/yscale,23000+m)
        for name,value in zip(('w0','b0','w1','b1','w2','b2'),params):arrays[f'm{m}_{name}']=value
        print('member',m,'validation normalized MSE',loss,flush=True)
    metadata=dict(features=list(FEATURES),ensemble=3,architecture=[16,48,32,2],train_episodes=len(paths),validation_episodes=len(vpaths),
        train_encounters=len(set(groups)),validation_encounters=len(set(vg)),train_samples=len(x),validation_samples=len(vx),
        train_files=paths,validation_files=vpaths,reference_digest=g.reference_digest,opponent_model_sha256=hashlib.sha256((ROOT/'artifacts/models/stage2.npz').read_bytes()).hexdigest(),
        conditional_on_ego_trajectory=True,validation_used_for_checkpoint_selection=True,test_data_used=False,api_calls=0)
    arrays['metadata']=np.array(json.dumps(metadata));np.savez_compressed(OUT/'model.npz',**arrays)
    np.savez_compressed(OUT/'dataset.npz',X=x,y=y,groups=groups,validation_X=vx,validation_y=vy,validation_groups=vg)
    model=OpponentForecast();prediction,spread=model.predict(vx)
    report=dict(metadata=metadata,validation_rmse_m=np.sqrt(np.mean((prediction-vy)**2,axis=0)).tolist(),
        constant_velocity_rmse_m=np.sqrt(np.mean(vy**2,axis=0)).tolist(),
        validation_error_p90_m=np.quantile(np.abs(prediction-vy),.9,axis=0).tolist(),ensemble_spread_mean_m=spread.mean(0).tolist())
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='metadata'},flush=True)

if __name__=='__main__':main()
