"""Conditional opponent-motion regression; local NumPy inference, no API.

Future ego positions are the proposed path at runtime and the executed path in
training. Opponent future positions are labels only. Predictions are conditional
forecasts, not causal guarantees that an unexecuted proposal will be followed.
"""
from pathlib import Path
import json
import numpy as np
from maneuver_core import delta_s
ROOT=Path(__file__).resolve().parent
FEATURES=('opponent_speed','opponent_offset','ego_relative_s','ego_offset','ego_speed','ego_heading_error',
          'curve_now','curve_1m','curve_3m','left_width','right_width','horizon',
          'ego_half_s','ego_half_d','ego_end_s','ego_end_d')

def features(g,os,od,ov,es,ed,ev,eyaw,h,half_s,half_d,end_s,end_d):
    h=np.asarray(h);n=h.size
    _,refyaw,k=g.sample(np.array([os,os+1.,os+3.]));left,right=g.widths(os)
    heading=float(np.arctan2(np.sin(eyaw-g.sample(es)[1]),np.cos(eyaw-g.sample(es)[1])))
    base=np.array([ov,od,float(delta_s(os,es,g.length)),ed,ev,heading,*k,float(left),float(right)])
    return np.column_stack([np.repeat(base[None,:],n,axis=0),h,
        delta_s(os,half_s,g.length),half_d,delta_s(os,end_s,g.length),end_d]).astype(np.float32)

class OpponentForecast:
    def __init__(self,path=None):
        self.path=Path(path or ROOT/'artifacts/learned_reaction_v2/model.npz')
        with np.load(self.path,allow_pickle=False) as z:
            self.meta=json.loads(str(z['metadata']));self.mean=z['mean'];self.scale=z['scale'];self.yscale=z['yscale']
            self.models=[[z[f'm{m}_{name}'] for name in ('w0','b0','w1','b1','w2','b2')] for m in range(self.meta['ensemble'])]
        if self.meta['features']!=list(FEATURES):raise ValueError('Forecast feature schema mismatch')
    def predict(self,x):
        a=(np.asarray(x,np.float32)-self.mean)/self.scale
        values=[]
        for w0,b0,w1,b1,w2,b2 in self.models:
            values.append((np.maximum(np.maximum(a@w0+b0,0)@w1+b1,0)@w2+b2)*self.yscale)
        values=np.asarray(values)
        return values.mean(axis=0),values.std(axis=0)
    def trajectory(self,g,target,obs,path,path_times,times):
        es,ed=map(float,g.project(float(obs['pose_x']),float(obs['pose_y'])))
        ps,pd=g.project(path[:,0],path[:,1]);ps=es+delta_s(es,ps,g.length)
        h=np.maximum(.05,np.minimum(1.5,np.asarray(times)))
        x=features(g,target.s,target.d,target.speed,es,ed,float(obs['linear_vel_x']),float(obs['pose_theta']),h,
            np.interp(h/2,path_times,ps),np.interp(h/2,path_times,pd),np.interp(h,path_times,ps),np.interp(h,path_times,pd))
        residual,spread=self.predict(x)
        # Force the forecast to start at the measured state; no pose teleport.
        blend=np.minimum(1.,np.asarray(times)/.25)[:,None]
        residual*=blend;spread*=blend
        s=target.s+target.speed*np.asarray(times)+residual[:,0]
        d=target.d+residual[:,1]
        s=np.maximum.accumulate(s)
        xy,_,_=g.sample(s,d);v=np.gradient(xy,axis=0)
        yaw=np.arctan2(v[:,1],v[:,0])
        return xy,yaw,spread,d
