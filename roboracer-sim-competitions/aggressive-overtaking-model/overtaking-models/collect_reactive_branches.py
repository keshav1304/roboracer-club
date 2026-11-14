"""Short exogenous ego-motion trials against the unchanged reactive Stage 2 rival.

Only the ego's training motion is predefined; every opponent runs avoidance.
Initial state is shared across proposed motions. Groups, not time samples, split.
"""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/reactive_branches_v2'

def trial(case):
    destination=OUT/f"case-{case['id']:03d}.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite collected trial: {destination}")
    from world import make_env,make_stage2_opponent,body_polygon
    from track_geometry import TrackGeometry,PathTracker
    from perception import LidarTracker
    from maneuver_core import delta_s,smoothstep
    from learned_opponent import features
    from benchmark import offline_only
    env=make_env(seed=case['seed']);g=TrackGeometry(env.unwrapped.track);rows=[];ys=[];records=[]
    with offline_only():
        for move in case['moves']:
            poses=[]
            for s,d in ((case['s'],case['d']),(case['s']+case['rel'],0.)):
                xy,yaw,_=g.sample(s,d);poses.append(np.r_[xy,yaw])
            poses=np.array(poses)
            if any(g.body_clearance(p[:2],p[2])<.08 for p in poses):continue
            obs,_=env.reset(options={'poses':poses})
            if body_polygon(obs['agent_0']).distance(body_polygon(obs['agent_1']))<.15:continue
            for agent in env.unwrapped.sim.agents:agent.state[3]=case['speed']
            obs,*_=env.step(np.array([[0.,case['speed']],[0.,case['speed']]]))
            angles=env.unwrapped.sim.agents[0].scan_angles
            rival=make_stage2_opponent(g,angles,2.2,obs['agent_1'])
            ego=PathTracker(g,lateral_acceleration=2.4);sensor=LidarTracker(g,angles)
            collision=False
            # Actual moving sensor history. The rival controller receives no
            # fabricated target or externally dictated avoidance action.
            for k in range(60):
                if k%5==0:
                    sensor.update(obs['agent_0'],dt=.05)
                    a=ego.control(obs['agent_0'],offset=case['d'],vmax=case['speed'],dt=.05)
                    b=rival.plan(obs['agent_1'],dt=.05)
                obs,*_=env.step(np.array([a,b]))
                if any(obs[f'agent_{i}']['collision'] for i in (0,1)) or body_polygon(obs['agent_0']).distance(body_polygon(obs['agent_1']))<.001:
                    collision=True;break
            if collision:records.append(dict(move=move,collision=True,phase='warmup'));continue
            target=sensor.target
            if target is None or target.confidence<.6:continue
            initial=dict(obs['agent_0']);es,ed=map(float,g.project(float(initial['pose_x']),float(initial['pose_y'])))
            def offsets(ss):
                return ed+(move['offset']-ed)*smoothstep(delta_s(es,ss,g.length)/move['entry'])
            trace=[];trace.append([0.,*[float(obs[f'agent_{i}'][key]) for i in (0,1) for key in ('pose_x','pose_y','pose_theta','linear_vel_x')]])
            for k in range(151):
                if k%5==0:
                    a=ego.control(obs['agent_0'],offset=offsets,vmax=2.4,dt=.05)
                    b=rival.plan(obs['agent_1'],dt=.05)
                obs,*_=env.step(np.array([a,b]))
                if k%5==4:trace.append([(k+1)*.01,*[float(obs[f'agent_{i}'][key]) for i in (0,1) for key in ('pose_x','pose_y','pose_theta','linear_vel_x')]])
                if any(obs[f'agent_{i}']['collision'] for i in (0,1)) or body_polygon(obs['agent_0']).distance(body_polygon(obs['agent_1']))<.001:
                    collision=True;break
            tr=np.array(trace);tt=tr[:,0]
            if len(tt)<6:records.append(dict(move=move,collision=collision,phase='maneuver',trace=trace));continue
            pe,pd=g.project(tr[:,1],tr[:,2]);po,od=g.project(tr[:,5],tr[:,6])
            pe=es+delta_s(es,pe,g.length);po=target.s+delta_s(target.s,po,g.length)
            h=np.array([.25,.5,.75,1.,1.5]);h=h<tt[-1]-.01;h=np.array([.25,.5,.75,1.,1.5])[h]
            if not len(h):continue
            x=features(g,target.s,target.d,target.speed,es,ed,float(initial['linear_vel_x']),float(initial['pose_theta']),h,
                np.interp(h/2,tt,pe),np.interp(h/2,tt,pd),np.interp(h,tt,pe),np.interp(h,tt,pd))
            y=np.column_stack([delta_s(target.s,np.interp(h,tt,po),g.length)-target.speed*h,np.interp(h,tt,od)-target.d])
            rows.extend(x.tolist());ys.extend(y.tolist())
            records.append(dict(move=move,collision=collision,phase='maneuver',trace=trace))
    env.close()
    data=dict(case=case,X=rows,y=ys,trials=records)
    with destination.open("x") as handle:
        handle.write(json.dumps(data,separators=(',',':')))
    return len(rows),len(records),sum(r['collision'] for r in records)

def main():
    OUT.mkdir(exist_ok=False);rng=np.random.default_rng(846721);cases=[]
    for i in range(160):
        s=(i+rng.uniform(.05,.95))*59.029902/160
        d=float(rng.choice([-.5,-.4,0.,.4,.5]));rel=float(rng.uniform(-.5,1.8))
        if d==0. and abs(rel)<.85:rel=.95
        moves=[dict(offset=float(z),entry=float(rng.choice([1.8,2.6,3.5]))) for z in rng.choice([-.5,-.35,0.,.35,.5],3,replace=False)]
        cases.append(dict(id=i,s=s,d=d,rel=rel,speed=float(rng.uniform(1.4,2.2)),seed=8467210+i,moves=moves,split='validation' if i%5==0 else 'train',opponent='reactive'))
    (OUT/'manifest.json').write_text(json.dumps(dict(cases=cases,all_opponents_reactive=True,exogenous_ego_maneuvers=True),indent=2))
    totals=np.zeros(3,int)
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,f in enumerate(as_completed([pool.submit(trial,c) for c in cases])):
            totals+=f.result()
            if i%20==19:print('completed states',i+1,'rows / maneuvers / contacts',totals.tolist(),flush=True)
    print('finished',totals.tolist(),flush=True)
if __name__=='__main__':main()
