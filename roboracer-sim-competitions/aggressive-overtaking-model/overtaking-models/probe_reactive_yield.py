"""Controlled pressure probes; every rival remains the original reactive model."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import collect_reactive_branches as collect
from world import make_env
from track_geometry import TrackGeometry

def probe_one(case):
    collect.OUT=Path("artifacts/reactive_pressure_probes")
    return collect.trial(case)

def main():
    collect.OUT=Path('artifacts/reactive_pressure_probes');collect.OUT.mkdir(exist_ok=True)
    env=make_env();g=TrackGeometry(env.unwrapped.track);cases=[]
    for i,s in enumerate(np.linspace(1.,g.length-1,16)):
        k=g.sample(s+np.linspace(0.,2.,5))[2];side=1 if k[np.argmax(abs(k))]>0 else -1
        cases.append(dict(id=i,s=float(s),d=side*.52,rel=.25,speed=1.8,seed=9567010+i,split='development',opponent='reactive',
            moves=[dict(offset=side*o,entry=entry) for o,entry in ((.52,1.),(.45,.8),(.40,.8),(0.,1.8))]))
    env.close();(collect.OUT/'manifest.json').write_text(json.dumps(dict(cases=cases,all_opponents_reactive=True),indent=2))
    with ProcessPoolExecutor(max_workers=2) as pool:
        for i,f in enumerate(as_completed([pool.submit(probe_one,c) for c in cases if not (collect.OUT/f"case-{c['id']:03d}.json").exists()])):print(i,f.result(),flush=True)
if __name__=='__main__':main()
