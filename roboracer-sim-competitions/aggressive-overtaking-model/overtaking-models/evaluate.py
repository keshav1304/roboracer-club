"""Seeded closed-loop evaluation with API network access disabled."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import socket
from unittest.mock import patch
import numpy as np

from bootstrap import ARTIFACTS
from run_experiment import Scenario, episode
from world import make_env
from track_geometry import TrackGeometry
from evaluation_metrics import passing_metrics, aggregate_metrics, TRAIN_START_SECTORS, RESERVED_START_SECTORS


def scenarios(stage, length, quick=False):
    # These START sectors are excluded from training and checkpoint selection.
    # Full closed-loop trials naturally continue through other track sectors.
    positions = [(sector + .40) * length/12 for sector in (1,5,9)]
    runs=[]
    for i,s in enumerate(positions):
        if stage==1:
            for j,lateral in enumerate((-.16,.16) if not quick else (0.,)):
                runs.append(Scenario(1,10000+i*10+j,s,2.6,lateral,2.4,0.,18.))
            runs.append(Scenario(1,11000+i,s,3.,0.,2.4,0.,12.,True))
        elif stage==2:
            for j,v in enumerate((1.6,2.,2.2) if not quick else (1.8,)):
                runs.append(Scenario(2,20000+i*10+j,s,1.6,0.,2.4,v,35.))
        else:
            for j,gap in enumerate((1.4,2.2) if not quick else (1.5,)):
                runs.append(Scenario(3,30000+i*10+j,s,gap,0.,2.4,2.4,35.,False,"reactive"))
            runs.append(Scenario(3,31000+i,s,1.4,0.,2.4,2.4,35.,False,"line"))
    return runs


def randomized_scenarios(stage, length, count, *, suite="heldout", seed=None):
    """Disjoint development/test starts, balanced across sectors and speed gaps."""
    sectors = TRAIN_START_SECTORS if suite == "tuning" else RESERVED_START_SECTORS
    seed = (20261116 if suite == "tuning" else 20261127) + stage if seed is None else seed
    rng = np.random.default_rng(seed)
    scenes = []
    for i in range(count):
        sector = sectors[i % len(sectors)]
        s = (sector + float(rng.uniform(.12, .85))) * length / 12
        gap = float(rng.uniform(2.3, 3.5) if stage == 1 else
                    rng.uniform(.9, 1.2) if stage == 3 and i % 2 == 0 else rng.uniform(1.2, 2.2))
        lateral = float(rng.uniform(-.2, .2)) if stage == 1 else 0.
        opponent_speed = (0. if stage == 1 else
                          float(np.clip((1.6, 2., 2.2)[(i + i // len(sectors)) % 3] + rng.uniform(-.06, .06), 1.5, 2.3))
                          if stage == 2 else 2.4)
        scenes.append(Scenario(stage, int(rng.integers(50000, 2**30)), s, gap, lateral,
                               2.4, opponent_speed, 18. if stage == 1 else 35.,
                               False, "reactive" if stage == 3 else "line"))
    return scenes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",type=int,choices=[1,2,3],required=True)
    p.add_argument("--quick",action="store_true")
    p.add_argument("--teacher",action="store_true")
    p.add_argument("--model",type=Path)
    p.add_argument("--output",default=None)
    p.add_argument("--random-trials",type=int,default=0)
    p.add_argument("--suite",choices=("regression", "tuning", "heldout"),default="regression",
                   help="Tuning uses only training start sectors; heldout uses reserved sectors with new seeds")
    p.add_argument("--seed",type=int)
    p.add_argument("--early-fraction",type=float,default=.5,
                   help="Report early passes completed within this fraction of a lap")
    args=p.parse_args()
    if args.random_trials < 0 or not 0 < args.early_fraction <= 1:
        p.error("Use nonnegative trial count and an early fraction in (0,1].")
    env=make_env(seed=9000+args.stage);g=TrackGeometry(env.unwrapped.track)
    prefix=args.output or f"evaluation_stage{args.stage}"
    results=[]
    traces=ARTIFACTS/"traces"; traces.mkdir(exist_ok=True,parents=True)
    def forbidden(*a,**k):
        raise RuntimeError("Network connections are forbidden during evaluation")
    scenes=scenarios(args.stage,g.length,args.quick)
    if args.random_trials or args.suite != "regression":
        scenes = randomized_scenarios(args.stage, g.length, args.random_trials or 12,
                                      suite=args.suite, seed=args.seed)
    try:
        with patch.object(socket.socket,"connect",forbidden),patch.object(socket.socket,"connect_ex",forbidden),patch.object(socket,"create_connection",forbidden):
            for i,scene in enumerate(scenes):
                res,frames,_,_=episode(scene,teacher=args.teacher,model_path=args.model,env=env,geometry=g)
                res["network_access"]="connections forbidden for entire evaluation"
                res["interaction_group"] = ("close" if scene.gap <= 1.2 else "approach") if scene.stage == 3 else "standard"
                res["passing_metrics"] = passing_metrics(res, frames, g, early_fraction=args.early_fraction)
                res["start_sector"] = int((scene.s % g.length) / g.length * 12)
                stem=f"{prefix}_{i:02d}"
                trace_path=traces/f"{stem}.trace.json"
                trace_path.write_text(json.dumps({"result":res,"frames":frames})+"\n")
                res["trace"]=str(trace_path.relative_to(ARTIFACTS))
                results.append(res)
                summary={"stage":args.stage,"trials":len(results),"requested_trials":len(scenes),
                         "evaluation_complete":len(results)==len(scenes),"passes":sum(r["success"] for r in results),
                         "collisions":sum(r["collision"] is not None for r in results),
                         "safe_blocked_stops":sum(r["safe_blocked_stop"] for r in results),
                         "start_sectors": list(TRAIN_START_SECTORS if args.suite == "tuning" else RESERVED_START_SECTORS),
                         "suite": args.suite, "training_track":"same fixed autodrive-default map",
                         "evaluation_role": ("development sweep; may inform tuning" if args.suite == "tuning" else
                                             "new-seed reserved-start probe; historical sector-level feedback means this is not an untouched track test" if args.suite == "heldout" else
                                             "reserved-sector regression suite; earlier runs informed controller fixes"),
                         "early_pass_threshold_track_fraction": args.early_fraction,
                         **aggregate_metrics(results),
                         "network_access":"blocked", "results":results}
                (ARTIFACTS/f"{prefix}.json").write_text(json.dumps(summary,indent=2)+"\n")
                print(f"Stage{args.stage} {i+1}: {res['outcome']} s={scene.s:.2f} gap={scene.gap} rival={scene.opponent_speed} clear={res['minimum_body_gap']:.3f} pass_t={res['passed_at']} early={res['passing_metrics']['early_pass']}",flush=True)
    finally:
        env.close()
    print(json.dumps({k:v for k,v in summary.items() if k!="results"},indent=2))


if __name__=="__main__":
    main()
