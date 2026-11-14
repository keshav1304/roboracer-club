"""Record comparable early/mid/final policies, or watch final drivers live."""
import argparse
import json
from bootstrap import ARTIFACTS, MODELS
from run_experiment import Scenario, episode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["1","2","3","all"],default="all")
    p.add_argument("--checkpoints",action="store_true")
    p.add_argument("--render",action="store_true")
    args=p.parse_args()
    stages=[1,2,3] if args.stage=="all" else [int(args.stage)]
    profiles={1:Scenario(1,901,0.,3.,0.,2.4,0.,18.),
              2:Scenario(2,902,48.6645623858,1.6,0.,2.4,2.,35.),
              3:Scenario(3,903,46.,1.2,0.,2.4,2.4,35.,False,"reactive")}
    # Choose a representative confirmed pass from the current evaluation,
    # explicitly marked illustrative. If no pass exists, retain a failed case.
    for stage in stages:
        report = ARTIFACTS / f"revised_stage{stage}.json"
        if report.exists():
            records = json.loads(report.read_text()).get("results", [])
            valid = [r for r in records if r.get("success")]
            choices = sorted(valid, key=lambda r:r["passed_at"]) if valid else records
            if choices:
                profiles[stage] = Scenario(**choices[len(choices)//2]["scenario"])
    paths=[]
    for stage in stages:
        for checkpoint in (["early","mid","final"] if args.checkpoints else ["final"]):
            model=MODELS/(f"stage{stage}.npz" if checkpoint=="final" else f"stage{stage}.{checkpoint}.npz")
            result,frames,_,_=episode(profiles[stage],model_path=model,render=args.render)
            result["demonstration_note"]="Selected illustrative scenario; aggregate success rates come from the full evaluation suite"
            result["checkpoint"]=checkpoint
            output=ARTIFACTS/f"demo_stage{stage}_{checkpoint}.trace.json"
            output.write_text(json.dumps({"result":result,"frames":frames})+"\n")
            paths.append(output)
            print(f"Stage {stage} {checkpoint}: {result['outcome']}",flush=True)
    if not args.render:
        # replay.py CLI is also available for combining arbitrary test traces.
        import subprocess,sys
        subprocess.run([sys.executable,"replay.py",*[str(x) for x in paths],"--output",str(ARTIFACTS/"demonstrations.html"),
                        "--poster",str(ARTIFACTS/"demonstrations.png")],check=True)


if __name__=="__main__":
    main()
