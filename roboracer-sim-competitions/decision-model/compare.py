"""Teacher state machine vs plain FTG ego on the same scenario grid.

Grid: opponent v_max 2/3/4/5 m/s × start gap 3/5/8 m, 60 s each, same seeds.
Reports, per ego: clean runs, runs with ≥1 overtake, mean time to first pass,
closest car-to-car gap, and the teacher's time in each state.

  python compare.py
  python compare.py --seconds 90 --speeds 3 4
"""
import argparse
import itertools
import warnings
from collections import Counter

import numpy as np

from run_teacher import race
from teacher import STATES

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0])
    ap.add_argument("--gaps", type=float, nargs="+", default=[3.0, 5.0, 8.0])
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    summary = {}
    for ego in ("ftg", "teacher"):
        rows, states = [], Counter()
        for v, g in itertools.product(args.speeds, args.gaps):
            res, _, _ = race(ego, seconds=args.seconds, opp_v=v, gap=g, verbose=False)
            rows.append((v, g, res))
            states.update({k: v2 for k, v2 in res["time_in_state"].items()})
            first = res["passes"][0] if res["passes"] else None
            print(f"{ego:7s} opp {v:.0f} m/s gap {g:.0f} m | passes {len(res['passes'])} first {first} | "
                  f"gap {res['car_gap']:.2f} m | collision {res['collision']}", flush=True)
        clean = sum(r["collision"] is None for _, _, r in rows)
        passed = [r for _, _, r in rows if r["passes"] and r["collision"] is None]
        summary[ego] = {
            "clean": f"{clean}/{len(rows)}",
            "clean passes": f"{len(passed)}/{len(rows)}",
            "mean first pass": f"{np.mean([r['passes'][0] for r in passed]):.2f} s" if passed else "-",
            "closest gap": f"{min(r['car_gap'] for _, _, r in rows):.2f} m",
            "states": states,
        }

    print("\n" + " " * 18 + "".join(f"{e:>16s}" for e in summary))
    for key in ("clean", "clean passes", "mean first pass", "closest gap"):
        print(f"{key:18s}" + "".join(f"{summary[e][key]:>16s}" for e in summary))
    tot = sum(summary["teacher"]["states"].values())
    print("\nteacher time in state: " + ", ".join(
        f"{s} {100 * summary['teacher']['states'][s] / tot:.1f}%" for s in STATES if summary["teacher"]["states"][s]))


if __name__ == "__main__":
    main()
