#!/usr/bin/env bash
# Train env_v2 policies on a RunPod CPU pod. Run from the repo root, or let the Dockerfile run it.
#
#   SEEDS="0 1 2" STEPS=40000000 bash pods/rl/runpod/train_runpod.sh
#
# Without Docker (a plain RunPod "Ubuntu + Python" CPU pod):
#   git clone -b claude/brave-cerf-mnebxb https://github.com/keshav1304/roboracer-club && cd roboracer-club
#   SETUP=1 bash pods/rl/runpod/train_runpod.sh
#
# Writes to /workspace (RunPod's persistent volume): track_cache/, runs/NAME/ (checkpoints, best_icra.zip,
# icra_eval.jsonl, cfg.json) and logs/. TensorBoard:  tensorboard --logdir /workspace/runs/tb --bind_all
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-40000000}
PRESET=${PRESET:-e2e_v2}
GEN=${GEN:-200}
HOLDOUT=${HOLDOUT:-icra}
NAME=${NAME:-$(date +%m%d)_${PRESET}}
OUT=${OUT:-/workspace}
NCPU=$(nproc)
NSEED=$(echo $SEEDS | wc -w)
WORKERS=${WORKERS:-$(( (NCPU - 1) / NSEED ))}
[ "$WORKERS" -lt 1 ] && WORKERS=1
export ENV_V2_TRACK_CACHE=${ENV_V2_TRACK_CACHE:-$OUT/track_cache}

if [ "${SETUP:-0}" = "1" ]; then
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -r pods/rl/runpod/requirements-v2.txt
  [ -d /opt/f1tenth_gym ] || git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym /opt/f1tenth_gym
  (cd /opt/f1tenth_gym && git checkout bdaec14 && pip install --no-deps -e .)
fi

cd pods/rl
mkdir -p "$OUT/logs" "$OUT/runs"
ln -sfn "$OUT/runs" train_v2/runs
echo "== $NCPU CPUs, seeds [$SEEDS], $WORKERS workers each, $STEPS steps, preset $PRESET, holdout $HOLDOUT"
echo "== checking the physics against AutoDRIVE first (env_v2/validate_v2.py)"
(cd env_v2 && python validate_v2.py | tee "$OUT/logs/validate_${NAME}.txt")
echo "== building $GEN generated tracks once"
python -c "import sys; sys.path.insert(0,'env_v2'); import tracks as T; [T.ensure_generated(i) for i in range($GEN)]"
for s in $SEEDS; do
  nohup python train_v2/train.py --name "${NAME}_s$s" --seed "$s" --steps "$STEPS" --workers "$WORKERS" \
        --preset "$PRESET" --gen "$GEN" --holdout "$HOLDOUT" > "$OUT/logs/${NAME}_s$s.log" 2>&1 &
  echo "seed $s -> $OUT/logs/${NAME}_s$s.log"
done
wait
for s in $SEEDS; do
  python train_v2/evaluate.py "train_v2/runs/${NAME}_s$s/best_icra.zip" | tee "$OUT/logs/${NAME}_s${s}_gates.txt" || true
done
