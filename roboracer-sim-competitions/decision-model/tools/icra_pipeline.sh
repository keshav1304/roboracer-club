#!/bin/bash
# Full ICRA 2026 pipeline: data → net + LightGBM → 2 DAgger rounds (3 s + 3 s branches,
# counterfactual weight 4) → 1000-race driving test on fresh scenarios.
# Run from decision-model/:  bash tools/icra_pipeline.sh 2>&1 | tee icra_pipeline.log
set -u   # no -e: a grep that filters out every line must not abort the run
PY=../gym-two-car-reactive/.venv/bin/python
MAP=icra2026-comp
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }

step "port track";            $PY tools/port_track.py $MAP
step "1. generate data";      $PY generate.py --map $MAP --episodes 1000 --workers 10 --out data/icra_v1 | tail -1
$PY dataset_stats.py data/icra_v1 | sed -n '1,12p'
step "2a. LightGBM";          $PY train_lgbm.py data/icra_v1 --out models/lgbm_state_icra.txt | grep -E "test:|trained"
step "2b. net (ICRA + Spielberg, from the Spielberg net)"
$PY train_nn.py data/icra_v1 --extra data/v2 --init models/race_decider_xs.pt --epochs 4 --batch 4096 --lr 1e-3 \
    --out models/race_decider_xs_icra.pt | grep -E "^\+|epoch|test:|aux"
step "3a. DAgger round 1";    $PY dagger.py --map $MAP --model models/race_decider_xs_icra.pt --episodes 300 --first-seed 1000 \
    --branch-s 3 --settle-s 3 --workers 10 --out data/icra_dag1 | tail -1
step "3b. fine-tune round 1"; $PY train_nn.py data/icra_v1 --extra data/v2 --extra data/icra_dag1 --init models/race_decider_xs_icra.pt \
    --cf-weight 4 --epochs 3 --batch 4096 --lr 7e-4 --out models/race_decider_xs_icra_d1.pt | grep -E "^\+|epoch|test:"
step "4a. DAgger round 2";    $PY dagger.py --map $MAP --model models/race_decider_xs_icra_d1.pt --episodes 300 --first-seed 1300 \
    --branch-s 3 --settle-s 3 --workers 10 --out data/icra_dag2 | tail -1
step "4b. fine-tune round 2"; $PY train_nn.py data/icra_v1 --extra data/v2 --extra data/icra_dag1 --extra data/icra_dag2 \
    --init models/race_decider_xs_icra_d1.pt --cf-weight 4 --epochs 3 --batch 4096 --lr 5e-4 \
    --out models/race_decider_xs_icra_d2.pt | grep -E "^\+|epoch|test:"
step "5. driving test: 1000 fresh ICRA races"
$PY eval_closed_loop.py --map $MAP --fresh-from 5000 --episodes 1000 --workers 10 \
    --model models/lgbm_state_icra.txt --model models/race_decider_xs_icra.pt \
    --model models/race_decider_xs_icra_d1.pt --model models/race_decider_xs_icra_d2.pt
step "done"
