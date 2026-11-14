#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/overtaking-models"
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
exec ../gym-two-car-reactive/.venv/bin/python ros_visualization/launch.py --stage 4 --speed .5 --response-experiment artifacts/learned_pressure_random_r1 "$@"
