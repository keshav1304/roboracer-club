#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
"$PYTHON_BIN" -m venv gym-two-car-reactive/.venv
gym-two-car-reactive/.venv/bin/python -m pip install --upgrade pip
gym-two-car-reactive/.venv/bin/python -m pip install -r requirements-runtime.txt
# The pinned runtime supplies headless OpenCV; avoid installing the conflicting GUI build.
gym-two-car-reactive/.venv/bin/python -m pip install --no-deps -e gym-two-car-reactive/f1tenth_gym
gym-two-car-reactive/.venv/bin/python verify_package.py
if command -v pixi >/dev/null 2>&1; then
  (cd overtaking-models/ros_visualization && pixi install --locked)
else
  echo 'Gym is ready. Install Pixi (https://pixi.sh) and run pixi install --locked in overtaking-models/ros_visualization for RViz.'
fi
