#!/usr/bin/env bash
# One-time setup: f1tenth_gym (v1.0.0 branch) in a local venv + the Spielberg map.
# Safe to rerun. Usage: ./setup.sh   (optionally PYTHON=/path/to/python3.12 ./setup.sh)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3.12}"
command -v "$PYTHON" >/dev/null || PYTHON=python3
echo ">> using $PYTHON ($("$PYTHON" -c 'import sys,platform; print(sys.version.split()[0], platform.machine())'))"

if [ ! -d f1tenth_gym ]; then
  git clone -b v1.0.0 --depth 1 https://github.com/f1tenth/f1tenth_gym.git
fi
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi
VPY=.venv/bin/python

# On Apple Silicon the interpreter must be native arm64, or numba/llvmlite break
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  arch=$("$VPY" -c 'import platform; print(platform.machine())')
  if [ "$arch" != "arm64" ]; then
    echo "!! venv Python is $arch, not arm64. Delete .venv and rerun with a native arm64 python3.12" >&2
    exit 1
  fi
fi

"$VPY" -m pip install -q --upgrade pip
"$VPY" -m pip install -q -e ./f1tenth_gym
"$VPY" -m pip install -q -r requirements.txt
# Swap opencv-python (pulled in by the gym) for the headless build: avoids
# duplicate-SDL warnings/crashes with pygame. Both share the cv2/ folder, so
# force-reinstall headless after the uninstall (otherwise a rerun breaks cv2).
"$VPY" -m pip uninstall -q -y opencv-python || true
"$VPY" -m pip install -q --force-reinstall --no-deps "opencv-python-headless>=4.9"
"$VPY" -c "import cv2; cv2.polylines" || { echo "!! cv2 is broken after install" >&2; exit 1; }

# Download Spielberg (the gym fetches it on first use) and fix its file name:
# the v1.0.0 loader wants Spielberg_map.yaml but the download ships Spielberg.yaml
"$VPY" -c "import gymnasium as gym; gym.make('f1tenth_gym:f1tenth-v0', config={'map': 'Spielberg'})" >/dev/null 2>&1 || true
M=f1tenth_gym/maps/Spielberg
[ -f "$M/Spielberg.yaml" ] || { echo "!! Spielberg map download failed" >&2; exit 1; }
cp -n "$M/Spielberg.yaml" "$M/Spielberg_map.yaml" 2>/dev/null || true

"$VPY" -c "import gymnasium as gym; gym.make('f1tenth_gym:f1tenth-v0', config={'map': 'Spielberg', 'num_agents': 2}); print('>> gym + Spielberg OK')" 2>&1 | grep -v '^objc'
echo ">> done. Next:  .venv/bin/python run_two_cars.py --render"
