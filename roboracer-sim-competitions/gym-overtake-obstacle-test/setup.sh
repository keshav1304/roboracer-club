#!/usr/bin/env bash
# One-time setup: f1tenth_gym (v1.0.0 branch) in a local venv + the PortoWide test track.
# Usage: ./setup.sh            (optionally PYTHON=/path/to/python3.12 ./setup.sh)
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
"$VPY" -m pip install -q --upgrade pip
"$VPY" -m pip install -q -e ./f1tenth_gym
# headless OpenCV avoids duplicate-SDL warnings/crashes with pygame
"$VPY" -m pip uninstall -q -y opencv-python || true
"$VPY" -m pip install -q -r requirements.txt

# On Apple Silicon the interpreter must be native arm64, or numba/llvmlite break
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  arch=$("$VPY" -c 'import platform; print(platform.machine())')
  if [ "$arch" != "arm64" ]; then
    echo "!! venv Python is $arch, not arm64. Delete .venv and rerun with a native arm64 python3.12" >&2
    exit 1
  fi
fi

"$VPY" build_track.py
"$VPY" -c "import gymnasium as gym; gym.make('f1tenth_gym:f1tenth-v0', config={'map': 'PortoWide', 'num_agents': 2}); print('>> gym + PortoWide OK')" 2>&1 | grep -v '^objc'
echo ">> done. Next:  .venv/bin/python check.py   or   .venv/bin/python race.py --render"
