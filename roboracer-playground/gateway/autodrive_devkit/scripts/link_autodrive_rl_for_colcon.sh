#!/usr/bin/env bash
# Put autodrive_rl on colcon's search path: src/autodrive_rl -> …/autodrive_rl
#
# Typical layouts:
#   A) Colcon ws with clone under src/ (your Docker case):
#        WS/src/autodrive_devkit/...   and RL at WS/src/autodrive_devkit/autodrive_rl
#      → symlink WS/src/autodrive_rl  →  autodrive_devkit/autodrive_rl
#   B) Monorepo: autodrive_devkit/ at WS root next to src/, RL inside devkit
#      → symlink WS/src/autodrive_rl  →  ../autodrive_devkit/autodrive_rl
#
# Usage:
#   ./scripts/link_autodrive_rl_for_colcon.sh [WORKSPACE_ROOT]
# If WORKSPACE_ROOT is omitted, walks upward from this script looking for layout A or B.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

detect_workspace() {
  local dir="$1"
  while [[ "$dir" != / ]]; do
    if [[ -f "$dir/src/autodrive_devkit/autodrive_rl/package.xml" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    if [[ -f "$dir/autodrive_devkit/autodrive_rl/package.xml" ]] && [[ -d "$dir/src" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}

if [[ "${1:-}" != "" ]]; then
  WS_ROOT="$(cd "$1" && pwd)"
else
  if ! WS_ROOT="$(detect_workspace "$SCRIPT_DIR")"; then
    echo "error: could not find a colcon workspace containing autodrive_devkit/autodrive_rl." >&2
    echo "  Pass workspace explicitly:  $0 /path/to/ws" >&2
    exit 1
  fi
fi

SRC_DIR="$WS_ROOT/src"
NESTED_RL="$SRC_DIR/autodrive_devkit/autodrive_rl"
FLAT_RL="$WS_ROOT/autodrive_devkit/autodrive_rl"

mkdir -p "$SRC_DIR"
cd "$SRC_DIR"
rm -f autodrive_rl

if [[ -f "$NESTED_RL/package.xml" ]]; then
  # Layout A: src/autodrive_devkit/ is the clone; RL is inside it
  ln -sfn autodrive_devkit/autodrive_rl autodrive_rl
elif [[ -f "$FLAT_RL/package.xml" ]]; then
  # Layout B: autodrive_devkit next to src/ at workspace root
  ln -sfn ../autodrive_devkit/autodrive_rl autodrive_rl
else
  echo "error: could not find autodrive_rl under:" >&2
  echo "  $NESTED_RL" >&2
  echo "  $FLAT_RL" >&2
  exit 1
fi

if [[ ! -f "$SRC_DIR/autodrive_rl/package.xml" ]]; then
  echo "error: symlink broken — $SRC_DIR/autodrive_rl/package.xml missing" >&2
  exit 1
fi

echo "Workspace: $WS_ROOT"
echo "Symlink:   $SRC_DIR/autodrive_rl -> $(readlink -f "$SRC_DIR/autodrive_rl")"
echo "ok. Run from $WS_ROOT:"
echo "  colcon build --symlink-install --packages-select autodrive_roboracer autodrive_rl"
