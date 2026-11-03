#!/bin/bash
# Localize on the saved room map and follow the circle in the middle gap.
# Flip SWB down after you set the pose in Foxglove. Ctrl+C stops the car.
set -e
cd "$(dirname "$0")"
source /opt/ros/humble/setup.bash
source "$HOME/osracer_ws/install/setup.bash"
source "$HOME/ros2_ws/install/setup.bash"
export AMENT_PREFIX_PATH="$HOME/ros2_ws/install/neoracer_ros2_driver:$HOME/ros2_ws/install/lakibeam1:${AMENT_PREFIX_PATH}"

# The live mapper and this node would both publish /map.
pkill -f "[a]sync_slam_toolbox_node" 2>/dev/null || true
pkill -f "[o]sracer_slam slam_toolbox.launch.py" 2>/dev/null || true
sleep 1

PP=""
AMCL=""
cleanup() {
  if [[ -n "$PP" ]]; then
    kill -INT "$PP" 2>/dev/null || true
    wait "$PP" 2>/dev/null || true
  fi
  if [[ -n "$AMCL" ]]; then
    kill "$AMCL" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python3 "$(dirname "$0")/circle_pursuit.py" &
PP=$!
sleep 2
if ! pgrep -f "[n]av2_amcl/amcl" >/dev/null; then
  ros2 run nav2_amcl amcl --ros-args --params-file "$(dirname "$0")/amcl_circle.yaml" &
  AMCL=$!
  sleep 2
fi
# A second start finds AMCL already configured. That is not a failure.
set +e
ros2 lifecycle set /amcl configure
ros2 lifecycle set /amcl activate
set -e
wait "$PP"
