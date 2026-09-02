#!/bin/bash
# RoboRacer Playground backend entrypoint.
# Starts the AutoDRIVE ROS 2 bridge (Socket.IO server on :4567) and the
# playground gateway (FastAPI WebSocket API on :8000). If either process
# dies, the container exits so the orchestrator (Fly/Docker) restarts it.
set -e

source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash
export XDG_RUNTIME_DIR=/tmp/runtime-root
mkdir -p "$XDG_RUNTIME_DIR"

echo "[playground] starting autodrive_bridge on :4567"
ros2 run autodrive_roboracer autodrive_bridge &
BRIDGE_PID=$!

echo "[playground] starting gateway on :${GATEWAY_PORT:-8000}"
python3 /home/playground_gateway/main.py &
GATEWAY_PID=$!

trap 'kill $BRIDGE_PID $GATEWAY_PID 2>/dev/null || true' SIGINT SIGTERM

# Exit when the first process exits
wait -n $BRIDGE_PID $GATEWAY_PID
EXIT_CODE=$?
echo "[playground] a core process exited (code=$EXIT_CODE); shutting down"
kill $BRIDGE_PID $GATEWAY_PID 2>/dev/null || true
exit $EXIT_CODE
