# CPU image for env_v2 training. The sim is LiDAR ray casting + numba physics, so CPU cores are what matter;
# a RunPod CPU pod with 16 to 32 vCPU is the right shape, a GPU pod would sit idle.
#   docker build -t penn-roboracer/rl-v2 -f pods/rl/runpod/Dockerfile .      (from the repo root)
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pods/rl/runpod/requirements-v2.txt /tmp/requirements-v2.txt
RUN pip install --no-cache-dir -r /tmp/requirements-v2.txt
RUN git clone -b dev-humble https://github.com/f1tenth/f1tenth_gym /opt/f1tenth_gym \
 && cd /opt/f1tenth_gym && git checkout bdaec14 && pip install --no-cache-dir --no-deps -e .
WORKDIR /workspace/roboracer-club
COPY . /workspace/roboracer-club
ENV ENV_V2_TRACK_CACHE=/workspace/track_cache
CMD ["bash", "pods/rl/runpod/train_runpod.sh"]
