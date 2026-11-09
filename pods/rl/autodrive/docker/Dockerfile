####################################################
#
#   AutoDRIVE RoboRacer PPO training image
#
#   Based on the upstream autodrive_devkit.Dockerfile:
#   https://github.com/AutoDRIVE-Ecosystem/AutoDRIVE-RoboRacer-Sim-Racing
#
#   The simulator connects to this container as a Socket.IO client. Training
#   talks to it directly, so ROS 2 is not in the training loop, but ros:humble
#   keeps Python 3.10 and the OS the same as the competition devkit image.
#
####################################################

FROM ros:humble

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-pip net-tools \
    && rm -rf /var/lib/apt/lists/*

# Bridge dependencies. The simulator speaks an old Socket.IO protocol, so these
# versions are pinned exactly as upstream pins them for Python 3.10.
RUN pip3 install --no-cache-dir \
    eventlet==0.33.3 \
    Flask==1.1.1 \
    Flask-SocketIO==4.1.0 \
    python-socketio==4.2.0 \
    python-engineio==3.13.0 \
    greenlet==1.1.0 \
    gevent==21.12.0 \
    gevent-websocket==0.10.1 \
    Jinja2==3.0.3 \
    itsdangerous==2.0.1 \
    werkzeug==2.0.3

# Training dependencies. PyTorch with CUDA 12.1 runs on the host driver through
# Docker Desktop's WSL 2 GPU passthrough.
RUN pip3 install --no-cache-dir \
    torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    gymnasium==0.29.1 \
    stable-baselines3==2.3.2 \
    tensorboard==2.17.1 \
    tqdm \
    rich

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace
EXPOSE 4567-4598 6006
