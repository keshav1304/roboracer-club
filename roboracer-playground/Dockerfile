####################################################
#
#   RoboRacer Playground Backend Dockerfile
#   (bundled AutoDRIVE Devkit + Playground Gateway)
#
#   Build context: the roboracer-playground/ directory itself.
#   Runs headless: autodrive_bridge (port 4567) +
#   FastAPI gateway (port 8000). Algorithms are
#   started/stopped by the gateway on demand.
#
####################################################

FROM ros:humble

# Install Debian packages
RUN apt update \
    && apt install -y --no-install-recommends \
    sudo \
    wget \
    curl \
    unzip \
    net-tools \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for the AutoDRIVE bridge
# (version-sensitive Socket.IO stack — do NOT upgrade these pins,
#  the Unity simulator client depends on this exact protocol version)
RUN pip3 install attrdict \
    && pip3 install numpy==1.22.2 \
    && pip3 install pillow \
    && pip3 install opencv-contrib-python-headless==4.10.0.84 \
    && pip3 install eventlet==0.33.3 \
    && pip3 install Flask==1.1.1 \
    && pip3 install Flask-SocketIO==4.1.0 \
    && pip3 install python-socketio==4.2.0 \
    && pip3 install python-engineio==3.13.0 \
    && pip3 install greenlet==1.1.0 \
    && pip3 install gevent==21.12.0 \
    && pip3 install gevent-websocket==0.10.1 \
    && pip3 install Jinja2==3.0.3 \
    && pip3 install itsdangerous==2.0.1 \
    && pip3 install werkzeug==2.0.3 \
    && pip3 install transforms3d

# Install ROS 2 dependencies (headless: no rviz)
# nav2-amcl + map-server localize Pure Pursuit on saved Race Lab maps
# packages.ros.org often drops large .debs (slam-toolbox still pulls
# rviz-ogre-vendor). Retry the transaction so already-fetched files are reused.
RUN set -eux; \
    printf '%s\n' \
      'Acquire::Retries "10";' \
      'Acquire::http::Timeout "120";' \
      'Acquire::https::Timeout "120";' \
      'Acquire::http::Pipeline-Depth "0";' \
      'Acquire::ForceIPv4 "true";' \
      > /etc/apt/apt.conf.d/99-ros-retries; \
    apt-get update; \
    n=0; \
    until apt-get install -y --no-install-recommends \
        ros-$ROS_DISTRO-tf-transformations \
        ros-$ROS_DISTRO-imu-tools \
        ros-$ROS_DISTRO-cv-bridge \
        ros-$ROS_DISTRO-slam-toolbox \
        ros-$ROS_DISTRO-nav2-amcl \
        ros-$ROS_DISTRO-nav2-map-server \
        ros-$ROS_DISTRO-nav2-lifecycle-manager \
        ros-$ROS_DISTRO-nav2-util; do \
        n=$((n+1)); \
        [ "$n" -lt 5 ] || exit 1; \
        echo "apt-get install failed (attempt $n/5); retrying in 15s..."; \
        sleep 15; \
        apt-get update; \
    done; \
    rm -rf /var/lib/apt/lists/*

# Install playground gateway dependencies
COPY gateway/requirements.txt /tmp/gateway-requirements.txt
RUN pip3 install -r /tmp/gateway-requirements.txt \
    && pip3 install "packaging>=24.0"

# Bundled AutoDRIVE Devkit (lives under gateway/autodrive_devkit)
COPY gateway/autodrive_devkit/. /home/autodrive_devkit/src/autodrive_devkit
RUN cd /home/autodrive_devkit \
    && rm -rf build install log src/autodrive_devkit/build src/autodrive_devkit/install \
    && bash -c "source /opt/ros/humble/setup.bash && colcon build"

# Playground gateway
COPY gateway/*.py /home/playground_gateway/

WORKDIR /home/autodrive_devkit
EXPOSE 4567 8000

COPY entrypoint.sh /home/playground_entrypoint.sh
RUN chmod +x /home/playground_entrypoint.sh
ENTRYPOINT ["/home/playground_entrypoint.sh"]
