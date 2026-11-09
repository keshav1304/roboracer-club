#!/usr/bin/env bash
#
# AutoDRIVE RoboRacer PPO launcher. Run it from Git Bash:   ./launch.bash
#
# The simulators run on Windows and the training runs in a Docker container with the GPU.
# Each simulator connects to its own port on the container, from 4567 upwards.

set -u
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"

# ---- settings ----------------------------------------------------------------------------------
# Your own paths go in launch.local.bash (copy launch.local.example.bash). It is not committed.
# Any of these can also be set from the environment, e.g.  TRACK=none ./launch.bash
[ -f launch.local.bash ] && . ./launch.local.bash
SIM_EXE="${SIM_EXE:-}"                     # simulator for tracks icra and practice
SIM_EXE_NONE="${SIM_EXE_NONE:-$SIM_EXE}"   # simulator for track none (a track with no centre line here)
TRACK="${TRACK:-icra}"                     # icra, practice or none
RACE_LINE="${RACE_LINE:-}"                 # optional race line CSV, e.g. tracks/icra/track3_minimum_curvature_world.csv
IMAGE="${IMAGE:-autodrive-ppo:latest}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"   # e.g. EXTRA_TRAIN_ARGS="--learning-rate 1e-4 --n-steps 4096"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"
BASE_PORT=4567
MAX_INSTANCES=8
MB_PER_INSTANCE=1200               # measured: free RAM fell 1.2 GB per simulator, trainer included
RAM_RESERVE_MB=1500                # keep this much free for Windows and other programs
STAGGER_S=6                        # gap between simulator starts, to avoid a burst of GPU allocations
CONTAINER=autodrive-ppo-job
TB_CONTAINER=autodrive-tensorboard
LOG_DIR="$ROOT/logs"
SIM_PIDS=()

export MSYS_NO_PATHCONV=1          # stop Git Bash rewriting container paths such as /workspace
export MSYS2_ARG_CONV_EXCL='*'

# ---- helpers -----------------------------------------------------------------------------------
say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }
fail() { printf '\033[1;31m%s\033[0m\n' "$*"; }

win_root() { (cd "$ROOT" && pwd -W); }

sim_exe() { if [ "$TRACK" = "none" ]; then printf '%s' "$SIM_EXE_NONE"; else printf '%s' "$SIM_EXE"; fi; }

# Arguments for train_ppo.py and eval_ppo.py that follow the track and race line settings.
track_args() {
    printf -- '--track %s' "$TRACK"
    [ -n "$RACE_LINE" ] && printf -- ' --race-line %s' "$RACE_LINE"
}

check_environment() {
    case "$(uname -s)" in
        MINGW*|MSYS*) ;;
        *) warn "This launcher is written for Git Bash on Windows; you are on $(uname -s)." ;;
    esac
    if [ ! -f "$(sim_exe)" ]; then
        fail "Simulator not found: '$(sim_exe)'"
        fail "Copy launch.local.example.bash to launch.local.bash and set the simulator path in it."
        return 1
    fi
    if [ -n "$RACE_LINE" ] && [ ! -f "$RACE_LINE" ]; then fail "Race line not found: $RACE_LINE"; return 1; fi
    if [ -n "$RACE_LINE" ] && [ "$TRACK" = "none" ]; then fail "Track none cannot use a race line."; return 1; fi
    docker info >/dev/null 2>&1 || { fail "Docker is not running. Start Docker Desktop and try again."; return 1; }
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        warn "The image $IMAGE does not exist yet, building it now."
        build_image || return 1
    fi
    mkdir -p "$LOG_DIR" models runs
}

build_image() {
    say "Building $IMAGE"
    docker build -t "$IMAGE" -f docker/Dockerfile docker
}

free_ram_mb() {
    powershell.exe -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)" 2>/dev/null | tr -d '\r'
}

gpu_args() {   # $1 = cuda or cpu
    [ "$1" = "cuda" ] && printf '%s' '--gpus all'
}

# Starts the job in a detached container, so that stopping is the same however the menu exits.
start_container() {    # $1 = number of ports, $2 = device, the rest = command
    local ports=$1 device=$2; shift 2
    local last=$((BASE_PORT + ports - 1))
    docker rm -f "$CONTAINER" >/dev/null 2>&1
    # shellcheck disable=SC2046
    docker run -d --name "$CONTAINER" $(gpu_args "$device") \
        -p "$BASE_PORT-$last:$BASE_PORT-$last" \
        -v "$(win_root):/workspace" -w /workspace -e PYTHONPATH=/workspace/rl \
        --stop-timeout 60 "$IMAGE" python3 -W ignore "$@" >/dev/null
}

start_sim() {          # $1 = port, $2 = headless or gui
    local port=$1 mode=$2 win_log
    win_log="$(cd "$LOG_DIR" && pwd -W)/sim_$port.log"
    if [ "$mode" = "headless" ]; then
        # -batchmode keeps a graphics device and connects by itself. Do not add -nographics: a
        # blue screen (0x10E, GPU memory) hit within seconds of first using it on this laptop.
        "$(sim_exe)" -batchmode -ip 127.0.0.1 -port "$port" -logFile "$win_log" >/dev/null 2>&1 &
    else
        "$(sim_exe)" -screen-fullscreen 0 -screen-width 1280 -screen-height 720 -logFile "$win_log" >/dev/null 2>&1 &
    fi
    SIM_PIDS[$((port - BASE_PORT))]=$!
}

stop_sims() {
    local pid winpid
    for pid in "${SIM_PIDS[@]:-}"; do
        [ -n "$pid" ] || continue
        winpid="$(cat "/proc/$pid/winpid" 2>/dev/null)"
        [ -n "$winpid" ] && taskkill /PID "$winpid" /F >/dev/null 2>&1
        kill "$pid" 2>/dev/null
    done
    SIM_PIDS=()
}

stop_job() {
    trap - INT TERM
    say "Stopping (the model is saved first, this can take a few seconds)"
    docker stop "$CONTAINER" >/dev/null 2>&1
    docker logs --tail 3 "$CONTAINER" 2>/dev/null
    docker rm -f "$CONTAINER" >/dev/null 2>&1
    stop_sims
}

# Follows the job, restarts any headless simulator that dies, and cleans up at the end.
watch_job() {          # $1 = headless or gui, $2 = number of simulators
    local mode=$1 count=$2 stopped=0 i pid
    trap 'stopped=1' INT TERM
    docker logs -f "$CONTAINER" 2>&1 | grep -v -E 'UserWarning|setattr\(self|_float_to_str' &
    local follower=$!
    while [ "$stopped" = 0 ] && [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]; do
        if [ "$mode" = "headless" ]; then
            for ((i = 0; i < count; i++)); do
                pid="${SIM_PIDS[$i]:-}"
                if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
                    warn "Simulator on port $((BASE_PORT + i)) exited, restarting it."
                    start_sim $((BASE_PORT + i)) headless
                fi
            done
        fi
        sleep 3
    done
    kill "$follower" 2>/dev/null
    stop_job
    say "Finished"
}

gui_connect_note() {
    warn ""
    warn "The simulator window does not connect by itself. In the window:"
    warn "  1. Click the menu icon at the top left."
    warn "  2. Check that the IP is 127.0.0.1 and the port is $BASE_PORT."
    warn "  3. Click 'Disconnected', which changes to 'Connected'."
    warn "  The mode button should read 'Autonomous'. Press Ctrl+C here to stop."
}

ask() {                # $1 = prompt, $2 = default
    local reply
    read -r -p "$1 [$2]: " reply
    printf '%s' "${reply:-$2}"
}

ask_device() {
    local d
    d="$(ask 'Train on (cuda/cpu)' cuda)"
    case "$d" in cuda|cpu) printf '%s' "$d" ;; *) printf 'cuda' ;; esac
}

list_models() {
    find models -name '*.zip' -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2- | head -15
}

choose_model() {
    local models=() m i=1 pick
    while IFS= read -r m; do models+=("$m"); done < <(list_models)
    if [ ${#models[@]} -eq 0 ]; then
        fail "No saved models in models/ yet." >&2
        return 1
    fi
    {
        echo "Saved models, newest first:"
        for m in "${models[@]}"; do echo "  $i) $m"; i=$((i + 1)); done
    } >&2
    read -r -p "Model number, or the path of a .zip [1]: " pick
    pick="${pick:-1}"
    if [ -f "$pick" ] && [ "${pick%.zip}" != "$pick" ]; then printf '%s' "$pick"; return 0; fi
    case "$pick" in ''|*[!0-9]*) fail "Not a number." >&2; return 1 ;; esac
    if [ "$pick" -lt 1 ] || [ "$pick" -gt ${#models[@]} ]; then
        fail "No such model." >&2
        return 1
    fi
    printf '%s' "${models[$((pick - 1))]}"
}

ask_instances() {      # prints the number of instances
    local free suggested n
    free="$(free_ram_mb)"; free="${free:-0}"
    suggested=$(( (free - RAM_RESERVE_MB) / MB_PER_INSTANCE ))
    [ "$suggested" -gt "$MAX_INSTANCES" ] && suggested=$MAX_INSTANCES
    [ "$suggested" -lt 1 ] && suggested=1
    echo "Free RAM: ${free} MB. Each simulator uses about ${MB_PER_INSTANCE} MB of RAM and 1.5 CPU cores." >&2
    n="$(ask 'Number of headless simulators' "$suggested")"
    case "$n" in ''|*[!0-9]*) n=$suggested ;; esac
    [ "$n" -lt 1 ] && n=1
    if [ "$n" -gt "$MAX_INSTANCES" ]; then
        warn "Limiting to $MAX_INSTANCES." >&2
        n=$MAX_INSTANCES
    fi
    if [ $(( n * MB_PER_INSTANCE + RAM_RESERVE_MB )) -gt "$free" ]; then
        warn "That is more than the $suggested that fit in free RAM. Close some programs first." >&2
    fi
    printf '%s' "$n"
}

run_headless_sims() {  # $1 = count
    local i
    for ((i = 0; i < $1; i++)); do
        echo "Starting simulator $((i + 1))/$1 on port $((BASE_PORT + i))"
        start_sim $((BASE_PORT + i)) headless
        [ "$i" -lt $(($1 - 1)) ] && sleep "$STAGGER_S"
    done
}

# ---- menu actions ------------------------------------------------------------------------------
train_gui() {
    local steps device resume_args=()
    check_environment || return
    steps="$(ask 'Total training steps' 1000000)"
    device="$(ask_device)"
    [ -n "${1:-}" ] && resume_args=(--resume "$1")
    say "Training with one simulator window"
    start_container 1 "$device" rl/train_ppo.py --num-envs 1 $(track_args) --total-steps "$steps" \
        --device "$device" "${resume_args[@]}" $EXTRA_TRAIN_ARGS || return
    start_sim "$BASE_PORT" gui
    gui_connect_note
    watch_job gui 1
}

train_headless() {
    local n steps device resume_args=()
    check_environment || return
    n="$(ask_instances)"
    steps="$(ask 'Total training steps' 1000000)"
    device="$(ask_device)"
    [ -n "${1:-}" ] && resume_args=(--resume "$1")
    say "Training with $n headless simulator(s)"
    start_container "$n" "$device" rl/train_ppo.py --num-envs "$n" $(track_args) --total-steps "$steps" \
        --device "$device" "${resume_args[@]}" $EXTRA_TRAIN_ARGS || return
    run_headless_sims "$n"
    echo "Press Ctrl+C to stop. The model is saved when you do. Progress: option 7 (TensorBoard)."
    watch_job headless "$n"
}

resume_training() {
    local model mode
    check_environment || return
    model="$(choose_model)" || return
    mode="$(ask 'Continue with (gui/headless)' headless)"
    if [ "$mode" = "gui" ]; then train_gui "$model"; else train_headless "$model"; fi
}

test_gui() {
    local model episodes
    check_environment || return
    model="$(choose_model)" || return
    episodes="$(ask 'Episodes' 3)"
    say "Testing $model in the simulator window"
    start_container 1 cuda rl/eval_ppo.py --model "$model" $(track_args) --episodes "$episodes" $EXTRA_EVAL_ARGS || return
    start_sim "$BASE_PORT" gui
    gui_connect_note
    watch_job gui 1
}

test_headless() {
    local model episodes
    check_environment || return
    model="$(choose_model)" || return
    episodes="$(ask 'Episodes' 5)"
    say "Testing $model headless"
    start_container 1 cuda rl/eval_ppo.py --model "$model" $(track_args) --episodes "$episodes" $EXTRA_EVAL_ARGS || return
    start_sim "$BASE_PORT" headless
    watch_job headless 1
}

check_setup() {
    check_environment || return
    say "GPU inside the container"
    docker run --rm --gpus all "$IMAGE" python3 -W ignore -c \
        "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    say "Scripted lap on a headless simulator (about a minute)"
    start_container 1 cpu rl/check_env.py --track "$TRACK" || return
    start_sim "$BASE_PORT" headless
    watch_job headless 1
}

tensorboard_start() {
    check_environment || return
    docker rm -f "$TB_CONTAINER" >/dev/null 2>&1
    docker run -d --name "$TB_CONTAINER" -p 6006:6006 -v "$(win_root):/workspace" -w /workspace "$IMAGE" \
        tensorboard --logdir runs --host 0.0.0.0 --port 6006 >/dev/null \
        && say "TensorBoard is at http://localhost:6006 (it keeps running until option 9)"
}

show_status() {
    say "Containers"
    docker ps --filter "name=autodrive" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    say "Simulators"
    tasklist /FI "IMAGENAME eq AutoDRIVE Simulator.exe" /FO TABLE 2>/dev/null
    say "GPU"
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
    say "Latest models"
    list_models | head -5
}

stop_everything() {
    say "Stopping every AutoDRIVE container and simulator"
    docker stop "$CONTAINER" >/dev/null 2>&1
    docker rm -f "$CONTAINER" "$TB_CONTAINER" >/dev/null 2>&1
    powershell.exe -NoProfile -Command "Get-Process -Name 'AutoDRIVE Simulator','UnityCrashHandler64' -ErrorAction SilentlyContinue | Stop-Process -Force" >/dev/null 2>&1
    SIM_PIDS=()
    echo "Done."
}

menu() {
    while true; do
        cat <<EOF

  ============  AutoDRIVE RoboRacer PPO  ============
   Track: $TRACK    Race line: ${RACE_LINE:-none}
   Train
     1) Train with the simulator window
     2) Train headless (choose the number of simulators)
     3) Continue training from a saved model
   Test
     4) Test a model in the simulator window
     5) Test a model headless (laps and lap times only)
   Tools
     6) Check the setup (GPU and a scripted lap)
     7) Start TensorBoard
     8) Status (containers, simulators, GPU, models)
     9) Stop everything
     b) Rebuild the Docker image
     t) Switch track (icra/practice/none)
     r) Set or clear the race line
     q) Quit
EOF
        read -r -p "Choice: " choice || exit 0
        case "$choice" in
            1) train_gui ;;
            2) train_headless ;;
            3) resume_training ;;
            4) test_gui ;;
            5) test_headless ;;
            6) check_setup ;;
            7) tensorboard_start ;;
            8) show_status ;;
            9) stop_everything ;;
            b) build_image ;;
            t) TRACK="$(ask 'Track (icra/practice/none)' "$TRACK")" ;;
            r) RACE_LINE="$(ask 'Race line CSV (enter - for none)' "${RACE_LINE:--}")"; [ "$RACE_LINE" = "-" ] && RACE_LINE="" ;;
            q) exit 0 ;;
            *) echo "Not an option." ;;
        esac
    done
}

menu
