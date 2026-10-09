#!/usr/bin/env bash
# Teacher servers for distillation (SGLang, one per teacher folder). Settings: mixrl/config.env, distillation
# section; placement: mixrl/run.sh domains. mixrl/run.sh distill starts them when they are not running.
#
#   mixrl/teachers.sh [start] [ROOT]   start the teachers of ROOT (default DISTILL_ROOT) that are not running
#   mixrl/teachers.sh status [ROOT]    which teachers answer
#   mixrl/teachers.sh stop             stop every teacher server
set -Eeuo pipefail

MIXRL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$MIXRL_DIR")
source "$MIXRL_DIR/config.env"
# Native servers (TEACHER_USE_DOCKER=0) log and keep their process ids here.
TEACHER_LOG_DIR=${TEACHER_LOG_DIR:-$BASE_DIR/runs/teachers}
TEACHER_START_SECONDS=${TEACHER_START_SECONDS:-1800}

usage() { sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2; }
fail() { echo "error: $*" >&2; exit 1; }

command=${1:-start}
[[ $# -gt 0 ]] && shift
case "$command" in
    start|status) [[ $# -le 1 ]] || usage ;;
    stop)
        [[ $# -eq 0 ]] || usage
        if [[ "$TEACHER_USE_DOCKER" == 1 ]]; then
            containers=$(docker ps -aq --filter 'name=^mixrl-teacher-[0-9]+$')
            [[ -n "$containers" ]] && docker rm -f $containers >/dev/null
        fi
        for pid_file in "$TEACHER_LOG_DIR"/teacher-*.pid; do
            [[ -f "$pid_file" ]] || continue
            kill "$(cat "$pid_file")" 2>/dev/null || true
            rm -f "$pid_file"
        done
        echo "Teacher servers stopped."
        exit 0
        ;;
    *) usage ;;
esac
ROOT=${1:-$DISTILL_ROOT}

# One line per server, in start order: port, GPU, memory setting, name (its domains joined by +), checkpoint, URL.
plan=$(cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT" python3 -m slime_plugins.chimera_mixrl.distill "$ROOT" --plan \
    --teacher-gpus "$TEACHER_GPUS" --teacher-port "$TEACHER_PORT" --teacher-memory "$TEACHER_GPU_MEMORY") \
    || fail "$ROOT: the distillation folder has problems (mixrl/run.sh domains $ROOT shows them)"

# The name a server announces (SGLang's --served-model-name), empty when nothing answers on the port.
served() {
    curl -sf --connect-timeout 5 "http://127.0.0.1:$1/v1/models" 2>/dev/null \
        | python3 -c 'import json, sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true
}

if [[ "$command" == status ]]; then
    missing=0
    while IFS=$'\t' read -r port gpu memory name path url; do
        now=$(served "$port")
        if [[ "$now" == "$name" ]]; then echo "ready    $name on GPU $gpu at $url ($path)"
        elif [[ -n "$now" ]]; then echo "WRONG    port $port serves '$now', expected $name"; missing=1
        else echo "down     $name (GPU $gpu, port $port)"; missing=1; fi
    done <<< "$plan"
    exit "$missing"
fi

# Chimera loads through SGLang's Transformers backend, with the rollout engines' precision settings, so a
# teacher identical to the student scores tokens as the student's own engines do.
serve_args() {
    local port=$1 memory=$2 name=$3 path=$4
    args=(--model-path "$path" --served-model-name "$name" --host 127.0.0.1 --port "$port" --trust-remote-code
        --model-impl transformers --enable-fp32-lm-head --context-length "$TRAIN_SEQUENCE_LENGTH"
        --mem-fraction-static "$memory" --random-seed "$MIXRL_SEED" --skip-server-warmup
        # Teachers only prefill (one forward over prompt + answer): no decode CUDA graphs to capture.
        --disable-cuda-graph --log-level-http warning)
}

wait_ready() {
    local port=$1 name=$2 log_hint=$3 check=$4
    for _ in $(seq 1 $((TEACHER_START_SECONDS / 5))); do
        [[ "$(served "$port")" == "$name" ]] && return 0
        if ! eval "$check"; then
            echo "error: teacher $name exited during startup; $log_hint" >&2
            return 1
        fi
        sleep 5
    done
    echo "error: teacher $name is not serving after $TEACHER_START_SECONDS s; $log_hint" >&2
    return 1
}

# A teacher holds one connection per answer being scored (a whole batch at once): the usual soft limit of 1024
# open files would fail them with "Too many open files". Native servers inherit this; containers get their own.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
started=0
# Servers sharing a GPU start one after another (see distill.servers), so every start waits for the previous.
while IFS=$'\t' read -r port gpu memory name path url; do
    now=$(served "$port")
    if [[ "$now" == "$name" ]]; then
        echo "teacher $name: already running on port $port"
        continue
    elif [[ -n "$now" ]]; then
        fail "port $port serves '$now', not teacher $name; stop it first: mixrl/teachers.sh stop"
    fi
    echo "teacher $name: starting on GPU $gpu, port $port, memory $memory ($path)"
    serve_args "$port" "$memory" "$name" "$path"
    if [[ "$TEACHER_USE_DOCKER" == 1 ]]; then
        container=mixrl-teacher-$port
        docker rm -f "$container" >/dev/null 2>&1 || true
        docker run -d --name "$container" --gpus "\"device=$gpu\"" --ipc=host --net=host --ulimit nofile=1048576:1048576 \
            -v "$path:$path:ro" -v "$TRANSFORMERS_DIR:/workspace/transformers:ro" -v "$REPO_ROOT:/workspace/slime:ro" \
            -e CHIMERA_TRANSFORMERS_ROOT=/workspace/transformers -e CHIMERA_MATCH_RMSNORM=1 \
            -e PYTHONPATH=/workspace/slime/examples/chimera/runtime:/workspace/slime:/root/Megatron-LM \
            -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -w /workspace/slime \
            "$SLIME_IMAGE" python3 -m sglang.launch_server "${args[@]}" >/dev/null
        wait_ready "$port" "$name" "log: docker logs $container" \
            "docker ps -q --filter name=^$container\$ | grep -q ." \
            || { docker logs --tail 40 "$container" >&2 || true; docker rm -f "$container" >/dev/null 2>&1; exit 1; }
    else
        mkdir -p "$TEACHER_LOG_DIR"
        log=$TEACHER_LOG_DIR/teacher-$port.log
        CUDA_VISIBLE_DEVICES=$gpu CHIMERA_TRANSFORMERS_ROOT=${CHIMERA_TRANSFORMERS_ROOT:-$TRANSFORMERS_DIR} \
            CHIMERA_MATCH_RMSNORM=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
            PYTHONPATH="$REPO_ROOT/examples/chimera/runtime:$REPO_ROOT${MEGATRON_ROOT:+:$MEGATRON_ROOT}${PYTHONPATH:+:$PYTHONPATH}" \
            nohup python3 -m sglang.launch_server "${args[@]}" >> "$log" 2>&1 < /dev/null &
        echo $! > "$TEACHER_LOG_DIR/teacher-$port.pid"
        wait_ready "$port" "$name" "log: $log" "kill -0 $! 2>/dev/null" || { tail -40 "$log" >&2; exit 1; }
    fi
    echo "teacher $name: ready at $url"
    started=$((started + 1))
done <<< "$plan"
echo "Teachers ready ($started started)."
