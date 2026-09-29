#!/usr/bin/env bash
# MixRL training runs. Settings: mixrl/config.env. Tasks: mixrl/tasks.json.
#
#   mixrl/run.sh tasks                 preview the task table (no Docker, no GPUs)
#   mixrl/run.sh preflight [RUN_NAME]  check tasks, data, reward service, judge and model; no GPUs
#   mixrl/run.sh start [RUN_NAME]      start a new run (default name: mixrl-<date>-<time>)
#   mixrl/run.sh resume RUN_NAME       continue a run from its latest checkpoint
#
# Any config.env value can be overridden for one command: LR=2e-6 mixrl/run.sh start my-run
set -Eeuo pipefail

MIXRL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$MIXRL_DIR")
source "$MIXRL_DIR/config.env"

usage() {
    sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}
fail() { echo "error: $*" >&2; exit 1; }

command=${1:-}
[[ $# -gt 0 ]] && shift
case "$command" in
    tasks)
        [[ $# -eq 0 ]] || usage
        cd "$REPO_ROOT"
        exec env PYTHONPATH="$REPO_ROOT" python3 -m slime_plugins.chimera_mixrl.tasks \
            "$MIXRL_DIR/tasks.json" --samples-per-prompt "$N_SAMPLES_PER_PROMPT"
        ;;
    preflight|start)
        [[ $# -le 1 ]] || usage
        RUN_NAME=${1:-mixrl-$(date +%Y%m%d-%H%M%S)}
        RESUME=0
        ;;
    resume)
        [[ $# -eq 1 ]] || usage
        RUN_NAME=$1
        RESUME=1
        ;;
    *) usage ;;
esac

# Host-side checks with plain messages, before any container starts.
[[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || fail "run name '$RUN_NAME': use letters, digits, . _ - only"
IFS=, read -ra gpu_ids <<< "$TRAIN_GPUS"
[[ ${#gpu_ids[@]} -eq $POLICY_GPUS ]] || fail "TRAIN_GPUS=$TRAIN_GPUS lists ${#gpu_ids[@]} GPUs but POLICY_GPUS=$POLICY_GPUS"
for path in "$BASE_DIR/models/$MODEL_NAME/hf/config.json" \
            "$BASE_DIR/models/$MODEL_NAME/mcore/latest_checkpointed_iteration.txt" \
            "$BASE_DIR/datasets/$DATASET_NAME/manifest.json" \
            "$TRANSFORMERS_DIR/src/transformers/models/chimera/__init__.py"; do
    [[ -f "$path" ]] || fail "missing $path (check BASE_DIR, MODEL_NAME, DATASET_NAME, TRANSFORMERS_DIR in mixrl/config.env)"
done
curl -sf --connect-timeout 5 "http://127.0.0.1:$REWARD_PORT/health" >/dev/null \
    || fail "reward service not reachable on port $REWARD_PORT; start it with mixrl/reward.sh"
run_dir=$BASE_DIR/runs/chimera/mixrl/$RUN_NAME
# A start that failed before any rollout or checkpoint (only logs/manifests) may be retried.
if [[ "$command" == start && -n "$(find "$run_dir/checkpoints" "$run_dir/rollouts" -type f -print -quit 2>/dev/null)" ]]; then
    fail "run $RUN_NAME already exists at $run_dir; pick a new name or: mixrl/run.sh resume $RUN_NAME"
fi
if [[ "$command" == resume && ! -f "$run_dir/checkpoints/latest_checkpointed_iteration.txt" ]]; then
    fail "no checkpoint to resume at $run_dir/checkpoints"
fi

# Hand every resolved config.env value (including command-line overrides) to the container.
env_file=$(mktemp)
followers=()
trap 'rm -f "$env_file"; for pid in "${followers[@]}"; do kill "$pid" 2>/dev/null || true; done' EXIT
while IFS= read -r name; do
    printf '%s=%s\n' "$name" "${!name}"
done < <(grep -oE '^[A-Z_][A-Z0-9_]*=' "$MIXRL_DIR/config.env" | tr -d =) > "$env_file"
printf 'RUN_NAME=%s\nRESUME=%s\nDATA_ROOT=/data\nCHIMERA_TRANSFORMERS_ROOT=/workspace/transformers\n' \
    "$RUN_NAME" "$RESUME" >> "$env_file"

docker_args=(--rm --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864
    -v "$BASE_DIR/models/$MODEL_NAME:/data/models/$MODEL_NAME:ro"
    -v "$BASE_DIR/datasets/$DATASET_NAME:/data/datasets/$DATASET_NAME:ro"
    -v "$TRANSFORMERS_DIR:/workspace/transformers:ro"
    -v "$REPO_ROOT:/workspace/slime"
    -w /workspace/slime --env-file "$env_file")
if [[ "$command" == preflight ]]; then
    # Full checks in the real image, written to a throwaway folder inside the container.
    docker_args+=(-e DRY_RUN=1 -e MIXRL_RUNS_ROOT=/tmp/mixrl-preflight)
    echo "Preflight for $RUN_NAME (no GPUs are used)"
else
    mkdir -p "$BASE_DIR/runs"
    # Inner quotes keep a comma-separated device list as one value for Docker.
    docker_args+=(--gpus "\"device=$TRAIN_GPUS\"" -v "$BASE_DIR/runs:/data/runs")
    if [[ -t 0 && -t 1 ]]; then docker_args+=(-it); fi
    echo "$command run $RUN_NAME on GPUs $TRAIN_GPUS; outputs: $run_dir"
    # The reward service's and Docker judge's output during this run, next to train.log.
    mkdir -p "$run_dir/logs" 2>/dev/null || true
    since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    for service in mixrl-reward-service:reward_service mixrl-judge-server:judge; do
        container=${service%%:*}
        if ! docker ps -q --filter "name=^$container$" | grep -q .; then continue; fi
        if [[ ! -w "$run_dir/logs" ]]; then
            echo "note: $run_dir/logs is not writable here; $container output stays in: docker logs $container"
            continue
        fi
        docker logs -f --since "$since" "$container" >> "$run_dir/logs/${service#*:}.log" 2>&1 &
        followers+=($!)
    done
fi
docker run "${docker_args[@]}" "$SLIME_IMAGE" bash mixrl/internal/launch.sh
