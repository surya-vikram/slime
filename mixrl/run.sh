#!/usr/bin/env bash
# MixRL training runs. Settings: mixrl/config.env. Tasks: mixrl/tasks.json.
#
#   mixrl/run.sh start [RUN_NAME]      everything for a new run: start the judge and reward service if they
#                                      are not up, check the tasks, preflight, train (name: mixrl-<date>-<time>)
#   mixrl/run.sh resume RUN_NAME       the same without the preflight, from the run's latest checkpoint
#   mixrl/run.sh preflight [RUN_NAME]  checks only: tasks, data, reward service, judge and model; no GPUs
#   mixrl/run.sh tasks                 preview the task table (no Docker, no GPUs)
#
# Any config.env value can be overridden for one command: LR=2e-6 mixrl/run.sh start my-run
# Running services are reused; mixrl/judge.sh or mixrl/reward.sh restarts one with new settings.
set -Eeuo pipefail

MIXRL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$MIXRL_DIR")
source "$MIXRL_DIR/config.env"

usage() {
    sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
}
fail() { echo "error: $*" >&2; exit 1; }
note() { echo "[$(date '+%F %T')] $*"; }

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
reward_up() { curl -sf --connect-timeout 5 "http://127.0.0.1:$REWARD_PORT/health" >/dev/null; }
judge_up() { curl -sf --connect-timeout 5 "http://$JUDGE_HOST:$JUDGE_PORT/v1/models" >/dev/null 2>&1; }
if [[ "$command" == preflight ]]; then
    reward_up || fail "reward service not reachable on port $REWARD_PORT; start it with mixrl/reward.sh"
fi
run_dir=$BASE_DIR/runs/chimera/mixrl/$RUN_NAME
# A start that failed before any rollout or checkpoint (only logs/manifests) may be retried.
if [[ "$command" == start && -n "$(find "$run_dir/checkpoints" "$run_dir/rollouts" -type f -print -quit 2>/dev/null)" ]]; then
    fail "run $RUN_NAME already exists at $run_dir; pick a new name or: mixrl/run.sh resume $RUN_NAME"
fi
if [[ "$command" == resume && ! -f "$run_dir/checkpoints/latest_checkpointed_iteration.txt" ]]; then
    fail "no checkpoint to resume at $run_dir/checkpoints"
fi
# Megatron resumes with the optimizer state; weights-only checkpoints have none (it failed with KeyError: 'optimizer').
if [[ "$command" == resume ]] && grep -q '^NO_SAVE_OPTIM=1$' "$run_dir/manifests/config.env" 2>/dev/null; then
    fail "run $RUN_NAME saved weights only (NO_SAVE_OPTIM=1), so it cannot resume; start a new run, and use NO_SAVE_OPTIM=0 for runs you want to resume"
fi

# start/resume: bring up what the run needs, as mixrl/judge.sh and mixrl/reward.sh would; their output
# goes to the run's logs folder and the terminal gets one line per step.
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)  # service output from here on is copied into the run's logs
if [[ "$command" != preflight ]]; then
    # A run folder made by an earlier container may be root-owned; then service logs go to a temp folder.
    logs=$run_dir/logs
    { mkdir -p "$logs" 2>/dev/null && [[ -w "$logs" ]]; } || { logs=$(mktemp -d); note "note: $run_dir/logs is not writable; service logs: $logs"; }
    tasks_text=$(cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT" python3 -m slime_plugins.chimera_mixrl.tasks \
        "$MIXRL_DIR/tasks.json" --samples-per-prompt "$N_SAMPLES_PER_PROMPT") || fail "mixrl/tasks.json: $tasks_text"
    printf '%s\n' "$tasks_text" > "$logs/tasks.txt"
    note "tasks: $(grep -m1 'tasks enabled' <<< "$tasks_text") (table: $logs/tasks.txt)"
    if judge_up; then
        note "judge: already running at $JUDGE_HOST:$JUDGE_PORT ($JUDGE_NAME)"
    elif grep -q '^judge: not needed' <<< "$tasks_text"; then
        note "judge: not needed by the enabled tasks; not started"
    elif [[ "$JUDGE_USE_DOCKER" == 1 ]]; then
        note "judge: starting on GPUs $JUDGE_GPUS (log: $logs/judge_start.log; this takes minutes)"
        "$MIXRL_DIR/judge.sh" > "$logs/judge_start.log" 2>&1 \
            || { tail -40 "$logs/judge_start.log" >&2; fail "the judge did not start"; }
        note "judge: ready"
    else
        note "judge: starting native vLLM on GPUs $JUDGE_GPUS (log: $logs/judge.log; this takes minutes)"
        nohup "$MIXRL_DIR/judge.sh" >> "$logs/judge.log" 2>&1 < /dev/null &
        judge_pid=$!
        until judge_up; do
            kill -0 "$judge_pid" 2>/dev/null || { tail -40 "$logs/judge.log" >&2; fail "the judge exited during startup"; }
            sleep 10
        done
        note "judge: ready (pid $judge_pid)"
    fi
    if reward_up; then
        note "reward service: already running on port $REWARD_PORT (mixrl/reward.sh restarts it with new settings)"
    else
        note "reward service: starting $REWARD_PROCESSES process(es) (log: $logs/reward_start.log)"
        "$MIXRL_DIR/reward.sh" > "$logs/reward_start.log" 2>&1 \
            || { tail -40 "$logs/reward_start.log" >&2; fail "the reward service did not start"; }
        note "reward service: $(grep -m1 'Reward service ready' "$logs/reward_start.log" || echo ready)"
    fi
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
# Launcher switches with no config.env entry (defaults in mixrl/internal/launch.sh) reach the container too
# when set for this command, e.g. MIXRL_PIPELINE_SECONDS=15 mixrl/run.sh start. Container paths and the
# run's identity are set above, never taken from the host.
container_owned=" DATA_ROOT CHIMERA_TRANSFORMERS_ROOT HF_CHECKPOINT MCORE_CHECKPOINT MEGATRON_ROOT MIXRL_DATA_DIR "
container_owned+="MIXRL_CODE_AUDIT_DIR MIXRL_TASKS_CONFIG MIXRL_RUNS_ROOT RUN_NAME RESUME DRY_RUN PREFLIGHT_ONLY "
known=" $(grep -oE '^[A-Z_][A-Z0-9_]*=' "$MIXRL_DIR/config.env" | tr -d = | tr '\n' ' ') "
for name in $(sed -nE 's/^(export )?([A-Z_][A-Z0-9_]*)=\$\{\2:-.*/\2/p' "$MIXRL_DIR/internal/launch.sh" | sort -u); do
    [[ "$container_owned" == *" $name "* || "$known" == *" $name "* ]] && continue
    known+="$name "
    if [[ -n "${!name+set}" ]]; then printf '%s=%s\n' "$name" "${!name}" >> "$env_file"; fi
done
# A MixRL-looking variable that is not a setting anywhere is most likely a typo: say so instead of ignoring it.
for name in $(compgen -e | grep -E '^(MIXRL|CHIMERA|SGLANG|JUDGE|REWARD)_' || true); do
    [[ "$known$container_owned" == *" $name "* ]] || echo "warning: $name is set but is not a MixRL setting; it has no effect" >&2
done

docker_args=(--rm --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576
    -v "$BASE_DIR/models/$MODEL_NAME:/data/models/$MODEL_NAME:ro"
    -v "$BASE_DIR/datasets/$DATASET_NAME:/data/datasets/$DATASET_NAME:ro"
    -v "$TRANSFORMERS_DIR:/workspace/transformers:ro"
    -v "$REPO_ROOT:/workspace/slime"
    -w /workspace/slime --env-file "$env_file")
# Full checks in the real image, written to a throwaway folder inside the container.
preflight_args=(-e DRY_RUN=1 -e MIXRL_RUNS_ROOT=/tmp/mixrl-preflight)
if [[ "$command" == preflight ]]; then
    echo "Preflight for $RUN_NAME (no GPUs are used)"
    exec docker run "${docker_args[@]}" "${preflight_args[@]}" "$SLIME_IMAGE" bash mixrl/internal/launch.sh
fi
if [[ "$command" == start ]]; then
    note "preflight: checking tasks, data, services and model (log: $logs/preflight.log)"
    docker run "${docker_args[@]}" "${preflight_args[@]}" "$SLIME_IMAGE" bash mixrl/internal/launch.sh \
        > "$logs/preflight.log" 2>&1 < /dev/null \
        || { tail -40 "$logs/preflight.log" >&2; fail "preflight failed; nothing was started"; }
    note "preflight: ok"
fi
mkdir -p "$BASE_DIR/runs"
# Inner quotes keep a comma-separated device list as one value for Docker.
docker_args+=(--gpus "\"device=$TRAIN_GPUS\"" -v "$BASE_DIR/runs:/data/runs")
if [[ -t 0 && -t 1 ]]; then docker_args+=(-it); fi
note "training: $command run $RUN_NAME on GPUs $TRAIN_GPUS; outputs: $run_dir"
# The reward service's and Docker judge's output during this run, next to train.log.
# Several reward-service processes (REWARD_PROCESSES > 1) log to reward_service-<i>.log.
services=()
for container in $(docker ps --format '{{.Names}}' --filter 'name=^mixrl-reward-service(-[0-9]+)?$' | sort); do
    services+=("$container:reward_service${container#mixrl-reward-service}")
done
services+=(mixrl-judge-server:judge)
for service in "${services[@]}"; do
    container=${service%%:*}
    if ! docker ps -q --filter "name=^$container$" | grep -q .; then continue; fi
    docker logs -f --since "$since" "$container" >> "$logs/${service#*:}.log" 2>&1 &
    followers+=($!)
done
docker run "${docker_args[@]}" "$SLIME_IMAGE" bash mixrl/internal/launch.sh
