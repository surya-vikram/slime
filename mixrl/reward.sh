#!/usr/bin/env bash
# Start the reward service (chimera-eval) in its own container. Settings: mixrl/config.env.
# It starts whether or not the judge is up: tasks that never call the judge train without it,
# and training refuses to start any enabled task that needs an unreachable judge.
# REWARD_PROCESSES > 1 starts that many processes on ports REWARD_PORT, REWARD_PORT+1, ...
# (each with its own cache), since one Python process grades on one core.
#
#   mixrl/reward.sh          start (or restart) the reward service
#   mixrl/reward.sh stop     stop it
set -Eeuo pipefail

MIXRL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$MIXRL_DIR/config.env"
CONTAINER=mixrl-reward-service
running_services() { docker ps -aq --filter "name=^$CONTAINER(-[0-9]+)?\$"; }

case "${1:-start}" in
    start) ;;
    stop)
        ids=$(running_services)
        if [[ -n "$ids" ]]; then docker rm -f $ids >/dev/null; echo "Stopped the reward service."; else echo "The reward service is not running."; fi
        exit 0 ;;
    *) sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac
(( REWARD_PROCESSES >= 1 )) || { echo "error: REWARD_PROCESSES must be at least 1" >&2; exit 1; }

DATA_DIR=$BASE_DIR/datasets/$DATASET_NAME
[[ -f "$DATA_DIR/manifest.json" ]] || { echo "error: dataset not found at $DATA_DIR" >&2; exit 1; }
[[ -f "$EVAL_REPO/eval_stack/reward_service.py" ]] || { echo "error: chimera-eval not found at $EVAL_REPO (EVAL_REPO)" >&2; exit 1; }
JUDGE_URL=http://$JUDGE_HOST:$JUDGE_PORT/v1
if curl -sf --connect-timeout 5 "$JUDGE_URL/models" >/dev/null 2>&1; then
    echo "Judge is up at $JUDGE_URL ($JUDGE_NAME)."
else
    echo "Judge not reachable at $JUDGE_URL: only judge-free tasks can train until it is up (mixrl/judge.sh)."
fi

ids=$(running_services); [[ -z "$ids" ]] || docker rm -f $ids >/dev/null
# Judge, request-thread and sandbox limits are totals, split across the processes.
per_process() { echo $(( ($1 + REWARD_PROCESSES - 1) / REWARD_PROCESSES )); }
names=()
for ((i = 0; i < REWARD_PROCESSES; i++)); do
    if (( REWARD_PROCESSES == 1 )); then name=$CONTAINER; cache=$BASE_DIR/cache/scorer_cache
    else name=$CONTAINER-$i; cache=$BASE_DIR/cache/scorer_cache/w$i; fi
    names+=("$name")
    mkdir -p "$cache"
    # The docker socket lets the service run APPS code in sibling sandbox containers on this host.
    docker run -d --name "$name" --net=host --ipc=host --restart=unless-stopped \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v "$EVAL_REPO:/opt/chimera-eval" \
        -v "$DATA_DIR:/data/datasets/$DATASET_NAME:ro" \
        -v "$cache:/data/cache/scorer_cache" \
        -w /opt/chimera-eval --entrypoint python3 \
        -e JUDGE_URL="$JUDGE_URL" -e JUDGE_NAME="$JUDGE_NAME" \
        -e JUDGE_CONTEXT="$JUDGE_CONTEXT" -e JUDGE_MAX_TOKENS="$JUDGE_MAX_TOKENS" \
        -e JUDGE_MAX_RETRY_TOKENS="$JUDGE_MAX_RETRY_TOKENS" -e JUDGE_CONCURRENCY="$(per_process "$JUDGE_CONCURRENCY")" \
        -e JUDGE_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}' -e CODE_IMAGE="$EVAL_IMAGE" \
        -e CODE_CONCURRENCY="$(per_process "$CODE_CONCURRENCY")" -e REQUEST_TIMEOUT="$JUDGE_REQUEST_TIMEOUT" \
        -e REQUEST_RETRIES="$JUDGE_REQUEST_RETRIES" -e JUDGE_ATTEMPTS="$JUDGE_ATTEMPTS" \
        "$EVAL_IMAGE" -m eval_stack.reward_service \
        --data-dir "/data/datasets/$DATASET_NAME" --cache-dir /data/cache/scorer_cache \
        --host 127.0.0.1 --port "$((REWARD_PORT + i))" --workers "$(per_process "$REWARD_WORKERS")" \
        --judge-revision "$JUDGE_NAME" >/dev/null
done

last_port=$((REWARD_PORT + REWARD_PROCESSES - 1))
echo "Waiting for the reward service on port$( (( REWARD_PROCESSES > 1 )) && echo "s $REWARD_PORT-$last_port" || echo " $REWARD_PORT")..."
for _ in $(seq 1 90); do
    ready=0
    for ((i = 0; i < REWARD_PROCESSES; i++)); do
        curl -sf "http://127.0.0.1:$((REWARD_PORT + i))/health" >/dev/null && ready=$((ready + 1))
    done
    if (( ready == REWARD_PROCESSES )); then
        python3 - "$(curl -sf "http://127.0.0.1:$REWARD_PORT/health")" "$REWARD_PROCESSES" <<'PY'
import json, sys
h, processes = json.loads(sys.argv[1]), int(sys.argv[2])
judge = h['judge']
where = f" ({processes} processes)" if processes > 1 else ""
print(f"Reward service ready{where}. Judge {judge['model']}: {'reachable' if judge['ready'] else 'not reachable'}.")
for task, error in sorted(h['task_errors'].items()):
    print(f"  cannot grade {task}: {error}")
PY
        exit 0
    fi
    for name in "${names[@]}"; do
        docker ps -q --filter "name=^$name\$" | grep -q . || { failed=$name; break 2; }
    done
    sleep 2
done
echo "error: reward service did not become ready; log of ${failed:-${names[0]}}:" >&2
docker logs --tail 40 "${failed:-${names[0]}}" >&2 || true
exit 1
