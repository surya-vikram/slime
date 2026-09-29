#!/usr/bin/env bash
# Start the reward service (chimera-eval) in its own container. Settings: mixrl/config.env.
# It starts whether or not the judge is up: tasks that never call the judge train without it,
# and training refuses to start any enabled task that needs an unreachable judge.
#
#   mixrl/reward.sh          start (or restart) the reward service
#   mixrl/reward.sh stop     stop it
set -Eeuo pipefail

MIXRL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$MIXRL_DIR/config.env"
CONTAINER=mixrl-reward-service

case "${1:-start}" in
    start) ;;
    stop) docker rm -f "$CONTAINER" >/dev/null 2>&1 && echo "Stopped $CONTAINER." || echo "$CONTAINER is not running."; exit 0 ;;
    *) sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac

DATA_DIR=$BASE_DIR/datasets/$DATASET_NAME
[[ -f "$DATA_DIR/manifest.json" ]] || { echo "error: dataset not found at $DATA_DIR" >&2; exit 1; }
[[ -f "$EVAL_REPO/eval_stack/reward_service.py" ]] || { echo "error: chimera-eval not found at $EVAL_REPO (EVAL_REPO)" >&2; exit 1; }
JUDGE_URL=http://$JUDGE_HOST:$JUDGE_PORT/v1
if curl -sf --connect-timeout 5 "$JUDGE_URL/models" >/dev/null 2>&1; then
    echo "Judge is up at $JUDGE_URL ($JUDGE_NAME)."
else
    echo "Judge not reachable at $JUDGE_URL: only judge-free tasks can train until it is up (mixrl/judge.sh)."
fi

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
mkdir -p "$BASE_DIR/cache/scorer_cache"
# The docker socket lets the service run APPS code in sibling sandbox containers on this host.
docker run -d --name "$CONTAINER" --net=host --ipc=host --restart=unless-stopped \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$EVAL_REPO:/opt/chimera-eval" \
    -v "$DATA_DIR:/data/datasets/$DATASET_NAME:ro" \
    -v "$BASE_DIR/cache/scorer_cache:/data/cache/scorer_cache" \
    -w /opt/chimera-eval --entrypoint python3 \
    -e JUDGE_URL="$JUDGE_URL" -e JUDGE_NAME="$JUDGE_NAME" \
    -e JUDGE_CONTEXT="$JUDGE_CONTEXT" -e JUDGE_MAX_TOKENS="$JUDGE_MAX_TOKENS" \
    -e JUDGE_MAX_RETRY_TOKENS="$JUDGE_MAX_RETRY_TOKENS" -e JUDGE_CONCURRENCY="$JUDGE_CONCURRENCY" \
    -e JUDGE_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}' -e CODE_IMAGE="$EVAL_IMAGE" \
    "$EVAL_IMAGE" -m eval_stack.reward_service \
    --data-dir "/data/datasets/$DATASET_NAME" --cache-dir /data/cache/scorer_cache \
    --host 127.0.0.1 --port "$REWARD_PORT" --workers "$REWARD_WORKERS" --judge-revision "$JUDGE_NAME" >/dev/null

echo "Waiting for the reward service on port $REWARD_PORT..."
for _ in $(seq 1 90); do
    if health=$(curl -sf "http://127.0.0.1:$REWARD_PORT/health"); then
        python3 - "$health" <<'PY'
import json, sys
h = json.loads(sys.argv[1])
judge = h['judge']
print(f"Reward service ready. Judge {judge['model']}: {'reachable' if judge['ready'] else 'not reachable'}.")
for task, error in sorted(h['task_errors'].items()):
    print(f"  cannot grade {task}: {error}")
PY
        exit 0
    fi
    if ! docker ps -q --filter "name=^$CONTAINER$" | grep -q .; then break; fi
    sleep 2
done
echo "error: reward service did not become ready; its log:" >&2
docker logs --tail 40 "$CONTAINER" >&2 || true
exit 1
