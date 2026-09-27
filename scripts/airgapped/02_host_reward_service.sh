#!/usr/bin/env bash
# ==============================================================================
# 02_host_reward_service.sh - Host chimera-eval Reward Microservice
# ==============================================================================
set -Eeuo pipefail

BASE_DIR="/nvme_zone3/home/ekamai1/chimera/mixrl"
SCORER_PORT="${SCORER_PORT:-18020}"
JUDGE_URL="${JUDGE_URL:-http://127.0.0.1:8025/v1}"
JUDGE_NAME="${JUDGE_NAME:-mixrl-judge}"
WORKERS="${WORKERS:-64}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-64}"

echo "=== Starting Reward Microservice (chimera-reward-service) ==="
echo "Target Judge: $JUDGE_URL (model: $JUDGE_NAME)"
echo "Workers: $WORKERS | Concurrency: $JUDGE_CONCURRENCY | Port: $SCORER_PORT"

# 1. Verify judge server is reachable before starting reward service
if ! curl -sf --connect-timeout 5 "$JUDGE_URL/models" >/dev/null 2>&1; then
    echo "[ERROR] Cannot connect to judge at: $JUDGE_URL/models" >&2
    echo "Please ensure the judge server (01_host_judge.sh) is running first." >&2
    exit 1
fi
echo "[OK] Judge server is live."

# 2. Stop any existing container instance
docker rm -f chimera-reward-service 2>/dev/null || true

# 3. Launch reward microservice daemon container
mkdir -p "$BASE_DIR/cache/scorer_cache"

docker run -d --name chimera-reward-service \
  --net=host \
  --ipc=host \
  --restart=unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$BASE_DIR/repos/chimera-eval:/opt/chimera-eval" \
  -v "$BASE_DIR/datasets/chimera-eval-data:/data/datasets/chimera-eval-data" \
  -v "$BASE_DIR/cache/scorer_cache:/data/cache/scorer_cache" \
  -w /opt/chimera-eval \
  --entrypoint python3 \
  -e JUDGE_URL="$JUDGE_URL" \
  -e JUDGE_NAME="$JUDGE_NAME" \
  -e JUDGE_CONTEXT=16384 \
  -e JUDGE_MAX_TOKENS=1024 \
  -e JUDGE_MAX_RETRY_TOKENS=2048 \
  -e JUDGE_CONCURRENCY="$JUDGE_CONCURRENCY" \
  -e JUDGE_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}' \
  suryavikram6/chimera-eval:0.1.1 \
  -m eval_stack.reward_service \
    --data-dir /data/datasets/chimera-eval-data \
    --cache-dir /data/cache/scorer_cache \
    --host 127.0.0.1 \
    --port "$SCORER_PORT" \
    --workers "$WORKERS" \
    --judge-revision "$JUDGE_NAME"

echo "Waiting for reward microservice readiness on port $SCORER_PORT..."
for i in {1..30}; do
    if curl -sf "http://127.0.0.1:$SCORER_PORT/health" >/dev/null 2>&1; then
        echo "[OK] Reward microservice is healthy!"
        curl -s "http://127.0.0.1:$SCORER_PORT/health" | jq . || curl -s "http://127.0.0.1:$SCORER_PORT/health"
        exit 0
    fi
    sleep 1
done

echo "[ERROR] Reward microservice failed to become ready within 30 seconds." >&2
docker logs chimera-reward-service
exit 1
