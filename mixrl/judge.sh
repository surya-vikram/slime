#!/usr/bin/env bash
# Start the judge (vLLM) used by judge-graded tasks. Settings: mixrl/config.env.
# Not needed when every enabled task is judge-free (mixrl/run.sh tasks shows which need it).
#
#   mixrl/judge.sh           start the judge (native vLLM, or JUDGE_USE_DOCKER=1 for the image)
#   mixrl/judge.sh stop      stop the Docker judge
set -Eeuo pipefail

MIXRL_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$MIXRL_DIR/config.env"
CONTAINER=mixrl-judge-server

case "${1:-start}" in
    start) ;;
    stop) docker rm -f "$CONTAINER" >/dev/null 2>&1 && echo "Stopped $CONTAINER." || echo "$CONTAINER is not running."; exit 0 ;;
    *) sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac

MODEL_PATH=$BASE_DIR/models/$JUDGE_MODEL_DIR
DRAFTER_PATH=$BASE_DIR/models/$JUDGE_MODEL_DIR-assistant
[[ -d "$MODEL_PATH" ]] || { echo "error: judge model not found at $MODEL_PATH (JUDGE_MODEL_DIR)" >&2; exit 1; }
IFS=, read -ra judge_gpus <<< "$JUDGE_GPUS"
[[ ${#judge_gpus[@]} -eq $((JUDGE_TP * JUDGE_DP)) ]] || {
    echo "error: JUDGE_GPUS=$JUDGE_GPUS lists ${#judge_gpus[@]} GPUs but JUDGE_TP x JUDGE_DP = $JUDGE_TP x $JUDGE_DP" >&2; exit 1; }
(( JUDGE_CONTEXT <= JUDGE_MAX_MODEL_LEN )) || { echo "error: JUDGE_CONTEXT must not exceed JUDGE_MAX_MODEL_LEN" >&2; exit 1; }

# Paths below are as the server sees them: the host path, or /models inside the container.
serve_args() {
    local models=$1
    args=(--host "$JUDGE_HOST" --port "$JUDGE_PORT" --served-model-name "$JUDGE_NAME" --dtype bfloat16
        --tensor-parallel-size "$JUDGE_TP" --data-parallel-size "$JUDGE_DP"
        --gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTILIZATION"
        --max-model-len "$JUDGE_MAX_MODEL_LEN" --kv-cache-dtype "$JUDGE_KV_CACHE_DTYPE"
        --max-num-batched-tokens "$JUDGE_MAX_NUM_BATCHED_TOKENS" --api-server-count "$JUDGE_API_SERVERS"
        --enable-prefix-caching --enable-auto-tool-choice
        --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer)
    # Like chimera-eval serve_gemma.sh: throughput mode; JUDGE_MAX_NUM_SEQS raises vLLM's request cap and the
    # reward service's KV-token budget (JUDGE_KV_CACHE_NUM_TOKENS) bounds the load.
    [[ -n "$JUDGE_MAX_NUM_SEQS" && "$JUDGE_MAX_NUM_SEQS" != 0 ]] && args+=(--max-num-seqs "$JUDGE_MAX_NUM_SEQS")
    [[ -n "$JUDGE_PERFORMANCE_MODE" ]] && args+=(--performance-mode "$JUDGE_PERFORMANCE_MODE")
    [[ "$JUDGE_LANGUAGE_MODEL_ONLY" == 1 ]] && args+=(--language-model-only)  # skip the vision encoder
    (( JUDGE_DP > 1 )) && args+=(--aggregate-engine-logging)
    if [[ "$JUDGE_SPECULATIVE" == 1 && -d "$DRAFTER_PATH" ]]; then
        # DFlash speculative decoding when the drafter model is present.
        args+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$models/$JUDGE_MODEL_DIR-assistant\",\"num_speculative_tokens\":$JUDGE_SPECULATIVE_TOKENS}")
    fi
}

echo "Judge $JUDGE_NAME on GPUs $JUDGE_GPUS (TP=$JUDGE_TP, DP=$JUDGE_DP) at http://$JUDGE_HOST:$JUDGE_PORT/v1, context $JUDGE_MAX_MODEL_LEN"
if [[ "$JUDGE_USE_DOCKER" == 1 ]]; then
    serve_args /models
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    # Docker would create a missing mount folder as root, and then reward.sh (run as you) could not
    # create its cache next to it in $BASE_DIR/cache.
    mkdir -p "$BASE_DIR/cache/vllm_cache"
    # Inner quotes keep a comma-separated device list as one value for Docker.
    docker run -d --name "$CONTAINER" --gpus "\"device=$JUDGE_GPUS\"" --ipc=host --net=host --restart=unless-stopped \
        -v "$BASE_DIR/models:/models:ro" -v "$BASE_DIR/cache/vllm_cache:/root/.cache/vllm" \
        "$JUDGE_IMAGE" "/models/$JUDGE_MODEL_DIR" "${args[@]}" >/dev/null
    echo "Started $CONTAINER; waiting for it to serve (up to 30 min; follow with: docker logs -f $CONTAINER)"
    # The restart policy would hide a startup crash as a silent restart loop: report it instead.
    for _ in $(seq 1 360); do
        if curl -sf --connect-timeout 5 "http://$JUDGE_HOST:$JUDGE_PORT/v1/models" >/dev/null 2>&1; then
            echo "Judge ready at http://$JUDGE_HOST:$JUDGE_PORT/v1 ($JUDGE_NAME)."
            exit 0
        fi
        restarts=$(docker inspect -f '{{.RestartCount}}' "$CONTAINER" 2>/dev/null || true)
        if [[ "${restarts:-0}" != 0 ]] || ! docker ps -q --filter "name=^$CONTAINER$" | grep -q .; then
            echo "error: the judge failed during startup; its log:" >&2
            docker logs --tail 40 "$CONTAINER" >&2 || true
            docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
            exit 1
        fi
        sleep 5
    done
    echo "error: the judge is still not serving after 30 min; see: docker logs $CONTAINER" >&2
    exit 1
else
    serve_args "$BASE_DIR/models"
    CUDA_VISIBLE_DEVICES="$JUDGE_GPUS" exec vllm serve "$MODEL_PATH" "${args[@]}"
fi
