#!/usr/bin/env bash
# ==============================================================================
# 01_host_judge.sh - Host Glimmer Judge via vLLM on 2xH200 GPUs
# ==============================================================================
set -Eeuo pipefail

BASE_DIR="/nvme_zone3/home/ekamai1/chimera/mixrl"
MODEL_PATH="${MODEL_PATH:-$BASE_DIR/models/Muse-Glimmer-30B}"
ASSISTANT_PATH="${ASSISTANT_PATH:-$BASE_DIR/models/Muse-Glimmer-30B-assistant}"
JUDGE_PORT="${JUDGE_PORT:-8025}"
JUDGE_GPUS="${JUDGE_GPUS:-4,5}"           # 2 dedicated H200 GPUs for the judge
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"        # Matches 64 judge concurrency slots
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
USE_DOCKER="${USE_DOCKER:-0}"             # 1: Use vllm Docker image, 0: native vllm CLI

echo "=== Starting Glimmer Judge Server (vLLM) ==="
echo "Model: $MODEL_PATH"
echo "GPUs: $JUDGE_GPUS | Port: $JUDGE_PORT | TP: $TENSOR_PARALLEL_SIZE | Context: $MAX_MODEL_LEN"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[ERROR] Judge model directory not found: $MODEL_PATH" >&2
    exit 1
fi

# Detect DFlash speculative decoding drafter model if present
EXTRA_ARGS=()
if [[ -d "$ASSISTANT_PATH" ]]; then
    echo "[INFO] DFlash assistant drafter detected at: $ASSISTANT_PATH"
    SPECULATIVE_CONFIG=$(python3 -c 'import json,sys; print(json.dumps({"method":"dflash","model":sys.argv[1],"num_speculative_tokens":int(sys.argv[2])}))' "$ASSISTANT_PATH" 15)
    EXTRA_ARGS+=(--speculative-config "$SPECULATIVE_CONFIG")
fi

if [[ "$USE_DOCKER" == 1 ]]; then
    echo "Launching vLLM in Docker container (mixrl-judge-server)..."
    docker rm -f mixrl-judge-server 2>/dev/null || true
    docker run -d --name mixrl-judge-server \
      --gpus "device=$JUDGE_GPUS" \
      --ipc=host \
      --net=host \
      --restart=unless-stopped \
      -v "$BASE_DIR/models:/models:ro" \
      -v "$BASE_DIR/cache/vllm_cache:/root/.cache/vllm" \
      vllm/vllm-openai:muse-glimmer \
      /models/Muse-Glimmer-30B \
      --host 127.0.0.1 \
      --port "$JUDGE_PORT" \
      --served-model-name mixrl-judge \
      --dtype bfloat16 \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --enable-prefix-caching \
      --generation-config auto \
      --reasoning-parser muse_glimmer \
      "${EXTRA_ARGS[@]}"
else
    echo "Launching native vLLM serve..."
    CUDA_VISIBLE_DEVICES="$JUDGE_GPUS" exec vllm serve "$MODEL_PATH" \
      --host 127.0.0.1 \
      --port "$JUDGE_PORT" \
      --served-model-name mixrl-judge \
      --dtype bfloat16 \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --enable-prefix-caching \
      --generation-config auto \
      --reasoning-parser muse_glimmer \
      "${EXTRA_ARGS[@]}"
fi
