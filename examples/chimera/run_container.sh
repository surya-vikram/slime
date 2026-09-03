#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "$REPO_ROOT/../.." && pwd)

SLIME_IMAGE=${SLIME_IMAGE:-slimerl/slime@sha256:f7f8ee9acde9645a6e88f0c703597e69a58d2892abff56071630c88f23d5068f}
TRANSFORMERS_ROOT=${TRANSFORMERS_ROOT:-$REPO_ROOT/../transformers}
HF_CHECKPOINT_HOST=${HF_CHECKPOINT_HOST:-$WORKSPACE_ROOT/models/qb_160b_pt}
MCORE_CHECKPOINT_HOST=${MCORE_CHECKPOINT_HOST:-$WORKSPACE_ROOT/models/qb_160b_pt_mcore}
PROMPT_DATA_HOST=${PROMPT_DATA_HOST:-$WORKSPACE_ROOT/data/dapo-math-17k/dapo-math-17k.jsonl}
AIME_DATA_HOST=${AIME_DATA_HOST:-$WORKSPACE_ROOT/data/aime-2024/aime-2024.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-$WORKSPACE_ROOT/runs/chimera-dapo}

required_paths=(
    "$TRANSFORMERS_ROOT/src/transformers/models/chimera/__init__.py"
    "$HF_CHECKPOINT_HOST/config.json"
    "$MCORE_CHECKPOINT_HOST"
    "$PROMPT_DATA_HOST"
    "$AIME_DATA_HOST"
)
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Required host path not found: $required_path" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_ROOT"

docker run --rm \
    --gpus all \
    --ipc host \
    --network host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$REPO_ROOT:/workspace/slime" \
    -v "$TRANSFORMERS_ROOT:/workspace/transformers:ro" \
    -v "$HF_CHECKPOINT_HOST:/workspace/model/hf:ro" \
    -v "$MCORE_CHECKPOINT_HOST:/workspace/model/mcore:ro" \
    -v "$PROMPT_DATA_HOST:/workspace/data/dapo-math-17k.jsonl:ro" \
    -v "$AIME_DATA_HOST:/workspace/data/aime-2024.jsonl:ro" \
    -v "$OUTPUT_ROOT:/workspace/output" \
    -e CHIMERA_TRANSFORMERS_ROOT=/workspace/transformers \
    -e HF_CHECKPOINT=/workspace/model/hf \
    -e MCORE_CHECKPOINT=/workspace/model/mcore \
    -e PROMPT_DATA=/workspace/data/dapo-math-17k.jsonl \
    -e AIME_DATA=/workspace/data/aime-2024.jsonl \
    -e SAVE_DIR=/workspace/output \
    -e RUN_MODE="${RUN_MODE:-smoke}" \
    -e MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" \
    -e RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}" \
    -e NUM_ROLLOUT \
    -e ROLLOUT_BATCH_SIZE \
    -e N_SAMPLES_PER_PROMPT \
    -e OVER_SAMPLING_BATCH_SIZE \
    -e ROLLOUT_MAX_RESPONSE_LEN \
    -e EVAL_MAX_RESPONSE_LEN \
    -e N_SAMPLES_PER_EVAL_PROMPT \
    -e EVAL_INTERVAL \
    -e SAVE_INTERVAL \
    -e MAX_TOKENS_PER_GPU \
    -e LR \
    -e SGLANG_MEM_FRACTION_STATIC \
    -e SGLANG_CUDA_GRAPH_MAX_BS \
    -w /workspace/slime \
    "$SLIME_IMAGE" \
    bash examples/chimera/run_dapo_2xh200.sh
