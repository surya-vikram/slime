#!/usr/bin/env bash
# ==============================================================================
# 04_run_training.sh - Launch Chimera 10B Adam MixRL Training on 4xH200
# ==============================================================================
set -Eeuo pipefail

BASE_DIR="/nvme_zone3/home/ekamai1/chimera/mixrl"
TRAINING_GPUS="${TRAINING_GPUS:-0,1,2,3}" # e.g. TRAINING_GPUS=0,1 with POLICY_GPUS=2

echo "=== Launching Chimera 10B Adam MixRL Training ==="
echo "Base Directory: $BASE_DIR"
echo "Training GPUs: $TRAINING_GPUS (devices $TRAINING_GPUS)"
echo "Tasks and batch: examples/chimera/mixrl_tasks.json (preview printed below)"

# Host overrides forwarded into the container only when set, e.g.
#   POLICY_GPUS=2 EXPERT_MODEL_PARALLEL_SIZE=2 RUN_NAME=gsm8k-01 bash 04_run_training.sh
OVERRIDES=(RUN_NAME RESUME POLICY_GPUS ROLLOUT_GPUS EXPERT_MODEL_PARALLEL_SIZE N_SAMPLES_PER_PROMPT
    NUM_ROLLOUT LR EVAL_INTERVAL SAVE_INTERVAL MIXRL_TASKS_CONFIG MIXRL_WALLCLOCK_SECONDS MIXRL_STOP_FILE
    MIXRL_INFLIGHT_GROUPS MIXRL_RESPONSE_CONCURRENCY MIXRL_REWARD_CONCURRENCY MIXRL_COLLECTION_TIMEOUT
    SGLANG_CUDA_GRAPH_MAX_BS SGLANG_MEM_FRACTION_STATIC)
OVERRIDE_ARGS=()
for name in "${OVERRIDES[@]}"; do OVERRIDE_ARGS+=(-e "$name"); done

mkdir -p "$BASE_DIR/runs"

docker run -it --rm \
  --gpus "device=$TRAINING_GPUS" \
  --ipc=host \
  --net=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$BASE_DIR/models/chimera-muon-nemotron-105k-yarn32k-iter3478:/data/models/chimera-muon-nemotron-105k-yarn32k-iter3478:ro" \
  -v "$BASE_DIR/datasets/chimera-eval-data:/data/datasets/chimera-eval-data:ro" \
  -v "$BASE_DIR/repos/transformers:/workspace/transformers:ro" \
  -v "$BASE_DIR/repos/slime:/workspace/slime" \
  -v "$BASE_DIR/runs:/data/runs" \
  -w /workspace/slime \
  -e DRY_RUN=0 \
  -e DATA_ROOT=/data \
  -e HF_CHECKPOINT=/data/models/chimera-muon-nemotron-105k-yarn32k-iter3478/hf \
  -e MCORE_CHECKPOINT=/data/models/chimera-muon-nemotron-105k-yarn32k-iter3478/mcore \
  -e MIXRL_DATA_DIR=/data/datasets/chimera-eval-data \
  -e MIXRL_CODE_AUDIT_DIR=/data/datasets/chimera-eval-data/audits/apps \
  -e CHIMERA_TRANSFORMERS_ROOT=/workspace/transformers \
  -e MIXRL_SCORER_URL=http://127.0.0.1:18020 \
  -e RUNS_ROOT=/data/runs \
  "${OVERRIDE_ARGS[@]}" \
  suryavikram6/slime:pinned \
  bash run_mixrl.sh "$@"
