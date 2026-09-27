#!/usr/bin/env bash
# ==============================================================================
# Chimera 10B Adam MixRL - Unified Container Entrypoint
# ==============================================================================
#
# Instructions:
# 1. Edit the configuration block below or provide environment variable overrides.
# 2. Ensure the reward service is running at MIXRL_SCORER_URL.
# 3. Run: bash run_mixrl.sh (or DRY_RUN=1 bash run_mixrl.sh for verification).
# ==============================================================================

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ------------------------------------------------------------------------------
# 1. Hardware & Cluster Topology
# ------------------------------------------------------------------------------
export POLICY_GPUS=${POLICY_GPUS:-4}                           # Total policy GPUs (actor + rollout)
export ROLLOUT_GPUS=${ROLLOUT_GPUS:-$POLICY_GPUS}              # Rollout engines (matches policy GPUs)
export EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1} # Expert parallelism: 1 (DP4/EP1) or 2 (DP2/EP2)
export COLOCATE=${COLOCATE:-1}                                 # 1: Actor & Rollout share GPUs (standard)
export RECIPE=mixrl
export MODEL_PROFILE=chimera
export CHIMERA_MODEL_SIZE=full

# ------------------------------------------------------------------------------
# 2. Filesystem Mounts (Explicit Paths - No Auto-Discovery)
# ------------------------------------------------------------------------------
# Default paths assume standard container volume mounts under /data or custom paths:
export DATA_ROOT=${DATA_ROOT:-/data}
export HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA_ROOT/models/chimera-muon-nemotron-105k-yarn32k-iter3478/hf}
export MCORE_CHECKPOINT=${MCORE_CHECKPOINT:-$DATA_ROOT/models/chimera-muon-nemotron-105k-yarn32k-iter3478/mcore}
export MIXRL_DATA_DIR=${MIXRL_DATA_DIR:-$DATA_ROOT/datasets/chimera-eval-data}
export MIXRL_CODE_AUDIT_DIR=${MIXRL_CODE_AUDIT_DIR:-$MIXRL_DATA_DIR/audits/apps}
export CHIMERA_TRANSFORMERS_ROOT=${CHIMERA_TRANSFORMERS_ROOT:-/workspace/transformers}
export RUNS_ROOT=${RUNS_ROOT:-$DATA_ROOT/runs}
export RUN_NAME=${RUN_NAME:-chimera-mixrl-4gpus-512x8-$(date +%Y%m%d-%H%M%S)}

# ------------------------------------------------------------------------------
# 3. Reward Microservice (Scorer) Endpoint
# ------------------------------------------------------------------------------
# Host running eval_stack.reward_service (which connects to the local vLLM judge)
export MIXRL_SCORER_URL=${MIXRL_SCORER_URL:-http://127.0.0.1:18020}

# ------------------------------------------------------------------------------
# 4. Batch Geometry & Quotas (512 Prompts x 8 Responses = 4,096 Samples/Step)
# ------------------------------------------------------------------------------
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-512}           # Prompts per boundary
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}         # Responses per prompt group
export OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-$ROLLOUT_BATCH_SIZE}

# 16-Domain training quotas scaled proportionally to sum to exactly 512:
DEFAULT_512_QUOTAS='{"gsm8k_train":48,"nemotron_math":48,"mcqa":32,"openqa":32,"science":16,"hotpot_train":96,"cascade_chat":64,"cascade_lists":16,"cascade_plans":16,"nvidia_multichallenge":32,"nvidia_multichallenge_advanced":16,"nemotron_if":32,"structured_train":16,"reasoning_gym":16,"calendar":16,"apps":16}'
export MIXRL_QUOTAS=${MIXRL_QUOTAS:-$DEFAULT_512_QUOTAS}

# 16K sequence budget allowances per route:
DEFAULT_16K_CAPS='{"gsm8k_train":8192,"nemotron_math":8192,"mcqa":4096,"openqa":4096,"science":8192,"hotpot_train":4096,"cascade_chat":8192,"cascade_lists":8192,"cascade_plans":8192,"nvidia_multichallenge":4096,"nvidia_multichallenge_advanced":4096,"nemotron_if":4096,"structured_train":8192,"reasoning_gym":8192,"calendar":8192,"apps":8192}'
export MIXRL_CAPS=${MIXRL_CAPS:-$DEFAULT_16K_CAPS}

# High-Throughput Concurrency (Prevents reward scoring bottleneck on 4,096 samples):
export MIXRL_REWARD_CONCURRENCY=${MIXRL_REWARD_CONCURRENCY:-64}     # Concurrent requests to reward service
export MIXRL_RESPONSE_CONCURRENCY=${MIXRL_RESPONSE_CONCURRENCY:-64} # Concurrent generation requests
export MIXRL_INFLIGHT_GROUPS=${MIXRL_INFLIGHT_GROUPS:-16}           # In-flight prompt groups in rollout

# ------------------------------------------------------------------------------
# 5. Optimization & Context Geometry
# ------------------------------------------------------------------------------
export NUM_ROLLOUT=${NUM_ROLLOUT:-100}                         # Total rollout boundaries
export LR=${LR:-1e-6}                                          # Adam learning rate
export WEIGHT_DECAY=${WEIGHT_DECAY:-0}                         # Locked Adam MixRL baseline: zero weight decay
export CLIP_GRAD=${CLIP_GRAD:-1.0}
export ADAM_BETA1=${ADAM_BETA1:-0.9}
export ADAM_BETA2=${ADAM_BETA2:-0.98}
export CONTEXT_PHASE=${CONTEXT_PHASE:-32k}                     # YaRN factor 4.0 (32,768 positional maximum)
export TRAIN_SEQUENCE_LENGTH=${TRAIN_SEQUENCE_LENGTH:-16384}   # Run sequence ceiling
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}
export MIXRL_CONTEXT_HEADROOM=${MIXRL_CONTEXT_HEADROOM:-1024}  # Reserved prompt headroom
export MIXRL_OBJECTIVE=${MIXRL_OBJECTIVE:-mimo}                # MiMo detached-IS objective
export CHIMERA_FP32_LM_HEAD=${CHIMERA_FP32_LM_HEAD:-1}         # Matched FP32 LM head projection
export CHIMERA_ROUTING_REPLAY=${CHIMERA_ROUTING_REPLAY:-1}     # Route-replay (R3) enabled

# ------------------------------------------------------------------------------
# 6. Evaluation, Checkpointing & Diagnostics Cadence
# ------------------------------------------------------------------------------
export EVAL_INTERVAL=${EVAL_INTERVAL:-10}                      # Quick eval (rl_val pass@4) every N boundaries
export MIXRL_MAIN_EVAL_INTERVAL=${MIXRL_MAIN_EVAL_INTERVAL:-50} # Main eval every N boundaries
export MIXRL_EVAL_SAMPLES=${MIXRL_EVAL_SAMPLES:-4}
export SAVE_INTERVAL=${SAVE_INTERVAL:-50}                      # Checkpoint save interval
export NO_SAVE_OPTIM=${NO_SAVE_OPTIM:-0}                       # 1: Save model weights only

# ------------------------------------------------------------------------------
# 7. Execution Controls
# ------------------------------------------------------------------------------
export DRY_RUN=${DRY_RUN:-0}                                   # 1: Manifest generation & preflight only
export RESUME=${RESUME:-0}                                     # 1: Resume from existing checkpoint in RUN_NAME

# ==============================================================================
# ENFORCE AIRGAPPED / OFFLINE ENVIRONMENT
# ==============================================================================
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVSHMEM_DISABLE_NCCL=1

# ==============================================================================
# PRE-FLIGHT VALIDATION & PREREQUISITE CHECKS
# ==============================================================================
echo "=== Chimera 10B Adam MixRL Preflight Check ==="
echo "Node GPUs: $POLICY_GPUS | Batch: $ROLLOUT_BATCH_SIZE prompts x $N_SAMPLES_PER_PROMPT responses | Sequence Cap: $TRAIN_SEQUENCE_LENGTH"

# 1. Verify required files exist at specified paths
REQUIRED_FILES=(
    "$HF_CHECKPOINT/config.json"
    "$MCORE_CHECKPOINT/latest_checkpointed_iteration.txt"
    "$MIXRL_DATA_DIR/manifest.json"
    "$MIXRL_DATA_DIR/splits/rl_train.jsonl"
    "$MIXRL_DATA_DIR/splits/rl_val.jsonl"
    "$CHIMERA_TRANSFORMERS_ROOT/src/transformers/models/chimera/__init__.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "[ERROR] Required file not found: $file" >&2
        echo "Please check your mounted paths or edit the configuration block at the top of run_mixrl.sh." >&2
        exit 1
    fi
done

# 2. Check reward service endpoint
if ! curl -sf --connect-timeout 5 "$MIXRL_SCORER_URL/health" >/dev/null 2>&1; then
    echo "[ERROR] Cannot connect to reward service at: $MIXRL_SCORER_URL/health" >&2
    echo "The reward service (eval_stack.reward_service) must be running before starting training." >&2
    echo "See AIRGAPPED_RUN.md for instructions on starting the reward service." >&2
    exit 1
fi
echo "[OK] Reward service is healthy at: $MIXRL_SCORER_URL"

# 3. Check visible GPUs when not in dry-run mode
if [[ "$DRY_RUN" != 1 ]]; then
    VISIBLE_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
    if (( VISIBLE_GPUS < POLICY_GPUS )); then
        echo "[ERROR] Requested POLICY_GPUS=$POLICY_GPUS, but only $VISIBLE_GPUS CUDA devices visible." >&2
        exit 1
    fi
    echo "[OK] Visible CUDA devices: $VISIBLE_GPUS"
fi

# ==============================================================================
# EXECUTE TRAINING LAUNCHER
# ==============================================================================
echo "Launching training pipeline..."
exec bash "$SCRIPT_DIR/examples/chimera/train.sh" "$@"
