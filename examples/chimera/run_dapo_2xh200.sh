#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

source "$REPO_ROOT/scripts/models/chimera.sh"

: "${CHIMERA_TRANSFORMERS_ROOT:?Set CHIMERA_TRANSFORMERS_ROOT to the Chimera Transformers checkout}"
: "${HF_CHECKPOINT:?Set HF_CHECKPOINT to the Chimera HF checkpoint}"
: "${MCORE_CHECKPOINT:?Set MCORE_CHECKPOINT to its Bridge-produced MCore torch_dist checkpoint}"
: "${PROMPT_DATA:?Set PROMPT_DATA to dapo-math-17k.jsonl}"
: "${AIME_DATA:?Set AIME_DATA to aime-2024.jsonl}"
: "${SAVE_DIR:?Set SAVE_DIR to a writable output directory}"

for required_file in \
    "$CHIMERA_TRANSFORMERS_ROOT/src/transformers/models/chimera/__init__.py" \
    "$HF_CHECKPOINT/config.json" \
    "$PROMPT_DATA" \
    "$AIME_DATA"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $required_file" >&2
        exit 1
    fi
done
if [[ ! -d "$MCORE_CHECKPOINT" ]]; then
    echo "MCore checkpoint directory not found: $MCORE_CHECKPOINT" >&2
    exit 1
fi

mkdir -p "$SAVE_DIR"

# Keep the image's Transformers and patched Megatron first-class. The runtime
# directory contributes only sitecustomize.py, which registers Chimera from the
# external model directory without replacing the installed Transformers tree.
export PYTHONPATH="$SCRIPT_DIR/runtime:$REPO_ROOT:/root/Megatron-LM"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVSHMEM_DISABLE_NCCL=1

python3 "$SCRIPT_DIR/preflight.py"

RUN_MODE=${RUN_MODE:-smoke}
case "$RUN_MODE" in
    smoke)
        NUM_ROLLOUT=${NUM_ROLLOUT:-1}
        ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-2}
        N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
        OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-4}
        ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-1024}
        EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-2048}
        N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT:-1}
        EVAL_INTERVAL=${EVAL_INTERVAL:-1}
        SAVE_INTERVAL=${SAVE_INTERVAL:-1}
        ;;
    train)
        NUM_ROLLOUT=${NUM_ROLLOUT:-3000}
        ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
        N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
        OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-16}
        ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-8192}
        EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-8192}
        N_SAMPLES_PER_EVAL_PROMPT=${N_SAMPLES_PER_EVAL_PROMPT:-8}
        EVAL_INTERVAL=${EVAL_INTERVAL:-20}
        SAVE_INTERVAL=${SAVE_INTERVAL:-20}
        ;;
    *)
        echo "RUN_MODE must be smoke or train, got: $RUN_MODE" >&2
        exit 1
        ;;
esac

if ((OVER_SAMPLING_BATCH_SIZE <= ROLLOUT_BATCH_SIZE)); then
    echo "OVER_SAMPLING_BATCH_SIZE must be greater than ROLLOUT_BATCH_SIZE for DAPO filtering" >&2
    exit 1
fi

GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))

CKPT_ARGS=(
    --hf-checkpoint "$HF_CHECKPOINT"
    --ref-load "$MCORE_CHECKPOINT"
    --load "$SAVE_DIR"
    --save "$SAVE_DIR"
    --save-interval "$SAVE_INTERVAL"
)

ROLLOUT_ARGS=(
    --prompt-data "$PROMPT_DATA"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type deepscaler
    --num-rollout "$NUM_ROLLOUT"
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
    --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
    --rollout-temperature 1.0
    --over-sampling-batch-size "$OVER_SAMPLING_BATCH_SIZE"
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --num-steps-per-rollout 1
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --balance-data
)

EVAL_ARGS=(
    --eval-interval "$EVAL_INTERVAL"
    --eval-prompt-data aime24 "$AIME_DATA"
    --n-samples-per-eval-prompt "$N_SAMPLES_PER_EVAL_PROMPT"
    --eval-max-response-len "$EVAL_MAX_RESPONSE_LEN"
    --eval-top-p 1.0
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.0
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-6}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    # Two independent TP=1 engines; Slime's router supplies rollout DP=2.
    --rollout-num-gpus 2
    --rollout-num-gpus-per-engine 1
    --sglang-model-impl transformers
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.7}"
    --sglang-cuda-graph-max-bs-decode "${SGLANG_CUDA_GRAPH_MAX_BS:-16}"
    --sglang-enable-metrics
)

MISC_ARGS=(
    --custom-model-provider-path slime_plugins.models.chimera.model_provider
    --model-name chimera
    --attention-backend flash
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --actor-num-nodes 1
    --actor-num-gpus-per-node 2
    --num-gpus-per-node 2
    --colocate
)

if [[ "$RUN_MODE" == smoke ]]; then
    MISC_ARGS+=(--ci-test --ci-train-rollout-logprob-abs-diff-threshold 0.1)
fi

ray stop --force >/dev/null 2>&1 || true
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT
ray start \
    --head \
    --node-ip-address "${MASTER_ADDR:-127.0.0.1}" \
    --num-gpus 2 \
    --disable-usage-stats \
    --dashboard-host 0.0.0.0 \
    --dashboard-port "${RAY_DASHBOARD_PORT:-8265}"

RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

print(
    json.dumps(
        {
            "env_vars": {
                key: os.environ[key]
                for key in (
                    "CHIMERA_TRANSFORMERS_ROOT",
                    "CUDA_DEVICE_MAX_CONNECTIONS",
                    "NVSHMEM_DISABLE_NCCL",
                    "PYTHONPATH",
                    "PYTHONUNBUFFERED",
                )
            }
        }
    )
)
PY
)

ray job submit \
    --address="http://${MASTER_ADDR:-127.0.0.1}:${RAY_DASHBOARD_PORT:-8265}" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- python3 "$REPO_ROOT/train.py" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"
