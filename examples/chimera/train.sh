#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

# Edit this block, or set the same names in the environment.
DATA_ROOT=${DATA_ROOT:-/datasets/megadata}
RUN_NAME=${RUN_NAME:-chimera-dapo-gsm8k-$(TZ=Asia/Kolkata date +%Y%m%d-%H%M%S)}
HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA_ROOT/models/chimera-10b-hf}
MCORE_CHECKPOINT=${MCORE_CHECKPOINT:-$DATA_ROOT/models/chimera-10b-mcore}
CHIMERA_TRANSFORMERS_ROOT=${CHIMERA_TRANSFORMERS_ROOT:-$(dirname "$REPO_ROOT")/transformers}

# Production defaults. This launcher intentionally supports one 8xH200 node.
NUM_ROLLOUT=${NUM_ROLLOUT:-3000}
LR=${LR:-1e-6}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-64}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-512}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-512}
EVAL_INTERVAL=${EVAL_INTERVAL:-20}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.70}
RESUME=${RESUME:-0}

EXPECTED_GPUS=8
RUNS_ROOT=${RUNS_ROOT:-$DATA_ROOT/runs/chimera/dapo-gsm8k}
RUN_DIR=$RUNS_ROOT/$RUN_NAME
SAVE_PATH=$RUN_DIR/checkpoints
TENSORBOARD_DIR=$RUN_DIR/tensorboard
LOG_DIR=$RUN_DIR/logs
MANIFEST_DIR=$RUN_DIR/manifests
ROLLOUT_DIR=$RUN_DIR/rollouts
TRAIN_DATA=$SCRIPT_DIR/data/gsm8k_train.jsonl
EVAL_DATA=${EVAL_DATA:-$SCRIPT_DIR/data/gsm8k_validation.jsonl}
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}

case "$RUN_NAME" in
    *[!A-Za-z0-9._-]* | "")
        echo "RUN_NAME may contain only letters, digits, dot, underscore, and dash: $RUN_NAME" >&2
        exit 1
        ;;
esac
if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
    echo "RESUME must be 0 or 1, got: $RESUME" >&2
    exit 1
fi
if ((OVER_SAMPLING_BATCH_SIZE <= ROLLOUT_BATCH_SIZE)); then
    echo "OVER_SAMPLING_BATCH_SIZE must exceed ROLLOUT_BATCH_SIZE for strict DAPO filtering" >&2
    exit 1
fi

for required_file in \
    "$CHIMERA_TRANSFORMERS_ROOT/src/transformers/models/chimera/__init__.py" \
    "$HF_CHECKPOINT/config.json" \
    "$MCORE_CHECKPOINT/latest_checkpointed_iteration.txt" \
    "$TRAIN_DATA" \
    "$EVAL_DATA"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $required_file" >&2
        exit 1
    fi
done
if [[ ! -d "$MEGATRON_ROOT/.git" ]]; then
    echo "The Slime image's Megatron checkout was not found at $MEGATRON_ROOT" >&2
    exit 1
fi

if [[ "$RESUME" == 1 ]]; then
    if [[ ! -f "$SAVE_PATH/latest_checkpointed_iteration.txt" ]]; then
        echo "RESUME=1 but no Slime checkpoint exists at $SAVE_PATH" >&2
        exit 1
    fi
elif [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "Fresh run directory is not empty: $RUN_DIR (choose a new RUN_NAME or set RESUME=1)" >&2
    exit 1
fi

mkdir -p "$SAVE_PATH" "$TENSORBOARD_DIR" "$LOG_DIR" "$MANIFEST_DIR" "$ROLLOUT_DIR"

CUDA_GRAPH_PATCH=$SCRIPT_DIR/patches/megatron-yarn-te-cuda-graph.patch
if git -C "$MEGATRON_ROOT" apply --unidiff-zero --reverse --check "$CUDA_GRAPH_PATCH" >/dev/null 2>&1; then
    echo "Megatron YaRN Transformer Engine CUDA-graph fix is already applied."
elif git -C "$MEGATRON_ROOT" apply --unidiff-zero --check "$CUDA_GRAPH_PATCH" >/dev/null 2>&1; then
    git -C "$MEGATRON_ROOT" apply --unidiff-zero "$CUDA_GRAPH_PATCH"
    echo "Applied the YaRN Transformer Engine CUDA-graph fix to $MEGATRON_ROOT."
else
    echo "The YaRN CUDA-graph patch does not match $MEGATRON_ROOT; refusing to run." >&2
    exit 1
fi

# Keep the image's pinned packages. sitecustomize registers only the external
# Chimera package; it does not replace the installed Transformers distribution.
export CHIMERA_TRANSFORMERS_ROOT HF_CHECKPOINT MEGATRON_ROOT TENSORBOARD_DIR
export PYTHONPATH="$SCRIPT_DIR/runtime:$REPO_ROOT:$MEGATRON_ROOT"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVSHMEM_DISABLE_NCCL=1

mapfile -t GPU_NAMES < <(python3 -c 'import torch; [print(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]')
if ((${#GPU_NAMES[@]} != EXPECTED_GPUS)); then
    echo "Chimera production launcher requires exactly $EXPECTED_GPUS visible GPUs; found ${#GPU_NAMES[@]}" >&2
    exit 1
fi
for gpu_name in "${GPU_NAMES[@]}"; do
    if [[ "$gpu_name" != *H200* ]]; then
        echo "Chimera production launcher requires H200 GPUs; found: $gpu_name" >&2
        exit 1
    fi
done

python3 "$SCRIPT_DIR/preflight.py"
source "$REPO_ROOT/scripts/models/chimera.sh"

GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))

CKPT_ARGS=(
    --hf-checkpoint "$HF_CHECKPOINT"
    --ref-load "$MCORE_CHECKPOINT"
    --save "$SAVE_PATH"
    --save-interval "$SAVE_INTERVAL"
)
if [[ "$RESUME" == 1 ]]; then
    CKPT_ARGS+=(--load "$SAVE_PATH")
else
    # The converted SFT checkpoint supplies model weights only. Start fresh
    # optimizer and RNG state while retaining optimizer state in new saves.
    CKPT_ARGS+=(--finetune --no-load-optim --no-load-rng)
fi

ROLLOUT_ARGS=(
    --prompt-data "$TRAIN_DATA"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rollout-seed 42
    --rm-type math
    --num-rollout "$NUM_ROLLOUT"
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
    --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
    --rollout-temperature 1.0
    --rollout-top-p 1.0
    --rollout-stop "<end_of_turn>"
    --over-sampling-batch-size "$OVER_SAMPLING_BATCH_SIZE"
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --num-steps-per-rollout 1
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --balance-data
    --log-passrate
)

EVAL_ARGS=(
    --eval-interval "$EVAL_INTERVAL"
    --eval-prompt-data gsm8k_validation "$EVAL_DATA"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len "$EVAL_MAX_RESPONSE_LEN"
    --eval-temperature 0.0
    --eval-top-p 1.0
    --eval-top-k 1
)

DAPO_ARGS=(
    --advantage-estimator grpo
    --calculate-per-token-loss
    --kl-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
)

PARALLEL_ARGS=(
    # With every model-parallel dimension at one, all eight actor ranks form
    # Megatron's data-parallel group. The optimizer state is sharded over DP=8.
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --use-dynamic-batch-size
    --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
    --cuda-graph-impl transformer_engine
    --cuda-graph-scope attn
    --cuda-graph-warmup-steps 1
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "$LR"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --use-precision-aware-optimizer
    --main-params-dtype fp32
    --main-grads-dtype fp32
    --exp-avg-dtype fp32
    --exp-avg-sq-dtype fp32
)

SGLANG_ARGS=(
    # Slime creates eight independent TP=1 engines: rollout replica DP=8.
    --rollout-num-gpus 8
    --rollout-num-gpus-per-engine 1
    --sglang-model-impl transformers
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC"
    --sglang-cuda-graph-max-bs-decode 32
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
    --actor-num-gpus-per-node 8
    --num-gpus-per-node 8
    --colocate
    --use-tensorboard
    --tensorboard-dir "$TENSORBOARD_DIR"
    --log-interval 1
)
if [[ "${DUMP_DETAILS:-0}" == 1 ]]; then
    MISC_ARGS+=(--dump-details "$ROLLOUT_DIR")
fi

TRAIN_COMMAND=(
    python3 "$REPO_ROOT/train.py"
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${EVAL_ARGS[@]}"
    "${DAPO_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${MISC_ARGS[@]}"
)

cp "$0" "$MANIFEST_DIR/train.sh"
git -C "$REPO_ROOT" rev-parse HEAD > "$MANIFEST_DIR/slime_commit.txt"
git -C "$CHIMERA_TRANSFORMERS_ROOT" rev-parse HEAD > "$MANIFEST_DIR/transformers_commit.txt"
git -C "$MEGATRON_ROOT" rev-parse HEAD > "$MANIFEST_DIR/megatron_image_commit.txt"
{
    printf 'DATA_ROOT=%q\n' "$DATA_ROOT"
    printf 'RUN_NAME=%q\n' "$RUN_NAME"
    printf 'RUN_DIR=%q\n' "$RUN_DIR"
    printf 'HF_CHECKPOINT=%q\n' "$HF_CHECKPOINT"
    printf 'MCORE_CHECKPOINT=%q\n' "$MCORE_CHECKPOINT"
    printf 'TRAIN_DATA=%q\n' "$TRAIN_DATA"
    printf 'EVAL_DATA=%q\n' "$EVAL_DATA"
    printf 'RESUME=%q\n' "$RESUME"
} > "$MANIFEST_DIR/run_paths.env"
printf '%q ' "${TRAIN_COMMAND[@]}" > "$MANIFEST_DIR/train_command.sh"
printf '\n' >> "$MANIFEST_DIR/train_command.sh"

echo "Run directory: $RUN_DIR"
echo "Megatron actor: DP=8, TP=PP=CP=EP=ETP=1, distributed optimizer"
echo "SGLang rollout: 8 independent TP=1 engines (DP=8)"
echo "DAPO batch: $ROLLOUT_BATCH_SIZE prompts x $N_SAMPLES_PER_PROMPT responses = $GLOBAL_BATCH_SIZE samples"

ray stop --force >/dev/null 2>&1 || true
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT
ray start \
    --head \
    --node-ip-address "${MASTER_ADDR:-127.0.0.1}" \
    --num-gpus "$EXPECTED_GPUS" \
    --disable-usage-stats \
    --dashboard-host 0.0.0.0 \
    --dashboard-port "${RAY_DASHBOARD_PORT:-8265}"

RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json
import os

keys = (
    "CHIMERA_TRANSFORMERS_ROOT",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "HF_CHECKPOINT",
    "MEGATRON_ROOT",
    "NVSHMEM_DISABLE_NCCL",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "TENSORBOARD_DIR",
)
print(json.dumps({"env_vars": {key: os.environ[key] for key in keys}}))
PY
)

ray job submit \
    --address="http://${MASTER_ADDR:-127.0.0.1}:${RAY_DASHBOARD_PORT:-8265}" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- "${TRAIN_COMMAND[@]}" 2>&1 | tee "$LOG_DIR/train.log"
