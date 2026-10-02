#!/usr/bin/env bash
# MixRL launcher that runs INSIDE the slime container. Do not run it by hand:
# mixrl/run.sh starts it with the right mounts and settings. Every setting you are
# meant to change has its default in mixrl/config.env; this file only adds internal
# switches, validates everything, and builds the Megatron + SGLang command.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MIXRL_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$MIXRL_DIR/.." && pwd)
CHIMERA_DIR=$REPO_ROOT/examples/chimera  # Chimera model support: runtime registration, patches, preflight

for removed in MIXRL_QUOTAS MIXRL_CAPS MIXRL_EVAL_QUOTAS MIXRL_MAIN_EVAL_QUOTAS ROLLOUT_BATCH_SIZE \
        MIXRL_EVAL_SAMPLES MIXRL_MAIN_EVAL_INTERVAL MIXRL_MAIN_EVAL_SAMPLES; do
    if [[ -n "${!removed:-}" ]]; then
        echo "$removed is no longer read for MixRL; set tasks, prompt counts and eval size in mixrl/tasks.json" >&2
        exit 1
    fi
done
source "$MIXRL_DIR/config.env"

# Internal switches (not in config.env): model profile, precision matching, scheduling.
MODEL_PROFILE=${MODEL_PROFILE:-chimera} # qwen3-0.6B: CPU/reference validation only.
export CHIMERA_MODEL_SIZE=${CHIMERA_MODEL_SIZE:-full} # tiny: canonical 8-layer local mechanics only.
DEFAULT_CHIMERA=0
if [[ "$MODEL_PROFILE" == chimera ]]; then DEFAULT_CHIMERA=1; fi
export CHIMERA_FP32_LM_HEAD=${CHIMERA_FP32_LM_HEAD:-$DEFAULT_CHIMERA} # Matched FP32 projection on actor AND rollout.
export CHIMERA_MATCH_RMSNORM=${CHIMERA_MATCH_RMSNORM:-$DEFAULT_CHIMERA} # SGLang RMSNorm rounding aligned with TE (rollout/train KL 3.0e-4 -> 1.8e-4).
export CHIMERA_MATCH_DENSE_SWIGLU=${CHIMERA_MATCH_DENSE_SWIGLU:-0} # Experimental rollout dense SwiGLU kernel; Tiny-qualified only.
export CHIMERA_SGLANG_FULL_BF16_REDUCTION=${CHIMERA_SGLANG_FULL_BF16_REDUCTION:-0} # Rollout PyTorch BF16 reduction only.
CHIMERA_ROUTING_REPLAY=${CHIMERA_ROUTING_REPLAY:-$DEFAULT_CHIMERA} # Baseline requires R3; 0 is an explicit diagnostic control.
CHAT_TEMPLATE_KWARGS=${CHAT_TEMPLATE_KWARGS:-'{}'}
DATA_ROOT=${DATA_ROOT:-/data}
RUN_NAME=${RUN_NAME:-mixrl-$(date +%Y%m%d-%H%M%S)}
HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA_ROOT/models/$MODEL_NAME/hf}
MCORE_CHECKPOINT=${MCORE_CHECKPOINT:-$DATA_ROOT/models/$MODEL_NAME/mcore}
CHIMERA_TRANSFORMERS_ROOT=${CHIMERA_TRANSFORMERS_ROOT:-/workspace/transformers}
MIXRL_DATA_DIR=${MIXRL_DATA_DIR:-$DATA_ROOT/datasets/$DATASET_NAME}
MIXRL_CODE_AUDIT_DIR=${MIXRL_CODE_AUDIT_DIR:-$MIXRL_DATA_DIR/audits/apps} # Reference+negative sandbox report.
if [[ -z "${MIXRL_SCORER_URL:-}" ]]; then
    # One URL per reward-service process (mixrl/reward.sh starts them on consecutive ports).
    scorer_urls=()
    for ((i = 0; i < ${REWARD_PROCESSES:-1}; i++)); do scorer_urls+=("http://127.0.0.1:$((REWARD_PORT + i))"); done
    MIXRL_SCORER_URL=$(IFS=,; echo "${scorer_urls[*]}")
fi
MIXRL_TASKS_CONFIG=${MIXRL_TASKS_CONFIG:-$MIXRL_DIR/tasks.json}
INITIAL_ACTOR_CHECKPOINT=${INITIAL_ACTOR_CHECKPOINT:-} # Fresh RL from other MCore weights (not optimizer).
SAVE_HF=${SAVE_HF:-0} # Opt-in native HF export beside MCore saves; Qwen validation only.
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-512} # Unused by MixRL (per-task caps); Slime requires it.
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-} # Optional per-engine request bound.
RESUME=${RESUME:-0}
MIXRL_EXTEND_CONSTANT_HORIZON=${MIXRL_EXTEND_CONSTANT_HORIZON:-0} # Explicit resume-only extension; LR/WD schedule stays frozen.
GPU_METRICS_INTERVAL=${GPU_METRICS_INTERVAL:-5} # Seconds; 0 disables nvidia-smi CSV sidecar.
# Opt-in budget starts before model initialization. Reserve includes final eval/save.
MIXRL_HEALTH_WAIT_SECONDS=${MIXRL_HEALTH_WAIT_SECONDS:-0} # >0: before a batch, wait this long for an unreachable judge/reward service instead of stopping.
MIXRL_REWARD_BACKOFF_MAX=${MIXRL_REWARD_BACKOFF_MAX:-8}   # Longest sleep (s) between /score retries.
MIXRL_PIPELINE_SECONDS=${MIXRL_PIPELINE_SECONDS:-30} # MIXRL_PIPELINE line interval during collection; 0 turns it off.
MIXRL_JUDGE_METRICS_URL=${MIXRL_JUDGE_METRICS_URL:-http://${JUDGE_HOST:-127.0.0.1}:${JUDGE_PORT:-8025}/metrics} # Judge load in that line.
export MIXRL_WALLCLOCK_SECONDS MIXRL_STOP_FILE MIXRL_KEEP_TRAIN_SAMPLES MIXRL_HEALTH_WAIT_SECONDS MIXRL_REWARD_BACKOFF_MAX
export MIXRL_PIPELINE_SECONDS MIXRL_JUDGE_METRICS_URL
export MIXRL_FINAL_RESERVE_SECONDS=${MIXRL_FINAL_RESERVE_SECONDS:-1200}
export MIXRL_INITIAL_UPDATE_SECONDS=${MIXRL_INITIAL_UPDATE_SECONDS:-300}
OFFLOAD_TRAIN=${OFFLOAD_TRAIN:-1} # Small-reference-model residency experiment only.
OFFLOAD_ROLLOUT=${OFFLOAD_ROLLOUT:-1}
EXECUTION_MODE=${EXECUTION_MODE:-sync} # async: native one-batch-lookahead experiment, Qwen only.
COLOCATE=${COLOCATE:-1}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$POLICY_GPUS}
CHIMERA_CONTEXT_OVERRIDE=${CHIMERA_CONTEXT_OVERRIDE:-0} # Use HF YaRN over historical MCore positional metadata; record both.
# MODEL_CONTEXT_LENGTH is what the preflight reads; it must equal TRAIN_SEQUENCE_LENGTH.
if [[ -n "${MODEL_CONTEXT_LENGTH:-}" && "$MODEL_CONTEXT_LENGTH" != "$TRAIN_SEQUENCE_LENGTH" ]]; then
    echo "MODEL_CONTEXT_LENGTH and TRAIN_SEQUENCE_LENGTH disagree" >&2; exit 1
fi
MODEL_CONTEXT_LENGTH=$TRAIN_SEQUENCE_LENGTH
MIXRL_REWARD_TIMEOUT=${MIXRL_REWARD_TIMEOUT:-600}
MIXRL_REWARD_ATTEMPTS=${MIXRL_REWARD_ATTEMPTS:-3}
export MIXRL_ROUTER_METRICS=${MIXRL_ROUTER_METRICS:-1} # Per-step expert-load balance over all ranks (MIXRL_ROUTER).
MIXRL_IS_POSITIVE_BOUNDS=${MIXRL_IS_POSITIVE_BOUNDS:-'[0.2,5.0]'}
MIXRL_IS_NEGATIVE_BOUNDS=${MIXRL_IS_NEGATIVE_BOUNDS:-'[0.2,5.0]'}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0} # Task/data/scorer checks only; no model, Ray or GPU work.
DRY_RUN=${DRY_RUN:-0} # Full checks and command build in the pinned image; no GPU checks or Ray.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

if [[ "$MODEL_PROFILE" != chimera && "$MODEL_PROFILE" != qwen3-0.6B ]]; then
    echo "Unsupported MODEL_PROFILE" >&2; exit 1
fi
if [[ "$CHIMERA_MODEL_SIZE" != full && "$CHIMERA_MODEL_SIZE" != tiny ]]; then
    echo "CHIMERA_MODEL_SIZE must be full or tiny" >&2; exit 1
fi
if [[ "$CHIMERA_MODEL_SIZE" == tiny && "$MODEL_PROFILE" != chimera ]]; then
    echo "Tiny Chimera is a Chimera mechanics-validation profile only" >&2; exit 1
fi
if [[ "$EXECUTION_MODE" != sync && "$EXECUTION_MODE" != async ]]; then
    echo "EXECUTION_MODE must be sync or async" >&2; exit 1
fi
if [[ "$COLOCATE" != 0 && "$COLOCATE" != 1 ]]; then
    echo "COLOCATE must be 0 or 1" >&2; exit 1
fi
if [[ "$NO_SAVE_OPTIM" != 0 && "$NO_SAVE_OPTIM" != 1 ]]; then
    echo "NO_SAVE_OPTIM must be 0 or 1" >&2; exit 1
fi
if [[ "$CHIMERA_FP32_LM_HEAD" != 0 && "$CHIMERA_FP32_LM_HEAD" != 1 ]]; then
    echo "CHIMERA_FP32_LM_HEAD must be 0 or 1" >&2; exit 1
fi
if [[ "$OFFLOAD_TRAIN" != 0 && "$OFFLOAD_TRAIN" != 1 || "$OFFLOAD_ROLLOUT" != 0 && "$OFFLOAD_ROLLOUT" != 1 ]]; then
    echo "OFFLOAD_TRAIN and OFFLOAD_ROLLOUT must be 0 or 1" >&2; exit 1
fi
if [[ "$EXECUTION_MODE" != sync && "$MIXRL_WALLCLOCK_SECONDS" != 0 ]]; then
    echo "Boundary wallclock stopping is synchronous-only" >&2; exit 1
fi
if [[ "$EXECUTION_MODE" == async ]]; then
    if [[ "$MODEL_PROFILE" != qwen3-0.6B || "$COLOCATE" != 0 || "$RESUME" != 0 || "$USE_ROLLOUT_LOGPROBS" != 1 || "$SAVE_INTERVAL" != "$NUM_ROLLOUT" ]]; then
        echo "Async smoke requires MixRL Qwen, COLOCATE=0, RESUME=0, USE_ROLLOUT_LOGPROBS=1, SAVE_INTERVAL=NUM_ROLLOUT (no mid-lookahead checkpoint)." >&2
        exit 1
    fi
fi
if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
    echo "RESUME must be 0 or 1, got: $RESUME" >&2
    exit 1
fi
case "$RUN_NAME" in
    *[!A-Za-z0-9._-]* | "")
        echo "Invalid RUN_NAME: $RUN_NAME (letters, digits, . _ - only)" >&2
        exit 1
        ;;
esac

EXPECTED_GPUS=$POLICY_GPUS
if [[ "$COLOCATE" == 0 ]]; then
    EXPECTED_GPUS=$((POLICY_GPUS + ROLLOUT_GPUS))
elif [[ "$ROLLOUT_GPUS" != "$POLICY_GPUS" ]]; then
    echo "Colocated runs require equal actor and rollout GPU counts" >&2; exit 1
fi
RUNS_ROOT=${MIXRL_RUNS_ROOT:-$DATA_ROOT/runs/chimera/mixrl}
RUN_DIR=$RUNS_ROOT/$RUN_NAME
SAVE_PATH=$RUN_DIR/checkpoints
TENSORBOARD_DIR=$RUN_DIR/tensorboard
LOG_DIR=$RUN_DIR/logs
MANIFEST_DIR=$RUN_DIR/manifests
ROLLOUT_DIR=$RUN_DIR/rollouts
TRAIN_DATA=$MIXRL_DATA_DIR/splits/rl_train.jsonl
EVAL_DATA=$MIXRL_DATA_DIR/splits/rl_val.jsonl
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}

# Everything printed from here to exit also goes to logs/train.log (appended on retry and
# resume). Cleanup steps registered below run first; then the log is flushed.
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/train.log") 2>&1
LOG_TEE_PID=$!
echo "=== $(date -u +%FT%TZ) mixrl/internal/launch.sh RUN_NAME=$RUN_NAME RESUME=$RESUME DRY_RUN=$DRY_RUN"
# Thousands of concurrent generation and grading connections: the usual soft limit of 1024 open
# files fails them with "Too many open files". Ray, SGLang and the rollout process inherit this.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
EXIT_STEPS=()
on_exit() {
    local status=$? step
    for step in "${EXIT_STEPS[@]}"; do eval "$step" || true; done
    exec >&- 2>&-
    wait "$LOG_TEE_PID" 2>/dev/null || true
    exit "$status"
}
trap on_exit EXIT

CONTEXT_OVERRIDE_ARGS=()
case "$CHIMERA_CONTEXT_OVERRIDE" in
    0) ;;
    1) CONTEXT_OVERRIDE_ARGS+=(--allow-mcore-context-override) ;;
    *) echo "CHIMERA_CONTEXT_OVERRIDE must be 0 or 1" >&2; exit 1 ;;
esac
CONTEXT_VALUES=$(PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m slime_plugins.models.chimera_context \
    --hf-checkpoint "$HF_CHECKPOINT" --mcore-checkpoint "${INITIAL_ACTOR_CHECKPOINT:-$MCORE_CHECKPOINT}" \
    --profile "$MODEL_PROFILE" --phase "$CONTEXT_PHASE" --sequence-cap "$TRAIN_SEQUENCE_LENGTH" "${CONTEXT_OVERRIDE_ARGS[@]}")
read -r RESOLVED_CONTEXT_PHASE MODEL_MAX_CONTEXT TRAIN_SEQUENCE_LENGTH <<< "$CONTEXT_VALUES"

# Preflight: task file, data, reward service and checkpoints, before any GPU work.
export CHIMERA_MIXRL_CONFIG="$MANIFEST_DIR/mixrl_config.json"
export MIXRL_DATA_DIR MIXRL_SCORER_URL MIXRL_TASKS_CONFIG MIXRL_TRUNCATION MIXRL_CODE_AUDIT_DIR
export MIXRL_SEED MIXRL_INFLIGHT_GROUPS MIXRL_RESPONSE_CONCURRENCY MIXRL_REFILL_ROUNDS MIXRL_CONTEXT_HEADROOM MIXRL_OVERSAMPLE
export MIXRL_LENGTH_PENALTY ROLLOUT_TEMPERATURE ROLLOUT_TOP_P ROLLOUT_TOP_K LR_WARMUP_STEPS
export MIXRL_COLLECTION_TIMEOUT MIXRL_REWARD_TIMEOUT MIXRL_REWARD_ATTEMPTS MIXRL_REWARD_CONCURRENCY
export MODEL_CONTEXT_LENGTH POLICY_GPUS EXPERT_MODEL_PARALLEL_SIZE N_SAMPLES_PER_PROMPT RUN_DIR
export MODEL_PROFILE CHAT_TEMPLATE_KWARGS HF_CHECKPOINT MCORE_CHECKPOINT LR MAX_TOKENS_PER_GPU
export WEIGHT_DECAY CLIP_GRAD ADAM_BETA1 ADAM_BETA2
export EXECUTION_MODE COLOCATE ROLLOUT_GPUS USE_ROLLOUT_LOGPROBS
export NUM_ROLLOUT RESUME MIXRL_EXTEND_CONSTANT_HORIZON INITIAL_ACTOR_CHECKPOINT
export MIXRL_OBJECTIVE MIXRL_IS_POSITIVE_BOUNDS MIXRL_IS_NEGATIVE_BOUNDS
export EVAL_INTERVAL MIXRL_EVAL_UPDATES CHIMERA_ROUTING_REPLAY
export CONTEXT_PHASE TRAIN_SEQUENCE_LENGTH CHIMERA_CONTEXT_OVERRIDE
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m slime_plugins.chimera_mixrl.configure
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then exit 0; fi
read -r ROLLOUT_BATCH_SIZE MIXRL_EVAL_SAMPLES < <(python3 -c 'import json, os
c = json.load(open(os.environ["CHIMERA_MIXRL_CONFIG"])); print(c["rollout_batch_size"], c["eval_samples"])')

for required_file in "$HF_CHECKPOINT/config.json" "$TRAIN_DATA" "$EVAL_DATA"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file not found: $required_file" >&2
        exit 1
    fi
done
if [[ "$MODEL_PROFILE" == chimera ]]; then
    test -f "$MCORE_CHECKPOINT/latest_checkpointed_iteration.txt"
fi
if [[ ! -d "$MEGATRON_ROOT/.git" ]]; then
    echo "The Slime image's Megatron checkout was not found at $MEGATRON_ROOT" >&2
    exit 1
fi

if [[ "$RESUME" == 1 ]]; then
    if [[ ! -f "$SAVE_PATH/latest_checkpointed_iteration.txt" ]]; then
        echo "RESUME=1 but no checkpoint exists at $SAVE_PATH" >&2
        exit 1
    fi
elif [[ -n "$(find "$SAVE_PATH" "$ROLLOUT_DIR" -type f -print -quit 2>/dev/null)" ]]; then
    # A start that failed before any rollout or checkpoint (only logs/manifests) may be retried.
    echo "Run $RUN_NAME already exists at $RUN_DIR; choose a new name or resume it" >&2
    exit 1
fi

mkdir -p "$SAVE_PATH" "$TENSORBOARD_DIR" "$LOG_DIR" "$MANIFEST_DIR" "$ROLLOUT_DIR"

if [[ "$MODEL_PROFILE" == chimera ]]; then
    test -f "$CHIMERA_TRANSFORMERS_ROOT/src/transformers/models/chimera/__init__.py"
    CUDA_GRAPH_PATCH=$CHIMERA_DIR/patches/megatron-yarn-te-cuda-graph.patch
    if git -C "$MEGATRON_ROOT" apply --unidiff-zero --reverse --check "$CUDA_GRAPH_PATCH" >/dev/null 2>&1; then
        echo "Megatron YaRN Transformer Engine CUDA-graph fix is already applied."
    elif git -C "$MEGATRON_ROOT" apply --unidiff-zero --check "$CUDA_GRAPH_PATCH" >/dev/null 2>&1; then
        git -C "$MEGATRON_ROOT" apply --unidiff-zero "$CUDA_GRAPH_PATCH"
        echo "Applied the YaRN Transformer Engine CUDA-graph fix to $MEGATRON_ROOT."
    else
        echo "The YaRN CUDA-graph patch does not match $MEGATRON_ROOT; refusing to run." >&2
        exit 1
    fi
fi

if [[ "$CHIMERA_ROUTING_REPLAY" == 1 ]]; then
    SGLANG_SOURCE=$(python3 -c 'from pathlib import Path; import sglang; print(Path(sglang.__file__).resolve().parents[2])')
    ROUTING_PATCH=$CHIMERA_DIR/patches/sglang-transformers-routing-capture.patch
    if git -C "$SGLANG_SOURCE" apply --unidiff-zero --reverse --check "$ROUTING_PATCH" >/dev/null 2>&1; then
        echo "SGLang Chimera routing capture patch already applied."
    elif git -C "$SGLANG_SOURCE" apply --unidiff-zero --check "$ROUTING_PATCH" >/dev/null 2>&1; then
        git -C "$SGLANG_SOURCE" apply --unidiff-zero "$ROUTING_PATCH"
    else
        echo "Pinned SGLang routing patch does not match; refusing replay." >&2; exit 1
    fi
fi

# Keep the image's pinned packages. sitecustomize registers only the external
# Chimera package; it does not replace the installed Transformers distribution.
export CHIMERA_TRANSFORMERS_ROOT HF_CHECKPOINT MEGATRON_ROOT TENSORBOARD_DIR
export PYTHONPATH="$REPO_ROOT:$MEGATRON_ROOT"
if [[ "$MODEL_PROFILE" == chimera ]]; then
    export PYTHONPATH="$CHIMERA_DIR/runtime:$PYTHONPATH"
fi
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVSHMEM_DISABLE_NCCL=1

if [[ "$DRY_RUN" != 1 ]]; then
mapfile -t GPU_NAMES < <(python3 -c 'import torch; [print(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]')
if ((${#GPU_NAMES[@]} != EXPECTED_GPUS)); then
    echo "MixRL requires exactly $EXPECTED_GPUS visible GPUs (POLICY_GPUS); found ${#GPU_NAMES[@]}" >&2
    exit 1
fi
for gpu_name in "${GPU_NAMES[@]}"; do
    if [[ "$gpu_name" != *H200* && !( "$MODEL_PROFILE" == chimera && "$CHIMERA_MODEL_SIZE" == tiny ) ]]; then
        echo "MixRL requires H200 GPUs; found: $gpu_name" >&2
        exit 1
    fi
done
fi

if [[ "$MODEL_PROFILE" == chimera ]]; then
    python3 "$CHIMERA_DIR/preflight.py"
fi
source "$REPO_ROOT/scripts/models/$MODEL_PROFILE.sh"
if [[ "$CHIMERA_ROUTING_REPLAY" == 1 ]]; then
    # TE fused top-k returns before Slime's replay hook in the pinned MCore.
    # Keep attention CUDA graphs; only disable router top-k fusion for R3.
    REPLAY_MODEL_ARGS=()
    for argument in "${MODEL_ARGS[@]}"; do
        if [[ "$argument" != --moe-router-fusion ]]; then REPLAY_MODEL_ARGS+=("$argument"); fi
    done
    MODEL_ARGS=("${REPLAY_MODEL_ARGS[@]}")
fi

GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))

CKPT_ARGS=(
    --hf-checkpoint "$HF_CHECKPOINT"
    --ref-load "$MCORE_CHECKPOINT"
    --save "$SAVE_PATH"
    --save-interval "$SAVE_INTERVAL"
)
if [[ "$NO_SAVE_OPTIM" == 1 ]]; then
    CKPT_ARGS+=(--no-save-optim)
fi
if [[ "$SAVE_HF" == 1 ]]; then
    [[ "$MODEL_PROFILE" == qwen3-0.6B ]] || { echo 'SAVE_HF is qualified only for the Qwen reference workflow.' >&2; exit 1; }
    CKPT_ARGS+=(--save-hf "$RUN_DIR/hf/iter_{rollout_id}")
elif [[ "$SAVE_HF" != 0 ]]; then
    echo 'SAVE_HF must be 0 or 1' >&2; exit 1
fi
if [[ "$RESUME" == 1 ]]; then
    CKPT_ARGS+=(--load "$SAVE_PATH")
else
    # The converted checkpoint supplies model weights only. Start fresh
    # optimizer and RNG state while retaining optimizer state in new saves.
    CKPT_ARGS+=(--finetune --no-load-optim --no-load-rng)
    if [[ "$MODEL_PROFILE" == qwen3-0.6B ]]; then
        # Slime already supports native qwen3 HF -> Megatron weight loading.
        CKPT_ARGS+=(--load "${INITIAL_ACTOR_CHECKPOINT:-$HF_CHECKPOINT}")
    fi
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
    --rollout-temperature "$ROLLOUT_TEMPERATURE"
    --rollout-top-p "$ROLLOUT_TOP_P"
    --rollout-top-k "$ROLLOUT_TOP_K"
    --num-steps-per-rollout 1
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --balance-data
    --apply-chat-template-kwargs "$CHAT_TEMPLATE_KWARGS"
    --data-source-path slime_plugins.chimera_mixrl.runtime.DataSource
    --rollout-function-path slime_plugins.chimera_mixrl.runtime.generate_rollout
    --custom-rm-path slime_plugins.chimera_mixrl.runtime.reward
    --custom-reward-post-process-path slime_plugins.chimera_mixrl.runtime.post_process_rewards
)
if [[ "$MODEL_PROFILE" == chimera ]]; then
    ROLLOUT_ARGS+=(--rollout-stop "<end_of_turn>")
fi

# Slime validates that an eval dataset is declared; the MixRL rollout selects
# enabled tasks' prompts from this frozen rl_val split, never from main_test.
EVAL_ARGS=(--eval-interval "$EVAL_INTERVAL" --eval-prompt-data mixrl "$EVAL_DATA"
    --n-samples-per-eval-prompt "$MIXRL_EVAL_SAMPLES")
if [[ "$EVAL_BEFORE_TRAIN" == 0 ]]; then EVAL_ARGS+=(--skip-eval-before-train); fi

if [[ "$MIXRL_OBJECTIVE" == mimo ]]; then
    OBJECTIVE_ARGS=(--advantage-estimator grpo --disable-grpo-std-normalization
        --kl-coef 0.0 --entropy-coef 0.0 --loss-type custom_loss
        --custom-loss-function-path slime_plugins.chimera_mixrl.objective.loss)
else
    # dapo: native DAPO objective, kept as a control.
    OBJECTIVE_ARGS=(--advantage-estimator grpo --calculate-per-token-loss --kl-coef 0.0 --entropy-coef 0.0
        --eps-clip 0.2 --eps-clip-high 0.28)
fi

PARALLEL_ARGS=(
    # With every model-parallel dimension at one, all actor ranks form Megatron's
    # data-parallel group; the optimizer state is sharded over it.
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size "$EXPERT_MODEL_PARALLEL_SIZE"
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
    --lr-warmup-iters "$LR_WARMUP_STEPS"
    --lr-decay-style constant
    --weight-decay "$WEIGHT_DECAY"
    --clip-grad "$CLIP_GRAD"
    --adam-beta1 "$ADAM_BETA1"
    --adam-beta2 "$ADAM_BETA2"
    --use-precision-aware-optimizer
    --main-params-dtype fp32
    --main-grads-dtype fp32
    --exp-avg-dtype fp32
    --exp-avg-sq-dtype fp32
)

SGLANG_ARGS=(
    # Slime creates one independent TP=1 engine per rollout GPU.
    --rollout-num-gpus "$ROLLOUT_GPUS"
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC"
    --sglang-cuda-graph-max-bs-decode "$SGLANG_CUDA_GRAPH_MAX_BS"
    --sglang-enable-metrics
    --sglang-context-length "$TRAIN_SEQUENCE_LENGTH"
)
if [[ -n "$SGLANG_MAX_RUNNING_REQUESTS" ]]; then
    SGLANG_ARGS+=(--sglang-max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS")
fi

MISC_ARGS=(
    --model-name "$MODEL_PROFILE"
    --attention-backend flash
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --actor-num-nodes 1
    --actor-num-gpus-per-node "$POLICY_GPUS"
    --num-gpus-per-node "$EXPECTED_GPUS"
    --use-tensorboard
    --tensorboard-dir "$TENSORBOARD_DIR"
    --log-interval 1
    --seq-length "$TRAIN_SEQUENCE_LENGTH"
    --max-position-embeddings "$MODEL_MAX_CONTEXT"
)
if [[ "$COLOCATE" == 1 ]]; then MISC_ARGS+=(--colocate); fi
if [[ "$OFFLOAD_TRAIN" == 0 ]]; then MISC_ARGS+=(--no-offload-train); fi
if [[ "$OFFLOAD_ROLLOUT" == 0 ]]; then MISC_ARGS+=(--no-offload-rollout); fi
if [[ "$USE_ROLLOUT_LOGPROBS" == 1 ]]; then MISC_ARGS+=(--use-rollout-logprobs); fi
TRAIN_ENTRYPOINT=$REPO_ROOT/train.py
if [[ "$EXECUTION_MODE" == async ]]; then
    TRAIN_ENTRYPOINT=$REPO_ROOT/train_async.py
    MISC_ARGS+=(--update-weights-interval 1)
fi
if [[ "$MODEL_PROFILE" == chimera ]]; then
    SGLANG_ARGS+=(--sglang-model-impl transformers)
    if [[ "$CHIMERA_FP32_LM_HEAD" == 1 ]]; then SGLANG_ARGS+=(--sglang-enable-fp32-lm-head); fi
    MISC_ARGS+=(--custom-model-provider-path slime_plugins.models.chimera.model_provider
        --freeze-params-name-list '(^|\.)router\.(weight|bias)$' --moe-z-loss-coeff 0.0
        --custom-megatron-before-train-step-hook-path slime_plugins.chimera_mixrl.routing.before_train_step)
fi
if [[ "$CHIMERA_ROUTING_REPLAY" == 1 ]]; then
    MISC_ARGS+=(--use-routing-replay --use-rollout-routing-replay)
    SGLANG_ARGS+=(--sglang-enable-return-routed-experts)
fi
if [[ "${DUMP_DETAILS:-0}" == 1 ]]; then
    MISC_ARGS+=(--dump-details "$ROLLOUT_DIR")
fi

TRAIN_COMMAND=(
    python3 "$TRAIN_ENTRYPOINT"
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${EVAL_ARGS[@]}"
    "${OBJECTIVE_ARGS[@]}"
    "${PARALLEL_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${MISC_ARGS[@]}"
)

# Everything needed to reproduce this run, next to its checkpoints.
cp "$MIXRL_TASKS_CONFIG" "$MANIFEST_DIR/tasks.json"
while IFS= read -r name; do
    printf '%s=%q\n' "$name" "${!name}"
done < <(grep -oE '^[A-Z_][A-Z0-9_]*=' "$MIXRL_DIR/config.env" | tr -d =) > "$MANIFEST_DIR/config.env"
tar --exclude=__pycache__ -cf "$MANIFEST_DIR/mixrl_source.tar" -C "$REPO_ROOT" \
    mixrl slime_plugins/chimera_mixrl "scripts/models/$MODEL_PROFILE.sh" train.py train_async.py \
    slime/backends/megatron_utils/model.py slime/ray/rollout.py slime_plugins/models/chimera.py \
    slime_plugins/models/chimera_context.py slime_plugins/models/chimera_geometry.py \
    slime_plugins/models/chimera_precision.py slime_plugins/models/chimera_sglang_precision.py \
    examples/chimera/runtime/sitecustomize.py \
    slime/backends/megatron_utils/actor.py slime/utils/routing_replay.py slime/backends/megatron_utils/loss.py \
    slime/backends/megatron_utils/cp_utils.py examples/chimera/patches
# Repos copied without .git (e.g. a GitHub zip) have no commit; mixrl_source.tar still holds the source.
commit_of() { git -c safe.directory="$1" -C "$1" rev-parse HEAD 2>/dev/null || echo "unknown: $1 is not a git checkout"; }
commit_of "$REPO_ROOT" > "$MANIFEST_DIR/slime_commit.txt"
if [[ "$MODEL_PROFILE" == chimera ]]; then
    commit_of "$CHIMERA_TRANSFORMERS_ROOT" > "$MANIFEST_DIR/transformers_commit.txt"
fi
commit_of "$MEGATRON_ROOT" > "$MANIFEST_DIR/megatron_image_commit.txt"
{
    printf 'DATA_ROOT=%q\n' "$DATA_ROOT"
    printf 'RUN_NAME=%q\n' "$RUN_NAME"
    printf 'RUN_DIR=%q\n' "$RUN_DIR"
    printf 'HF_CHECKPOINT=%q\n' "$HF_CHECKPOINT"
    printf 'MCORE_CHECKPOINT=%q\n' "$MCORE_CHECKPOINT"
    printf 'TRAIN_DATA=%q\n' "$TRAIN_DATA"
    printf 'EVAL_DATA=%q\n' "$EVAL_DATA"
    printf 'RESUME=%q\n' "$RESUME"
    printf 'CONTEXT_PHASE=%q\n' "$RESOLVED_CONTEXT_PHASE"
    printf 'MODEL_MAX_CONTEXT=%q\n' "$MODEL_MAX_CONTEXT"
    printf 'TRAIN_SEQUENCE_LENGTH=%q\n' "$TRAIN_SEQUENCE_LENGTH"
} > "$MANIFEST_DIR/run_paths.env"
printf '%q ' "${TRAIN_COMMAND[@]}" > "$MANIFEST_DIR/train_command.sh"
printf '\n' >> "$MANIFEST_DIR/train_command.sh"

echo "Run directory: $RUN_DIR"
echo "Checkpoint context: $RESOLVED_CONTEXT_PHASE maximum=$MODEL_MAX_CONTEXT; run sequence cap=$TRAIN_SEQUENCE_LENGTH"
echo "Megatron actor: dense-DP=$POLICY_GPUS, expert-DP=$((POLICY_GPUS / EXPERT_MODEL_PARALLEL_SIZE)), TP=PP=CP=ETP=1, EP=$EXPERT_MODEL_PARALLEL_SIZE, distributed optimizer"
echo "SGLang rollout: $ROLLOUT_GPUS independent TP=1 engines, mode=$EXECUTION_MODE colocate=$COLOCATE"
echo "Batch: $ROLLOUT_BATCH_SIZE prompts x $N_SAMPLES_PER_PROMPT responses = $GLOBAL_BATCH_SIZE samples"
if [[ "$DRY_RUN" == 1 ]]; then
    echo "Dry run only: command/manifests written; no Ray services or training started."
    exit 0
fi

ray stop --force >/dev/null 2>&1 || true
GPU_METRICS_PID=
if [[ "$GPU_METRICS_INTERVAL" != 0 ]]; then
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv --loop="$GPU_METRICS_INTERVAL" >> "$LOG_DIR/gpu_metrics.csv" &
    GPU_METRICS_PID=$!
fi
EXIT_STEPS+=('if [[ -n "$GPU_METRICS_PID" ]]; then kill "$GPU_METRICS_PID" 2>/dev/null; fi'
    'ray stop --force >/dev/null 2>&1'
    # Ray's own logs (raylet, GCS, workers) die with the container; keep them with the run.
    'tar -czf "$LOG_DIR/ray_logs-$(date -u +%Y%m%dT%H%M%SZ).tar.gz" -C "${RAY_TMPDIR:-/tmp}/ray/session_latest" logs 2>/dev/null')
# Ray's own services and the submit client do not need Chimera. Without the runtime
# sitecustomize each would import torch and Transformers (~0.8 GB apiece); job
# processes still get the full PYTHONPATH from the runtime env below.
RAY_SERVICE_PYTHONPATH="$REPO_ROOT:$MEGATRON_ROOT"
# Plain text in train.log: no color codes around Ray worker prefixes or CLI output.
export RAY_COLOR_PREFIX=0
PYTHONPATH="$RAY_SERVICE_PYTHONPATH" ray start \
    --head \
    --log-color false \
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
    "RAY_COLOR_PREFIX",
    "TENSORBOARD_DIR",
    "MIXRL_WALLCLOCK_SECONDS",
    "MIXRL_FINAL_RESERVE_SECONDS",
    "MIXRL_INITIAL_UPDATE_SECONDS",
    "MIXRL_STOP_FILE",
    "MIXRL_KEEP_TRAIN_SAMPLES",
    "MIXRL_HEALTH_WAIT_SECONDS",
    "MIXRL_REWARD_BACKOFF_MAX",
    "MIXRL_PIPELINE_SECONDS",
    "MIXRL_JUDGE_METRICS_URL",
    "CHIMERA_MATCH_DENSE_SWIGLU",
    "CHIMERA_SGLANG_FULL_BF16_REDUCTION",
)
values = {key: os.environ[key] for key in keys}
if os.environ.get("CHIMERA_MIXRL_CONFIG"):
    values["CHIMERA_MIXRL_CONFIG"] = os.environ["CHIMERA_MIXRL_CONFIG"]
    values["CHIMERA_ROUTING_REPLAY"] = os.environ["CHIMERA_ROUTING_REPLAY"]
    values["MIXRL_ROUTER_METRICS"] = os.environ["MIXRL_ROUTER_METRICS"]
    values["CHIMERA_MODEL_SIZE"] = os.environ["CHIMERA_MODEL_SIZE"]
    values["CHIMERA_FP32_LM_HEAD"] = os.environ["CHIMERA_FP32_LM_HEAD"]
    values["CHIMERA_MATCH_RMSNORM"] = os.environ["CHIMERA_MATCH_RMSNORM"]
print(json.dumps({"env_vars": values}))
PY
)

PYTHONPATH="$RAY_SERVICE_PYTHONPATH" ray job submit \
    --log-color false \
    --address="http://${MASTER_ADDR:-127.0.0.1}:${RAY_DASHBOARD_PORT:-8265}" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- "${TRAIN_COMMAND[@]}"
