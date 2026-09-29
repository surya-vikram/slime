#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

# Edit this block, or set the same names in the environment.
RECIPE=${RECIPE:-mixrl} # mixrl | gsm8k (control) | sft (Qwen reference only).
MODEL_PROFILE=${MODEL_PROFILE:-chimera} # qwen3-0.6B: reference MixRL validation only.
if [[ "$RECIPE" == mixrl ]]; then
    for removed in MIXRL_QUOTAS MIXRL_CAPS MIXRL_EVAL_QUOTAS MIXRL_MAIN_EVAL_QUOTAS ROLLOUT_BATCH_SIZE \
            MIXRL_EVAL_SAMPLES MIXRL_MAIN_EVAL_INTERVAL MIXRL_MAIN_EVAL_SAMPLES; do
        if [[ -n "${!removed:-}" ]]; then
            echo "$removed is no longer read for MixRL; set tasks, prompt counts and eval size in the task file (MIXRL_TASKS_CONFIG)" >&2
            exit 1
        fi
    done
fi
export CHIMERA_MODEL_SIZE=${CHIMERA_MODEL_SIZE:-full} # tiny: canonical 8-layer local mechanics only.
DEFAULT_FP32_LM_HEAD=0
if [[ "$MODEL_PROFILE" == chimera && "$RECIPE" == mixrl ]]; then DEFAULT_FP32_LM_HEAD=1; fi
export CHIMERA_FP32_LM_HEAD=${CHIMERA_FP32_LM_HEAD:-$DEFAULT_FP32_LM_HEAD} # Locked Chimera MixRL: matched FP32 projection on actor AND rollout; BF16 weights.
export CHIMERA_MATCH_RMSNORM=${CHIMERA_MATCH_RMSNORM:-0} # Opt-in SGLang rounding aligned with TE RMSNorm.
export CHIMERA_MATCH_DENSE_SWIGLU=${CHIMERA_MATCH_DENSE_SWIGLU:-0} # Experimental rollout dense SwiGLU kernel; Tiny-qualified only.
export CHIMERA_SGLANG_FULL_BF16_REDUCTION=${CHIMERA_SGLANG_FULL_BF16_REDUCTION:-0} # Rollout PyTorch BF16 reduction only; does not configure native TE.
CHAT_TEMPLATE_KWARGS=${CHAT_TEMPLATE_KWARGS:-'{}'}
if [[ "$RECIPE" == sft && "$CHAT_TEMPLATE_KWARGS" == '{}' ]]; then
    CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
fi
DATA_ROOT=${DATA_ROOT:-/datasets/megadata}
RUN_NAME=${RUN_NAME:-chimera-$RECIPE-$(TZ=Asia/Kolkata date +%Y%m%d-%H%M%S)}
HF_CHECKPOINT=${HF_CHECKPOINT:-$DATA_ROOT/models/chimera-10b-hf}
MCORE_CHECKPOINT=${MCORE_CHECKPOINT:-$DATA_ROOT/models/chimera-10b-mcore}
CHIMERA_TRANSFORMERS_ROOT=${CHIMERA_TRANSFORMERS_ROOT:-$(dirname "$REPO_ROOT")/transformers}

# Production topology defaults to one 8xH200 node; POLICY_GPUS can be reduced.
DEFAULT_UPDATES=3000
DEFAULT_EVAL_INTERVAL=20
DEFAULT_SAVE_INTERVAL=100
if [[ "$RECIPE" == mixrl ]]; then
    DEFAULT_UPDATES=100
    DEFAULT_EVAL_INTERVAL=10
    DEFAULT_SAVE_INTERVAL=50
fi
DEFAULT_SAMPLES=8
if [[ "$RECIPE" == mixrl && "$MODEL_PROFILE" == qwen3-0.6B ]]; then DEFAULT_SAMPLES=4; fi
NUM_ROLLOUT=${NUM_ROLLOUT:-$DEFAULT_UPDATES}
LR=${LR:-1e-6}
WEIGHT_DECAY=${WEIGHT_DECAY:-0} # Locked MixRL baseline: no weight decay.
CLIP_GRAD=${CLIP_GRAD:-1.0}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.98}
SFT_LR=${SFT_LR:-1e-5}
SFT_DATA=${SFT_DATA:-} # Token-admitted JSONL + adjacent .manifest.json, rl_train only.
SFT_WARMUP_FRACTION=${SFT_WARMUP_FRACTION:-0.05}
SFT_SAVE_INTERVAL=${SFT_SAVE_INTERVAL:-} # Empty: final-only; 1 for isolated resume qualification.
INITIAL_ACTOR_CHECKPOINT=${INITIAL_ACTOR_CHECKPOINT:-} # Fresh RL from SFT MCore weights, not optimizer.
SAVE_HF=${SAVE_HF:-0} # Opt-in native HF export beside MCore saves; Qwen validation only.
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32} # MixRL: replaced by the task file's enabled prompt sum after preflight.
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-$DEFAULT_SAMPLES}
OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-64}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-512}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-512}
EVAL_INTERVAL=${EVAL_INTERVAL:-$DEFAULT_EVAL_INTERVAL}
SAVE_INTERVAL=${SAVE_INTERVAL:-$DEFAULT_SAVE_INTERVAL}
NO_SAVE_OPTIM=${NO_SAVE_OPTIM:-0} # 1: save model weights only; avoids very large Adam-state checkpoints.
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.70}
SGLANG_CUDA_GRAPH_MAX_BS=${SGLANG_CUDA_GRAPH_MAX_BS:-32}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-} # Optional per-engine request bound.
RESUME=${RESUME:-0}
MIXRL_EXTEND_CONSTANT_HORIZON=${MIXRL_EXTEND_CONSTANT_HORIZON:-0} # Explicit resume-only extension; LR/WD schedule stays frozen.
GPU_METRICS_INTERVAL=${GPU_METRICS_INTERVAL:-5} # Seconds; 0 disables nvidia-smi CSV sidecar.
# Opt-in budget starts before model initialization. Reserve includes final eval/save.
export MIXRL_WALLCLOCK_SECONDS=${MIXRL_WALLCLOCK_SECONDS:-0}
export MIXRL_FINAL_RESERVE_SECONDS=${MIXRL_FINAL_RESERVE_SECONDS:-1200}
export MIXRL_INITIAL_UPDATE_SECONDS=${MIXRL_INITIAL_UPDATE_SECONDS:-300}
export MIXRL_STOP_FILE=${MIXRL_STOP_FILE:-}
# Small-reference-model residency experiment only; defaults retain native offload.
OFFLOAD_TRAIN=${OFFLOAD_TRAIN:-1}
OFFLOAD_ROLLOUT=${OFFLOAD_ROLLOUT:-1}

# MixRL v0: configure here; scorer uses the separate chimera-eval environment/image.
POLICY_GPUS=${POLICY_GPUS:-8} # MixRL can validate DP4; judge GPUs are not in this visible pool.
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1} # Set 2 on the two-H200 16K qualification run.
EXECUTION_MODE=${EXECUTION_MODE:-sync} # async: native one-batch-lookahead experiment, Qwen only.
COLOCATE=${COLOCATE:-1}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$POLICY_GPUS}
USE_ROLLOUT_LOGPROBS=${USE_ROLLOUT_LOGPROBS:-0}
# auto reads the checkpoint phase. Explicit 8k|32k|64k|128k must MATCH it.
# All Chimera phases use YaRN, including factor 1 at 8K.
CONTEXT_PHASE=${CONTEXT_PHASE:-auto}
CHIMERA_CONTEXT_OVERRIDE=${CHIMERA_CONTEXT_OVERRIDE:-0} # Explicitly use HF YaRN over historical MCore positional metadata; record both.
# Total prompt/history + response cap, NOT the model's positional maximum.
# MixRL is DP-only: hard ceiling 16384, also bounded by checkpoint context.
# MODEL_CONTEXT_LENGTH remains a backwards-compatible alias for this run cap.
TRAIN_SEQUENCE_LENGTH=${TRAIN_SEQUENCE_LENGTH:-${MODEL_CONTEXT_LENGTH:-16384}}
if [[ -n "${MODEL_CONTEXT_LENGTH:-}" && "$MODEL_CONTEXT_LENGTH" != "$TRAIN_SEQUENCE_LENGTH" ]]; then
    echo "MODEL_CONTEXT_LENGTH and TRAIN_SEQUENCE_LENGTH disagree" >&2; exit 1
fi
MODEL_CONTEXT_LENGTH=$TRAIN_SEQUENCE_LENGTH
MIXRL_DATA_DIR=${MIXRL_DATA_DIR:-$DATA_ROOT/datasets/chimera-eval-data}
MIXRL_CODE_AUDIT_DIR=${MIXRL_CODE_AUDIT_DIR:-$MIXRL_DATA_DIR/audits/apps} # Reference+negative sandbox report.
MIXRL_SCORER_URL=${MIXRL_SCORER_URL:-http://127.0.0.1:18020}
# Which tasks train, prompts per step, response caps and eval prompt counts. The
# rollout batch is the sum of enabled prompts. Preview: python3 -m slime_plugins.chimera_mixrl.tasks
MIXRL_TASKS_CONFIG=${MIXRL_TASKS_CONFIG:-$SCRIPT_DIR/mixrl_tasks.json}
MIXRL_TRUNCATION=${MIXRL_TRUNCATION:-mask} # mask: exclude caps from loss AND group stats; zero: fixed-budget learning.
MIXRL_SEED=${MIXRL_SEED:-42}
MIXRL_CONTEXT_HEADROOM=${MIXRL_CONTEXT_HEADROOM:-1024} # Reserved, not added to generation allowance.
MIXRL_INFLIGHT_GROUPS=${MIXRL_INFLIGHT_GROUPS:-4}
MIXRL_RESPONSE_CONCURRENCY=${MIXRL_RESPONSE_CONCURRENCY:-8}
MIXRL_MAX_ATTEMPTS=${MIXRL_MAX_ATTEMPTS:-100} # Legacy compatibility only; fixed batches never refill.
MIXRL_COLLECTION_TIMEOUT=${MIXRL_COLLECTION_TIMEOUT:-1800}
MIXRL_REWARD_TIMEOUT=${MIXRL_REWARD_TIMEOUT:-600}
MIXRL_REWARD_ATTEMPTS=${MIXRL_REWARD_ATTEMPTS:-3}
MIXRL_REWARD_CONCURRENCY=${MIXRL_REWARD_CONCURRENCY:-8} # Match the default Glimmer/scorer request capacity.
MIXRL_OBJECTIVE=${MIXRL_OBJECTIVE:-mimo} # mimo: fixed detached-IS/group-token objective; dapo: control.
export MIXRL_ROUTER_METRICS=${MIXRL_ROUTER_METRICS:-1} # Per-rank/layer/microbatch full-input expert counts.
MIXRL_IS_POSITIVE_BOUNDS=${MIXRL_IS_POSITIVE_BOUNDS:-'[0.2,5.0]'}
MIXRL_IS_NEGATIVE_BOUNDS=${MIXRL_IS_NEGATIVE_BOUNDS:-'[0.2,5.0]'}
CHIMERA_ROUTING_REPLAY=${CHIMERA_ROUTING_REPLAY:-$([[ "$MODEL_PROFILE" == chimera && "$RECIPE" == mixrl ]] && echo 1 || echo 0)} # Baseline requires R3; 0 is an explicit diagnostic control.
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0} # MixRL config/data/scorer checks; no Ray or GPU launch.
DRY_RUN=${DRY_RUN:-0} # Build manifests/command in the pinned image, skip GPU checks and Ray.

if [[ "$RECIPE" != gsm8k && "$RECIPE" != mixrl && "$RECIPE" != sft ]]; then
    echo "RECIPE must be gsm8k, mixrl or sft" >&2
    exit 1
fi
if [[ "$MODEL_PROFILE" != chimera && "$MODEL_PROFILE" != qwen3-0.6B ]]; then
    echo "Unsupported MODEL_PROFILE" >&2; exit 1
fi
if [[ "$CHIMERA_MODEL_SIZE" != full && "$CHIMERA_MODEL_SIZE" != tiny ]]; then
    echo "CHIMERA_MODEL_SIZE must be full or tiny" >&2; exit 1
fi
if [[ "$CHIMERA_MODEL_SIZE" == tiny && ( "$MODEL_PROFILE" != chimera || "$RECIPE" != mixrl ) ]]; then
    echo "Tiny Chimera is a MixRL mechanics-validation profile only" >&2; exit 1
fi
if [[ "$RECIPE" == sft && ( "$MODEL_PROFILE" != qwen3-0.6B || "$EXECUTION_MODE" != sync ) ]]; then
    echo "SFT adapter currently qualifies synchronous Qwen3-0.6B only" >&2; exit 1
fi
if [[ "$MODEL_PROFILE" != chimera && "$RECIPE" == gsm8k ]]; then
    echo "Reference model requires RECIPE=mixrl" >&2; exit 1
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
    if [[ "$RECIPE" != mixrl || "$MODEL_PROFILE" != qwen3-0.6B || "$COLOCATE" != 0 || "$RESUME" != 0 || "$USE_ROLLOUT_LOGPROBS" != 1 || "$SAVE_INTERVAL" != "$NUM_ROLLOUT" ]]; then
        echo "Async smoke requires MixRL Qwen, COLOCATE=0, RESUME=0, USE_ROLLOUT_LOGPROBS=1, SAVE_INTERVAL=NUM_ROLLOUT (no mid-lookahead checkpoint)." >&2
        exit 1
    fi
fi

EXPECTED_GPUS=8
RUNS_ROOT=${RUNS_ROOT:-$DATA_ROOT/runs/chimera/dapo-gsm8k}
if [[ "$RECIPE" == mixrl ]]; then
    EXPECTED_GPUS=$POLICY_GPUS
    if [[ "$COLOCATE" == 0 ]]; then
        EXPECTED_GPUS=$((POLICY_GPUS + ROLLOUT_GPUS))
    elif [[ "$ROLLOUT_GPUS" != "$POLICY_GPUS" ]]; then
        echo "Colocated smoke requires equal actor and rollout GPU counts" >&2; exit 1
    fi
    RUNS_ROOT=${MIXRL_RUNS_ROOT:-$DATA_ROOT/runs/chimera/mixrl}
fi
if [[ "$RECIPE" == sft ]]; then
    EXPECTED_GPUS=$POLICY_GPUS
    RUNS_ROOT=${SFT_RUNS_ROOT:-$DATA_ROOT/runs/chimera/sft}
fi
RUN_DIR=$RUNS_ROOT/$RUN_NAME
SAVE_PATH=$RUN_DIR/checkpoints
TENSORBOARD_DIR=$RUN_DIR/tensorboard
LOG_DIR=$RUN_DIR/logs
MANIFEST_DIR=$RUN_DIR/manifests
ROLLOUT_DIR=$RUN_DIR/rollouts
TRAIN_DATA=$SCRIPT_DIR/data/gsm8k_train.jsonl
EVAL_DATA=${EVAL_DATA:-$SCRIPT_DIR/data/gsm8k_validation.jsonl}
if [[ "$RECIPE" == mixrl ]]; then
    TRAIN_DATA=$MIXRL_DATA_DIR/splits/rl_train.jsonl
    EVAL_DATA=$MIXRL_DATA_DIR/splits/rl_val.jsonl
fi
if [[ "$RECIPE" == sft ]]; then
    TRAIN_DATA=$SFT_DATA
    EVAL_DATA=$SFT_DATA # Required-file check only; SFT never evaluates this data.
    export SFT_DATA CHAT_TEMPLATE_KWARGS ROLLOUT_BATCH_SIZE TRAIN_SEQUENCE_LENGTH HF_CHECKPOINT
    NUM_ROLLOUT=$(PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
from slime_plugins.chimera_mixrl.core import digest
from slime_plugins.chimera_mixrl.sft import validate_provenance
p = Path(os.environ['SFT_DATA'])
m = json.loads(Path(str(p) + '.manifest.json').read_text())
with p.open() as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
assert digest(rows) == m['data_hash'], 'SFT artifact changed'
assert m['batch_size'] == int(os.environ['ROLLOUT_BATCH_SIZE']), 'SFT batch differs from frozen one-pass plan'
assert len(rows) == m['rows'] == m['batches'] * m['batch_size'], 'SFT one-pass batch count inconsistent'
assert m['sequence_cap'] == int(os.environ['TRAIN_SEQUENCE_LENGTH']), 'SFT context differs'
assert m['chat_template_kwargs'] == json.loads(os.environ['CHAT_TEMPLATE_KWARGS']), 'SFT template differs'
assert m.get('tokenizer_files'), 'SFT manifest lacks frozen tokenizer hashes'
tokenizer = Path(os.environ['HF_CHECKPOINT'])
for name, expected in m['tokenizer_files'].items():
    assert Path(name).name == name, 'Invalid tokenizer filename'
    assert hashlib.sha256((tokenizer / name).read_bytes()).hexdigest() == expected, 'SFT tokenizer changed: ' + name
for row in rows:
    validate_provenance(row['metadata'])
print(m['batches'])
PY
)
    N_SAMPLES_PER_PROMPT=1
    LR=$SFT_LR
    SAVE_INTERVAL=${SFT_SAVE_INTERVAL:-$NUM_ROLLOUT}
fi
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}

case "$RUN_NAME" in
    *[!A-Za-z0-9._-]* | "")
        echo "Invalid RUN_NAME: $RUN_NAME" >&2
        exit 1
        ;;
esac
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
if [[ "$RECIPE" == mixrl ]]; then
    export CHIMERA_MIXRL_CONFIG="$MANIFEST_DIR/mixrl_config.json"
    export MIXRL_DATA_DIR MIXRL_SCORER_URL MIXRL_TASKS_CONFIG MIXRL_TRUNCATION
    export MIXRL_CODE_AUDIT_DIR
    export MIXRL_SEED MIXRL_INFLIGHT_GROUPS MIXRL_RESPONSE_CONCURRENCY MIXRL_MAX_ATTEMPTS
    export MIXRL_CONTEXT_HEADROOM
    export MIXRL_COLLECTION_TIMEOUT MIXRL_REWARD_TIMEOUT MIXRL_REWARD_ATTEMPTS
    export MIXRL_REWARD_CONCURRENCY
    export MODEL_CONTEXT_LENGTH POLICY_GPUS EXPERT_MODEL_PARALLEL_SIZE N_SAMPLES_PER_PROMPT RUN_DIR
    export MODEL_PROFILE CHAT_TEMPLATE_KWARGS HF_CHECKPOINT MCORE_CHECKPOINT LR MAX_TOKENS_PER_GPU
    export WEIGHT_DECAY CLIP_GRAD ADAM_BETA1 ADAM_BETA2
    export EXECUTION_MODE COLOCATE ROLLOUT_GPUS USE_ROLLOUT_LOGPROBS
    export NUM_ROLLOUT RESUME MIXRL_EXTEND_CONSTANT_HORIZON
    export INITIAL_ACTOR_CHECKPOINT
    export MIXRL_OBJECTIVE MIXRL_IS_POSITIVE_BOUNDS MIXRL_IS_NEGATIVE_BOUNDS
    export EVAL_INTERVAL
    export MIXRL_EVAL_UPDATES
    export CHIMERA_ROUTING_REPLAY
    export CONTEXT_PHASE TRAIN_SEQUENCE_LENGTH
    export CHIMERA_CONTEXT_OVERRIDE
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m slime_plugins.chimera_mixrl.configure
    if [[ "$PREFLIGHT_ONLY" == 1 ]]; then exit 0; fi
    read -r ROLLOUT_BATCH_SIZE MIXRL_EVAL_SAMPLES < <(python3 -c 'import json, os
c = json.load(open(os.environ["CHIMERA_MIXRL_CONFIG"])); print(c["rollout_batch_size"], c["eval_samples"])')
fi

if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
    echo "RESUME must be 0 or 1, got: $RESUME" >&2
    exit 1
fi
if [[ "$RECIPE" == gsm8k ]] && ((OVER_SAMPLING_BATCH_SIZE <= ROLLOUT_BATCH_SIZE)); then
    echo "OVER_SAMPLING_BATCH_SIZE must exceed ROLLOUT_BATCH_SIZE for strict DAPO filtering" >&2
    exit 1
fi

for required_file in \
    "$HF_CHECKPOINT/config.json" \
    "$TRAIN_DATA" \
    "$EVAL_DATA"; do
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
        echo "RESUME=1 but no Slime checkpoint exists at $SAVE_PATH" >&2
        exit 1
    fi
    if [[ "$RECIPE" == sft ]]; then
        SAVED_ITERATION=$(<"$SAVE_PATH/latest_checkpointed_iteration.txt")
        if [[ ! "$SAVED_ITERATION" =~ ^[0-9]+$ ]] || [[ ! -f "$SAVE_PATH/rollout/global_dataset_state_dict_${SAVED_ITERATION}.pt" ]]; then
            echo "SFT resume requires the matching saved sampler cursor; refusing to repeat examples." >&2
            exit 1
        fi
        cmp --silent "$SFT_DATA.manifest.json" "$MANIFEST_DIR/sft_data.json" || {
            echo "SFT resume manifest differs from the saved run." >&2; exit 1;
        }
    fi
elif [[ -d "$RUN_DIR" ]] && find "$RUN_DIR" -type f ! -name mixrl_config.json -print -quit | grep -q .; then
    echo "Fresh run directory is not empty: $RUN_DIR (choose a new RUN_NAME or set RESUME=1)" >&2
    exit 1
fi

mkdir -p "$SAVE_PATH" "$TENSORBOARD_DIR" "$LOG_DIR" "$MANIFEST_DIR" "$ROLLOUT_DIR"

CUDA_GRAPH_PATCH=$SCRIPT_DIR/patches/megatron-yarn-te-cuda-graph.patch
if [[ "$MODEL_PROFILE" == chimera ]]; then
test -f "$CHIMERA_TRANSFORMERS_ROOT/src/transformers/models/chimera/__init__.py"
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

if [[ "$RECIPE" == mixrl && "$CHIMERA_ROUTING_REPLAY" == 1 ]]; then
    SGLANG_SOURCE=$(python3 -c 'from pathlib import Path; import sglang; print(Path(sglang.__file__).resolve().parents[2])')
    ROUTING_PATCH=$SCRIPT_DIR/patches/sglang-transformers-routing-capture.patch
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
    export PYTHONPATH="$SCRIPT_DIR/runtime:$PYTHONPATH"
fi
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVSHMEM_DISABLE_NCCL=1

if [[ "$DRY_RUN" != 1 ]]; then
mapfile -t GPU_NAMES < <(python3 -c 'import torch; [print(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]')
if ((${#GPU_NAMES[@]} != EXPECTED_GPUS)); then
    echo "Chimera production launcher requires exactly $EXPECTED_GPUS visible GPUs; found ${#GPU_NAMES[@]}" >&2
    exit 1
fi
for gpu_name in "${GPU_NAMES[@]}"; do
    if [[ "$gpu_name" != *H200* && !( "$MODEL_PROFILE" == chimera && "$CHIMERA_MODEL_SIZE" == tiny && "$RECIPE" == mixrl ) ]]; then
        echo "Chimera production launcher requires H200 GPUs; found: $gpu_name" >&2
        exit 1
    fi
done
fi

if [[ "$MODEL_PROFILE" == chimera ]]; then
    python3 "$SCRIPT_DIR/preflight.py"
fi
source "$REPO_ROOT/scripts/models/$MODEL_PROFILE.sh"
if [[ "$RECIPE" == mixrl && "$CHIMERA_ROUTING_REPLAY" == 1 ]]; then
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
    # The converted SFT checkpoint supplies model weights only. Start fresh
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
    --rollout-temperature 1.0
    --rollout-top-p 1.0
    --num-steps-per-rollout 1
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --balance-data
)
if [[ "$RECIPE" == gsm8k ]]; then
    ROLLOUT_ARGS+=(--log-passrate --over-sampling-batch-size "$OVER_SAMPLING_BATCH_SIZE"
        --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
fi
if [[ "$MODEL_PROFILE" == chimera ]]; then
    ROLLOUT_ARGS+=(--rollout-stop "<end_of_turn>")
fi
ROLLOUT_ARGS+=(--apply-chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")

if [[ "$RECIPE" == mixrl ]]; then
    ROLLOUT_ARGS+=(
        --data-source-path slime_plugins.chimera_mixrl.runtime.DataSource
        --rollout-function-path slime_plugins.chimera_mixrl.runtime.generate_rollout
        --custom-rm-path slime_plugins.chimera_mixrl.runtime.reward
        --custom-reward-post-process-path slime_plugins.chimera_mixrl.runtime.post_process_rewards
    )
fi

EVAL_ARGS=(
    --eval-interval "$EVAL_INTERVAL"
    --eval-prompt-data gsm8k_validation "$EVAL_DATA"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len "$EVAL_MAX_RESPONSE_LEN"
    --eval-temperature 0.0
    --eval-top-p 1.0
    --eval-top-k 1
)
if [[ "$RECIPE" == mixrl ]]; then
    # Slime validates that an eval dataset is declared; our custom hook selects
    # enabled routes from this frozen rl_val split, never from main_test.
    EVAL_ARGS=(--eval-interval "$EVAL_INTERVAL" --eval-prompt-data mixrl "$EVAL_DATA"
        --n-samples-per-eval-prompt "$MIXRL_EVAL_SAMPLES")
fi

DAPO_ARGS=(
    --advantage-estimator grpo
    --calculate-per-token-loss
    --kl-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
)
if [[ "$RECIPE" == mixrl && "$MIXRL_OBJECTIVE" == mimo ]]; then
    DAPO_ARGS=(--advantage-estimator grpo --disable-grpo-std-normalization
        --kl-coef 0.0 --entropy-coef 0.0 --loss-type custom_loss
        --custom-loss-function-path slime_plugins.chimera_mixrl.objective.loss)
fi
if [[ "$RECIPE" == sft ]]; then
    ROLLOUT_ARGS=(--prompt-data "$SFT_DATA" --input-key messages --metadata-key metadata
        --rollout-function-path slime_plugins.chimera_mixrl.sft.generate_rollout
        --rollout-shuffle --rollout-seed 42 --num-rollout "$NUM_ROLLOUT"
        --rollout-batch-size "$ROLLOUT_BATCH_SIZE" --n-samples-per-prompt 1
        --global-batch-size "$GLOBAL_BATCH_SIZE" --apply-chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
    EVAL_ARGS=()
    DAPO_ARGS=(--loss-type sft_loss --calculate-per-token-loss
        --disable-compute-advantages-and-returns --debug-train-only)
fi

PARALLEL_ARGS=(
    # With every model-parallel dimension at one, all eight actor ranks form
    # Megatron's data-parallel group. The optimizer state is sharded over DP=8.
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
if [[ "$RECIPE" == sft ]]; then
    OPTIMIZER_ARGS+=(--lr-warmup-fraction "$SFT_WARMUP_FRACTION")
fi

SGLANG_ARGS=(
    # Slime creates eight independent TP=1 engines: rollout replica DP=8.
    --rollout-num-gpus "$ROLLOUT_GPUS"
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC"
    --sglang-cuda-graph-max-bs-decode "$SGLANG_CUDA_GRAPH_MAX_BS"
    --sglang-enable-metrics
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
    MISC_ARGS+=(--custom-model-provider-path slime_plugins.models.chimera.model_provider)
    if [[ "$RECIPE" == mixrl ]]; then
        MISC_ARGS+=(--freeze-params-name-list '(^|\.)router\.(weight|bias)$' --moe-z-loss-coeff 0.0
            --custom-megatron-before-train-step-hook-path slime_plugins.chimera_mixrl.routing.before_train_step)
    fi
fi
if [[ "$RECIPE" == mixrl && "$CHIMERA_ROUTING_REPLAY" == 1 ]]; then
    MISC_ARGS+=(--use-routing-replay --use-rollout-routing-replay)
    SGLANG_ARGS+=(--sglang-enable-return-routed-experts)
fi
SGLANG_ARGS+=(--sglang-context-length "$TRAIN_SEQUENCE_LENGTH")
MISC_ARGS+=(--seq-length "$TRAIN_SEQUENCE_LENGTH" --max-position-embeddings "$MODEL_MAX_CONTEXT")
if [[ "${DUMP_DETAILS:-0}" == 1 ]]; then
    MISC_ARGS+=(--dump-details "$ROLLOUT_DIR")
fi

TRAIN_COMMAND=(
    python3 "$TRAIN_ENTRYPOINT"
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
if [[ "$RECIPE" == mixrl ]]; then cp "$MIXRL_TASKS_CONFIG" "$MANIFEST_DIR/mixrl_tasks.json"; fi
if [[ "$RECIPE" == mixrl || "$RECIPE" == sft ]]; then
    tar --exclude=__pycache__ -cf "$MANIFEST_DIR/mixrl_source.tar" -C "$REPO_ROOT" \
        slime_plugins/chimera_mixrl "scripts/models/$MODEL_PROFILE.sh" train.py train_async.py \
        slime/backends/megatron_utils/model.py slime/ray/rollout.py slime_plugins/models/chimera.py \
        slime_plugins/models/chimera_context.py slime_plugins/models/chimera_geometry.py \
        slime_plugins/models/chimera_precision.py slime_plugins/models/chimera_sglang_precision.py \
        examples/chimera/runtime/sitecustomize.py \
        slime/backends/megatron_utils/actor.py slime/utils/routing_replay.py slime/backends/megatron_utils/loss.py \
        slime/backends/megatron_utils/cp_utils.py examples/chimera/patches
fi
if [[ "$RECIPE" == sft ]]; then cp "$SFT_DATA.manifest.json" "$MANIFEST_DIR/sft_data.json"; fi
git -c safe.directory="$REPO_ROOT" -C "$REPO_ROOT" rev-parse HEAD > "$MANIFEST_DIR/slime_commit.txt"
if [[ "$MODEL_PROFILE" == chimera ]]; then
    git -c safe.directory="$CHIMERA_TRANSFORMERS_ROOT" -C "$CHIMERA_TRANSFORMERS_ROOT" rev-parse HEAD > "$MANIFEST_DIR/transformers_commit.txt"
fi
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
    printf 'CONTEXT_PHASE=%q\n' "$RESOLVED_CONTEXT_PHASE"
    printf 'MODEL_MAX_CONTEXT=%q\n' "$MODEL_MAX_CONTEXT"
    printf 'TRAIN_SEQUENCE_LENGTH=%q\n' "$TRAIN_SEQUENCE_LENGTH"
} > "$MANIFEST_DIR/run_paths.env"
printf '%q ' "${TRAIN_COMMAND[@]}" > "$MANIFEST_DIR/train_command.sh"
printf '\n' >> "$MANIFEST_DIR/train_command.sh"

echo "Run directory: $RUN_DIR"
echo "Checkpoint context: $RESOLVED_CONTEXT_PHASE maximum=$MODEL_MAX_CONTEXT; run sequence cap=$TRAIN_SEQUENCE_LENGTH"
echo "Megatron actor: dense-DP=$POLICY_GPUS, expert-DP=$((POLICY_GPUS / EXPERT_MODEL_PARALLEL_SIZE)), TP=PP=CP=ETP=1, EP=$EXPERT_MODEL_PARALLEL_SIZE, distributed optimizer"
if [[ "$RECIPE" == sft ]]; then
    echo "SFT: verified final-turn-only targets, one pass, no SGLang generation or judging"
else
    echo "SGLang rollout: $ROLLOUT_GPUS independent TP=1 engines, mode=$EXECUTION_MODE colocate=$COLOCATE"
fi
echo "$RECIPE batch: $ROLLOUT_BATCH_SIZE prompts x $N_SAMPLES_PER_PROMPT responses = $GLOBAL_BATCH_SIZE samples"
if [[ "$DRY_RUN" == 1 ]]; then
    echo "Dry run only: command/manifests written; no Ray services or training started."
    exit 0
fi

ray stop --force >/dev/null 2>&1 || true
GPU_METRICS_PID=
if [[ "$GPU_METRICS_INTERVAL" != 0 ]]; then
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv --loop="$GPU_METRICS_INTERVAL" > "$LOG_DIR/gpu_metrics.csv" 2> "$LOG_DIR/gpu_metrics.err" &
    GPU_METRICS_PID=$!
fi
trap 'if [[ -n "$GPU_METRICS_PID" ]]; then kill "$GPU_METRICS_PID" 2>/dev/null || true; fi; ray stop --force >/dev/null 2>&1 || true' EXIT
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
    "MIXRL_WALLCLOCK_SECONDS",
    "MIXRL_FINAL_RESERVE_SECONDS",
    "MIXRL_INITIAL_UPDATE_SECONDS",
    "MIXRL_STOP_FILE",
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

ray job submit \
    --address="http://${MASTER_ADDR:-127.0.0.1}:${RAY_DASHBOARD_PORT:-8265}" \
    --runtime-env-json="$RUNTIME_ENV_JSON" \
    -- "${TRAIN_COMMAND[@]}" 2>&1 | tee "$LOG_DIR/train.log"
