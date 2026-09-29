# MixRL staged validation (full mixture not yet GPU-qualified)

Entry point: `examples/chimera/train.sh`. Edit its configuration block or export
the same variables. `RECIPE=mixrl` is the default; `RECIPE=gsm8k` remains a control.
There is no Chimera MixRL working-stage tag yet. See IMPLEMENTATION_STATUS.md.
The current scope is synchronous training only. Small reference-model GPU smoke
tests have run, but full-mixture learning and final Chimera qualification remain
pending. Finish local preparation before requesting another GPU allocation.

## Choosing tasks

Everything about tasks lives in one file, `examples/chimera/mixrl_tasks.json`
(point `MIXRL_TASKS_CONFIG` at a different copy if needed):

- `tasks.<name>`: `enabled`, `prompts_per_step` (drawn from `about.train_pool`),
  `eval_prompts` (`"all"` by default, or a number up to `about.val_pool`; more is
  capped with a note), `max_response_tokens`, and an `about` block: one-line
  summary, how it is graded, the answer format the grader needs, its requirements,
  domain, verifier, judge use and pool sizes.
- `domains.<name>`: a one-line summary; the preview totals each domain.
- `eval`: `samples_per_prompt`. The preview shows what an eval costs relative to a
  training step (prompts x samples x response cap); it never blocks a launch.

The rollout batch is the sum of enabled `prompts_per_step`. `ROLLOUT_BATCH_SIZE`,
`MIXRL_QUOTAS`, `MIXRL_CAPS`, `MIXRL_*EVAL_QUOTAS`, `MIXRL_EVAL_SAMPLES` and
`MIXRL_MAIN_EVAL_*` are refused. Samples per prompt for training, eval cadence
(`EVAL_INTERVAL`) and the optimizer stay in `train.sh`. Preview without launching:

```bash
python3 -m slime_plugins.chimera_mixrl.tasks [path] --samples-per-prompt 8
```

Launch stops, naming the task and the reason, when:

- `about` disagrees with rl_train/rl_val or with the reward service;
- an enabled task needs the judge and `/health` reports it unreachable (checked
  again before every rollout, so a judge that dies mid-run stops the next step);
- the reward service's startup grading check failed for an enabled task
  (missing package, checker data or code sandbox).

There is one reward-service setup: it always has its judge configured. Tasks that
never call the judge (gsm8k_train, mcqa, nemotron_if, calendar, apps) train
while the judge is down; the rest cannot start until it is up.

| judge | tasks |
|---|---|
| none | gsm8k_train, mcqa, nemotron_if, calendar, apps |
| on_miss (exact match first, judge on a miss) | reasoning_gym |
| to_pass (parser can fail, judge must pass) | structured_train |
| always | nemotron_math, openqa, science, hotpot_train, cascade_chat/lists/plans, nvidia_multichallenge(_advanced) |

The policy never exceeds 16,384 tokens (prompt + response). The judge sees the
prompt, the response and references; long multi-turn and structured tasks need
about 21K judge tokens at full response length, so run the judge and
`JUDGE_CONTEXT` at 32,768 or more (the judge supports up to 131,072).

The task file expects the v4 data published on Hugging Face
(`surya-vikram/chimera-eval-data`; download the revision pinned in
`AIRGAPPED_RUN.md`): rl_val grown from 128 to 512 prompts (56-57 per domain) by
moving seeded rows out of rl_train, only rows a response can pass; main_test is
unchanged; the move is recorded under `val_growth` in `manifest.json`. Launch
checks every task's pool sizes against the downloaded splits. Per-task audit notes
are in `MIXRL_TASK_AUDIT.md`.

Grading rules that affect what earns reward (reward service, shared with main_test):

- `<think>...</think>`: the policy is a non-thinking model that sometimes writes
  these tags. Only the answer after the last closed block is graded, so tags never
  cost reward. Reasoning that never closes, or closes with no answer after it, is
  unfinished: like a truncated response it follows the truncation policy (masked in
  training by default) and scores 0 in eval (`incomplete_rate`). `think_rate` is
  logged per task in training and eval.
- mcqa: 1.0 for the correct label in the format the prompt asks for, 0.5 in the
  other explicit format ("Answer: X" vs `\boxed{X}`), 0 otherwise; pass@k counts
  only the requested format.
- structured_train: data counts even inside explanation (a fenced block, or an
  embedded JSON value). The last one given is the answer and must match the schema;
  an earlier valid block does not rescue it. CSV cells are converted to the
  schema's types.
- calendar: the final calendar may come with explanation; the last JSON value given
  is checked.
- multichallenge: the judge gets each rubric criterion with the completed
  conversation (the response as its final turn) once; the rubric's expected verdict
  is applied by the grader and never shown to the judge.
- gsm8k keeps "exactly one `\boxed{}`": the model must follow the instruction. Rows no response can pass (CSV with nested fields, TOML with a
  top-level array schema: 830 rows) are quarantined at startup and never train.

Each task samples its own seeded pass over its pool. When a pool runs out the
task reshuffles and starts a new pass (`MIXRL_PASS` log line; `mixrl/<task>/pass`,
`pass_progress` and `deferred` metrics). A row whose family is already in the
current batch waits for the next batch instead of being skipped for the pass.

## Current Chimera synchronous entrypoint

Inside the pinned Slime image, mount this repository, the matching Transformers
checkout and your data directory. Keep the image's Megatron dependencies; do not
replace the entire installed stack. The launcher registers only Chimera and applies
the checked narrow capture fixes. Configure the scorer below first, then:

```bash
DATA_ROOT=/datasets/megadata \
HF_CHECKPOINT=/datasets/megadata/models/chimera-10b-hf \
MCORE_CHECKPOINT=/datasets/megadata/models/chimera-10b-mcore \
CHIMERA_TRANSFORMERS_ROOT=/workspace/transformers \
MIXRL_CODE_AUDIT_DIR=/datasets/megadata/datasets/chimera-eval-data/audits/apps \
MIXRL_SCORER_URL=http://127.0.0.1:18020 \
CONTEXT_PHASE=32k POLICY_GPUS=8 RUN_NAME=chimera-mixrl-001 \
bash examples/chimera/train.sh
```

Paths above are examples, not required filesystem locations. On the rented
container use a DATA_ROOT under `/home/jovyan`. Edit the configuration block in
`train.sh` or export overrides. Start with `DRY_RUN=1` to validate/record the command
without launching Ray. The HF config must already contain the approved YaRN
32K/factor-4 settings. Use `CHIMERA_CONTEXT_OVERRIDE=1` only for the approved
historical MCore positional mismatch, not to bypass missing metadata.

Defaults: full Chimera, synchronous MixRL, 8 policy GPUs, all model-parallel sizes1,
the task file's 512 prompts x8 responses, 100 rollout boundaries, 16K sequence/1K
headroom, LR1e-6, WD0, clip1, betas0.9/0.98, MiMo objective and R3 replay enabled.
The prompt total times responses must divide by the policy GPU count. Scorer/judge
GPU allocation is separate.

Evaluation runs at baseline, every10 rollout boundaries and at the final (or
early-stop) boundary, always on the same panel: each trained domain's
`eval_prompts` from rl_val x `eval.samples_per_prompt` responses, sized by the
task file. These are rollout-boundary counts: an all-constant batch does not
increment the optimizer/scheduler. Evaluation never borrows from main_test. It reports per-domain metrics,
equal-domain aggregate, binary-only pass@k and truncation/error rates separately.

Fresh initialization loads weights without optimizer; `RESUME=1` restores the
saved optimizer and sampler with an immutable run config. Synchronous Chimera
saves include matching runtime `run_config.yaml` at root/iteration, and serialize
evaluation arguments as plain metadata for standalone Bridge compatibility.
Export with the existing Megatron Chimera export workflow and matching architecture
contract (RL balance=none/z-loss=0 is supported); do not invent a second converter.

See `LOCAL_VALIDATION.md` for measured local evidence and `IMPLEMENTATION_STATUS.md`
for outstanding acceptance gates. `CHIMERA_MODEL_SIZE=tiny` is mechanics-only.

## Historical published-reference SFT warm-start (not current scope)

Use published completed demonstrations before generating new targets. Source
selection/download preparation does not need training GPUs. Do not mix the earlier
Qwen3.5-2B self-generated pilot into the approved training artifact.

The SFT adapter accepts either task-grader provenance or pinned published-reference
provenance. A published reference is **not** labeled as an independently verified
judge success. Record publisher/revision/row hash and any actual verification
(e.g. APPS sandbox or human HelpSteer ratings). Preserve source licenses.

Before freezing: exclude overlaps with both `rl_val` and `main_test`, including
conversation user turns and wrapped prompts; reject malformed/non-text/tool
trajectories, inspect quality, and tokenize with the actual policy tokenizer.
The current non-thinking warm-start preserves final answer text and excludes
publisher reasoning blocks. Only the final assistant answer and EOS receive loss;
earlier turns are context only. Exact rollout-prefix/full-template token equality
is required. The native per-message Qwen3 masker did not preserve that equality.

Freeze candidates using `python3 -m slime_plugins.chimera_mixrl.sft --help`.
Choose a batch-divisible, coverage-checked artifact; no last-batch wrapping.
The JSONL and adjacent `.manifest.json` are external data, not repository files.
For the reference-model GPU gate, edit/export these central launcher fields:

```bash
RECIPE=sft MODEL_PROFILE=qwen3-0.6B POLICY_GPUS=2 \
  HF_CHECKPOINT="$DATA_ROOT/models/qwen3-0.6b" \
  SFT_DATA="$DATA_ROOT/datasets/sft/train.jsonl" \
  TRAIN_SEQUENCE_LENGTH=16384 MAX_TOKENS_PER_GPU=16384 \
  ROLLOUT_BATCH_SIZE=32 SFT_LR=1e-5 RUN_NAME=qwen-published-sft \
  bash examples/chimera/train.sh
```

The manifest must match batch/context/template settings and frozen tokenizer hashes.
`RESUME=1` also requires the saved identical SFT manifest and matching sampler cursor;
missing cursor is a hard failure, not permission to repeat the data. `SAVE_HF=1`
enables the native Qwen HF export alongside MCore for standalone evaluation; default
is off and Chimera export remains separately unqualified. `SFT_SAVE_INTERVAL=1`
is available for an isolated resume fixture; leave empty for final-only SFT saving.
This is native SFT loss,
one pass, no rollout generation or judging. Actual GPU loading/training remains
a qualification gate; the command has only been dry-run/native-parser checked.
Fresh MixRL uses `INITIAL_ACTOR_CHECKPOINT` pointing to the saved SFT MCore folder,
`RESUME=0`, fresh optimizer/RNG. Later RL continuation uses `RESUME=1` with the same
immutable configuration, optimizer and sampler state.

## Checkpoint phase and run sequence budget

`CONTEXT_PHASE=auto` reads the HF checkpoint; explicitly selecting `8k`, `32k`,
`64k` or `128k` checks that it matches, never rewrites it. All Chimera phases use
YaRN (factors 1, 4, 8, 16; original context 8192). The imported MCore checkpoint's
generated `run_config.yaml` must agree with HF geometry. Missing or contradictory
metadata fails before Ray/GPU startup; do not substitute a repository template.

For an explicitly approved positional change, `CHIMERA_CONTEXT_OVERRIDE=1`
allows historical MCore positional metadata to differ from the canonical HF
runtime config. The manifest records every old/new field. It does not rewrite
weights, relax architecture validation, or certify long-context quality. Conflicting
root/iteration metadata still fails. An `INITIAL_ACTOR_CHECKPOINT` is validated as
the actual weight source rather than silently validating only MCORE_CHECKPOINT.

`TRAIN_SEQUENCE_LENGTH=16384` is the default total prompt/history + response budget.
For example, a 32K checkpoint may use `CONTEXT_PHASE=32k
TRAIN_SEQUENCE_LENGTH=16384 MAX_TOKENS_PER_GPU=16384`; it remains YaRN factor 4.
This does not perform context-extension training. `MODEL_CONTEXT_LENGTH` is a
legacy alias for the sequence budget; conflicting settings are rejected.

The actor receives the checkpoint positional maximum separately from the run
sequence length. SGLang receives the run context cap but keeps the checkpoint's
YaRN geometry. MixRL admission measures the actual formatted prompt plus response
cap before sampling. DP-only preflight requires the sequence cap not exceed
`MAX_TOKENS_PER_GPU`: Slime's packing budget alone allows oversized single samples
and is not an OOM guard. The budget itself is not a proven memory-capacity estimate.

CP is out of scope: MixRL rejects a run sequence cap above **16,384**, even for
a longer-context checkpoint. Full canonical-tokenizer training admission at this
ceiling retains **86,450/86,647** rows with current 2,048/4,096 response allowances.
There are **zero length exclusions**; the 197 exclusions are separate Python
verification-quality exclusions. These original counts used no extra headroom.

The current **16K buffered profile** doubles response allowances to **4,096/8,192**
and reserves **1,024 unused tokens**, configurable as `MIXRL_CONTEXT_HEADROOM`.
Full re-audit still retains **86,450 training rows and all 128 validation rows**:
no additional length exclusions. Admission requires actual chat-formatted prompt
tokens + task response cap + headroom <= run context. Thus 8,192-response routes
allow at most 7,168 prompt/history tokens, and 4,096-response routes allow 11,264.
These caps include thinking and answer tokens; early stopping remains normal.
Headroom is not secretly added to generation, and no prompt is shortened.

8K checkpoints keep the smaller response caps; the new headroom also applies to
them, so the historical 473 exclusions without headroom are not the current 8K
count. Re-audit after changing context, response caps, tokenizer or chat template.
Larger allowances reduce cap pressure but do not guarantee no truncated response.

The canonical Megatron workflow skill guided these invariants. Slime's image
dependencies remain pinned; the existing YaRN CUDA-capture patch is unchanged.

## Prerequisites and topology

Use the pinned Slime image in `RUNBOOK.md`; do not replace its Megatron checkout.
Mount this Slime checkout and the **v3** frozen dataset. The separate local
`chimera-eval` checkout supplies `eval_stack.reward_service`; the published evaluator
image 0.1.1 does **not** contain this unreleased service yet.

Remote SGLang -> local scoring service -> local judge. Responses are scored as
they finish, with bounded concurrency. Optimizer updates remain synchronous.
The local Qwen3.5-2B judge validates plumbing, not production judgment quality.

Run locally from `chimera-eval` with its grader dependencies installed:

```bash
JUDGE_URL=http://127.0.0.1:8010/v1 JUDGE_NAME=eval-long2b \
JUDGE_CONTEXT=32768 JUDGE_MAX_TOKENS=1024 JUDGE_MAX_RETRY_TOKENS=2048 \
JUDGE_CONCURRENCY=2 JUDGE_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}' \
python -m eval_stack.reward_service --data-dir "$DATA_DIR" \
  --cache-dir "$SCORER_CACHE" --port 8020 --workers 4 \
  --judge-revision "$IMMUTABLE_JUDGE_ID"
```

Keep the service running, then establish the reverse tunnel **from the local host**:

```bash
ssh -NT -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 -R 127.0.0.1:18020:127.0.0.1:8020 root@REMOTE
```

Use host networking if SSH terminates outside the training container. Never bind
the unauthenticated scorer publicly. Check remote `curl -f localhost:18020/health`
before launching Ray. This launcher owns/stops Ray in its dedicated container;
do not share that container with another Ray job.

## Small reference model first

Copy the cached Qwen3-0.6B HF checkpoint, Slime working tree, and v3 dataset to the
remote persistent data root. No Chimera code patch or separate conversion is
needed for Qwen: this Slime revision already implements native Qwen3 HF loading.
That loading path has been exercised in the earlier GPU smoke; a full-mixture
run and final Chimera validation are still required.

Example settings inside the remote container (all artifacts under the data root):

```bash
export DATA_ROOT=/home/jovyan/chimera-mixrl
export RECIPE=mixrl MODEL_PROFILE=qwen3-0.6B POLICY_GPUS=4
export HF_CHECKPOINT=$DATA_ROOT/models/qwen3-0.6b
export MIXRL_DATA_DIR=$DATA_ROOT/datasets/chimera-eval-data
export MIXRL_SCORER_URL=http://127.0.0.1:18020
export CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
export CONTEXT_PHASE=auto TRAIN_SEQUENCE_LENGTH=8192 N_SAMPLES_PER_PROMPT=4
# A copy of mixrl_tasks.json with only mcqa enabled, prompts_per_step 4, max_response_tokens 2048:
export MIXRL_TASKS_CONFIG=$DATA_ROOT/mcqa_smoke_tasks.json
export MIXRL_MAX_ATTEMPTS=32 MIXRL_COLLECTION_TIMEOUT=600
export MIXRL_RESPONSE_CONCURRENCY=8 MIXRL_INFLIGHT_GROUPS=2
export NUM_ROLLOUT=2 EVAL_INTERVAL=1 SAVE_INTERVAL=1
export RUN_NAME=qwen-mcqa-smoke-001
PREFLIGHT_ONLY=1 bash examples/chimera/train.sh
bash examples/chimera/train.sh
```

Preflight verifies data, scorer identity, model architecture, batch divisibility,
and hashes checkpoint bytes. It does **not** test CUDA, loading or training.
Real-tokenizer admission is checked by the datasource before rollout. Exact
restart uses `RESUME=1`, the same run name/config and original `NUM_ROLLOUT`.
Megatron validates the saved scheduler horizon even with constant LR; extending
it needs a separately explicit scheduler policy, not a silent larger rollout count.
Fresh runs load weights only; saves retain optimizer/RNG/sampler state.

The single-route command above is diagnostic only. For full MixRL, use the
default task file: 512 sampled prompt groups across all 16 routes/nine domains.
There is no replacement sampling: constant-reward groups are fully loss-masked.
Sampling is seeded without replacement within each route pass, with cursors and
waiting rows saved for resume. Exhausted routes reshuffle independently and keep
their sampling quotas; slots never spill into another domain. Hard/easy
weighting is deferred.

Before enabling APPS, run this once in the evaluator environment with Docker
access (dataset code runs only in the restricted child container):

```bash
PYTHONPATH=. python tests/audit_mixrl_apps.py \
  --data-dir "$DATA_DIR" --output-dir "$DATA_DIR/audits/apps" --workers 4
```

Set `MIXRL_CODE_AUDIT_DIR` to that output directory in the training launcher.
Preflight verifies every audit row hash, rejects incomplete/infrastructure-failed
audits, and excludes rows without a passing reference/negative control. Explicit
non-unique-output hints are conservatively excluded because this small APPS
adapter uses exact whitespace-normalized output, not task-specific checkers.
This is a runtime admission manifest, not a modification of the published split.

Defaults remain eight policy GPUs. Smoke DP4 is configurable; every model-parallel
dimension is one, with distributed optimizer and one TP1 rollout engine per GPU.
Chimera retains its established MCore checkpoint/Transformers registration path.

## Reward and truncation contracts

- Same versioned evaluator graders: MCQA extraction, mathematical verification,
  and OpenQA judging. No gold/reference fields are sent to the target model.
- Grading removes a final tokenizer EOS string only when the completed sample's
  last token ID is that EOS. Raw rollout text, token IDs, log-probabilities and
  training masks are preserved. Do not feed this transport marker into strict
  JSON/calendar graders or strip arbitrary special tokens from the answer.
- `MIXRL_TRUNCATION=mask`: capped-response tokens have zero loss; censored scores
  are excluded from sibling mean/std. Keep completed siblings. Accept only groups
  with at least two eligible responses and nonzero reward spread.
- `zero` is an explicit alternative; fixed-budget **evaluation** always scores
  capped responses zero. Judge/transport failures are errors, never zero rewards.
- Generate exactly the sampled quotas; do not replace constant groups or reroll
  individual responses until successful. Constant groups remain in the fixed DP
  batch with zero token masks and zero advantages. MiMo advantages are scaled by
  total groups / informative groups before DP partitioning, so they do not dilute
  the informative-prompt mean. An entirely uninformative batch skips optimizer
  and scheduler execution. Bounded waves retain proposal order.
- Native diagnostic means retain the fixed response-count reporting denominator.
  Divide `importance_ratio`, `entropy`, `importance_masked_fraction` and
  `train_rollout_logprob_abs_diff` by `effective_response_fraction` to interpret
  them over loss-active responses. Do not apply that correction to loss, which
  already has its prompt-normalization correction.
- `MIXRL_OBJECTIVE=mimo` selects mean-centered advantages, detached importance
  weights, sign-specific fixed bounds (both default `[0.2,5.0]`), zero KL and
  prompt-balanced group-token reduction. `dapo` selects the original standardized
  advantages/per-token PPO loss with 0.2/0.28 clipping as a comparison control.
  These are different objectives, not interchangeable names.
- MiMo normalization uses a per-response advantage multiplier
  `group_size * eligible_response_tokens / eligible_group_tokens` with the
  native per-response mean and response-count outer normalization. This equals
  a prompt mean of group-token means. Cap masks precede denominator construction;
  later importance masks do not change that denominator. Keep one Slime rollout
  ID per response; do not collapse all siblings to one scheduling ID.
- Router projections freeze through Slime's native parameter patterns. Bias
  update rate and auxiliary router z-loss are zero. A before-step hook checks
  frozen state has not changed and expert parameters remain trainable.
- `CHIMERA_ROUTING_REPLAY=1` is the default Chimera MixRL path. It applies a
  narrow SGLang Transformers expert-capture patch, maps physical decoder layers
  2..24 (not compressed MoE indices 0..22), validates token/layer/top-k shapes
  and distinct in-range IDs, and uses Slime's native replay. Router top-k fusion
  is disabled because the pinned TE fused route bypasses Slime's replay hook;
  attention CUDA graphs remain enabled. Captured IDs use compressed non-pickle
  persistence. **Do not claim replay is GPU-qualified from CPU tests.**
- Periodic eval uses only enabled routes in `rl_val`, no filtering. Main-test
  never enters training or this monitoring path.
- `domains.<name>.eval_prompts` sets each trained domain's fixed eval panel.
  Selection is deterministic from the seed. Failed judging retries the same
  persisted evaluation draws, not a new easier sample. `evaluation.json` records
  per-task and per-domain scores, binary-only pass@k, cap rates, and the equal
  domain aggregate. Native fractional-reward pass-rate logging is disabled.
- Training logs include raw proposed rewards, acceptance/cap rates, collection
  time, entropy, importance ratios/masked fraction and logprob discrepancy.
  Exceptions leave a `failure.json` and persist completed generations for retry.
- GAR is a later correctness-gated experiment. Adaptive bounds remain deferred;
  token-level shaping, async policy training and MOPD are outside current scope.

## GPU acceptance and time budget

See `VALIDATION_PLAN.md` for the two-H200 full-mixture commands and ordered gates.

Request an explicit GPU count (maximum four, smallest useful topology preferred).
Budget the next session for **60–90 minutes maximum**, not a guaranteed runtime.
Reserve the final 15 minutes for synchronous save and copying small artifacts.
Check time before each stage; stop rather than launch a stage that cannot finish.

1. Record hardware, image/code/data hashes and tunnel health. Measure startup.
2. Single-route updates: verify loaded weights, finite gradients/loss, changed
   policy weights, behavior logprobs, `train_rollout_logprob_abs_diff`, CUDA graph
   activity and no OOM/NaN. Inspect actual completions and rewards.
3. Restart from a saved iteration: prove optimizer and sampler continuation;
   no repeated accepted groups or stale-policy cache reuse.
4. Full nine-domain run: exact accepted quotas, cap rates, raw rewards,
   rejection counts and timestamps showing generation/judging overlap.
5. Copy logs, resolved configs, admission/collection metrics and selected rollout
   samples. No large checkpoints copied back without explicit request.

Only tag a stage after its GPU gates pass. A Qwen pass does not validate Chimera's
MoE conversion, graph/YaRN path or expert-routing replay. Small smoke tests cannot
establish reward quality or statistically meaningful learning improvements.

Checkpoint boundary checks matter even when no optimizer step occurs. A fully
masked first batch must initialize empty Adam state before saving, without a
dummy update. After checkpoint-time rollout offload, SGLang's weight onload only
reallocates storage: rebroadcast the unchanged actor before evaluation or another
rollout. The synchronous skip branch handles both cases. Do not remove that
broadcast because "the policy did not change"; it restores discarded weight data.
The tiny two-H200 acceptance covers this boundary and subsequent resume; see
`IMPLEMENTATION_STATUS.md` for separate full-model acceptance status.

## Experimental async comparison

**Deferred, not in the current execution plan.** The following documents existing
experimental code only; do not rent GPUs or run it as part of the synchronous work.
Earlier the user authorized a sequential four-H200 comparison. First run the colocated DP4
audit above. Then use `EXECUTION_MODE=async COLOCATE=0 POLICY_GPUS=2 ROLLOUT_GPUS=2
USE_ROLLOUT_LOGPROBS=1` with a fresh run name and `SAVE_INTERVAL=NUM_ROLLOUT`.
This selects upstream `train_async.py`, not a new scheduler. Its one-batch
lookahead overlaps next-batch generation with the current optimizer update;
`--update-weights-interval 1` drains generation before every weight swap.
No fully asynchronous trajectory queue or mid-generation weight changes are used.

Async smoke saves only at the final drained boundary. Mid-lookahead save/resume
is deliberately prohibited because the sampler can be ahead of the optimizer;
exact async restart needs separate queue/checkpoint work. Synchronous resume is
still part of the audit. Native behavior logprobs enter the PPO ratio for async.
Compare against `EXECUTION_MODE=sync COLOCATE=0 POLICY_GPUS=2 ROLLOUT_GPUS=2
USE_ROLLOUT_LOGPROBS=1` to separate scheduling from allocation differences.
Record both run startup and steady-state costs; a few updates are not learning
evidence. Do not compare DP4-colocated throughput as if topology were identical.

## Locked routing and LM-head alignment

For `RECIPE=mixrl MODEL_PROFILE=chimera`, the launcher defaults
`CHIMERA_ROUTING_REPLAY=1` and `CHIMERA_FP32_LM_HEAD=1`. These are the selected
baseline, not optional performance experiments. Set either to `0` only for an
explicit diagnostic A/B run. The 0.6B reference and GSM8K control do not inherit
the Chimera-only FP32 default.

Routing replay records each rollout's ordered expert IDs from SGLang and reuses
those exact IDs in the actor. The launcher applies the pinned SGLang capture
patch and disables only fused router top-k (which bypasses replay); attention
CUDA graphs remain available. Replay must remain enabled through backward. The
FP32 option changes only LM-head projection arithmetic on actor and rollout;
weights stay BF16 and actor gradients use Megatron's FP32 main-grad buffers.
Confirm `CHIMERA_HEAD_RUNTIME` and SGLang's `enable_fp32_lm_head=true` at startup.

On 2026-09-27, the final YaRN checkpoint was tested on two H200s, with the full
MCore actor initialized EP=2, TP=PP=CP=ETP=1. A 4,096-token prompt plus 256-token
response was compared (4,351 causal input positions, 25 layers, top-4). Natural
MCore top-k differed from SGLang on 3,465 token-layer expert sets. Under replay,
all 4,351 × 23 MoE-layer ordered routes matched exactly. The ordered captured
routes were also replayed through a teacher-forced SGLang request with zero
mismatches.

On the same replayed sequence, matched default-head selected-token logprob gap
was mean `1.0015e-4`, P95 `5.5001e-4`, max `3.2664e-3`; matched FP32 projection
gap was mean `5.0360e-5`, P95 `2.8898e-4`, max `9.3161e-4` (256 response tokens).
This is about 2× lower mean and 3.5× lower maximum. The captured SGLang server
had FP32-head serving disabled; its FP32 values were calculated directly from
the captured pre-head states. Thus the FP32 arithmetic and actor path were
measured, but a live SGLang server started with the FP32 flag remains an
acceptance check.

Layer-boundary tracing found relative-L2 error already at 0.1352% after dense
layer 0, before any MoE. At layer 24 it was 3.8496% with natural routing and
0.8970% with replay; pre-LM-head error was 1.5273% and 0.9603%, respectively.
Replay fixes route mismatch, not all backend numeric drift. The next diagnosis
is the first dense block's RMSNorm/QKV/YaRN/attention/output/residual/MLP
boundaries. Keep RMSNorm rounding alignment off by default until that A/B is
isolated; do not combine more numerical switches into this baseline.

The two exact route-repeatability captures used a 4,352-token sequence with
Radix cache disabled (one run had CUDA graphs enabled); a shorter prior default-
cache capture showed decode-route differences. Exact repeatability with the
production Radix-cache configuration remains open. Per-trajectory capture and
replay are exact and do not require separate requests to choose the same routes.

Router-freeze evidence: the configured regex sets router projection parameters
`requires_grad=False`; bias-update rate and router load-balancing are zero; and
the step hook verifies router/bias tensors before and after optimizer steps.
All six focused replay/freeze tests passed, including an Adam update proving
router weights stay fixed while expert weights update and a deliberate bias
mutation is rejected. A full 10B optimizer step was not part of this numerical
diagnostic; retain that as an integrated-training acceptance gate.

### Other numerical experiments (not enabled in the baseline)

`CHIMERA_MATCH_RMSNORM=1` switches only Chimera's SGLang RMSNorm rounding to
cast after weight multiplication, matching the tested TE RMSNorm arithmetic.
It preserves epsilon and weights. Default0; keep it disabled until the isolated
layer-boundary diagnosis is complete. Any switch change requires a fresh run,
not a silent resume.

Two additional rollout-only experiments default to0:
`CHIMERA_MATCH_DENSE_SWIGLU=1` uses SGLang's fused SiLU-and-multiply kernel in
Chimera's dense MLPs, preserving gate/up/down weights and leaving fused MoE
kernels unchanged. `CHIMERA_SGLANG_FULL_BF16_REDUCTION=1` disables PyTorch CUDA
BF16 reduced-precision matmul reduction in the Chimera SGLang worker before model
initialization/capture. This process-wide PyTorch setting does not configure
Megatron's native Transformer Engine GEMMs. CLI helpers register the hook; it
activates only when a Chimera SGLang adapter is constructed. Standalone HF models
and the Megatron actor are not patched. Both options are recorded in immutable
MixRL configuration and propagated to Ray/SGLang workers. They are Tiny-qualified
experiments; full trained checkpoint and end-to-end training qualification remain
pending. Inspect `CHIMERA_PRECISION` stderr diagnostics for actual activation.

Keep full-support temperature1 sampling for v0 (enforced by runtime). Restricted
top-p candidate replay is not implemented. Do not widen ratio bounds to hide drift.

### Test commands

```bash
python3 -m unittest discover -s tests -p 'test_chimera_mixrl*.py' -v
python3 -m unittest discover -s tests -p 'test_chimera_context.py' -v
bash -n examples/chimera/train.sh
# Runtime tests require torch; otherwise unittest explicitly skips them.
```

`tests/check_mixrl_admission.py` additionally checks the real tokenizer and v3
dataset without using a GPU. Pass `--config` with the resolved launch manifest
to audit the full mixture (including code exclusions); `--context 16384` performs
a diagnostic length comparison without changing the original run manifest.
It needs the installed Slime processing dependencies.

`tests/live_mixrl_routes.py` exercises an already hosted local target and the
shared scorer on one rl_val example from every route. The target sees messages
only, never verifier references. This is plumbing validation, not model quality
or judge calibration. Capped cases are reported explicitly.
