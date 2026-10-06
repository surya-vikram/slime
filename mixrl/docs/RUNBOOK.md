# How MixRL works

The commands are in [`mixrl/README.md`](../README.md) and the airgapped setup in
[AIRGAPPED.md](AIRGAPPED.md). This page explains what the settings mean and the
contracts the code enforces. Scope: synchronous training; the full mixture and final
Chimera qualification on H200s are still pending.

## Tasks and eval

Everything about tasks lives in `mixrl/tasks.json`:

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
(`EVAL_INTERVAL`) and the optimizer are in `mixrl/config.env`. Preview without
launching: `mixrl/run.sh tasks`.

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

The task file expects the v5 data published on Hugging Face
(`surya-vikram/chimera-eval-data`; download the revision pinned in
[AIRGAPPED.md](AIRGAPPED.md)). v4 grew rl_val from 128 to 512 prompts (56-57 per domain) by
moving seeded rows out of rl_train, only rows a response can pass (recorded under
`val_growth` in `manifest.json`); v5 removes the 1,094 rl_train/rl_val rows no response
can pass (structured schemas CSV/TOML cannot express or that contradict themselves, APPS
with no passing or several valid outputs; listed in `removed_rows.json`), leaving 506
rl_val prompts. main_test is unchanged. Launch
checks every task's pool sizes against the downloaded splits. Per-task audit notes
are in [TASK_AUDIT.md](TASK_AUDIT.md).

Grading rules that affect what earns reward (reward service, shared with main_test):

- `<think>...</think>`: the policy is a non-thinking model that sometimes writes
  these tags. Only the answer after the last closed block is graded, so tags never
  cost reward. Reasoning that never closes, or closes with no answer after it, is
  unfinished: like a truncated response it follows the truncation policy (scored 0
  in training by default) and scores 0 in eval (`incomplete_rate`). `think_rate` is
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
- gsm8k keeps "exactly one `\boxed{}`": the model must follow the instruction.
- Rows no response can pass (CSV with nested fields, TOML with a top-level array
  schema: 830 rows) are quarantined at startup and never train.

Each task samples its own seeded pass over its pool. When a pool runs out the
task reshuffles and starts a new pass (`MIXRL_PASS` log line; `mixrl/<task>/pass`,
`pass_progress` and `deferred` metrics). A row whose family is already in the
current batch waits for the next batch instead of being skipped for the pass.

## Reward and truncation contracts

- Same versioned evaluator graders: MCQA extraction, mathematical verification,
  and OpenQA judging. No gold/reference fields are sent to the target model.
- Grading removes the completed sample's stop marker: a trailing configured stop
  string (Chimera's `<end_of_turn>`), else a final tokenizer EOS string when the
  last token ID is that EOS. Raw rollout text, token IDs (which keep the marker, so
  the policy learns to end its turn), log-probabilities and training masks are
  preserved. Do not strip arbitrary special tokens from the answer.
- `MIXRL_TRUNCATION=zero` (default, as in MiMo's and DAPO's public recipes): a
  response with no finished answer (cut off at its cap, or reasoning that never
  closed) is a wrong answer: reward 0, counted in its group's mean, trained with a
  negative advantage. `mask` is the alternative: such responses get zero loss and
  are excluded from sibling statistics. A capped response is scored 0 without a
  reward-service request (the grader's fixed-budget rule never reads it); eval
  always scores it 0. Judge/transport failures are errors, never zero rewards.
- A group trains only with at least two eligible responses and nonzero outcome
  spread. `MIXRL_LENGTH_PENALTY=1` applies MiMo's group-relative length penalty
  (paper Eq. 4, public recipe values; matches MiMo's code exactly): in groups where
  more than half passed (score >= 0.5), a passing response more than 30% longer
  than the median passing length loses up to 0.1, ramping as t^1.5 to the full
  deduction at twice the median. It shifts the scores used for advantages only;
  logged and eval scores stay raw, and it never makes an all-correct group trainable.
- Refill (`MIXRL_REFILL_ROUNDS`, default 1; 0 = off): a group without outcome
  spread is replaced by a fresh prompt of the same task while the task is short of
  `prompts_per_step` informative groups (counting groups still generating), up to
  (1 + rounds) x `prompts_per_step` prompts per task per step. Then the step
  continues with what it has; a short task fills its slots with its constant
  groups (zero loss) so the DP batch keeps its fixed size. Groups are resolved in
  proposal order, so a retried step proposes the same prompts and reuses its saved
  responses. MiMo advantages are scaled by total groups / informative groups before
  DP partitioning, so padding does not dilute the informative-prompt mean. An
  entirely uninformative batch skips optimizer and scheduler execution.
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
- Periodic eval uses only enabled tasks in `rl_val`, no filtering. Main-test
  never enters training or this monitoring path.
- `tasks.<name>.eval_prompts` sets each task's fixed eval panel ("all" by default).
  Selection is deterministic from the seed. Failed judging retries the same
  persisted evaluation draws, not a new easier sample. `evaluation.json` records
  per-task and per-domain scores, binary-only pass@k, cap rates, and the equal
  domain aggregate. Native fractional-reward pass-rate logging is disabled.
- Training logs include raw proposed rewards, acceptance/cap rates, collection
  time, entropy, importance ratios/masked fraction and logprob discrepancy.
  Exceptions leave a `failure.json` and persist completed generations for retry.
- GAR is a later correctness-gated experiment. Adaptive bounds remain deferred;
  token-level shaping, async policy training and MOPD are outside current scope.

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
must equal it if set; conflicting settings are rejected.

The actor receives the checkpoint positional maximum separately from the run
sequence length. SGLang receives the run context cap but keeps the checkpoint's
YaRN geometry. MixRL admission measures the actual formatted prompt plus response
cap before sampling. DP-only preflight requires the sequence cap not exceed
`MAX_TOKENS_PER_GPU`: Slime's packing budget alone allows oversized single samples
and is not an OOM guard. The budget itself is not a proven memory-capacity estimate.

CP is out of scope: MixRL rejects a run sequence cap above **16,384**, even for
a longer-context checkpoint. Response caps are 4,096/8,192 per task (`tasks.json`)
with **1,024 reserved tokens** (`MIXRL_CONTEXT_HEADROOM`). Admission requires actual
chat-formatted prompt tokens + task response cap + headroom <= run context; at 16K
this drops 1 multichallenge and 7 structured_train prompts (see TASK_AUDIT.md). Thus 8,192-response routes
allow at most 7,168 prompt/history tokens, and 4,096-response routes allow 11,264.
These caps include thinking and answer tokens; early stopping remains normal.
Headroom is not secretly added to generation, and no prompt is shortened.



## Locked routing and LM-head alignment

For Chimera the launcher defaults `CHIMERA_ROUTING_REPLAY=1` and
`CHIMERA_FP32_LM_HEAD=1`. These are the selected baseline, not optional
performance experiments. Set either to `0` only for an explicit diagnostic A/B
run. The 0.6B reference profile does not inherit them.

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

### Train/rollout consistency (measured 2026-09-29, 2xH200, SFT iteration 3478)

Each training step prints `MIXRL_CONSISTENCY` (per-token KL k1/k3 between SGLang's
behaviour log-probs and Megatron's, mean/max |log-prob| and |prob| gap, max by
response position and by likelihood, and `routes_overridden_by_replay`: the share of
token-layer expert choices where Megatron's own router would have picked other experts than
the rollout's; training always uses the rollout's) and `MIXRL_WORST_TOKENS` (the eight largest gaps with position,
token id and both log-probs). Live measurements, one step each, about 50-100K tokens:

| Run | KL k3 | mean |d| | max |d| | tokens |d|>0.1 | routes replay changed |
|---|---|---|---|---|---|
| route replay, top-p 1.0 | (not logged) | 0.0132 | 0.70 | 0.46% | 8.0% |
| route replay, top-p 1.0, RMSNorm aligned | 1.8e-4 | 0.0113 | 0.58 | 0.22% | 9.1% |
| route + top-p 0.95 replay | (not logged) | 0.0111 | 0.43 | 0.27% | not counted |

For reference, the R3 paper (arXiv 2510.11370) reports SGLang/Megatron KL 7.5e-4
for Qwen3-30B-A3B with R3 (1.5e-3 without) and 6.4e-4 for dense Qwen3-8B, and finds
tokens with probability ratio above 2 even for dense models. No token here reached
|d| > 1; the largest gaps sit mid-response, mostly deep in long responses, not at the
first or last token. Top-p candidate-set replay adds no gap.

YaRN (factor 4, original 8K, max 32K; run sequence cap 16K) was compared directly:
SGLang's HF `ChimeraRotaryEmbedding` and Megatron's `YarnRotaryEmbedding` use the
same positions (packed sequences restart at 0; slime sizes the table by the longest
sequence) and the same scale (1.138629), and their fp32 cos/sin agree to 2e-6. In
bf16 they are bit-identical below position 1,664 and differ by occasional single-ulp
roundings beyond it.

`CHIMERA_MATCH_RMSNORM=1` (default for Chimera) makes SGLang's RMSNorm cast after the
weight multiplication, as TE does. Forward-only parity on 12 prompts gave KL k3
1.8e-4 with it and 3.0e-4 without; the live run above confirms it. Any switch change
requires a fresh run, not a silent resume.

### Other numerical experiments (not enabled in the baseline)

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

Rollouts sample at `ROLLOUT_TEMPERATURE=1.0`, `ROLLOUT_TOP_P=0.95`, `ROLLOUT_TOP_K=20`
(MiMo's public code recipe; top-k 20 is also Qwen3's default); `rl_val` eval samples the
same way. On this checkpoint (about 200B pretraining tokens) top-p 0.95 alone gave
candidate sets with median 8 but p90 978 ids, and 9.1% of sampled tokens ranked beyond
20 (HotpotQA, 13.6K tokens); top-k 20 caps the replay payload and that tail. Top-k needs
top-p < 1: slime records and replays candidate sets only then. With top-p < 1, SGLang
records each generated token's top-p candidate set and returns its log-prob
renormalized over that set (plus the sampled token); the loss renormalizes the
actor's log-prob over the same set (MiMo's top-p candidate-set replay; slime's
trainer force-keeps the target token the same way). The runtime refuses a response
without aligned candidate sets. `top_p_set_mean` in `MIXRL_COLLECTION` is the mean
set size per generated token: MiMo reports under 5 at top-p 0.97 for its model. It
also sizes the replay payload (Ray transfer and saved responses), so check it in the
first H200 steps; a random-weight model gives ~44K (nearly the whole vocabulary).
Temperature scaling is applied identically on both sides. Do not widen ratio bounds
to hide drift.

## Experimental async comparison

**Deferred, not in the current execution plan.** The following documents existing
experimental code only; do not rent GPUs or run it as part of the synchronous work.
It uses `EXECUTION_MODE=async COLOCATE=0 POLICY_GPUS=2 ROLLOUT_GPUS=2
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

## GPU acceptance gates

1. Record hardware, image/code/data revisions. Measure startup.
2. First updates: loaded weights, finite loss and gradients, changed policy weights,
   behavior logprobs, `train_rollout_logprob_abs_diff`, CUDA-graph activity, no
   OOM/NaN, frozen router. Inspect actual completions and rewards.
3. Resume from a saved iteration: optimizer and sampler continue exactly; no
   repeated groups or stale-policy cache reuse.
4. Full mixture: exact quotas, cap rates, raw rewards and timestamps showing
   generation/judging overlap.

A Qwen pass does not validate Chimera's MoE conversion, graph/YaRN path or expert
replay; small smoke tests cannot establish learning improvements.

Checkpoint boundary checks matter even when no optimizer step occurs. A fully
masked first batch must initialize empty Adam state before saving, without a
dummy update. After checkpoint-time rollout offload, SGLang's weight onload only
reallocates storage: rebroadcast the unchanged actor before evaluation or another
rollout. The synchronous skip branch handles both cases; do not remove that
broadcast because "the policy did not change".

## Tests and developer validation

```bash
python3 -m unittest discover -s tests -p 'test_chimera_mixrl*.py'
python3 -m unittest tests.test_mixrl_scripts          # run.sh / reward.sh / judge.sh with stubs
python3 -m unittest discover -s tests -p 'test_chimera_context.py'
bash -n mixrl/internal/launch.sh
# Runtime tests require CPU torch; otherwise unittest explicitly skips them.
```

`MODEL_PROFILE=qwen3-0.6B` (set when calling `mixrl/internal/launch.sh` directly
inside the slime container) is a small reference profile for mechanics checks;
`CHIMERA_MODEL_SIZE=tiny` is Chimera mechanics-only. `tests/check_mixrl_admission.py`
checks the real tokenizer against the data without a GPU (`--config` takes a
run's `manifests/mixrl_config.json`). `tests/live_mixrl_routes.py` sends one rl_val
prompt per task through an already hosted model and the reward service: plumbing
validation, not model quality.

Older design and validation notes (SFT warm-start, the GSM8K DAPO control, local
validation logs) are in `examples/chimera/legacy/`.
