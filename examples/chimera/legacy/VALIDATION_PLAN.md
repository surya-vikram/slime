# Synchronous MixRL GPU qualification

## Current plan — supersedes historical Qwen/SFT instructions below

2026-09-26: local canonical tiny checks are recorded in LOCAL_VALIDATION.md.
Request one machine with **2 H200s**, not four initially. Stage code/data before
rental; allow 60–90 minutes maximum, including a final15-minute artifact/save
reserve. This is a budget, not an ETA. No SFT, async, MOPD or long quality run.

1. Run tiny Chimera through the integrated synchronous Ray loop on DP2. Verify
   captured rollout routes, actual MiMo backward, per-rank frozen router/bias,
   expert updates, gradient reduction, weight broadcast and asynchronous scoring.
2. Save/resume and compare optimizer/sampler boundaries; evaluate before/after.
   Random weights may produce constant rewards: a separately labeled controlled
   mechanics fixture may exercise backward, never serve as learning evidence.
3. Load full Chimera weights-only with the approved 32K YaRN/16K sequence cap.
   Check route-forced numerical agreement on identical tokens and inspect the
   local BF16 maximum-difference caveat before setting full-model acceptance.
4. Exercise attention graph replay after capture, offload/onload, conservative
   token budget, then measured 16K capacity. Do not switch to CP/EP or alter caps
   silently on OOM. Stop and report the smallest viable memory profile.
5. One short full-mixture update/eval/save cycle if time permits; copy small logs,
   metrics, manifest and samples. No checkpoint copying or working-stage tag until
   the relevant gates pass. No claim of learning from a mechanics smoke test.

## Historical reference-model plan — retained for provenance, do not execute

Status (2026-09-26): the dense Qwen reference completed eight fixed-batch DP2
updates across all16 routes. SFT warm-start, revised scheduling, graceful stopping
and exact resume still need GPU validation. Final Chimera MoE is not GPU-qualified.
Do not tag it as working based on a dry run. Configuration lives
in `train.sh`; scoring setup and contracts live in `MIXRL_RUNBOOK.md`.

## Allocation and preparation

Request **two H200 GPUs on one node**, inside the pinned Slime container. One GPU
cannot test DP gradients/optimizer sharding. Do not request four initially.
Next allocation: separately budget20–45min for SFT/qualification, then a bounded
three-hour RL window including final20min evaluation/save reserve. Confirm rental
ceiling before launch. Stage data/code locally first. Judge may share a tested
memory partition on one H200 if local judging dominates latency. No asynchronous-
policy experiment or GAR experiment in this allocation. The prior90min smoke plan
below is historical; use the current SFT/MixRL plan for experiment duration.
Account additionally for setup and the larger A/B/C stage panels: approximately
4–4.5h total is the initial estimate, revised after timing. Local published SFT
is frozen at2048rows/64batches; the separate64-row fixture qualifies save/resume.
The142-row curriculum and128-row production-transfer panels must be reported
separately; a256-row nine-domain main-test subset is available for stage comparison.

Image used for the local native-argument/patch checks:

```text
slimerl/slime@sha256:f7f8ee9acde9645a6e88f0c703597e69a58d2892abff56071630c88f23d5068f
```

Before reserving GPUs, stage the Slime working tree, cached Qwen3-0.6B HF
checkpoint, frozen v3 dataset and completed APPS reference audit. Transfer the
working tree, not only the last Git commit: this implementation is uncommitted.
Keep the dataset manifest and splits together; existing manifest validation may
need main_test present, but main_test is never sampled by RL. Copy the APPS audit
under `datasets/chimera-eval-data/audits/apps` or set `MIXRL_CODE_AUDIT_DIR`.
All remote writes go under `/home/jovyan/chimera-mixrl`; no host-specific paths
are embedded in production scripts. Keep Slime's pinned Megatron/dependencies.

## Gate 1: full-mixture reference run

From the transferred Slime checkout, in a clean shell:

```bash
export DATA_ROOT=/home/jovyan/chimera-mixrl
export RECIPE=mixrl MODEL_PROFILE=qwen3-0.6B EXECUTION_MODE=sync
export POLICY_GPUS=2 ROLLOUT_GPUS=2 COLOCATE=1
export HF_CHECKPOINT="$DATA_ROOT/models/qwen3-0.6b"
export MIXRL_DATA_DIR="$DATA_ROOT/datasets/chimera-eval-data"
export MIXRL_CODE_AUDIT_DIR="$MIXRL_DATA_DIR/audits/apps"
export MIXRL_SCORER_URL=http://127.0.0.1:18020
export CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
export CONTEXT_PHASE=auto TRAIN_SEQUENCE_LENGTH=8192 MAX_TOKENS_PER_GPU=8192
export MIXRL_OBJECTIVE=mimo N_SAMPLES_PER_PROMPT=4
export NUM_ROLLOUT=2 EVAL_INTERVAL=1 SAVE_INTERVAL=1
export MIXRL_MAX_ATTEMPTS=24 MIXRL_COLLECTION_TIMEOUT=600
export MIXRL_RESPONSE_CONCURRENCY=8 MIXRL_REWARD_CONCURRENCY=8
export RUN_NAME=qwen-fullmix-dp2-gate-001
# A copy of examples/chimera/mixrl_tasks.json for this gate: all 16 tasks enabled,
# prompts_per_step summing to 32, eval sized in its domains/eval sections (all nine domains).
export MIXRL_TASKS_CONFIG=$DATA_ROOT/fullmix_gate_tasks.json
PREFLIGHT_ONLY=1 bash examples/chimera/train.sh
DRY_RUN=1 bash examples/chimera/train.sh
bash examples/chimera/train.sh
```

Verify all 16 route quotas, four siblings, overlapping generation/grading,
finite loss/gradients, nonzero expert/dense updates, synchronized rollout weights,
and no NaN/OOM. Inspect behavior/current logprob discrepancy before and after
the first update; do not infer alignment merely from finite losses. Keep actual
graph-capture/replay evidence and check graphs remain active after capture.

Fixed sampled quotas now replace strict refill filtering. Constant groups stay
in the batch with zero loss/advantages. Average over informative prompts; an
entirely constant batch skips optimizer/scheduler but preserves save/eval actions.
Do not refill groups, spill domain slots, weaken correctness or fabricate variance.
Sampling cycles independently per route, without repeats inside its own epoch.

## Gate 2: synchronous resume

Use `RESUME=1` with the same run name, immutable configuration and original
`NUM_ROLLOUT=2` horizon. Test continuation from a completed non-final save boundary,
not a completed run's final marker (which has no work left). Stop cleanly at that
boundary, or use a separately preserved copy of its complete checkpoint **and
sampler state**, including the correct iteration marker. Do not edit only a marker
while leaving a later sampler state. Do not overwrite the original evidence.

Require logs identifying the loaded iteration, optimizer/scheduler/RNG state,
sampler continuation and next rollout. No duplicate accepted groups or stale
policy-cache reuse. Fresh SFT imports are weights-only; exact RL resume is not.

## Gate 3: useful learning evidence, only if time permits

Measure update and eval time first. Choose a fresh run's horizon once, fitting
the remaining budget with a 15-minute reserve. Keep the task file's domain eval sizes fixed
across compared runs so baseline/final evaluation uses the same prompts.
Keep four samples, per-domain means, binary pass@4 and equal-domain aggregate.
Compare the same prompts/scoring protocol and report cap rates beside scores.
Neither sixteen smoke prompts nor one improving draw establishes learning.
No guarantee of measurable improvement from this short integration experiment.

If local judging dominates, stop and report its measured share. Do not spend
the allocation waiting indefinitely or silently move the judge onto a policy GPU.
A later one-policy/one-judge layout is a different topology and needs an explicit
configuration; it does not qualify DP=2 throughput.

## Chimera-specific gate (requires final checkpoint)

The available legacy NoPE checkpoint cannot qualify the final YaRN architecture.
Need matching final YaRN HF and converted MCore checkpoints with generated
`run_config.yaml`. Request their paths before allocating GPUs for this gate.

Start with DP=2 and the checkpoint's phase, a sequence budget within its native
context and the 16K per-GPU admission limit. Check all exported tensors and
eager/captured HF–MCore logits/logprobs. Check frozen router projections and
correction bias against the initial state **after the final update**, not just
before intermediate steps. Check changed expert weights and exact resume.

Then use a fresh named run with `CHIMERA_ROUTING_REPLAY=1`: verify physical
decoder-layer mapping, per-token top-k IDs, replay through native Slime backward,
and CUDA graph behavior. Router top-k fusion is disabled for this gate because
it bypasses the native replay hook; attention graphs remain enabled. Synthetic
tiny models can test other YaRN geometries but cannot certify real extended
checkpoint quality. CP and >16K training are not qualified by these DP tests.

## Stop and handoff

Stop on corrupt grades, missing routes, retry exhaustion, nonfinite gradients,
weight-sync/replay failures, or a stage that cannot fit before the reserve.
At the deadline copy logs, manifests/source archive, admission/audit identities,
collection/evaluation metrics, failure records and selected rollout samples.
Do not transfer large checkpoints without permission. Document exact commands,
versions, timings, failures and remaining gates before the machine is closed.
Tag only independently passed stages; no tag for an incomplete full-mixture run.
