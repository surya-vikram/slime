# Synchronous Chimera MixRL implementation

Updated 2026-09-27. Local checks and tiny-model two-H200 integration pass. A
full YaRN checkpoint route-replay/precision diagnostic now also passes on two
H200s; full-size optimizer/training acceptance remains pending. Not yet
production-qualified.
This ledger supplements MIXRL_RUNBOOK.md and the workspace design documents.
No ScaleRL, SFT, async-policy, MOPD, GAR or token-shaping implementation in scope.

## Locked contract

- One entrypoint: `examples/chimera/train.sh`.
- YaRN maximum 32768/factor 4/original 8192; train sequence 16384,
  admission headroom 1024; DP only, FP32 distributed optimizer, weight decay 0.
- Frozen router weights and correction bias; experts trainable; recorded expert replay.
- MiMo-style detached importance weighting, fixed sign-specific bounds [0.2, 5],
  informative-prompt mean of group-token-normalized losses, zero KL/entropy.
- Fixed per-route quotas and independent seeded epochs. Constant groups have zero
  loss; all-constant batches skip optimizer/scheduler. RL truncations masked.
- Asynchronous per-response scoring; synchronous policy updates and checkpoints.
- Quick evaluation on 128 rl_val prompts, four responses, binary pass@4;
  equal-domain score, separate continuous quality scores and cap rate.

## Current work and evidence

- [x] Expose weight decay 0, gradient clip, Adam betas; pin them in immutable run config.
- [x] Explicit positional metadata override with old/new provenance; no checkpoint rewrites.
  Missing/inconsistent checkpoint metadata still fails. Runtime HF geometry stays canonical.
- [x] Replay persistence accepts trusted tiny geometry (8 layers/top-2) as well as
  production geometry (25/top-4), rejects cross-architecture caches.
- [x] Check frozen router/bias after optimizer steps, including final update.
- [x] Expanded pinned-image regression: 89 tests passed; Megatron architecture: 20 passed.
- [x] Tiny local GPU forward/backward, capture, native replay and frozen-state tests.
- [x] Tiny HF/MCore numerical comparison, live SGLang/HF replay comparison,
  and exact 229-tensor Bridge roundtrip. See numerical caveat below.
- [x] Production launcher defaults, explicit canonical tiny profile, replay diagnostics.
- [x] Quick/main tiers, baseline/final evaluation including early-stop boundaries;
  scorer-source packaging and all-16-route fixture coverage.
- [x] Native tiny actor expert updates, synchronous save, exact resumed vs uninterrupted
  model/optimizer tensors, standalone Bridge export of the updated actor.
- [x] Full canonical-tokenizer admission: 86,450 training and 128 validation prompts;
  only the existing 197 APPS exclusions, no additional length removals.
- [x] Local report and current runbook: LOCAL_VALIDATION.md and MIXRL_RUNBOOK.md.
- [x] Tiny two-H200 integration: authentic capped-batch skip/save, then controlled
  synthetic-reward native MiMo updates, DP2 broadcast/offload and checkpoint resume.
  Loaded iteration2, continued update3 and saved. Mean train/rollout logprob gaps
  0.00330/0.00360/0.00360; finite grad norms6.45/6.64/6.87. Tensor audit:12 router
  states unchanged,12 stacked expert tensors changed,79 model tensors finite.
- [x] Fix pre-first-update optimizer save (including pinned MCore/new TE signature)
  and rebroadcast unchanged weights after skipped-save rollout offload. Without
  the latter, SGLang onload allocates memory but does not restore its weight data.
  Current focused MixRL regression suite:79 tests pass.
- [x] Full YaRN checkpoint (iter3478),4,352-token two-H200/EP2 forward trace:
  recorded SGLang expert IDs replayed exactly across23 MoE layers and4,351
  positions. Natural expert-set choices differed on3,465 token-layer rows.
  Matched FP32 head reduced selected-token logprob gap from mean/P95/max
  1.0015e-4/5.5001e-4/3.2664e-3 to5.0360e-5/2.8898e-4/9.3161e-4.
  Layer-boundary comparison found0.1352% relative-L2 discrepancy at dense
  layer0 before MoE; final layer dropped from3.8496% natural to0.8970% replay.
  See `MIXRL_RUNBOOK.md` for scope and caveats. Live SGLang FP32-head startup
  and production Radix-cache route-repeatability remain unverified.
- [ ] Full-size optimizer/training acceptance: actual frozen-router/bias behavior
  across a 10B optimizer update, replayed backward, finite gradients, policy
  broadcast, CUDA-graph activity after capture, 16K capacity, save/resume and a
  real-reward learning trend. No learning-improvement claim yet.

HF/MCore tiny BF16 logits were close, not identical: cosine 0.999935,
maximum absolute difference 0.023438. This is above the strict 0.01 absolute
parity guideline; do not label it a strict parity pass. Live SGLang/HF replayed
logprobs passed the declared 0.05 maximum/0.01 mean thresholds on six short cases.

CPU fixtures do not certify actual SGLang capture, TE attention graphs, distributed
optimizer or checkpoint parity. Random tiny weights test mechanics, not model quality.
Keep those gates explicit rather than treating unit-test success as end-to-end success.
