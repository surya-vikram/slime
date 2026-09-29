# Canonical tiny Chimera local validation

2026-09-26. Mechanics evidence, not a learning/quality result or full H200 acceptance.
Uses the geometry in Megatron-LM `examples/chimera/tiny_chimera.sh`: 8 layers,
first2 dense, hidden512, FFN2048, 8 attention heads/2 KV heads, head64, 8 experts,
top2, expert FFN256, vocab50176, untied embeddings, QK norm, RMS epsilon1e-5.
Do not inherit pretraining optimizer or router auxiliary losses from that script.

Hardware: local RTX PRO 1000 Blackwell laptop GPU, 8GB. Pinned Slime image ID:
`sha256:f7f8ee9acde9645a6e88f0c703597e69a58d2892abff56071630c88f23d5068f`.
Bridge checks used `suryavikram6/megatron-gemma:v2-fixed` with the existing local
Megatron-LM/Bridge/Transformers checkouts. No remote GPU used for these results.

## Results

| Check | Observed result |
|---|---|
| Slime focused regression | 89 passed |
| Megatron architecture/export contract | 20 passed |
| Reward-service focused regression | 5 passed; not judge-quality certification |
| HF tiny YaRN 8K/32K/64K/128K | Finite short-probe backward, expert gradients, no router gradients |
| Fixed-weight HF repeated logits / rotary CUDA graph | Maximum difference0 |
| Native Megatron TopKRouter / Slime R3 | Exact weights/routes; upstream gradients retained; experts update, router/bias frozen |
| Actual SGLang capture buffers | CUDA graph replay preserves changed routes in tiny/full layouts |
| Bridge HF→MCore→HF | All229 tensors bitwise identical |
| HF vs MCore BF16 next-token logits | Same argmax, cosine0.999935, max difference0.023438, mean0.004042 |
| Live SGLang graph decode | Capture completed; subsequent decode logs `cuda graph: True` |
| Live SGLang vs HF forced-route logprobs | Six short cases, worst max0.015613; each mean<0.004 |
| Native tiny actor update | Finite synthetic loss/gradient; 66 expert tensors change, router/bias unchanged |
| Native actor repeat with R3 | Exact full-model repeated logits, maximum difference0 |
| Native save/resume vs two uninterrupted updates | Exact model hash/scheduler; 85 checkpoint tensors including6 optimizer tensors identical |
| Updated native actor standalone Bridge export | Passed; plain saved eval metadata avoids requiring Slime on export |
| Production data admission, actual tokenizer | 86450train/128val; existing197 APPS exclusions only; no added length exclusions |

The HF/MCore maximum exceeds the strict 0.01 absolute parity guideline. Preserve
that caveat; cosine similarity and matching argmax do not erase it. BF16 kernels
and routing are not claimed bitwise equivalent across backends. SGLang/HF replay
used predeclared max0.05/mean0.01 logprob tolerances. One natural-routing case
reached max0.0862, illustrating why route-forced agreement is measured separately.

The native actor uses a **synthetic squared-logit loss** to isolate update and
resume mechanics. Its decreasing loss is not evidence of MixRL improvement.
MiMo gradients, masks and native packed/DP reduction equivalence are separately
checked by analytic regression fixtures. All16 routes/nine domains are covered
by collector/evaluation fixtures and existing scorer anchors.

## Reproduction and artifacts

Inside the pinned Slime image with this repo and Transformers mounted:

```bash
CHIMERA_TRANSFORMERS_ROOT=/workspace/transformers \
LOCAL_VALIDATION_DIR=/datasets/megadata/local-validation \
bash examples/chimera/validate_local.sh
```

This runs CPU regressions and component GPU probes. It does not pretend to launch
the complete Ray training stack. Set `LOCAL_GPU_CHECKS=0` for CPU regressions only.
The remaining standalone diagnostics have explicit `--help` interfaces:

- `tests/local_chimera_parse_command.py`: feed a `DRY_RUN=1` generated command.
- `tests/local_chimera_sglang_parity.py`: live tiny SGLang endpoint plus matching HF.
- `tests/local_chimera_actor.py`: native actor from that command, `--updates` and
  `--resume-from`; use the image YaRN patch and the same checkpoint/data mounts.
- `tests/local_chimera_resume_compare.py`: compare saved model and optimizer tensors.

Generate/import/export tiny weights with the **existing** Transformers tiny
exporter and Megatron-Bridge workflow; no separate conversion implementation.
Use the Megatron Chimera export wrapper for tokenizer copying, not just the
Bridge model-weight export CLI.

Local evidence directory (not a required script path):
`workspace/chimera-local-validation-20260926` contains full logs, roundtrip tensor
report, SGLang parity JSON, native actor update/resume reports, portable export,
admission and dry-run manifests. Component reports are also under
`workspace/docs/chimera-posttraining/local-validation-20260926`.
Artifacts/checkpoints are not repository content and have not been pushed.

## Remaining acceptance

Use one **2×H200 node** for real DP2 and the integrated Ray rollout→scoring→MiMo
backward→broadcast loop, TE attention graph replay, offload and full-model16K
memory qualification. Then integrated sampler/optimizer restart and short eval.
These remain open; separate component tests do not prove their composition.
Keep full-model parity acceptance explicit. See VALIDATION_PLAN.md for the bounded
allocation; no SFT, async, MOPD or extended learning run is authorized by this test.
