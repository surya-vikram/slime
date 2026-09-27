# Chimera MixRL

The canonical Chimera training entrypoint is synchronous MixRL. It targets the
final YaRN-on-RoPE model and defaults to eight policy GPUs with DP-only
Megatron, one TP=1 SGLang engine per GPU, expert-route replay, and the matched
FP32 LM-head path. The prompt mix, token budgets, and evaluation cadence are
configured in one block at the top of `train.sh`.

- Run `examples/chimera/train.sh` inside the pinned Slime container.
- Follow `examples/chimera/MIXRL_RUNBOOK.md` for the training contract and
  validation status; `examples/chimera/RUNBOOK.md` covers the pinned image,
  mounts, checkpoint layout, and operational checks.
- Regenerate the committed dataset with `examples/chimera/prepare_gsm8k.py`.

The separate GSM8K/DAPO control recipe remains available; it is not the Chimera
MixRL default.
