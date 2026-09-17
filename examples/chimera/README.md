# Chimera reinforcement learning

This directory contains the production Chimera 10B DAPO/GSM8K integration.
It supports only the final YaRN-on-RoPE architecture and a colocated 8-GPU
DP-only topology.

- Run `examples/chimera/train.sh` inside the pinned Slime container.
- Follow `examples/chimera/RUNBOOK.md` for mounts, checkpoint layout, and
  operational checks.
- Regenerate the committed dataset with `examples/chimera/prepare_gsm8k.py`.
