# Legacy Chimera material

Kept for reference; nothing here is used by MixRL (`mixrl/`). Paths and commands in
these files refer to the layout before `mixrl/` existed (`examples/chimera/train.sh`,
`run_mixrl.sh`, `scripts/airgapped/`), and the GSM8K DAPO and SFT recipes they mention
were removed from the launcher.

- `RUNBOOK.md`, `VALIDATION_PLAN.md`, `LOCAL_VALIDATION.md`, `IMPLEMENTATION_STATUS.md`,
  `CHECKPOINT_COMPATIBILITY.md`: earlier design, validation and checkpoint notes.
- `data/`, `prepare_gsm8k.py`: GSM8K splits for the removed DAPO control recipe.
- `compare_hf_logits.py`, `validate_local.sh`: local parity and debug tools.
