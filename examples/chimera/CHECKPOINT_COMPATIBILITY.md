# Checkpoints from newer Megatron versions

Slime's Megatron checkpoint loader registers a metadata-only compatibility enum
before reading native MCore checkpoints. This supports newer checkpoints whose
saved args/configs reference `megatron.core.transformer.enums.InferenceCudaGraphScope`
when that class is absent from the pinned image.

The exact upstream values are preserved: `none=1`, `layer=2`, `block=3`. An enum
already supplied by Megatron is never replaced. Unknown values still fail.
This does not enable inference graphs, change training numerics, rewrite checkpoint
files, or provide general compatibility with arbitrary newer checkpoint formats.

For a fresh RL run from SFT, continue loading weights without optimizer/RNG state
using the launcher's existing finetune path. The compatibility registration also
applies to ordinary native checkpoint loads; it does not change resume semantics.

Only load trusted checkpoints: Megatron common metadata uses pickle.

## Verification

```bash
python -m pytest -q tests/test_checkpoint_metadata_compat.py
```

The regression covers a newer serialized enum failing on an older module, successful
deserialization after registration, preserved values and roundtrip, idempotence,
preservation of an existing enum, rejection of unknown values, and loader ordering.

The actual Chimera SFT iteration3478 `common.pt` was successfully loaded read-only
on the pinned Slime container after registration. This verifies the reported metadata
failure; full model loading and rollout/training remain separate integration checks.
