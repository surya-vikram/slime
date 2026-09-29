# Chimera model support

Files the MixRL launcher uses to run Chimera. To train, see [`mixrl/README.md`](../../mixrl/README.md).

- `runtime/sitecustomize.py`: registers the external Chimera package with Transformers and SGLang.
- `patches/`: the Megatron YaRN CUDA-graph fix and the SGLang expert-route capture patch,
  applied automatically by `mixrl/internal/launch.sh`.
- `preflight.py`: dependency and configuration checks the launcher runs before training.
- `legacy/`: older validation notes, the GSM8K DAPO control data, and debug tools. Not used by MixRL.
