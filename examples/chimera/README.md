# Chimera 10B DAPO on 2xH200

This integration keeps the Slime image's dependency stack intact:

- `/root/Megatron-LM` remains Slime's patched MCore checkout.
- Transformers remains the version installed in the image. Only the external
  `transformers.models.chimera` package is added to its model search path.
- Megatron-Bridge is not imported by the Slime job.

The actor uses pure data parallelism across two GPUs: TP, PP, CP, EP, and ETP
are all one. Rollout inference is one Slime router backed by two independent
SGLang TP=1 replicas. Do not set `--sglang-dp-size`; SGLang's similarly named
DP-attention mode is not model-replica data parallelism.

## Checkpoints and data

Use the established Chimera import workflow in
`Megatron-LM/examples/chimera/RUNBOOK.md` to convert the HF checkpoint to an
MCore `torch_dist` checkpoint. The Slime job consumes the HF directory through
`--hf-checkpoint` and the converted directory through `--ref-load`; it does not
perform another conversion.

The HF checkpoint must report the trained `rms_norm_eps` value of `1e-5`.
Download DAPO-Math-17K and AIME-2024 as JSONL files with `prompt` and `label`
fields, following Slime's standard math examples.

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir ~/workspace/data/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir ~/workspace/data/aime-2024
```

## Launch

Set the host paths when they differ from the defaults and run the pinned Slime
container:

```bash
export MCORE_CHECKPOINT_HOST=/path/to/chimera_sft_torch_dist
export HF_CHECKPOINT_HOST=/path/to/chimera_sft_hf
export PROMPT_DATA_HOST=/path/to/dapo-math-17k.jsonl
export AIME_DATA_HOST=/path/to/aime-2024.jsonl
bash examples/chimera/run_container.sh
```

`RUN_MODE=smoke` is the default and requests one short DAPO/GRPO update with
Slime's alignment checks enabled. Set `RUN_MODE=train` for the longer defaults.
Batch sizes, response lengths, learning rate, SGLang memory fraction, and save
location remain overridable through the environment variables used by
`run_dapo_2xh200.sh`.
