# Chimera 10B DAPO on 2xH200

This integration keeps the Slime image's dependency stack intact:

The current compatibility patch is validated against
`slimerl/slime@sha256:f7f8ee9acde9645a6e88f0c703597e69a58d2892abff56071630c88f23d5068f`.
If the remote container uses a different Megatron source revision, the
launcher fails before training instead of applying an uncertain patch.

- `/root/Megatron-LM` remains Slime's patched MCore checkout. The launcher
  applies only the YaRN TE CUDA-graph fix from Chimera commit `ef2ed0b9e` to
  that checkout and refuses to run if the expected source does not match.
- Transformers remains the version installed in the image. Only the external
  `transformers.models.chimera` package is added to its model search path.
- Megatron-Bridge is not imported by the Slime job.

The Megatron actor uses EP=2 so its experts are sharded across the two H200s;
TP, PP, CP, and ETP remain one. MCore still infers a dense-parameter DP group
of two, while expert-DP is one. Rollout inference is one Slime router backed
by two independent SGLang TP=1 replicas, giving rollout replica DP=2. Do not
set `--sglang-dp-size`; SGLang's similarly named DP-attention mode is not
model-replica data parallelism.

Megatron captures its attention modules with Transformer Engine CUDA graphs,
matching the established Chimera training recipe. The runtime patch ensures
YaRN contributes a real rotary tensor during capture instead of silently
falling through to `None`.

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

Run directly inside the existing Slime container. Do not start a nested Docker
container. Set the paths visible inside that container:

```bash
export CHIMERA_TRANSFORMERS_ROOT=/workspace/repos/transformers
export HF_CHECKPOINT=/datasets/megadata/hf_models/chimera-10b
export MCORE_CHECKPOINT=/datasets/megadata/checkpoints/chimera-sft
export PROMPT_DATA=/datasets/megadata/rl/dapo-math-17k.jsonl
export AIME_DATA=/datasets/megadata/rl/aime-2024.jsonl
export SAVE_DIR=/datasets/megadata/runs/chimera-dapo
bash examples/chimera/run_dapo_2xh200.sh
```

`RUN_MODE=smoke` is the default and requests one short DAPO/GRPO update with
Slime's alignment checks enabled. Set `RUN_MODE=train` for the longer defaults.
Batch sizes, response lengths, learning rate, SGLang memory fraction, and save
location remain overridable through the environment variables used by
`run_dapo_2xh200.sh`.

## 2xH200 export and forward verification

Use a fresh `SAVE_DIR` and `RUN_MODE=verify` before training. This uses LR=0
and performs four checks against the matching HF/MCore checkpoint pair:

1. SGLang's weight checker requires every initial MCore-to-SGLang tensor to be
   transferred exactly.
2. Slime's CI alignment compares the Megatron and rollout forward log
   probabilities.
3. The established Chimera `architecture_contract.py compare-hf` command
   requires the Slime-exported HF checkpoint to preserve every key, shape,
   dtype, tensor value, and SHA256 digest.
4. The source and exported HF checkpoints must produce bitwise-identical
   logits for the same fixed input.

```bash
export RUN_MODE=verify
export CHIMERA_MEGATRON_ROOT=/workspace/repos/Megatron-LM
export SAVE_DIR=/datasets/megadata/runs/chimera-export-verification
bash examples/chimera/run_dapo_2xh200.sh
```

Reports are written to `$SAVE_DIR/export-verification/tensor-preservation.json`
and `forward-logits.json`. The normal GRPO learning rate is used only by smoke
and train modes; verify mode always forces it to zero.
