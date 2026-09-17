# Chimera 10B DAPO/GSM8K runbook

This recipe uses one colocated 8xH200 node. Megatron uses pure DP=8 with its
distributed optimizer (`TP=PP=CP=EP=ETP=1`). Slime runs eight independent
SGLang TP=1 engines, so rollout inference is also DP=8. The only supported
model is the final Chimera YaRN-on-RoPE architecture.

## Inputs

- A final Chimera Hugging Face checkpoint.
- Its matching MCore `torch_dist` checkpoint, produced by the established
  Megatron-Bridge flow in `Megatron-LM/examples/chimera/RUNBOOK.md`. Only model
  weights are loaded for a fresh RL run; optimizer and RNG state start fresh.
- The sibling Chimera Transformers checkout. Megatron-Bridge is not a runtime
  dependency.

GSM8K is already committed under `examples/chimera/data/`. The periodic eval
split is a deterministic 512-row holdout from official training data; the
official 1,319-row test split is kept untouched for final evaluation.

## Container

```bash
docker pull slimerl/slime@sha256:f7f8ee9acde9645a6e88f0c703597e69a58d2892abff56071630c88f23d5068f

docker run --rm -it --gpus all --network host --ipc host \
  --shm-size 64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /absolute/path/to/repos:/workspace/repos \
  -v /absolute/path/to/megadata:/datasets/megadata \
  -w /workspace/repos/slime \
  slimerl/slime@sha256:f7f8ee9acde9645a6e88f0c703597e69a58d2892abff56071630c88f23d5068f \
  bash
```

The repositories mount must contain sibling `slime` and `transformers`
checkouts on their `chimera` branches. Do not replace `/root/Megatron-LM` in
the image: Slime depends on that patched revision. The launcher applies only
the checked-in YaRN CUDA-graph compatibility patch and verifies it before use.

## Run

Edit the small configuration block at the top of `examples/chimera/train.sh`,
or set the equivalent environment variables, then run:

```bash
RUN_NAME=gsm8k-dapo-001 \
HF_CHECKPOINT=/datasets/megadata/models/chimera-10b-hf \
MCORE_CHECKPOINT=/datasets/megadata/models/chimera-10b-mcore \
bash examples/chimera/train.sh
```

For an exact restart of a checkpoint created by this launcher, reuse the run
name and add `RESUME=1`. Resume restores the saved distributed-optimizer state;
fresh SFT imports never attempt to load optimizer state.

To evaluate a learned MCore checkpoint on the untouched official test split,
use a new run name and point the weights-only input at that checkpoint:

```bash
RUN_NAME=gsm8k-final-eval NUM_ROLLOUT=0 EVAL_INTERVAL=1 \
EVAL_DATA="$PWD/examples/chimera/data/gsm8k_test.jsonl" \
MCORE_CHECKPOINT=/datasets/megadata/runs/chimera/dapo-gsm8k/gsm8k-dapo-001/checkpoints \
bash examples/chimera/train.sh
```

Each run is self-contained:

```text
/datasets/megadata/runs/chimera/dapo-gsm8k/<run-name>/
  checkpoints/  logs/train.log  tensorboard/  manifests/  rollouts/
```

The manifest records the three source revisions, resolved paths, launcher
snapshot, and exact Slime command. Set `DUMP_DETAILS=1` only for short debug
runs because per-rollout tensor dumps grow quickly.

Monitor TensorBoard and `logs/train.log` for held-out `eval/*` accuracy,
training reward/pass rate, accepted-versus-filtered DAPO groups, gradient norm,
Megatron/SGLang log-probability mismatch, NaN/OOM, and CUDA-graph capture. The
strict DAPO filter deliberately keeps only prompt groups containing both a
correct and an incorrect answer; if the policy has zero GSM8K capability,
collection can keep sampling instead of producing a training batch.
