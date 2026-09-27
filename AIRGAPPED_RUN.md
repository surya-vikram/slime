# Chimera 10B Adam MixRL: Airgapped / Production Run Guide

This guide provides the complete, turn-key instructions for launching Chimera 10B Adam MixRL on an **airgapped / no-internet** multi-GPU machine. All paths, scripts, and container mounts are prebaked for the production filesystem root:

```
/nvme_zone3/home/ekamai1/chimera/mixrl
```

---

## 1. Architecture & Service Topology

In an airgapped environment, no external internet access or Hugging Face Hub calls are made. All inter-service communications execute over local network sockets (`127.0.0.1`):

```mermaid
flowchart TD
    subgraph S1["Step 1: Judge Server (2xH200)"]
        J["vLLM Glimmer Server (GPUs 4,5)\nhttp://127.0.0.1:8025/v1\nModel: mixrl-judge (Muse-Glimmer-30B)"]
    end

    subgraph S2["Step 2: Reward Microservice"]
        RS["chimera-eval daemon\neval_stack.reward_service\nhttp://127.0.0.1:18020\n(64 Workers | 64 Judge Concurrency)"]
    end

    subgraph S3["Step 3: MixRL Training (4xH200)"]
        ML["Slime Training Cluster (GPUs 0,1,2,3)\n- 4 Policy GPUs (Actor + SGLang Rollout)\n- 512 Prompts x 8 Responses (4,096 samples/step)\n- YaRN 32K context cap (16,384 sequence limit)\n- Adam Optimizer (LR 1e-6, WD 0)"]
    end

    ML -->|"1. Preflight check: GET /health & /admission"| RS
    ML -->|"2. Score rollout batches: POST /score"| RS
    RS -->|"Forward open-ended queries (OpenQA, Cascade)"| J
    RS -->|"Deterministic verification (Math, MCQA, Code, IFBench)"| RS
    RS -->|"Return normalized rewards [0, 1]"| ML
```

---

## 2. Required Docker Images on Host

Before executing the scripts on an airgapped system, ensure the following Docker images are already loaded in the host's Docker daemon (`docker images`):

| Image Name | Size | Role / Service | Invoking Script | Sandboxing / Access |
| :--- | :--- | :--- | :--- | :--- |
| `suryavikram6/slime:pinned` | ~25 GB | Slime MixRL Training & Dry-Run Preflight | `03_run_preflight_dryrun.sh`<br>`04_run_training.sh`<br>`05_resume_training.sh` | Runs on GPUs 0,1,2,3 with `--ipc=host --ulimit memlock=-1`. Contains pinned Megatron-LM, SGLang, and Te/CUDA-graph patches. |
| `suryavikram6/chimera-eval:0.1.1` | ~6.5 GB | Reward Microservice & Code Sandbox Worker | `02_host_reward_service.sh`<br>*(also dynamically invoked by `graders.py`)* | Runs daemon on port 18020 with 64 workers. Also used as the isolated runner image for APPS code execution (`docker run --network none --read-only`). |
| `vllm/vllm-openai:muse-glimmer` *(optional)* | ~15 GB | Glimmer Judge Server (vLLM) | `01_host_judge.sh` *(only if `USE_DOCKER=1`)* | Runs vLLM OpenAI-compatible server on GPUs 4,5 at port 8025 with `--reasoning-parser muse_glimmer`. (Not needed if running vLLM bare-metal). |

### Verification Command on Airgapped Host:
```bash
docker images --format "table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}" | grep -E "slime|chimera-eval|vllm"
```

Expected output:
```
suryavikram6/slime:pinned          <image-id>   ~25GB
suryavikram6/chimera-eval:0.1.1    <image-id>   ~6.5GB
vllm/vllm-openai:muse-glimmer      <image-id>   ~15GB  # (if using Docker for judge)
```

> [!NOTE]
> If transporting images to the airgapped node via tarballs:
> ```bash
> # On connected workstation:
> docker save suryavikram6/slime:pinned -o slime-pinned.tar
> docker save suryavikram6/chimera-eval:0.1.1 -o chimera-eval-0.1.1.tar
> docker save vllm/vllm-openai:muse-glimmer -o vllm-glimmer.tar
>
> # On airgapped host:
> docker load -i slime-pinned.tar
> docker load -i chimera-eval-0.1.1.tar
> docker load -i vllm-glimmer.tar
> ```

---

## 3. Host Filesystem Directory Structure

The entire post-training setup resides under `/nvme_zone3/home/ekamai1/chimera/mixrl`:

```
/nvme_zone3/home/ekamai1/chimera/mixrl/
├── models/
│   ├── chimera-muon-nemotron-105k-yarn32k-iter3478/
│   │   ├── hf/                             # config.json, 5 safetensors shards, tokenizer
│   │   └── mcore/                          # iter_0003478/, run_config.yaml, latest_checkpointed_iteration.txt
│   ├── Muse-Glimmer-30B/                   # Glimmer judge weights (served on 2xH200)
│   └── Muse-Glimmer-30B-assistant/         # (Optional) DFlash speculative drafter weights
├── datasets/
│   └── chimera-eval-data/
│       ├── manifest.json                   # Frozen dataset inventory
│       ├── splits/
│       │   ├── rl_train.jsonl              # 86,647 training records
│       │   └── rl_val.jsonl                # 128 quick-monitoring records
│       └── audits/
│           └── apps/                       # summary.json + 1,376 audited APPS JSON records (197 quarantined)
├── repos/
│   ├── slime/                              # Slime checkout on branch 'chimera' (scripts/airgapped/, run_mixrl.sh)
│   ├── transformers/                       # Pinned Chimera Transformers package
│   └── chimera-eval/                       # Evaluator repo (eval_stack.reward_service)
├── cache/
│   ├── scorer_cache/                       # Persistent scoring cache
│   └── vllm_cache/                         # vLLM model cache
├── runs/                                   # Training output (checkpoints, tensorboard, logs)
└── scripts/airgapped/                      # Prebaked turn-key launch scripts
    ├── 01_host_judge.sh                    # Host Glimmer judge via vLLM on GPUs 4,5
    ├── 02_host_reward_service.sh           # Host reward microservice daemon on port 18020
    ├── 03_run_preflight_dryrun.sh          # Zero-FLOP dry-run preflight in Slime container
    ├── 04_run_training.sh                  # Launch 4xH200 MixRL training on GPUs 0,1,2,3
    └── 05_resume_training.sh               # Resume interrupted training from checkpoint
```

> [!IMPORTANT]
> Ensure `/nvme_zone3/home/ekamai1/chimera/mixrl/datasets/chimera-eval-data/audits/apps` is a **real, self-contained directory** containing `summary.json` and the 1,376 audit JSON files (not an external symlink), so it resolves cleanly within container volume mounts.

---

## 4. Prebaked Execution Scripts

All executable scripts are located in [`scripts/airgapped/`](scripts/airgapped/) and have all paths hardcoded to `/nvme_zone3/home/ekamai1/chimera/mixrl`.

### Step 1: Host the Glimmer Judge via vLLM (2×H200)
Run the judge server script:

```bash
cd /nvme_zone3/home/ekamai1/chimera/mixrl/repos/slime
bash scripts/airgapped/01_host_judge.sh
```

#### What `01_host_judge.sh` Does:
* Allocates **2 dedicated GPUs** (`CUDA_VISIBLE_DEVICES=4,5`) with `--tensor-parallel-size 2`.
* Binds to `127.0.0.1:8025` serving under the registered name `--served-model-name mixrl-judge`.
* Configures Glimmer reasoning parser: `--reasoning-parser muse_glimmer`.
* Enables continuous batching concurrency matching the reward service: `--max-num-seqs 64`, `--max-num-batched-tokens 8192`, `--max-model-len 32768`.
* Automatically detects and enables **DFlash speculative decoding** if `Muse-Glimmer-30B-assistant` is present, boosting judge throughput up to ~2,500 tokens/sec.
* To run inside the official vLLM Glimmer Docker image instead of bare-metal, pass `USE_DOCKER=1`:
  ```bash
  USE_DOCKER=1 bash scripts/airgapped/01_host_judge.sh
  ```

#### Verify Judge Readiness:
```bash
curl -s http://127.0.0.1:8025/v1/models | jq .
# Expected: {"object": "list", "data": [{"id": "mixrl-judge", ...}]}
```

---

### Step 2: Start the Reward Microservice Daemon
Once the judge is live on port `8025`, start the reward microservice:

```bash
cd /nvme_zone3/home/ekamai1/chimera/mixrl/repos/slime
bash scripts/airgapped/02_host_reward_service.sh
```

#### What `02_host_reward_service.sh` Does:
* Verifies `http://127.0.0.1:8025/v1/models` responds and contains `mixrl-judge`.
* Launches `chimera-reward-service` daemon container (`suryavikram6/chimera-eval:0.1.1`) on `--net=host`.
* Overrides default entrypoint with `--entrypoint python3 -w /opt/chimera-eval` to execute `eval_stack.reward_service` directly.
* Binds `/var/run/docker.sock` for secure Docker-in-Docker APPS code execution sandboxing.
* Runs **64 parallel worker processes** (`--workers 64`) and maintains **64 concurrent judge streams** (`JUDGE_CONCURRENCY=64`), eliminating scoring bottlenecks for the 4,096-sample rollout.
* Polls `/health` until HTTP 200 is confirmed.

#### Verify Reward Service:
```bash
# Health check:
curl -s http://127.0.0.1:18020/health
# Expected output: {"protocol_id": "...", "status": "ready"}

# Admission check:
curl -s http://127.0.0.1:18020/admission | head -c 200
# Expected output: {"protocol_id": "...", "excluded_rows": {...}}
```

---

### Step 3: Run Zero-FLOP Dry-Run Verification
Before allocating GPU memory for training, run the zero-FLOP dry-run preflight:

```bash
cd /nvme_zone3/home/ekamai1/chimera/mixrl/repos/slime
bash scripts/airgapped/03_run_preflight_dryrun.sh
```

#### What `03_run_preflight_dryrun.sh` Does:
* Starts the `suryavikram6/slime:pinned` container with mounted paths.
* Verifies SHA256 checksums of all 20 GB model shards.
* Confirms YaRN 32K context parameters and the 16,384 sequence limit.
* Validates that all 16 domain training quotas sum to exactly 512 ($512 \times 8 = 4,096$ samples).
* Connects to `http://127.0.0.1:18020/health` and verifies the 197 APPS code exclusions are quarantined.
* Verifies Megatron YaRN TE CUDA-graph patch and SGLang routing capture patch are applied.
* Completes in ~15–20 seconds with **0 VRAM consumed** and exits with code 0:
  ```
  Megatron actor: dense-DP=4, expert-DP=4, TP=PP=CP=ETP=1, EP=1, distributed optimizer
  SGLang rollout: 4 independent TP=1 engines, mode=sync colocate=1
  mixrl batch: 512 prompts x 8 responses = 4096 samples
  Dry run only: command/manifests written; no Ray services or training started.
  ```

---

### Step 4: Launch Full Production Training (4×H200)
Once the dry-run passes, launch full training:

```bash
cd /nvme_zone3/home/ekamai1/chimera/mixrl/repos/slime
bash scripts/airgapped/04_run_training.sh
```

#### What `04_run_training.sh` Does:
* Launches the `suryavikram6/slime:pinned` container with dedicated access to **GPUs 0, 1, 2, 3** (`--gpus '"device=0,1,2,3"'`).
* Runs `bash run_mixrl.sh` with the Adam optimizer baseline:
  * LR: `1e-6`, Weight Decay: `0.0`, Clip Grad: `1.0`.
  * Batch: 512 prompts $\times$ 8 responses = 4,096 samples/step.
  * Matched FP32 LM-head projection (`CHIMERA_FP32_LM_HEAD=1`).
  * Route replay (`CHIMERA_ROUTING_REPLAY=1`).
  * Concurrency: `MIXRL_REWARD_CONCURRENCY=64`, `MIXRL_RESPONSE_CONCURRENCY=64`.
* All trailing CLI arguments (e.g. `NUM_ROLLOUT=200`) are passed directly through to the training script.

---

### Step 5: Resuming Interrupted Training
If training is stopped or interrupted, resume seamlessly from the latest saved checkpoint:

```bash
cd /nvme_zone3/home/ekamai1/chimera/mixrl/repos/slime
bash scripts/airgapped/05_resume_training.sh <RUN_NAME>

# Example:
bash scripts/airgapped/05_resume_training.sh chimera-mixrl-4gpus-512x8-20260927-140000
```
This reloads model weights, Adam optimizer states, route sampler cursors, and the frozen horizon scheduler without skipping or duplicating data.

---

## 5. Step 1 Monitoring & Health Checklist

During rollout 1 and optimizer step 1, verify the following telemetry in the terminal and `/nvme_zone3/home/ekamai1/chimera/mixrl/runs/$RUN_NAME/`:

1. **Frozen Router Invariants:**
   Verify terminal log confirms `router.weight` and `router.bias` maintain `param.grad is None` and zero mutation before and after the optimizer step (< 1ms execution time).
2. **Scoring Concurrency:**
   All 4,096 samples finish scoring in **< 10 seconds** across the 64 workers (deterministic domains complete in < 0.5s; judge domains pipeline in ~4–6s).
3. **Importance Weight Bounds:**
   Check logged pre-clip importance ratio quantiles (P10, P50, P90, P99); fraction clipped at `[0.2, 5.0]` is < 5%.
4. **Peak VRAM Headroom:**
   Check `/nvme_zone3/home/ekamai1/chimera/mixrl/runs/$RUN_NAME/logs/gpu_metrics.csv` to confirm peak memory stabilizes around **~82 GiB / 141 GiB** on each of the 4 training H200 GPUs.
5. **Periodic Validation:**
   In-run evaluation automatically evaluates 128 prompts from `rl_val` (pass@4) every 10 rollout boundaries.

---

## 6. Monitoring & Telemetry Artifacts

While training is active, artifacts and metrics are written to `/nvme_zone3/home/ekamai1/chimera/mixrl/runs/$RUN_NAME/`:

* **TensorBoard:**
  ```bash
  tensorboard --logdir /nvme_zone3/home/ekamai1/chimera/mixrl/runs/$RUN_NAME/tensorboard --port 6006
  ```
* **Hardware Telemetry:**
  `/nvme_zone3/home/ekamai1/chimera/mixrl/runs/$RUN_NAME/logs/gpu_metrics.csv` records GPU utilization, memory usage, and power draw every 5 seconds.
* **Evaluation Summaries:**
  Every `EVAL_INTERVAL` boundaries (default 10), quick evaluation evaluates 128 prompts from `rl_val` (pass@4). Results and domain breakdowns are saved in `/nvme_zone3/home/ekamai1/chimera/mixrl/runs/$RUN_NAME/rollouts/`.

---

## 7. Troubleshooting & Common Pitfalls

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `cli.py: error: argument command: invalid choice: 'python3'` | `chimera-eval` image has default CLI entrypoint | `02_host_reward_service.sh` automatically overrides this with `--entrypoint python3 -w /opt/chimera-eval`. |
| `FileNotFoundError: .../audits/apps/summary.json` | Dataset mount lacks the APPS audit directory or uses an unmounted host symlink | Ensure `/nvme_zone3/home/ekamai1/chimera/mixrl/datasets/chimera-eval-data/audits/apps/` is a real, self-contained directory containing `summary.json` and 1,376 JSON files. |
| `Configured judge model is not served` | vLLM served model name does not match `JUDGE_NAME` | Ensure `01_host_judge.sh` serves `--served-model-name mixrl-judge` and verify via `curl http://127.0.0.1:8025/v1/models`. |
| `Cannot connect to reward service at: http://127.0.0.1:18020` | Reward container not running or port collision | Run `bash scripts/airgapped/02_host_reward_service.sh` and verify with `curl http://127.0.0.1:18020/health`. |
