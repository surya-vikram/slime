# GSM8K-only run on 8×H200 (airgapped)

You need three transfers, two edits, and four commands. Everything is pushed; the GSM8K-only path was checked on 2×H200 but hasn't run on 8 GPUs yet.

### 1. On the connected machine

```bash
git clone --depth 1 --branch chimera https://github.com/surya-vikram/slime.git         # eee4261 or later
git clone --depth 1 --branch main    https://github.com/surya-vikram/chimera-eval.git   # 6a8b545
tar -czf mixrl-repos.tar.gz slime chimera-eval

hf download surya-vikram/chimera-eval-data --repo-type dataset \
  --revision 9f204f733a762c3766407636e1fbb4f8aa41dc9e --local-dir chimera-eval-data
tar --exclude=.cache -czf chimera-eval-data.tar.gz -C chimera-eval-data .    # ~122 MB
```

- **Transformers repo and model:** unchanged, keep the ones you have.
- **Docker images:** you only need the slime and chimera-eval images; skip the judge image.

### 2. On the airgapped host

```bash
BASE_DIR=/nvme_zone3/home/ekamai1/chimera/mixrl   # the BASE_DIR in mixrl/config.env
cd $BASE_DIR/repos && mv slime slime.old && mv chimera-eval chimera-eval.old && tar -xzf mixrl-repos.tar.gz
cd $BASE_DIR/datasets && mv chimera-eval-data chimera-eval-data-v4 && mkdir chimera-eval-data \
  && tar -xzf chimera-eval-data.tar.gz -C chimera-eval-data
wc -l chimera-eval-data/splits/*.jsonl      # 3982 main_test, 85175 rl_train, 506 rl_val
```

The new `tasks.json` requires v5. Launch checks the pool sizes of all 16 tasks, including disabled ones, so v4 data would fail.

### 3. Edit `mixrl/config.env`

```bash
TRAIN_GPUS=0,1,2,3,4,5,6,7          # no judge, so all 8 GPUs train
POLICY_GPUS=8                       # must equal the number of TRAIN_GPUS
EXPERT_MODEL_PARALLEL_SIZE=2        # keep: the only value tested (expert-DP becomes 4)
MIXRL_INFLIGHT_GROUPS=64            # throughput only, results unchanged
MIXRL_RESPONSE_CONCURRENCY=512      # 64 would be only 8 requests per SGLang engine
SGLANG_CUDA_GRAPH_MAX_BS=64
```

Leave the rest at its defaults:
- LR 1e-6 with a 10-step warmup;
- temperature 1.0, top-p 0.95, top-k 20;
- 8 responses per prompt, 16K sequence length;
- 100 steps, which is about one pass over GSM8K's 4,984 prompts at 48 per step.

### 4. Enable only GSM8K in `mixrl/tasks.json`

```bash
cd $BASE_DIR/repos/slime
python3 - <<'EOF'
import json
p = 'mixrl/tasks.json'
t = json.load(open(p))
for name, task in t['tasks'].items():
    task['enabled'] = name == 'gsm8k_train'
open(p, 'w').write(json.dumps(t, indent=2, ensure_ascii=False) + '\n')
EOF
```

GSM8K's entry is 48 prompts per step, an 8,192-token cap, and eval on all 29 validation prompts × 4 samples.

### 5. Run

```bash
cd $BASE_DIR/repos/slime
mixrl/judge.sh stop                  # only if an old judge container is holding a GPU
mixrl/run.sh tasks                   # expect "1 of 16 tasks enabled" and "judge: not needed"
mixrl/reward.sh                      # "Judge not reachable" is expected and fine;
                                     # it must NOT print "cannot grade gsm8k_train"
mixrl/run.sh preflight gsm8k-01
mixrl/run.sh start gsm8k-01
```

Don't run `mixrl/judge.sh`. The log is at `$BASE_DIR/runs/chimera/mixrl/gsm8k-01/logs/train.log`.

### What to check in the log

| Line | Healthy |
|---|---|
| `MIXRL_EVAL` (step 0) | the baseline score |
| `MIXRL_COLLECTION` | reward rising over steps; `accepted` groups well above 0 of 48; `top_p_set_mean` ≤ 20 |
| `MIXRL_CONSISTENCY` | `kl_k3` roughly 1e-4 to 1e-3; `tokens_over_1` ≈ 0 |
| `MIXRL_TRAIN` | `importance_masked_fraction` ≈ 0; LR climbing from 0 to 1e-6 over the first 10 steps |

Also note the `routes_overridden_by_replay` value on the first steps. It was 42% on the tiny test and 3–9% on bigger batches, so the 8-GPU number will tell us whether the tiny test's value was a small-batch artifact.
