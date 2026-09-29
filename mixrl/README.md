# MixRL

Everything you run for MixRL training is in this folder.

```
mixrl/
├── config.env        every setting you change: paths, GPUs, steps, LR, concurrency, judge, ports
├── tasks.json        which tasks train, prompts per step, eval size, and what each task is
├── judge.sh          start the judge (vLLM)
├── reward.sh         start the reward service
├── run.sh            tasks | preflight | start | resume
├── internal/launch.sh   runs inside the training container; never run by hand
└── docs/             AIRGAPPED.md (setup and transfer), RUNBOOK.md (how it works), TASK_AUDIT.md, GSM8K.md (GSM8K-only run on 8×H200)
```

## Running

From the slime checkout (first-time setup and data download: [docs/AIRGAPPED.md](docs/AIRGAPPED.md)):

```bash
mixrl/judge.sh                  # only if an enabled task needs the judge
mixrl/reward.sh                 # reward service; says if the judge is reachable
mixrl/run.sh tasks              # preview: tasks, prompts per step, eval size, judge use
mixrl/run.sh preflight          # check everything without GPUs
mixrl/run.sh start gsm8k-01     # new run named gsm8k-01 (name optional)
mixrl/run.sh resume gsm8k-01    # continue it from its latest checkpoint
```

The run name only names the output folder, `$BASE_DIR/runs/chimera/mixrl/<name>/`
(checkpoints, logs, evals, and a frozen copy of the settings and tasks it ran with).
`start` refuses a name that already has rollouts or checkpoints (a start that failed
before its first step can reuse its name); `resume` continues one with the same settings.

## Choosing what to train

In `tasks.json`, per task:

| Field | Meaning |
|---|---|
| `enabled` | `true` / `false` |
| `prompts_per_step` | prompts this task adds to each step, from `about.train_pool`; the batch is the sum |
| `eval_prompts` | `"all"` (default) or a number up to `about.val_pool` |
| `max_response_tokens` | response cap; prompt + response must fit in 16,384 tokens |

`about` describes the task (summary, grading, answer format, requirements, judge use,
pool sizes); launch checks it against the data and the reward service, so don't edit it.

## Settings

Edit `config.env`, or override for one command:

```bash
LR=2e-6 NUM_ROLLOUT=200 mixrl/run.sh start run-b
```

Defaults are set for 2 training H200s (`TRAIN_GPUS=0,1`) and a judge on GPU 2.
Training won't start, and says why, when a task needs a judge that is down, the reward
service can't grade a task, the data doesn't match `tasks.json`, or GPUs and paths
don't line up.

### Recipe settings (MiMo-V2.6 public recipes)

| Setting | Default | What it does |
|---|---|---|
| `MIXRL_TRUNCATION` | `zero` | A response with no finished answer (cut off at its cap, or `<think>` never closed) scores 0 and counts in its group like any wrong answer. `mask` leaves it out of the loss instead. Capped responses are never sent to the reward service. |
| `MIXRL_LENGTH_PENALTY` | `1` | MiMo's group-relative length penalty: in groups where most answers pass, a correct answer more than 30% longer than the median correct one loses up to 0.1 (full at twice the median). Advantages only; logged scores stay raw. |
| `MIXRL_REFILL_ROUNDS` | `2` | Replace groups without reward spread by new prompts of the same task, up to (1 + N) x `prompts_per_step` per task per step, then continue with what there is. `0` = off. |
| `ROLLOUT_TEMPERATURE`, `ROLLOUT_TOP_P`, `ROLLOUT_TOP_K` | `1.0`, `0.95`, `20` | Rollout and eval sampling (MiMo's code recipe; top-k 20 is also Qwen3's default). Top-p < 1 replays each token's candidate set in the loss (MiMo); top-k caps that set at 20 ids and needs top-p < 1. |
| `LR_WARMUP_STEPS` | `10` | Linear LR warmup, then constant (DAPO's recipe at LR 1e-6). Adam starts without optimizer state; the first step only initializes its moments (LR 0). |

These are part of the recipe: `resume` refuses a changed value.

### Rollout log-probs (`USE_ROLLOUT_LOGPROBS`)

The MiMo loss weights each token by the ratio of two log-probs: the actor's, from the
training forward pass, and SGLang's, recorded when the token was generated. It never
uses a separately recomputed "old" log-prob.

- `0` (default): before training, the actor runs one extra forward pass over the whole
  batch to recompute log-probs. With `MIXRL_OBJECTIVE=mimo` and one optimizer step per
  batch the loss does not read them (they match the training pass), so the pass only
  costs time: up to about a quarter of the actor's compute per step.
- `1`: skip that pass. Loss, advantages and the `train_rollout_logprob_abs_diff`
  metric are unchanged; routing replay then applies the recorded expert routes in the
  training forward and backward only.
- With `MIXRL_OBJECTIVE=dapo` keep `0`: there the PPO ratio's old log-prob is the
  recomputed one, and `1` would fold SGLang/Megatron numeric differences into the
  clipped ratio.

Status: `0` until confirmed on the H200s. To confirm, start a new run with
`USE_ROLLOUT_LOGPROBS=1` and check that the first steps finish (no "R3 routing replay
was not consumed exactly once") and that `MIXRL_TRAIN` loss and
`train_rollout_logprob_abs_diff` look like a `0` run's. It is part of the recipe, so
`resume` refuses a changed value.

## Watching a run

All logs for a run are in `$BASE_DIR/runs/chimera/mixrl/<name>/logs/`:

```
train.log            everything the run printed: launcher, Ray, Megatron, SGLang and the MIXRL_* lines
reward_service.log   the reward service's output during the run
judge.log            the judge's output during the run (Docker judge; a native judge logs in its terminal)
gpu_metrics.csv      GPU utilization, memory and power every 5 s
ray_logs-*.tar.gz    Ray's internal logs (raylet, GCS, workers), saved when the run exits
```

Resume and retried starts append to the same files. `train.log` is also what the
terminal shows; its MIXRL_* lines are one JSON object each:

| Line | When | Contents |
|---|---|---|
| `MIXRL_COLLECTION` | each step | per task: groups `attempted`, informative (`accepted`, `acceptance_rate`), `all_correct`, `all_wrong`, `constant`; refill: `refilled` (replacement prompts), `padding` (constant groups used to fill the batch); `raw_reward_mean`, `cap_rate`; `length_penalized`, `length_penalty_mean`; `top_p_set_mean`; pool `pass`, `deferred`; `think_rate` |
| `MIXRL_GROUP` | each prompt group | decision (`accepted`, `constant`, or `replaced` by a refill), its responses' rewards, lengths and cap flags |
| `MIXRL_TRAIN` | each optimizer step | loss, `grad_norm`, `lr-pg_*`, entropy, importance ratio and clip fractions per direction, `train_rollout_logprob_abs_diff` (`active/*`: per contributing token) |
| `MIXRL_ROUTER` | each optimizer step | expert load per MoE layer, whole batch over all ranks (MiMo section 5.4): `cv`, `peak` (max/mean), `cold` (share of experts under 0.1x mean), and token counts |
| `MIXRL_CONSISTENCY` | each optimizer step | rollout (SGLang) vs training (Megatron) agreement: per-token KL `kl_k3`, mean/max \|log-prob\| and \|prob\| gap, tokens over 0.1 / 1; `routes_overridden_by_replay` = share of expert choices Megatron would have made differently (replay uses the rollout's). Healthy: KL flat around 1e-4 |
| `MIXRL_WORST_TOKENS` | each optimizer step | the eight largest gaps with position, token id and both log-probs |
| `MIXRL_SKIP` | step with no reward spread | no group was informative; optimizer and LR schedule untouched |
| `MIXRL_EVAL` | each eval | per task and domain: `mean_score`, `pass@k`, `cap_rate`, `incomplete_rate`, `think_rate`; `equal_domain_mean` |
| `MIXRL_TIMING` | each step | generation and grading times, tokens per second |
| `MIXRL_PASS` / `MIXRL_FAILURE` / `MIXRL_STOP` | when they happen | a task reshuffled its pool / a rollout failed (with the error) / a clean stop |

Slime also prints `rollout N: {...}`, `eval N: {...}` and `perf N: {...}` summaries;
TensorBoard (`<run>/tensorboard`) has all of the above as curves. Eval results are in
`rollouts/eval-*/evaluation.json`. With the router frozen, `MIXRL_ROUTER` should stay
flat from step to step; a rising `cv`, `peak` or `cold` is the collapse MiMo saw with a
trainable router. To stop cleanly at a step boundary, set `MIXRL_WALLCLOCK_SECONDS` or
create `MIXRL_STOP_FILE`.
