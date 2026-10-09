# MixRL

Everything you run for MixRL training is in this folder.

```
mixrl/
├── config.env        every setting you change: paths, GPUs, steps, LR, concurrency, judge, ports
├── tasks.json        which tasks train, prompts per step, eval size, and what each task is
├── judge.sh          start the judge (vLLM)
├── reward.sh         start the reward service
├── run.sh            start | resume (starts judge and reward service as needed) | preflight | tasks | domains | distill
├── teachers.sh       start | status | stop the teacher servers for distillation
├── internal/launch.sh   runs inside the training container; never run by hand
├── mopd/             multi-teacher on-policy distillation (MOPD): process, inputs, running (README.md)
└── docs/             AIRGAPPED.md (setup and transfer), RUNBOOK.md (how it works), TASK_AUDIT.md, GSM8K.md (GSM8K-only run on 8×H200)
```

## Running

From the slime checkout (first-time setup and data download: [docs/AIRGAPPED.md](docs/AIRGAPPED.md)):

```bash
mixrl/run.sh start gsm8k-01     # new run named gsm8k-01 (name optional): see below
mixrl/run.sh resume gsm8k-01    # continue it from its latest checkpoint (no preflight)
mixrl/run.sh tasks              # preview only: tasks, prompts per step, eval size, judge use
mixrl/run.sh preflight          # checks only, no GPUs (start runs them too)
mixrl/run.sh domains            # distillation: preview and check DISTILL_ROOT (mopd/README.md)
mixrl/run.sh distill d-01       # distillation run: teacher servers instead of judge and reward service
```

`start` does every step in order, one terminal line each, and stops at the first that fails:
start the judge if an enabled task needs it and it is not already up, start the reward
service if it is not up, show the task summary, run the preflight, then train. Services
that are already running are reused; after changing their settings restart one with
`mixrl/judge.sh` or `mixrl/reward.sh` (`stop` stops it). Their startup output goes to the
run's `logs/` folder (`judge_start.log`, `reward_start.log`, `preflight.log`, `tasks.txt`).

`resume` needs checkpoints with optimizer state, which the default `NO_SAVE_OPTIM=0` saves
(~130 GB per checkpoint; Megatron keeps every one, so delete old `checkpoints/iter_*` folders
as the run goes). `NO_SAVE_OPTIM=1` saves weights only (~19 GB), and `resume` refuses such a run.
It also needs the settings and the code the run started with: keep the repo unchanged
until the run ends. A refused resume names what differs (a code change shows up as
`launcher_hash`, `implementation_hash`, ...).

To run longer than `NUM_ROLLOUT`, resume with a larger step count:

```bash
NUM_ROLLOUT=500 MIXRL_EXTEND_CONSTANT_HORIZON=1 mixrl/run.sh resume run-a
```

Only the step count may change. The LR schedule keeps the run's original horizon (after
any warmup the LR is constant, so nothing changes), and the run records the extension in
`manifests/horizon_extension_<steps>.json`.

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

Launcher switches that have no `config.env` entry (in `mixrl/internal/launch.sh`, e.g.
`MIXRL_PIPELINE_SECONDS`) can be set the same way. A `MIXRL_`/`CHIMERA_`/`SGLANG_`/`JUDGE_`/`REWARD_`
variable that is no setting at all gets a warning, so a typo does not pass silently. Each run's
terminal shows a `settings` line with what its processes actually use (from the resolved config
and the train command).

Defaults are set for 8 H200s split 5 + 3: training and generation on `TRAIN_GPUS=0,1,2,3,4`, three judge
replicas on GPUs 5-7. The dense judge is compute-bound; with the cascade tasks off and long rubrics judged in one call,
three replicas keep grading about as fast as five GPUs generate. With 5 training GPUs, prompts per step x 16 must
divide by 5 (the launcher refuses otherwise).
The cascade quality tasks are off by default: they have no reference answers, so the judge has to work each answer
out itself (long, noisy verdicts that can reward confident wrong answers) until references are added.
The judge gets one attempt per verdict (4,096 tokens, brief reasoning). A response it cannot judge (cut off or
unreadable) is masked at once, without retries: left out of its group's statistics and the loss while the rest of
the group trains. Each task's prompts_per_step keeps it at or below 3 passes over its pool in the default 100 steps.
Training won't start, and says why, when a task needs a judge that is down, the reward
service can't grade a task, the data doesn't match `tasks.json`, or GPUs and paths
don't line up.

### Recipe settings (MiMo-V2.6 public recipes)

| Setting | Default | What it does |
|---|---|---|
| `MIXRL_TRUNCATION` | `zero` | A response with no finished answer (cut off at its cap, or `<think>` never closed) scores 0 and counts in its group like any wrong answer. `mask` leaves it out of the loss instead. Capped responses are never sent to the reward service. |
| `MIXRL_LENGTH_PENALTY` | `1` | MiMo's group-relative length penalty: in groups where most answers pass, a correct answer more than 30% longer than the median correct one loses up to 0.1 (full at twice the median). Advantages only; logged scores stay raw. |
| `MIXRL_REFILL_ROUNDS` | `1` | Replace groups without reward spread by new prompts of the same task, up to (1 + N) x `prompts_per_step` per task per step, then continue with what there is. `0` = off. |
| `ROLLOUT_TEMPERATURE`, `ROLLOUT_TOP_P`, `ROLLOUT_TOP_K` | `1.0`, `0.95`, `20` | Rollout and eval sampling (MiMo's code recipe; top-k 20 is also Qwen3's default). Top-p < 1 replays each token's candidate set in the loss (MiMo); top-k caps that set at 20 ids and needs top-p < 1. |
| `LR_WARMUP_STEPS` | `0` | Constant LR from the first step (MiMo-7B, DeepSeekMath GRPO and verl's default; at LR 1e-6 with gradient clipping a fresh Adam needs no ramp, and in 100 steps a 10-step warmup would cost a tenth of the run). `>0`: linear warmup over that many steps first (DAPO used 20). |

These are part of the recipe: `resume` refuses a changed value.

### Rollout log-probs (`USE_ROLLOUT_LOGPROBS`)

The MiMo loss weights each token by the ratio of two log-probs: the actor's, from the
training forward pass, and SGLang's, recorded when the token was generated. It never
uses a separately recomputed "old" log-prob.

- `0`: before training, the actor runs one extra forward pass over the whole
  batch to recompute log-probs. With `MIXRL_OBJECTIVE=mimo` and one optimizer step per
  batch the loss does not read them (they match the training pass), so the pass only
  costs time: up to about a quarter of the actor's compute per step.
- `1` (default): skip that pass. Loss, advantages and the `train_rollout_logprob_abs_diff`
  metric are unchanged; routing replay then applies the recorded expert routes in the
  training forward and backward only.
- `MIXRL_OBJECTIVE=dapo` needs `0` (launch refuses `1`): there the PPO ratio's old
  log-prob is the recomputed one, and `1` would fold SGLang/Megatron numeric
  differences into the clipped ratio.

Confirmed on 2 H200s (EP 2, R3 and top-p replay on), 3 steps each way on the same
tasks: no R3 replay errors, rollout/training KL 1.1-1.5e-4 and mean log-prob gap
0.006-0.007 with either value, no importance-ratio masking; the pass was 25-42% of the
training phase. It is part of the recipe, so `resume` refuses a changed value.

## Watching a run

All logs for a run are in `$BASE_DIR/runs/chimera/mixrl/<name>/logs/`:

```
train.log              everything the run printed: launcher, Ray, Megatron, SGLang and the MIXRL_* lines
console.log            what the terminal showed (the concise view below)
metrics.jsonl          every MIXRL_* record, one JSON object per line with its time and kind, plus the trainer's timings
reward_service.log     the reward service: its settings, a stats line a minute, every failed request
                       (reward_service-<i>.log per process with REWARD_PROCESSES > 1)
judge.log              the judge's output during the run
judge_start.log, reward_start.log, preflight.log, tasks.txt   what start did before training
gpu_metrics.csv        GPU utilization, memory and power every 5 s
ray_logs-*.tar.gz      Ray's internal logs (raylet, GCS, every worker in full), saved when the run exits
```

Each reward-service process also keeps its whole log, across runs, in its cache folder
(`$BASE_DIR/scorer_cache/.../reward_service.log`).

The terminal shows a concise view (`MIXRL_CONSOLE=full` shows every line instead), in the
Megatron-LM style of `key: value` fields:

```
[2026-10-02 10:47:48] MixRL run mix-16r (new) | model: zoro2 | training GPUs: 6 (EP 1) | SGLang engines: 6 | batch: 1008 prompts x 16 = 16128 samples | steps: 1000 | eval every 10 | save every 20
[2026-10-02 10:47:48] settings | tasks: mixrl/tasks.json | rollout: temperature 1.0, top-p 0.95, top-k 20 | R3 replay: on | refill rounds: 2 | oversample: 0.3 | in flight: 3200 responses, 1400 grading, 100000 groups | reward services: 4 | SGLang per engine: 1024 running, request cap 534, CUDA graphs to 1024, memory 0.80 | distributed post: on | keep train responses: 0 | lr: 1e-06 | max tokens/GPU: 16384
[2026-10-02 10:53:40] ready | startup: 5m52s | models loaded, weights in SGLang
[2026-10-02 10:54:40] step 1 rollout | 1m00s | done: 911 | generating: 2332 | grading: 373 | gen: 2.2k tok/s | sglang: 2047 running, 286 waiting, KV 16% | judge: 415 running, KV 6%
[2026-10-02 11:13:22] step    1/1000 | step time: 20m02s | ETA: 13d21h | reward: 0.412 | loss: -6.6485E-02 | grad norm: 0.154 | entropy: 0.655 | lr: 5.00E-08 | logprob diff: 0.0050 | rollout KL: 1.20E-04 | IS masked: 0% | router cv: 1.04
[2026-10-02 11:13:22]                 | rollout: 10m44s | train: 4m11s (12.3k tok/s) | sync: 24s | save: 52s | groups: 90/167 informative, 77 padded | refills: 240 | responses: 7248 | resp len: 264 | capped: 3.1% | gen: 3.0k tok/s | peak KV: sglang 48%, judge 11%
[2026-10-02 11:13:22]                  reward by task (informative/quota) | gsm8k_train 0.62 48/48 | nemotron_math 0.31 40/48 | ...
[2026-10-02 12:40:00] eval after step 10 | score: 0.553 | math 0.710 | knowledge 0.480 | ...
[2026-10-02 11:10:00] WARNING reward service request failed, retrying: attempt 2/12: ...
[2026-10-02 11:20:00] ERROR RuntimeError: CUDA out of memory ... | traceback: train.log line 18234
```

`step time` is the wall clock of the whole step (rollout, training, weight sync, and
save and eval when they happen) and `ETA` the mean of the last ten steps times the steps
left. Resume and retried starts append to the same files. The MIXRL_* lines in
`train.log` (and `metrics.jsonl`) are one JSON object each:

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
| `MIXRL_STEP` | each step | wall-clock seconds of the step and of its phases: `rollout`, `train`, `sync` (weights to SGLang), `save`, `eval` |
| `MIXRL_PIPELINE` | every 30 s of a rollout | responses waiting to generate, generating, waiting to grade, grading, done; tokens/s; SGLang and judge running requests and KV use |
| `MIXRL_READY` | once | startup finished (models loaded, first weights in SGLang) |
| `MIXRL_GRADE_FAILED` / `MIXRL_REWARD_RETRY` / `MIXRL_HEALTH_WAIT` | when they happen | a response that could not be graded, masked (`ungradable: true`: the judge could not judge it, never retried) / a reward-service request retried after a 5xx or network error / waiting for a service before a step |
| `MIXRL_PASS` / `MIXRL_FAILURE` / `MIXRL_STOP` | when they happen | a task reshuffled its pool / a rollout failed (with the error) / a clean stop |

Slime also prints `rollout N: {...}`, `eval N: {...}` and `perf N: {...}` summaries;
TensorBoard (`<run>/tensorboard`) has all of the above as curves. Eval results are in
`rollouts/eval-*/evaluation.json`. With the router frozen, `MIXRL_ROUTER` should stay
flat from step to step; a rising `cv`, `peak` or `cold` is the collapse MiMo saw with a
trainable router. To stop cleanly at a step boundary, set `MIXRL_WALLCLOCK_SECONDS` or
create `MIXRL_STOP_FILE`.
