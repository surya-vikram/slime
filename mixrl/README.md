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
└── docs/             AIRGAPPED.md (setup and transfer), RUNBOOK.md (how it works), TASK_AUDIT.md
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
`start` refuses an existing name; `resume` continues one with the same settings.

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

## Watching a run

`$BASE_DIR/runs/chimera/mixrl/<name>/logs/train.log`: search `MIXRL_EVAL` (eval scores),
`MIXRL_COLLECTION` (per-step rewards, `think_rate`), `MIXRL_PASS` (a task reshuffled its
pool) and `MIXRL_FAILURE`. Eval results are in `rollouts/eval-*/evaluation.json`. To stop
cleanly at a step boundary, set `MIXRL_WALLCLOCK_SECONDS` or create `MIXRL_STOP_FILE`.
