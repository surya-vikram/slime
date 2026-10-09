# Multi-teacher on-policy distillation (MOPD)

Status: process agreed 2026-10-09 after the research report (`~/reports/Multi teacher on policy distillation
recipes.md`); implemented on branch `mopd`. This file is the spec the code follows.

Distillation merges several domain teachers into one student. Each teacher is our own model after RL on one
set of tasks, trained with MixRL from the same starting checkpoint as the student. The student then learns all
tasks at once from its own answers, scored token by token by the teacher of each prompt's task. This is
the recipe of MiMo-V2-Flash, Kimi K3, Nemotron 3 Ultra and the MOPD paper: domain RL makes the teachers, one
distillation run merges them without the see-saw of mixed RL.

The run looks like a MixRL run: same `mixrl/run.sh`, same `config.env`, same terminal view and logs
(`train.log`, `console.log`, `metrics.jsonl`, `MIXRL_*` records). Only the inputs (one config file,
`mixrl/distill.json`), the scoring (a teacher instead of the judge and reward service) and the loss differ.
Running it on the airgapped machine: [RUN.md](RUN.md).

## Agreed process

| # | decision | why |
|---|---|---|
| 1 | Sampled-token distillation: per response token, `A = clip(log q_teacher(y) - log pi_student(y), -5, 5)`, both log-probs over the full vocabulary | the universal recipe (MiMo, K3, Nemotron, MOPD paper); one float per token from the teacher |
| 2 | Temperature 1.0, pinned: the teacher scores at 1.0 and distillation refuses `ROLLOUT_TEMPERATURE` other than 1. The student samples as in MixRL: top-p 0.95, top-k 20, candidate-set replay, R3 | SGLang's prefill log-probs ignore temperature (always 1.0) |
| 3 | No Self teacher: only prompts where one of the teachers beats the student are distilled | a Self teacher only serves prompts no teacher covers; there are none |
| 4 | The end-of-turn token is trained, as in MixRL | the student learns when the teachers stop |
| 5 | Batch: each task's `prompts_per_step`, one answer each, at most 3 passes over a task's prompts in the run | published range 512-2,048 prompts, N=1 |
| 6 | LR 1e-6 with a 10-step linear warm-up (distillation only) | early large gradients are a known failure mode |
| 7 | Each task's answer cap about the teacher's p99 answer length (default: the task's MixRL cap, which the teacher trained with) | eval shows the teacher's own lengths next to the cap |
| 8 | Pure distillation, no outcome reward | only MiMo adds one, with an unpublished weight |
| 9 | Eval without grading: per task KL to the teacher, answer length next to the teacher's, stop-before-cap rate, clipped share; accuracy afterwards with chimera-eval on saved checkpoints | the judge would need GPUs the teachers use |

## How a step works

```
mixed batch: prompts_per_step prompts from every task in the config
   │
   ▼
student (SGLang, current weights) writes one answer per prompt at temperature 1, top-p 0.95, top-k 20;
keeps each token's log-prob mu and candidate set S (the tokens it could have sampled), and the expert routes
   │
   ▼  each answer goes to its task's teacher (one teacher per answer, never an average)
teacher (SGLang, frozen) reads prompt + answer once (prefill, no generation) and returns q, the full-vocabulary
log-prob at temperature 1 of every token the student wrote
   │
   ▼
Megatron: one forward + backward over the whole batch (routes replayed, R3), loss below; one optimizer step
   │
   ▼
new weights to SGLang, next step
```

## The loss

For every response token `y` (end-of-turn included), from the one training forward pass:

```
lp_full = log softmax(logits)[y]                  no gradient: the student's probability over all tokens
lp_set  = log softmax(logits restricted to S)[y]  with gradient: the distribution y was actually sampled from
A       = clip(q - lp_full, -5, 5)                how much more (or less) the teacher likes y
ratio   = exp(lp_set - mu)                        constant: Megatron vs SGLang on the same token
keep    = 0.2 <= ratio <= 5                       engines disagree badly: the token is dropped
term    = -keep * ratio * A * lp_set
loss    = mean over answers of (mean over the answer's tokens of term)
```

- `A` compares like with like: `q` is a full-vocabulary probability, so the student's must be too. With
  `lp_set` in its place every token would lose ln(mass of S) and a teacher identical to the student would
  not give zero.
- The gradient goes through `lp_set` because `y` was drawn from the candidate set; through `lp_full` it would
  also move probability into or out of the set as a whole. This is MixRL's candidate-set replay, unchanged.
- `ratio` and `keep` are MixRL's importance weight and mask (`masked_terms` in `objective.py`), unchanged;
  only the advantage differs. With one optimizer step per batch the ratio measures engine noise only.
- Pushing up `lp_set` where `A > 0` raises tokens the teacher is surer of, lowers the others; a teacher equal
  to the student gives `A = 0` everywhere (up to engine noise) and nothing moves.

## GPU layout on 8 x H200: 6 + 2

| GPUs | role |
|---|---|
| 0-5 | student: Megatron training and 6 SGLang rollout engines, sharing the GPUs as in MixRL (EP 1) |
| 6-7 | teachers: one SGLang server per teacher, up to 3 per GPU (~20 GB each in bf16), always on, frozen |

No judge and no reward service run during distillation.

## Inputs: one config file, full paths

`mixrl/distill.json` (or the file `DISTILL_CONFIG` names) says what to distil. Every path is a full path
(starting with `/`); nothing is copied or linked, each container mounts the paths as they are.

```json
{
  "student": {
    "hf": "/nvme_zone3/home/ekamai1/chimera/mixrl/models/zoro3/hf",
    "mcore": "/nvme_zone3/home/ekamai1/chimera/mixrl/models/zoro3/mcore"
  },
  "splits": "/nvme_zone3/home/ekamai1/chimera/mixrl/datasets/chimera-eval-data/splits",
  "teachers": {
    "four_tasks": "/full/path/to/four_tasks/hf",
    "multiturn": "/full/path/to/multiturn/hf"
  },
  "tasks": {
    "gsm8k_train": {"teacher": "four_tasks", "prompts_per_step": 150, "max_response_tokens": 1024, "eval_prompts": "all"},
    "mcqa": {"teacher": "four_tasks", "prompts_per_step": 280, "max_response_tokens": 2048, "eval_prompts": "all"},
    "nemotron_if": {"teacher": "four_tasks", "prompts_per_step": 299, "max_response_tokens": 2048, "eval_prompts": "all"},
    "calendar": {"teacher": "four_tasks", "prompts_per_step": 89, "max_response_tokens": 2048, "eval_prompts": "all"},
    "nvidia_multichallenge": {"teacher": "multiturn", "prompts_per_step": 22, "max_response_tokens": 2048, "eval_prompts": "all"}
  }
}
```

| field | what | checked before any GPU work |
|---|---|---|
| `student.hf` | student HF checkpoint: SGLang loads it; teachers are compared with it | full path; `config.json` exists |
| `student.mcore` | the same model as a Megatron checkpoint, for training | full path; `latest_checkpointed_iteration.txt` exists |
| `splits` | the data release's splits folder: training prompts from `rl_train.jsonl`, eval prompts from `rl_val.jsonl`; `main_test.jsonl` is never read | full path; both files exist (and the release's `manifest.json` next to the folder, when present, must match) |
| `teachers` | name → HF checkpoint; each teacher is one SGLang server | full paths; the student's architecture (`config.json`), tokenizer and chat template |
| `tasks` | task name (the `task` field of the splits) → its teacher and its prompts per step | prompts in `rl_train` and `rl_val`; the teacher is listed; every teacher scores a task |

Per task, `teacher` and `prompts_per_step` are required; `max_response_tokens` (default: the task's cap in
`mixrl/tasks.json`) and `eval_prompts` (`"all"` of its `rl_val` prompts, or a number) are optional. Two teachers
may point at the same folder; each is still its own server.

A teacher must be RL'd from the student's checkpoint: the MOPD paper saw training collapse with a teacher from
another model family. The weights cannot prove where a teacher came from, so this is on whoever writes the file.

## Settings: `mixrl/config.env`, distillation section

| setting | default | meaning |
|---|---|---|
| `DISTILL_CONFIG` | `mixrl/distill.json` | the config file above (relative to the slime checkout, or a full path) |
| `DISTILL_TRAIN_GPUS` | `0,1,2,3,4,5` | student GPUs |
| `TEACHER_GPUS` | `6,7` | teacher GPUs, taken in turn in the order the teachers are listed, up to 3 per GPU |
| `TEACHER_PORT` | `8100` | teacher *i* (in the order listed) listens on `TEACHER_PORT + i` |
| `TEACHER_GPU_MEMORY` | `0.85` | share of each teacher GPU its servers use, split evenly between them |
| `TEACHER_USE_DOCKER` | `1` | 1: teacher servers in `SLIME_IMAGE` containers; 0: native processes (inside a container) |
| `DISTILL_SAMPLES_PER_PROMPT` | `1` | student answers per prompt |
| `DISTILL_ADV_CLIP` | `5` | cap on \|teacher log-prob - student log-prob\| per token |
| `DISTILL_LR_WARMUP_STEPS` | `10` | linear LR warm-up steps (MixRL's `LR_WARMUP_STEPS` stays 0) |

Shared with MixRL: `NUM_ROLLOUT`, `LR`, `EVAL_INTERVAL`, `EVAL_BEFORE_TRAIN`, `SAVE_INTERVAL`, `NO_SAVE_OPTIM`,
rollout sampling, routing replay, the importance bounds, throughput settings. Ignored in distillation:
`MODEL_NAME` and `DATASET_NAME` (the config names the student and the splits), the task file's enabled flags
and quotas, judge and reward-service settings, refills, oversampling, the length penalty and the truncation
policy (a capped answer still gets its per-token signal). Any setting can be overridden for one command.

## Running

```bash
mixrl/run.sh distill-plan                # check the config and preview it: tasks, teachers, GPUs, batch, passes
mixrl/run.sh distill [RUN_NAME]          # start the teacher servers if needed, preflight, train
mixrl/run.sh resume RUN_NAME             # MixRL and distillation runs alike
mixrl/teachers.sh [status|stop]          # the teacher servers: start (default), check, stop
```

Outputs go to `$BASE_DIR/runs/chimera/distill/<run>/` with MixRL's layout: `logs/` (train.log, console.log,
metrics.jsonl, distill_plan.txt, teacher server logs), `checkpoints/`, `rollouts/` (eval answers with teacher
scores, the teachers' own eval answers), `manifests/` (a copy of the config and the resolved settings; resume
refuses a changed config, teacher or setting).

The terminal shows MixRL's three lines per step, with distillation numbers:

```
step  3/100 | step time: 6m10s | ETA: 9h55m | KL: 0.214 | loss: ... | grad norm: ... | entropy: ... | lr: ... |
              logprob diff: ... | rollout KL: ... | IS masked: 0% | clipped: 0.3% | router cv: ...
            | rollout: ... | train: ... | sync: ... | answers: 840 | resp len: 610 | capped: 2% | gen: ... tok/s
            | KL by task (resp len, capped) | gsm8k_train 0.180 (300, 0%) | mcqa 0.262 (650, 1%) | ...
eval after step 10 | KL: 0.170 | gsm8k_train KL 0.150 len 290/700 (teacher 280/690) stop 100% (teacher 100%) clipped 0.2% | ...
```

`KL` is the mean over answers of the per-token `lp_full - q` (an estimate of the reverse KL to the teacher);
it should fall. `clipped` is the share of tokens at the ±5 clip.

## Checks before any GPU time

- Every path in the config is a full path and holds what it should (table above).
- Every teacher has the student's architecture, tokenizer and chat template.
- Every task has prompts in `rl_train` and `rl_val`; prompt plus cap fits the 16K context (longer prompts are
  left out and counted in the run's manifest).
- The batch (sum of `prompts_per_step` x `DISTILL_SAMPLES_PER_PROMPT`) divides by the student GPUs.
- A task that would use its prompts more than 3 times over the run is warned about (no refills).
- `ROLLOUT_TEMPERATURE` is 1; student and teacher GPUs do not overlap.
- Every teacher server answers under its teacher's name and serves the checkpoint it was started with.

## Evaluation

Before training (`EVAL_BEFORE_TRAIN`), every `EVAL_INTERVAL` steps and at the end, on each task's `rl_val`
prompts (`eval_prompts` of them), one answer per prompt sampled as in training:

| metric | catches |
|---|---|
| KL to the teacher: per answer the mean of `lp_full - q`, averaged over answers (both from SGLang prefills at temperature 1) | the progress signal; should fall |
| answer length mean and p99, next to the teacher's own answers to the same prompts | length inflation; a cap below the teacher's p99 |
| stop-before-cap rate (answers that end their turn before the cap), next to the teacher's | stopping drift, repetition loops |
| share of tokens at the ±5 clip | tokens the teacher strongly rejects |

The teacher's own answers are generated once, at the first evaluation, and kept in `rollouts/teacher-eval.json`.
Accuracy is measured afterwards with chimera-eval on the saved checkpoints.

## Validation

1. CPU tests (`tests/test_chimera_mixrl_distill*.py`, the console and script tests).
2. Local end to end with a tiny Chimera as student and teachers: `run.sh distill` with the teachers and the
   trainer in docker; KL stays at the engine-noise floor when teacher = student.
3. A 10-step run with the real teachers on the airgapped machine ([RUN.md](RUN.md)) before the full run.

## Not in version 1

- An outcome-reward term (MiMo's `+ alpha x A_ORM`).
- Graded evaluation during the run (the judge needs the teacher GPUs).
- Top-k or full-vocabulary distillation (the MOPD paper found top-k no better).
- A Self teacher for prompts no teacher covers.
- Multi-prefix rollouts for multi-turn data (MiMo's MOPD2).
- A second round (the distilled student initializing better teachers).

## Implementation map

| file | role |
|---|---|
| `mixrl/distill.json` | the config (as pushed: zoro3, the release's splits, teacher paths to fill in) |
| `slime_plugins/chimera_mixrl/distill.py` | read and check the config, batch, teacher plan, preview (standard library) |
| `mixrl/teachers.sh` | start, check and stop the teacher servers |
| `mixrl/run.sh`, `mixrl/internal/launch.sh` | `distill-plan`, `distill`; `MIXRL_MODE=distill` in the container |
| `slime_plugins/chimera_mixrl/configure.py` | resolve the config and the teacher servers into the run's settings |
| `slime_plugins/chimera_mixrl/runtime.py` | the tasks' prompts, teacher scoring instead of grading, distillation eval |
| `slime_plugins/chimera_mixrl/objective.py` | `distill_loss`: the loss above, per-task KL and clipped share |
| `mixrl/internal/console.py` | the terminal lines above |

## References

- MiMo-V2-Flash technical report (MOPD): https://arxiv.org/abs/2601.02780
- MOPD: Multi-Teacher On-Policy Distillation for Capability Integration: https://arxiv.org/abs/2606.30406
- Kimi K3: https://arxiv.org/abs/2607.24653
- Nemotron 3 Ultra: https://arxiv.org/abs/2606.15007
- NeMo-RL MOPD: https://docs.nvidia.com/nemo/rl/latest/about/algorithms/mopd.html
- Revisiting On-Policy Distillation, failure modes and fixes: https://arxiv.org/abs/2603.25562
- slime's single-teacher OPD: `--use-opd`, `slime/rollout/on_policy_distillation.py`
