# Multi-teacher on-policy distillation (MOPD)

Status: process agreed 2026-10-09 after the research report (`~/reports/Multi teacher on policy distillation
recipes.md`); implemented on branch `mopd`. This file is the spec the code follows.

Distillation merges several domain teachers into one student. Each teacher is our own model after RL on one
domain, trained with MixRL from the same starting checkpoint as the student. The student then learns every
domain at once from its own answers, scored token by token by the teacher of each prompt's domain. This is
the recipe of MiMo-V2-Flash, Kimi K3, Nemotron 3 Ultra and the MOPD paper: domain RL makes the teachers, one
distillation run merges them without the see-saw of mixed RL.

The run looks like a MixRL run: same `mixrl/run.sh`, same `config.env`, same terminal view and logs
(`train.log`, `console.log`, `metrics.jsonl`, `MIXRL_*` records). Only the inputs, the scoring (a teacher
instead of the judge and reward service) and the loss differ.

## Agreed process

| # | decision | why |
|---|---|---|
| 1 | Sampled-token distillation: per response token, `A = clip(log q_teacher(y) - log pi_student(y), -5, 5)`, both log-probs over the full vocabulary | the universal recipe (MiMo, K3, Nemotron, MOPD paper); one float per token from the teacher |
| 2 | Temperature 1.0, pinned: the teacher scores at 1.0 and distillation refuses `ROLLOUT_TEMPERATURE` other than 1. The student samples as in MixRL: top-p 0.95, top-k 20, candidate-set replay, R3 | SGLang's prefill log-probs ignore temperature (always 1.0) |
| 3 | No Self teacher: only prompts where one of the teachers beats the student are distilled | a Self teacher only serves prompts no teacher covers; there are none |
| 4 | The end-of-turn token is trained, as in MixRL | the student learns when the teachers stop |
| 5 | Batch: each domain's `prompts_per_step`; about 1,200 prompts per step to start, one answer each | published range 512-2,048 prompts, N=1 |
| 6 | LR 1e-6 with a 10-step linear warm-up (distillation only) | early large gradients are a known failure mode |
| 7 | Each domain's answer cap about the teacher's p99 answer length | eval shows the teacher's own lengths next to the cap |
| 8 | Pure distillation, no outcome reward | only MiMo adds one, with an unpublished weight |
| 9 | Eval without grading: per domain KL to the teacher, answer length next to the teacher's, stop-before-cap rate, clipped share; accuracy afterwards with chimera-eval on saved checkpoints | the judge would need GPUs the teachers use |

## How a step works

```
mixed batch: prompts_per_step prompts from every enabled domain
   │
   ▼
student (SGLang, current weights) writes one answer per prompt at temperature 1, top-p 0.95, top-k 20;
keeps each token's log-prob mu and candidate set S (the tokens it could have sampled), and the expert routes
   │
   ▼  each answer goes to its domain's teacher (one teacher per answer, never an average)
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

## Inputs: one folder, read-only

```
$DISTILL_ROOT/
├── student/
│   ├── hf/                 required: HF checkpoint (rollout engines)
│   └── mcore/              required: Megatron checkpoint of the same model (training)
└── domains/
    ├── teacher_x/
    │   ├── prompts.jsonl   required: training prompts
    │   ├── eval.jsonl      required: held-out prompts (same format, never trained on)
    │   ├── domain.json     required: this domain's settings
    │   └── teacher/        required: HF checkpoint (teacher server); may be a symlink
    └── teacher_y/
        └── ...same four entries
```

**`prompts.jsonl`, `eval.jsonl`**: one prompt per line; no answers or rubrics, the teacher is the supervision.

```json
{"id": "x-000123", "messages": [{"role": "user", "content": "..."}], "max_response_tokens": 2048}
```

`id` (unique) and `messages` (system/user/assistant turns, ending on a user turn) are required;
`max_response_tokens` is optional per row. Only prompts where this domain's teacher beats the student belong
here (decision 3); choosing them is part of data preparation.

**`domain.json`**:

```json
{"enabled": true, "prompts_per_step": 600, "max_response_tokens": 2048, "eval_prompts": "all"}
```

| field | meaning | default |
|---|---|---|
| `enabled` | include this domain in the run | `true` |
| `prompts_per_step` | prompts from this domain in each step (one answer each): its share of the batch | required |
| `max_response_tokens` | the student's answer cap in this domain (a row can override it); about the teacher's p99 | 2048 |
| `eval_prompts` | `eval.jsonl` prompts each evaluation uses: `"all"` or a number | `"all"` |

**Teachers**: an HF checkpoint of the student's architecture, tokenizer and chat template, RL'd from
`student/` (the MOPD paper saw training collapse with a teacher from another model family; the weights
cannot prove where a teacher came from, so this is on whoever fills `teacher/`). Two domains may share one
teacher folder (a symlink); it is served once.

## Settings: `mixrl/config.env`, distillation section

| setting | default | meaning |
|---|---|---|
| `DISTILL_ROOT` | `$BASE_DIR/distill` | the input folder above |
| `DISTILL_TRAIN_GPUS` | `0,1,2,3,4,5` | student GPUs |
| `TEACHER_GPUS` | `6,7` | teacher GPUs; teachers are packed onto them by load, up to 3 per GPU |
| `TEACHER_PORT` | `8100` | teacher server *i* listens on `TEACHER_PORT + i` |
| `TEACHER_GPU_MEMORY` | `0.85` | share of each teacher GPU its servers use, split evenly between them |
| `TEACHER_USE_DOCKER` | `1` | 1: teacher servers in `SLIME_IMAGE` containers; 0: native processes (inside a container) |
| `DISTILL_SAMPLES_PER_PROMPT` | `1` | student answers per prompt |
| `DISTILL_ADV_CLIP` | `5` | cap on \|teacher log-prob - student log-prob\| per token |
| `DISTILL_LR_WARMUP_STEPS` | `10` | linear LR warm-up steps (MixRL's `LR_WARMUP_STEPS` stays 0) |

Shared with MixRL: `NUM_ROLLOUT`, `LR`, `EVAL_INTERVAL`, `EVAL_BEFORE_TRAIN`, `SAVE_INTERVAL`, `NO_SAVE_OPTIM`,
rollout sampling, routing replay, the importance bounds, throughput settings. Ignored in distillation: the task
file, judge and reward-service settings, refills, oversampling, the length penalty and the truncation policy
(a capped answer still gets its per-token signal). Any setting can be overridden for one command.

## Running

```bash
mixrl/run.sh domains                     # preview DISTILL_ROOT: domains, batch, teacher placement, passes, checks
mixrl/run.sh distill [RUN_NAME]          # start the teacher servers if needed, preflight, train
mixrl/run.sh resume RUN_NAME             # MixRL and distillation runs alike
mixrl/teachers.sh                        # (re)start the teacher servers for DISTILL_ROOT; stop: mixrl/teachers.sh stop
```

Outputs go to `$BASE_DIR/runs/chimera/distill/<run>/` with MixRL's layout: `logs/` (train.log, console.log,
metrics.jsonl, teacher server logs), `checkpoints/`, `rollouts/` (eval answers with teacher scores),
`manifests/` (resolved config with the teacher plan; resume refuses a changed folder, teacher or setting).

The terminal shows MixRL's three lines per step, with distillation numbers:

```
step  3/100 | step time: 6m10s | ETA: 9h55m | KL: 0.214 | loss: ... | grad norm: ... | entropy: ... | lr: ... |
              logprob diff: ... | rollout KL: ... | IS masked: 0% | clipped: 0.3% | router cv: ...
            | rollout: ... | train: ... | sync: ... | answers: 1200 | resp len: 840 | capped: 2% | gen: ... tok/s
            | KL by domain (resp len, capped) | teacher_x 0.180 (900, 1%) | teacher_y 0.262 (600, 3%)
eval after step 10 | KL: 0.170 | teacher_x KL 0.150 len 812/1650 (teacher 790/1580) stop 99% (teacher 100%) clipped 0.2% | ...
```

`KL` is the mean over answers of the per-token `lp_full - q` (an estimate of the reverse KL to the teacher);
it should fall. `clipped` is the share of tokens at the ±5 clip.

## Checks before any GPU time

- `student/hf`, `student/mcore` and every enabled domain's four entries exist.
- Every teacher has the student's architecture (`config.json`), tokenizer and chat template.
- Prompt ids are unique within a domain and `eval.jsonl` shares no id with `prompts.jsonl`; messages are well
  formed; prompt plus cap fits the 16K context (longer prompts are left out and counted in the manifest).
- The batch (sum of `prompts_per_step` x `DISTILL_SAMPLES_PER_PROMPT`) divides by the student GPUs.
- A domain that would use its prompts more than 3 times over the run is warned about (no refills).
- `ROLLOUT_TEMPERATURE` is 1.
- Every teacher server answers and serves the expected checkpoint under the expected name.

## Evaluation

Before training (`EVAL_BEFORE_TRAIN`), every `EVAL_INTERVAL` steps and at the end, on each domain's
`eval.jsonl` (`eval_prompts` of them), one answer per prompt sampled as in training:

| metric | catches |
|---|---|
| KL to the teacher: per answer the mean of `lp_full - q`, averaged over answers (both from SGLang prefills at temperature 1) | the progress signal; should fall |
| answer length mean and p99, next to the teacher's own answers to the same prompts | length inflation; a cap below the teacher's p99 |
| stop-before-cap rate (answers that end their turn before the cap), next to the teacher's | stopping drift, repetition loops |
| share of tokens at the ±5 clip | tokens the teacher strongly rejects |

The teacher's own answers are generated once, at the first evaluation, and kept in `rollouts/teacher-eval.json`.
Accuracy is measured afterwards with chimera-eval on the saved checkpoints.

## Validation

1. CPU tests (`tests/test_chimera_mixrl_distill*.py`, the console and launcher tests).
2. Teacher = student, on up to 4 H200s (3 student GPUs + 1 teacher GPU): two domains whose `teacher/` links to
   `student/hf` (one server). Expected: train and eval KL at the engine-noise floor (about the rollout KL),
   clipped share 0, and over 10 steps with LR on, flat eval metrics, entropy and length.
3. A short pilot with the real teachers before the full run.

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
| `slime_plugins/chimera_mixrl/distill.py` | read and check the folder, batch, teacher placement, preview and plan (standard library) |
| `mixrl/teachers.sh` | start, check and stop the teacher servers for a plan |
| `mixrl/run.sh`, `mixrl/internal/launch.sh` | the `distill` command; `MIXRL_MODE=distill` in the container |
| `slime_plugins/chimera_mixrl/configure.py` | resolve the folder, teacher plan and settings into the run's config |
| `slime_plugins/chimera_mixrl/runtime.py` | domain prompts, teacher scoring instead of grading, distillation eval |
| `slime_plugins/chimera_mixrl/objective.py` | `distill_loss`: the loss above, per-domain KL and clipped share |
| `mixrl/internal/console.py` | the terminal lines above |

## References

- MiMo-V2-Flash technical report (MOPD): https://arxiv.org/abs/2601.02780
- MOPD: Multi-Teacher On-Policy Distillation for Capability Integration: https://arxiv.org/abs/2606.30406
- Kimi K3: https://arxiv.org/abs/2607.24653
- Nemotron 3 Ultra: https://arxiv.org/abs/2606.15007
- NeMo-RL MOPD: https://docs.nvidia.com/nemo/rl/latest/about/algorithms/mopd.html
- Revisiting On-Policy Distillation, failure modes and fixes: https://arxiv.org/abs/2603.25562
- slime's single-teacher OPD: `--use-opd`, `slime/rollout/on_policy_distillation.py`
