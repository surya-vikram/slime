# Multi-teacher on-policy distillation (MOPD)

Status: design, agreed 2026-10-09; not implemented yet. This file is the spec the implementation follows.

Distillation merges several domain teachers into one student. Each teacher is our own model after RL on
one domain (math, knowledge, instructions, ...), trained with MixRL from the same starting checkpoint as
the student. The student then learns all domains at once from its own answers, scored token by token by
the teacher of each prompt's domain. This is the recipe of MiMo-V2-Flash (MOPD), Kimi K3 and the MOPD
paper: domain RL makes the teachers, one distillation run merges them without the see-saw of mixed RL.

## How a step works

```
mixed batch: prompts_per_step prompts from every enabled domain
   │
   ▼
student (SGLang, current weights) writes one answer per prompt; keeps each token's log-prob
   │
   ▼  each answer goes to its domain's teacher (one teacher per answer, never an average)
teacher (SGLang, frozen) reads prompt + answer once (prefill, no generation) and returns the
log-prob of every token the student wrote
   │
   ▼
per-token advantage  A_t = clip(teacher logp - student logp, -5, 5)   (fixed numbers, no gradient)
   │
   ▼
Megatron: one forward + backward over the whole batch, loss = -mean(w_t * A_t * log pi(y_t)),
w_t = the usual SGLang/Megatron consistency mask, routing replayed from the rollout (R3); one update
   │
   ▼
new weights to SGLang, next step
```

- A token the teacher finds more likely than the student does is pushed up, a less likely one down.
  Every token gets its own signal (RL spreads one score over the whole answer), so it needs several
  times fewer samples (MOPD paper: ~25-30K per domain vs 150-180K for mixed RL).
- The student's log-probs come free from sampling (as with `USE_ROLLOUT_LOGPROBS=1`); the only extra
  work is the teacher prefill, which overlaps generation the way grading does in MixRL.
- No judge, no reward service, no answer keys, no refills: one answer per prompt.
- Every domain is in every batch, so no domain is forgotten while another is learned. A domain's
  `prompts_per_step` sets how strongly its teacher pulls on the student.

## GPU layout on 8 x H200: 6 + 2

| GPUs | role |
|---|---|
| 0-5 | student: Megatron training and 6 SGLang rollout engines, sharing the GPUs as in MixRL (EP 1) |
| 6-7 | teachers: one SGLang server per teacher, up to 3 per GPU (~20 GB each in bf16), always on, frozen |

## Inputs: one folder, read-only

```
$DISTILL_ROOT/
├── student/
│   ├── hf/                 required: HF checkpoint (rollout engines)
│   └── mcore/              required: Megatron checkpoint of the same model (training)
└── domains/
    ├── math/
    │   ├── prompts.jsonl   required: training prompts
    │   ├── eval.jsonl      required: held-out prompts (same format, never trained on)
    │   ├── domain.json     required: this domain's settings
    │   └── teacher/        required: HF checkpoint (teacher server); may be a symlink
    ├── knowledge/
    │   └── ...same four entries
    └── instruction/
        └── ...same four entries
```

Run outputs (logs, checkpoints, rollouts) go to `$BASE_DIR/runs/<run>/` as for MixRL.

**`prompts.jsonl`, `eval.jsonl`**: one prompt per line; no answers or rubrics, the teacher is the
supervision.

```json
{"id": "math-000123", "messages": [{"role": "user", "content": "..."}], "max_response_tokens": 2048}
```

`id` (unique) and `messages` (system/user/assistant turns, ending on a user turn) are required;
`max_response_tokens` is optional per row.

**`domain.json`**:

```json
{"enabled": true, "prompts_per_step": 512, "max_response_tokens": 2048, "eval_prompts": "all"}
```

| field | meaning | default |
|---|---|---|
| `enabled` | include this domain in the run | `true` |
| `prompts_per_step` | prompts from this domain in each step (one answer each): its share of the batch | required |
| `max_response_tokens` | the student's answer cap in this domain (a row can override it) | 2048 |
| `eval_prompts` | `eval.jsonl` prompts each evaluation uses: `"all"` or a number | `"all"` |

**Teachers**: any HF checkpoint of the student's architecture works in `teacher/`. Ours come from MixRL
runs on one domain's tasks, started from `student/`, converted to HF (Megatron-LM's
`examples/chimera/export.sh`, or `mixrl/run.sh export`). Two domains may share one teacher folder
(a symlink); it is served once.

## Settings: `mixrl/config.env`, distillation section

| setting | default | meaning |
|---|---|---|
| `DISTILL_ROOT` | `$BASE_DIR/distill` | the input folder above |
| `DISTILL_TRAIN_GPUS` | `0,1,2,3,4,5` | student GPUs |
| `TEACHER_GPUS` | `6,7` | teacher GPUs; teachers are packed onto them, up to 3 per GPU |
| `TEACHER_PORT` | `8100` | teacher *i* serves on `TEACHER_PORT + i` |
| `DISTILL_SAMPLES_PER_PROMPT` | `1` | answers per prompt |
| `DISTILL_ADV_CLIP` | `5` | cap on \|teacher logp - student logp\| per token |
| `DISTILL_MASK_SPECIAL_TOKENS` | `1` | no distillation signal on `<end_of_turn>` and other special tokens |

Shared with MixRL, unchanged: `NUM_ROLLOUT`, `LR` (1e-6, no warmup), `EVAL_INTERVAL`, `SAVE_INTERVAL`,
`NO_SAVE_OPTIM`, rollout sampling (temperature 1.0, top-p 0.95, top-k 20), routing replay,
`USE_ROLLOUT_LOGPROBS`. Any setting can be overridden for one command, as for MixRL.

## Running

```bash
mixrl/run.sh domains $DISTILL_ROOT          # preview: domains, prompts, teachers, batch, passes, checks
mixrl/run.sh distill $DISTILL_ROOT <run>    # start teacher servers if needed, check, train
mixrl/run.sh resume <run>                   # the root, teachers and settings are pinned in the run's manifest
```

Making teachers (ordinary MixRL runs, then a conversion):

```bash
MIXRL_TASKS_CONFIG=<a tasks file with one domain's tasks> mixrl/run.sh start teach_math_v1
mixrl/run.sh export teach_math_v1 $DISTILL_ROOT/domains/math/teacher [step]
```

## Checks before any GPU time

- `student/hf`, `student/mcore` and every enabled domain's four entries exist.
- Every teacher has the student's architecture (`config.json`), tokenizer and chat template: the
  per-token log-probs must line up.
- Prompt ids are unique within a domain and `eval.jsonl` shares no id with `prompts.jsonl`; messages
  are well formed; prompt plus `max_response_tokens` fits the 16K context.
- The batch (sum of `prompts_per_step` x `DISTILL_SAMPLES_PER_PROMPT`) divides by the student GPUs.
- A domain that would use its prompts more than 3 times over the run is warned about
  (`prompts_per_step x NUM_ROLLOUT / prompts`; there are no refills).
- Teacher servers answer. Teacher weights are hashed into the run's manifest (resume refuses a
  changed teacher) and shown on the settings line.

A teacher must come from the same starting checkpoint as the student: the MOPD paper saw training
collapse with a stronger teacher from another model family. The weights cannot prove where a teacher
came from, so this is on whoever fills `teacher/`.

## Evaluation

Every `EVAL_INTERVAL` steps, on each domain's `eval.jsonl`: the student's reverse KL to its teacher,
answer length, and the share of answers that stop before the cap. No judge or answer keys needed.

## Not in version 1

- An outcome-reward term (MiMo's `+ alpha x A_ORM`): needs answer keys and graders.
- Graded evaluation (accuracy) on the domains: needs answer keys in `eval.jsonl`.
- Top-k distillation over the teacher's 64 most likely tokens: the MOPD paper found it no better.
- A second round (the first student initializing better teachers).

## References

- MiMo-V2-Flash technical report (MOPD): https://arxiv.org/abs/2601.02780
- MOPD: Multi-Teacher On-Policy Distillation for Capability Integration: https://arxiv.org/abs/2606.30406
- Kimi K3: https://arxiv.org/abs/2607.24653
- NeMo-RL MOPD: https://docs.nvidia.com/nemo/rl/latest/about/algorithms/mopd.html
- Revisiting On-Policy Distillation, failure modes and fixes: https://arxiv.org/abs/2603.25562
- slime's single-teacher OPD, which this extends: `--use-opd`, `slime/rollout/on_policy_distillation.py`
