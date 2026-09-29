# MixRL task and eval audit (2026-09-29)

Each task was traced from its data row to its reward: source adapter
(`chimera-eval/eval_stack/sources.py`), prompt rendering and context admission
(`slime_plugins/chimera_mixrl/runtime.py`), generation, reward service, grader
(`eval_stack/graders.py`, `native_math.py`, `calendar.py`, `code_worker.py`,
`judging.py`), and group reward/eval aggregation. Evidence:

- v3/v4 splits (every row), token lengths measured with the Chimera tokenizer and
  chat template (`<start_of_turn>role ... <end_of_turn>`, no thinking support);
- the recovered 2xH200 run `sft3478-selected-mixrl-b384-main48-02` (15 steps,
  4 evals, 192 raw eval responses).

The task file's `about` blocks (`summary`, `grading`, `answer_format`, `requires`,
`judge`) carry the per-task result of this audit. This document keeps the evidence.

## Shared path (every task)

1. Row -> chat template -> prompt tokens. Admitted only if
   prompt + `max_response_tokens` + 1,024 headroom <= 16,384; no prompt truncation.
2. Generation at temperature 1, top-p 0.95 (candidate sets replayed in the loss),
   up to `max_response_tokens`.
3. Grading text = response minus the stop marker (`<end_of_turn>` or EOS). A
   response that hits the cap scores 0 without grading, in training (a wrong
   answer in its group, `MIXRL_TRUNCATION=zero`) and in eval (`cap_rate`).
4. Reward service grades from its own copy of the row (gold never travels);
   invalid or failed grading is an error, never a zero.
5. Eval: same caps and sampling, fixed rl_val panel, mean score per task, equal
   weight per prompt within a domain and per domain overall; pass@k for binary tasks.

## Eval and test size per domain

The old rl_val had 1-26 prompts per task (128 total): too few to read a domain.
v4 moves seeded rows from rl_train into rl_val (families never shared; main_test
unchanged; every old rl_val row kept), capped at 512 in total and balanced by
domain. 100 per domain would need about 900, so the preview notes it instead of
enforcing it. Each task evaluates all of its validation prompts by default.

| domain | trained tasks (rl_val per task) | rl_val | main_test benchmarks |
|---|---|---|---|
| math | gsm8k_train 29, nemotron_math 28 | 57 | 600: gsm8k 400, math500 200 |
| knowledge | mcqa 19, openqa 19, science 19 | 57 | 640: arc 160, mmlu_pro 280, triviaqa 200 |
| grounding | hotpot_train 57 | 57 | 400: hotpot validation |
| quality | cascade_chat 19, cascade_lists 19, cascade_plans 19 | 57 | 390: biggen |
| multiturn | nvidia_multichallenge 29, _advanced 28 | 57 | 400: multichallenge 200, multi_if 200 |
| instruction | nemotron_if 57 | 57 | 400: ifeval 200, ifbench 200 |
| structure | structured_train 56 | 56 | 194: structeval |
| logic | reasoning_gym 28, calendar 28 | 56 | 360: bbh (12 x 30) |
| python | apps 52 | 52 | 163: humanevalplus |

main_test also has 435 long-context prompts with no training task. Its response
budgets (8,192-16,384) are at or above every training cap, so a long correct
answer is not cut off at test time.

Eval cost: the recovered run's steps took 340-460 s for 384 samples (rollout
224-349 s, actor update ~27 s); an eval of 48 samples took 32-45 s. The launch
rule therefore compares generated tokens (prompts x samples x cap): with the
default 16-task mix and every task on "all", an eval is 506 prompts x 4 samples,
0.52 of a step; gsm8k alone at 64 prompts/step evaluates 29 prompts at 0.23 of a step.

## Per task

Token columns: prompt tokens p50/p99, rows dropped at 16K, judge prompt at full
response length (p50/max, Chimera tokenizer, approximate for the judge's own).

| task | verifier / judge | prompt p50/p99 | dropped at 16K | judge prompt at cap |
|---|---|---|---|---|
| gsm8k_train | math / none | 83/152 | 0 | - |
| nemotron_math | equivalence / always | 98/419 | 0 | 8.7K/10.2K |
| mcqa | choice / none | 285/669 | 0 | - |
| openqa | equivalence / always | 126/278 | 0 | 4.6K/5.2K |
| science | equivalence / always | 75/252 | 0 | 8.7K/9.5K |
| hotpot_train | grounded / always | 1,324/2,274 | 0 | 5.9K/8.4K |
| cascade_chat | quality / always | 49/1,524 | 0 | 8.6K/15.1K |
| cascade_lists | quality / always | 110/2,320 | 0 | 8.7K/12.9K |
| cascade_plans | quality / always | 195/2,681 | 0 | 8.8K/12.2K |
| nvidia_multichallenge | rubric / always, 2 calls | 1,630/6,164 | 1 | 8.0K/40.6K* |
| nvidia_multichallenge_advanced | rubric / always, up to 6 calls | 3,615/6,415 | 0 | 12.3K/18.9K |
| nemotron_if | instruction / none | 108/1,620 | 0 | - |
| structured_train | structure / to_pass | 1,492/6,562 | 7 | 10.4K/18.4K |
| reasoning_gym | exact / on_miss | 119/398 | 0 | 8.7K/9.0K |
| calendar | calendar / none | 2,543/4,255 | 0 | - |
| apps | apps / none | 518/1,278 | 0 | - |

\* The 40.6K case is the one prompt already dropped at 16K.

Notes by task (answer formats and requirements are also in the task file):

- **gsm8k_train**: `explicit_final_box_v2` scores 0 unless the response contains
  exactly one `\boxed{`, reasoning included. math_verify compares it with gold.
- **nemotron_math**: no output format in the data; the judge sees the last
  "Answer:" line if any, else the whole response. A multi-line final answer after
  "Answer:" reaches the judge truncated to that line.
- **mcqa**: half the rows want "Answer: X", half `\boxed{X}`; a correct answer in
  the other format used to score 0 (seen in the recovered run) and now earns 0.5.
  Last match wins; "Answer: The answer is B" extracts "T" and scores 0.
- **openqa**: boxed regex allows two nested brace levels; deeper LaTeX fails
  extraction (score 0 without judging). Rare for these text answers.
- **science**: nine per-row formats, including "last `**bold**` span"; bolding a
  heading after the answer sends the heading to the judge.
- **hotpot_train**: judge reads the whole response and the passages; length is
  not penalized.
- **cascade_***: data rubric says "Do not reward verbosity"; the judge instruction
  says not to lower a score for a longer correct explanation. Net: length neutral.
  Prompts that ask for brevity are judged on it.
- **nvidia_multichallenge(_advanced)**: every rubric check quoted the whole
  conversation, the payload added it again (roughly doubling judge tokens), and the
  check showed the judge the expected verdict ("Expected answer: YES"), anchoring it.
  Now the judge gets the criterion plus the completed conversation once; the grader
  applies the expected verdict.
  With `JUDGE_CONTEXT=16384` (the old airgapped default) 44 and 99 rows could
  overflow the judge on a long answer, which stops training; now 32,768.
- **nemotron_if**: IFBench checkers apply constraints to the whole response;
  extra text can legitimately break length/format constraints.
- **structured_train**: the whole response had to parse, so "Here is the JSON:"
  scored 0; valid data inside explanation now counts. CSV was parsed into a list
  of strings, so 1,190 of 1,332 CSV rows could never pass (object schemas,
  integer or nested fields) and 138 TOML rows asked for a top-level array TOML
  cannot express. CSV cells are now converted to schema types (640 CSV rows
  passable); the remaining 830 impossible rows, and 67 self-contradictory schemas, were
  quarantined and are removed in dataset v5. 263 rows could
  overflow a 16K judge context. Formats: JSON, YAML, TOML, CSV.
- **reasoning_gym**: seven allow-listed exact-answer generators; misses go to the
  judge with the whole response, so verbose correct answers pass when the judge is up.
- **calendar**: any prose around the JSON list used to score 0; the last JSON value
  given is now checked.
- **apps**: 122 of 1,376 rows failed the reference audit and 75 more were excluded
  as possibly non-unique output (197 total, removed in dataset v5, which keeps 1,179); stdout compared token by token,
  15 s per test, Docker sandbox required.

## Cross-cutting finding: reasoning tags from a non-thinking model (handled)

The policy is a non-thinking instruct model and its chat template has no thinking
mode, yet it writes `<think>...</think>` in 19% of recovered eval responses (a
quarter never close before the cap), most likely learned from reasoning traces in
the SFT mixture (Cascade2 math/science). This is a model defect, not a mode to
support, but it must not cost reward either. Graders read the whole text, so the
tags cost reward where the format is strict (gsm8k sees two boxes when the model
boxes inside and after the tags; calendar/structure fail to parse). Now, when an
answer follows a closed block, only the answer is graded; otherwise the tags are
removed and the rest graded. Side effect: in strict tasks, reasoning inside the
tags is exempt from format checks that plain-text reasoning still faces (a plain
response with two boxes still scores 0 on gsm8k). `think_rate` is logged per task
in training (`mixrl/<task>/think_rate`) and eval (`evaluation.json`) to watch it.

## Changes made

- Judge: one reward-service setup; `/health` reports judge reachability, which
  tasks can call the judge, and startup grading-check failures. Training refuses
  enabled tasks it cannot grade, at launch and before every rollout.
- Judge context default 32,768 (`JUDGE_CONTEXT` in `mixrl/config.env`).
- rl_val grown to 512 (v4, published on Hugging Face; seeds and targets recorded in
  the manifest's `val_growth`); domains and a single eval tier.
- Dataset v5 (2026-09-29): 1,094 rows no response can pass removed (already excluded by
  training): structured 5,946 -> 5,050 train, 57 -> 56 val; apps 1,319 -> 1,127 train,
  57 -> 52 val. rl_val 506. Science references lost leftover `**` markdown.
- `think_rate` per task in training metrics and eval summaries.
- Reasoning tags never cost reward; reasoning with no answer is unfinished and
  scored 0 like truncation (2026-09-29, following MiMo/DAPO; previously masked). mcqa partial credit (0.5) for the other explicit format.
  Structured data and calendar JSON inside explanation count (last answer given),
  CSV type conversion, 830 impossible structure rows quarantined. Multichallenge
  judge reads the completed conversation once, without the expected verdict.
- Eval chosen per task like training: `eval_prompts` "all" by default; eval size
  rules removed (the preview only reports cost).

## Proposed grader changes (not made; they change rewards)

1. Equivalence tasks: send the judge the extracted answer plus the end of the
   response, so a stray later "Answer:" line or bold heading cannot hide it.

Decided against: accepting several identical `\boxed{}` on gsm8k (the prompt asks for
exactly one; the model must follow it).
