# Distillation on the airgapped 8×H200

One run: fill in `mixrl/distill.json`, check it, train 10 steps, send the logs. Design and settings:
[README.md](README.md).

Layout: the student trains and generates on GPUs 0-5; the teachers serve on GPUs 6 and 7 (the first teacher in
the file on GPU 6, port 8100; the second on GPU 7, port 8101). No judge or reward service is used.

## 1. Transfer

On the connected machine:

```bash
git clone --depth 1 --branch mopd https://github.com/surya-vikram/slime.git slime-mopd
tar -czf slime-mopd.tar.gz slime-mopd
```

On the airgapped host, next to your MixRL checkout (which stays as it is):

```bash
BASE_DIR=/nvme_zone3/home/ekamai1/chimera/mixrl
cd $BASE_DIR/repos && tar -xzf slime-mopd.tar.gz && cd slime-mopd
```

The slime image, the transformers checkout, zoro3 and the data release are already on the machine.

## 2. Fill in `mixrl/distill.json`

Full paths only:

- `student`: zoro3's `hf` and `mcore` folders (filled in; check them).
- `splits`: the release's `splits` folder with `rl_train.jsonl`, `rl_val.jsonl`, `main_test.jsonl` (filled in;
  check it). Training uses `rl_train`, evaluation `rl_val`; `main_test` is never read.
- `teachers`: replace the two `/full/path/to/...` entries with each teacher's HF checkpoint folder. A teacher
  must be RL'd from zoro3 and exported to HF like zoro3's (Megatron-LM `examples/chimera/export.sh`).
- `tasks`: each task's teacher and prompts per step (840 in total, divisible by the 6 student GPUs).

To test with one teacher, keep only that teacher and its tasks, with a total still divisible by 6.

## 3. Check and run

```bash
mixrl/judge.sh stop           # only if a judge container holds GPUs 6-7
NUM_ROLLOUT=10 mixrl/run.sh distill-plan
NUM_ROLLOUT=10 EVAL_INTERVAL=5 NO_SAVE_OPTIM=1 mixrl/run.sh distill pilot-01
```

`distill-plan` checks every path, task and teacher (architecture, tokenizer, chat template against zoro3) and
prints the plan; it must end without an `error:` line. `distill` starts the teacher servers (a few minutes
each), runs the preflight, then trains 10 steps with evaluations before training, after step 5 and after step
10.

What a healthy run shows in the terminal:

| line | healthy |
|---|---|
| `teachers: Teachers ready (N started)` | every teacher up |
| `teachers answered the eval prompts` | once, at the first evaluation |
| `eval before step 1` | a KL per task, clipped small (under about 1%) |
| `step N/10` | KL above 0 and falling, clipped small, IS masked about 0%, lr rising from 1e-7 to 1e-6 |
| `KL by task` | every task listed |
| `eval after step 5`, `eval after step 10` | KL lower than before training; lengths moving toward the teacher's |

An error stops the run with an `ERROR ...` line pointing into `logs/train.log`; send the logs either way.

## 4. Send the logs

```bash
cd $BASE_DIR/runs/chimera/distill
tar -czf pilot-01-logs.tar.gz pilot-01/logs pilot-01/manifests pilot-01/rollouts
```

The terminal view (`logs/console.log`), the full log (`train.log`), every metric (`metrics.jsonl`), the teacher
servers' logs, the plan, the resolved config and the evaluation answers; no checkpoints.

## 5. Afterwards

```bash
mixrl/teachers.sh stop                                          # the teacher servers stay up between runs
rm -rf $BASE_DIR/runs/chimera/distill/pilot-01/checkpoints
```
