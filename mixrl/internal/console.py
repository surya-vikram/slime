#!/usr/bin/env python3
"""Concise terminal view of a MixRL run.

launch.sh pipes the run's whole output through this filter after tee has written it to logs/train.log,
so train.log keeps every line. The terminal gets, Megatron-LM style (`key: value | ...`):

  startup lines from the launcher, then `ready` with the startup time
  a progress line during each rollout (every MIXRL_CONSOLE_PROGRESS_SECONDS, default 60)
  three lines per training step: training metrics with the step time and ETA, timing and rollout
  statistics, and reward per task
  evaluations, task passes, warnings (grading retries/failures, waiting for services) and errors
  (the exception line of each traceback, with its line number in train.log)

It also writes logs/console.log (what it printed) and logs/metrics.jsonl (every MIXRL_* record as one
JSON object per line, plus the trainer's timing). MIXRL_CONSOLE=full prints every line instead.
A line it cannot read is passed over; it never stops the run.
"""
import argparse
import ast
import json
import os
import re
import statistics
import sys
import time

RAY_PREFIX = re.compile(r'^\((?P<actor>[^) ]+)(?: pid=\d+)?(?:, ip=[^)]*)?\)\s?')
EXCEPTION = re.compile(r'^(?:[A-Za-z_][\w.]*\.)?[A-Z]\w*(?:Error|Exception|Exit|Interrupt|Died\w*)\b(?::.*)?$')
ERROR_TEXT = re.compile(r'^(?:error|ERROR|Error)\b[: ]|CUDA out of memory|\bKilled\b|^Job \S+ failed|ActorDiedError'
                        r'|RayTaskError|NCCL (?:error|timeout)|Watchdog caught collective operation timeout')
JOB_END = re.compile(r"^Job '\S+' (?:succeeded|failed|stopped)")
PERF = re.compile(r'\bperf (\d+): (\{.*\})\s*$')


def duration(seconds):
    if seconds is None:
        return '-'
    seconds = int(round(seconds))
    if seconds < 60:
        return f'{seconds}s'
    if seconds < 3600:
        return f'{seconds // 60}m{seconds % 60:02d}s'
    if seconds < 86400:
        return f'{seconds // 3600}h{seconds % 3600 // 60:02d}m'
    return f'{seconds // 86400}d{seconds % 86400 // 3600:02d}h'


def rate(value):
    return f'{value / 1000:.1f}k' if value >= 1000 else f'{value:.0f}'


def pct(value):
    return '-' if value is None else f'{100 * value:.0f}%'


def num(value, spec='.3f'):
    return '-' if value is None else format(value, spec)


class Console:
    def __init__(self, out, console_log=None, metrics=None, offset=0, full=False, progress_seconds=60.0,
                 clock=time.time):
        self.out, self.console_log, self.metrics = out, console_log, metrics
        self.line_number, self.full, self.progress_seconds, self.clock = offset, full, progress_seconds, clock
        self.started = clock()
        self.steps = {}           # rollout_id -> records gathered for its step lines
        self.step_seconds = []    # for the ETA
        self.last_progress, self.rates = None, []
        self.traceback = None     # train.log line where the current traceback began
        self.warned = {}          # message -> (last printed, repeats since)

    # ---- output -------------------------------------------------------------------------------------
    def write(self, line):
        if self.out is None:
            return
        try:
            self.out.write(line + '\n')
            self.out.flush()
        except OSError:  # terminal gone (SSH dropped): keep writing the files
            self.out = None

    def say(self, text):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.clock()))}] {text}'
        self.write(line)
        if self.console_log:
            self.console_log.write(line + '\n')
            self.console_log.flush()

    def warn(self, text, key=None):
        """Warnings repeat (a retry storm, a waiting loop); print each at most once a minute."""
        key = key or text
        now, (last, repeats) = self.clock(), self.warned.get(key, (None, 0))
        if last is not None and now - last < 60:
            self.warned[key] = (last, repeats + 1)
            return
        self.warned[key] = (now, 0)
        self.say(f'WARNING {text}' + (f' (and {repeats} more like it)' if repeats else ''))

    def record(self, kind, value):
        if self.metrics:
            self.metrics.write(json.dumps({'time': round(self.clock(), 3), 'kind': kind, **value}) + '\n')
            self.metrics.flush()

    # ---- input --------------------------------------------------------------------------------------
    def feed(self, raw):
        self.line_number += 1
        line = raw.rstrip('\r\n')
        if self.full:
            self.write(line)
            if self.console_log:
                self.console_log.write(line + '\n')
        try:
            self.handle(line)
        except Exception as error:  # a malformed line must never stop the run
            if not self.full:
                self.say(f'(console could not read train.log line {self.line_number}: {type(error).__name__})')

    def handle(self, line):
        match = RAY_PREFIX.match(line)
        actor = match.group('actor') if match else None
        text = line[match.end():] if match else line
        if text.startswith('MIXRL_'):
            kind, _, payload = text.partition(' ')
            value = None
            if payload.startswith('{'):
                try:
                    value = json.loads(payload)
                except ValueError:
                    value = None
            if isinstance(value, dict):
                self.record(kind[6:].lower(), value)
            if not self.full:
                self.mixrl(kind[6:], value, payload)
            return
        perf = PERF.search(text)
        if perf and actor and actor.startswith('MegatronTrain'):
            values = ast.literal_eval(perf.group(2))
            record = {'rollout_id': int(perf.group(1)), **{k.removeprefix('perf/'): v for k, v in values.items()}}
            self.record('trainer_perf', record)
            self.steps.setdefault(int(perf.group(1)), {})['perf'] = record
            return
        if self.full:
            return
        self.errors(text)

    def errors(self, text):
        if 'Traceback (most recent call last)' in text:
            self.traceback = self.line_number
            return
        if self.traceback is not None:
            if EXCEPTION.match(text.strip()) and not text.startswith((' ', '\t')):
                self.say(f'ERROR {text.strip()[:300]} | traceback: train.log line {self.traceback}')
                self.traceback = None
            elif self.line_number - self.traceback > 400:
                self.traceback = None
            return
        if JOB_END.match(text):
            self.say(text.strip())
        elif ERROR_TEXT.search(text):
            self.say(f'ERROR {text.strip()[:300]} | train.log line {self.line_number}')

    # ---- MIXRL records ------------------------------------------------------------------------------
    def mixrl(self, kind, value, payload):
        if kind == 'SAY':
            self.say(payload)
        elif value is None:
            if kind in ('STOP', 'SKIP'):
                self.say(f'{kind.lower()} {payload}')
            elif kind == 'REWARD_RETRY':
                self.warn(f'reward service request failed, retrying: {payload}', key='reward_retry')
        elif kind == 'READY':
            self.say(f'ready | startup: {duration(self.clock() - self.started)} | models loaded, weights in SGLang')
        elif kind == 'PIPELINE':
            self.progress(value)
        elif kind in ('TIMING', 'COLLECTION', 'ROUTER', 'CONSISTENCY'):
            self.steps.setdefault(value['rollout_id'], {})[kind.lower()] = value
        elif kind == 'TRAIN':
            self.steps.setdefault(value['rollout_id'], {})['train'] = value
        elif kind == 'STEP':
            self.step(value)
        elif kind == 'EVAL':
            self.evaluation(value)
        elif kind == 'PASS':
            self.say(f'task {value["task"]} | started pass {value["pass"]} over its {value["pool"]} prompts')
        elif kind == 'GRADE_FAILED':
            self.warn(f'grading failed, response dropped | task: {value.get("task")} | row: {value.get("row")} | '
                      f'{str(value.get("error", ""))[:200]}', key=f'grade_failed:{value.get("task")}')
        elif kind == 'HEALTH_WAIT':
            self.warn(f'waiting for services before the next batch: {json.dumps(value)[:300]}', key='health_wait')
        elif kind == 'FAILURE':
            self.say(f'ERROR rollout {value.get("rollout_id")} failed: {value.get("type")}: '
                     f'{str(value.get("error"))[:300]}')

    def progress(self, p):
        rid, evaluating = p.get('rollout_id', -1), p.get('phase') == 'eval'
        step = self.steps.setdefault(rid, {})
        if not evaluating:  # peak KV of the training rollout only
            peaks = step.setdefault('peaks', {})
            for service in ('sglang', 'judge'):
                load = p.get(service) or {}
                if load.get('kv_usage') is not None:
                    peaks[service] = max(peaks.get(service, 0.0), load['kv_usage'])
        # Tokens count when a response finishes, so one heartbeat's rate jumps; average since the last line.
        self.rates.append(p.get('gen_tokens_per_s', 0))
        now = self.clock()
        # Heartbeats come every MIXRL_PIPELINE_SECONDS with some jitter: allow 10% so a line is not skipped.
        if self.last_progress is not None and now - self.last_progress < 0.9 * self.progress_seconds:
            return
        self.last_progress, gen_rate, self.rates = now, statistics.mean(self.rates), []
        what = self.eval_label(rid) if evaluating else f'step {rid + 1} rollout'
        parts = [what, duration(p.get('seconds')), f'done: {p.get("done", 0)}', f'generating: {p.get("generating", 0)}',
                 f'grading: {p.get("grading", 0)}']
        if p.get('gen_queued'):
            parts.append(f'waiting to generate: {p["gen_queued"]}')
        if p.get('grade_queued'):
            parts.append(f'waiting to grade: {p["grade_queued"]}')
        parts.append(f'gen: {rate(gen_rate)} tok/s')
        for service in ('sglang', 'judge'):
            load = p.get(service)
            if load:
                waiting = f', {load["waiting"]:.0f} waiting' if load.get('waiting') else ''
                parts.append(f'{service}: {load.get("running", 0):.0f} running{waiting}, KV {pct(load.get("kv_usage"))}')
        if p.get('loop_lag_ms', 0) >= 1000:
            parts.append(f'event loop lag: {p["loop_lag_ms"] / 1000:.1f}s')
        self.say(' | '.join(parts))

    def step(self, s):
        rid, info = s['rollout_id'], self.steps.pop(s['rollout_id'], {})
        self.last_progress, self.rates = None, []
        self.step_seconds.append(s['seconds'])
        remaining = s.get('num_rollout', 0) - rid - 1
        recent = self.step_seconds[-10:]
        eta = duration(statistics.mean(recent) * remaining) if remaining > 0 else 'done'
        width = len(str(s.get('num_rollout', rid + 1)))
        head = f'step {rid + 1:>{width}}/{s.get("num_rollout", "?")}'
        routes = (info.get('collection') or {}).get('routes', {})
        responses = sum(r.get('responses', 0) for r in routes.values())
        reward = (sum(r.get('raw_reward_mean', 0) * r.get('responses', 0) for r in routes.values()) / responses
                  if responses else None)
        train = info.get('train') or {}
        router, consistency = info.get('router') or {}, info.get('consistency') or {}
        lr = next((v for k, v in train.items() if k.startswith('lr-')), None)
        if s.get('skipped'):
            self.say(f'{head} | step time: {duration(s["seconds"])} | ETA: {eta} | skipped: no informative groups, '
                     f'optimizer unchanged | reward: {num(reward)}')
        else:
            self.say(' | '.join([head, f'step time: {duration(s["seconds"])}', f'ETA: {eta}', f'reward: {num(reward)}',
                                 f'loss: {num(train.get("loss"), ".4E")}', f'grad norm: {num(train.get("grad_norm"))}',
                                 f'entropy: {num(train.get("entropy"))}', f'lr: {num(lr, ".2E")}',
                                 f'logprob diff: {num(train.get("train_rollout_logprob_abs_diff"), ".4f")}',
                                 f'rollout KL: {num(consistency.get("kl_k3"), ".2E")}',
                                 f'IS masked: {pct(train.get("importance_masked_fraction"))}',
                                 f'router cv: {num(router.get("cv_mean"), ".2f")}']))
        timing, perf, peaks = info.get('timing') or {}, info.get('perf') or {}, info.get('peaks') or {}
        quota = sum(r.get('accepted', 0) + r.get('padding', 0) for r in routes.values())
        accepted = sum(r.get('accepted', 0) for r in routes.values())
        padding = sum(r.get('padding', 0) for r in routes.values())
        capped = sum(r.get('capped', 0) for r in routes.values())
        failed = sum(r.get('grade_failed', 0) for r in routes.values())
        parts = [' ' * len(head), f'rollout: {duration(s.get("rollout"))}']
        train_part = f'train: {duration(s.get("train"))}'
        if perf.get('actor_train_tok_per_s'):
            train_part += f' ({rate(perf["actor_train_tok_per_s"])} tok/s)'
        parts += [train_part, f'sync: {duration(s.get("sync"))}']
        for phase in ('save', 'eval'):
            if s.get(phase):
                parts.append(f'{phase}: {duration(s[phase])}')
        if routes:
            parts += [f'groups: {accepted}/{quota} informative' + (f', {padding} padded' if padding else ''),
                      f'refills: {sum(r.get("refilled", 0) for r in routes.values())}', f'responses: {responses}',
                      f'resp len: {timing.get("generated_tokens", 0) / max(1, responses):.0f}',
                      f'capped: {pct(capped / max(1, responses))}',
                      f'gen: {rate(timing.get("generated_tokens_per_second", 0))} tok/s']
        if peaks:
            parts.append('peak KV: ' + ', '.join(f'{k} {pct(v)}' for k, v in peaks.items()))
        if failed:
            parts.append(f'grade failed: {failed}')
        self.say(' | '.join(parts))
        if routes:
            self.say(' | '.join([' ' * len(head) + ' reward by task (informative/quota)'] + [
                f'{task} {r.get("raw_reward_mean", 0):.2f} {r.get("accepted", 0)}/{r.get("accepted", 0) + r.get("padding", 0)}'
                for task, r in routes.items()]))

    def eval_label(self, rid):
        """An evaluation tagged rollout_id runs before that step's rollout (the baseline) or after the step."""
        return f'eval after step {rid + 1}' if 'collection' in self.steps.get(rid, {}) else f'eval before step {rid + 1}'

    def evaluation(self, e):
        domains = e.get('domains') or {}
        parts = [self.eval_label(e.get('rollout_id', 0)), f'score: {num(e.get("equal_domain_mean"))}']
        parts += [f'{d} {v.get("mean_score", 0):.3f}' for d, v in domains.items() if isinstance(v, dict)]
        if e.get('ungraded_prompts'):
            parts.append(f'ungraded prompts: {e["ungraded_prompts"]}')
        self.say(' | '.join(parts))


def settings_line(config, command, env):
    """The settings a run actually uses, from what its processes read: the resolved config (rollout workers)
    and the train command (Megatron, SGLang), not the environment they came from."""
    def flag(name, absent='-'):
        if name not in command:
            return absent
        value = command[command.index(name) + 1] if command.index(name) + 1 < len(command) else ''
        return 'on' if not value or value.startswith('--') else value
    urls = config.get('scorer_urls') or [config.get('scorer_url')]
    return ' | '.join([
        'settings', f'tasks: {config.get("tasks_file")}',
        f'rollout: temperature {config.get("rollout_temperature")}, top-p {config.get("rollout_top_p")}, '
        f'top-k {config.get("rollout_top_k")}',
        f'R3 replay: {"on" if config.get("routing_replay") else "OFF"}',
        f'refill rounds: {config.get("refill_rounds")}', f'oversample: {config.get("oversample", 0)}',
        f'in flight: {config.get("response_concurrency")} responses, {config.get("reward_concurrency")} grading, '
        f'{config.get("inflight_groups")} groups',
        f'reward services: {len(urls)}',
        f'SGLang per engine: {flag("--sglang-max-running-requests", "auto")} running, request cap '
        f'{flag("--sglang-server-concurrency")}, CUDA graphs to {flag("--sglang-cuda-graph-max-bs-decode")}, '
        f'memory {flag("--sglang-mem-fraction-static")}',
        f'distributed post: {flag("--use-distributed-post", "off")}',
        f'keep train responses: {env.get("MIXRL_KEEP_TRAIN_SAMPLES", "0")}',
        f'lr: {config.get("lr")}', f'max tokens/GPU: {config.get("max_tokens_per_gpu")}'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--log-dir')
    parser.add_argument('--offset', type=int, default=0, help='lines already in train.log before this run')
    parser.add_argument('--settings', nargs=2, metavar=('CONFIG_JSON', 'TRAIN_COMMAND_SH'),
                        help='print the settings line for a resolved config and train command, then exit')
    a = parser.parse_args()
    if a.settings:
        import shlex
        config, command = json.load(open(a.settings[0])), shlex.split(open(a.settings[1]).read())
        print(settings_line(config, command, os.environ))
        return
    full = os.environ.get('MIXRL_CONSOLE', 'concise') == 'full'
    progress = float(os.environ.get('MIXRL_CONSOLE_PROGRESS_SECONDS', '60'))
    with open(os.path.join(a.log_dir, 'console.log'), 'a') as console_log, \
            open(os.path.join(a.log_dir, 'metrics.jsonl'), 'a') as metrics:
        console = Console(sys.stdout, console_log, metrics, a.offset, full, progress)
        for raw in sys.stdin.buffer:
            console.feed(raw.decode('utf-8', errors='replace'))


if __name__ == '__main__':
    main()
