"""Narrow Slime hooks. Imports GPU-stack modules only inside runtime functions."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import logging
import math
import os
import re
from pathlib import Path
import urllib.request
import urllib.error
import time

from .core import PriorityGate, RouteSampler, collect, digest, group_rewards, length_penalties, load_split, write_json
from .routes import THINK_TAG, evaluation_summary, validate_route
from .tasks import blocked
from .objective import group_length_scales
from .records import load_sample, save_sample


class QuietHttpx(logging.Filter):
    """slime's HTTP client logs every request at INFO, thousands per step (1,728 of 4,328 lines in one
    step's train.log); keep only the requests that did not get a 2xx answer."""
    SUCCESS = re.compile(r'"HTTP/[0-9.]+ 2[0-9][0-9]\b')

    def filter(self, record):
        return record.levelno > logging.INFO or not self.SUCCESS.search(record.getMessage())


logging.getLogger('httpx').addFilter(QuietHttpx())


class ServiceHTTPError(RuntimeError):
    """HTTP error status with the service's response body. urllib's HTTPError keeps an open
    socket reader, so Ray cannot pickle it and the real reason was lost ('cannot pickle
    BufferedReader instances'); this one pickles."""

    def __init__(self, url, code, body):
        super().__init__(url, code, body)
        self.url, self.code, self.body = url, code, body

    def __str__(self):
        return f'HTTP {self.code} from {self.url}: {self.body}'


def request(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read(4096).decode(errors='replace')
        exc.close()
        raise ServiceHTTPError(url, exc.code, body) from None


_config_cache = {}


def config():
    # Written once at launch and never changed during a run, but read per response and per
    # microbatch: parse it once per file version instead of on every call.
    path = os.environ['CHIMERA_MIXRL_CONFIG']
    stamp = os.stat(path).st_mtime_ns
    cached = _config_cache.get(path)
    if cached is None or cached[0] != stamp:
        cached = _config_cache[path] = (stamp, json.loads(Path(path).read_text()))
    return cached[1]


def scorer_url(c, request_id=None):
    """Reward-service base URL. With several reward-service processes (each its own cache),
    a request and every retry of it go to the same process, chosen by its request id."""
    urls = c.get('scorer_urls') or [c['scorer_url']]
    if request_id is None or len(urls) == 1:
        return urls[0]
    return urls[int(request_id[:8], 16) % len(urls)]


# A response the reward service could not grade: the judge could not judge it (HTTP 422, never
# retried: it would be asked the same question again) or every retry failed. It never gets a score:
# training masks it (left out of its group's statistics and the loss; the rest of the group trains,
# and a group with fewer than two graded responses becomes padding), eval scores its prompt on the rest.
GRADE_FAILED = 'grade_failed'


def masked(group):
    return [grade_failed(s) for s in group]

# Reward-service requests at once for tasks that never call the judge (exact checks, the APPS
# sandbox), on top of MIXRL_REWARD_CONCURRENCY: they grade in milliseconds to seconds and must not
# queue behind judge calls (in the 16-task run they waited up to ~10 min for a slot).
QUICK_GRADING_SLOTS = 512


def grade_failed(sample):
    return (sample.metadata.get('grade') or {}).get('status') == GRADE_FAILED


_scoring_pool = None


def scoring_pool(size):
    # asyncio.to_thread uses the loop's default executor: min(32, cpu_count + 4) threads,
    # which silently capped reward-service requests (and so the judge) at 32 in flight.
    global _scoring_pool
    if _scoring_pool is None or _scoring_pool._max_workers < size:
        _scoring_pool = ThreadPoolExecutor(max_workers=size, thread_name_prefix='mixrl-score')
    return _scoring_pool


_persist_pool = None


def persist_pool():
    # Response files are written off the event loop: serializing a long response with its
    # expert routes takes tens to hundreds of milliseconds and would stall every request.
    global _persist_pool
    if _persist_pool is None:
        _persist_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='mixrl-save')
    return _persist_pool


class DataSource:
    def __init__(self, args):
        from slime.utils.processing_utils import load_tokenizer
        self.args = args
        self.c = config()
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        rows, self.manifest = load_split(self.c['data_dir'], 'rl_train')
        val, _ = load_split(self.c['data_dir'], 'rl_val')
        if {r['family_id'] for r in rows} & {r['family_id'] for r in val}:
            raise ValueError('Train/validation family overlap')
        self.prompts = {}
        self.excluded = []
        quarantine = set(self.c.get('excluded_row_ids', []))
        def admitted(row):
            if row['task'] not in self.c['quotas']:
                return False
            if row['id'] in quarantine:
                reason = self.c.get('code_exclusion_reasons', {}).get(row['id'], 'scorer_metadata_quarantine')
                self.excluded.append({'id': row['id'], 'task': row['task'], 'reason': reason})
                return False
            validate_route(row, self.c['routes'])
            cap = self.c['caps'][row['task']]
            prompt = self.tokenizer.apply_chat_template(
                row['messages'], tokenize=False, add_generation_prompt=True,
                **self.c.get('chat_template_kwargs', {}))
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            headroom = self.c.get('context_headroom', 0)
            if len(tokens) + cap + headroom > self.c['context']:
                self.excluded.append({'id': row['id'], 'task': row['task'], 'prompt_tokens': len(tokens), 'cap': cap,
                                      'headroom': headroom})
                return False
            self.prompts[row['id']] = (prompt, tokens)
            return True
        admitted_rows = [r for r in rows if admitted(r)]
        self.val = [r for r in val if admitted(r)]
        selected_val = []
        for route in self.c['quotas']:
            pool = [r for r in self.val if r['task'] == route]
            # Context admission or quarantine can leave fewer rows than the task file's val_pool.
            count = min(self.c['eval_quotas'][route], len(pool))
            pool.sort(key=lambda r: digest([self.c['seed'], 'eval', r['id']]))
            selected_val.extend(pool[:count])
        self.val = selected_val
        if any(not any(r['task'] == t for r in self.val) for t in self.c['quotas']):
            raise ValueError('An enabled route has no context-admitted rl_val rows')
        self.sampler = RouteSampler(admitted_rows, self.c['quotas'], self.c['seed'])
        self.identity = digest({'config': self.c, 'manifest': self.manifest,
                                'sampler': self.sampler.fingerprint,
                                'prompt_tokens': {k: v[1] for k, v in self.prompts.items()}})
        self.run_dir = Path(self.c['run_dir'])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.run_dir / 'manifests' / 'mixrl_admission.json',
                   {'identity': self.identity, 'excluded': self.excluded,
                    'context': self.c['context'], 'caps': self.c['caps'],
                    'context_headroom': self.c.get('context_headroom', 0),
                    'train_rows': len(admitted_rows), 'val_rows': len(self.val),
                    'train_by_route': {t: sum(r['task'] == t for r in admitted_rows) for t in self.c['quotas']},
                    'val_by_route': {t: sum(r['task'] == t for r in self.val) for t in self.c['quotas']}})

    def __len__(self):
        return sum(map(len, self.sampler.pools.values()))

    def get_samples(self, num_samples):
        raise RuntimeError('MixRL uses route-aware proposals, not the default datasource iterator')

    def add_samples(self, samples):
        if samples:
            raise RuntimeError('Cross-version partial rollout replay is disabled')

    def save(self, rollout_id):
        write_json(Path(self.args.save) / 'rollout' / f'mixrl_{rollout_id}.json',
                   {'identity': self.identity, 'sampler': self.sampler.snapshot()})

    def load(self, rollout_id=None):
        # Slime calls datasource.load even for fresh --finetune HF/MCore imports.
        # Only a genuine RL restart has sampler state to restore.
        if not self.args.load or getattr(self.args, 'finetune', False):
            return
        path = Path(self.args.load) / 'rollout' / f'mixrl_{rollout_id}.json'
        if not path.exists():
            raise RuntimeError(f'Exact resume requires sampler state: {path}')
        saved = json.loads(path.read_text())
        if saved['identity'] != self.identity:
            raise ValueError('Cannot resume with changed tokenizer/data/scorer/training config')
        self.sampler.restore(saved['sampler'])

    def samples(self, row, group_id, rollout_id, split='rl_train', count=None):
        from slime.utils.types import Sample
        prompt, tokens = self.prompts[row['id']]
        count = count or self.args.n_samples_per_prompt
        return [Sample(group_index=group_id, index=group_id * count + i,
                       prompt=prompt, tokens=list(tokens),
                       metadata={'mixrl': {'row_id': row['id'], 'row_hash': digest(row),
                                           'task': row['task'], 'binary': row['binary'], 'split': split, 'sample': i,
                                           'policy_version': rollout_id, 'group_id': group_id,
                                           'cap': self.c['caps'][row['task']],
                                           'identity': self.identity}})
                for i in range(count)]


async def reward(args, sample, **kwargs):
    from slime.utils.types import Sample
    c = config()
    if sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
        raise RuntimeError('Aborted/failed generation cannot be assigned a reward')
    meta = sample.metadata['mixrl']
    if sample.status == Sample.Status.TRUNCATED:
        # Cut off at its cap: no finished answer. The reward service's fixed-budget rule
        # scores this 0 without looking at the text, so skip the request (and any judge call).
        now = time.time()
        sample.metadata.update(reward_started_at=now, reward_finished_at=now)
        sample.metadata['grade'] = {'status': 'valid', 'score': 0., 'passed': False if meta.get('binary', True) else None,
                                    'components': {'failure': 'candidate_truncated', 'scoring_policy': 'fixed_budget_v1',
                                                   'graded_by': 'mixrl_runtime'}}
        return 0.
    payload = {'request_id': digest([meta['identity'], meta['split'], meta['policy_version'],
                                     meta['group_id'], meta['row_id'], meta['sample']]),
               'protocol_id': c['scorer_protocol'], 'split': meta['split'],
               'row_id': meta['row_id'], 'row_hash': meta['row_hash'],
               'response': {'text': sample.metadata.get('grading_text', sample.response),
                            'finish_reason': 'length' if sample.status == Sample.Status.TRUNCATED else 'stop'}}
    sample.metadata['reward_started_at'] = time.time()
    # Same immutable response/identity on retry, never generate a replacement answer.
    result = error = None
    for attempt in range(c['reward_attempts']):
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                scoring_pool(c.get('reward_concurrency', 8) + QUICK_GRADING_SLOTS), request,
                scorer_url(c, payload['request_id']) + '/score', payload, c['reward_timeout'])
            break
        except ServiceHTTPError as exc:
            error = exc
            if exc.code == 422:
                break  # the judge answered but could not judge this response (cut off): retries only repeat that
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise  # 4xx: a malformed or conflicting request is a bug, not a grading fault
        except (OSError, TimeoutError) as exc:
            error = exc
        print(f'MIXRL_REWARD_RETRY attempt {attempt + 1}/{c["reward_attempts"]}: {type(error).__name__}: {error}', flush=True)
        if attempt + 1 < c['reward_attempts']:
            await asyncio.sleep(min(2 ** attempt, int(os.environ.get('MIXRL_REWARD_BACKOFF_MAX', '8'))))
    if result is None:
        reason = f'{type(error).__name__}: {error}'[:2000]
        ungradable = isinstance(error, ServiceHTTPError) and error.code == 422
        sample.metadata['reward_finished_at'] = time.time()
        sample.metadata['grade'] = {'status': GRADE_FAILED, 'error': reason, 'ungradable': ungradable}
        print('MIXRL_GRADE_FAILED ' + json.dumps({'task': meta['task'], 'split': meta['split'], 'row': meta['row_id'],
                                                 'sample': meta['sample'], 'ungradable': ungradable,
                                                 'error': reason}), flush=True)
        return 0.  # placeholder only; grade_failed() keeps it out of every loss and score
    if result['protocol_id'] != c['scorer_protocol'] or result['request_id'] != payload['request_id']:
        raise RuntimeError('Mismatched reward response identity')
    grade = result['grade']
    score = grade.get('score')
    if grade.get('status') != 'valid' or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
        raise RuntimeError('Invalid required grade')
    sample.metadata['grade'] = grade
    sample.metadata['reward_finished_at'] = time.time()
    return float(score)


def unfinished(sample):
    """No finished answer: cut off at the cap, or reasoning tags that never reached an answer.
    Both follow the truncation policy: 'zero' (default) scores them 0 in their group like any
    wrong answer; 'mask' leaves them out of the group statistics and the loss."""
    from slime.utils.types import Sample
    grade = sample.metadata.get('grade') or {}
    return sample.status == Sample.Status.TRUNCATED or grade.get('components', {}).get('incomplete') is True


def post_process_rewards(args, samples):
    from slime.utils.types import Sample
    c = config()
    raw, normalized = [], []
    n = args.n_samples_per_prompt
    if len(samples) != n * args.rollout_batch_size:
        raise ValueError('Incomplete accepted batch')
    def outcome(group, standardize=True, penalties=None):
        skip = masked(group)
        if len(group) - sum(skip) < 2:
            # Ungraded padding group: no scores, no advantages, no loss.
            return [0.] * len(group), [0.] * len(group), [False] * len(group), False
        return group_rewards([s.metadata['grade'] for s in group], [unfinished(s) for s in group],
                             c['truncation'], standardize, penalties, masked=skip)

    informative = sum(outcome(samples[start:start+n])[-1] for start in range(0, len(samples), n))
    # Native reduction divides by the fixed global response count. Compensate
    # once, before DP partitioning, to average only informative prompt groups.
    batch_scale = args.rollout_batch_size / informative if informative else 0.
    if c.get('objective', 'dapo') != 'mimo':
        # Native DAPO's per-token denominator already excludes masked tokens.
        batch_scale = 1.
    for start in range(0, len(samples), n):
        group = samples[start:start + n]
        validate_group(group, n)
        # Length penalties were computed at collection time from the same scores and lengths.
        scores, advantages, eligible, usable = outcome(
            group, args.grpo_std_normalization if c.get('objective', 'dapo') == 'dapo' else False,
            [s.metadata.get('length_penalty', 0.) for s in group])
        eligible = [keep and usable for keep in eligible]
        for sample, keep in zip(group, eligible):
            if not keep:
                sample.loss_mask = [0] * sample.response_length
        if c.get('objective', 'dapo') == 'mimo' and usable:
            lengths = [sum(s.loss_mask) if s.loss_mask is not None else s.response_length for s in group]
            scales = group_length_scales(lengths, eligible)
            for sample, advantage, scale in zip(group, advantages, scales):
                sample.metadata['base_advantage'] = advantage
                sample.metadata['group_length_scale'] = scale
            advantages = [a * scale for a, scale in zip(advantages, scales)]
        raw.extend(scores)
        normalized.extend([a * batch_scale if usable else 0. for a in advantages])
    return raw, normalized


def validate_group(group, n):
    if len(group) != n:
        raise ValueError('Incomplete group')
    metas = [s.metadata['mixrl'] for s in group]
    identities = {(m['identity'], m['policy_version'], m['group_id'], m['row_id'], m['split']) for m in metas}
    if len(identities) != 1 or [m['sample'] for m in metas] != list(range(n)):
        raise ValueError('Mixed/duplicate/out-of-order response identities')


def _load_numbers(load, out):
    # SGLang /v1/loads payloads differ across versions: collect the fields wherever they sit.
    if isinstance(load, list):
        for item in load:
            _load_numbers(item, out)
    elif isinstance(load, dict):
        for key, value in load.items():
            if key in ('num_running_reqs', 'num_waiting_reqs') and isinstance(value, (int, float)):
                out[key] = out.get(key, 0) + value
            elif key == 'token_usage' and isinstance(value, (int, float)):
                out.setdefault('token_usage', []).append(value)
            elif isinstance(value, (dict, list)):
                _load_numbers(value, out)


async def _sglang_load(args):
    from slime.utils.http_utils import get
    workers = await get(f'http://{args.sglang_router_ip}:{args.sglang_router_port}/workers')
    out = {}
    for worker in workers.get('workers', []):
        _load_numbers(await get(f"{worker['url']}/v1/loads?include=core"), out)
    usage = out.pop('token_usage', None)
    return {'running': out.get('num_running_reqs'), 'waiting': out.get('num_waiting_reqs'),
            'kv_usage': round(sum(usage) / len(usage), 3) if usage else None}


def _judge_load(url):
    # vLLM Prometheus text: sum running/waiting over engines, mean KV-cache usage.
    with urllib.request.urlopen(url, timeout=5) as response:
        text = response.read().decode()
    sums, kv = {}, []
    for line in text.splitlines():
        if line.startswith('#') or ' ' not in line:
            continue
        name, value = line.split('{', 1)[0].split(' ', 1)[0], line.rsplit(' ', 1)[1]
        try:
            number = float(value)
        except ValueError:
            continue
        if name in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
            sums[name.split('_')[-1]] = sums.get(name.split('_')[-1], 0) + number
        elif name in ('vllm:kv_cache_usage_perc', 'vllm:gpu_cache_usage_perc'):
            kv.append(number)
    return {'running': sums.get('running'), 'waiting': sums.get('waiting'),
            'kv_usage': round(sum(kv) / len(kv), 3) if kv else None}


class LoopProfiler:
    """Samples the event-loop thread's stack every 20 ms (stdlib only; py-spy needs ptrace).
    snapshot() returns the share of samples where the loop was busy and the busiest functions
    since the previous snapshot. Idle samples sit in the selector waiting for I/O."""

    def __init__(self, interval=.02):
        import collections, sys, threading
        self.target, self.interval = threading.get_ident(), interval
        self.counts, self.lock, self.stop = collections.Counter(), threading.Lock(), threading.Event()
        def run():
            while not self.stop.wait(self.interval):
                frame = sys._current_frames().get(self.target)
                if frame is None:
                    continue
                chain, f = [], frame
                while f is not None and len(chain) < 4:
                    chain.append(f'{f.f_code.co_filename.rsplit("/", 1)[-1]}:{f.f_code.co_name}')
                    f = f.f_back
                with self.lock:
                    self.counts[' < '.join(chain)] += 1
        threading.Thread(target=run, daemon=True, name='mixrl-loop-profiler').start()

    def snapshot(self, top=6):
        with self.lock:
            counts, self.counts = self.counts, type(self.counts)()
        total = sum(counts.values())
        if not total:
            return None
        idle = sum(v for k, v in counts.items() if k.startswith('selectors.py:select'))
        busy = [(k, v) for k, v in counts.most_common() if not k.startswith('selectors.py:select')][:top]
        return {'busy_share': round(1 - idle / total, 3), 'top': [[k, round(v / total, 3)] for k, v in busy]}

    def close(self):
        self.stop.set()


async def pipeline_heartbeat(args, rollout_id, phase, flow, interval):
    """Every `interval` seconds: where the responses are, how late the event loop runs, and how
    busy SGLang and the judge are. A large loop lag means this process, not the GPUs, is the
    bottleneck; few running requests with a long grade queue means grading is."""
    loop, started, tokens = asyncio.get_running_loop(), time.monotonic(), 0
    judge_url = os.environ.get('MIXRL_JUDGE_METRICS_URL', '')
    profiler = LoopProfiler() if os.environ.get('MIXRL_PROFILE', '0') == '1' else None
    try:
        await _heartbeat_loop(args, rollout_id, phase, flow, interval, loop, started, tokens, judge_url, profiler)
    finally:
        if profiler:
            profiler.close()


async def _heartbeat_loop(args, rollout_id, phase, flow, interval, loop, started, tokens, judge_url, profiler):
    while True:
        tick = loop.time()
        await asyncio.sleep(interval)
        lag = loop.time() - tick - interval
        line = {'rollout_id': rollout_id, 'phase': phase, 'seconds': round(time.monotonic() - started),
                **{k: v for k, v in flow.items() if k != 'gen_tokens'},
                'gen_tokens_per_s': round((flow['gen_tokens'] - tokens) / interval), 'loop_lag_ms': round(lag * 1000)}
        tokens = flow['gen_tokens']
        try:
            line['sglang'] = await asyncio.wait_for(_sglang_load(args), 5)
        except Exception:
            pass
        if judge_url:
            try:
                line['judge'] = await asyncio.wait_for(asyncio.to_thread(_judge_load, judge_url), 6)
            except Exception:
                pass
        if profiler:
            line['loop_profile'] = profiler.snapshot()
        print('MIXRL_PIPELINE ' + json.dumps(line), flush=True)


async def stop_generation(args):
    """Abort whatever SGLang still runs and wait until every engine is idle."""
    from slime.backends.sglang_utils.server_control import abort_servers_until_idle
    from slime.utils.http_utils import get
    response = await get(f'http://{args.sglang_router_ip}:{args.sglang_router_port}/workers')
    await abort_servers_until_idle([worker['url'] for worker in response['workers']])


async def gather_cancel(coros):
    tasks = [asyncio.create_task(c) for c in coros]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _rollout(args, rollout_id, source, evaluation=False):
    from slime.rollout.sglang_rollout import GenerateState, abort
    from slime.rollout.base_types import RolloutFnTrainOutput, RolloutFnEvalOutput
    from slime.utils.types import Sample
    c = source.c
    if args.partial_rollout or args.group_rm:
        raise ValueError('Partial/group-RM modes are not supported')
    if bool(args.use_rollout_routing_replay) != bool(c.get('routing_replay', 0)):
        raise ValueError('Routing-replay launcher/config mismatch')
    # Behaviour log-probs must describe exactly the distribution sampled from: temperature is
    # applied on both sides, and top-p < 1 replays each token's candidate set in the loss (MiMo).
    top_k = getattr(args, 'rollout_top_k', -1)
    if (args.rollout_temperature != c.get('rollout_temperature', 1.0) or args.rollout_top_p != c.get('rollout_top_p', 1.0)
            or top_k != c.get('rollout_top_k', -1) or (top_k > 0 and args.rollout_top_p == 1)):
        raise ValueError('Rollout sampling differs from the resolved MixRL config (top-k needs top-p < 1 for replay)')
    replay_top_p = args.rollout_top_p < 1 and not evaluation
    state = GenerateState(args)
    snapshot = source.sampler.snapshot()
    phase = 'eval' if evaluation else 'train'
    # weight_versions disambiguates baseline and post-update eval at rollout id 0.
    # The caller's eval sequence below additionally creates distinct durable identities.
    eval_counter_path = source.run_dir / 'rollouts' / 'eval_counter.json'
    if evaluation:
        counter = json.loads(eval_counter_path.read_text()) if eval_counter_path.exists() else 0
        version = f'{rollout_id}-eval-{counter}'
    else:
        version = rollout_id
    directory = source.run_dir / 'rollouts' / f'{phase}-{version}'
    directory.mkdir(parents=True, exist_ok=True)
    # Per-response files: always for eval (few, and the ones worth reading); for training only
    # with MIXRL_KEEP_TRAIN_SAMPLES=1. A training response with its expert routes and top-p sets
    # is MBs and tens to hundreds of ms to serialize; the batch itself is passed in memory.
    keep_files = evaluation or os.environ.get('MIXRL_KEEP_TRAIN_SAMPLES', '0') == '1'
    # Checked before every batch: the judge can go down mid-run, and that must stop
    # training before generation, never turn judge-graded answers into zeros.
    # A judge or reward service that is restarting gets MIXRL_HEALTH_WAIT_SECONDS to return.
    deadline = time.monotonic() + float(os.environ.get('MIXRL_HEALTH_WAIT_SECONDS', '0'))
    urls = c.get('scorer_urls') or [c['scorer_url']]
    while True:
        reasons, protocols = {}, set()
        for url in urls:
            try:
                health = await asyncio.to_thread(request, url + '/health')
                protocols.add(health['protocol_id'])
                reasons.update({f'{task} ({url})' if len(urls) > 1 else task: reason
                                for task, reason in blocked(c['routes'], health).items()})
            except (OSError, ServiceHTTPError, ValueError, KeyError) as exc:
                reasons[f'reward service {url}'] = f'{type(exc).__name__}: {exc}'
        if not reasons or time.monotonic() >= deadline:
            break
        print('MIXRL_HEALTH_WAIT ' + json.dumps(reasons), flush=True)
        await asyncio.sleep(30)
    if reasons:
        raise RuntimeError('Reward service cannot grade enabled tasks: '
                           + '; '.join(f'{task}: {reason}' for task, reason in reasons.items()))
    if protocols != {c['scorer_protocol']}:
        raise RuntimeError('Scorer protocol mismatch before rollout')
    # Earlier-proposed groups are generated and graded first (lower priority value), so groups
    # finish one after another and keep-or-refill decisions arrive throughout the step.
    slots = PriorityGate(c['response_concurrency'])
    scoring_slots = PriorityGate(c.get('reward_concurrency', 8))
    quick_slots = PriorityGate(QUICK_GRADING_SLOTS)
    judge_free = {task for task, route in c['routes'].items() if route.get('judge') == 'none'}
    priority = lambda sample: sample.metadata.get('mixrl_priority', 0)
    # Where every response is right now, for the MIXRL_PIPELINE line.
    flow = {'gen_queued': 0, 'generating': 0, 'grade_queued': 0, 'grading': 0, 'done': 0, 'gen_tokens': 0}

    async def score(sample):
        # Slime retains the stop marker in decoded response text for token/logprob
        # alignment, and the tokens keep it so the policy learns to end its turn.
        # Graders consume ordinary assistant content, not serialization markers:
        # Chimera ends a turn with the <end_of_turn> stop string, other models with EOS.
        text = sample.response
        eos = getattr(source.tokenizer, 'eos_token', None)
        eos_id = getattr(source.tokenizer, 'eos_token_id', None)
        if sample.status == Sample.Status.COMPLETED:
            stop = next((s for s in getattr(args, 'rollout_stop', None) or [] if s and text.endswith(s)), None)
            if stop:
                text = text[:-len(stop)]
            elif eos and sample.tokens and sample.tokens[-1] == eos_id and text.endswith(eos):
                text = text[:-len(eos)]
        sample.metadata['grading_text'] = text
        sample.metadata['reward_queued_at'] = time.time()
        gate = quick_slots if sample.metadata['mixrl']['task'] in judge_free else scoring_slots
        async with gate.slot(priority(sample), waiting=(flow, 'grade_queued')):
            flow['grading'] += 1
            try:
                return await reward(args, sample)
            finally:
                flow['grading'] -= 1

    async def generate_one(sample):
        meta = sample.metadata['mixrl']
        path = directory / f'{sample.index}.json'
        sample.metadata['generation_queued_at'] = time.time()
        async with slots.slot(priority(sample), waiting=(flow, 'gen_queued')):
            # Retry completed generations without rerolling after judge/network failure.
            if keep_files and path.exists():
                saved = load_sample(path, num_layers=getattr(args, 'num_layers', 25),
                                    router_topk=getattr(args, 'moe_router_topk', 4))
                if saved.metadata['mixrl'] != meta:
                    raise ValueError('Saved response identity/config conflict')
                sample = saved
                if sample.status in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
                    sample.metadata['reused_generation'] = True
                    return sample
            params = copy.deepcopy(state.sampling_params)
            params['max_new_tokens'] = meta['cap']
            if evaluation:
                # Eval samples like training but never trains: no candidate sets to return.
                params.pop('custom_params', None)
            seed_key = (['eval', meta['row_id'], meta['sample']] if evaluation
                        else [version, meta['group_id'], meta['sample']])
            params['sampling_seed'] = (c['seed'] + int(digest(seed_key)[:8], 16)) % (2**31)
            sample.metadata['generation_started_at'] = time.time()
            # Separate generation from RM to persist tokens/logprobs before grading.
            from slime.rollout.sglang_rollout import generate
            flow['generating'] += 1
            try:
                async with state.semaphore:
                    with state.dp_rank_context():
                        sample = await generate(args, sample, params)
            finally:
                flow['generating'] -= 1
            sample.metadata['generation_finished_at'] = time.time()
            flow['gen_tokens'] += sample.response_length
            if sample.status not in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED):
                raise RuntimeError('Generation did not terminate normally')
            if sample.response_length < 1 or len(sample.rollout_log_probs or []) != sample.response_length:
                raise RuntimeError('Missing generated tokens/behavior log probabilities')
            if not all(math.isfinite(p) for p in sample.rollout_log_probs):
                raise RuntimeError('Nonfinite behavior log probabilities')
            if replay_top_p:
                ids, offsets = sample.rollout_top_p_token_ids, sample.rollout_top_p_token_offsets
                if (ids is None or offsets is None or len(offsets) != sample.response_length + 1
                        or int(offsets[-1]) != len(ids)):
                    raise RuntimeError('Missing/misaligned top-p candidate sets for replay')
            if c.get('routing_replay'):
                experts = sample.rollout_routed_experts
                expected = (len(sample.tokens) - 1, args.num_layers, args.moe_router_topk)
                if experts is None or tuple(experts.shape) != expected:
                    raise RuntimeError('Missing/misaligned rollout expert path')
                routed = experts[:, 2:, :]
                if (routed < 0).any() or (routed >= args.num_experts).any():
                    raise RuntimeError('Out-of-range rollout expert IDs')
                ordered = routed.sort(dim=-1).values
                if (ordered[..., 1:] == ordered[..., :-1]).any():
                    raise RuntimeError('Duplicate/missing top-k experts in captured path')
                # Dense decoder columns have no routes; clear unused capture
                # buffer bytes before persistence, never alter actual MoE IDs.
                experts[:, :2, :] = 0
            if keep_files:
                # Persist before grading so a step that fails while grading regrades, not rerolls.
                await asyncio.get_running_loop().run_in_executor(persist_pool(), write_record, path, sample)
            return sample

    def write_record(path, sample):
        if evaluation:
            # Eval responses are kept to be read (text, grade, lengths); no routes or top-p sets.
            sample = copy.copy(sample)
            sample.rollout_routed_experts = None
            sample.rollout_top_p_token_ids = sample.rollout_top_p_token_offsets = None
        save_sample(path, sample)

    async def one(sample):
        # Group concurrency bounds the pending judge backlog. Generation slots
        # cover generation only, so slow grading does not idle free model slots.
        sample = await generate_one(sample)
        if sample.reward is None:
            sample.reward = await score(sample)
        if keep_files:
            await asyncio.get_running_loop().run_in_executor(
                persist_pool(), write_record, directory / f'{sample.index}.json', sample)
        flow['done'] += 1
        return sample

    async def execute(group):
        return await gather_cancel(one(s) for s in group)

    interval = float(os.environ.get('MIXRL_PIPELINE_SECONDS', '30'))
    heartbeat = (asyncio.create_task(pipeline_heartbeat(args, rollout_id, phase, flow, interval))
                 if interval > 0 else None)
    try:
        if evaluation:
            data = {}
            evaluation_groups = []
            panel, eval_samples = source.val, c['eval_samples']
            # One timeout for the full panel, rolling bounded group dispatch.
            group_slots = asyncio.Semaphore(c['inflight_groups'])
            async def evaluate_group(group):
                async with group_slots:
                    return await execute(group)
            groups = [source.samples(r, i, version, 'rl_val', eval_samples)
                      for i, r in enumerate(panel)]
            for i, group in enumerate(groups):
                for sample in group:
                    sample.metadata['mixrl_priority'] = i
            completed = await asyncio.wait_for(
                gather_cancel(evaluate_group(g) for g in groups), c['collection_timeout'])
            for row, group in zip(panel, completed):
                evaluation_groups.append((row, group))
                graded = [s for s in group if not grade_failed(s)]  # ungradable responses are never scored
                if not graded:
                    continue  # counted per task in the summary
                entry = data.setdefault(row['task'], {'rewards': [], 'truncated': [], 'samples': []})
                entry['rewards'].extend(s.reward for s in graded)
                entry['truncated'].extend(s.status == Sample.Status.TRUNCATED for s in graded)
                entry['samples'].extend(graded)
            summary = evaluation_summary(evaluation_groups)
            summary['samples_per_prompt'] = eval_samples
            write_json(directory / 'evaluation.json', summary)
            print('MIXRL_EVAL ' + json.dumps({'rollout_id': rollout_id, **summary}), flush=True)
            # Native Slime still logs task means; these entries expose the
            # equal-domain aggregate without treating quality scores as passes.
            data['equal_domain_mean'] = {'rewards': [summary['equal_domain_mean']]}
            for domain, metrics in summary['domains'].items():
                for key, value in metrics.items():
                    if key != 'prompts':
                        data[f'domain/{domain}/{key}'] = {'rewards': [value]}
            write_json(eval_counter_path, counter + 1)
            return RolloutFnEvalOutput(data=data)
        seen = set()
        proposals = iter(range(10**9))
        def propose(route):
            row, index = source.sampler.take(route, seen)
            group, order = source.samples(row, index, version), next(proposals)
            for sample in group:
                sample.metadata['mixrl_priority'] = order
            return group
        def assess(group):
            validate_group(group, args.n_samples_per_prompt)
            capped = [unfinished(s) for s in group]
            skip = masked(group)
            if len(group) - sum(skip) < 2:
                return False, [0.] * len(group), capped, True
            scores, _, _, usable = group_rewards([s.metadata['grade'] for s in group], capped, c['truncation'],
                                                 masked=skip)
            return usable, scores, capped, False, skip
        def event(route, job, group, decision):
            print('MIXRL_GROUP ' + json.dumps({
                'rollout_id': rollout_id, 'route': route, 'group': group[0].group_index,
                'decision': decision, 'rewards': [s.reward for s in group],
                'response_tokens': [s.response_length for s in group],
                'capped': [unfinished(s) for s in group]}), flush=True)
            with (directory / 'collection.jsonl').open('a') as stream:
                stream.write(json.dumps({'route': route, 'group': group[0].group_index,
                                         'row': group[0].metadata['mixrl']['row_id'], 'decision': decision}) + '\n')
        spares = {r: math.ceil(q * c.get('oversample', 0)) for r, q in c['quotas'].items()}
        groups, metrics = await collect(c['quotas'], propose, execute, assess,
                                        inflight=c['inflight_groups'], refill_rounds=c.get('refill_rounds', 0),
                                        timeout=c['collection_timeout'], event=event, spares=spares)
        if any(m.get('cancelled') for m in metrics.values()):
            # Cancelled spares and surplus refills can still be generating in SGLang (with distributed post their
            # requests are not cancelled at all). The trainer frees SGLang's memory next; under a running batch
            # that is a CUDA illegal memory access. Stop them and wait until every engine is idle.
            await asyncio.wait_for(stop_generation(args), 120)
        # Length penalties and loss masks are fixed here, before conversion and logging,
        # so reward normalization reads exactly what the logs report.
        for group in groups:
            usable, scores = assess(group)[:2]
            skip = masked(group)
            deltas = [0.] * len(group)
            if c.get('length_penalty') and usable:
                # Over the graded responses only; a masked response is neither penalized nor counted.
                graded = [i for i, m in enumerate(skip) if not m]
                for i, delta in zip(graded, length_penalties([scores[i] for i in graded],
                                                             [group[i].response_length for i in graded],
                                                             c['length_penalty'])):
                    deltas[i] = delta
            for sample, delta, mask in zip(group, deltas, skip):
                sample.metadata['length_penalty'] = delta
                if not usable or mask or (c['truncation'] == 'mask' and unfinished(sample)):
                    sample.loss_mask = [0] * sample.response_length
        for route in metrics:
            position = source.sampler.position(route)
            metrics[route].update(position)
            batch = [s for g in groups for s in g if s.metadata['mixrl']['task'] == route]
            texts = [s.metadata['grading_text'] for s in batch]
            metrics[route]['think_rate'] = sum(THINK_TAG in t for t in texts) / max(1, len(texts))
            penalized = [s.metadata['length_penalty'] for s in batch if s.metadata['length_penalty'] < 0]
            metrics[route]['length_penalized'] = len(penalized)
            metrics[route]['length_penalty_mean'] = sum(penalized) / len(penalized) if penalized else 0.
            if replay_top_p:
                # Mean top-p candidate-set size per generated token (MiMo reports < 5 at top-p 0.97).
                metrics[route]['top_p_set_mean'] = (sum(int(s.rollout_top_p_token_offsets[-1]) for s in batch)
                                                   / max(1, sum(s.response_length for s in batch)))
            for number in range(snapshot['state'][route]['epoch'] + 2, position['pass'] + 1):
                # The pool ran out during this batch; the task reshuffled and began a new pass.
                print('MIXRL_PASS ' + json.dumps({'rollout_id': rollout_id, 'task': route, 'pass': number,
                                                  'pool': len(source.sampler.pools[route])}), flush=True)
        write_json(directory / 'metrics.json', metrics)
        from .telemetry import collection_summary
        timing = collection_summary([s for g in groups for s in g],
            max(m['collection_seconds'] for m in metrics.values()), sum(m['accepted'] for m in metrics.values()))
        write_json(directory / 'timing.json', timing)
        print('MIXRL_TIMING ' + json.dumps({'rollout_id': rollout_id, **timing}), flush=True)
        print('MIXRL_COLLECTION ' + json.dumps({'rollout_id': rollout_id, 'routes': metrics}), flush=True)
        flattened = {f'mixrl/{route}/{k}': v for route, m in metrics.items() for k, v in m.items()}
        flattened['mixrl/skip_optimizer'] = int(not any(m['accepted'] for m in metrics.values()))
        flattened['mixrl/informative_groups'] = sum(m['accepted'] for m in metrics.values())
        return RolloutFnTrainOutput(samples=groups, metrics=flattened)
    except BaseException as failure:
        print('MIXRL_FAILURE ' + json.dumps({'rollout_id': rollout_id, 'type': type(failure).__name__,
                                            'error': str(failure)}), flush=True)
        source.sampler.restore(snapshot)
        write_json(directory / 'failure.json', {'type': type(failure).__name__, 'error': str(failure)})
        # Ensure outstanding server work cannot survive a failed synchronous batch.
        try:
            await asyncio.wait_for(abort(args, rollout_id), 60)
        except Exception as cleanup_error:
            write_json(directory / 'abort_error.json', {'error': str(cleanup_error)})
        raise
    finally:
        if heartbeat is not None:
            heartbeat.cancel()
        state.reset()


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.utils.async_utils import run
    return run(_rollout(args, rollout_id, data_source, evaluation))
