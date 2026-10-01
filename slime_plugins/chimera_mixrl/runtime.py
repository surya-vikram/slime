"""Narrow Slime hooks. Imports GPU-stack modules only inside runtime functions."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import urllib.request
import urllib.error
import time

from .core import RouteSampler, collect, digest, group_rewards, length_penalties, load_split, write_json
from .routes import THINK_TAG, evaluation_summary, validate_route
from .tasks import blocked
from .objective import group_length_scales
from .records import load_sample, save_sample


def request(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def config():
    return json.loads(Path(os.environ['CHIMERA_MIXRL_CONFIG']).read_text())


_scoring_pool = None


def scoring_pool(size):
    # asyncio.to_thread uses the loop's default executor: min(32, cpu_count + 4) threads,
    # which silently capped reward-service requests (and so the judge) at 32 in flight.
    global _scoring_pool
    if _scoring_pool is None or _scoring_pool._max_workers < size:
        _scoring_pool = ThreadPoolExecutor(max_workers=size, thread_name_prefix='mixrl-score')
    return _scoring_pool


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
    for attempt in range(c['reward_attempts']):
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                scoring_pool(c.get('reward_concurrency', 8)), request, c['scorer_url'] + '/score', payload, c['reward_timeout'])
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt + 1 == c['reward_attempts']:
                raise
        except (OSError, TimeoutError):
            if attempt + 1 == c['reward_attempts']:
                raise
        await asyncio.sleep(min(2 ** attempt, 8))
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
    informative = sum(group_rewards(
        [s.metadata['grade'] for s in samples[start:start+n]],
        [unfinished(s) for s in samples[start:start+n]],
        c['truncation'])[-1] for start in range(0, len(samples), n))
    # Native reduction divides by the fixed global response count. Compensate
    # once, before DP partitioning, to average only informative prompt groups.
    batch_scale = args.rollout_batch_size / informative if informative else 0.
    if c.get('objective', 'dapo') != 'mimo':
        # Native DAPO's per-token denominator already excludes masked tokens.
        batch_scale = 1.
    for start in range(0, len(samples), n):
        group = samples[start:start + n]
        validate_group(group, n)
        capped = [unfinished(s) for s in group]
        # Length penalties were computed at collection time from the same scores and lengths.
        scores, advantages, eligible, usable = group_rewards(
            [s.metadata['grade'] for s in group], capped, c['truncation'],
            args.grpo_std_normalization if c.get('objective', 'dapo') == 'dapo' else False,
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
    if not evaluation and os.environ.get('MIXRL_KEEP_TRAIN_SAMPLES', '1') == '0':
        # A finished step's per-response files (tokens, log-probs, expert routes, top-p sets:
        # up to MBs each) only serve a retry of that step. Keep its summaries and collection log.
        for old in (source.run_dir / 'rollouts').glob('train-*'):
            step = old.name.split('-', 1)[1]
            if step.isdigit() and int(step) < rollout_id:
                for path in old.glob('*.json'):
                    if path.stem.isdigit():
                        path.unlink()
    health = await asyncio.to_thread(request, c['scorer_url'] + '/health')
    if health['protocol_id'] != c['scorer_protocol']:
        raise RuntimeError('Scorer protocol mismatch before rollout')
    # Checked before every batch: the judge can go down mid-run, and that must stop
    # training before generation, never turn judge-graded answers into zeros.
    reasons = blocked(c['routes'], health)
    if reasons:
        raise RuntimeError('Reward service cannot grade enabled tasks: '
                           + '; '.join(f'{task}: {reason}' for task, reason in reasons.items()))
    slots = asyncio.Semaphore(c['response_concurrency'])
    scoring_slots = asyncio.Semaphore(c.get('reward_concurrency', 8))

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
        async with scoring_slots:
            return await reward(args, sample)

    async def generate_one(sample):
        meta = sample.metadata['mixrl']
        path = directory / f'{sample.index}.json'
        sample.metadata['generation_queued_at'] = time.time()
        async with slots:
            # Retry completed generations without rerolling after judge/network failure.
            if path.exists():
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
            async with state.semaphore:
                with state.dp_rank_context():
                    sample = await generate(args, sample, params)
            sample.metadata['generation_finished_at'] = time.time()
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
            save_sample(path, sample)
            return sample

    async def one(sample):
        # Group concurrency bounds the pending judge backlog. Generation slots
        # cover generation only, so slow grading does not idle free model slots.
        sample = await generate_one(sample)
        if sample.reward is None:
            sample.reward = await score(sample)
        save_sample(directory / f'{sample.index}.json', sample)
        return sample

    async def execute(group):
        return await gather_cancel(one(s) for s in group)

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
            completed = await asyncio.wait_for(
                gather_cancel(evaluate_group(g) for g in groups), c['collection_timeout'])
            for row, group in zip(panel, completed):
                evaluation_groups.append((row, group))
                entry = data.setdefault(row['task'], {'rewards': [], 'truncated': [], 'samples': []})
                entry['rewards'].extend(s.reward for s in group)
                entry['truncated'].extend(s.status == Sample.Status.TRUNCATED for s in group)
                entry['samples'].extend(group)
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
        def propose(route):
            row, index = source.sampler.take(route, seen)
            return source.samples(row, index, version)
        def assess(group):
            validate_group(group, args.n_samples_per_prompt)
            capped = [unfinished(s) for s in group]
            scores, _, _, usable = group_rewards([s.metadata['grade'] for s in group], capped, c['truncation'])
            return usable, scores, capped
        def event(route, job, group, decision):
            print('MIXRL_GROUP ' + json.dumps({
                'rollout_id': rollout_id, 'route': route, 'group': group[0].group_index,
                'decision': decision, 'rewards': [s.reward for s in group],
                'response_tokens': [s.response_length for s in group],
                'capped': [unfinished(s) for s in group]}), flush=True)
            with (directory / 'collection.jsonl').open('a') as stream:
                stream.write(json.dumps({'route': route, 'group': group[0].group_index,
                                         'row': group[0].metadata['mixrl']['row_id'], 'decision': decision}) + '\n')
        groups, metrics = await collect(c['quotas'], propose, execute, assess,
                                        inflight=c['inflight_groups'], refill_rounds=c.get('refill_rounds', 0),
                                        timeout=c['collection_timeout'], event=event)
        # Length penalties and loss masks are fixed here, before conversion and logging,
        # so reward normalization reads exactly what the logs report.
        for group in groups:
            usable, scores, _ = assess(group)
            deltas = [0.] * len(group)
            if c.get('length_penalty') and usable:
                deltas = length_penalties(scores, [s.response_length for s in group], c['length_penalty'])
            for sample, delta in zip(group, deltas):
                sample.metadata['length_penalty'] = delta
                if not usable or (c['truncation'] == 'mask' and unfinished(sample)):
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
        state.reset()


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.utils.async_utils import run
    return run(_rollout(args, rollout_id, data_source, evaluation))
