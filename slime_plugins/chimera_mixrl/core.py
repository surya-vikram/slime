"""CPU-testable, deterministic quota collection and group reward semantics."""
import asyncio
import hashlib
import json
import math
import random
import statistics
import time
from collections import deque
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_split(root, split):
    if split not in ('rl_train', 'rl_val'):
        raise ValueError('MixRL cannot load main_test for training/monitoring')
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    with (root / 'splits' / f'{split}.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if digest(rows) != manifest['splits'][split]['hash']:
        raise ValueError(f'{split}: frozen manifest mismatch')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate record identity')
    return rows, manifest


class RouteSampler:
    """Independent seeded passes per task, resumable without cross-task spillover.

    Each pass visits every row once in a fresh seeded order. A row whose family is
    already in the current batch (a batch that straddles two passes) is deferred to
    the next batch instead of being dropped for the whole pass.
    """
    def __init__(self, rows, routes, seed=42):
        self.pools = {r: sorted([x for x in rows if x['task'] == r], key=lambda x: x['id']) for r in routes}
        if any(not p for p in self.pools.values()):
            raise ValueError('An enabled route has no admitted rows')
        self.rows = {r: {x['id']: x for x in p} for r, p in self.pools.items()}
        self.seed = seed
        self.state = {r: {'epoch': 0, 'offset': 0, 'deferred': []} for r in routes}
        self.group_index = 0
        self.fingerprint = digest({'pools': {r: [x['id'] for x in p] for r, p in self.pools.items()}, 'seed': seed})
        self.orders = {}

    def snapshot(self):
        return json.loads(json.dumps({'fingerprint': self.fingerprint, 'state': self.state,
                                     'group_index': self.group_index}))

    def restore(self, snapshot):
        if snapshot['fingerprint'] != self.fingerprint or set(snapshot['state']) != set(self.pools):
            raise ValueError('Sampler checkpoint does not match admitted data/seed/routes')
        for route, state in snapshot['state'].items():
            deferred = state.get('deferred')
            if (set(state) != {'epoch', 'offset', 'deferred'} or state['epoch'] < 0
                    or not 0 <= state['offset'] <= len(self.pools[route]) or not isinstance(deferred, list)
                    or len(set(deferred)) != len(deferred) or any(i not in self.rows[route] for i in deferred)):
                raise ValueError('Invalid sampler cursor')
        self.state = json.loads(json.dumps(snapshot['state']))
        self.group_index = snapshot['group_index']

    def position(self, route):
        """1-based pass number, fraction of the current pass drawn, and rows waiting for the next batch."""
        state = self.state[route]
        return {'pass': state['epoch'] + 1, 'pass_progress': state['offset'] / len(self.pools[route]),
                'deferred': len(state['deferred'])}

    def _order(self, route, epoch):
        key = (route, epoch)
        if key not in self.orders:
            order = list(range(len(self.pools[route])))
            random.Random(f'{self.seed}:{route}:{epoch}').shuffle(order)
            self.orders = {k: v for k, v in self.orders.items() if k[0] != route}
            self.orders[key] = order
        return self.orders[key]

    def _accept(self, row, seen):
        seen.add(row['family_id'])
        index = self.group_index
        self.group_index += 1
        return row, index

    def take(self, route, seen):
        pool, cursor = self.pools[route], self.state[route]
        for row_id in cursor['deferred']:
            row = self.rows[route][row_id]
            if row['family_id'] not in seen:
                cursor['deferred'].remove(row_id)
                return self._accept(row, seen)
        for _ in range(len(pool) * 2):
            if cursor['offset'] == len(pool):
                cursor['epoch'] += 1
                cursor['offset'] = 0
            row = pool[self._order(route, cursor['epoch'])[cursor['offset']]]
            cursor['offset'] += 1
            if row['family_id'] not in seen:
                return self._accept(row, seen)
            if row['id'] not in cursor['deferred']:
                cursor['deferred'].append(row['id'])
        raise RuntimeError(f'{route}: distinct family pool exhausted within this update')


def group_rewards(grades, capped, policy='mask', standardize=True):
    """Cap masking excludes censored scores from sibling statistics, not just loss.

    This is an explicit Chimera adaptation. Raw evaluation scores remain untouched.
    None/invalid judgments must not be supplied as a valid zero.
    """
    if policy not in ('mask', 'zero') or len(grades) != len(capped) or len(grades) < 2:
        raise ValueError('Invalid group configuration')
    scores = []
    for grade in grades:
        score = grade.get('score')
        if grade.get('status') != 'valid' or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid required reward; no partial group or fabricated zero')
        scores.append(float(score))
    scores = [0. if cap else score for score, cap in zip(scores, capped)]
    eligible = [not cap or policy == 'zero' for cap in capped]
    valid = [s for s, keep in zip(scores, eligible) if keep]
    if len(valid) < 2:
        return scores, [0.] * len(scores), eligible, False
    mean = statistics.mean(valid)
    std = statistics.stdev(valid)
    # Epsilon is only numerical protection, not permission for judge-noise rewards.
    advantages = [(s - mean) / (std + 1e-6) if standardize else s - mean for s in scores]
    advantages = [a if keep else 0. for a, keep in zip(advantages, eligible)]
    return scores, advantages, eligible, max(valid) - min(valid) > 1e-8


async def collect(quotas, propose, execute, assess, *, inflight=4, max_attempts=100,
                  timeout=600, event=None):
    """Fixed sampled quotas; constant groups are retained for zero-loss masking.

    Preselect the entire batch before execution. Bounded rolling dispatch releases
    a slot as soon as its group finishes; results/events retain proposal order.
    There is no finish-time selection, replacement or cross-policy leftover.
    """
    if not quotas or any(type(v) is not int or v < 1 for v in quotas.values()):
        raise ValueError('Quotas must be positive integer sampled-group counts')
    if inflight < 1 or max_attempts < 1 or timeout <= 0:
        raise ValueError('Invalid collection limits')
    accepted = {r: [] for r in quotas}
    attempts = {r: 0 for r in quotas}
    routes = deque(quotas)
    metrics = {r: {'attempted': 0, 'accepted': 0, 'constant': 0, 'surplus': 0,
                   'all_correct': 0, 'all_wrong': 0,
                   'score_sum': 0., 'responses': 0, 'capped': 0} for r in quotas}

    async def run():
        started = time.monotonic()
        jobs = []
        while any(attempts[r] < quotas[r] for r in quotas):
            route = next(r for r in routes if attempts[r] < quotas[r])
            while routes[0] != route:
                routes.rotate(-1)
            routes.rotate(-1)
            jobs.append((route, propose(route)))
            attempts[route] += 1
        slots = asyncio.Semaphore(inflight)
        async def bounded(job):
            async with slots:
                return await execute(job)
        tasks = [asyncio.create_task(bounded(job)) for _, job in jobs]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        for (route, job), group in zip(jobs, results):
            usable, scores, capped = assess(group)
            m = metrics[route]
            m['attempted'] += 1
            m['score_sum'] += sum(scores)
            m['responses'] += len(scores)
            m['capped'] += sum(capped)
            m['all_correct'] += int(not any(capped) and all(s == 1 for s in scores))
            m['all_wrong'] += int(not any(capped) and all(s == 0 for s in scores))
            decision = 'constant'
            accepted[route].append(group)
            if usable:
                decision = 'accepted'
            m[decision] += 1
            if event:
                event(route, job, group, decision)
        for m in metrics.values():
            m['raw_reward_mean'] = m['score_sum'] / max(1, m['responses'])
            m['acceptance_rate'] = m['accepted'] / max(1, m['attempted'])
            m['cap_rate'] = m['capped'] / max(1, m['responses'])
            m['collection_seconds'] = time.monotonic() - started
        return [g for r in quotas for g in accepted[r]], metrics

    return await asyncio.wait_for(run(), timeout)
