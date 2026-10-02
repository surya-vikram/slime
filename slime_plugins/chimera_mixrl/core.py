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


# MiMo-V2.6 group-relative length penalty (paper Eq. 4; public recipe values from
# XiaomiMiMo/verl recipes/general/config/general.yaml, verl/utils/length_penalty.py).
MIMO_LENGTH_PENALTY = {'max_penalty': 0.1, 'deadzone': 0.3, 'saturate': 1.0, 'exponent': 1.5,
                       'pass_threshold': 0.5, 'anchor_quantile': 0.5, 'min_pass_rate': 0.5}


def length_penalties(scores, lengths, cfg=MIMO_LENGTH_PENALTY):
    """Non-positive reward deltas for one prompt group; only passing responses are penalized.

    Applies when more than min_pass_rate of the group passed. The anchor is the
    anchor_quantile of the passing responses' generated-token counts; a passing
    response whose relative excess over it exceeds the deadzone loses
    max_penalty * t**exponent, with t ramping from 0 at the deadzone to 1 at saturate.
    """
    if len(scores) != len(lengths) or not scores:
        raise ValueError('Length penalty needs one length per score')
    deltas = [0.] * len(scores)
    passed = [i for i, s in enumerate(scores) if s >= cfg['pass_threshold']]
    if not passed or len(passed) / len(scores) <= cfg['min_pass_rate']:
        return deltas
    ordered = sorted(lengths[i] for i in passed)
    position = cfg['anchor_quantile'] * (len(ordered) - 1)  # numpy.percentile's linear rule
    low = math.floor(position)
    anchor = ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)
    if anchor <= 0:
        return deltas
    for i in passed:
        excess = max(0., (lengths[i] - anchor) / anchor)
        if excess > cfg['deadzone']:
            t = min((excess - cfg['deadzone']) / (cfg['saturate'] - cfg['deadzone']), 1.)
            deltas[i] = -cfg['max_penalty'] * t ** cfg['exponent']
    return deltas


def group_rewards(grades, capped, policy='mask', standardize=True, penalties=None):
    """Group-relative advantages from validated scores.

    capped marks responses without a finished answer (cut off at the cap, or reasoning
    that never closed). 'zero' scores them 0 inside the group (MiMo/DAPO: no answer is a
    wrong answer); 'mask' excludes them from sibling statistics and the loss.
    penalties (e.g. length_penalties) shift the scores used for advantages only; whether
    the group is informative is decided by the outcome scores. Raw scores are returned
    unchanged. None/invalid judgments must not be supplied as a valid zero.
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
    if len(valid) < 2 or max(valid) - min(valid) <= 1e-8:
        return scores, [0.] * len(scores), eligible, False
    shaped = [s + d for s, d in zip(scores, penalties)] if penalties else scores
    shaped_valid = [s for s, keep in zip(shaped, eligible) if keep]
    mean = statistics.mean(shaped_valid)
    std = statistics.stdev(shaped_valid)
    # Epsilon is only numerical protection, not permission for judge-noise rewards.
    advantages = [(s - mean) / (std + 1e-6) if standardize else s - mean for s in shaped]
    advantages = [a if keep else 0. for a, keep in zip(advantages, eligible)]
    return scores, advantages, eligible, True


class PriorityGate:
    """A semaphore that hands free slots to the lowest priority value first (FIFO among equals).

    Used so responses of earlier-proposed groups are generated and graded first: groups then
    finish one after another and their keep-or-refill decisions arrive steadily, instead of
    every group being partly graded until the end of the step."""

    def __init__(self, slots):
        if slots < 1:
            raise ValueError('PriorityGate needs at least one slot')
        self.free, self.waiters, self.count = slots, [], 0

    async def acquire(self, priority=0):
        import heapq
        if self.free > 0 and not self.waiters:
            self.free -= 1
            return
        future = asyncio.get_running_loop().create_future()
        self.count += 1
        heapq.heappush(self.waiters, (priority, self.count, future))
        try:
            await future
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                self.release()  # granted just as it was cancelled: pass the slot on
            raise

    def release(self):
        import heapq
        while self.waiters:
            _, _, future = heapq.heappop(self.waiters)
            if not future.done():
                future.set_result(None)
                return
        self.free += 1

    def slot(self, priority=0):
        gate = self

        class _Slot:
            async def __aenter__(self):
                await gate.acquire(priority)

            async def __aexit__(self, *exc):
                gate.release()
        return _Slot()


async def collect(quotas, propose, execute, assess, *, inflight=4, refill_rounds=0,
                  timeout=600, event=None, spares=None):
    """Per task, collect `quota` groups for the batch, preferring informative ones.

    The whole initial batch is proposed up front and generated with bounded rolling
    dispatch. With refill_rounds > 0, a group without reward spread is replaced by a
    fresh prompt from the same task while the task is still short (informative plus
    still-pending groups below its quota), up to (1 + refill_rounds) * quota prompts
    per task per step. A task that stays short fills its remaining slots with its
    constant groups (zero loss), so every step has the same batch shape.

    Groups are assessed as soon as they finish, and a replacement is dispatched at once, so
    a slow group no longer holds back refills for groups proposed after it. The refill rule
    (informative + pending < quota, i.e. attempts < quota + uniform groups so far) depends only
    on how many groups turned out uniform, not on their order, so the same replacement prompts
    are proposed as with in-order resolution. The batch is still taken in proposal order, so
    it depends only on outcomes: a retried step proposes and selects the same prompts.
    Informative groups never exceed the quota.

    spares[route] > 0 oversamples: that many extra groups stay in flight beyond those still
    needed, so a uniform group's replacement is already running instead of starting after it
    is graded. Spares are proposed after the groups they back up, so the batch (the first
    `quota` informative groups in proposal order) is unchanged; once those are known, groups
    proposed after them are cancelled. Surplus informative groups are dropped.
    """
    if not quotas or any(type(v) is not int or v < 1 for v in quotas.values()):
        raise ValueError('Quotas must be positive integer sampled-group counts')
    if inflight < 1 or type(refill_rounds) is not int or refill_rounds < 0 or timeout <= 0:
        raise ValueError('Invalid collection limits')
    attempts = {r: 0 for r in quotas}
    pending = {r: 0 for r in quotas}
    informative = {r: [] for r in quotas}
    constant = {r: [] for r in quotas}
    failed = {r: [] for r in quotas}
    budget = {r: quotas[r] * (1 + refill_rounds) for r in quotas}
    spares = {r: max(0, int((spares or {}).get(r, 0))) for r in quotas}
    routes = deque(quotas)
    metrics = {r: {'attempted': 0, 'accepted': 0, 'constant': 0, 'refilled': 0, 'padding': 0,
                   'all_correct': 0, 'all_wrong': 0, 'grade_failed': 0, 'cancelled': 0,
                   'score_sum': 0., 'responses': 0, 'capped': 0} for r in quotas}

    async def run():
        started = time.monotonic()
        slots = asyncio.Semaphore(inflight)
        jobs = []      # (route, job, task) in proposal order
        running = {}   # task -> proposal index

        async def bounded(job):
            async with slots:
                return await execute(job)

        def dispatch(route):
            job = propose(route)
            task = asyncio.create_task(bounded(job))
            running[task] = len(jobs)
            jobs.append((route, job, task))
            attempts[route] += 1
            pending[route] += 1

        cancelled = []

        def short(route):
            # Keep the groups still needed (if every pending one turns out informative), plus spares.
            needed = quotas[route] - len(informative[route])
            return needed > 0 and pending[route] < needed + spares[route] and attempts[route] < budget[route]

        def prune(route):
            # Once the first `quota` informative groups in proposal order are known, nothing
            # proposed after them can enter the batch: cancel those still running.
            if len(informative[route]) < quotas[route]:
                return
            last = sorted(i for i, _ in informative[route])[quotas[route] - 1]
            mine = [(task, i) for task, i in running.items() if jobs[i][0] == route]
            if any(i < last for _, i in mine):
                return
            for task, i in mine:
                running.pop(task)
                pending[route] -= 1
                task.cancel()
                cancelled.append(task)
                metrics[route]['cancelled'] += 1
                if event:
                    event(route, jobs[i][1], jobs[i][1], 'cancelled')

        def resolve(index, route, job, group):
            outcome = assess(group)
            usable, scores, capped = outcome[:3]
            m = metrics[route]
            m['attempted'] += 1
            if len(outcome) > 3 and outcome[3]:
                # Some response could not be graded: no scores to count or train on.
                # Replace it like a constant group; use it only as zero-loss padding.
                failed[route].append((index, group))
                m['grade_failed'] += 1
                decision = 'grade_failed'
                if short(route):
                    dispatch(route)
                    m['refilled'] += 1
                    decision = 'grade_failed_replaced'
            else:
                m['score_sum'] += sum(scores)
                m['responses'] += len(scores)
                m['capped'] += sum(capped)
                m['all_correct'] += int(not any(capped) and all(s == 1 for s in scores))
                m['all_wrong'] += int(not any(capped) and all(s == 0 for s in scores))
                if usable:
                    informative[route].append((index, group))
                    decision = 'accepted'
                else:
                    constant[route].append((index, group))
                    decision = 'constant'
                    if short(route):
                        dispatch(route)
                        m['refilled'] += 1
                        decision = 'replaced'
                m['accepted' if usable else 'constant'] += 1
            if event:
                event(route, job, group, decision)

        initial = {r: min(quotas[r] + spares[r], budget[r]) for r in quotas}
        try:
            while any(attempts[r] < initial[r] for r in quotas):
                route = next(r for r in routes if attempts[r] < initial[r])
                while routes[0] != route:
                    routes.rotate(-1)
                routes.rotate(-1)
                dispatch(route)
            while running:
                done, _ = await asyncio.wait(list(running), return_when=asyncio.FIRST_COMPLETED)
                # Groups that finish together are handled in proposal order.
                for task in sorted(done, key=lambda t: running.get(t, -1)):
                    if task not in running:
                        continue  # pruned while this batch of finished groups was handled
                    index = running.pop(task)
                    route, job, _ = jobs[index]
                    group = task.result()
                    pending[route] -= 1
                    resolve(index, route, job, group)
                    prune(route)
            await asyncio.gather(*cancelled, return_exceptions=True)
        except BaseException:
            tasks = [task for _, _, task in jobs]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        ordered = lambda pairs: [group for _, group in sorted(pairs, key=lambda pair: pair[0])]
        batch = []
        for route, quota in quotas.items():
            chosen = ordered(informative[route])[:quota]
            padding = (ordered(constant[route]) + ordered(failed[route]))[:quota - len(chosen)]
            metrics[route]['padding'] = len(padding)
            batch.extend(chosen + padding)
        for m in metrics.values():
            m['raw_reward_mean'] = m['score_sum'] / max(1, m['responses'])
            m['acceptance_rate'] = m['accepted'] / max(1, m['attempted'])
            m['cap_rate'] = m['capped'] / max(1, m['responses'])
            m['collection_seconds'] = time.monotonic() - started
        return batch, metrics

    return await asyncio.wait_for(run(), timeout)
