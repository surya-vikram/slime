"""No torch/Ray/SGLang required for the v0 contract tests."""
import asyncio
import collections
import copy
import unittest

from slime_plugins.chimera_mixrl.core import RouteSampler, collect, group_rewards


def grade(score):
    return {'status': 'valid', 'score': score, 'passed': None}


class GroupTests(unittest.TestCase):
    def test_mask_excludes_cap_from_sibling_baseline(self):
        raw, advantages, mask, usable = group_rewards([grade(1), grade(1), grade(0)], [False, False, True])
        self.assertEqual(raw, [1., 1., 0.])
        self.assertEqual(advantages, [0., 0., 0.])
        self.assertEqual(mask, [True, True, False])
        self.assertFalse(usable)

    def test_mask_preserves_variable_completed_siblings(self):
        _, advantages, mask, usable = group_rewards([grade(1), grade(0), grade(0)], [False, False, True])
        self.assertTrue(usable)
        self.assertAlmostEqual(sum(advantages), 0.)
        self.assertEqual(advantages[-1], 0.)
        self.assertFalse(mask[-1])

    def test_zero_policy_is_explicitly_different(self):
        _, adv, mask, usable = group_rewards([grade(1), grade(1), grade(0)], [False, False, True], 'zero')
        self.assertTrue(usable)
        self.assertTrue(all(mask))
        self.assertLess(adv[-1], 0)

    def test_invalid_or_insufficient_cannot_be_training_group(self):
        for invalid in ({'status': 'error'}, grade(float('nan')), grade(True), grade(1.1)):
            with self.assertRaises(ValueError):
                group_rewards([grade(1), invalid], [False, False])
        self.assertFalse(group_rewards([grade(1), grade(0)], [False, True])[-1])
        self.assertFalse(group_rewards([grade(0), grade(0)], [True, True])[-1])

    def test_quality_not_forced_to_binary(self):
        self.assertTrue(group_rewards([grade(.5), grade(.75)], [False, False])[-1])

    def test_zero_policy_scores_unfinished_as_wrong_inside_the_group(self):
        raw, adv, eligible, usable = group_rewards([grade(1), grade(1), grade(1)], [False, False, True], 'zero',
                                                   standardize=False)
        self.assertEqual((raw, eligible, usable), ([1., 1., 0.], [True, True, True], True))
        self.assertAlmostEqual(adv[2], -2 / 3)

    def test_penalties_shape_advantages_but_not_informativeness(self):
        # Every answer correct: no outcome spread, so a length penalty alone never trains the group.
        _, adv, _, usable = group_rewards([grade(1)] * 3, [False] * 3, 'zero', False, [0., 0., -.1])
        self.assertFalse(usable)
        self.assertEqual(adv, [0.] * 3)
        raw, adv, _, usable = group_rewards([grade(1), grade(1), grade(0)], [False] * 3, 'zero', False, [0., -.1, 0.])
        self.assertTrue(usable)
        self.assertEqual(raw, [1., 1., 0.])  # logged scores stay raw
        for value, expected in zip(adv, (1 - 1.9 / 3, .9 - 1.9 / 3, -1.9 / 3)):
            self.assertAlmostEqual(value, expected)


class LengthPenaltyTests(unittest.TestCase):
    """MiMo-V2.6 Eq. 4 with its public recipe values (checked against MiMo's verl code)."""

    def test_only_long_correct_answers_in_mostly_passing_groups(self):
        from slime_plugins.chimera_mixrl.core import length_penalties
        # Passing lengths 100, 100, 200: anchor (median) 100. 200 is +100%: saturated, -0.1.
        self.assertEqual(length_penalties([1, 1, 1, 0], [100, 100, 200, 5000]), [0., 0., -.1, 0.])
        # +65%: t = (0.65 - 0.3) / 0.7 = 0.5, penalty 0.1 * 0.5 ** 1.5.
        deltas = length_penalties([1, 1, 1, 1], [100, 100, 165, 100])
        self.assertAlmostEqual(deltas[2], -.1 * .5 ** 1.5)
        # Within the 30% deadzone, or a group where only half passed: no penalty.
        self.assertEqual(length_penalties([1, 1, 1], [100, 100, 130]), [0.] * 3)
        self.assertEqual(length_penalties([1, 1, 0, 0], [100, 400, 10, 10]), [0.] * 4)
        # Partial credit at the 0.5 pass threshold counts as passing; failed answers are never penalized.
        self.assertEqual(length_penalties([.5, 1, 1, 0], [100, 100, 900, 9000])[2:], [-.1, 0.])


class SamplerTests(unittest.TestCase):
    def test_small_route_cycles_independently_without_cross_domain_spill(self):
        rows = [{'id': str(i), 'family_id': str(i), 'task': 'a' if i < 2 else 'b'} for i in range(8)]
        sampler = RouteSampler(rows, ['a', 'b'])
        ids = []
        for _ in range(8):
            row, _ = sampler.take('a', set())
            ids.append(row['id'])
            saved = sampler.snapshot()
            sampler = RouteSampler(rows, ['a', 'b'])
            sampler.restore(saved)
        self.assertEqual(len(set(ids[:2])), 2)
        self.assertEqual(set(ids), {'0', '1'})
        self.assertEqual(sampler.state['b']['epoch'], 0)
        self.assertEqual(sampler.state['b']['offset'], 0)
        sampler.take('a', set())
        self.assertEqual(sampler.state['a']['epoch'], 4)

    def test_first_route_epoch_unique_across_batches_and_resume(self):
        rows = [{'id': str(i), 'family_id': str(i), 'task': 'a'} for i in range(12)]
        sampler = RouteSampler(rows, ['a'])
        ids = []
        for batch in range(4):
            seen = set()
            ids.extend(sampler.take('a', seen)[0]['id'] for _ in range(3))
            saved = sampler.snapshot()
            sampler = RouteSampler(rows, ['a'])
            sampler.restore(saved)
        self.assertEqual(len(set(ids)), 12)
        self.assertEqual(sampler.state['a']['epoch'], 0)

    def test_exact_restart_and_no_family_reuse(self):
        rows = [{'id': str(i), 'family_id': str(i), 'task': 'a'} for i in range(12)]
        a = RouteSampler(rows, ['a'])
        seen = set()
        a.take('a', seen)
        saved = a.snapshot()
        b = RouteSampler(rows, ['a'])
        b.restore(saved)
        other_seen = set(seen)
        self.assertEqual(a.take('a', seen), b.take('a', other_seen))
        # A fresh comparison covers cycling while a shared seen set prevents duplicates.
        a = RouteSampler(rows, ['a'])
        seen = set()
        identities = [a.take('a', seen)[0]['id'] for _ in range(12)]
        self.assertEqual(len(set(identities)), 12)
        seen = {str(i) for i in range(12)}
        with self.assertRaises(RuntimeError):
            a.take('a', seen)
        broken = copy.deepcopy(saved)
        broken['fingerprint'] = 'different'
        with self.assertRaises(ValueError):
            b.restore(broken)

    def test_clashing_row_leads_the_next_batch(self):
        rows = [{'id': str(i), 'family_id': str(i), 'task': 'a'} for i in range(7)]
        sampler = RouteSampler(rows, ['a'])
        order = [rows[i]['id'] for i in sampler._order('a', 0)]
        # Every family but the pass's last row is already in this batch, so the six drawn first wait.
        row, _ = sampler.take('a', set(order[:-1]))
        self.assertEqual(row['id'], order[-1])
        self.assertEqual(sampler.state['a']['deferred'], order[:-1])
        seen = set()
        following = [sampler.take('a', seen)[0]['id'] for _ in range(6)]
        self.assertEqual(following, order[:-1])
        self.assertEqual(sampler.position('a'), {'pass': 1, 'pass_progress': 1.0, 'deferred': 0})

    def test_batches_across_pass_boundaries_lose_no_rows(self):
        deferred_happened = False
        for seed in range(30):
            rows = [{'id': str(i), 'family_id': str(i), 'task': 'a'} for i in range(7)]
            sampler = RouteSampler(rows, ['a'], seed)
            served = collections.Counter()
            for _ in range(60):
                seen = set()
                batch = [sampler.take('a', seen)[0]['id'] for _ in range(3)]
                self.assertEqual(len(set(batch)), 3)
                served.update(batch)
                state = sampler.state['a']
                waiting = state.get('deferred', [])
                deferred_happened |= bool(waiting)
                # Every row drawn from a pass is either served or still waiting; none is dropped.
                self.assertEqual(sum(served.values()) + len(waiting), state['epoch'] * 7 + state['offset'])
            self.assertLessEqual(max(served.values()) - min(served.values()), 2)
        self.assertTrue(deferred_happened)

    def test_waiting_rows_survive_resume(self):
        rows = [{'id': str(i), 'family_id': str(i), 'task': 'a'} for i in range(7)]
        for seed in range(30):
            a = RouteSampler(rows, ['a'], seed)
            for _ in range(20):
                seen = set()
                for _ in range(3):
                    a.take('a', seen)
                if a.state['a']['deferred']:
                    break
            if not a.state['a']['deferred']:
                continue
            b = RouteSampler(rows, ['a'], seed)
            b.restore(a.snapshot())
            for _ in range(10):
                seen_a, seen_b = set(), set()
                self.assertEqual([a.take('a', seen_a)[0]['id'] for _ in range(3)],
                                 [b.take('a', seen_b)[0]['id'] for _ in range(3)])
            for corrupt in (['missing'], ['0', '0']):
                broken = a.snapshot()
                broken['state']['a']['deferred'] = corrupt
                with self.assertRaises(ValueError):
                    b.restore(broken)
            return
        self.fail('No seed produced a waiting row')


class CollectorTests(unittest.IsolatedAsyncioTestCase):
    async def test_rolling_dispatch_does_not_wait_for_slow_wave_sibling(self):
        proposed = []
        third_started = asyncio.Event()
        async def execute(job):
            self.assertEqual(len(proposed), 4)  # Whole fixed batch selected first.
            if job == 0:
                await asyncio.wait_for(third_started.wait(), .5)
            if job == 2:
                third_started.set()
            return job
        def propose(route):
            proposed.append(len(proposed))
            return proposed[-1]
        groups, metrics = await collect({'a': 4}, propose, execute,
            lambda g: (True, [0., 1.], [False, False]), inflight=2)
        self.assertEqual(groups, [0, 1, 2, 3])
        self.assertEqual(metrics['a']['attempted'], 4)

    async def scenario(self, reverse):
        counters = {'a': 0, 'b': 0}
        events = []
        active = 0
        maximum = 0
        def propose(route):
            counters[route] += 1
            return route, counters[route]
        async def execute(job):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(.002 if (job[1] % 2 == reverse) else .0001)
            active -= 1
            return job
        groups, metrics = await collect({'a': 3, 'b': 2}, propose, execute,
            lambda g: (g[1] % 2 == 0, [0., 1.], [False, False]), inflight=4,
            event=lambda r, j, g, d: events.append((r, j, d)))
        self.assertLessEqual(maximum, 4)
        self.assertEqual([metrics[r]['attempted'] for r in ('a', 'b')], [3, 2])
        self.assertEqual([metrics[r]['accepted'] for r in ('a', 'b')], [1, 1])
        self.assertEqual(len(groups), 5)
        return groups, metrics, events

    async def test_completion_order_does_not_change_admission(self):
        first, second = await self.scenario(False), await self.scenario(True)
        for result in (first, second):
            for metrics in result[1].values():
                self.assertGreater(metrics.pop('collection_seconds'), 0)
        # Same batch and counts; events are logged as groups finish, so only their set matches.
        self.assertEqual(first[:2], second[:2])
        self.assertEqual(sorted(first[2]), sorted(second[2]))

    async def test_constant_groups_do_not_trigger_replacement(self):
        async def execute(job):
            return job
        groups, metrics = await collect({'a': 1}, lambda r: r, execute,
                          lambda g: (False, [0., 0.], [False, False]))
        self.assertEqual(groups, ['a'])
        self.assertEqual(metrics['a']['attempted'], 1)
        self.assertEqual(metrics['a']['accepted'], 0)

    async def refill(self, outcomes, quotas, rounds, delays=None):
        """outcomes[route][k] is whether that route's k-th proposal has reward spread."""
        proposals = {r: 0 for r in quotas}
        events = []
        def propose(route):
            proposals[route] += 1
            return route, proposals[route]
        async def execute(job):
            await asyncio.sleep((delays or {}).get(job, .0001))
            return job
        def assess(job):
            route, number = job
            usable = outcomes[route][number - 1] if number <= len(outcomes[route]) else False
            return usable, [1., 0.] if usable else [0., 0.], [False, False]
        groups, metrics = await collect(quotas, propose, execute, assess, inflight=3, refill_rounds=rounds,
                                        event=lambda r, j, g, d: events.append((j, d)))
        return groups, metrics, events

    async def test_refill_replaces_constant_groups_until_the_quota_is_informative(self):
        groups, metrics, events = await self.refill({'a': [False, True, True, True]}, {'a': 2}, 2)
        self.assertEqual(groups, [('a', 2), ('a', 3)])
        self.assertEqual((metrics['a']['attempted'], metrics['a']['refilled'], metrics['a']['padding']), (3, 1, 0))
        self.assertEqual(events[0], (('a', 1), 'replaced'))

    async def test_refill_budget_is_bounded_then_pads_with_constant_groups(self):
        # A task the model never gets any spread on: (1 + 2) x quota prompts, then carry on.
        groups, metrics, _ = await self.refill({'a': [], 'b': [True]}, {'a': 2, 'b': 1}, 2)
        self.assertEqual(groups, [('a', 1), ('a', 2), ('b', 1)])
        self.assertEqual((metrics['a']['attempted'], metrics['a']['refilled'], metrics['a']['padding']), (6, 4, 2))
        self.assertEqual((metrics['b']['attempted'], metrics['b']['refilled']), (1, 0))

    async def test_refill_off_keeps_the_fixed_batch(self):
        groups, metrics, events = await self.refill({'a': [False, True]}, {'a': 2}, 0)
        self.assertEqual(groups, [('a', 2), ('a', 1)])
        self.assertEqual((metrics['a']['attempted'], metrics['a']['refilled'], metrics['a']['padding']), (2, 0, 1))
        self.assertNotIn('replaced', [d for _, d in events])

    async def test_refill_never_overshoots_and_ignores_completion_order(self):
        # Proposal 1 is constant and finishes first while 2 (informative) is still running:
        # counting pending groups, one replacement is enough; both orders propose the same prompts.
        outcomes = {'a': [False, True, True, True, True]}
        slow_first = await self.refill(outcomes, {'a': 2}, 3, delays={('a', 1): .02})
        slow_second = await self.refill(outcomes, {'a': 2}, 3, delays={('a', 2): .02})
        for groups, metrics, events in (slow_first, slow_second):
            self.assertEqual(groups, [('a', 2), ('a', 3)])
            self.assertEqual(metrics['a']['attempted'], 3)
        self.assertEqual(sorted(slow_first[2]), sorted(slow_second[2]))

    async def test_refill_starts_before_a_slow_earlier_group_finishes(self):
        # Proposal 1 is a straggler; proposal 2 is uniform and fast. Its replacement must start
        # right away, not after the straggler (which in-order resolution waited for).
        timeline = []
        proposals = {'a': 0}
        def propose(route):
            proposals[route] += 1
            timeline.append(('proposed', proposals[route]))
            return route, proposals[route]
        async def execute(job):
            await asyncio.sleep(.05 if job[1] == 1 else .001)
            timeline.append(('finished', job[1]))
            return job
        def assess(job):
            usable = job[1] != 2
            return usable, [1., 0.] if usable else [0., 0.], [False, False]
        groups, metrics = await collect({'a': 2}, propose, execute, assess, inflight=4, refill_rounds=1)
        self.assertEqual(groups, [('a', 1), ('a', 3)])
        self.assertLess(timeline.index(('proposed', 3)), timeline.index(('finished', 1)))

    async def test_batch_and_proposals_do_not_depend_on_finishing_order(self):
        import random
        outcomes = {'a': [False, True, False, False, True, True, False, True, True, True],
                    'b': [False, False, False, True, False, True, True, True, True, True]}
        reference = None
        for seed in range(20):
            rng = random.Random(seed)
            delays = {(r, k): rng.choice([.0001, .001, .004]) for r in outcomes for k in range(1, 11)}
            groups, metrics, events = await self.refill(outcomes, {'a': 3, 'b': 2}, 2, delays=delays)
            for m in metrics.values():
                m.pop('collection_seconds')
            proposed = sorted(j for j, _ in events)
            result = (groups, metrics, proposed)
            if reference is None:
                reference = result
            self.assertEqual(result, reference)

    async def test_timeout_cancels_work(self):
        stopped = []
        async def execute(job):
            try:
                await asyncio.sleep(10)
            finally:
                stopped.append(job)
        with self.assertRaises(TimeoutError):
            await collect({'a': 1}, lambda r: r, execute, lambda _: None, inflight=2, timeout=.01)
        self.assertEqual(len(stopped), 1)

    async def test_reward_error_never_replaced_by_zero(self):
        async def execute(job):
            raise RuntimeError('judge unavailable')
        with self.assertRaisesRegex(RuntimeError, 'judge unavailable'):
            await collect({'a': 1}, lambda r: r, execute, lambda _: self.fail('must not assess error'))


if __name__ == '__main__':
    unittest.main()
