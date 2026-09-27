"""No torch/Ray/SGLang required for the v0 contract tests."""
import asyncio
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
            max_attempts=20, event=lambda r, j, g, d: events.append((r, j, d)))
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
        self.assertEqual(first, second)

    async def test_constant_groups_do_not_trigger_replacement(self):
        async def execute(job):
            return job
        groups, metrics = await collect({'a': 1}, lambda r: r, execute,
                          lambda g: (False, [0., 0.], [False, False]), max_attempts=3)
        self.assertEqual(groups, ['a'])
        self.assertEqual(metrics['a']['attempted'], 1)
        self.assertEqual(metrics['a']['accepted'], 0)

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
