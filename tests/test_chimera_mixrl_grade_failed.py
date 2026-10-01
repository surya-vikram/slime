"""Ungradable responses: dropped from training (refilled or zero-loss padding), left out of eval.

Pure Python: a minimal stand-in for slime.utils.types.Sample replaces the torch import.
"""
import asyncio
import enum
import http.server
import json
import os
import socketserver
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch


class _Status(enum.Enum):
    COMPLETED = 'completed'
    TRUNCATED = 'truncated'


class _Sample:
    Status = _Status

    def __init__(self, status=_Status.COMPLETED, grade=None, meta=None, length=4):
        self.status, self.response_length, self.loss_mask = status, length, None
        self.response, self.reward = 'answer', None
        self.metadata = {'mixrl': meta or {}}
        if grade is not None:
            self.metadata['grade'] = grade


def _install_fake_slime():
    names = ['slime', 'slime.utils', 'slime.utils.types']
    saved = {n: sys.modules.get(n) for n in names}
    for name in names:
        sys.modules[name] = types.ModuleType(name)
    sys.modules['slime.utils.types'].Sample = _Sample
    return saved


def _restore(saved):
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


SAVED = _install_fake_slime()
from slime_plugins.chimera_mixrl import core, routes, runtime  # noqa: E402


def tearDownModule():
    _restore(SAVED)


def valid(score):
    return {'status': 'valid', 'score': float(score), 'passed': score >= 1}


FAILED = {'status': runtime.GRADE_FAILED, 'error': 'ServiceHTTPError: HTTP 503: judge timed out'}


def group(route, group_id, grades, n=4):
    meta = {'identity': 'i', 'policy_version': 0, 'group_id': group_id, 'row_id': f'{route}-{group_id}',
            'split': 'rl_train', 'task': route}
    return [_Sample(grade=g, meta={**meta, 'sample': i}) for i, g in enumerate(grades[:n])]


class CollectTests(unittest.IsolatedAsyncioTestCase):
    async def run_collect(self, outcomes, quota, refill_rounds):
        """outcomes: per proposal, 'informative' | 'constant' | 'failed'."""
        proposals = iter(outcomes)
        events = []

        def propose(route):
            return next(proposals)

        async def execute(kind):
            return kind

        def assess(kind):
            if kind == 'failed':
                return False, [0., 0.], [False, False], True
            return kind == 'informative', [0., 1.], [False, False], False

        batch, metrics = await core.collect({'t': quota}, propose, execute, assess, inflight=8,
                                            refill_rounds=refill_rounds, timeout=10,
                                            event=lambda r, j, g, d: events.append(d))
        return batch, metrics['t'], events

    async def test_failed_group_is_refilled(self):
        batch, m, events = await self.run_collect(['informative', 'failed', 'informative'], 2, 1)
        self.assertEqual(batch, ['informative', 'informative'])
        self.assertEqual((m['grade_failed'], m['refilled'], m['accepted']), (1, 1, 2))
        self.assertIn('grade_failed_replaced', events)

    async def test_failed_group_pads_only_after_constant_groups(self):
        batch, m, _ = await self.run_collect(['failed', 'constant', 'informative'], 3, 0)
        self.assertEqual(batch, ['informative', 'constant', 'failed'])
        self.assertEqual((m['padding'], m['grade_failed']), (2, 1))

    async def test_failed_group_is_not_counted_in_scores(self):
        _, m, _ = await self.run_collect(['failed', 'informative'], 2, 0)
        self.assertEqual((m['responses'], m['score_sum'], m['attempted']), (2, 1., 2))


class PostProcessTests(unittest.TestCase):
    def test_failed_padding_group_has_no_loss_or_advantage(self):
        os.environ['CHIMERA_MIXRL_CONFIG'] = self.config_path
        good = group('t', 0, [valid(1), valid(0), valid(1), valid(0)])
        bad = group('t', 1, [valid(1), FAILED, valid(0), valid(1)])
        args = types.SimpleNamespace(n_samples_per_prompt=4, rollout_batch_size=2, grpo_std_normalization=False)
        raw, normalized = runtime.post_process_rewards(args, good + bad)
        self.assertEqual(raw[4:], [0.] * 4)
        self.assertEqual(normalized[4:], [0.] * 4)
        self.assertTrue(all(s.loss_mask == [0] * s.response_length for s in bad))
        self.assertTrue(any(a != 0 for a in normalized[:4]))
        # Only the graded group counts as informative: the MiMo batch scale is 2 / 1.
        self.assertAlmostEqual(abs(normalized[0]), abs(good[0].metadata['base_advantage'] * 2
                                                       * good[0].metadata['group_length_scale']))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.tmp.name, 'config.json')
        with open(self.config_path, 'w') as f:
            json.dump({'truncation': 'zero', 'objective': 'mimo'}, f)

    def tearDown(self):
        os.environ.pop('CHIMERA_MIXRL_CONFIG', None)
        self.tmp.cleanup()


class EvaluationSummaryTests(unittest.TestCase):
    def test_prompt_with_ungradable_response_is_left_out(self):
        row = {'task': 'a', 'domain': 'd', 'binary': True}
        graded = [_Sample(grade=valid(1)), _Sample(grade=valid(0))]
        ungraded = [_Sample(grade=valid(1)), _Sample(grade=FAILED)]
        summary = routes.evaluation_summary([(row, graded), (row, ungraded)])
        self.assertEqual(summary['tasks']['a']['prompts'], 1)
        self.assertEqual(summary['tasks']['a']['ungraded_prompts'], 1)
        self.assertEqual(summary['tasks']['a']['mean_score'], 0.5)
        self.assertEqual(summary['ungraded_prompts'], 1)

    def test_task_with_only_ungradable_prompts_is_reported_not_scored(self):
        a = {'task': 'a', 'domain': 'd', 'binary': True}
        b = {'task': 'b', 'domain': 'e', 'binary': True}
        summary = routes.evaluation_summary([(a, [_Sample(grade=valid(1))]), (b, [_Sample(grade=FAILED)])])
        self.assertEqual(summary['tasks']['b'], {'prompts': 0, 'ungraded_prompts': 1})
        self.assertNotIn('e', summary['domains'])
        self.assertEqual(summary['equal_domain_mean'], 1.)


class RewardExhaustionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls = 0
        test = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                test.calls += 1
                self.rfile.read(int(self.headers['Content-Length']))
                body = json.dumps({'error': 'EndpointError: judge timed out'}).encode()
                self.send_response(test.status)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.status = 503
        self.server = socketserver.ThreadingTCPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.tmp = tempfile.TemporaryDirectory()
        path = os.path.join(self.tmp.name, 'config.json')
        with open(path, 'w') as f:
            json.dump({'scorer_url': f'http://127.0.0.1:{self.server.server_address[1]}',
                       'scorer_protocol': 'p', 'reward_attempts': 3, 'reward_timeout': 5,
                       'reward_concurrency': 4}, f)
        self.env = patch.dict(os.environ, {'CHIMERA_MIXRL_CONFIG': path, 'MIXRL_REWARD_BACKOFF_MAX': '0'})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def sample(self):
        return _Sample(meta={'identity': 'i', 'split': 'rl_train', 'policy_version': 0, 'group_id': 0,
                             'row_id': 'r', 'row_hash': 'h', 'sample': 0, 'task': 't', 'binary': True})

    async def test_grading_fault_after_all_retries_marks_sample_ungraded(self):
        sample = self.sample()
        score = await runtime.reward(None, sample)
        self.assertEqual(self.calls, 3)
        self.assertEqual(score, 0.)
        self.assertTrue(runtime.grade_failed(sample))
        self.assertIn('judge timed out', sample.metadata['grade']['error'])

    async def test_client_error_still_stops(self):
        self.status = 400
        with self.assertRaises(runtime.ServiceHTTPError):
            await runtime.reward(None, self.sample())
        self.assertEqual(self.calls, 1)


if __name__ == '__main__':
    unittest.main()
