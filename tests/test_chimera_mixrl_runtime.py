"""Real Slime Sample contracts with simulated SGLang/tokenizer/scoring boundaries.

Run in a torch environment (cached vLLM image is sufficient); no GPU required.
This intentionally does not claim a live Megatron/SGLang integration test.
"""
import asyncio
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

from slime_plugins.chimera_mixrl.core import digest, write_json
from slime_plugins.chimera_mixrl import runtime


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch; run in cached vLLM image on CPU')
class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_judging_does_not_hold_generation_slot(self):
        self.c['response_concurrency'] = 1
        original = runtime.reward
        all_generated = asyncio.Event()
        from slime.rollout import sglang_rollout
        original_generate = sglang_rollout.generate
        async def generate(*args):
            sample = await original_generate(*args)
            if self.generated == 6:
                all_generated.set()
            return sample
        async def delayed_reward(*args):
            await asyncio.wait_for(all_generated.wait(), 1)
            return await original(*args)
        with patch.object(sglang_rollout, 'generate', generate), patch.object(runtime, 'reward', delayed_reward):
            await runtime._rollout(self.args, 0, self.source)
        self.assertEqual(self.judged, 6)

    async def test_eval_seeds_stable_but_policy_cache_identity_changes(self):
        from slime.rollout import sglang_rollout
        original = sglang_rollout.generate
        seeds, identities = [], []
        async def generate(args, sample, params):
            seeds.append(params['sampling_seed'])
            identities.append(sample.metadata['mixrl']['policy_version'])
            return await original(args, sample, params)
        with patch.object(sglang_rollout, 'generate', generate):
            await runtime._rollout(self.args, 0, self.source, evaluation=True)
            await runtime._rollout(self.args, 8, self.source, evaluation=True)
        self.assertEqual(seeds[:2], seeds[2:])
        self.assertNotEqual(identities[:2], identities[2:])

    async def test_every_eval_uses_the_same_panel_and_samples(self):
        self.c['eval_samples'] = 3
        for rollout_id in (0, 1, 4, 99):
            await runtime._rollout(self.args, rollout_id, self.source, evaluation=True)
            reports = list((self.source.run_dir / 'rollouts').glob(f'eval-{rollout_id}-eval-*/evaluation.json'))
            self.assertEqual(len(reports), 1)
            summary = json.loads(reports[0].read_text())
            self.assertNotIn('tier', summary)
            self.assertEqual((summary['samples_per_prompt'], summary['tasks']['mcqa']['prompts']), (3, 1))

    async def test_unreachable_judge_stops_before_generation(self):
        self.c['routes']['mcqa']['judge'] = 'always'
        self.health['judge']['ready'] = False
        for evaluation in (False, True):
            with self.subTest(evaluation=evaluation), \
                    self.assertRaisesRegex(RuntimeError, "mcqa: needs the judge \\(always\\), but judge 'fixture-judge'"):
                await runtime._rollout(self.args, 0, self.source, evaluation=evaluation)
        self.assertEqual(self.generated, 0)
        self.health['judge']['ready'] = True
        self.health['task_errors'] = {'mcqa': 'No module named regex'}
        with self.assertRaisesRegex(RuntimeError, 'mcqa: grading check failed: No module named regex'):
            await runtime._rollout(self.args, 0, self.source)
        self.assertEqual(self.generated, 0)

    async def test_full_nine_domain_collection_and_evaluation(self):
        from slime_plugins.chimera_mixrl import tasks
        routes = tasks.resolved(tasks.load())['routes']
        self.c.update(quotas={t: 1 for t in routes}, caps={t: 8 for t in routes}, routes=routes,
                      eval_quotas={t: 1 for t in routes})
        self.args.rollout_batch_size = len(routes)
        manifest = {'splits': {}}
        for split in ('rl_train', 'rl_val'):
            rows = []
            for t, route in routes.items():
                rows.append(dict(id=f'{split}-{t}', family_id=f'{split}-{t}', task=t,
                                 domain=route['domain'], verifier=route['verifier'],
                                 binary=route['reward'] == 'binary',
                                 messages=[{'role': 'user', 'content': 'Question'}]))
            (self.data / 'splits' / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            manifest['splits'][split] = {'hash': digest(rows)}
        write_json(self.data / 'manifest.json', manifest)
        source = runtime.DataSource(self.args)
        output = await runtime._rollout(self.args, 0, source)
        self.assertEqual({g[0].metadata['mixrl']['task'] for g in output.samples}, set(routes))
        raw, normalized = runtime.post_process_rewards(self.args, [s for g in output.samples for s in g])
        self.assertEqual(len(raw), 48)
        evaluation = await runtime._rollout(self.args, 0, source, evaluation=True)
        self.assertEqual(evaluation.data['equal_domain_mean']['rewards'], [.5])

    async def test_pool_exhaustion_starts_a_logged_new_pass(self):
        # Eight fixture rows at two prompts per step: rollouts 0-3 are pass 1, rollout 4 starts pass 2.
        drawn, passes = [], []
        for rollout_id in range(5):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                output = await runtime._rollout(self.args, rollout_id, self.source)
            drawn.append([g[0].metadata['mixrl']['row_id'] for g in output.samples])
            passes.append(output.metrics['mixrl/mcqa/pass'])
            logged = [json.loads(line.split(' ', 1)[1]) for line in stream.getvalue().splitlines()
                      if line.startswith('MIXRL_PASS ')]
            expected = [{'rollout_id': 4, 'task': 'mcqa', 'pass': 2, 'pool': 8}] if rollout_id == 4 else []
            self.assertEqual(logged, expected)
        self.assertEqual(passes, [1, 1, 1, 1, 2])
        self.assertEqual(len({row for batch in drawn[:4] for row in batch}), 8)
        self.assertEqual(output.metrics['mixrl/mcqa/pass_progress'], 2 / 8)
        self.assertEqual(output.metrics['mixrl/mcqa/deferred'], 0)
        # Samples 1-2 of each group of 3 carry reasoning tags; sample 2 is truncated.
        self.assertAlmostEqual(output.metrics['mixrl/mcqa/think_rate'], 2 / 3)

    async def test_unfinished_reasoning_is_masked_like_truncation(self):
        original = runtime.request
        def request(url, payload=None, timeout=1):
            result = original(url, payload, timeout)
            if payload and payload['response']['text'] == 'B':
                result['grade'] = {'status': 'valid', 'score': 0., 'passed': False,
                                   'components': {'failure': 'unfinished_reasoning', 'incomplete': True}}
            return result
        with patch.object(runtime, 'request', side_effect=request):
            output = await runtime._rollout(self.args, 0, self.source)
        for group in output.samples:
            # Sample 0 is unfinished and sample 2 truncated: both masked; group of one left is constant.
            self.assertEqual([runtime.unfinished(s) for s in group], [True, False, True])
            self.assertTrue(all(s.loss_mask == [0, 0] for s in group))
        self.assertEqual(output.metrics['mixrl/mcqa/cap_rate'], 2 / 3)

    async def test_fresh_weight_import_does_not_restore_sampler(self):
        source = runtime.DataSource(self.args)
        self.args.load = '/original-hf-without-sampler'
        self.args.finetune = True
        source.load(-1)
        self.args.finetune = False
        with self.assertRaisesRegex(RuntimeError, 'sampler state'):
            source.load(0)

    async def test_admission_reserves_headroom_at_exact_boundary(self):
        self.c['context_headroom'] = 16
        self.c['context'] = max(self.c['caps'].values()) + 2 + 16
        source = runtime.DataSource(self.args)  # fixture prompt is exactly two tokens
        self.assertEqual(len(source), len(self.source))
        self.c['context'] -= 1
        with self.assertRaisesRegex(ValueError, 'no context-admitted'):
            runtime.DataSource(self.args)

    def setUp(self):
        from slime.utils.types import Sample
        self.Sample = Sample
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.generated = 0
        self.judged = 0
        self.fail_judge = False
        self.started = []
        self.finished = []
        rows = []
        for i in range(8):
            rows.append({'id': f'{i:064x}', 'family_id': str(i), 'task': 'mcqa', 'domain': 'knowledge',
                         'messages': [{'role': 'user', 'content': f'Question {i}'}],
                         'verifier': 'choice', 'verification': {'answer': 'B', 'labels': ['A', 'B']},
                         'source': 'fixture', 'binary': True})
        val = [dict(rows[0], id='f' * 64, family_id='heldout')]
        for name, records in (('rl_train', rows), ('rl_val', val)):
            path = self.data / 'splits' / f'{name}.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(r) + '\n' for r in records))
        write_json(self.data / 'manifest.json', {'splits': {name: {'hash': digest(rs)}
                   for name, rs in (('rl_train', rows), ('rl_val', val))}})
        self.c = {'data_dir': str(self.data), 'run_dir': str(self.root / 'run'),
                  'quotas': {'mcqa': 2}, 'caps': {'mcqa': 8}, 'context': 32, 'seed': 42,
                  'eval_quotas': {'mcqa': 1},
                  'routes': {'mcqa': {'domain': 'knowledge', 'verifier': 'choice', 'judge': 'none',
                                      'reward': 'binary'}},
                  'scorer_protocol': 'fixture', 'scorer_url': 'http://fixture', 'truncation': 'mask',
                  'inflight_groups': 2, 'response_concurrency': 4, 'max_attempts': 8,
                  'collection_timeout': 5, 'reward_timeout': 1, 'reward_attempts': 1, 'eval_samples': 2}
        self.args = types.SimpleNamespace(hf_checkpoint='fixture', save=str(self.root / 'ckpt'), load=None,
                  n_samples_per_prompt=3, rollout_batch_size=2, grpo_std_normalization=True,
                  partial_rollout=False, group_rm=False, use_rollout_routing_replay=False,
                  rollout_top_p=1., rollout_temperature=1.)
        tokenizer = types.SimpleNamespace(
            apply_chat_template=lambda messages, **kw: messages[-1]['content'],
            encode=lambda prompt, **kw: [7, 8])
        processing = types.ModuleType('slime.utils.processing_utils')
        processing.load_tokenizer = lambda *a, **kw: tokenizer
        state = types.SimpleNamespace(semaphore=asyncio.Semaphore(8), sampling_params={},
                                      dp_rank_context=contextlib.nullcontext, reset=lambda: None)
        async def generate(args, sample, params):
            self.generated += 1
            self.started.append(sample.index)
            await asyncio.sleep(.001 if sample.metadata['mixrl']['sample'] == 0 else .02)
            sample.response = 'B' if sample.metadata['mixrl']['sample'] == 0 else '<think>\nmaybe</think>\nA'
            sample.response_length = 2
            sample.tokens += [9, 10]
            sample.rollout_log_probs = [-.5, -.7]
            sample.loss_mask = [1, 1]
            sample.status = Sample.Status.COMPLETED if sample.metadata['mixrl']['sample'] < 2 else Sample.Status.TRUNCATED
            sample.weight_versions = ['fixture-1']
            self.finished.append(sample.index)
            return sample
        async def abort(*args):
            return []
        generation = types.ModuleType('slime.rollout.sglang_rollout')
        generation.GenerateState = lambda args: state
        generation.generate = generate
        generation.abort = abort
        self.health = {'protocol_id': 'fixture', 'judge': {'model': 'fixture-judge', 'ready': True},
                       'task_errors': {}}
        def request(url, payload=None, timeout=1):
            if payload is None:
                return self.health
            self.judged += 1
            if self.fail_judge:
                raise RuntimeError('fixture judge outage')
            score = float(payload['response']['text'] == 'B' and payload['response']['finish_reason'] == 'stop')
            return {'protocol_id': 'fixture', 'request_id': payload['request_id'],
                    'grade': {'status': 'valid', 'score': score, 'passed': bool(score)}}
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.dict('sys.modules', {'slime.utils.processing_utils': processing,
                                                           'slime.rollout.sglang_rollout': generation}))
        self.stack.enter_context(patch.object(runtime, 'config', return_value=self.c))
        self.stack.enter_context(patch.object(runtime, 'request', side_effect=request))
        self.source = runtime.DataSource(self.args)

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    async def test_constant_group_mask_and_global_normalization(self):
        self.c['objective'] = 'mimo'
        output = await runtime._rollout(self.args, 0, self.source)
        flat = [s for g in output.samples for s in g]
        _, before = runtime.post_process_rewards(self.args, flat)
        for s in output.samples[1]:
            s.metadata['grade']['score'] = 0.
        _, after = runtime.post_process_rewards(self.args, flat)
        self.assertEqual(after[:3], [2*x for x in before[:3]])
        self.assertEqual(after[3:], [0., 0., 0.])
        self.assertTrue(all(s.loss_mask == [0, 0] for s in output.samples[1]))

    async def test_all_constant_fixed_batch_signals_skip_without_resampling(self):
        def constant(url, payload=None, timeout=1):
            if payload is None:
                return {'protocol_id': 'fixture'}
            return {'protocol_id': 'fixture', 'request_id': payload['request_id'],
                    'grade': {'status': 'valid', 'score': 0., 'passed': False}}
        with patch.object(runtime, 'request', side_effect=constant):
            output = await runtime._rollout(self.args, 0, self.source)
        self.assertEqual(self.generated, 6)
        self.assertEqual(output.metrics['mixrl/skip_optimizer'], 1)
        self.assertEqual(self.source.sampler.group_index, 2)
        self.assertTrue(all(not any(s.loss_mask) for g in output.samples for s in g))

    async def test_full_collection_masks_and_resume(self):
        output = await runtime._rollout(self.args, 0, self.source)
        self.assertEqual(len(output.samples), 2)
        flat = [s for g in output.samples for s in g]
        raw, advantages = runtime.post_process_rewards(self.args, flat)
        self.assertEqual(raw, [1., 0., 0.] * 2)
        self.assertEqual(advantages[2], 0.)
        self.assertEqual(flat[2].loss_mask, [0, 0])
        # A response's grading begins while later sibling generation is in progress.
        self.assertLess(flat[0].metadata['reward_started_at'], flat[1].metadata['generation_finished_at'])
        self.source.save(0)
        restored = runtime.DataSource(self.args)
        self.args.load = self.args.save
        restored.load(0)
        self.assertEqual(restored.sampler.snapshot(), self.source.sampler.snapshot())
        self.assertTrue(all(s.metadata['mixrl']['split'] == 'rl_train' for s in flat))

    async def test_grading_strips_only_verified_terminal_eos_not_training_tokens(self):
        from slime.rollout import sglang_rollout
        original = sglang_rollout.generate
        async def with_eos(*args):
            sample = await original(*args)
            sample.response += '<EOS>'
            return sample
        self.source.tokenizer.eos_token = '<EOS>'
        self.source.tokenizer.eos_token_id = 10
        with patch.object(sglang_rollout, 'generate', with_eos):
            output = await runtime._rollout(self.args, 0, self.source)
        for group in output.samples:
            self.assertEqual(group[0].reward, 1.)
            self.assertEqual(group[0].metadata['grading_text'], 'B')
            self.assertEqual(group[0].response, 'B<EOS>')
            self.assertEqual(group[0].tokens[-1], 10)
            self.assertEqual(len(group[0].rollout_log_probs), 2)
            self.assertTrue(group[2].metadata['grading_text'].endswith('<EOS>'))

    async def test_grading_strips_the_end_of_turn_stop_string_not_training_tokens(self):
        # Chimera's chat turns end with <end_of_turn>; SGLang stops on it as a string and
        # (no_stop_trim) keeps it in the response. The 2026-09-27 H200 run graded it verbatim.
        from slime.rollout import sglang_rollout
        original = sglang_rollout.generate
        async def with_end_of_turn(*args):
            sample = await original(*args)
            sample.response += '<end_of_turn>'
            return sample
        self.args.rollout_stop = ['<end_of_turn>']
        with patch.object(sglang_rollout, 'generate', with_end_of_turn):
            output = await runtime._rollout(self.args, 0, self.source)
        for group in output.samples:
            self.assertEqual(group[0].reward, 1.)
            self.assertEqual(group[0].metadata['grading_text'], 'B')
            self.assertEqual(group[0].response, 'B<end_of_turn>')
            self.assertEqual(len(group[0].rollout_log_probs), 2)
            # A truncated response never reached the stop; its text is graded as generated.
            self.assertTrue(group[2].metadata['grading_text'].endswith('<end_of_turn>'))

    async def test_reward_concurrency_is_independent_and_bounded(self):
        self.c['reward_concurrency'] = 1
        original = runtime.request
        lock = threading.Lock()
        active = maximum = 0
        def slow_request(url, payload=None, timeout=1):
            nonlocal active, maximum
            if payload is None:
                return original(url, payload, timeout)
            with lock:
                active += 1
                maximum = max(maximum, active)
            try:
                time.sleep(.01)
                return original(url, payload, timeout)
            finally:
                with lock:
                    active -= 1
        with patch.object(runtime, 'request', side_effect=slow_request):
            await runtime._rollout(self.args, 0, self.source)
        self.assertEqual(maximum, 1)
        self.assertEqual(self.judged, 6)

    async def test_failed_batch_reuses_persisted_completions(self):
        before = self.source.sampler.snapshot()
        self.fail_judge = True
        with self.assertRaisesRegex(RuntimeError, 'outage'):
            await runtime._rollout(self.args, 0, self.source)
        self.assertEqual(before, self.source.sampler.snapshot())
        saved = list((self.root / 'run' / 'rollouts' / 'train-0').glob('[0-9]*.json'))
        self.assertTrue(saved)
        saved_ids = {json.loads(p.read_text())['index'] for p in saved}
        self.started.clear()
        self.fail_judge = False
        output = await runtime._rollout(self.args, 0, self.source)
        self.assertEqual(len(output.samples), 2)
        self.assertFalse(saved_ids & set(self.started), 'persisted responses must be regraded, not rerolled')

    async def test_eval_unfiltered_separate_from_training(self):
        before = self.source.sampler.snapshot()
        output = await runtime._rollout(self.args, 0, self.source, evaluation=True)
        self.assertEqual(output.data['mcqa']['rewards'], [1., 0.])
        self.assertEqual(before, self.source.sampler.snapshot())
        self.assertEqual(output.data['mcqa']['samples'][0].metadata['mixrl']['split'], 'rl_val')
        await runtime._rollout(self.args, 0, self.source, evaluation=True)
        self.assertTrue((self.root / 'run' / 'rollouts' / 'eval-0-eval-1').exists())

    async def test_failed_eval_reuses_same_draws(self):
        self.fail_judge = True
        with self.assertRaisesRegex(RuntimeError, 'outage'):
            await runtime._rollout(self.args, 0, self.source, evaluation=True)
        directory = self.root / 'run' / 'rollouts' / 'eval-0-eval-0'
        saved_ids = {json.loads(p.read_text())['index'] for p in directory.glob('[0-9]*.json')}
        self.started.clear()
        self.fail_judge = False
        await runtime._rollout(self.args, 0, self.source, evaluation=True)
        self.assertFalse(saved_ids & set(self.started))


if __name__ == '__main__':
    unittest.main()
