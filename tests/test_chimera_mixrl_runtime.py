"""Real Slime Sample contracts with simulated SGLang/tokenizer/scoring boundaries.

Run in a torch environment (cached vLLM image is sufficient); no GPU required.
This intentionally does not claim a live Megatron/SGLang integration test.
"""
import asyncio
import contextlib
import importlib.util
import io
import json
import os
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
        self.assertEqual(self.judged, 4)  # the 2 truncated responses are scored 0 without a request

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
                  'inflight_groups': 2, 'response_concurrency': 4, 'refill_rounds': 0,
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

    async def test_cancelled_groups_are_stopped_in_sglang_before_returning(self):
        real, stopped = runtime.collect, []
        async def with_cancel(*a, **kw):
            groups, metrics = await real(*a, **kw)
            metrics['mcqa']['cancelled'] = 1  # a spare was cancelled while still generating
            return groups, metrics
        async def stop(args):
            stopped.append(True)
        with patch.object(runtime, 'collect', with_cancel), patch.object(runtime, 'stop_generation', stop):
            await runtime._rollout(self.args, 0, self.source)
        self.assertEqual(stopped, [True])
        with patch.object(runtime, 'stop_generation', stop):  # nothing cancelled: nothing to stop
            await runtime._rollout(self.args, 1, self.source)
        self.assertEqual(stopped, [True])

    async def test_stop_generation_aborts_every_engine_until_idle(self):
        aborted = []
        async def get(url):
            self.assertEqual(url, 'http://router:9/workers')
            return {'workers': [{'url': 'http://a'}, {'url': 'http://b'}]}
        async def until_idle(urls):
            aborted.extend(urls)
        args = types.SimpleNamespace(sglang_router_ip='router', sglang_router_port=9)
        with patch('slime.utils.http_utils.get', get), \
                patch('slime.backends.sglang_utils.server_control.abort_servers_until_idle', until_idle):
            await runtime.stop_generation(args)
        self.assertEqual(aborted, ['http://a', 'http://b'])

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

    async def test_training_responses_are_written_only_when_kept_and_eval_always_without_routes(self):
        rollouts = Path(self.c['run_dir']) / 'rollouts'
        with patch.dict(os.environ):
            os.environ.pop('MIXRL_KEEP_TRAIN_SAMPLES', None)  # default: summaries only
            await runtime._rollout(self.args, 0, self.source)
        names = sorted(p.name for p in (rollouts / 'train-0').iterdir())
        self.assertFalse([n for n in names if n.split('.')[0].isdigit()], names)
        self.assertIn('collection.jsonl', names)
        with patch.dict(os.environ, {'MIXRL_KEEP_TRAIN_SAMPLES': '1'}):
            await runtime._rollout(self.args, 1, self.source)
        self.assertTrue(any(p.stem.isdigit() for p in (rollouts / 'train-1').glob('*.json')))
        await runtime._rollout(self.args, 1, self.source, evaluation=True)
        saved = [json.loads(p.read_text()) for p in next(rollouts.glob('eval-*')).glob('[0-9]*.json')]
        self.assertTrue(saved)
        self.assertTrue(all(r.get('rollout_routed_experts') is None for r in saved))
        self.assertTrue(all(r.get('reward') is not None for r in saved))

    async def test_truncated_responses_score_zero_without_a_request(self):
        output = await runtime._rollout(self.args, 0, self.source)
        self.assertEqual(self.judged, 4)
        for group in output.samples:
            truncated = group[2]
            self.assertEqual(truncated.reward, 0.)
            self.assertEqual(truncated.metadata['grade']['passed'], False)
            self.assertEqual(truncated.metadata['grade']['components']['failure'], 'candidate_truncated')

    async def test_zero_policy_trains_on_unfinished_responses_as_wrong_answers(self):
        self.c['truncation'] = 'zero'
        output = await runtime._rollout(self.args, 0, self.source)
        flat = [s for g in output.samples for s in g]
        self.assertTrue(all(s.loss_mask == [1, 1] for s in flat))
        _, advantages = runtime.post_process_rewards(self.args, flat)
        self.assertGreater(advantages[0], 0)
        self.assertLess(advantages[2], 0)  # the truncated response is pushed down like any wrong answer

    async def test_length_penalty_from_collection_reaches_the_advantages(self):
        from slime.rollout import sglang_rollout
        from slime.utils.types import Sample
        from slime_plugins.chimera_mixrl.core import MIMO_LENGTH_PENALTY
        self.c.update(truncation='zero', length_penalty=dict(MIMO_LENGTH_PENALTY))
        async def generate(args, sample, params):
            # Two correct answers of 2 and 8 tokens and one wrong 2-token answer.
            number = sample.metadata['mixrl']['sample']
            length = 8 if number == 1 else 2
            sample.response = 'A' if number == 2 else 'B'
            sample.response_length = length
            sample.tokens += [9] * length
            sample.rollout_log_probs = [-.5] * length
            sample.loss_mask = [1] * length
            sample.status = Sample.Status.COMPLETED
            return sample
        with patch.object(sglang_rollout, 'generate', generate):
            output = await runtime._rollout(self.args, 0, self.source)
        expected = -.1 * ((.6 - .3) / .7) ** 1.5  # anchor 5 tokens, +60%
        for group in output.samples:
            self.assertEqual([s.reward for s in group], [1., 1., 0.])  # logged scores stay raw
            penalties = [s.metadata['length_penalty'] for s in group]
            self.assertEqual(penalties[0], 0.)
            self.assertAlmostEqual(penalties[1], expected)
        self.assertEqual(output.metrics['mixrl/mcqa/length_penalized'], 2)
        self.assertAlmostEqual(output.metrics['mixrl/mcqa/length_penalty_mean'], expected)
        flat = [s for g in output.samples for s in g]
        _, advantages = runtime.post_process_rewards(self.args, flat)
        self.assertGreater(advantages[0], advantages[1])
        self.assertGreater(advantages[1], 0)

    async def test_top_p_replay_requires_candidate_sets_and_eval_does_not_request_them(self):
        import torch
        from slime.rollout import sglang_rollout
        self.c['rollout_top_p'] = self.args.rollout_top_p = .95
        state_params = {'custom_params': {'return_top_p_token_ids': True}, 'top_p': .95}
        self.stack.enter_context(patch.dict(sglang_rollout.GenerateState(self.args).sampling_params, state_params))
        original = sglang_rollout.generate
        requested = []
        async def with_sets(args, sample, params):
            requested.append('custom_params' in params)
            sample = await original(args, sample, params)
            sample.rollout_top_p_token_ids = torch.tensor([9, 4, 10], dtype=torch.int32)
            sample.rollout_top_p_token_offsets = torch.tensor([0, 2, 3], dtype=torch.int32)
            return sample
        with patch.object(sglang_rollout, 'generate', with_sets):
            output = await runtime._rollout(self.args, 0, self.source)
            self.assertEqual(output.metrics['mixrl/mcqa/top_p_set_mean'], 1.5)
            self.assertTrue(all(requested))
            requested.clear()
            await runtime._rollout(self.args, 0, self.source, evaluation=True)
            self.assertFalse(any(requested))
        with self.assertRaisesRegex(RuntimeError, 'top-p candidate sets'):
            await runtime._rollout(self.args, 1, self.source)
        self.args.rollout_top_p = 1.
        with self.assertRaisesRegex(ValueError, 'Rollout sampling differs'):
            await runtime._rollout(self.args, 1, self.source)

    async def test_top_k_is_allowed_only_with_top_p_replay_and_must_match_config(self):
        import torch
        from slime.rollout import sglang_rollout
        original = sglang_rollout.generate
        async def with_sets(args, sample, params):
            sample = await original(args, sample, params)
            sample.rollout_top_p_token_ids = torch.tensor([9, 10], dtype=torch.int32)
            sample.rollout_top_p_token_offsets = torch.tensor([0, 1, 2], dtype=torch.int32)
            return sample
        self.c.update(rollout_top_p=.95, rollout_top_k=20)
        self.args.rollout_top_p, self.args.rollout_top_k = .95, 20
        with patch.object(sglang_rollout, 'generate', with_sets):
            await runtime._rollout(self.args, 0, self.source)
        for top_p, top_k in ((.95, 40), (1., 20)):
            self.c.update(rollout_top_p=top_p, rollout_top_k=20)
            self.args.rollout_top_p, self.args.rollout_top_k = top_p, top_k
            with self.subTest(top_p=top_p, top_k=top_k), self.assertRaisesRegex(ValueError, 'Rollout sampling differs'):
                await runtime._rollout(self.args, 1, self.source)

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
        self.assertEqual(self.judged, 4)

    @patch.dict(os.environ, {'MIXRL_KEEP_TRAIN_SAMPLES': '1'})
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



class ServiceRoutingTests(unittest.IsolatedAsyncioTestCase):
    def test_each_request_and_its_retries_go_to_one_reward_process(self):
        from slime_plugins.chimera_mixrl.core import digest
        c = {'scorer_url': 'http://a', 'scorer_urls': ['http://a', 'http://b', 'http://c']}
        ids = [digest(['run', i]) for i in range(300)]
        chosen = [runtime.scorer_url(c, i) for i in ids]
        self.assertEqual(chosen, [runtime.scorer_url(c, i) for i in ids])  # stable across retries
        counts = {u: chosen.count(u) for u in c['scorer_urls']}
        self.assertTrue(all(70 < n < 130 for n in counts.values()), counts)  # spread evenly
        self.assertEqual(runtime.scorer_url({'scorer_url': 'http://a'}, ids[0]), 'http://a')

    async def test_pipeline_line_reports_flow_and_loop_lag(self):
        import contextlib, io
        flow = {'gen_queued': 3, 'generating': 5, 'grade_queued': 1, 'grading': 2, 'done': 7, 'gen_tokens': 0}
        args = types.SimpleNamespace(sglang_router_ip='127.0.0.1', sglang_router_port=1)
        out = io.StringIO()
        with patch.dict(os.environ, {'MIXRL_JUDGE_METRICS_URL': 'http://127.0.0.1:1/metrics'}), contextlib.redirect_stdout(out):
            task = asyncio.create_task(runtime.pipeline_heartbeat(args, 4, 'train', flow, .05))
            flow['gen_tokens'] = 50
            await asyncio.sleep(.2)
            task.cancel()
        line = json.loads(next(l for l in out.getvalue().splitlines() if l.startswith('MIXRL_PIPELINE ')).split(' ', 1)[1])
        self.assertEqual((line['rollout_id'], line['phase'], line['generating'], line['grading']), (4, 'train', 5, 2))
        self.assertGreaterEqual(line['loop_lag_ms'], 0)
        self.assertNotIn('sglang', line)  # unreachable endpoints are left out, never fatal
        self.assertNotIn('judge', line)

    async def test_loop_profile_in_the_pipeline_line(self):
        import contextlib, io
        flow = {'gen_queued': 0, 'generating': 0, 'grade_queued': 0, 'grading': 0, 'done': 0, 'gen_tokens': 0}
        args = types.SimpleNamespace(sglang_router_ip='127.0.0.1', sglang_router_port=1)
        out = io.StringIO()
        with patch.dict(os.environ, {'MIXRL_PROFILE': '1', 'MIXRL_JUDGE_METRICS_URL': ''}), contextlib.redirect_stdout(out):
            task = asyncio.create_task(runtime.pipeline_heartbeat(args, 0, 'train', flow, .15))
            end = time.monotonic() + .4
            while time.monotonic() < end:  # keep the loop busy in a recognizable function
                sum(range(20000))
                await asyncio.sleep(0)
            task.cancel()
        lines = [json.loads(l.split(' ', 1)[1]) for l in out.getvalue().splitlines() if l.startswith('MIXRL_PIPELINE ')]
        profile = next(l['loop_profile'] for l in lines if l.get('loop_profile'))
        self.assertGreater(profile['busy_share'], 0)
        self.assertTrue(profile['top'])

    def test_judge_metrics_parsing(self):
        text = """# HELP vllm:num_requests_running x
vllm:num_requests_running{engine="0",model_name="j"} 12.0
vllm:num_requests_running{engine="1",model_name="j"} 8.0
vllm:num_requests_waiting{engine="0",model_name="j"} 3.0
vllm:kv_cache_usage_perc{engine="0",model_name="j"} 0.5
vllm:kv_cache_usage_perc{engine="1",model_name="j"} 0.7
"""
        class Response(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False
        with patch.object(runtime.urllib.request, 'urlopen', lambda url, timeout: Response(text.encode())):
            self.assertEqual(runtime._judge_load('http://j/metrics'), {'running': 20.0, 'waiting': 3.0, 'kv_usage': 0.6})


if __name__ == '__main__':
    unittest.main()
