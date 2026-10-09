"""Distillation training paths (mixrl/mopd/README.md) on CPU: the loss, teacher scoring in the rollout, evaluation
and the resolved config, with simulated SGLang/teacher servers. Needs torch and slime (the pinned image)."""
import asyncio
import contextlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from slime_plugins.chimera_mixrl import distill
from slime_plugins.chimera_mixrl.objective import active_diagnostics

HAVE_TORCH = importlib.util.find_spec('torch') is not None


def backend(current, entropy=None):
    """slime's log-prob module, replaced: the loss's gradient path returns `current` (lp_set)."""
    module = types.ModuleType('slime.backends.megatron_utils.loss')
    module.get_log_probs_and_entropy = lambda *a, **kw: (None, {
        'log_probs': list(current), 'entropy': [entropy if entropy is not None else c.detach() * 0 for c in current]})
    module.get_rollout_top_p_logprob_kwargs = lambda args, batch: {}
    return module


@unittest.skipUnless(HAVE_TORCH, 'requires CPU torch')
class LossTests(unittest.TestCase):
    def setUp(self):
        self.args = types.SimpleNamespace(calculate_per_token_loss=False, kl_coef=0., entropy_coef=0.,
                                          context_parallel_size=1, tensor_model_parallel_size=1, rollout_temperature=1.)
        self.c = {'mode': 'distill', 'adv_clip': 5., 'domains': {'teacher_x': {}, 'teacher_y': {}},
                  'is_positive_bounds': [.2, 5.], 'is_negative_bounds': [.2, 5.]}

    def run_loss(self, batch, current, logits, sum_of_sample_mean=None):
        from slime_plugins.chimera_mixrl import objective, runtime
        with patch.dict(sys.modules, {'slime.backends.megatron_utils.loss': backend(current)}), \
                patch.object(runtime, 'config', return_value=self.c):
            return objective.distill_loss(self.args, batch, logits, sum_of_sample_mean or (lambda x: x.mean()))

    def example(self):
        """The 3-token example from the design discussion: the student writes `Paris` `.` `<end_of_turn>`.
        Vocabulary: 0 Paris, 1 Lyon, 2 Rome / newline, 3 '.', 4 '!', 5 <end_of_turn>; prompt token 1."""
        import torch
        probs = torch.tensor([[.50, .30, .15, .05 / 3, .05 / 3, .05 / 3],
                              [.0075, .0075, .0075, .90, .07, .0075],
                              [.005, .005, .28, .005, .005, .70],
                              [1 / 6] * 6])  # last row predicts past the answer: unused
        logits = probs.log()[None]
        tokens = torch.tensor([1, 0, 3, 5])
        lp_set = torch.tensor([.50 / .95, .90 / .97, .70 / .98]).log().requires_grad_()
        batch = {'unconcat_tokens': [tokens], 'total_lengths': [4], 'response_lengths': [3], 'loss_masks': [torch.ones(3)],
                 'rollout_log_probs': [torch.tensor([-.645, -.076, -.337])],
                 'teacher_log_probs': [torch.tensor([.80, .60, .95]).log()], 'advantages': [torch.zeros(3)]}
        return batch, lp_set, logits

    def test_the_worked_example_pushes_each_token_as_derived(self):
        import torch
        batch, lp_set, logits = self.example()
        value, metrics = self.run_loss(batch, [lp_set], logits)
        value.backward()
        # A = ln q - ln p_full: +0.470, -0.405, +0.305; gradient on lp_set = -ratio * A / 3.
        torch.testing.assert_close(lp_set.grad, torch.tensor([-.157, .135, -.102]), atol=1e-3, rtol=0)
        # With the candidate-set log-prob in A, every token would lose ln(mass) (Paris: +0.419 instead of +0.470).
        self.assertAlmostEqual(metrics['distill_kl'].item(), -(.470 - .405 + .305) / 3, places=2)
        self.assertEqual(metrics['distill_clipped'].item(), 0.)
        self.assertAlmostEqual(value.item(), (.303 - .031 + .103) / 3, places=2)

    def test_a_teacher_equal_to_the_student_moves_nothing(self):
        import torch
        batch, lp_set, logits = self.example()
        batch['teacher_log_probs'] = [torch.log_softmax(logits[0, :3], -1)[torch.arange(3), torch.tensor([0, 3, 5])]]
        value, metrics = self.run_loss(batch, [lp_set], logits)
        value.backward()
        torch.testing.assert_close(lp_set.grad, torch.zeros(3))
        self.assertAlmostEqual(metrics['distill_kl'].item(), 0., places=6)

    def test_clip_and_importance_mask(self):
        import torch
        batch, lp_set, logits = self.example()
        batch['teacher_log_probs'] = [torch.tensor([.001, .60, .95]).log()]  # A = ln(0.001/0.5) = -6.2 -> -5
        batch['rollout_log_probs'][0][1] = lp_set[1].item() - math.log(6)     # ratio 6: outside [0.2, 5]
        value, metrics = self.run_loss(batch, [lp_set], logits)
        value.backward()
        ratio0 = math.exp(lp_set[0].item() + .645)
        torch.testing.assert_close(lp_set.grad[:2], torch.tensor([5 * ratio0 / 3, 0.]), atol=1e-6, rtol=1e-5)
        self.assertAlmostEqual(metrics['distill_clipped'].item(), 1 / 3, places=6)
        self.assertAlmostEqual(metrics['importance_masked_fraction'].item(), 1 / 3, places=6)

    def test_full_vocab_log_probs_over_packed_sequences(self):
        import torch
        from slime_plugins.chimera_mixrl.objective import full_vocab_log_probs
        torch.manual_seed(0)
        logits = torch.randn(1, 12, 7)  # two sequences (5 and 6 tokens) and one padding row
        tokens = [torch.randint(7, (5,)), torch.randint(7, (6,))]
        out = full_vocab_log_probs(logits, tokens, [5, 6], [2, 4], chunk=3)
        reference = torch.log_softmax(logits[0], -1)
        torch.testing.assert_close(out[0], reference[[2, 3], tokens[0][3:]])
        torch.testing.assert_close(out[1], reference[[6, 7, 8, 9], tokens[1][2:]])

    def test_per_domain_kl_survives_the_global_reduction(self):
        import torch
        logits = torch.zeros(1, 7, 4)  # uniform: lp_full = ln 1/4 everywhere
        lp_set = [torch.full((2,), math.log(.25), requires_grad=True), torch.full((3,), math.log(.25), requires_grad=True)]
        batch = {'unconcat_tokens': [torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2, 3])], 'total_lengths': [3, 4],
                 'response_lengths': [2, 3], 'loss_masks': [torch.ones(2), torch.ones(3)],
                 'rollout_log_probs': [x.detach().clone() for x in lp_set],
                 'teacher_log_probs': [torch.full((2,), math.log(.5)), torch.tensor([math.log(.25)] * 2 + [-20.])],
                 'advantages': [torch.zeros(2), torch.ones(3)]}  # domain index 0, then 1
        per_sample = lambda x: sum(part.mean() for part in x.split([2, 3]))  # slime's sum of per-answer means
        _, metrics = self.run_loss(batch, lp_set, logits, per_sample)
        reduced = {f'train/{k}': v.item() / 2 for k, v in metrics.items()}  # / global batch size
        out = active_diagnostics(reduced)
        self.assertAlmostEqual(out['train/distill/teacher_x/kl'], math.log(.25) - math.log(.5), places=5)
        self.assertAlmostEqual(out['train/distill/teacher_y/kl'], (math.log(.25) + 20) / 3, places=5)
        self.assertAlmostEqual(out['train/distill/teacher_y/clipped'], 1 / 3, places=5)
        self.assertEqual(out['train/distill/teacher_x/clipped'], 0.)

    def test_refuses_a_mixrl_config_or_another_temperature(self):
        batch, lp_set, logits = self.example()
        self.args.rollout_temperature = .7
        with self.assertRaisesRegex(ValueError, 'temperature 1'):
            self.run_loss(batch, [lp_set], logits)
        self.args.rollout_temperature = 1.
        self.c.pop('mode')
        with self.assertRaisesRegex(ValueError, 'distillation config'):
            self.run_loss(batch, [lp_set], logits)


@unittest.skipUnless(HAVE_TORCH, 'requires torch and slime')
class RolloutTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from slime.utils.types import Sample
        from slime_plugins.chimera_mixrl import runtime
        self.runtime, self.Sample = runtime, Sample
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.c = {'mode': 'distill', 'run_dir': str(root / 'run'), 'context': 32, 'seed': 42, 'adv_clip': 5.,
                  'domains': {}, 'teachers': {}, 'quotas': {'teacher_x': 2, 'teacher_y': 1},
                  'caps': {'teacher_x': 8, 'teacher_y': 4}, 'eval_quotas': {'teacher_x': 1, 'teacher_y': 1},
                  'eval_samples': 1, 'routes': {}, 'inflight_groups': 4, 'response_concurrency': 4,
                  # MixRL settings that distillation ignores: no refills, spares, masks or length penalty.
                  'refill_rounds': 2, 'oversample': 1., 'truncation': 'mask', 'length_penalty': {'max_penalty': .1},
                  'collection_timeout': 5, 'reward_timeout': 1, 'reward_attempts': 2, 'reward_concurrency': 8}
        for i, name in enumerate(('teacher_x', 'teacher_y')):
            base = root / name
            base.mkdir()
            rows = [{'id': f'p{j}', 'messages': [{'role': 'user', 'content': f'{name} {j}'}]} for j in range(4)]
            rows[0]['max_response_tokens'] = 3
            (base / 'prompts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            (base / 'eval.jsonl').write_text(json.dumps({'id': 'e0', 'messages': [{'role': 'user', 'content': 'held'}]}) + '\n')
            self.c['domains'][name] = {'prompts': str(base / 'prompts.jsonl'), 'eval': str(base / 'eval.jsonl'),
                                       'prompts_hash': 'h', 'eval_hash': 'h'}
            self.c['teachers'][name] = {'url': f'http://teacher-{i}', 'server': name}
        self.args = types.SimpleNamespace(hf_checkpoint='fixture', save=str(root / 'ckpt'), load=None,
                                          n_samples_per_prompt=1, rollout_batch_size=3, partial_rollout=False,
                                          group_rm=False, use_rollout_routing_replay=False, rollout_top_p=1.,
                                          rollout_temperature=1., sglang_router_ip='router', sglang_router_port=1,
                                          rollout_stop=['<end_of_turn>'])
        self.requests, self.served = [], {'http://teacher-0': 'teacher_x', 'http://teacher-1': 'teacher_y'}
        self.shift = 0
        tokenizer = types.SimpleNamespace(apply_chat_template=lambda messages, **kw: messages[-1]['content'],
                                          encode=lambda prompt, **kw: [7, 8])
        processing = types.ModuleType('slime.utils.processing_utils')
        processing.load_tokenizer = lambda *a, **kw: tokenizer
        state = types.SimpleNamespace(semaphore=asyncio.Semaphore(8), sampling_params={},
                                      dp_rank_context=contextlib.nullcontext, reset=lambda: None)

        async def generate(args, sample, params):
            sample.response, sample.response_length = 'ok<end_of_turn>', 2
            sample.tokens = sample.tokens + [9, 10]
            sample.rollout_log_probs, sample.loss_mask = [-.5, -.7], [1, 1]
            sample.status = Sample.Status.COMPLETED if sample.metadata['mixrl']['row_id'][-1] != '1' else Sample.Status.TRUNCATED
            return sample

        async def abort(*args):
            return []
        generation = types.ModuleType('slime.rollout.sglang_rollout')
        generation.GenerateState, generation.generate, generation.abort = (lambda args: state), generate, abort

        def request(url, payload=None, timeout=1):
            self.requests.append((url, payload))
            base = url.removesuffix('/v1/models').removesuffix('/generate')
            if url.endswith('/v1/models'):
                return {'data': [{'id': self.served[base]}]}
            if payload['sampling_params']['max_new_tokens'] > 0:  # a teacher answering an eval prompt itself
                return {'meta_info': {'completion_tokens': 5, 'finish_reason': {'type': 'stop', 'matched': '<end_of_turn>'}}}
            tokens = payload['input_ids'][payload['logprob_start_len']:]
            value = {'http://router:1': -1.0}.get(base, -0.25)  # the student scores lower than the teachers
            return {'meta_info': {'input_token_logprobs': [[None if i == 0 else value, t + self.shift, None]
                                                           for i, t in enumerate(tokens)]}}
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.dict('sys.modules', {'slime.utils.processing_utils': processing,
                                                           'slime.rollout.sglang_rollout': generation}))
        self.stack.enter_context(patch.object(runtime, 'config', return_value=self.c))
        self.stack.enter_context(patch.object(runtime, 'request', side_effect=request))
        self.source = runtime.DataSource(self.args)

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    async def test_every_drawn_prompt_is_scored_by_its_domains_teacher(self):
        output = await self.runtime._rollout(self.args, 0, self.source)
        flat = [s for g in output.samples for s in g]
        self.assertEqual([s.metadata['mixrl']['task'] for s in flat], ['teacher_x', 'teacher_x', 'teacher_y'])
        # Ids are prefixed by domain; a prompt's own max_response_tokens overrides its domain's cap.
        caps = {s.metadata['mixrl']['row_id']: s.metadata['mixrl']['cap'] for s in flat}
        self.assertEqual({k: v for k, v in caps.items() if k.endswith(':p0')}, {k: 3 for k in caps if k.endswith(':p0')})
        self.assertTrue(all(v == 8 for k, v in caps.items() if k.startswith('teacher_x:') and not k.endswith(':p0')))
        for s in flat:
            self.assertEqual(s.teacher_log_probs, [-.25, -.25])
            self.assertEqual(s.loss_mask, [1, 1])  # capped answers train too (truncation=mask is ignored)
        scored = [(u, p) for u, p in self.requests if u.endswith('/generate')]
        self.assertEqual(len(scored), 3)  # no refills or spares, and training needs no student prefill
        self.assertTrue(all(p['sampling_params'] == {'temperature': 1.0, 'max_new_tokens': 0} and p['logprob_start_len'] == 1
                            for _, p in scored))
        raw, labels = self.runtime.post_process_rewards(self.args, flat)
        self.assertEqual((raw, labels), ([0.] * 3, [0., 0., 1.]))  # the advantage slot carries the domain index
        metrics = output.metrics
        self.assertEqual(metrics['mixrl/skip_optimizer'], 0)
        self.assertEqual(metrics['mixrl/teacher_x/length_mean'], 2)
        self.assertNotIn('mixrl/teacher_x/raw_reward_mean', metrics)

    async def test_misaligned_or_missing_teacher_log_probs_stop_the_step(self):
        self.shift = 1
        with self.assertRaisesRegex(RuntimeError, 'do not line up'):
            await self.runtime._rollout(self.args, 0, self.source)
        sample = self.Sample(index=0, prompt='', tokens=[7, 8, 9], response_length=1, metadata={'mixrl': {'task': 'teacher_x'}})
        with self.assertRaisesRegex(ValueError, 'one teacher log-prob per answer token'):
            self.runtime.post_process_rewards(types.SimpleNamespace(n_samples_per_prompt=1, rollout_batch_size=1), [sample])

    async def test_a_replaced_teacher_stops_the_run_before_generation(self):
        self.served['http://teacher-1'] = 'something_else'
        with self.assertRaisesRegex(RuntimeError, 'Teachers cannot score: teacher teacher_y'):
            await self.runtime._rollout(self.args, 0, self.source)
        self.assertFalse(any(u.endswith('/generate') for u, _ in self.requests))

    async def test_evaluation_compares_student_and_teacher_and_keeps_the_teachers_answers(self):
        printed = []
        with patch('builtins.print', lambda *a, **kw: printed.append(' '.join(map(str, a)))):
            output = await self.runtime._rollout(self.args, 0, self.source, evaluation=True)
            await self.runtime._rollout(self.args, 0, self.source, evaluation=True)
        summary = json.loads(next(p for p in printed if p.startswith('MIXRL_EVAL ')).split(' ', 1)[1])
        x = summary['domains']['teacher_x']
        self.assertEqual(summary['mode'], 'distill')
        self.assertAlmostEqual(x['kl'], -1.0 + .25)  # student prefill - teacher prefill, per token
        self.assertEqual((x['length_mean'], x['stop_rate'], x['teacher_length_mean'], x['teacher_stop_rate']), (2, 1., 5, 1.))
        self.assertAlmostEqual(output.data['kl']['rewards'][0], -.75)
        # The teachers answer the eval prompts once per run.
        answered = [p for _, p in self.requests if p and p['sampling_params']['max_new_tokens'] > 0]
        self.assertEqual(len(answered), 2)
        self.assertEqual(answered[0]['sampling_params']['stop'], ['<end_of_turn>'])
        self.assertTrue((Path(self.c['run_dir']) / 'rollouts' / 'teacher-eval.json').exists())


@unittest.skipUnless(HAVE_TORCH, 'requires the pinned image (configure imports the Chimera helpers)')
class ConfigTests(unittest.TestCase):
    def setUp(self):
        from tests.test_chimera_mixrl_distill import CONFIG, checkpoint, prompts
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        checkpoint(self.root / 'student' / 'hf')
        (self.root / 'student' / 'mcore').mkdir(parents=True)
        (self.root / 'student' / 'mcore' / 'latest_checkpointed_iteration.txt').write_text('1')
        for name, per_step in (('teacher_x', 6), ('teacher_y', 3)):
            base = self.root / 'domains' / name
            base.mkdir(parents=True)
            prompts(base / 'prompts.jsonl', 30, name)
            prompts(base / 'eval.jsonl', 5, name + '-eval')
            (base / 'domain.json').write_text(json.dumps({'prompts_per_step': per_step, 'eval_prompts': 2}))
            (base / 'teacher').symlink_to(self.root / 'student' / 'hf')  # teacher = student: one server
        self.env = {'MIXRL_DISTILL_DIR': str(self.root), 'DISTILL_ADV_CLIP': '5', 'ROLLOUT_TEMPERATURE': '1.0',
                    'MIXRL_TEACHER_URLS': 'teacher_x+teacher_y=http://127.0.0.1:8100'}
        self.served = {'http://127.0.0.1:8100': 'teacher_x+teacher_y'}

    def tearDown(self):
        self.temp.cleanup()

    def resolve(self):
        from slime_plugins.chimera_mixrl import configure
        def request(url, payload=None, timeout=1):
            if url.endswith('/get_model_info'):
                return {'model_path': '/teacher'}
            return {'data': [{'id': self.served[url.rsplit('/v1/models', 1)[0]], 'root': 'served name'}]}
        with patch.object(configure, 'request', side_effect=request):
            return configure.distill_inputs(self.env)[0]

    def test_folder_and_servers_become_the_trainers_config(self):
        c = self.resolve()
        self.assertEqual((c['mode'], c['quotas'], c['rollout_batch_size'], c['eval_quotas'], c['eval_samples']),
                         ('distill', {'teacher_x': 6, 'teacher_y': 3}, 9, {'teacher_x': 2, 'teacher_y': 2}, 1))
        self.assertEqual(c['teachers']['teacher_y'], {'url': 'http://127.0.0.1:8100', 'server': 'teacher_x+teacher_y',
                                                      'checkpoint': '/teacher'})
        self.assertEqual(len(c['domains']['teacher_x']['prompts_hash']), 64)

    def test_refuses_another_temperature_or_a_missing_or_wrong_server(self):
        self.env['ROLLOUT_TEMPERATURE'] = '0.8'
        with self.assertRaisesRegex(ValueError, 'ROLLOUT_TEMPERATURE=1'):
            self.resolve()
        self.env['ROLLOUT_TEMPERATURE'] = '1'
        self.served['http://127.0.0.1:8100'] = 'teacher_x'
        with self.assertRaisesRegex(ValueError, "serves \\['teacher_x'\\], expected teacher_x\\+teacher_y"):
            self.resolve()
        self.env['MIXRL_TEACHER_URLS'] = 'teacher_x=http://127.0.0.1:8100'
        with self.assertRaisesRegex(ValueError, 'No teacher server for domain\\(s\\) teacher_y'):
            self.resolve()


if __name__ == '__main__':
    unittest.main()
