"""Analytic gradients and actual native reducer, CPU only."""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from slime_plugins.chimera_mixrl.objective import group_length_scales, masked_terms


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires CPU torch')
class ObjectiveTests(unittest.TestCase):
    def test_active_diagnostics_use_global_fraction_without_rescaling_loss(self):
        from slime_plugins.chimera_mixrl.objective import active_diagnostics
        metrics = {'train/effective_response_fraction': .25, 'train/importance_ratio': .25,
                   'train/entropy': .5, 'train/loss': -.02}
        active = active_diagnostics(metrics)
        self.assertEqual(active['train/active/importance_ratio'], 1.)
        self.assertEqual(active['train/active/entropy'], 2.)
        self.assertNotIn('train/active/loss', active)
        self.assertEqual(active_diagnostics({'train/effective_response_fraction': 0}), {})

    def test_actual_loss_wrapper_uses_behavior_and_reports_diagnostics(self):
        import sys
        import types
        import torch
        from unittest.mock import patch
        from slime_plugins.chimera_mixrl.objective import loss
        from slime_plugins.chimera_mixrl import runtime
        current = torch.tensor([-1., -2.], requires_grad=True)
        backend = types.ModuleType('slime.backends.megatron_utils.loss')
        backend.get_log_probs_and_entropy = lambda *a, **kw: (None, {
            'log_probs': [current], 'entropy': [torch.ones(2)]})
        backend.get_rollout_top_p_logprob_kwargs = lambda args, batch: {}
        args = SimpleNamespace(calculate_per_token_loss=False, kl_coef=0., entropy_coef=0.,
                               context_parallel_size=1)
        batch = dict(unconcat_tokens=[], total_lengths=[3], response_lengths=[2],
                     rollout_log_probs=[current.detach().clone()], advantages=[torch.tensor([1., -1.])],
                     loss_masks=[torch.ones(2)])
        with patch.dict(sys.modules, {backend.__name__: backend}), patch.object(runtime, 'config', return_value={
                'is_positive_bounds': [.2, 5.], 'is_negative_bounds': [.2, 5.]}):
            value, metrics = loss(args, batch, None, lambda x: x.mean())
        value.backward()
        torch.testing.assert_close(current.grad, torch.tensor([-.5, .5]))
        self.assertEqual(metrics['train_rollout_logprob_abs_diff'].item(), 0.)
        self.assertEqual(metrics['importance_masked_fraction'].item(), 0.)

    def test_loss_renormalizes_over_recorded_top_p_candidate_sets(self):
        import sys
        import types
        import torch
        from unittest.mock import patch
        from slime_plugins.chimera_mixrl.objective import loss
        from slime_plugins.chimera_mixrl import runtime
        current = torch.tensor([-1., -2.], requires_grad=True)
        received = {}
        def log_probs(*args, **kwargs):
            received.update(kwargs)
            return None, {'log_probs': [current], 'entropy': [torch.ones(2)]}
        backend = types.ModuleType('slime.backends.megatron_utils.loss')
        backend.get_log_probs_and_entropy = log_probs
        backend.get_rollout_top_p_logprob_kwargs = lambda args, batch: {
            'top_p_token_ids': batch['rollout_top_p_token_ids'], 'top_p_token_offsets': batch['rollout_top_p_token_offsets']}
        args = SimpleNamespace(calculate_per_token_loss=False, kl_coef=0., entropy_coef=0., context_parallel_size=1)
        batch = dict(unconcat_tokens=[], total_lengths=[3], response_lengths=[2],
                     rollout_log_probs=[current.detach().clone()], advantages=[torch.tensor([1., -1.])],
                     rollout_top_p_token_ids=[[9, 4, 10]], rollout_top_p_token_offsets=[[0, 2, 3]],
                     loss_masks=[torch.ones(2)])
        with patch.dict(sys.modules, {backend.__name__: backend}), patch.object(runtime, 'config', return_value={
                'is_positive_bounds': [.2, 5.], 'is_negative_bounds': [.2, 5.]}):
            loss(args, batch, None, lambda x: x.mean())
        self.assertEqual((received['top_p_token_ids'], received['top_p_token_offsets']), ([[9, 4, 10]], [[0, 2, 3]]))

    def test_detached_weight_and_sign_specific_mask(self):
        import torch
        current = torch.tensor([-1., -2., -3., -4.], requires_grad=True)
        behavior = (current.detach() - torch.tensor([1., 3., 2., .1]).log()).requires_grad_()
        advantages = torch.tensor([1., 1., -1., -1.], requires_grad=True)
        terms, ratio, keep = masked_terms(current, behavior, advantages, (.2, 2.), (.2, 3.))
        terms.sum().backward()
        torch.testing.assert_close(current.grad, torch.tensor([-1., 0., 2., 0.]))
        self.assertIsNone(behavior.grad)
        self.assertIsNone(advantages.grad)
        self.assertEqual(keep.tolist(), [True, False, True, False])

    def test_native_reducer_partition_and_dp_gradient_equivalence(self):
        import torch
        from typing import Callable
        path = Path(__file__).resolve().parents[1] / 'slime/backends/megatron_utils/cp_utils.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
                    and n.name == 'get_sum_of_sample_mean')
        env = dict(torch=torch, Callable=Callable,
                   mpu=SimpleNamespace(get_context_parallel_world_size=lambda: 1))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
        reducer = env['get_sum_of_sample_mean']
        lengths = [2, 5, 3, 4, 7, 1, 6, 2]
        masks = [torch.ones(n) for n in lengths]
        masks[2].zero_()  # capped sibling contributes neither loss nor denominator
        base = [1., -.5, 0., -.5, 1., -1., .5, -.5]
        scales = (group_length_scales(lengths[:4], [True, True, False, True]) +
                  group_length_scales(lengths[4:], [True] * 4))
        x = torch.linspace(-3, -1, sum(lengths), requires_grad=True)
        parts = x.split(lengths)
        ref = sum(sum((parts[i] * masks[i] * -base[i]).sum() for i in ids) /
                  sum(masks[i].sum() for i in ids) for ids in (range(4), range(4, 8))) / 2
        reference_grad = torch.autograd.grad(ref, x, retain_graph=True)[0]
        # Unequal lengths, arbitrary sample order, two ranks, two microbatches.
        ranks = [[[0, 5], [2, 7]], [[1, 6], [3, 4]]]
        loss = 0
        for microbatches in ranks:
            for ids in microbatches:
                reduce = reducer([lengths[i] + 2 for i in ids], [lengths[i] for i in ids],
                                 [masks[i] for i in ids], [masks[i].sum() for i in ids], False)
                terms = torch.cat([-parts[i] * base[i] * scales[i] for i in ids])
                # Native loss_func: *num_mbs/global_batch_size*dp_size;
                # Megatron accumulates /num_mbs then DP averages /dp_size.
                loss = loss + reduce(terms) * 2 / 8 * 2 / 2 / 2
        torch.testing.assert_close(loss, ref)
        torch.testing.assert_close(torch.autograd.grad(loss, x)[0], reference_grad)

    def test_masks_do_not_renormalize_denominator(self):
        self.assertEqual(group_length_scales([2, 6, 9], [True, True, False]), [.75, 2.25, 0.])

    def test_entire_masked_rank_does_not_dilute_informative_prompt(self):
        import torch
        from typing import Callable
        path = Path(__file__).resolve().parents[1] / 'slime/backends/megatron_utils/cp_utils.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
                    and n.name == 'get_sum_of_sample_mean')
        env = dict(torch=torch, Callable=Callable,
                   mpu=SimpleNamespace(get_context_parallel_world_size=lambda: 1))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
        x = torch.tensor([-1., -2., -3., -4.], requires_grad=True)
        loss = 0.
        for ids, active in (([0, 1], False), ([2, 3], True)):
            masks = [torch.tensor([float(active)]) for _ in ids]
            reduce = env['get_sum_of_sample_mean']([2, 2], [1, 1], masks,
                                                    [m.sum() for m in masks], False)
            terms = -x[ids] * torch.tensor([1., -1.]) * 2  # total groups / informative groups
            loss = loss + reduce(terms) / 4  # DP-scaled accumulation then DP average
        reference = (-x[2] + x[3]) / 2
        torch.testing.assert_close(loss, reference)
        torch.testing.assert_close(torch.autograd.grad(loss, x, retain_graph=True)[0],
                                   torch.autograd.grad(reference, x)[0])

    def test_frozen_router_preserves_upstream_and_expert_gradients(self):
        import torch
        from slime_plugins.chimera_mixrl import routing
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.router = torch.nn.Linear(3, 2, bias=False)
                self.router.register_buffer('expert_bias', torch.zeros(2))
                self.experts = torch.nn.Linear(2, 1, bias=False)
                self.router.weight.requires_grad_(False)
            def forward(self, x):
                return self.experts(self.router(x))
        model = Tiny()
        args = SimpleNamespace(moe_router_bias_update_rate=0., moe_router_load_balancing_type='none')
        routing._snapshots.clear()
        routing.before_train_step(args, 0, 0, [model], None, None)
        x = torch.ones(1, 3, requires_grad=True)
        model(x).sum().backward()
        self.assertIsNone(model.router.weight.grad)
        self.assertIsNotNone(model.experts.weight.grad)
        self.assertIsNotNone(x.grad)
        routing.before_train_step(args, 1, 0, [model], None, None)
        model.router.expert_bias.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            routing.before_train_step(args, 2, 0, [model], None, None)

    def test_router_hooks_report_whole_step_expert_balance(self):
        import contextlib
        import io
        import json
        import os
        import sys
        import types
        import torch
        from unittest.mock import patch
        from slime_plugins.chimera_mixrl import routing

        class Router(torch.nn.Module):
            topk = 1

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(4, 2), requires_grad=False)
                self.register_buffer('expert_bias', torch.zeros(4))

            def forward(self, choice):
                routing_map = torch.nn.functional.one_hot(choice, 4).bool()
                return routing_map.float(), routing_map

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.router, self.experts = Router(), torch.nn.Linear(2, 2)

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()

        model = torch.nn.Module()
        model.decoder = torch.nn.Module()
        model.decoder.layers = torch.nn.ModuleList([Layer(), Layer()])
        args = SimpleNamespace(moe_router_bias_update_rate=0., moe_router_load_balancing_type='none',
                               num_steps_per_rollout=1)
        from slime_plugins.chimera_mixrl import objective
        objective.GAP.clear()  # no loss ran in this step: no consistency line
        logged = []
        observability = types.ModuleType('slime.observability')
        observability.logging_utils = types.SimpleNamespace(log=lambda a, metrics, step_key: logged.append(metrics))
        routing._snapshots.clear()
        with patch.dict(os.environ, {'MIXRL_ROUTER_METRICS': '1'}), patch.dict(sys.modules, {
                'slime.observability': observability}):
            routing.before_train_step(args, 3, 0, [model], None, None)
            first, second = (layer.mlp.router for layer in model.decoder.layers)
            for choices in ([0, 0, 1], [0, 2]):  # two microbatches
                first(torch.tensor(choices))
            second(torch.tensor([1, 2, 3, 0, 1, 2, 3, 0]))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                routing.after_train_step(args, 3, 0, [model])
        (line,) = [l for l in output.getvalue().splitlines() if l.startswith('MIXRL_ROUTER ')]
        summary = json.loads(line.split(' ', 1)[1])
        self.assertEqual(summary['counts'], {'0': [3, 1, 1, 0], '1': [2, 2, 2, 2]})
        self.assertEqual(summary['layers']['0'], {'cv': 0.8718, 'peak': 2.4, 'cold': 0.25})
        self.assertEqual(summary['layers']['1'], {'cv': 0., 'peak': 1., 'cold': 0.})
        self.assertEqual((summary['worst_layer'], summary['peak_max']), ('0', 2.4))
        (router_metrics,) = [m for m in logged if 'router/cv_mean' in m]
        self.assertEqual(router_metrics['train/step'], 3)
        self.assertEqual(router_metrics['router/layer_0/cold'], 0.25)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires CPU torch')
class ConsistencyTests(unittest.TestCase):
    def test_logprob_gap_and_route_agreement_line(self):
        import contextlib
        import io
        import json
        import math
        import sys
        import types
        import torch
        from unittest.mock import patch
        from slime.utils import routing_replay
        from slime_plugins.chimera_mixrl import objective, routing
        objective.GAP.clear()
        # Two microbatches; masked tokens (third of the first) never count.
        objective.record_gap(torch.tensor([-1., -2., -9.]), torch.tensor([-1.05, -2., 0.]),
                             [torch.tensor([1, 1]), torch.tensor([0])])
        objective.record_gap(torch.tensor([-.5]), torch.tensor([-2.]), [torch.tensor([1])])
        routing_replay.ROUTE_AGREEMENT.update(sets=200, mismatched=torch.tensor(3))
        logged = []
        observability = types.ModuleType('slime.observability')
        observability.logging_utils = types.SimpleNamespace(log=lambda a, metrics, step_key: logged.append(metrics))
        output = io.StringIO()
        with patch.dict(sys.modules, {'slime.observability': observability}), contextlib.redirect_stdout(output):
            routing.report_consistency(SimpleNamespace(num_steps_per_rollout=1), 4, 0)
        line = json.loads(output.getvalue().split('MIXRL_CONSISTENCY ', 1)[1])
        self.assertEqual((line['tokens'], line['tokens_over_0.1'], line['tokens_over_1']), (3, 1, 1))
        self.assertAlmostEqual(line['logprob_abs_diff_max'], 1.5)
        self.assertAlmostEqual(line['logprob_abs_diff_mean'], (.05 + 1.5) / 3, places=6)
        self.assertAlmostEqual(line['prob_abs_diff_max'], math.exp(-.5) - math.exp(-2.), places=6)
        self.assertEqual(line['routes_overridden_by_replay'], .015)
        self.assertEqual(logged[0]['train/step'], 4)
        self.assertEqual((objective.GAP, routing_replay.ROUTE_AGREEMENT), ({}, {'sets': 0, 'mismatched': 0, 'active': False}))


class RouterBalanceTests(unittest.TestCase):
    def test_mimo_load_statistics(self):
        from slime_plugins.chimera_mixrl.routing import load_balance, load_summary
        self.assertEqual(load_balance([3, 1, 1, 0]), {'cv': 0.8718, 'peak': 2.4, 'cold': 0.25})
        self.assertIsNone(load_balance([0, 0]))
        summary = load_summary({'0:decoder.layers.2.mlp.router': [4, 4], '0:decoder.layers.3.mlp.router': [8, 0]})
        self.assertEqual(summary['layers']['3'], {'cv': 1.0, 'peak': 2.0, 'cold': 0.5})
        self.assertEqual((summary['cv_mean'], summary['cold_mean'], summary['worst_layer']), (0.5, 0.25, '3'))


if __name__ == '__main__':
    unittest.main()
