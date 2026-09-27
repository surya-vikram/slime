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
        args = SimpleNamespace(calculate_per_token_loss=False, kl_coef=0., entropy_coef=0.,
                               context_parallel_size=1)
        batch = dict(unconcat_tokens=[], total_lengths=[3], response_lengths=[2],
                     rollout_log_probs=[current.detach().clone()], advantages=[torch.tensor([1., -1.])])
        with patch.dict(sys.modules, {backend.__name__: backend}), patch.object(runtime, 'config', return_value={
                'is_positive_bounds': [.2, 5.], 'is_negative_bounds': [.2, 5.]}):
            value, metrics = loss(args, batch, None, lambda x: x.mean())
        value.backward()
        torch.testing.assert_close(current.grad, torch.tensor([-.5, .5]))
        self.assertEqual(metrics['train_rollout_logprob_abs_diff'].item(), 0.)
        self.assertEqual(metrics['importance_masked_fraction'].item(), 0.)

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


if __name__ == '__main__':
    unittest.main()
