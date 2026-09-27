import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from slime_plugins.chimera_mixrl.sglang_capture import capture, decoder_layer


class MappingTests(unittest.TestCase):
    def test_physical_not_compressed_layer_mapping(self):
        self.assertEqual(decoder_layer('model.layers.2.mlp.experts'), 2)
        self.assertEqual(decoder_layer('model.layers.24.mlp.experts'), 24)
        for name in ('model.layers.0.mlp.experts', 'model.layers.25.mlp.experts', 'experts.0'):
            with self.assertRaises(ValueError):
                decoder_layer(name)

    def test_capture_calls_global_capturer(self):
        calls = []
        capturer = types.SimpleNamespace(capture=lambda **kw: calls.append(kw))
        module = types.ModuleType('sglang.srt.state_capturer.routed_experts')
        module.get_global_experts_capturer = lambda: capturer
        with patch.dict('sys.modules', {module.__name__: module}):
            capture('model.layers.7.mlp.experts', 'fixture_ids')
            self.assertEqual(calls, [{'layer_id': 7, 'topk_indices': 'fixture_ids'}])
            module.get_global_experts_capturer = lambda: None
            capture('ignored_when_capture_disabled', None)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires CPU torch')
class PersistenceTests(unittest.TestCase):
    def test_tiny_geometry_roundtrip_and_cross_architecture_rejection(self):
        import torch
        from slime.utils.types import Sample
        from slime_plugins.chimera_mixrl.records import load_sample, save_sample
        sample = Sample(tokens=[1, 2, 3, 4], response='answer', response_length=2)
        sample.rollout_routed_experts = torch.arange(48, dtype=torch.int32).reshape(3, 8, 2) % 8
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'sample.json'
            save_sample(path, sample)
            restored = load_sample(path, num_layers=8, router_topk=2)
            torch.testing.assert_close(restored.rollout_routed_experts, sample.rollout_routed_experts)
            with self.assertRaisesRegex(ValueError, 'shape/codec'):
                load_sample(path)

    def test_router_frozen_through_update_and_bias_mutation_detected(self):
        import torch
        from slime_plugins.chimera_mixrl.routing import before_train_step, _snapshots
        module = torch.nn.Module()
        module.router = torch.nn.Linear(4, 8, bias=False)
        module.router.weight.requires_grad_(False)
        module.router.register_buffer('expert_bias', torch.zeros(8))
        module.experts = torch.nn.Linear(4, 4, bias=False)
        args = types.SimpleNamespace(moe_router_bias_update_rate=0,
                                     moe_router_load_balancing_type='none',
                                     use_rollout_routing_replay=True, moe_router_fusion=False)
        optimizer = torch.optim.AdamW(module.parameters(), lr=.01, weight_decay=0)
        original = module.router.weight.detach().clone()
        expert_original = module.experts.weight.detach().clone()
        try:
            for step in range(2):
                before_train_step(args, 0, step, [module], optimizer, None)
                module.experts(torch.ones(2, 4)).square().sum().backward()
                optimizer.step()
                optimizer.zero_grad()
                before_train_step(args, 0, step, [module], optimizer, None)
            self.assertTrue(torch.equal(module.router.weight, original))
            self.assertFalse(torch.equal(module.experts.weight, expert_original))
            module.router.expert_bias[0] = 1
            with self.assertRaisesRegex(RuntimeError, 'Frozen router/bias changed'):
                before_train_step(args, 0, 2, [module], optimizer, None)
        finally:
            _snapshots.pop((id(module),), None)

    def test_native_replay_uses_captured_experts_not_current_argmax(self):
        import os
        import torch
        from slime.utils import routing_replay
        selected = torch.tensor([[2, 0]])
        replay = types.SimpleNamespace(pop_forward=lambda: selected)
        def forbidden(*args, **kwargs):
            raise AssertionError('Fresh routing must not run during replay')
        wrapped = routing_replay.get_routing_replay_compute_topk(forbidden)
        with patch.object(routing_replay, 'ROUTING_REPLAY', replay), patch.dict(os.environ,
                ENABLE_ROUTING_REPLAY='1', ROUTING_REPLAY_STAGE='replay_forward'):
            weights, indices = wrapped(torch.tensor([[.2, .9, .1]]), 2)
        torch.testing.assert_close(indices, selected)
        torch.testing.assert_close(weights, torch.tensor([[.1, .2]]))

    def test_compressed_expert_path_roundtrip_and_corruption(self):
        import torch
        import json
        from slime.utils.types import Sample
        from slime_plugins.chimera_mixrl.records import load_sample, save_sample
        sample = Sample(tokens=[1, 2, 3, 4], response='answer', response_length=2)
        sample.rollout_routed_experts = torch.arange(300, dtype=torch.int32).reshape(3, 25, 4) % 32
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'sample.json'
            save_sample(path, sample)
            restored = load_sample(path)
            torch.testing.assert_close(restored.rollout_routed_experts, sample.rollout_routed_experts)
            record = json.loads(path.read_text())
            record['rollout_routed_experts']['shape'][0] = 7
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                load_sample(path)


if __name__ == '__main__':
    unittest.main()
