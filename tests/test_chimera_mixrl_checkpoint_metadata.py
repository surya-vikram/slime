import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import yaml

from slime_plugins.chimera_mixrl.checkpoint_metadata import initialize_lazy_optimizer_state, portable_save_args, write_metadata
from slime_plugins.models.chimera_context import resolve_context
from test_chimera_context import hf_config, mcore_config


class CheckpointMetadataTests(unittest.TestCase):
    def test_new_te_initializer_preserves_state_and_remainder_dtype_rule(self):
        from collections import defaultdict
        from unittest.mock import Mock, patch
        class Param:
            def __init__(self, dtype):
                self.dtype = dtype
        bf16, fp32, trained = Param('bf16'), Param('fp32'), Param('bf16')
        state = defaultdict(dict, {trained: {'moment': 7}})
        calls = []
        def native(param, store_param_remainders):
            calls.append((param, store_param_remainders))
            state[param]['master_param'] = 3
        inner = SimpleNamespace(param_groups=[{'params': [bf16, fp32, trained]}],
            state=state, initialize_state=native, store_param_remainders=True)
        old_callback = Mock(side_effect=AssertionError('old TE signature must not be used'))
        opt = SimpleNamespace(optimizer=inner, init_state_fn=old_callback,
            config=SimpleNamespace(use_precision_aware_optimizer=True))
        with patch.dict('sys.modules', {'torch': SimpleNamespace(bfloat16='bf16')}):
            initialize_lazy_optimizer_state(opt)
            initialize_lazy_optimizer_state(opt)
        self.assertEqual(calls, [(bf16, True), (fp32, False)])
        self.assertEqual(state[trained], {'moment': 7})
        old_callback.assert_not_called()

    def test_lazy_save_initializes_without_advancing_or_overwriting_state(self):
        from unittest.mock import Mock
        inner = SimpleNamespace(state={'new': {}, 'trained': {'moment': 7}}, step=Mock())
        def initialize(opt, config):
            for state in opt.state.values():
                if not state:
                    state.update(master_param=3, moment=0)
        child = SimpleNamespace(optimizer=inner, config=object(), init_state_fn=initialize)
        optimizer = SimpleNamespace(chained_optimizers=[child])
        initialize_lazy_optimizer_state(optimizer)
        initialize_lazy_optimizer_state(optimizer)
        self.assertEqual(inner.state, {'new': {'master_param': 3, 'moment': 0}, 'trained': {'moment': 7}})
        inner.step.assert_not_called()
        initialize_lazy_optimizer_state(None)

    def test_portable_save_restores_live_objects_even_on_failure(self):
        @dataclass
        class Eval:
            name: str
        original = [Eval('fixture')]
        args = SimpleNamespace(async_save=False, eval_datasets=original)
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            with portable_save_args(args):
                self.assertEqual(args.eval_datasets, [{'name': 'fixture'}])
                raise RuntimeError('fixture failure')
        self.assertIs(args.eval_datasets, original)
        args.async_save = True
        with self.assertRaises(ValueError):
            with portable_save_args(args):
                pass

    def test_runtime_phase_written_without_rewriting_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'imported'
            source.mkdir()
            (source / 'latest_checkpointed_iteration.txt').write_text('0')
            document = mcore_config('8k')
            document['model'].update(num_layers=8, hidden_size=512, ffn_hidden_size=2048,
                num_attention_heads=8, num_query_groups=2, kv_channels=64,
                num_moe_experts=8, moe_router_topk=2, moe_ffn_hidden_size=256,
                vocab_size=50176, layernorm_epsilon=1e-5, moe_z_loss_coeff=0.001)
            source_yaml = source / 'run_config.yaml'
            source_yaml.write_text(yaml.safe_dump(document))
            original = source_yaml.read_bytes()
            config = dict(model_profile='chimera', execution_mode='sync', chimera_model_size='tiny',
                          context=16384, weight_decay=0, expert_model_parallel_size=2,
                          checkpoint_context=resolve_context(hf_config('32k'), 'chimera',
                              sequence_cap=16384, mcore_root=source, allow_mcore_context_override=True))
            save = root / 'saved'
            iteration = save / 'iter_0000009'
            iteration.mkdir(parents=True)
            write_metadata(config, save, 9)
            self.assertEqual(source_yaml.read_bytes(), original)
            self.assertEqual((save / 'run_config.yaml').read_bytes(), (iteration / 'run_config.yaml').read_bytes())
            actual = yaml.safe_load((iteration / 'run_config.yaml').read_text())['model']
            self.assertEqual(actual['seq_length'], 32768)
            self.assertEqual(actual['yarn_rotary_scaling_factor'], 4.)
            self.assertEqual(actual['moe_z_loss_coeff'], 0.001)  # source value, as Megatron's export requires
            self.assertEqual(actual['moe_aux_loss_coeff'], 0.)
            self.assertEqual(actual['moe_router_load_balancing_type'], 'none')
            self.assertEqual(actual['expert_model_parallel_size'], 2)
            self.assertTrue((iteration / 'chimera_mixrl_identity.json').is_file())
            config['chimera_model_size'] = 'full'
            with self.assertRaisesRegex(ValueError, 'architecture mismatch'):
                write_metadata(config, save, 9)


if __name__ == '__main__':
    unittest.main()
