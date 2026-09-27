"""Execute native packing and sampler code on CPU; not a CUDA qualification."""
import ast
import copy
import importlib.util
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from collections.abc import Sequence
import unittest

ROOT = Path(__file__).resolve().parents[1]


def extract(path, names, env):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), env)
    return env


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires CPU torch')
class NativeSFTTests(unittest.TestCase):
    def test_native_sft_loss_gradient_excludes_masked_tokens(self):
        import torch
        from typing import Callable
        values = torch.tensor([-1., -2., -3., -4.], requires_grad=True)
        mask = torch.tensor([0., 1., 1., 0.])
        env = extract('slime/backends/megatron_utils/loss.py', {'sft_loss_function'}, dict(
            torch=torch, Namespace=SimpleNamespace, RolloutBatch=dict, Callable=Callable,
            get_log_probs_and_entropy=lambda *a, **kw: (None, {'log_probs': [values]})))
        batch = dict(unconcat_tokens=[], total_lengths=[6], response_lengths=[4])
        loss, metrics = env['sft_loss_function'](SimpleNamespace(), batch, values,
                                               lambda x: (x * mask).sum() / mask.sum())
        loss.backward()
        self.assertEqual(loss.item(), 2.5)
        self.assertFalse(metrics['loss'].requires_grad)
        torch.testing.assert_close(values.grad, torch.tensor([0., -.5, -.5, 0.]))

    def test_packed_boundaries_and_next_token_masks(self):
        import torch
        env = extract('slime/backends/megatron_utils/data.py', {'get_batch', 'DataIterator'}, dict(
            torch=torch, F=torch.nn.functional, Sequence=Sequence, RolloutBatch=dict,
            PackedSeqParams=SimpleNamespace,
            mpu=SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 1,
                get_context_parallel_world_size=lambda: 1, get_context_parallel_rank=lambda: 0),
            accelerator=SimpleNamespace(device=lambda: 'cpu', current_device=lambda: 'cpu'),
            slice_with_cp=lambda tensor, pad: tensor))
        # Final response includes EOS; earlier assistant turns are prompt, never targets.
        data = dict(tokens=[torch.tensor([1, 2, 3, 4, 9]), torch.tensor([5, 6, 7, 9])],
                    loss_masks=[torch.ones(2), torch.ones(1)],
                    total_lengths=[5, 4], response_lengths=[2, 1])
        for allgather in (False, True):
            batch = env['get_batch'](env['DataIterator'](data, [[0, 1]]), list(data),
                                     pad_multiplier=8, allgather_cp=allgather)
            self.assertEqual(batch['tokens'].shape, (1, 16))
            self.assertEqual(batch['packed_seq_params'].cu_seqlens_q.tolist(), [0, 5, 9, 16])
            self.assertEqual(batch['full_loss_masks'][0].tolist(),
                             [0, 0, 1, 1, 0, 0, 0, 1, 0] + [0] * 7)
            self.assertEqual(batch['full_loss_masks'].sum().item(), 3)

    def test_native_sampler_resume_exact_next_batch_and_unique_pass(self):
        import torch
        class Dataset:
            def __init__(self, *a, **kw):
                self.samples = [SimpleNamespace(prompt=str(i)) for i in range(8)]
            def __len__(self):
                return len(self.samples)
        env = extract('slime/rollout/data_source.py', {'RolloutDataSource'}, dict(
            torch=torch, os=os, copy=copy, Path=Path, DataSource=object,
            Dataset=Dataset, Sample=SimpleNamespace, logger=logging.getLogger(__name__),
            load_tokenizer=lambda *a, **kw: None, load_processor=lambda *a, **kw: None))
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(rollout_global_dataset=True, prompt_data='fixture', hf_checkpoint='',
                dump_details=None, rollout_max_prompt_len=None, input_key='messages', multimodal_keys=None,
                label_key=None, metadata_key='metadata', tool_key=None, apply_chat_template=False,
                apply_chat_template_kwargs={}, rollout_seed=42, rollout_shuffle=False,
                n_samples_per_prompt=1, save=directory, load=directory)
            source = env['RolloutDataSource'](args)
            first = source.get_samples(4)
            source.save(0)
            expected = source.get_samples(4)
            resumed = env['RolloutDataSource'](args)
            resumed.load(0)
            actual = resumed.get_samples(4)
            normalize = lambda groups: [(s.prompt, s.index, s.group_index) for g in groups for s in g]
            self.assertEqual(normalize(actual), normalize(expected))
            self.assertEqual(len(set(s.prompt for g in first + actual for s in g)), 8)
            self.assertEqual(resumed.epoch_id, 0)


if __name__ == '__main__':
    unittest.main()
