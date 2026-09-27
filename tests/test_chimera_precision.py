import unittest
import torch
from slime_plugins.models.chimera_precision import _FP32Projection


class ProjectionTests(unittest.TestCase):
    def test_runtime_registration_preserves_json_stdout(self):
        import contextlib
        import io
        import os
        import sys
        from pathlib import Path
        from types import ModuleType
        from unittest.mock import patch
        provider = ModuleType('slime_plugins.models.chimera')
        provider.register_transformers = lambda: None
        precision = ModuleType('slime_plugins.models.chimera_sglang_precision')
        called = []
        for name in ('install_rms_alignment', 'install_dense_swiglu_alignment', 'install_full_bf16_reduction'):
            def install(name=name):
                called.append(name)
                print('library initialization log')
            setattr(precision, name, install)
        script = Path(__file__).resolve().parents[1] / 'examples/chimera/runtime/sitecustomize.py'
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.dict(sys.modules, {provider.__name__: provider, precision.__name__: precision}), \
                patch.dict(os.environ, {'CHIMERA_MATCH_RMSNORM': '1', 'CHIMERA_MATCH_DENSE_SWIGLU': '1',
                                       'CHIMERA_SGLANG_FULL_BF16_REDUCTION': '1'}), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(script.read_text(), str(script), 'exec'), {})
        self.assertEqual(stdout.getvalue(), '')
        self.assertIn('library initialization log', stderr.getvalue())
        self.assertEqual(called, ['install_rms_alignment', 'install_dense_swiglu_alignment',
                                  'install_full_bf16_reduction'])

    def test_runtime_experimental_options_default_off(self):
        import os
        import sys
        from pathlib import Path
        from types import ModuleType
        from unittest.mock import patch
        provider = ModuleType('slime_plugins.models.chimera')
        provider.register_transformers = lambda: None
        # Any attempted precision import/installer lookup will fail.
        precision = ModuleType('slime_plugins.models.chimera_sglang_precision')
        script = Path(__file__).resolve().parents[1] / 'examples/chimera/runtime/sitecustomize.py'
        with patch.dict(sys.modules, {provider.__name__: provider, precision.__name__: precision}), \
                patch.dict(os.environ, {}, clear=True):
            exec(compile(script.read_text(), str(script), 'exec'), {})

    def test_full_precision_forward_and_accumulation(self):
        torch.manual_seed(19)
        x = torch.randn(2, 5, 16, dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(31, 16, dtype=torch.bfloat16, requires_grad=True)
        w.main_grad = torch.zeros_like(w, dtype=torch.float32)
        w.grad_added_to_main_grad = False
        direction = torch.randn(2, 5, 31)
        expected_dw = direction.reshape(-1, 31).t() @ x.detach().float().reshape(-1, 16)
        expected_dx = (direction @ w.detach().float()).to(x.dtype)
        for _ in range(2):
            output = _FP32Projection.apply(x, w, True)
            self.assertEqual(output.dtype, torch.float32)
            torch.testing.assert_close(output, torch.nn.functional.linear(x.float(), w.float()), rtol=0, atol=0)
            output.backward(direction)
        torch.testing.assert_close(w.main_grad, 2 * expected_dw)
        torch.testing.assert_close(x.grad, 2 * expected_dx)
        self.assertTrue(w.grad_added_to_main_grad)
        self.assertEqual(torch.count_nonzero(w.grad), 0)

    def test_unfused(self):
        x = torch.randn(3, 4, requires_grad=True)
        w = torch.randn(5, 4, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda a, b: _FP32Projection.apply(a, b, False), (x, w),
            eps=1e-3, atol=1e-3, rtol=1e-2))


class RolloutPrecisionTests(unittest.TestCase):
    def test_adapter_scoped_dense_kernel_and_bf16_reduction_preserve_parameters(self):
        import contextlib
        import io
        import sys
        from types import ModuleType, SimpleNamespace
        from unittest.mock import patch
        from slime_plugins.models.chimera_sglang_precision import (
            install_dense_swiglu_alignment, install_full_bf16_reduction,
        )

        class ChimeraMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
                self.up_proj = torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
                self.down_proj = torch.nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)

            def forward(self, x):
                return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

        class Base:
            def __init__(self, config):
                self.model = torch.nn.Module()
                self.model.mlp = ChimeraMLP()
                self.model.shared_experts = ChimeraMLP()
                self.model.fused_moe = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
                self.initial_parameters = [(id(p), p.detach().clone()) for p in self.model.parameters()]

        adapter = SimpleNamespace(TransformersBase=Base)
        models = ModuleType('sglang.srt.models')
        models.transformers = adapter
        modeling = ModuleType('transformers.models.chimera.modeling_chimera')
        modeling.ChimeraMLP = ChimeraMLP
        activation = ModuleType('sglang.jit_kernel.activation')
        calls = []

        def fused_stub(gate_up):
            calls.append((gate_up.shape, gate_up.dtype))
            gate, up = gate_up.chunk(2, dim=-1)
            return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate_up.dtype)

        activation.silu_and_mul = fused_stub
        config = SimpleNamespace(model_type='chimera', hidden_act='silu')
        original_flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            with patch.dict(sys.modules, {models.__name__: models, modeling.__name__: modeling,
                                          activation.__name__: activation}), \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                install_dense_swiglu_alignment()
                wrapper = Base.__init__
                install_full_bf16_reduction()
                install_dense_swiglu_alignment()
                self.assertIs(wrapper, Base.__init__)  # one constructor hook
                self.assertTrue(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
                other = Base(SimpleNamespace(model_type='other', hidden_act='silu'))
                self.assertNotIn('forward', other.model.mlp.__dict__)
                self.assertTrue(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
                model = Base(config)
                self.assertFalse(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
                self.assertIn('forward', model.model.mlp.__dict__)
                self.assertNotIn('forward', model.model.shared_experts.__dict__)
                self.assertNotIn('forward', model.model.fused_moe.__dict__)
                for (identity, value), parameter in zip(model.initial_parameters, model.model.parameters()):
                    self.assertEqual(id(parameter), identity)
                    self.assertEqual(parameter.dtype, torch.bfloat16)
                    torch.testing.assert_close(parameter, value, rtol=0, atol=0)
                x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
                dense = model.model.mlp
                gate, up = dense.gate_proj(x), dense.up_proj(x)
                expected = dense.down_proj((torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype))
                torch.testing.assert_close(dense(x), expected, rtol=0, atol=0)
                self.assertEqual(calls, [(torch.Size([2, 3, 16]), torch.bfloat16)])
                self.assertNotIn('forward', ChimeraMLP().__dict__)  # standalone HF not patched
                with self.assertRaisesRegex(ValueError, 'silu/swish'):
                    Base(SimpleNamespace(model_type='chimera', hidden_act='gelu'))
            self.assertEqual(stdout.getvalue(), '')
            self.assertIn('native Transformer Engine reduction is unchanged', stderr.getvalue())
            self.assertIn('fused dense SwiGLU on 1 Chimera MLPs', stderr.getvalue())
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_flag


if __name__ == '__main__':
    unittest.main()
