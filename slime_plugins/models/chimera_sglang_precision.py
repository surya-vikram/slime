"""Chimera-only opt-in precision experiments for the SGLang HF adapter.

New dense-SwiGLU/reduction options are Tiny-qualified experiments; full trained
checkpoint qualification is pending. They never replace checkpoint parameters.
"""


def _install_rollout_precision(*, dense_swiglu=False, full_bf16_reduction=False):
    import functools
    import sys
    import types

    import torch
    from sglang.srt.models import transformers as adapter

    original = adapter.TransformersBase.__init__
    options = getattr(original, '_chimera_precision_options', None)
    if options is None:
        options = {'dense_swiglu': False, 'full_bf16_reduction': False}

        @functools.wraps(original)
        def initialize(self, config, *args, **kwargs):
            text_config = getattr(config, 'text_config', config)
            chimera = getattr(text_config, 'model_type', None) == 'chimera'
            if chimera and options['full_bf16_reduction']:
                # Set in the actual SGLang model process, before initialization
                # or graph capture. This process-wide PyTorch/cuBLAS preference
                # does not configure native Megatron Transformer Engine GEMMs.
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
                print('CHIMERA_PRECISION: SGLang PyTorch BF16 reduced-precision reduction=False; '
                      'native Transformer Engine reduction is unchanged', file=sys.stderr, flush=True)
            if chimera and options['dense_swiglu'] and getattr(text_config, 'hidden_act', None) not in ('silu', 'swish'):
                raise ValueError('Chimera dense SwiGLU alignment requires silu/swish activation')
            original(self, config, *args, **kwargs)
            if chimera and options['dense_swiglu']:
                from sglang.jit_kernel.activation import silu_and_mul
                from transformers.models.chimera.modeling_chimera import ChimeraMLP

                def dense_forward(module, x):
                    gate_up = torch.cat((module.gate_proj(x), module.up_proj(x)), dim=-1)
                    return module.down_proj(silu_and_mul(gate_up))

                count = 0
                for name, module in self.model.named_modules():
                    if name.rsplit('.', 1)[-1] == 'mlp' and isinstance(module, ChimeraMLP):
                        module.forward = types.MethodType(dense_forward, module)
                        count += 1
                if not count:
                    raise RuntimeError('No Chimera dense MLP found for SwiGLU alignment')
                print(f'CHIMERA_PRECISION: SGLang fused dense SwiGLU on {count} Chimera MLPs; '
                      'fused MoE kernels unchanged', file=sys.stderr, flush=True)

        initialize._chimera_precision_options = options
        adapter.TransformersBase.__init__ = initialize
    options['dense_swiglu'] |= dense_swiglu
    options['full_bf16_reduction'] |= full_bf16_reduction


def install_dense_swiglu_alignment():
    """Register only; patch surviving dense modules inside SGLang construction."""
    _install_rollout_precision(dense_swiglu=True)


def install_full_bf16_reduction():
    """Register a Chimera SGLang worker setting, not a global actor-side switch."""
    _install_rollout_precision(full_bf16_reduction=True)


def install_rms_alignment():
    from sglang.srt.models import transformers as adapter
    if getattr(adapter.replace_rms_norm_class, '_chimera_aligned', False):
        return
    original = adapter.replace_rms_norm_class

    def replace(norm, hidden_size):
        result = original(norm, hidden_size)
        if norm.__class__.__module__.startswith('transformers.models.chimera.'):
            if not hasattr(result, 'cast_x_before_out_mul'):
                raise RuntimeError('Unsupported SGLang RMSNorm: rounding flag missing')
            result.cast_x_before_out_mul = False
        return result

    replace._chimera_aligned = True
    adapter.replace_rms_norm_class = replace
    import sys
    print('CHIMERA_PRECISION: SGLang Chimera RMSNorm cast after weight multiplication', file=sys.stderr, flush=True)
