"""Opt-in FP32 output projection for the DP-only Chimera actor.

Parameters/checkpoints remain BF16. Only projection arithmetic and accumulation
are FP32. Preserve Megatron DDP's fused-main-grad hook contract.
"""
import types

import torch


class _FP32Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, fused):
        ctx.save_for_backward(inputs, weight)
        ctx.fused = fused
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return torch.nn.functional.linear(inputs.float(), weight.float())

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            grad = grad_output.reshape(-1, weight.shape[0]).float()
            x = inputs.reshape(-1, weight.shape[1]).float()
            dx = (grad @ weight.float()).reshape_as(inputs).to(inputs.dtype)
            if ctx.fused:
                if not hasattr(weight, 'main_grad') or weight.main_grad.dtype != torch.float32:
                    raise RuntimeError('FP32 Chimera head requires FP32 Megatron main_grad')
                weight.main_grad.addmm_(grad.t(), x)
                if hasattr(weight, 'grad_added_to_main_grad'):
                    weight.grad_added_to_main_grad = True
                    # Zero dummy triggers the DDP hook without double accumulation,
                    # including the zero_out_wgrad path used by Megatron.
                    dw = torch.zeros_like(weight)
                else:
                    dw = None
            else:
                dw = (grad.t() @ x).to(weight.dtype)
        return dx, dw, None


def install_fp32_head(model):
    """Install before DDP wrapping; do not replace or retype any parameter."""
    if not getattr(model, 'post_process', True):
        return
    layer = model.output_layer
    if getattr(layer, '_chimera_fp32_head', False):
        return
    config = layer.config
    if (config.tensor_model_parallel_size != 1 or config.pipeline_model_parallel_size != 1
            or config.context_parallel_size != 1 or layer.sequence_parallel
            or config.defer_embedding_wgrad_compute or layer.bias is not None):
        raise ValueError('Chimera FP32 head supports DP-only, no bias or deferred wgrad')

    def forward(self, input_, weight=None, runtime_gather_output=None):
        projection_weight = self.weight if weight is None else weight
        output = _FP32Projection.apply(input_, projection_weight,
                                      self.gradient_accumulation_fusion)
        if not getattr(self, '_chimera_precision_logged', False):
            print(f'CHIMERA_HEAD_RUNTIME input={input_.dtype} weight={projection_weight.dtype} '
                  f'output={output.dtype} tf32={torch.backends.cuda.matmul.allow_tf32} '
                  f'matmul_precision={torch.get_float32_matmul_precision()}', flush=True)
            self._chimera_precision_logged = True
        return output, None

    layer.forward = types.MethodType(forward, layer)
    layer._chimera_fp32_head = True
    print('CHIMERA_PRECISION: FP32 LM-head forward/backward; original parameter dtype; DP-only', flush=True)
