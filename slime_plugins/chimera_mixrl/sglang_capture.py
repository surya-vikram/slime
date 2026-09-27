"""Capture logical expert IDs from SGLang's Transformers MoE adapter.

Layer indices are physical decoder indices, NOT compressed MoE-layer indices.
The GPU copy occurs inside the MoE custom op so CUDA graphs capture it too.
"""


def decoder_layer(layer_name, num_layers=25):
    parts = layer_name.split('.')
    if len(parts) != 5 or parts[:2] != ['model', 'layers'] or parts[3:] != ['mlp', 'experts']:
        raise ValueError(f'Unexpected Chimera expert module path: {layer_name}')
    layer = int(parts[2])
    if num_layers not in (8, 25) or not 2 <= layer < num_layers:
        raise ValueError(f'Expected a Chimera MoE decoder layer, got {layer}')
    return layer


def capture(layer_name, topk_ids):
    from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

    capturer = get_global_experts_capturer()
    if capturer is not None:
        capturer.capture(layer_id=decoder_layer(layer_name, getattr(capturer, 'num_layers', 25)),
                         topk_indices=topk_ids)
