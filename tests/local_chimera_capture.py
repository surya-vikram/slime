"""GPU test: Chimera hook -> real SGLang capture buffers -> CUDA graph replay.

Tests buffer capture and causal-row retrieval, not full SGLang model execution.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    from sglang.srt.state_capturer.base import BaseTopkCapturer
    from sglang.srt.state_capturer.routed_experts import set_global_experts_capturer
    from slime_plugins.chimera_mixrl.sglang_capture import capture
    from slime_plugins.chimera_mixrl.core import write_json

    results = []
    for layers, topk, experts in ((8, 2, 8), (25, 4, 32)):
        capturer = BaseTopkCapturer(num_tokens=32, max_batch_size=8, num_layers=layers,
                                   topk_size=topk, device='cuda', name='chimera-test')
        set_global_experts_capturer(capturer)
        indices = torch.arange(4 * topk, device='cuda', dtype=torch.int32).reshape(4, topk) % experts
        def run():
            for layer in range(2, layers):
                capture(f'model.layers.{layer}.mlp.experts', indices)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        # Replaying must capture new IDs, not the warmup/capture-time values.
        indices.add_(1).remainder_(experts)
        capturer.device_cache.buffer.zero_()
        graph.replay()
        torch.cuda.synchronize()
        actual = capturer.device_cache.buffer[:4]
        assert torch.equal(actual[:, 2:], indices[:, None].expand(-1, layers - 2, -1))
        assert not actual[:, :2].any()
        positions = torch.tensor([7, 3, 11, 2], device='cuda')
        capturer.on_forward_end(SimpleNamespace(out_cache_loc=positions), True, 4)
        pool = SimpleNamespace(req_to_token=torch.tensor([[7, 3, 11, 2, 0]], device='cuda'))
        restored = capturer.get_topk(0, 5, pool)
        assert restored.shape == (4, layers, topk)
        assert torch.equal(restored, actual.cpu())
        results.append(dict(layers=layers, topk=topk, changed_ids_captured=True,
                            dense_layers_zero=True, causal_rows=4,
                            physical_layer_mapping=True, host_roundtrip_exact=True))
        del graph
    set_global_experts_capturer(None)
    report = dict(scope=__doc__, gpu=torch.cuda.get_device_name(), profiles=results)
    write_json(Path(args.output), report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
