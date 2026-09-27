"""Opt-in tiny random Chimera GPU check; NOT Megatron/TE or checkpoint parity."""

import argparse
import gc
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transformers-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    from slime_plugins.models.chimera import register_transformers
    from slime_plugins.models.chimera_context import PHASES
    from slime_plugins.chimera_mixrl.core import write_json

    register_transformers(args.transformers_root)
    from transformers.models.chimera import ChimeraConfig, ChimeraForCausalLM
    # Leave the existing local inference service alone. Fail rather than consume
    # its reserved memory; this smoke intentionally uses no optimizer state.
    torch.cuda.set_per_process_memory_fraction(.08)
    results = []
    for phase, (maximum, factor) in PHASES.items():
        torch.manual_seed(42)
        config = ChimeraConfig(num_hidden_layers=8, hidden_size=512, intermediate_size=2048,
            num_attention_heads=8, num_key_value_heads=2, head_dim=64,
            n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=256,
            first_k_dense_replace=2, last_k_dense_replace=0, n_shared_experts=0,
            shared_expert_intermediate_size=0, context_phase=phase,
            max_position_embeddings=maximum, router_load_balancing_type='none',
            router_bias_update_rate=0., rms_norm_eps=1e-5, use_cache=False)
        config._attn_implementation = 'eager'
        model = ChimeraForCausalLM(config).to(device='cuda', dtype=torch.bfloat16)
        for name, param in model.named_parameters():
            if name.endswith('gate.weight'):
                param.requires_grad_(False)
        tokens = torch.randint(10, config.vocab_size, (1, 16), device='cuda')
        positions = torch.arange(maximum - 16, maximum, device='cuda')[None]
        output = model(input_ids=tokens, position_ids=positions, use_cache=False)
        with torch.no_grad():
            repeated_logits = model(input_ids=tokens, position_ids=positions, use_cache=False).logits
        fixed_weight_error = (output.logits.detach().float() - repeated_logits.float()).abs().max().item()
        if fixed_weight_error != 0:
            raise RuntimeError(f'Fixed-weight repeat logits differ: {fixed_weight_error}')
        del repeated_logits
        loss = output.logits.float().square().mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if any(p.grad is not None for name, p in model.named_parameters() if name.endswith('gate.weight')):
            raise RuntimeError('Frozen HF router acquired gradients')
        if not any(p.grad is not None for name, p in model.named_parameters() if '.experts.' in name):
            raise RuntimeError('HF expert gradients missing')
        if not torch.isfinite(loss) or not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Nonfinite/missing local tiny-model gradients')
        # Capture just HF YaRN cos/sin, then change positions and replay. This
        # does not substitute for the separate Megatron TE attention graph gate.
        rotary = model.model.rotary_emb
        x = torch.zeros(1, 16, 64, device='cuda', dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                rotary(x, positions)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = rotary(x, positions)
        positions.copy_(torch.arange(16, device='cuda')[None])
        graph.replay()
        eager = rotary(x, positions)
        torch.cuda.synchronize()
        error = max((a.float() - b.float()).abs().max().item() for a, b in zip(captured, eager))
        if error != 0:
            raise RuntimeError(f'Local HF rotary graph replay mismatch: {error}')
        results.append(dict(phase=phase, factor=factor, loss=loss.item(),
                            gradient_tensors=len(gradients), hf_rotary_graph_max_error=error,
                            fixed_weight_repeat_logits_max_error=fixed_weight_error,
                            router_gradients_absent=True, expert_gradients_present=True))
        print(json.dumps(results[-1]), flush=True)
        del model, output, loss, gradients, rotary, x, graph, captured, eager
        gc.collect()
        torch.cuda.empty_cache()
    write_json(Path(args.output), {'scope': __doc__, 'gpu': torch.cuda.get_device_name(), 'phases': results})


if __name__ == '__main__':
    main()
