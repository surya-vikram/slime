"""Local GPU test of the pinned native Megatron router and Slime R3 hook.

Canonical tiny geometry, not a full model/SGLang/TE-attention parity claim.
Run inside the pinned Slime image with PYTHONPATH pointing to this checkout.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    os.environ['ENABLE_ROUTING_REPLAY'] = '1'
    os.environ['ROUTING_REPLAY_STAGE'] = 'record'
    import torch
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.moe.router import TopKRouter
    from slime.utils.routing_replay import RoutingReplay
    from slime_plugins.chimera_mixrl.routing import before_train_step, after_train_step
    from slime_plugins.chimera_mixrl.core import write_json

    torch.manual_seed(42)
    rendezvous = tempfile.TemporaryDirectory(prefix='chimera-router-')
    torch.distributed.init_process_group('nccl', rank=0, world_size=1,
                                        init_method='file://' + str(Path(rendezvous.name) / 'rdzv'))
    config = TransformerConfig(
        num_layers=8, hidden_size=512, num_attention_heads=8, num_query_groups=2,
        kv_channels=64, ffn_hidden_size=2048, num_moe_experts=8,
        moe_ffn_hidden_size=256, moe_router_topk=2, moe_router_score_function='sigmoid',
        moe_router_topk_scaling_factor=2.5, moe_router_dtype='fp32',
        moe_router_enable_expert_bias=True, moe_router_bias_update_rate=0.,
        moe_router_load_balancing_type='none', moe_aux_loss_coeff=0.,
        moe_z_loss_coeff=0., moe_router_fusion=False, add_bias_linear=False,
        params_dtype=torch.float32, use_cpu_initialization=True)
    world = torch.distributed.group.WORLD
    groups = SimpleNamespace(tp=world, cp=world, tp_cp=world, tp_dp_cp=world)
    model = torch.nn.Module()
    model.router = TopKRouter(config, pg_collection=groups).cuda()
    model.router.weight.requires_grad_(False)
    # Trainable expert-output surrogate isolates the native router from dispatch
    # kernels; full grouped-GEMM model validation is a separate gate.
    model.experts = torch.nn.Linear(8, 4, bias=False).cuda()
    options = SimpleNamespace(moe_router_bias_update_rate=0.,
                              moe_router_load_balancing_type='none',
                              use_rollout_routing_replay=True, moe_router_fusion=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
    x = torch.randn(16, 1, 512, device='cuda', requires_grad=True)
    original_weight = model.router.weight.detach().clone()
    original_bias = model.router.expert_bias.clone()
    original_experts = model.experts.weight.detach().clone()
    before_train_step(options, 0, 0, [model], optimizer, None)
    probs, assignment = model.router(x)
    replay = model.router.routing_replay
    torch.cuda.synchronize()
    recorded = replay.top_indices_list[0].to(device='cuda', dtype=torch.int64)
    reference = torch.sigmoid(torch.nn.functional.linear(x.reshape(-1, 512), original_weight))
    selected = reference.gather(1, recorded)
    expected = torch.zeros_like(reference).scatter(1, recorded, selected / selected.sum(-1, keepdim=True) * 2.5)
    torch.testing.assert_close(probs, expected, rtol=1e-5, atol=1e-6)
    os.environ['ROUTING_REPLAY_STAGE'] = 'replay_forward'
    repeated, repeated_map = model.router(x)
    torch.testing.assert_close(repeated, probs, rtol=0, atol=0)
    assert torch.equal(repeated_map, assignment)
    # Simulate recomputation with changed hidden states: replay must retain the
    # behavior path even when fresh argmax would choose different experts.
    os.environ['ROUTING_REPLAY_STAGE'] = 'fallthrough'
    _, fresh_map = model.router(-x)
    assert not torch.equal(fresh_map, assignment)
    os.environ['ROUTING_REPLAY_STAGE'] = 'replay_backward'
    recomputed, replayed_map = model.router(-x)
    assert torch.equal(replayed_map, assignment)
    RoutingReplay.assert_all_consumed()
    loss = model.experts(recomputed).square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert model.router.weight.grad is None
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    after_train_step(options, 0, 0, [model])
    assert torch.equal(original_weight, model.router.weight)
    assert torch.equal(original_bias, model.router.expert_bias)
    assert not torch.equal(original_experts, model.experts.weight)
    result = dict(scope=__doc__, gpu=torch.cuda.get_device_name(), tokens=16,
                  experts=8, topk=2, native_sigmoid_weight_max_error=(probs - expected).abs().max().item(),
                  fixed_weight_replay_max_error=(probs - repeated).abs().max().item(),
                  changed_input_route_preserved=True, frozen_router_and_bias=True,
                  upstream_gradient=True, expert_updated=True, gradient_norm=float(norm),
                  forward_backward_records_consumed=True)
    write_json(Path(args.output), result)
    print(json.dumps(result), flush=True)
    RoutingReplay.clear_all()
    torch.distributed.destroy_process_group()
    rendezvous.cleanup()


if __name__ == '__main__':
    main()
