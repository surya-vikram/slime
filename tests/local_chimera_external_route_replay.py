"""Replay real SGLang-captured Chimera routes through native Megatron R3.

The SGLang route JSON is produced by ``local_chimera_sglang_parity.py`` or an
equivalent live endpoint capture. This is a route-plumbing test, not a full
Transformer-layer/logit parity test.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.environ["ENABLE_ROUTING_REPLAY"] = "1"
    os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"

    import torch
    from megatron.core import mpu
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.transformer_config import TransformerConfig
    from slime.backends.megatron_utils.cp_utils import prepare_routed_experts_for_routing_replay
    from slime.utils import routing_replay
    from slime.utils.routing_replay import RoutingReplay
    from slime_plugins.chimera_mixrl.core import write_json

    record = json.loads(Path(args.routes_json).read_text())
    tokens = torch.tensor(record["all_tokens"], dtype=torch.long)
    routes = torch.tensor(record["routed_experts"], dtype=torch.int32)
    if routes.ndim != 3 or tuple(routes.shape[1:]) != (25, 4):
        raise ValueError(f"expected captured routes [tokens-1,25,4], got {tuple(routes.shape)}")
    if routes.shape[0] != tokens.numel() - 1:
        raise ValueError(f"route rows {routes.shape[0]} != token count - 1 ({tokens.numel()-1})")
    moe_routes = routes[:, 2:, :]
    if not bool(torch.all((moe_routes >= 0) & (moe_routes < 32))):
        raise ValueError("captured MoE expert id outside [0,32)")
    if not bool(torch.all(torch.sort(moe_routes, dim=-1).values[..., 1:] !=
                          torch.sort(moe_routes, dim=-1).values[..., :-1])):
        raise ValueError("captured top-4 IDs are not unique")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    world_size = torch.distributed.get_world_size()
    if world_size != 2:
        raise ValueError(f"this production-topology check requires EP=2; got world_size={world_size}")
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=2,
        create_gloo_process_groups=False,
    )

    # Same sequence packing helper the actor's fill_routing_replay uses.
    packed = prepare_routed_experts_for_routing_replay(
        [routes],
        [tokens],
        num_experts=32,
        data_pad_size_multiplier=128,
        sequence_parallel=False,
        allgather_cp=False,
    )
    expected_padded_rows = ((routes.shape[0] + 127) // 128) * 128
    if packed.shape != (expected_padded_rows, 25, 4):
        raise AssertionError(
            f"unexpected actor-aligned route shape {tuple(packed.shape)}; "
            f"expected ({expected_padded_rows},25,4)"
        )
    if not torch.equal(packed[: routes.shape[0], 2:, :], routes[:, 2:, :]):
        raise AssertionError("actor packing changed the captured causal-token routes")

    config = TransformerConfig(
        num_layers=25,
        hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=2,
        kv_channels=128,
        ffn_hidden_size=8192,
        num_moe_experts=32,
        moe_ffn_hidden_size=2048,
        moe_router_topk=4,
        moe_router_score_function="sigmoid",
        moe_router_topk_scaling_factor=2.5,
        moe_router_dtype="fp32",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=0.0,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_z_loss_coeff=0.0,
        moe_router_fusion=False,
        add_bias_linear=False,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
    )
    groups = SimpleNamespace(
        tp=mpu.get_tensor_model_parallel_group(),
        cp=mpu.get_context_parallel_group(),
        tp_cp=mpu.get_tensor_and_context_parallel_group(),
        tp_dp_cp=mpu.get_tensor_and_data_parallel_group(with_context_parallel=True),
    )
    router = TopKRouter(config, pg_collection=groups).cuda()
    router.weight.requires_grad_(False)
    for parameter in router.parameters():
        parameter.requires_grad_(False)

    if len(RoutingReplay.all_routing_replays) != 1:
        raise AssertionError(f"expected one native Megatron replay hook, got {len(RoutingReplay.all_routing_replays)}")
    replay = router.routing_replay
    if replay is None:
        raise AssertionError("Megatron router did not register Slime R3 replay")
    for layer_id in range(2, 25):
        replay.record(packed[:, layer_id, :])

    captured_internal_indices = []
    original_capture = routing_replay._capture_ordered_topk

    def capture_internal_indices(indices):
        captured_internal_indices.append(indices.detach().clone())

    routing_replay._capture_ordered_topk = capture_internal_indices

    torch.manual_seed(711 + torch.distributed.get_rank())
    hidden = torch.randn((packed.shape[0], 1, 2048), device="cuda", requires_grad=True)
    forward_equal = 0
    os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
    forward_outputs = []
    for layer_id in range(2, 25):
        probabilities, indices = router(hidden)
        expected = packed[:, layer_id, :].to(device="cuda")
        actual = captured_internal_indices[-1].reshape_as(expected).to(dtype=expected.dtype)
        if not torch.equal(actual, expected):
            raise AssertionError(f"forward replay differs at physical MoE layer {layer_id}")
        expected_map = torch.zeros((expected.shape[0], 32), dtype=torch.bool, device="cuda")
        expected_map.scatter_(1, expected.long(), True)
        if not torch.equal(indices.reshape(expected.shape[0], 32), expected_map):
            raise AssertionError(f"forward routing map differs at physical MoE layer {layer_id}")
        forward_equal += expected.numel()
        forward_outputs.append(probabilities)

    os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
    backward_equal = 0
    backward_outputs = []
    for layer_id in range(2, 25):
        probabilities, indices = router(-hidden)
        expected = packed[:, layer_id, :].to(device="cuda")
        actual = captured_internal_indices[-1].reshape_as(expected).to(dtype=expected.dtype)
        if not torch.equal(actual, expected):
            raise AssertionError(f"backward replay differs at physical MoE layer {layer_id}")
        expected_map = torch.zeros((expected.shape[0], 32), dtype=torch.bool, device="cuda")
        expected_map.scatter_(1, expected.long(), True)
        if not torch.equal(indices.reshape(expected.shape[0], 32), expected_map):
            raise AssertionError(f"backward routing map differs at physical MoE layer {layer_id}")
        backward_equal += expected.numel()
        backward_outputs.append(probabilities)

    loss = sum(output.float().sum() for output in backward_outputs)
    loss.backward()
    if hidden.grad is None or not bool(torch.isfinite(hidden.grad).all()):
        raise AssertionError("no finite upstream gradient through replayed router scores")
    RoutingReplay.assert_all_consumed()
    routing_replay._capture_ordered_topk = original_capture
    report = {
        "passed": True,
        "scope": "real SGLang route IDs -> Slime actor route packing -> native Megatron R3; no full actor model",
        "world_size": world_size,
        "expert_parallel_size": 2,
        "gpu": torch.cuda.get_device_name(local_rank),
        "captured_shape": list(routes.shape),
        "actor_packed_shape": list(packed.shape),
        "moe_layers_replayed": 23,
        "forward_ids_equal": forward_equal,
        "backward_ids_equal": backward_equal,
        "captured_route_ids_per_stage": int(routes.shape[0] * 23 * 4),
        "padded_route_rows": int(packed.shape[0] - routes.shape[0]),
        "all_replay_records_consumed": True,
        "upstream_gradient_finite": True,
        "checkpoint_architecture": record.get("model"),
        "checkpoint_position_type": record.get("position_embedding_type", "unspecified"),
    }
    if torch.distributed.get_rank() == 0:
        write_json(Path(args.output), report)
        print(json.dumps(report), flush=True)
    torch.distributed.barrier()
    RoutingReplay.clear_all()
    mpu.destroy_model_parallel()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
