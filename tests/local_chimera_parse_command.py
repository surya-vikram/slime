"""Validate a DRY_RUN manifest through the real pinned Slime/Megatron parser."""
import argparse
import json
import shlex
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command_file')
    local = parser.parse_args()
    words = shlex.split(Path(local.command_file).read_text())
    if len(words) < 3 or Path(words[1]).name != 'train.py':
        raise ValueError('Expected a synchronous train.py command manifest')
    from slime_plugins.models.chimera import register_transformers
    register_transformers()
    sys.argv = words[1:]
    from slime.utils.arguments import parse_args
    args = parse_args()
    assert args.weight_decay == 0 and args.clip_grad == 1
    assert args.kl_coef == 0 and args.entropy_coef == 0
    assert args.tensor_model_parallel_size == args.pipeline_model_parallel_size == 1
    assert args.context_parallel_size == args.expert_model_parallel_size == args.expert_tensor_parallel_size == 1
    assert args.use_distributed_optimizer and args.use_rollout_routing_replay
    assert args.custom_loss_function_path == 'slime_plugins.chimera_mixrl.objective.loss'
    print(json.dumps(dict(parsed=True, layers=args.num_layers, experts=args.num_experts,
                         topk=args.moe_router_topk, seq_length=args.seq_length,
                         max_position_embeddings=args.max_position_embeddings,
                         weight_decay=args.weight_decay, gradient_clip=args.clip_grad)), flush=True)


if __name__ == '__main__':
    main()
