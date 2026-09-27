"""Single-GPU native Slime tiny actor: import, R3 repeat, update and save.

Synthetic squared-logit loss isolates mechanics; this is NOT a MixRL learning run.
Uses the production command manifest and optimizer with a 16-token probe.
"""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    if 'torch_memory_saver' not in os.environ.get('LD_PRELOAD', ''):
        from torch_memory_saver import configure_subprocess
        # Match the environment Slime normally gives its Ray actor workers.
        with configure_subprocess():
            subprocess.run([sys.executable, *sys.argv], check=True,
                           env={**os.environ, 'TMS_INIT_ENABLE': '1',
                                'TMS_INIT_ENABLE_CPU_BACKUP': '1', 'NCCL_CUMEM_ENABLE': '0'})
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command-file', required=True)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--updates', type=int, default=1)
    parser.add_argument('--resume-from')
    options = parser.parse_args()
    command = Path(options.command_file)
    os.environ['CHIMERA_MIXRL_CONFIG'] = str(command.with_name('mixrl_config.json'))
    os.environ['ENABLE_ROUTING_REPLAY'] = '1'
    os.environ['ROUTING_REPLAY_STAGE'] = 'record'
    os.environ['MIXRL_ROUTER_METRICS'] = '1'
    import torch
    from slime_plugins.models.chimera import register_transformers
    register_transformers()
    sys.argv = shlex.split(command.read_text())[1:]
    from slime.utils.arguments import parse_args
    from slime.backends.megatron_utils.initialize import init
    from slime.backends.megatron_utils.model import initialize_model_and_optimizer, save
    from slime_plugins.chimera_mixrl.routing import before_train_step, after_train_step
    from slime.utils.routing_replay import RoutingReplay
    from slime_plugins.chimera_mixrl.core import write_json
    from megatron.core.distributed import finalize_model_grads

    args = parse_args()
    assert args.num_layers == 8 and args.num_experts == 8 and args.moe_router_topk == 2
    args.save = options.save_dir
    if options.resume_from:
        args.load = options.resume_from
        args.finetune = False
        args.no_load_optim = False
        args.no_load_rng = False
    args.rank = args.local_rank = 0
    torch.cuda.set_device(0)
    rendezvous = tempfile.TemporaryDirectory(prefix='chimera-actor-')
    torch.distributed.init_process_group('nccl', rank=0, world_size=1,
                                        init_method='file://' + str(Path(rendezvous.name) / 'rdzv'))
    init(args)
    model, optimizer, scheduler, loaded = initialize_model_and_optimizer(args, 'actor')
    assert len(model) == 1
    module = model[0]
    module.eval()
    ids = torch.arange(12, 28, device='cuda')[None]
    positions = torch.arange(16, device='cuda')[None]
    with torch.no_grad():
        first = module(input_ids=ids, position_ids=positions, attention_mask=None)
        os.environ['ROUTING_REPLAY_STAGE'] = 'replay_forward'
        repeated = module(input_ids=ids, position_ids=positions, attention_mask=None)
    error = (first.float() - repeated.float()).abs().max().item()
    assert error == 0, error
    if options.updates < 1:
        raise ValueError('--updates must be positive')
    steps = []
    for step in range(options.updates):
        RoutingReplay.clear_all()
        module.train()
        module.zero_grad_buffer()
        optimizer.zero_grad()
        before_train_step(args, 0, step, model, optimizer, scheduler)
        old_experts = {n: p.detach().cpu().clone() for n, p in module.named_parameters() if '.experts.' in n}
        os.environ['ROUTING_REPLAY_STAGE'] = 'record'
        logits = module(input_ids=ids, position_ids=positions, attention_mask=None)
        loss = logits.float().square().mean()
        loss.backward()
        finalize_model_grads(model)
        successful, norm, _ = optimizer.step()
        assert successful and torch.isfinite(torch.tensor(norm))
        scheduler.step(increment=1)
        after_train_step(args, 0, step, model)
        changed = sum(not torch.equal(p.detach().cpu(), old_experts[n])
                      for n, p in module.named_parameters() if n in old_experts)
        assert changed > 0
        steps.append(dict(loss=loss.item(), gradient_norm=float(norm), changed_expert_tensors=changed))
        del logits, loss, old_experts
    iteration = loaded + options.updates if options.resume_from else options.updates - 1
    save(iteration, model, optimizer, scheduler)
    weight_hash = hashlib.sha256()
    for name, value in sorted(module.named_parameters()):
        weight_hash.update(name.encode())
        weight_hash.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    report = dict(scope=__doc__, loaded_iteration=loaded, fixed_weight_r3_logits_max_error=error,
                  steps=steps, parameter_sha256=weight_hash.hexdigest(), scheduler=scheduler.state_dict(),
                  router_and_bias_unchanged=True, native_optimizer_step=True,
                  checkpoint=str(Path(args.save).resolve()), gpu=torch.cuda.get_device_name())
    write_json(Path(options.output), report)
    print(json.dumps(report), flush=True)
    torch.distributed.destroy_process_group()
    rendezvous.cleanup()


if __name__ == '__main__':
    main()
