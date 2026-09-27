"""Fail-fast invariants for frozen Chimera routers (not expert-path replay)."""

_snapshots = {}
_loads = {}


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    check_frozen(args, model)
    import os
    if os.environ.get('MIXRL_ROUTER_METRICS', '0') != '1':
        return
    for chunk, module in enumerate(model):
        for name, router in module.named_modules():
            if name.rsplit('.', 1)[-1] != 'router' or not hasattr(router, 'topk'):
                continue
            state = _loads.setdefault(id(router), {'counts': [], 'active': False,
                                                   'name': f'{chunk}:{name}'})
            state['counts'].clear()
            state['active'] = True
            if not state.get('registered'):
                def capture(module, inputs, output, state=state):
                    # Full input-token load including padding, not just response
                    # tokens; exclude activation recomputation from telemetry.
                    if state['active'] and os.environ.get('ROUTING_REPLAY_STAGE') != 'replay_backward':
                        state['counts'].append(output[1].detach().sum(dim=0))
                router.register_forward_hook(capture)
                state['registered'] = True


def after_train_step(args, rollout_id, step_id, model):
    import json
    import torch
    check_frozen(args, model)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    for module in model:
        for router in module.modules():
            state = _loads.get(id(router))
            if state is None or not state['active']:
                continue
            state['active'] = False
            if state['counts']:
                counts = torch.stack(state['counts']).cpu().tolist()
                print('MIXRL_ROUTER_LOAD ' + json.dumps(dict(
                    rollout_id=rollout_id, step_id=step_id, rank=rank, layer=state['name'],
                    scope='full_input_including_padding_per_microbatch', counts=counts)), flush=True)
            state['counts'].clear()


def check_frozen(args, model):
    import torch

    frozen = {}
    expert_trainable = 0
    for chunk, module in enumerate(model):
        for name, parameter in module.named_parameters():
            if '.router.' in '.' + name and name.rsplit('.', 1)[-1] in ('weight', 'bias'):
                if parameter.requires_grad:
                    raise RuntimeError(f'Router projection is not frozen: {name}')
                frozen[f'{chunk}:{name}'] = parameter
            if '.experts.' in '.' + name and parameter.requires_grad:
                expert_trainable += 1
        for name, buffer in module.named_buffers():
            if name.endswith(('expert_bias', 'e_score_correction_bias')):
                frozen[f'{chunk}:{name}'] = buffer
    if not frozen or not expert_trainable:
        raise RuntimeError('Expected frozen router state and trainable experts')
    if args.moe_router_bias_update_rate != 0 or args.moe_router_load_balancing_type not in ('none', ['none']):
        raise RuntimeError('RL must not update router correction bias')
    if getattr(args, 'use_rollout_routing_replay', False) and args.moe_router_fusion:
        raise RuntimeError('Fused router top-k bypasses the pinned Slime replay hook')
    key = tuple(id(m) for m in model)
    current = {name: tensor.detach().cpu().clone() for name, tensor in frozen.items()}
    previous = _snapshots.get(key)
    if previous is not None:
        if previous.keys() != current.keys() or any(not torch.equal(previous[n], current[n]) for n in current):
            raise RuntimeError('Frozen router/bias changed between optimizer updates')
    else:
        _snapshots[key] = current
