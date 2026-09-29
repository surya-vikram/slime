"""Fail-fast invariants for frozen Chimera routers (not expert-path replay)."""

_snapshots = {}
_loads = {}


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    check_frozen(args, model)
    import os
    from slime.utils import routing_replay
    routing_replay.ROUTE_AGREEMENT['active'] = True
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


def load_balance(counts):
    """MiMo section 5.4 expert-load statistics for one layer's per-expert token counts:
    coefficient of variation, peak load (max / mean) and fraction of cold experts (< 0.1 x mean)."""
    n = len(counts)
    mean = sum(counts) / n
    if mean <= 0:
        return None
    std = (sum((c - mean) ** 2 for c in counts) / n) ** .5
    return {'cv': round(std / mean, 4), 'peak': round(max(counts) / mean, 4),
            'cold': round(sum(c < .1 * mean for c in counts) / n, 4)}


def load_summary(layers):
    """{layer name: per-expert counts} -> one step's summary; layers keyed by decoder index."""
    import re
    per_layer, counts = {}, {}
    for name, values in layers.items():
        match = re.search(r'layers\.(\d+)\.', name)
        key = match.group(1) if match else name
        stats = load_balance(values)
        if stats is not None:
            per_layer[key], counts[key] = stats, [int(v) for v in values]
    if not per_layer:
        return None
    worst = max(per_layer, key=lambda k: per_layer[k]['cv'])
    return {'cv_mean': round(sum(s['cv'] for s in per_layer.values()) / len(per_layer), 4),
            'peak_max': max(s['peak'] for s in per_layer.values()),
            'cold_mean': round(sum(s['cold'] for s in per_layer.values()) / len(per_layer), 4),
            'worst_layer': worst, 'layers': per_layer, 'counts': counts}


def report_consistency(args, rollout_id, step_id):
    """One line per optimizer step on train/rollout agreement, over every rank's tokens:
    per-token |log-prob| and |prob| gap between the actor and SGLang's behaviour log-probs
    (after top-p replay), and how often the training router would have chosen experts other
    than the replayed rollout routes: the share of expert choices replay overrode (the routes
    actually used are always the rollout's; 0 without routing replay)."""
    import json
    import torch
    from slime.utils import routing_replay
    from .objective import GAP
    agreement = routing_replay.ROUTE_AGREEMENT
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    def value(x):
        return torch.as_tensor(x, dtype=torch.float64, device=device).reshape(())
    by_kind = ('max_first', 'max_middle', 'max_last', 'max_likely', 'max_tail')
    maxes = torch.stack([value(GAP.get(k, 0.)) for k in ('logprob_max', 'prob_max', *by_kind)])
    worst = GAP.get('worst', [])
    sums = torch.stack([value(GAP.get(k, 0.)) for k in ('logprob_sum', 'tokens', 'over_0.1', 'over_1', 'k1_sum', 'k3_sum')]
                       + [value(agreement['sets']), value(agreement['mismatched'])])
    GAP.clear()
    agreement.update(sets=0, mismatched=0, active=False)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(maxes, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(sums)
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, worst)
        worst = [w for rank in gathered for w in rank]
        if torch.distributed.get_rank() != 0:
            return
    worst = sorted(worst, key=lambda w: -w['abs_logprob_diff'])[:8]
    logprob_sum, tokens, over_small, over_large, k1_sum, k3_sum, sets, mismatched = sums.tolist()
    if not tokens:
        return
    line = {'tokens': int(tokens), 'kl_k1': k1_sum / tokens, 'kl_k3': k3_sum / tokens,
            'logprob_abs_diff_mean': logprob_sum / tokens,
            'logprob_abs_diff_max': maxes[0].item(), 'prob_abs_diff_max': maxes[1].item(),
            'tokens_over_0.1': int(over_small), 'tokens_over_1': int(over_large),
            'route_sets': int(sets), 'routes_overridden_by_replay': mismatched / sets if sets else 0.,
            **{f'logprob_abs_diff_{k}': v for k, v in zip(by_kind, maxes[2:].tolist())}}
    print('MIXRL_CONSISTENCY ' + json.dumps({'rollout_id': rollout_id, 'step_id': step_id, **line}), flush=True)
    if worst:
        print('MIXRL_WORST_TOKENS ' + json.dumps({'rollout_id': rollout_id, 'step_id': step_id, 'tokens': worst}),
              flush=True)
    from slime.observability import logging_utils
    metrics = {f'consistency/{k}': v for k, v in line.items()}
    metrics['train/step'] = rollout_id * (getattr(args, 'num_steps_per_rollout', None) or 1) + step_id
    logging_utils.log(args, metrics, step_key='train/step')


def after_train_step(args, rollout_id, step_id, model):
    import json
    import torch
    check_frozen(args, model)
    report_consistency(args, rollout_id, step_id)
    names, totals = [], []
    for module in model:
        for router in module.modules():
            state = _loads.get(id(router))
            if state is None or not state['active']:
                continue
            state['active'] = False
            # This rank's microbatches; every rank reaches here with the same layers in the same order.
            total = torch.zeros(router.weight.shape[0], dtype=torch.float64, device=router.weight.device)
            for counts in state['counts']:
                total += counts.to(total.dtype)
            names.append(state['name'])
            totals.append(total)
            state['counts'].clear()
    if not totals:
        return
    loads = torch.stack(totals)
    if torch.distributed.is_initialized():
        # DP-only layout (TP=PP=CP=1): ranks route disjoint tokens, so the sum is the whole batch.
        torch.distributed.all_reduce(loads)
        if torch.distributed.get_rank() != 0:
            return
    summary = load_summary(dict(zip(names, loads.cpu().tolist())))
    if summary is None:
        return
    # Tokens as routed in the training forward (replayed rollout routes under R3), padding included.
    print('MIXRL_ROUTER ' + json.dumps({'rollout_id': rollout_id, 'step_id': step_id, **summary}), flush=True)
    from slime.observability import logging_utils
    metrics = {'router/cv_mean': summary['cv_mean'], 'router/peak_max': summary['peak_max'],
               'router/cold_mean': summary['cold_mean']}
    for layer, stats in summary['layers'].items():
        metrics.update({f'router/layer_{layer}/{k}': v for k, v in stats.items()})
    metrics['train/step'] = rollout_id * (getattr(args, 'num_steps_per_rollout', None) or 1) + step_id
    logging_utils.log(args, metrics, step_key='train/step')


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
