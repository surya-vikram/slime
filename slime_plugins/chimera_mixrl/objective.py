"""Fixed-bound MiMo-style objective; native DAPO remains a separate control.

No adaptive controller, quality redistribution or token-level shaping.
"""


def active_diagnostics(metrics, prefix='train/'):
    """Apply only after native global DP/microbatch reduction, never rank-locally."""
    out = {}
    # Distillation: per-domain means over answers, from distill_loss's per-domain sums and answer counts.
    head = prefix + 'distill_answers/'
    for key, answers in metrics.items():
        if key.startswith(head) and answers > 0:
            domain = key[len(head):]
            for name in ('kl', 'clipped'):
                out[f'{prefix}distill/{domain}/{name}'] = metrics[f'{prefix}distill_{name}/{domain}'] / answers
    fraction = metrics.get(prefix + 'effective_response_fraction', 0.)
    if fraction <= 0:
        return out
    return out | {prefix + 'active/' + name: metrics[prefix + name] / fraction
                  for name in ('entropy', 'importance_ratio', 'importance_masked_fraction',
                               'importance_positive_low_fraction', 'importance_positive_high_fraction',
                               'importance_negative_low_fraction', 'importance_negative_high_fraction',
                               'train_rollout_logprob_abs_diff') if prefix + name in metrics}


# Train/rollout log-prob gap over loss-active response tokens, accumulated across this
# rank's microbatches; routing.after_train_step reduces it over ranks and prints it.
GAP = {}


def record_gap(current, behavior, masks, lengths=None, totals=None, tokens=None, worst=8):
    """Per-token |log-prob| and |prob| gap over loss-active tokens; with lengths, also the max gap
    by response position (first/middle/last token) and by likelihood (behaviour log-prob below -5
    is a tail token), and the worst tokens with their context."""
    import torch
    with torch.no_grad():
        active = torch.cat([m.to(current.device) for m in masks]).bool()
        full = (current.detach().float() - behavior.detach().float()).abs()
        diff = full[active]
        if not diff.numel():
            return
        prob = (current.detach().float().exp() - behavior.detach().float().exp())[active].abs()
        def keep_max(key, value):
            GAP[key] = torch.maximum(GAP[key], value) if key in GAP else value
        keep_max('logprob_max', diff.max())
        keep_max('prob_max', prob.max())
        # Per-token KL(rollout || train) estimators over tokens sampled by the rollout engine,
        # as reported in the literature (e.g. R3): k1 = log mu - log pi, k3 = r - 1 - log r, r = pi / mu.
        log_ratio = (current.detach().float() - behavior.detach().float())[active]
        for key, value in (('logprob_sum', diff.sum()), ('tokens', float(diff.numel())),
                           ('over_0.1', (diff > .1).sum().float()), ('over_1', (diff > 1).sum().float()),
                           ('k1_sum', (-log_ratio).sum()), ('k3_sum', (log_ratio.exp() - 1 - log_ratio).sum())):
            GAP[key] = GAP.get(key, 0.) + value
        if lengths is None:
            return
        device = current.device
        position = torch.cat([torch.arange(n, device=device) for n in lengths])
        last = torch.cat([torch.arange(n, device=device) == n - 1 for n in lengths])
        tail = behavior.detach().float() < -5
        zero = torch.zeros((), device=device)
        for key, selected in (('max_first', position == 0), ('max_last', last), ('max_middle', (position > 0) & ~last),
                              ('max_tail', tail), ('max_likely', ~tail)):
            chosen = full[active & selected]
            keep_max(key, chosen.max() if chosen.numel() else zero)
        values, index = full.masked_fill(~active, -1.).topk(min(worst, int(active.sum())))
        ends = torch.cumsum(torch.tensor(list(lengths), device=device), 0)
        for value, i in zip(values.tolist(), index.tolist()):
            sample = int(torch.searchsorted(ends, torch.tensor(i, device=device), right=True))
            pos = int(position[i])
            prompt = int(totals[sample]) - int(lengths[sample])
            GAP.setdefault('worst', []).append({
                'abs_logprob_diff': round(value, 5), 'rollout_logprob': round(float(behavior[i]), 5),
                'train_logprob': round(float(current[i]), 5), 'response_position': pos,
                'response_length': int(lengths[sample]), 'prompt_length': prompt,
                'token_id': int(tokens[sample][prompt + pos]) if tokens is not None and sample < len(tokens) else None})
        GAP['worst'] = sorted(GAP['worst'], key=lambda w: -w['abs_logprob_diff'])[:worst]


def group_length_scales(lengths, eligible):
    """Compensate native per-response mean + response-count outer normalization.

    With n responses, advantage *= n * response_length / eligible_group_length.
    Native sum(response loss / response_length) / (prompts*n) then equals
    sum(prompt token losses / eligible_group_length) / prompts, regardless of
    packing/DP assignment. Cap masks are applied before computing these lengths;
    importance masks must NOT change them afterwards.
    """
    if len(lengths) != len(eligible) or not lengths or any(x < 0 for x in lengths):
        raise ValueError('Invalid group token lengths')
    denominator = sum(length for length, keep in zip(lengths, eligible) if keep)
    if denominator <= 0:
        raise ValueError('No eligible generated tokens')
    n = len(lengths)
    return [n * length / denominator if keep else 0. for length, keep in zip(lengths, eligible)]


def masked_terms(current, behavior, advantages, positive=(.2, 5.), negative=(.2, 5.)):
    import torch

    for low, high in (positive, negative):
        if not 0 < low <= 1 <= high:
            raise ValueError('Importance bounds must contain one and be positive')
    # Both the ratio and its selection mask are constants in differentiation.
    delta = (current.detach() - behavior.detach()).float()
    ratio = delta.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError('Nonfinite importance ratios; fail rather than hide instability')
    low = torch.where(advantages >= 0, positive[0], negative[0])
    high = torch.where(advantages >= 0, positive[1], negative[1])
    keep = (ratio >= low) & (ratio <= high)
    terms = -(keep * ratio * advantages.detach()) * current
    return terms, ratio, keep


def full_vocab_log_probs(logits, unconcat_tokens, total_lengths, response_lengths, chunk=4096):
    """lp_full of mixrl/mopd/README.md: each answer token's log-prob over the whole vocabulary, from the training
    logits, without gradient. CP = TP = 1 (MixRL's layout): every position and the whole vocabulary are local.
    Sequences are packed back to back as in get_log_probs_and_entropy; row j predicts token j + 1."""
    import torch
    with torch.no_grad():
        flat, out, offset = logits.squeeze(0), [], 0
        for tokens, total, response in zip(unconcat_tokens, total_lengths, response_lengths, strict=True):
            rows = flat[offset + total - response - 1:offset + total - 1]
            target = torch.as_tensor(tokens[total - response:total], device=flat.device).long()
            values = []
            for i in range(0, response, chunk):
                part = rows[i:i + chunk].float()
                values.append(part.gather(-1, target[i:i + chunk, None]).squeeze(-1) - torch.logsumexp(part, -1))
            out.append(torch.cat(values) if values else flat.new_zeros(0))
            offset += total
    return out


def loss(args, batch, logits, sum_of_sample_mean):
    return _loss(args, batch, logits, sum_of_sample_mean)


def distill_loss(args, batch, logits, sum_of_sample_mean):
    """Multi-teacher on-policy distillation (mixrl/mopd/README.md): this loss with the teacher's per-token
    advantage A = clip(q - lp_full, -clip, clip) in place of the group-relative one; the importance weight, its
    mask, candidate-set replay and routing replay are unchanged."""
    return _loss(args, batch, logits, sum_of_sample_mean, distill=True)


def _loss(args, batch, logits, sum_of_sample_mean, distill=False):
    import torch

    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, get_rollout_top_p_logprob_kwargs
    from .runtime import config

    c = config()
    if args.calculate_per_token_loss or args.kl_coef != 0 or args.entropy_coef != 0:
        raise ValueError('MiMo fixed objective requires per-response reduction, zero KL/entropy loss')
    if args.context_parallel_size != 1:
        raise ValueError('CP objective qualification is not yet complete')
    if distill and (c.get('mode') != 'distill' or getattr(args, 'tensor_model_parallel_size', 1) != 1
                    or args.rollout_temperature != 1):
        raise ValueError('distill_loss needs a distillation config, TP=1 and temperature 1')
    if distill:
        # Before the log-prob pass below, which may reuse buffers: lp_full reads the untouched logits.
        full = torch.cat(full_vocab_log_probs(logits, batch['unconcat_tokens'], batch['total_lengths'],
                                              batch['response_lengths']))
    # With top-p < 1, renormalize each token's log-prob over the candidate set recorded at
    # rollout, as SGLang did for the behaviour log-prob (MiMo's top-p candidate-set replay).
    _, values = get_log_probs_and_entropy(
        logits, args=args, unconcat_tokens=batch['unconcat_tokens'],
        total_lengths=batch['total_lengths'], response_lengths=batch['response_lengths'],
        with_entropy=True, **get_rollout_top_p_logprob_kwargs(args, batch))
    current = torch.cat(values['log_probs'])
    behavior = torch.cat(batch['rollout_log_probs'])
    advantages = torch.cat(batch['advantages'])
    if distill:
        # The advantage slot carries each answer's domain index (runtime.distill_rewards), for the metrics only.
        domains = advantages.detach()
        gap = torch.cat(batch['teacher_log_probs']).to(full) - full  # q - lp_full
        advantages = gap.clamp(-c['adv_clip'], c['adv_clip'])
    record_gap(current, behavior, batch['loss_masks'], batch['response_lengths'],
               batch['total_lengths'], batch['unconcat_tokens'])
    terms, ratio, keep = masked_terms(current, behavior, advantages,
                                     c['is_positive_bounds'], c['is_negative_bounds'])
    value = sum_of_sample_mean(terms)
    # Diagnostics use the original masks and native per-response means. They are
    # intentionally not multiplied by the prompt-length correction in advantages.
    metrics = {'loss': value.detach(), 'pg_loss': value.detach(),
               'entropy': sum_of_sample_mean(torch.cat(values['entropy'])).detach(),
               'effective_response_fraction': sum_of_sample_mean(torch.ones_like(current)).detach(),
               'importance_ratio': sum_of_sample_mean(ratio).detach(),
               'importance_masked_fraction': sum_of_sample_mean((~keep).float()).detach(),
               'train_rollout_logprob_abs_diff': sum_of_sample_mean((current.detach() - behavior).abs()).detach()}
    # Fractions of active response tokens, not conditional on advantage sign.
    # Normalize once after native DP reduction via active_diagnostics.
    for sign, selected, bounds in (
        ('positive', advantages > 0, c['is_positive_bounds']),
        ('negative', advantages < 0, c['is_negative_bounds']),
    ):
        for tail, condition in (('low', ratio < bounds[0]), ('high', ratio > bounds[1])):
            metrics[f'importance_{sign}_{tail}_fraction'] = sum_of_sample_mean(
                (selected & condition).float()).detach()
    if distill:
        # KL: per answer the mean of lp_full - q (a reverse-KL estimate), averaged over answers like the loss.
        # Per domain: sums and answer counts here, divided after the global reduction (active_diagnostics).
        clipped = (gap.abs() >= c['adv_clip']).float()
        metrics['distill_kl'] = sum_of_sample_mean(-gap).detach()
        metrics['distill_clipped'] = sum_of_sample_mean(clipped).detach()
        for i, name in enumerate(c['domains']):
            mine = (domains == i).float()
            metrics[f'distill_answers/{name}'] = sum_of_sample_mean(mine).detach()
            metrics[f'distill_kl/{name}'] = sum_of_sample_mean(-gap * mine).detach()
            metrics[f'distill_clipped/{name}'] = sum_of_sample_mean(clipped * mine).detach()
    return value, metrics
