"""Fixed-bound MiMo-style objective; native DAPO remains a separate control.

No adaptive controller, quality redistribution or token-level shaping.
"""


def active_diagnostics(metrics, prefix='train/'):
    """Apply only after native global DP/microbatch reduction, never rank-locally."""
    fraction = metrics.get(prefix + 'effective_response_fraction', 0.)
    if fraction <= 0:
        return {}
    return {prefix + 'active/' + name: metrics[prefix + name] / fraction
            for name in ('entropy', 'importance_ratio', 'importance_masked_fraction',
                         'importance_positive_low_fraction', 'importance_positive_high_fraction',
                         'importance_negative_low_fraction', 'importance_negative_high_fraction',
                         'train_rollout_logprob_abs_diff') if prefix + name in metrics}


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


def loss(args, batch, logits, sum_of_sample_mean):
    import torch

    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy
    from .runtime import config

    c = config()
    if args.calculate_per_token_loss or args.kl_coef != 0 or args.entropy_coef != 0:
        raise ValueError('MiMo fixed objective requires per-response reduction, zero KL/entropy loss')
    if args.context_parallel_size != 1:
        raise ValueError('CP objective qualification is not yet complete')
    _, values = get_log_probs_and_entropy(
        logits, args=args, unconcat_tokens=batch['unconcat_tokens'],
        total_lengths=batch['total_lengths'], response_lengths=batch['response_lengths'],
        with_entropy=True)
    current = torch.cat(values['log_probs'])
    behavior = torch.cat(batch['rollout_log_probs'])
    advantages = torch.cat(batch['advantages'])
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
    return value, metrics
