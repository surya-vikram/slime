"""Collection timing summaries, separate from optimization/reward decisions."""
import math


def quantile(values, q):
    values = sorted(values)
    if not values:
        return 0.
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def collection_summary(samples, seconds, informative):
    timings = {}
    for name, start, end in (
        ('generation_queue', 'generation_queued_at', 'generation_started_at'),
        ('generation', 'generation_started_at', 'generation_finished_at'),
        ('judge_queue', 'reward_queued_at', 'reward_started_at'),
        ('judge', 'reward_started_at', 'reward_finished_at'),
    ):
        values = [s.metadata[end] - s.metadata[start] for s in samples
                  if start in s.metadata and end in s.metadata and not s.metadata.get('reused_generation')]
        timings[name] = {'count': len(values), 'p50_seconds': quantile(values, .5),
                         'p95_seconds': quantile(values, .95)}
    tokens = sum(s.response_length for s in samples if not s.metadata.get('reused_generation'))
    return {'seconds': seconds, 'generated_tokens': tokens,
            'generated_tokens_per_second': tokens / max(seconds, 1e-9),
            'informative_groups_per_second': informative / max(seconds, 1e-9),
            'reused_responses': sum(bool(s.metadata.get('reused_generation')) for s in samples),
            'timings': timings}
