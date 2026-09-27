"""Frozen-data route contracts and domain-balanced evaluation summaries."""

import statistics

# Route -> (domain, existing evaluator verifier). No new reward implementation.
ROUTES = {
    'gsm8k_train': ('math', 'math'), 'nemotron_math': ('math', 'equivalence'),
    'mcqa': ('knowledge', 'choice'), 'openqa': ('knowledge', 'equivalence'),
    'science': ('knowledge', 'equivalence'), 'hotpot_train': ('grounding', 'grounded'),
    'cascade_chat': ('quality', 'quality'), 'cascade_lists': ('quality', 'quality'),
    'cascade_plans': ('quality', 'quality'),
    'nvidia_multichallenge': ('multiturn', 'rubric'),
    'nvidia_multichallenge_advanced': ('multiturn', 'rubric'),
    'nemotron_if': ('instruction', 'instruction'), 'structured_train': ('structure', 'structure'),
    'reasoning_gym': ('logic', 'exact'), 'calendar': ('logic', 'calendar'),
    'apps': ('python', 'apps'),
}


def validate_route(row):
    expected = ROUTES.get(row['task'])
    if expected != (row['domain'], row['verifier']):
        raise ValueError(f"Unknown/mismatched training route: {row['task']}")
    if row.get('turns'):
        raise ValueError('Interactive trajectories are eval-only; training uses frozen conversation history')
    messages = row.get('messages')
    if not messages or messages[-1]['role'] != 'user':
        raise ValueError('Training conversation must end in a user prompt')
    if any(m['role'] not in ('system', 'user', 'assistant') or not isinstance(m['content'], str)
           for m in messages):
        raise ValueError('Invalid training conversation')
    expected_binary = row['verifier'] != 'quality'
    if row.get('binary') is not expected_binary:
        raise ValueError('Route binary/quality metadata mismatch')


def evaluation_summary(groups):
    """All prompts equal within a domain; all enabled domains equal overall.

    Each item is (frozen row, completed Samples); no dynamic filtering. pass@k
    here is observed any-pass among all k draws, only for binary routes.
    """
    tasks, domains = {}, {}
    for row, samples in groups:
        if not samples:
            raise ValueError('Missing evaluation responses')
        grades = [s.metadata['grade'] for s in samples]
        if any(g.get('status') != 'valid' for g in grades):
            raise ValueError('Evaluation cannot aggregate failed grading')
        item = {'mean_score': statistics.mean(g['score'] for g in grades),
                'cap_rate': statistics.mean(s.status.name == 'TRUNCATED' for s in samples)}
        if row['binary']:
            if any(type(g.get('passed')) is not bool for g in grades):
                raise ValueError('Binary route lacks an explicit pass verdict')
            item[f'pass@{len(samples)}'] = float(any(g['passed'] for g in grades))
        tasks.setdefault(row['task'], []).append(item)
        domains.setdefault(row['domain'], []).append(item)

    def aggregate(items):
        return {k: statistics.mean(x[k] for x in items if k in x)
                for k in sorted(set().union(*(x.keys() for x in items)))} | {'prompts': len(items)}

    task_scores = {k: aggregate(v) for k, v in tasks.items()}
    domain_scores = {k: aggregate(v) for k, v in domains.items()}
    return {'tasks': task_scores, 'domains': domain_scores,
            'equal_domain_mean': statistics.mean(x['mean_score'] for x in domain_scores.values()),
            'aggregation': 'prompt mean within each domain, equal mean across enabled domains; no pass@k quality score'}
