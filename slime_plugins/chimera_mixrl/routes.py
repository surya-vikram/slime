"""Training-row contracts and domain-balanced evaluation summaries."""

import statistics

# The policy is a non-thinking instruct model; reasoning tags are an SFT artifact to watch, not a mode.
THINK_TAG = '<think>'


def validate_route(row, routes):
    """routes: {task: about subset} for enabled tasks, from the task file."""
    route = routes.get(row['task'])
    if route is None or (row['domain'], row['verifier']) != (route['domain'], route['verifier']):
        raise ValueError(f"Unknown/mismatched training route: {row['task']}")
    if row.get('turns'):
        raise ValueError('Interactive trajectories are eval-only; training uses frozen conversation history')
    messages = row.get('messages')
    if not messages or messages[-1]['role'] != 'user':
        raise ValueError('Training conversation must end in a user prompt')
    if any(m['role'] not in ('system', 'user', 'assistant') or not isinstance(m['content'], str)
           for m in messages):
        raise ValueError('Invalid training conversation')
    if row.get('binary') is not (route['reward'] == 'binary'):
        raise ValueError('Route binary/graded metadata mismatch')


def evaluation_summary(groups):
    """All prompts equal within a domain; all enabled domains equal overall.

    Each item is (frozen row, completed Samples); no dynamic filtering. pass@k
    here is observed any-pass among all k draws, only for binary routes.
    """
    tasks, domains, ungraded, masked = {}, {}, {}, {}
    for row, samples in groups:
        if not samples:
            raise ValueError('Missing evaluation responses')
        all_samples = samples
        # A response the reward service could not grade is never a zero: the prompt is scored on its
        # other responses (pass@k then counts only those), or left out when none could be graded.
        samples = [s for s in samples if s.metadata['grade'].get('status') != 'grade_failed']
        if len(samples) < len(all_samples):
            masked[row['task']] = masked.get(row['task'], 0) + len(all_samples) - len(samples)
        if not samples:
            ungraded[row['task']] = ungraded.get(row['task'], 0) + 1
            continue
        grades = [s.metadata['grade'] for s in samples]
        if any(g.get('status') != 'valid' for g in grades):
            raise ValueError('Evaluation cannot aggregate failed grading')
        item = {'mean_score': statistics.mean(g['score'] for g in grades),
                'cap_rate': statistics.mean(s.status.name == 'TRUNCATED' for s in samples),
                'incomplete_rate': statistics.mean(g.get('components', {}).get('incomplete') is True for g in grades),
                'think_rate': statistics.mean(THINK_TAG in s.metadata.get('grading_text', '') for s in samples)}
        if row['binary']:
            if any(type(g.get('passed')) is not bool for g in grades):
                raise ValueError('Binary route lacks an explicit pass verdict')
            item[f'pass@{len(all_samples)}'] = float(any(g['passed'] for g in grades))
        tasks.setdefault(row['task'], []).append(item)
        domains.setdefault(row['domain'], []).append(item)

    def aggregate(items):
        return {k: statistics.mean(x[k] for x in items if k in x)
                for k in sorted(set().union(*(x.keys() for x in items)))} | {'prompts': len(items)}

    task_scores = {k: aggregate(v) for k, v in tasks.items()}
    for task, count in ungraded.items():
        task_scores.setdefault(task, {'prompts': 0})['ungraded_prompts'] = count
    for task, count in masked.items():
        task_scores.setdefault(task, {'prompts': 0})['masked_responses'] = count
    domain_scores = {k: aggregate(v) for k, v in domains.items()}
    return {'tasks': task_scores, 'domains': domain_scores,
            'equal_domain_mean': (statistics.mean(x['mean_score'] for x in domain_scores.values())
                                  if domain_scores else 0.),
            'ungraded_prompts': sum(ungraded.values()), 'masked_responses': sum(masked.values()),
            'aggregation': 'prompt mean within each domain, equal mean across enabled domains; no pass@k quality score; '
                           'an ungradable response is left out of its prompt, a prompt without graded responses is left out'}
