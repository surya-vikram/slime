"""The MixRL task file: which tasks train, how many prompts each, how eval is sized, and what each task is.

Only task and eval-size choices live here. Everything else (batch geometry, eval
cadence, optimizer) stays in examples/chimera/train.sh. Preview a file with:

    python3 -m slime_plugins.chimera_mixrl.tasks [path] [--samples-per-prompt N]
"""
import argparse
import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / 'examples' / 'chimera' / 'mixrl_tasks.json'
JUDGE_USES = ('none', 'on_miss', 'to_pass', 'always')
REWARDS = ('binary', 'graded')
EVAL = {'samples_per_prompt'}
FIELDS = {'enabled', 'prompts_per_step', 'eval_prompts', 'max_response_tokens', 'about'}
ABOUT = {'summary', 'grading', 'answer_format', 'requires', 'domain', 'verifier', 'judge', 'reward',
         'train_pool', 'val_pool'}


def _positive(value):
    return type(value) is int and value >= 1


def _text(value):
    return isinstance(value, str) and value.strip()


def load(path=DEFAULT_PATH):
    """Read and structurally validate a task file; returns {'eval', 'domains', 'tasks'}."""
    path = Path(path)
    spec = json.loads(path.read_text())
    if not isinstance(spec, dict) or set(spec) - {'_readme'} != {'eval', 'domains', 'tasks'}:
        raise ValueError(f'{path}: expected "eval", "domains" and "tasks"')
    rules = spec['eval']
    if not isinstance(rules, dict) or set(rules) != EVAL or not _positive(rules['samples_per_prompt']):
        raise ValueError(f'{path}: eval must be exactly {{"samples_per_prompt": <positive integer>}}')
    for name, domain in spec['domains'].items():
        if not isinstance(domain, dict) or set(domain) != {'summary'} or not _text(domain['summary']):
            raise ValueError(f'{path}: domain {name} needs exactly a summary')
    tasks = spec['tasks']
    for name, task in tasks.items():
        where = f'{path}: {name}'
        if not isinstance(task, dict) or set(task) != FIELDS:
            raise ValueError(f'{where}: fields must be exactly {sorted(FIELDS)}')
        if type(task['enabled']) is not bool:
            raise ValueError(f'{where}: enabled must be true or false')
        for key in ('prompts_per_step', 'max_response_tokens'):
            if not _positive(task[key]):
                raise ValueError(f'{where}: {key} must be a positive integer')
        if task['eval_prompts'] != 'all' and not _positive(task['eval_prompts']):
            raise ValueError(f'{where}: eval_prompts must be "all" or a positive integer')
        about = task['about']
        if not isinstance(about, dict) or set(about) != ABOUT:
            raise ValueError(f'{where}: about must have exactly {sorted(ABOUT)}')
        for key in ('summary', 'grading', 'answer_format', 'domain', 'verifier'):
            if not _text(about[key]):
                raise ValueError(f'{where}: about.{key} must be a nonempty string')
        if not isinstance(about['requires'], list) or not all(_text(r) for r in about['requires']):
            raise ValueError(f'{where}: about.requires must be a list of strings')
        if about['domain'] not in spec['domains']:
            raise ValueError(f'{where}: about.domain {about["domain"]} is not listed in domains')
        if about['judge'] not in JUDGE_USES or about['reward'] not in REWARDS:
            raise ValueError(f'{where}: about.judge must be one of {JUDGE_USES}; about.reward one of {REWARDS}')
        if not _positive(about['train_pool']) or not _positive(about['val_pool']):
            raise ValueError(f'{where}: about.train_pool and about.val_pool must be positive integers')
        if task['prompts_per_step'] > about['train_pool']:
            raise ValueError(f'{where}: prompts_per_step {task["prompts_per_step"]} exceeds train_pool {about["train_pool"]}')
    unused = sorted(set(spec['domains']) - {t['about']['domain'] for t in tasks.values()})
    if unused:
        raise ValueError(f'{path}: domains without tasks: {", ".join(unused)}')
    if not any(task['enabled'] for task in tasks.values()):
        raise ValueError(f'{path}: no task is enabled')
    eval_counts(spec)
    return spec


def enabled(spec):
    return {name: task for name, task in spec['tasks'].items() if task['enabled']}


def by_domain(spec, tasks):
    groups = {}
    for name, task in tasks.items():
        groups.setdefault(task['about']['domain'], {})[name] = task
    return groups


def eval_counts(spec):
    """Eval prompts per enabled task: "all" is its whole validation pool; a number is capped at it."""
    return {n: t['about']['val_pool'] if t['eval_prompts'] == 'all' else min(t['eval_prompts'], t['about']['val_pool'])
            for n, t in enabled(spec).items()}


def eval_cost(spec, samples_per_prompt):
    """Eval's worst-case generated tokens as a fraction of one training step's."""
    chosen = enabled(spec)
    counts = eval_counts(spec)
    k = spec['eval']['samples_per_prompt']
    step = sum(t['prompts_per_step'] * samples_per_prompt * t['max_response_tokens'] for t in chosen.values())
    run = sum(counts[n] * k * t['max_response_tokens'] for n, t in chosen.items())
    return {'prompts': sum(counts.values()), 'samples': sum(counts.values()) * k,
            'step_samples': sum(t['prompts_per_step'] for t in chosen.values()) * samples_per_prompt,
            'steps': run / step}


def routes(tasks):
    """Per-task facts used to validate data rows."""
    return {n: {k: t['about'][k] for k in ('domain', 'verifier', 'judge', 'reward')} for n, t in tasks.items()}


def resolved(spec):
    """The enabled-task settings the trainer consumes."""
    chosen = enabled(spec)
    return {'quotas': {n: t['prompts_per_step'] for n, t in chosen.items()},
            'caps': {n: t['max_response_tokens'] for n, t in chosen.items()},
            'eval_quotas': eval_counts(spec),
            'eval_samples': spec['eval']['samples_per_prompt'],
            'routes': routes(chosen),
            'rollout_batch_size': sum(t['prompts_per_step'] for t in chosen.values())}


def check_data(spec, train, val):
    """about must describe the frozen splits, and every task in them must be listed."""
    tasks = spec['tasks']
    for split, rows in (('rl_train', train), ('rl_val', val)):
        unlisted = sorted({r['task'] for r in rows} - tasks.keys())
        if unlisted:
            raise ValueError(f'{split} has tasks missing from the task file: {", ".join(unlisted)}')
    for name, task in tasks.items():
        about = task['about']
        for split, rows, key in (('rl_train', train, 'train_pool'), ('rl_val', val, 'val_pool')):
            mine = [r for r in rows if r['task'] == name]
            if len(mine) != about[key]:
                raise ValueError(f'{name}: about.{key} is {about[key]} but {split} has {len(mine)} rows')
            for row in mine:
                if (row['domain'], row['verifier']) != (about['domain'], about['verifier']):
                    raise ValueError(f'{name}: about says {about["domain"]}/{about["verifier"]} but '
                                     f'{split} row {row["id"]} is {row["domain"]}/{row["verifier"]}')
                if row.get('binary') is not (about['reward'] == 'binary'):
                    raise ValueError(f'{name}: about.reward {about["reward"]} disagrees with {split} row {row["id"]}')
        families = len({r['family_id'] for r in train if r['task'] == name})
        if task['enabled'] and families < task['prompts_per_step']:
            raise ValueError(f'{name}: prompts_per_step {task["prompts_per_step"]} exceeds {families} distinct families')


def blocked(routes, health):
    """Enabled tasks the reward service cannot grade right now, with the reason."""
    judge = health.get('judge') or {}
    reasons = {task: f'grading check failed: {error}'
               for task, error in (health.get('task_errors') or {}).items() if task in routes}
    if not judge.get('ready'):
        for task, route in routes.items():
            if route['judge'] != 'none':
                reasons.setdefault(task, f'needs the judge ({route["judge"]}), but judge '
                                         f'{judge.get("model")!r} is not reachable')
    return reasons


def check_scorer(spec, health):
    """The reward service is the authority on judge use and on what it can grade."""
    reported = health.get('task_judge')
    if not isinstance(reported, dict) or not isinstance(health.get('judge'), dict):
        raise ValueError('Reward service does not report judge status; update chimera-eval')
    for name, task in spec['tasks'].items():
        if reported.get(name) != task['about']['judge']:
            raise ValueError(f'{name}: task file says judge={task["about"]["judge"]} but the reward '
                             f'service grades it as judge={reported.get(name)}')
    reasons = blocked(routes(enabled(spec)), health)
    if reasons:
        raise ValueError('Training cannot start; the reward service cannot grade these enabled tasks:\n'
                         + '\n'.join(f'  {task}: {reason}' for task, reason in reasons.items())
                         + '\nFix the reward service or disable these tasks in the task file.')


def _grid(rows):
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ['  '.join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths)))
             for r in rows]
    lines.insert(1, '-' * len(lines[0]))
    return lines


def table(spec, samples_per_prompt=None):
    chosen = enabled(spec)
    counts = eval_counts(spec)
    groups = by_domain(spec, spec['tasks'])
    domain_rows = [('domain', 'tasks on', 'prompts/step', 'eval prompts', 'val pool', 'judge')]
    for domain, tasks in groups.items():
        on = {n: t for n, t in tasks.items() if t['enabled']}
        needs = sorted({t['about']['judge'] for t in on.values()} - {'none'}, key=JUDGE_USES.index)
        domain_rows.append((domain, f'{len(on)}/{len(tasks)}', str(sum(t['prompts_per_step'] for t in on.values())),
                            str(sum(counts[n] for n in on)) if on else '-',
                            str(sum(t['about']['val_pool'] for t in on.values())) if on else '-',
                            ('needed' if needs else 'no') if on else '-'))
    task_rows = [('task', 'domain', 'on', 'prompts', 'train pool', 'eval', 'val pool', 'resp_cap', 'judge', 'steps/pass')]
    for name, t in spec['tasks'].items():
        a = t['about']
        task_rows.append((name, a['domain'], 'yes' if t['enabled'] else '-', str(t['prompts_per_step']),
                          str(a['train_pool']), str(counts.get(name, '-')), str(a['val_pool']),
                          str(t['max_response_tokens']), a['judge'], f"{a['train_pool'] / t['prompts_per_step']:.0f}"))
    lines = _grid(domain_rows) + [''] + _grid(task_rows) + ['']
    lines += [f'{n}: {t["about"]["summary"]}' for n, t in chosen.items()]
    batch = sum(t['prompts_per_step'] for t in chosen.values())
    total = f'{len(chosen)} of {len(spec["tasks"])} tasks enabled; {batch} prompts per step'
    lines += ['', total + (f' x {samples_per_prompt} responses = {batch * samples_per_prompt} samples'
                           if samples_per_prompt else '')]
    k = spec['eval']['samples_per_prompt']
    evaluated = f'eval: {sum(counts.values())} prompts x {k} samples = {sum(counts.values()) * k} per eval'
    if samples_per_prompt:
        evaluated += f'; up to {eval_cost(spec, samples_per_prompt)["steps"]:.2f} training steps of tokens'
    needs = [n for n, t in chosen.items() if t['about']['judge'] != 'none']
    lines += [evaluated, 'judge: needed by ' + ', '.join(needs) if needs else 'judge: not needed']
    return '\n'.join(lines + notes(spec))


def notes(spec):
    """Capped eval requests; information only."""
    counts = eval_counts(spec)
    return [f'note: {name} asks for {task["eval_prompts"]} eval prompts; its validation pool has {counts[name]}'
            for name, task in enabled(spec).items()
            if task['eval_prompts'] != 'all' and task['eval_prompts'] > counts[name]]


def main():
    parser = argparse.ArgumentParser(description='Preview and validate a MixRL task file.')
    parser.add_argument('path', nargs='?', default=str(DEFAULT_PATH))
    parser.add_argument('--samples-per-prompt', type=int)
    args = parser.parse_args()
    print(table(load(args.path), args.samples_per_prompt))


if __name__ == '__main__':
    main()
