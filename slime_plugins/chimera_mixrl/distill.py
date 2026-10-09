"""Multi-teacher on-policy distillation inputs: read and check the distillation config (mixrl/distill.json), plan
the batch and where the teachers run. Rules: mixrl/mopd/README.md. Preview with `mixrl/run.sh distill-plan`, or:

    python3 -m slime_plugins.chimera_mixrl.distill CONFIG [--steps N] [--policy-gpus N]
        [--teacher-gpus 6,7] [--teacher-port 8100] [--teacher-memory 0.85] [--plan | --paths]

--plan prints the teacher servers (mixrl/teachers.sh and mixrl/run.sh read it); --paths prints the folders the
training container mounts. Standard library only: it runs on the host before any container starts. The trainer
reads the same config (configure.distill_inputs) and summarizes evaluations with eval_summary().
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

FIELDS = {'student', 'splits', 'teachers', 'tasks'}
TASK_FIELDS = {'teacher', 'prompts_per_step', 'max_response_tokens', 'eval_prompts'}
TASK_FILE = Path(__file__).resolve().parents[2] / 'mixrl' / 'tasks.json'  # MixRL's caps are the default caps
# Calendar stopped improving on eval around its third pass over its prompts; keep every task at or below.
MAX_PASSES = 3
TEACHERS_PER_GPU = 3  # ~20 GB of bf16 weights each, with room for prefill KV on an H200
# Files that must be byte-identical between student and teacher: the per-token log-probs only line up
# when both read the same token ids through the same chat template.
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json',
                   'added_tokens.json', 'chat_template.jinja', 'chat_template.json')
# config.json keys that may differ between checkpoints of one architecture.
CONFIG_VOLATILE = {'transformers_version', '_name_or_path', 'torch_dtype', 'dtype', 'use_cache'}
# Teacher names become server names (SGLang --served-model-name) and MIXRL_TEACHER_URLS entries.
NAME = re.compile(r'^[A-Za-z0-9._-]+$')


def _positive(value):
    return type(value) is int and value >= 1


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_checkpoints(student, teacher):
    """Problems that make a teacher's per-token log-probs incomparable with the student's."""
    problems = []
    try:
        mine, theirs = (json.loads((p / 'config.json').read_text()) for p in (student, teacher))
    except (OSError, json.JSONDecodeError) as error:
        return [f'cannot read config.json: {error}']
    differing = sorted(k for k in (set(mine) | set(theirs)) - CONFIG_VOLATILE if mine.get(k) != theirs.get(k))
    if differing:
        problems.append(f'architecture differs from the student in config.json: {differing[:8]}')
    for name in TOKENIZER_FILES:
        a, b = student / name, teacher / name
        if a.exists() != b.exists():
            problems.append(f'{name} is in {"the student" if a.exists() else "the teacher"} only')
        elif a.exists() and _digest(a) != _digest(b):
            problems.append(f'{name} differs from the student\'s (token ids or chat template would not line up)')
    if not any(teacher.glob('*.safetensors')):
        problems.append('no *.safetensors weights')
    return problems


def _path(value, what, problems, must=None):
    """A full path from the config; `must` is a file that has to exist inside it."""
    if not isinstance(value, str) or not value.startswith('/'):
        problems.append(f'{what}: give the full path (starting with /), got {value!r}')
        return None
    path = Path(value)
    if must and not (path / must).is_file():
        problems.append(f'{what}: {path / must} not found')
    return path


def task_counts(splits, tasks):
    """{(split, task): (prompts, characters)} for the given tasks in rl_train and rl_val."""
    counts = {}
    for split in ('rl_train', 'rl_val'):
        with open(splits / f'{split}.jsonl', encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('task') in tasks:
                    n, chars = counts.get((split, row['task']), (0, 0))
                    counts[(split, row['task'])] = (n + 1, chars + sum(len(m['content']) for m in row['messages']))
    return counts


def load(path, check_teachers=True):
    """Read and check a distillation config. Returns {'config', 'student', 'splits', 'teachers', 'tasks'};
    raises with every problem. check_teachers=False skips reading the teacher folders: the trainer container
    does not mount them (the host checked them, and the servers prove what they serve)."""
    path = Path(path)
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'{path}: {error}') from None
    if not isinstance(config, dict) or set(config) - {'_readme'} != FIELDS:
        raise ValueError(f'{path}: expected exactly {sorted(FIELDS)} (and an optional "_readme")')
    problems = []
    student = config['student']
    if not isinstance(student, dict) or set(student) != {'hf', 'mcore'}:
        raise ValueError(f'{path}: "student" must be {{"hf": "/full/path", "mcore": "/full/path"}}')
    hf = _path(student['hf'], 'student.hf', problems, 'config.json')
    mcore = _path(student['mcore'], 'student.mcore', problems, 'latest_checkpointed_iteration.txt')
    splits = _path(config['splits'], 'splits', problems)
    if splits:
        problems += [f'splits: {splits / f"{s}.jsonl"} not found' for s in ('rl_train', 'rl_val')
                     if not (splits / f'{s}.jsonl').is_file()]
    teachers = {}
    if not isinstance(config['teachers'], dict) or not config['teachers']:
        problems.append('"teachers" must name at least one teacher: {"name": "/full/path/to/hf"}')
    else:
        for name, folder in config['teachers'].items():
            if not NAME.match(name):
                problems.append(f'teacher {name}: use letters, digits, . _ - only in teacher names')
            teachers[name] = _path(folder, f'teacher {name}', problems, 'config.json' if check_teachers else None)
    defaults = {n: t['max_response_tokens'] for n, t in json.loads(TASK_FILE.read_text())['tasks'].items()}
    tasks = {}
    if not isinstance(config['tasks'], dict) or not config['tasks']:
        problems.append('"tasks" must list at least one task: {"task": {"teacher": "name", "prompts_per_step": N}}')
    else:
        for name, task in config['tasks'].items():
            where = f'task {name}'
            if not isinstance(task, dict) or not {'teacher', 'prompts_per_step'} <= set(task) <= TASK_FIELDS:
                problems.append(f'{where}: needs "teacher" and "prompts_per_step"; may have "max_response_tokens", '
                                f'"eval_prompts"')
                continue
            if task['teacher'] not in config['teachers']:
                problems.append(f'{where}: teacher {task["teacher"]!r} is not in "teachers"')
            if not _positive(task['prompts_per_step']):
                problems.append(f'{where}: "prompts_per_step" must be a positive integer')
            cap = task.get('max_response_tokens', defaults.get(name))
            if not _positive(cap):
                problems.append(f'{where}: "max_response_tokens" must be a positive integer '
                                f'(mixrl/tasks.json has no cap for this task)')
            evals = task.get('eval_prompts', 'all')
            if evals != 'all' and not _positive(evals):
                problems.append(f'{where}: "eval_prompts" must be "all" or a positive integer')
            tasks[name] = {'teacher': task['teacher'], 'prompts_per_step': task['prompts_per_step'],
                           'max_response_tokens': cap, 'eval_prompts': evals}
    unused = sorted(set(teachers) - {t['teacher'] for t in tasks.values()})
    if unused:
        problems.append(f'teacher(s) {", ".join(unused)} score no task')
    if not problems:
        counts = task_counts(splits, tasks)
        for name, task in tasks.items():
            (pool, chars), (held, _) = counts.get(('rl_train', name), (0, 0)), counts.get(('rl_val', name), (0, 0))
            if not pool or not held:
                problems.append(f'task {name}: {pool} rl_train and {held} rl_val prompts in {splits}')
                continue
            if task['eval_prompts'] != 'all' and task['eval_prompts'] > held:
                problems.append(f'task {name}: eval_prompts {task["eval_prompts"]} > its {held} rl_val prompts')
            task.update(pool=pool, eval_pool=held, prompt_chars=chars / pool)
    if check_teachers and not problems:
        for name, folder in teachers.items():
            problems += [f'teacher {name}: {problem}' for problem in compare_checkpoints(hf, folder)]
    if problems:
        raise ValueError('\n'.join(problems))
    return {'config': path, 'student': {'hf': hf, 'mcore': mcore}, 'splits': splits, 'teachers': teachers,
            'tasks': tasks}


def batch_prompts(spec):
    return sum(t['prompts_per_step'] for t in spec['tasks'].values())


def max_passes(task, steps):
    """Times a task uses its prompts in `steps` steps (no refills in distillation)."""
    return task['prompts_per_step'] * steps / task['pool']


def servers(spec, gpus, port, memory=0.85, host='127.0.0.1'):
    """One server per teacher, in the order listed: GPUs taken in turn, port `port` + i. SGLang sizes its cache
    from what is left on the GPU (KV = mem_fraction_static x GPU memory - memory in use), so servers sharing a GPU
    start one after another, the k-th of n with k/n of `memory`: an equal share each."""
    names = list(spec['teachers'])
    if len(names) > len(gpus) * TEACHERS_PER_GPU:
        raise ValueError(f'{len(names)} teachers do not fit {len(gpus)} teacher GPUs at {TEACHERS_PER_GPU} per GPU')
    plan = [{'name': name, 'path': spec['teachers'][name], 'gpu': gpus[i % len(gpus)], 'port': port + i,
             'url': f'http://{host}:{port + i}', 'tasks': [t for t, v in spec['tasks'].items() if v['teacher'] == name]}
            for i, name in enumerate(names)]
    for p in plan:
        mates = [q for q in plan if q['gpu'] == p['gpu']]
        p['memory'] = round(memory * (mates.index(p) + 1) / len(mates), 3)
    return plan


def parse_urls(text):
    """MIXRL_TEACHER_URLS: 'name=url,name=url' -> {teacher name: url}."""
    out = {}
    for item in filter(None, (part.strip() for part in text.split(','))):
        name, sep, url = item.partition('=')
        if not sep or not url.startswith('http') or not NAME.match(name):
            raise ValueError(f'MIXRL_TEACHER_URLS: expected name=http://host:port, got {item!r}')
        out[name] = url.rstrip('/')
    return out


def percentile(values, q):
    """numpy.percentile's default (linear) rule; values need not be sorted."""
    ordered = sorted(values)
    position = q / 100 * (len(ordered) - 1)
    low = math.floor(position)
    return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (position - low)


def answer_stats(answers):
    """answers: [{'tokens': response length, 'stopped': ended its turn before the cap}] -> length and stop rate."""
    lengths = [a['tokens'] for a in answers]
    return {'answers': len(answers), 'length_mean': sum(lengths) / len(lengths),
            'length_p99': percentile(lengths, 99), 'stop_rate': sum(a['stopped'] for a in answers) / len(answers)}


def eval_summary(answers, teacher=None, clip=5.):
    """One evaluation, per task and overall. answers: {task: [{'tokens', 'stopped', 'gaps'}]}, gaps being the
    per-token student - teacher log-probs (both full vocabulary, temperature 1). KL is the mean over answers of
    each answer's mean gap (the loss weighs answers the same way); clipped is the share of tokens at the clip.
    teacher: {task: answer_stats of the teacher's own answers}, shown next to the student's."""
    tasks = {}
    for name, items in answers.items():
        if not items:
            continue
        tokens = sum(len(a['gaps']) for a in items)
        tasks[name] = {**answer_stats(items),
                       'kl': sum(sum(a['gaps']) / len(a['gaps']) for a in items) / len(items),
                       'clipped': sum(abs(g) >= clip for a in items for g in a['gaps']) / max(1, tokens)}
        for key, value in ((teacher or {}).get(name) or {}).items():
            if key != 'answers':
                tasks[name]['teacher_' + key] = value
    return {'mode': 'distill', 'domains': tasks,
            'kl': sum(t['kl'] for t in tasks.values()) / len(tasks) if tasks else None,
            'aggregation': 'per task: mean over answers of the per-token student - teacher log-prob; '
                           'overall: equal mean over tasks'}


def check_batch(spec, samples_per_prompt, policy_gpus):
    total = batch_prompts(spec) * samples_per_prompt
    if total % policy_gpus:
        raise ValueError(f'the batch ({batch_prompts(spec)} prompts x {samples_per_prompt} = {total} answers) must divide '
                         f'by the {policy_gpus} student GPUs: change prompts_per_step of some task')
    return total


def _grid(rows, left=(0,)):
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ['  '.join(c.ljust(w) if i in left else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))).rstrip()
             for r in rows]
    lines.insert(1, '-' * len(lines[0]))
    return lines


def table(spec, steps, samples_per_prompt=1, policy_gpus=None, teacher_gpus=(6, 7), port=8100, memory=0.85):
    rows = [('task', 'teacher', 'prompts/step', 'prompts', 'eval prompts', 'max response', f'passes in {steps} steps')]
    for name, t in spec['tasks'].items():
        evals = t['eval_pool'] if t['eval_prompts'] == 'all' else t['eval_prompts']
        rows.append((name, t['teacher'], str(t['prompts_per_step']), str(t['pool']), str(evals),
                     str(t['max_response_tokens']), f'{max_passes(t, steps):.1f}'))
    lines = [f'config: {spec["config"]}', f'student: {spec["student"]["hf"]} (Megatron: {spec["student"]["mcore"]})',
             f'prompts: {spec["splits"]} (training: rl_train; eval: rl_val)', ''] + _grid(rows) + ['']
    plan = servers(spec, list(teacher_gpus), port, memory)
    lines += _grid([('teacher', 'GPU', 'port', 'memory', 'tasks', 'checkpoint')]
                   + [(p['name'], str(p['gpu']), str(p['port']), str(p['memory']), ','.join(p['tasks']), str(p['path']))
                      for p in plan], left=(0, 4, 5))
    total = batch_prompts(spec) * samples_per_prompt
    lines += ['', f'{len(spec["tasks"])} tasks; {batch_prompts(spec)} prompts per step x {samples_per_prompt} answers = '
                  f'{total} answers; {len(plan)} teacher servers on GPUs {",".join(str(g) for g in teacher_gpus)}']
    if policy_gpus and total % policy_gpus:
        lines.append(f'error: {total} answers do not divide by the {policy_gpus} student GPUs; change prompts_per_step')
    lines += [f'note: {name} can use its {t["pool"]} prompts up to {max_passes(t, steps):.1f} times in {steps} steps '
              f'(more than {MAX_PASSES}: it may memorise them); lower its prompts_per_step'
              for name, t in spec['tasks'].items() if max_passes(t, steps) > MAX_PASSES]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Preview and check a multi-teacher distillation config.')
    parser.add_argument('config')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--samples-per-prompt', type=int, default=1)
    parser.add_argument('--policy-gpus', type=int, default=6)
    parser.add_argument('--teacher-gpus', default='6,7')
    parser.add_argument('--teacher-port', type=int, default=8100)
    parser.add_argument('--teacher-memory', type=float, default=0.85)
    parser.add_argument('--teacher-host', default='127.0.0.1')
    parser.add_argument('--plan', action='store_true',
                        help='print one line per teacher server: port, GPU, memory share, name, checkpoint, URL')
    parser.add_argument('--paths', action='store_true',
                        help='print the student HF and Megatron folders and the splits folder, one per line')
    args = parser.parse_args()
    gpus = [int(g) for g in args.teacher_gpus.split(',') if g.strip()]
    try:
        spec = load(args.config, check_teachers=not args.paths)
        if args.paths:
            print('\n'.join(str(p) for p in (spec['student']['hf'], spec['student']['mcore'], spec['splits'])))
            return
        if args.plan:
            for p in servers(spec, gpus, args.teacher_port, args.teacher_memory, args.teacher_host):
                print('\t'.join(str(p[k]) for k in ('port', 'gpu', 'memory', 'name', 'path', 'url')))
            return
        text = table(spec, args.steps, args.samples_per_prompt, args.policy_gpus, gpus, args.teacher_port,
                     args.teacher_memory)
    except ValueError as error:
        raise SystemExit(f'{args.config}:\n{error}')
    print(text)
    if 'error:' in text:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
