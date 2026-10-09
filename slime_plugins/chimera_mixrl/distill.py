"""Multi-teacher on-policy distillation inputs: read and check a distillation folder, plan the batch and
where the teachers run. Layout and rules: mixrl/mopd/README.md. Preview with `mixrl/run.sh domains`, or:

    python3 -m slime_plugins.chimera_mixrl.distill ROOT [--steps N] [--samples-per-prompt N]
        [--policy-gpus N] [--teacher-gpus 6,7] [--teacher-port 8100]

Standard library only: the preview runs on the host before any container starts.
"""
import argparse
import hashlib
import json
from pathlib import Path

DOMAIN_FIELDS = {'enabled', 'prompts_per_step', 'max_response_tokens', 'eval_prompts'}
DEFAULTS = {'enabled': True, 'max_response_tokens': 2048, 'eval_prompts': 'all'}
ROLES = ('system', 'user', 'assistant')
# Calendar stopped improving on eval around its third pass over its prompts; keep every domain at or below.
MAX_PASSES = 3
TEACHERS_PER_GPU = 3  # ~20 GB of bf16 weights each, with room for prefill KV on an H200
# Files that must be byte-identical between student and teacher: the per-token log-probs only line up
# when both read the same token ids through the same chat template.
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json',
                   'added_tokens.json', 'chat_template.jinja', 'chat_template.json')
# config.json keys that may differ between checkpoints of one architecture.
CONFIG_VOLATILE = {'transformers_version', '_name_or_path', 'torch_dtype', 'dtype', 'use_cache'}


def _positive(value):
    return type(value) is int and value >= 1


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_prompts(path):
    """Validate a prompts/eval jsonl file. Returns (ids, mean prompt characters) or raises with every problem."""
    ids, chars, problems = set(), 0, []
    with open(path, encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            where = f'{path.name} line {number}'
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                problems.append(f'{where}: not JSON ({error.msg})')
                continue
            if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'].strip():
                problems.append(f'{where}: needs a nonempty string "id"')
                continue
            if row['id'] in ids:
                problems.append(f'{where}: duplicate id {row["id"]!r}')
            ids.add(row['id'])
            messages = row.get('messages')
            if (not isinstance(messages, list) or not messages
                    or any(not isinstance(m, dict) or m.get('role') not in ROLES or not isinstance(m.get('content'), str)
                           for m in messages)):
                problems.append(f'{where}: "messages" must be a list of {{role: system|user|assistant, content: str}}')
            elif messages[-1]['role'] != 'user':
                problems.append(f'{where}: the last message must be from the user (the student answers it)')
            else:
                chars += sum(len(m['content']) for m in messages)
            if 'max_response_tokens' in row and not _positive(row['max_response_tokens']):
                problems.append(f'{where}: "max_response_tokens" must be a positive integer')
            unknown = set(row) - {'id', 'messages', 'max_response_tokens'}
            if unknown:
                problems.append(f'{where}: unknown fields {sorted(unknown)}')
            if len(problems) >= 20:
                problems.append(f'{path.name}: stopping after 20 problems')
                break
    if not ids and not problems:
        problems.append(f'{path.name}: no prompts')
    if problems:
        raise ValueError('\n'.join(problems))
    return ids, chars / len(ids)


def read_settings(path):
    try:
        settings = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'{path}: {error}') from None
    if not isinstance(settings, dict) or set(settings) - DOMAIN_FIELDS:
        raise ValueError(f'{path}: allowed fields are {sorted(DOMAIN_FIELDS)}')
    settings = {**DEFAULTS, **settings}
    if type(settings['enabled']) is not bool:
        raise ValueError(f'{path}: "enabled" must be true or false')
    if not _positive(settings.get('prompts_per_step')):
        raise ValueError(f'{path}: "prompts_per_step" is required, a positive integer (this domain\'s share of each batch)')
    if not _positive(settings['max_response_tokens']):
        raise ValueError(f'{path}: "max_response_tokens" must be a positive integer')
    if settings['eval_prompts'] != 'all' and not _positive(settings['eval_prompts']):
        raise ValueError(f'{path}: "eval_prompts" must be "all" or a positive integer')
    return settings


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


def load(root):
    """Read and check a distillation folder. Returns {'root', 'student', 'domains'}; raises with every problem."""
    root = Path(root)
    problems = []
    student = {'hf': root / 'student' / 'hf', 'mcore': root / 'student' / 'mcore'}
    if not (student['hf'] / 'config.json').is_file():
        problems.append(f'{student["hf"]}: the student HF checkpoint (config.json) is missing')
    if not (student['mcore'] / 'latest_checkpointed_iteration.txt').is_file():
        problems.append(f'{student["mcore"]}: the student Megatron checkpoint (latest_checkpointed_iteration.txt) is missing')
    domains = {}
    folder = root / 'domains'
    names = sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
    if not names:
        problems.append(f'{folder}: no domain folders')
    for name in names:
        base = folder / name
        missing = [f for f in ('prompts.jsonl', 'eval.jsonl', 'domain.json') if not (base / f).is_file()]
        if not (base / 'teacher').is_dir():
            missing.append('teacher/')
        if missing:
            problems.append(f'domain {name}: missing {", ".join(missing)}')
            continue
        try:
            settings = read_settings(base / 'domain.json')
            train_ids, chars = read_prompts(base / 'prompts.jsonl')
            eval_ids, _ = read_prompts(base / 'eval.jsonl')
        except ValueError as error:
            problems.append(f'domain {name}:\n  ' + str(error).replace('\n', '\n  '))
            continue
        shared = train_ids & eval_ids
        if shared:
            problems.append(f'domain {name}: {len(shared)} eval.jsonl ids are also training prompts '
                            f'(e.g. {sorted(shared)[0]!r}); eval must be held out')
        if settings['eval_prompts'] != 'all' and settings['eval_prompts'] > len(eval_ids):
            problems.append(f'domain {name}: eval_prompts {settings["eval_prompts"]} > {len(eval_ids)} eval prompts')
        teacher = (base / 'teacher').resolve()
        if settings['enabled'] and (student['hf'] / 'config.json').is_file():
            problems += [f'domain {name}: teacher {problem}' for problem in compare_checkpoints(student['hf'], teacher)]
        domains[name] = {**settings, 'prompts': base / 'prompts.jsonl', 'eval': base / 'eval.jsonl',
                         'teacher': teacher, 'pool': len(train_ids), 'eval_pool': len(eval_ids),
                         'prompt_chars': chars}
    if names and not any(d['enabled'] for d in domains.values()) and not problems:
        problems.append('no domain is enabled')
    if problems:
        raise ValueError('\n'.join(problems))
    return {'root': root, 'student': student, 'domains': domains}


def enabled(spec):
    return {name: d for name, d in spec['domains'].items() if d['enabled']}


def batch_prompts(spec):
    return sum(d['prompts_per_step'] for d in enabled(spec).values())


def max_passes(domain, steps):
    """Times a domain uses its prompts in `steps` steps (no refills in distillation)."""
    return domain['prompts_per_step'] * steps / domain['pool']


def teachers(spec):
    """One server per distinct teacher folder: {path: {'name', 'domains', 'load'}}. Two domains whose
    teacher/ resolves to the same folder share it. Load ~ tokens to prefill per step."""
    out = {}
    for name, d in enabled(spec).items():
        entry = out.setdefault(d['teacher'], {'name': name, 'domains': [], 'load': 0.})
        if name not in entry['domains']:
            entry['domains'].append(name)
        entry['load'] += d['prompts_per_step'] * (d['prompt_chars'] / 4 + d['max_response_tokens'])
    for entry in out.values():
        entry['name'] = '+'.join(entry['domains'])
    return out


def placement(spec, gpus, port, per_gpu=TEACHERS_PER_GPU):
    """Pack teachers onto GPUs by load: the heaviest first, each onto the least-loaded GPU with room.
    Returns [{'name', 'path', 'domains', 'gpu', 'port'}] in port order."""
    servers = sorted(teachers(spec).items(), key=lambda kv: -kv[1]['load'])
    if len(servers) > len(gpus) * per_gpu:
        raise ValueError(f'{len(servers)} teachers do not fit {len(gpus)} teacher GPUs at {per_gpu} per GPU: '
                         f'add teacher GPUs or share a teacher folder between domains')
    load, count, plan = {g: 0. for g in gpus}, {g: 0 for g in gpus}, []
    for i, (path, entry) in enumerate(servers):
        gpu = min((g for g in gpus if count[g] < per_gpu), key=lambda g: (load[g], count[g]))
        load[gpu] += entry['load']
        count[gpu] += 1
        plan.append({'name': entry['name'], 'path': path, 'domains': entry['domains'], 'gpu': gpu, 'port': port + i})
    return plan


def check_batch(spec, samples_per_prompt, policy_gpus):
    total = batch_prompts(spec) * samples_per_prompt
    if total % policy_gpus:
        raise ValueError(f'the batch ({batch_prompts(spec)} prompts x {samples_per_prompt} = {total} answers) must divide '
                         f'by the {policy_gpus} student GPUs: change prompts_per_step in some domain.json')
    return total


def _grid(rows, left=(0,)):
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ['  '.join(c.ljust(w) if i in left else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))).rstrip()
             for r in rows]
    lines.insert(1, '-' * len(lines[0]))
    return lines


def table(spec, steps, samples_per_prompt=1, policy_gpus=None, teacher_gpus=(6, 7), port=8100):
    rows = [('domain', 'on', 'prompts/step', 'prompts', 'eval prompts', 'max response', 'teacher', f'max passes in {steps} steps')]
    names = {d: e['name'] for e in teachers(spec).values() for d in e['domains']}
    for name, d in spec['domains'].items():
        evals = d['eval_pool'] if d['eval_prompts'] == 'all' else d['eval_prompts']
        rows.append((name, 'yes' if d['enabled'] else '-', str(d['prompts_per_step']), str(d['pool']), str(evals),
                     str(d['max_response_tokens']), names.get(name, '-'), f'{max_passes(d, steps):.1f}'))
    lines = [f'student: {spec["student"]["hf"].parent}', ''] + _grid(rows) + ['']
    plan = placement(spec, list(teacher_gpus), port)
    lines += _grid([('teacher', 'GPU', 'port', 'domains', 'checkpoint')]
                   + [(p['name'], str(p['gpu']), str(p['port']), ','.join(p['domains']), str(p['path'])) for p in plan],
                   left=(0, 3, 4))
    on = enabled(spec)
    total = batch_prompts(spec) * samples_per_prompt
    lines += ['', f'{len(on)} of {len(spec["domains"])} domains enabled; {batch_prompts(spec)} prompts per step x '
                  f'{samples_per_prompt} answers = {total} answers; {len(plan)} teacher servers on GPUs '
                  f'{",".join(str(g) for g in teacher_gpus)}']
    if policy_gpus and total % policy_gpus:
        lines.append(f'error: {total} answers do not divide by the {policy_gpus} student GPUs; change prompts_per_step')
    lines += [f'note: {name} can use its {d["pool"]} prompts up to {max_passes(d, steps):.1f} times in {steps} steps '
              f'(more than {MAX_PASSES}: it may memorise them); lower its prompts_per_step'
              for name, d in on.items() if max_passes(d, steps) > MAX_PASSES]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Preview and check a multi-teacher distillation folder.')
    parser.add_argument('root')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--samples-per-prompt', type=int, default=1)
    parser.add_argument('--policy-gpus', type=int, default=6)
    parser.add_argument('--teacher-gpus', default='6,7')
    parser.add_argument('--teacher-port', type=int, default=8100)
    args = parser.parse_args()
    gpus = [int(g) for g in args.teacher_gpus.split(',') if g.strip()]
    try:
        spec = load(args.root)
        text = table(spec, args.steps, args.samples_per_prompt, args.policy_gpus, gpus, args.teacher_port)
    except ValueError as error:
        raise SystemExit(f'{args.root}:\n{error}')
    print(text)
    if 'error:' in text:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
