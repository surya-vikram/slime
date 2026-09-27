"""Verified final-turn SFT using the exact rollout chat template.

The native Qwen3 per-message mask builder changes historical thinking blocks.
For this final-turn-only adapter, require the full rendered conversation to have
the rollout prompt as an exact token prefix; never guess offsets or truncate.
"""
import copy
import json
import math

from .core import digest


def validate_provenance(metadata):
    """Published references are provenance, not fabricated model-judge grades."""
    if metadata.get('split') != 'rl_train':
        raise ValueError('SFT requires train-only provenance')
    if metadata.get('scorer_protocol'):
        return
    source = metadata.get('reference_source', {})
    revision = source.get('revision', '')
    row_hash = source.get('row_hash', metadata.get('source_row_hash', ''))
    if (not source.get('repo') or len(revision) != 40 or len(row_hash) != 64
            or not metadata.get('family_id')
            or not (metadata.get('verification') or metadata.get('verification_status'))):
        raise ValueError('Missing pinned published-reference provenance')


def demonstration(row, response, grade, protocol, forbidden_families):
    if row.get('split', 'rl_train') != 'rl_train' or row['family_id'] in forbidden_families:
        raise ValueError('SFT source overlaps held-out data or is not rl_train')
    if not protocol or response.get('finish_reason') != 'stop' or not response.get('text', '').strip():
        raise ValueError('Incomplete/unidentified SFT demonstration')
    score = grade.get('score')
    if (grade.get('status') != 'valid' or type(score) not in (int, float)
            or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError('Invalid demonstration grade')
    accepted = (grade.get('passed') is True and score == 1.) if row['binary'] else (
        score >= .75 and grade.get('components', {}).get('acceptability') is True)
    if not accepted:
        raise ValueError('Demonstration failed its task-aware quality gate')
    messages = copy.deepcopy(row['messages'])
    if not messages or messages[-1]['role'] != 'user':
        raise ValueError('SFT prompt must end with user')
    for message in messages:
        message['step_loss_mask'] = 0
    messages.append({'role': 'assistant', 'content': response['text'], 'step_loss_mask': 1})
    return {'messages': messages, 'metadata': {
        'parent_id': row['id'], 'family_id': row['family_id'], 'task': row['task'],
        'source_row_hash': digest(row), 'response_hash': digest(response),
        'grade': grade, 'scorer_protocol': protocol, 'split': 'rl_train'}}


def tokenize_final(tokenizer, messages, template_kwargs, max_length):
    if not messages or messages[-1]['role'] != 'assistant' or not messages[-1]['content'].strip():
        raise ValueError('Missing final assistant target')
    if any(m.get('step_loss_mask') != 0 for m in messages[:-1]) or messages[-1].get('step_loss_mask') != 1:
        raise ValueError('Only the final assistant turn may carry SFT loss')
    forbidden = {'tokenize', 'add_generation_prompt', 'return_dict'} & template_kwargs.keys()
    if forbidden:
        raise ValueError('Template kwargs override required tokenization controls')
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=True,
        add_generation_prompt=True, return_dict=False, **template_kwargs)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=False, return_dict=False, **template_kwargs)
    if tokens[:len(prefix)] != prefix:
        raise ValueError('Full SFT conversation does not match rollout prompt token prefix')
    if len(tokens) > max_length or len(tokens) <= len(prefix):
        raise ValueError('SFT sequence exceeds context or has an empty target')
    tail = tokens[len(prefix):]
    if tokenizer.eos_token_id not in tail:
        raise ValueError('Final assistant target is missing EOS')
    return tokens, [0] * len(prefix) + [1] * len(tail)


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    if evaluation or not args.rollout_global_dataset:
        raise ValueError('SFT adapter requires a training-only global dataset')
    from slime.utils.processing_utils import load_tokenizer
    from slime.utils.mask_utils import get_response_lengths
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    kwargs = getattr(args, 'apply_chat_template_kwargs', {}) or {}
    if isinstance(kwargs, str):
        kwargs = json.loads(kwargs)
    groups = data_buffer.get_samples(args.rollout_batch_size)
    for group in groups:
        if len(group) != 1:
            raise ValueError('SFT needs exactly one verified demonstration per prompt')
        sample = group[0]
        validate_provenance(sample.metadata)
        tokens, mask = tokenize_final(tokenizer, sample.prompt, kwargs, args.seq_length)
        sample.tokens = tokens
        sample.response_length = get_response_lengths([mask])[0]
        sample.loss_mask = mask[-sample.response_length:]
        sample.reward = 0
    return groups


def freeze_candidates(candidates, tokenizer, kwargs, max_length, batch_size):
    """Token-admit without truncation; report all exclusions and dropped tail rows."""
    from collections import Counter
    if batch_size < 1:
        raise ValueError('Positive SFT batch size required')
    accepted, excluded, seen = [], [], set()
    for row in candidates:
        meta = row['metadata']
        validate_provenance(meta)
        if meta['family_id'] in seen:
            raise ValueError('Duplicate SFT family')
        seen.add(meta['family_id'])
        try:
            tokens, mask = tokenize_final(tokenizer, row['messages'], kwargs, max_length)
        except ValueError as exc:
            excluded.append({'id': meta['parent_id'], 'reason': str(exc)})
            continue
        row = copy.deepcopy(row)
        row['metadata'].update(total_tokens=len(tokens), supervised_tokens=sum(mask))
        accepted.append(row)
    # Native Slime batches wrap at epoch boundaries. Never wrap this one-pass SFT.
    usable = len(accepted) // batch_size * batch_size
    dropped = [r['metadata']['parent_id'] for r in accepted[usable:]]
    accepted = accepted[:usable]
    if not accepted or {r['metadata']['task'] for r in accepted} != {r['metadata']['task'] for r in candidates}:
        raise ValueError('Token admission/batch rounding removed a route; inspect candidates or reduce batch')
    counts, token_counts = Counter(), Counter()
    for row in accepted:
        counts[row['metadata']['task']] += 1
        token_counts[row['metadata']['task']] += row['metadata']['supervised_tokens']
    return accepted, {'rows': len(accepted), 'batches': len(accepted) // batch_size,
        'batch_size': batch_size, 'sequence_cap': max_length, 'chat_template_kwargs': kwargs,
        'counts': dict(counts), 'supervised_tokens': dict(token_counts),
        'excluded': excluded, 'batch_tail_excluded_ids': dropped, 'data_hash': digest(accepted)}


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from transformers import AutoTokenizer
    from .core import write_json
    parser = argparse.ArgumentParser(description='Freeze verified SFT candidates after exact token admission')
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sequence-cap', type=int, default=16384)
    parser.add_argument('--chat-template-kwargs', type=json.loads, default={'enable_thinking': False})
    options = parser.parse_args()
    target = Path(options.output)
    if target.exists():
        raise ValueError('Refusing to overwrite frozen SFT artifact')
    rows, report = freeze_candidates(json.loads(Path(options.candidates).read_text()),
        AutoTokenizer.from_pretrained(options.tokenizer), options.chat_template_kwargs,
        options.sequence_cap, options.batch_size)
    report['tokenizer_files'] = {p.name: __import__('hashlib').sha256(p.read_bytes()).hexdigest()
        for p in Path(options.tokenizer).glob('*') if p.name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    write_json(str(target) + '.manifest.json', report)
    print(json.dumps(report, indent=2))
