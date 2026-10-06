"""Resolve mixrl/config.env, the task file and the launcher settings before any GPU service starts."""
import json
import os
import hashlib
import math
from pathlib import Path

from slime_plugins.models.chimera_context import resolve_context
from slime_plugins.models.chimera_geometry import validate_geometry, validate_mcore_geometry

from .core import MIMO_LENGTH_PENALTY, digest, load_split, write_json
from .runtime import request
from .routes import validate_route
from . import tasks as task_file
from .code_admission import code_exclusions


def scorer_urls(env):
    """MIXRL_SCORER_URL: one reward-service URL, or several separated by commas."""
    urls = [u.strip().rstrip('/') for u in env['MIXRL_SCORER_URL'].split(',') if u.strip()]
    if not urls:
        raise ValueError('MIXRL_SCORER_URL is empty')
    return urls


def checkpoint_identity(root):
    """Pin bytes, not just paths; deliberately read once during preflight."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Checkpoint directory missing: {root}')
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and '.cache' not in path.relative_to(root).parts:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
            files[str(path.relative_to(root))] = h.hexdigest()
    if not files:
        raise ValueError(f'Checkpoint directory empty: {root}')
    return {'path': str(root), 'files': files}


def resolve():
    env = os.environ
    tasks_path = Path(env.get('MIXRL_TASKS_CONFIG', str(task_file.DEFAULT_PATH))).resolve()
    spec = task_file.load(tasks_path)
    c = {'tasks_file': str(tasks_path), **task_file.resolved(spec)}
    quotas, caps = c['quotas'], c['caps']
    c.update({'data_dir': env['MIXRL_DATA_DIR'],
        'run_dir': env['RUN_DIR'], 'scorer_url': scorer_urls(env)[0],
        'truncation': env['MIXRL_TRUNCATION'], 'seed': int(env['MIXRL_SEED']),
        'context': int(env['MODEL_CONTEXT_LENGTH']),
        'inflight_groups': int(env['MIXRL_INFLIGHT_GROUPS']),
        'response_concurrency': int(env['MIXRL_RESPONSE_CONCURRENCY']),
        'refill_rounds': int(env['MIXRL_REFILL_ROUNDS']),
        'collection_timeout': int(env['MIXRL_COLLECTION_TIMEOUT']),
        'reward_timeout': int(env['MIXRL_REWARD_TIMEOUT']),
        'reward_attempts': int(env['MIXRL_REWARD_ATTEMPTS']),
        'reward_concurrency': int(env.get('MIXRL_REWARD_CONCURRENCY', '8')),
        'samples_per_prompt': int(env['N_SAMPLES_PER_PROMPT']),
        'policy_gpus': int(env['POLICY_GPUS']),
        'expert_model_parallel_size': int(env.get('EXPERT_MODEL_PARALLEL_SIZE', '1'))})
    c['model_profile'] = env['MODEL_PROFILE']
    c['chimera_model_size'] = env.get('CHIMERA_MODEL_SIZE', 'full')
    if c['chimera_model_size'] not in ('full', 'tiny'):
        raise ValueError('CHIMERA_MODEL_SIZE must be full or tiny')
    c['context_headroom'] = int(env.get('MIXRL_CONTEXT_HEADROOM', '0'))
    if not 0 <= c['context_headroom'] < c['context'] - max(caps.values()):
        raise ValueError('Context headroom must leave space for prompt and full response allowance')
    if c['context'] > 16384:
        raise ValueError('DP-only MixRL requires prompt plus response within 16384 tokens; CP is out of scope')
    c['objective'] = env.get('MIXRL_OBJECTIVE', 'dapo')
    c['eval_interval'] = int(env.get('EVAL_INTERVAL', '10'))
    c['eval_updates'] = [int(value) for value in env.get('MIXRL_EVAL_UPDATES', '').split(',') if value]
    if c['eval_interval'] < 1:
        raise ValueError('EVAL_INTERVAL must be positive')
    if (c['eval_updates'] != sorted(set(c['eval_updates'])) or
            any(update < 1 or update > int(env.get('NUM_ROLLOUT', '1')) for update in c['eval_updates'])):
        raise ValueError('MIXRL_EVAL_UPDATES must be sorted unique 1-based updates within NUM_ROLLOUT')
    c['routing_replay'] = int(env.get('CHIMERA_ROUTING_REPLAY', '0'))
    c['router_metrics'] = int(env.get('MIXRL_ROUTER_METRICS', '1'))
    if c['router_metrics'] not in (0, 1):
        raise ValueError('MIXRL_ROUTER_METRICS must be 0 or 1')
    if c['routing_replay'] not in (0, 1) or (c['routing_replay'] and c['model_profile'] != 'chimera'):
        raise ValueError('Expert-path replay is Chimera-only, opt-in 0/1')
    c['is_positive_bounds'] = json.loads(env.get('MIXRL_IS_POSITIVE_BOUNDS', '[0.2,5.0]'))
    c['is_negative_bounds'] = json.loads(env.get('MIXRL_IS_NEGATIVE_BOUNDS', '[0.2,5.0]'))
    if c['objective'] not in ('dapo', 'mimo'):
        raise ValueError('MIXRL_OBJECTIVE must be dapo or mimo')
    for bounds in (c['is_positive_bounds'], c['is_negative_bounds']):
        if (not isinstance(bounds, list) or len(bounds) != 2 or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds) or
            not 0 < bounds[0] <= 1 <= bounds[1]):
            raise ValueError('Invalid fixed importance bounds')
    c['execution_mode'] = env.get('EXECUTION_MODE', 'sync')
    c['colocate'] = int(env.get('COLOCATE', '1'))
    c['rollout_gpus'] = int(env.get('ROLLOUT_GPUS', env['POLICY_GPUS']))
    c['use_rollout_logprobs'] = int(env.get('USE_ROLLOUT_LOGPROBS', '0'))
    if c['objective'] == 'dapo' and c['use_rollout_logprobs']:
        # DAPO's PPO ratio reads the recomputed old log-probs; SGLang's would fold numeric differences into it.
        raise ValueError('MIXRL_OBJECTIVE=dapo needs USE_ROLLOUT_LOGPROBS=0')
    c['optimizer_schedule_rollouts'] = int(env.get('NUM_ROLLOUT', '5'))
    if c['rollout_gpus'] < 1 or c['execution_mode'] not in ('sync', 'async'):
        raise ValueError('Invalid scheduling configuration')
    c['chat_template_kwargs'] = json.loads(env['CHAT_TEMPLATE_KWARGS'])
    if not isinstance(c['chat_template_kwargs'], dict):
        raise ValueError('CHAT_TEMPLATE_KWARGS must be a JSON object')
    if {'tokenize', 'add_generation_prompt'} & c['chat_template_kwargs'].keys():
        raise ValueError('Chat template kwargs cannot override tokenization/generation mode')
    c['lr'] = float(env['LR'])
    c['weight_decay'] = float(env.get('WEIGHT_DECAY', '0'))
    c['clip_grad'] = float(env.get('CLIP_GRAD', '1.0'))
    c['adam_beta1'] = float(env.get('ADAM_BETA1', '0.9'))
    c['adam_beta2'] = float(env.get('ADAM_BETA2', '0.98'))
    if c['weight_decay'] != 0:
        raise ValueError('The approved MixRL baseline requires WEIGHT_DECAY=0')
    if not math.isfinite(c['clip_grad']) or c['clip_grad'] <= 0:
        raise ValueError('CLIP_GRAD must be finite and positive')
    if any(not math.isfinite(c[k]) or not 0 <= c[k] < 1 for k in ('adam_beta1', 'adam_beta2')):
        raise ValueError('Adam beta values must be finite and in [0, 1)')
    c['max_tokens_per_gpu'] = int(env['MAX_TOKENS_PER_GPU'])
    if not math.isfinite(c['lr']) or c['lr'] <= 0 or c['max_tokens_per_gpu'] < 1:
        raise ValueError('Invalid optimizer LR or token microbatch budget')
    for key in ('context', 'inflight_groups', 'response_concurrency',
                'collection_timeout', 'reward_timeout', 'reward_attempts', 'reward_concurrency', 'eval_samples', 'policy_gpus'):
        if c[key] < 1:
            raise ValueError(f'{key} must be positive')
    if (c['expert_model_parallel_size'] < 1 or
            c['policy_gpus'] % c['expert_model_parallel_size']):
        raise ValueError('EXPERT_MODEL_PARALLEL_SIZE must positively divide POLICY_GPUS')
    if c['samples_per_prompt'] < 2 or c['truncation'] not in ('mask', 'zero'):
        raise ValueError('Invalid group size or truncation policy')
    oversample = float(env.get('MIXRL_OVERSAMPLE', '0'))
    if not 0 <= oversample <= 2:
        raise ValueError('MIXRL_OVERSAMPLE must be between 0 (off) and 2 (spares as a fraction of each quota)')
    if oversample:
        c['oversample'] = oversample  # 0 keeps the resolved config unchanged
    if c['refill_rounds'] < 0:
        raise ValueError('MIXRL_REFILL_ROUNDS must be 0 (off) or a positive number of rounds')
    length_penalty = env.get('MIXRL_LENGTH_PENALTY', '0')
    if length_penalty not in ('0', '1'):
        raise ValueError('MIXRL_LENGTH_PENALTY must be 0 or 1')
    # MiMo-V2.6 group-relative length penalty with its public recipe values.
    c['length_penalty'] = dict(MIMO_LENGTH_PENALTY) if length_penalty == '1' else None
    c['rollout_temperature'] = float(env.get('ROLLOUT_TEMPERATURE', '1.0'))
    c['rollout_top_p'] = float(env.get('ROLLOUT_TOP_P', '1.0'))
    if not (math.isfinite(c['rollout_temperature']) and c['rollout_temperature'] > 0
            and math.isfinite(c['rollout_top_p']) and 0 < c['rollout_top_p'] <= 1):
        raise ValueError('ROLLOUT_TEMPERATURE must be positive and ROLLOUT_TOP_P in (0, 1]')
    c['rollout_top_k'] = int(env.get('ROLLOUT_TOP_K', '-1'))
    if c['rollout_top_k'] == 0 or c['rollout_top_k'] < -1:
        raise ValueError('ROLLOUT_TOP_K must be -1 (off) or a positive number of tokens')
    if c['rollout_top_k'] > 0 and c['rollout_top_p'] == 1:
        # Slime records and replays candidate sets only when top-p < 1; top-k alone would leave the
        # loss renormalizing over the full vocabulary while SGLang sampled from the top k.
        raise ValueError('ROLLOUT_TOP_K needs ROLLOUT_TOP_P < 1, which turns on candidate-set replay')
    c['lr_warmup_steps'] = int(env.get('LR_WARMUP_STEPS', '0'))
    if c['lr_warmup_steps'] < 0:
        raise ValueError('LR_WARMUP_STEPS must be 0 or positive')
    if c['rollout_batch_size'] * c['samples_per_prompt'] % c['policy_gpus']:
        raise ValueError('v0 requires response batch divisible by policy DP; do not silently round')
    if max(caps.values()) >= c['context']:
        raise ValueError('Output budget leaves no prompt space')
    if not (Path(env['HF_CHECKPOINT']) / 'config.json').is_file():
        raise ValueError('HF checkpoint config.json missing')
    c['hf_checkpoint'] = checkpoint_identity(env['HF_CHECKPOINT'])
    if env.get('INITIAL_ACTOR_CHECKPOINT'):
        initial = Path(env['INITIAL_ACTOR_CHECKPOINT'])
        if not (initial / 'latest_checkpointed_iteration.txt').is_file():
            raise ValueError('INITIAL_ACTOR_CHECKPOINT requires a saved MCore checkpoint')
        c['initial_actor_checkpoint'] = checkpoint_identity(initial)
    if c['model_profile'] == 'chimera':
        if not (Path(env['MCORE_CHECKPOINT']) / 'latest_checkpointed_iteration.txt').is_file():
            raise ValueError('Converted Chimera checkpoint required')
        c['mcore_checkpoint'] = checkpoint_identity(env['MCORE_CHECKPOINT'])
        validate_geometry(json.loads((Path(env['HF_CHECKPOINT']) / 'config.json').read_text()),
                          c['chimera_model_size'])
    elif c['model_profile'] == 'qwen3-0.6B':
        hf_config = json.loads((Path(env['HF_CHECKPOINT']) / 'config.json').read_text())
        expected = dict(model_type='qwen3', num_hidden_layers=28, hidden_size=1024,
                        intermediate_size=3072, num_attention_heads=16, num_key_value_heads=8,
                        head_dim=128, vocab_size=151936, rms_norm_eps=1e-6,
                        rope_theta=1000000, tie_word_embeddings=True)
        if any(hf_config.get(k) != v for k, v in expected.items()):
            raise ValueError('HF config does not match native qwen3-0.6B model definition')
        if c['context'] > hf_config['max_position_embeddings']:
            raise ValueError('Reference context exceeds native model window')
        c['mcore_checkpoint'] = None # Native HF loading for the reference model only.
    else:
        raise ValueError('Unsupported model profile')
    hf_config = json.loads((Path(env['HF_CHECKPOINT']) / 'config.json').read_text())
    sequence_cap = int(env.get('TRAIN_SEQUENCE_LENGTH', str(c['context'])))
    if sequence_cap != c['context']:
        raise ValueError('MODEL_CONTEXT_LENGTH and TRAIN_SEQUENCE_LENGTH disagree')
    context_override = env.get('CHIMERA_CONTEXT_OVERRIDE', '0')
    if context_override not in ('0', '1'):
        raise ValueError('CHIMERA_CONTEXT_OVERRIDE must be 0 or 1')
    c['checkpoint_context'] = resolve_context(
        hf_config, c['model_profile'], env.get('CONTEXT_PHASE', 'auto'), sequence_cap,
        (env.get('INITIAL_ACTOR_CHECKPOINT') or env['MCORE_CHECKPOINT']) if c['model_profile'] == 'chimera' else None,
        allow_mcore_context_override=context_override == '1')
    if c['model_profile'] == 'chimera':
        import yaml
        metadata_path = Path(c['checkpoint_context']['mcore_context_provenance']['source'])
        validate_mcore_geometry(yaml.safe_load(metadata_path.read_text())['model'], c['chimera_model_size'])
    if c['context'] > c['max_tokens_per_gpu']:
        raise ValueError('DP-only sequence cap exceeds MAX_TOKENS_PER_GPU; packing does not split long samples')
    rows, manifest = load_split(c['data_dir'], 'rl_train')
    val, _ = load_split(c['data_dir'], 'rl_val')
    task_file.check_data(spec, rows, val)
    for row in rows + val:
        if row['task'] in quotas:
            validate_route(row, c['routes'])
    if {r['family_id'] for r in rows} & {r['family_id'] for r in val}:
        raise ValueError('Train/validation overlap')
    urls = scorer_urls(env)
    if len(urls) > 1:
        c['scorer_urls'] = urls  # several reward-service processes; one URL keeps the old config
    health = request(urls[0] + '/health', timeout=30)
    task_file.check_scorer(spec, health)
    c['scorer_protocol'] = health['protocol_id']
    c['judge'] = health['judge']['model']
    admission = request(urls[0] + '/admission', timeout=30)
    for url in urls[1:]:
        # Every process must grade identically: same data, settings and judge.
        other = request(url + '/health', timeout=30)
        if other['protocol_id'] != c['scorer_protocol'] or request(url + '/admission', timeout=30) != admission:
            raise ValueError(f'Reward service {url} differs from {urls[0]}; restart them together with mixrl/reward.sh')
    if admission['protocol_id'] != c['scorer_protocol']:
        raise ValueError('Scoring service changed during preflight')
    c['excluded_row_ids'] = sorted(admission['excluded_rows'])
    if 'apps' in quotas:
        excluded, audit_hash = code_exclusions(rows + val, env.get('MIXRL_CODE_AUDIT_DIR'))
        c['code_audit_hash'] = audit_hash
        c['code_exclusion_reasons'] = excluded
        c['excluded_row_ids'] = sorted(set(c['excluded_row_ids']) | excluded.keys())
    c['data_hash'] = digest(manifest)
    c['implementation_hash'] = digest({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))})
    repo = Path(__file__).resolve().parents[2]
    c['launcher_hash'] = digest((repo / 'mixrl/internal/launch.sh').read_text())
    c['training_loop_hash'] = digest((repo / ('train_async.py' if c['execution_mode'] == 'async' else 'train.py')).read_text())
    c['backend_model_hash'] = digest((repo / 'slime/backends/megatron_utils/model.py').read_text())
    c['rollout_manager_hash'] = digest((repo / 'slime/ray/rollout.py').read_text())
    c['backend_loss_hash'] = digest((repo / 'slime/backends/megatron_utils/loss.py').read_text())
    c['backend_reducer_hash'] = digest((repo / 'slime/backends/megatron_utils/cp_utils.py').read_text())
    c['routing_patch_hash'] = digest((repo / 'examples/chimera/patches/sglang-transformers-routing-capture.patch').read_text())
    c['model_definition_hash'] = digest((repo / 'scripts/models' / (c['model_profile'] + '.sh')).read_text())
    c['context_contract_hash'] = digest((repo / 'slime_plugins/models/chimera_context.py').read_text())
    c['geometry_contract_hash'] = digest((repo / 'slime_plugins/models/chimera_geometry.py').read_text())
    c['backend_actor_hash'] = digest((repo / 'slime/backends/megatron_utils/actor.py').read_text())
    c['native_replay_hash'] = digest((repo / 'slime/utils/routing_replay.py').read_text())
    c['chimera_provider_hash'] = digest((repo / 'slime_plugins/models/chimera.py').read_text())
    for key in ('CHIMERA_FP32_LM_HEAD', 'CHIMERA_MATCH_RMSNORM',
                'CHIMERA_MATCH_DENSE_SWIGLU', 'CHIMERA_SGLANG_FULL_BF16_REDUCTION'):
        if env.get(key, '0') not in ('0', '1'):
            raise ValueError(f'{key} must be 0 or 1')
        c[key.lower()] = env.get(key, '0') == '1'
    c['precision_adapter_hash'] = digest({name: (repo / 'slime_plugins/models' / name).read_text()
        for name in ('chimera_precision.py', 'chimera_sglang_precision.py')})
    c['runtime_registration_hash'] = digest((repo / 'examples/chimera/runtime/sitecustomize.py').read_text())
    path = Path(env['CHIMERA_MIXRL_CONFIG'])
    if path.exists() and env.get('MIXRL_EXTEND_CONSTANT_HORIZON') == '1':
        previous = json.loads(path.read_text())
        old_horizon = previous['optimizer_schedule_rollouts']
        new_horizon = c['optimizer_schedule_rollouts']
        if env.get('RESUME') != '1' or new_horizon < old_horizon:
            raise ValueError('Horizon extension requires RESUME=1 and a nondecreasing horizon')
        # Preserve the whole scheduler identity/units, including WD. The native
        # backend reads this frozen horizon while the driver may run longer.
        c['optimizer_schedule_rollouts'] = old_horizon
        if c != previous:
            changed = sorted(k for k in set(previous) | set(c) if previous.get(k) != c.get(k))
            raise ValueError(f'Horizon extension cannot change any other resolved setting; differs in {", ".join(changed[:10])}')
        write_json(path.parent / f'horizon_extension_{new_horizon}.json', {
            'original_schedule_rollouts': old_horizon, 'execution_rollouts': new_horizon,
            'constant_lr': c['lr'], 'config_hash': digest(previous)})
    if path.exists() and (previous := json.loads(path.read_text())) != c:
        changed = sorted(k for k in set(previous) | set(c) if previous.get(k) != c.get(k))
        raise ValueError(f'Existing resolved config differs in {", ".join(changed[:10])}; '
                         'resume with the settings the run started with, or choose a fresh run name')
    write_json(path, c)
    print(f'MixRL tasks ({tasks_path})\n' + task_file.table(spec, c['samples_per_prompt'],
                                                     int(os.environ.get('NUM_ROLLOUT', 0)) or None, c['refill_rounds']))
    print(f'Reward service judge: {c["judge"]} ({"reachable" if health["judge"]["ready"] else "not reachable"}); '
          f'{len(c["excluded_row_ids"])} quarantined rows; resolved config: {path}')


if __name__ == '__main__':
    resolve()
