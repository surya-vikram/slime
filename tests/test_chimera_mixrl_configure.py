"""CPU-only launch contracts; checkpoint files here are explicitly fixtures."""
import contextlib
import copy
import io
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from slime_plugins.chimera_mixrl import configure
from slime_plugins.chimera_mixrl.core import digest, write_json

REPO = Path(__file__).resolve().parents[1]
ROUTES = {'mcqa': ('knowledge', 'choice', 'none'), 'hotpot_train': ('grounding', 'grounded', 'always')}


def task_entry(task, enabled):
    domain, verifier, judge = ROUTES[task]
    return {'enabled': enabled, 'prompts_per_step': 1, 'eval_prompts': 'all', 'max_response_tokens': 64,
            'about': {'summary': f'{task} fixture', 'grading': 'fixture', 'answer_format': 'fixture',
                      'requires': [], 'domain': domain, 'verifier': verifier, 'judge': judge,
                      'reward': 'binary', 'train_pool': 1, 'val_pool': 1}}


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.hf = root / 'hf'
        self.mcore = root / 'mcore'
        write_json(self.hf / 'config.json', dict(model_type='qwen3', num_hidden_layers=28,
            hidden_size=1024, intermediate_size=3072, num_attention_heads=16, num_key_value_heads=8,
            head_dim=128, vocab_size=151936, rms_norm_eps=1e-6, rope_theta=1000000,
            tie_word_embeddings=True, max_position_embeddings=32768))
        self.mcore.mkdir()
        (self.mcore / 'latest_checkpointed_iteration.txt').write_text('1')
        data = root / 'data'
        manifest = {'splits': {}}
        for split in ('rl_train', 'rl_val'):
            rows = [{'id': f'{split}-{task}', 'family_id': f'{split}-{task}', 'task': task, 'domain': domain,
                     'verifier': verifier, 'binary': True,
                     'messages': [{'role': 'user', 'content': 'Question'}]}
                    for task, (domain, verifier, _) in ROUTES.items()]
            path = data / 'splits' / f'{split}.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            manifest['splits'][split] = {'hash': digest(rows)}
        write_json(data / 'manifest.json', manifest)
        self.tasks_path = root / 'tasks.json'
        self.tasks = {'eval': {'samples_per_prompt': 2},
                      'domains': {'knowledge': {'summary': 'fixture'}, 'grounding': {'summary': 'fixture'}},
                      'tasks': {'mcqa': task_entry('mcqa', True), 'hotpot_train': task_entry('hotpot_train', False)}}
        write_json(self.tasks_path, self.tasks)
        self.health = {'protocol_id': 'fixture', 'status': 'ready', 'excluded_rows': {},
                       'judge': {'model': 'fixture-judge', 'ready': False}, 'task_errors': {},
                       'task_judge': {task: judge for task, (_, _, judge) in ROUTES.items()}}
        self.output = root / 'run/config.json'
        self.env = dict(MIXRL_TASKS_CONFIG=str(self.tasks_path),
            MIXRL_DATA_DIR=str(data), RUN_DIR=str(root / 'run'), MIXRL_SCORER_URL='http://fixture',
            MIXRL_TRUNCATION='mask', MIXRL_SEED='42', MODEL_CONTEXT_LENGTH='256',
            MIXRL_INFLIGHT_GROUPS='1', MIXRL_RESPONSE_CONCURRENCY='2', MIXRL_REFILL_ROUNDS='2',
            MIXRL_COLLECTION_TIMEOUT='10', MIXRL_REWARD_TIMEOUT='5', MIXRL_REWARD_ATTEMPTS='2',
            N_SAMPLES_PER_PROMPT='2', POLICY_GPUS='2',
            EXPERT_MODEL_PARALLEL_SIZE='1',
            MODEL_PROFILE='qwen3-0.6B', CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}', LR='1e-6',
            MAX_TOKENS_PER_GPU='256', HF_CHECKPOINT=str(self.hf), MCORE_CHECKPOINT=str(self.mcore),
            CHIMERA_MIXRL_CONFIG=str(self.output))

    def edit_tasks(self, change, whole=False):
        spec = copy.deepcopy(self.tasks)
        change(spec if whole else spec['tasks'])
        write_json(self.tasks_path, spec)

    def resolve(self, health=None, **updates):
        with patch.dict(os.environ, dict(self.env, **updates)), contextlib.redirect_stdout(io.StringIO()), \
             patch.object(configure, 'request', return_value=health or self.health):
            configure.resolve()
        return json.loads(self.output.read_text())

    def test_roundtrip_and_model_template_pinned(self):
        c = self.resolve()
        self.assertEqual(c, self.resolve())
        self.assertEqual(c['chat_template_kwargs'], {'enable_thinking': False})
        self.assertIn('config.json', c['hf_checkpoint']['files'])
        self.assertEqual(c['expert_model_parallel_size'], 1)
        self.output.unlink()
        self.assertEqual(self.resolve(EXPERT_MODEL_PARALLEL_SIZE='2')['expert_model_parallel_size'], 2)

    def test_task_file_drives_quotas_caps_eval_and_batch(self):
        self.edit_tasks(lambda tasks: tasks['mcqa'].update(max_response_tokens=96))
        c = self.resolve()
        self.assertEqual((c['quotas'], c['caps'], c['rollout_batch_size']), ({'mcqa': 1}, {'mcqa': 96}, 1))
        self.assertEqual((c['eval_quotas'], c['eval_samples']), ({'mcqa': 1}, 2))
        self.assertNotIn('main_eval_quotas', c)
        self.assertEqual(c['routes'], {'mcqa': {'domain': 'knowledge', 'verifier': 'choice',
                                                'judge': 'none', 'reward': 'binary'}})
        self.assertEqual((c['tasks_file'], c['judge']), (str(self.tasks_path.resolve()), 'fixture-judge'))

    def test_judge_task_refused_without_judge(self):
        self.edit_tasks(lambda tasks: tasks['hotpot_train'].update(enabled=True))
        with self.assertRaisesRegex(ValueError, "hotpot_train: needs the judge \\(always\\), but judge "
                                                "'fixture-judge' is not reachable"):
            self.resolve()
        c = self.resolve(dict(self.health, judge={'model': 'fixture-judge', 'ready': True}))
        self.assertEqual((c['rollout_batch_size'], c['judge']), (2, 'fixture-judge'))
        # Judge-free tasks never need one.
        self.output.unlink()
        self.edit_tasks(lambda tasks: None)
        self.assertEqual(self.resolve()['quotas'], {'mcqa': 1})

    def test_task_file_must_match_data_and_scorer(self):
        cases = {
            'about.train_pool is 2': lambda t: t['mcqa']['about'].update(train_pool=2),
            'about says knowledge/equivalence': lambda t: t['mcqa']['about'].update(verifier='equivalence'),
            'about.reward graded disagrees': lambda t: t['mcqa']['about'].update(reward='graded'),
            'task file says judge=none': lambda t: t['hotpot_train']['about'].update(judge='none'),
        }
        def drop_hotpot(spec):
            spec['tasks'].pop('hotpot_train')
            spec['domains'].pop('grounding')
        cases['missing from the task file: hotpot_train'] = lambda spec: drop_hotpot(spec)
        for message, change in cases.items():
            with self.subTest(message):
                whole = message.startswith('missing')
                self.edit_tasks(change, whole=whole)
                with self.assertRaisesRegex(ValueError, message):
                    self.resolve()
        self.edit_tasks(lambda tasks: None)
        with self.assertRaisesRegex(ValueError, 'does not report judge status'):
            self.resolve({'protocol_id': 'fixture', 'excluded_rows': {}})

    def test_grading_errors_stop_launch_but_eval_size_does_not(self):
        with self.assertRaisesRegex(ValueError, 'mcqa: grading check failed: No module named regex'):
            self.resolve(dict(self.health, task_errors={'mcqa': 'No module named regex'}))
        self.edit_tasks(lambda spec: spec['eval'].update(samples_per_prompt=16), whole=True)
        self.assertEqual(self.resolve()['eval_samples'], 16)

    def test_explicit_extension_preserves_frozen_identity(self):
        original = self.resolve(NUM_ROLLOUT='10')
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.resolve(NUM_ROLLOUT='20')
        extended = self.resolve(NUM_ROLLOUT='20', RESUME='1', MIXRL_EXTEND_CONSTANT_HORIZON='1')
        self.assertEqual(original, extended)
        record = json.loads((self.output.parent / 'horizon_extension_20.json').read_text())
        self.assertEqual(record['execution_rollouts'], 20)
        with self.assertRaises(ValueError):
            self.resolve(NUM_ROLLOUT='20', RESUME='1', MIXRL_EXTEND_CONSTANT_HORIZON='1', LR='2e-6')
        with self.assertRaises(ValueError):
            self.resolve(NUM_ROLLOUT='9', RESUME='1', MIXRL_EXTEND_CONSTANT_HORIZON='1')

    def test_reject_resume_after_lr_task_or_checkpoint_change(self):
        self.resolve()
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.resolve(LR='2e-6')
        self.edit_tasks(lambda tasks: tasks['mcqa'].update(max_response_tokens=32))
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.resolve()
        self.edit_tasks(lambda tasks: None)
        (self.hf / 'weights.fixture').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.resolve()

    def test_invalid_batch_context_and_template(self):
        with self.assertRaisesRegex(ValueError, 'within 16384 tokens'):
            self.resolve(MODEL_CONTEXT_LENGTH='32768', TRAIN_SEQUENCE_LENGTH='32768',
                         MAX_TOKENS_PER_GPU='32768')
        for changes in ({'POLICY_GPUS': '3'}, {'EXPERT_MODEL_PARALLEL_SIZE': '3'},
                        {'WEIGHT_DECAY': '.1'}, {'CLIP_GRAD': 'nan'}, {'ADAM_BETA2': '1'},
                        {'EVAL_INTERVAL': '0'},
                        {'CHAT_TEMPLATE_KWARGS': '[]'}, {'LR': 'nan'},
                        {'CHAT_TEMPLATE_KWARGS': '{"tokenize":true}'},
                        {'TRAIN_SEQUENCE_LENGTH': '128'}, {'MAX_TOKENS_PER_GPU': '128'},
                        {'MIXRL_CONTEXT_HEADROOM': '-1'}, {'MIXRL_CONTEXT_HEADROOM': '256'},
                        {'MIXRL_REFILL_ROUNDS': '-1'}, {'MIXRL_LENGTH_PENALTY': 'yes'},
                        {'ROLLOUT_TOP_P': '0'}, {'ROLLOUT_TOP_P': '1.5'}, {'ROLLOUT_TEMPERATURE': '0'},
                        {'ROLLOUT_TOP_K': '0'}, {'ROLLOUT_TOP_K': '-2'}, {'ROLLOUT_TOP_K': '20', 'ROLLOUT_TOP_P': '1.0'},
                        {'LR_WARMUP_STEPS': '-1'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.resolve(**changes)
        self.edit_tasks(lambda tasks: tasks['mcqa'].update(max_response_tokens=256))
        with self.assertRaisesRegex(ValueError, 'space'):  # headroom or prompt space, whichever trips first
            self.resolve()

    def test_requires_conversion(self):
        (self.mcore / 'latest_checkpointed_iteration.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Convert'):
            self.resolve(MODEL_PROFILE='chimera')

    def test_refuses_changed_scorer(self):
        with patch.dict(os.environ, self.env), patch.object(configure, 'request',
                side_effect=[self.health, {'protocol_id': 'b', 'excluded_rows': {}}]):
            with self.assertRaisesRegex(ValueError, 'changed'):
                configure.resolve()

    def launch(self, health, **updates):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(health).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            env = dict(os.environ, **self.env)
            # Exercise the actual new defaults, not fixture generation counts.
            env.pop('N_SAMPLES_PER_PROMPT', None)
            env.update(PREFLIGHT_ONLY='1', EXECUTION_MODE='sync',
                       CONTEXT_PHASE='auto', TRAIN_SEQUENCE_LENGTH='256',
                       MIXRL_CONTEXT_HEADROOM='16',
                       DATA_ROOT=self.temp.name, RUN_NAME='launcher-preflight',
                       MIXRL_RUNS_ROOT=str(Path(self.temp.name) / 'runs'),
                       MIXRL_SCORER_URL=f'http://127.0.0.1:{server.server_port}', **updates)
            result = subprocess.run(['bash', str(REPO / 'mixrl/internal/launch.sh')],
                                    env=env, capture_output=True, text=True, timeout=20)
            return result, Path(env['MIXRL_RUNS_ROOT']) / env['RUN_NAME'] / 'manifests/mixrl_config.json'
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_real_launcher_cpu_preflight(self):
        result, path = self.launch(self.health)
        self.assertEqual(result.returncode, 0, result.stderr)
        config = json.loads(path.read_text())
        self.assertEqual(config['samples_per_prompt'], 16)  # mixrl/config.env default
        self.assertEqual(config['eval_samples'], 2)
        self.assertEqual(config['context_headroom'], 16)
        self.assertEqual(config['rollout_batch_size'], 1)
        self.assertEqual(config['checkpoint_context'],
                         {'phase': 'native', 'model_max_context': 32768, 'sequence_cap': 256})
        self.assertIn('1 of 2 tasks enabled; 1 prompts per step x 16 responses = 16 samples', result.stdout)
        self.assertIn('judge: not needed', result.stdout)
        self.assertIn('mcqa: mcqa fixture', result.stdout)

    def test_launcher_rejects_removed_task_variables(self):
        for name, value in (('MIXRL_QUOTAS', '{"mcqa":1}'), ('MIXRL_CAPS', '{"mcqa":64}'),
                            ('MIXRL_EVAL_QUOTAS', '{}'), ('ROLLOUT_BATCH_SIZE', '1'),
                            ('MIXRL_EVAL_SAMPLES', '4'), ('MIXRL_MAIN_EVAL_INTERVAL', '50')):
            with self.subTest(name):
                result, path = self.launch(self.health, **{name: value})
                self.assertEqual(result.returncode, 1)
                self.assertIn(f'{name} is no longer read for MixRL', result.stderr)
                self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
