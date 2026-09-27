"""CPU-only launch contracts; checkpoint files here are explicitly fixtures."""
import contextlib
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
            rows = [{'id': split, 'family_id': split, 'task': 'mcqa', 'domain': 'knowledge',
                     'verifier': 'choice', 'binary': True,
                     'messages': [{'role': 'user', 'content': 'Question'}]}]
            path = data / 'splits' / f'{split}.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rows[0]) + '\n')
            manifest['splits'][split] = {'hash': digest(rows)}
        write_json(data / 'manifest.json', manifest)
        self.output = root / 'run/config.json'
        self.env = dict(MIXRL_QUOTAS='{"mcqa":1}', MIXRL_CAPS='{"mcqa":64}',
            MIXRL_DATA_DIR=str(data), RUN_DIR=str(root / 'run'), MIXRL_SCORER_URL='http://fixture',
            MIXRL_TRUNCATION='mask', MIXRL_SEED='42', MODEL_CONTEXT_LENGTH='256',
            MIXRL_INFLIGHT_GROUPS='1', MIXRL_RESPONSE_CONCURRENCY='2', MIXRL_MAX_ATTEMPTS='4',
            MIXRL_COLLECTION_TIMEOUT='10', MIXRL_REWARD_TIMEOUT='5', MIXRL_REWARD_ATTEMPTS='2',
            MIXRL_EVAL_SAMPLES='2', N_SAMPLES_PER_PROMPT='2', POLICY_GPUS='2',
            EXPERT_MODEL_PARALLEL_SIZE='1', ROLLOUT_BATCH_SIZE='1',
            MODEL_PROFILE='qwen3-0.6B', CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}', LR='1e-6',
            MAX_TOKENS_PER_GPU='256', HF_CHECKPOINT=str(self.hf), MCORE_CHECKPOINT=str(self.mcore),
            CHIMERA_MIXRL_CONFIG=str(self.output))

    def resolve(self, **updates):
        with patch.dict(os.environ, dict(self.env, **updates)), contextlib.redirect_stdout(io.StringIO()), \
             patch.object(configure, 'request', return_value={'protocol_id': 'fixture', 'excluded_rows': {}}):
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

    def test_reject_resume_after_lr_or_checkpoint_change(self):
        self.resolve()
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.resolve(LR='2e-6')
        (self.hf / 'weights.fixture').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.resolve()

    def test_invalid_batch_context_and_template(self):
        with self.assertRaisesRegex(ValueError, 'within 16384 tokens'):
            self.resolve(MODEL_CONTEXT_LENGTH='32768', TRAIN_SEQUENCE_LENGTH='32768',
                         MAX_TOKENS_PER_GPU='32768')
        for changes in ({'POLICY_GPUS': '3'}, {'EXPERT_MODEL_PARALLEL_SIZE': '3'},
                        {'MIXRL_CAPS': '{"mcqa":256}'},
                        {'WEIGHT_DECAY': '.1'}, {'CLIP_GRAD': 'nan'}, {'ADAM_BETA2': '1'},
                        {'EVAL_INTERVAL': '7', 'MIXRL_MAIN_EVAL_INTERVAL': '50'},
                        {'MIXRL_MAIN_EVAL_SAMPLES': '0'},
                        {'CHAT_TEMPLATE_KWARGS': '[]'}, {'LR': 'nan'},
                        {'CHAT_TEMPLATE_KWARGS': '{"tokenize":true}'},
                        {'TRAIN_SEQUENCE_LENGTH': '128'}, {'MAX_TOKENS_PER_GPU': '128'},
                        {'MIXRL_CONTEXT_HEADROOM': '-1'}, {'MIXRL_CONTEXT_HEADROOM': '256'},
                        {'MIXRL_MAX_ATTEMPTS': '0'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.resolve(**changes)

    def test_requires_conversion(self):
        (self.mcore / 'latest_checkpointed_iteration.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Convert'):
            self.resolve(MODEL_PROFILE='chimera')

    def test_refuses_changed_scorer(self):
        with patch.dict(os.environ, self.env), patch.object(configure, 'request',
                side_effect=[{'protocol_id': 'a'}, {'protocol_id': 'b', 'excluded_rows': {}}]):
            with self.assertRaisesRegex(ValueError, 'changed'):
                configure.resolve()

    def test_real_launcher_cpu_preflight(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({'protocol_id': 'fixture', 'excluded_rows': {}}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            repo = Path(__file__).resolve().parents[1]
            env = dict(os.environ, **self.env)
            env.update(RECIPE='mixrl', PREFLIGHT_ONLY='1', EXECUTION_MODE='sync',
                       CONTEXT_PHASE='auto', TRAIN_SEQUENCE_LENGTH='256',
                       MIXRL_CONTEXT_HEADROOM='16',
                       DATA_ROOT=self.temp.name, RUN_NAME='launcher-preflight',
                       MIXRL_RUNS_ROOT=str(Path(self.temp.name) / 'runs'),
                       MIXRL_SCORER_URL=f'http://127.0.0.1:{server.server_port}')
            # Exercise the actual new defaults, not fixture generation counts.
            env.pop('N_SAMPLES_PER_PROMPT', None)
            env.pop('MIXRL_EVAL_SAMPLES', None)
            result = subprocess.run(['bash', str(repo / 'examples/chimera/train.sh')],
                                    env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads((Path(env['MIXRL_RUNS_ROOT']) / env['RUN_NAME'] /
                                 'manifests/mixrl_config.json').read_text())
            self.assertEqual(config['samples_per_prompt'], 4)
            self.assertEqual(config['eval_samples'], 4)
            self.assertEqual(config['context_headroom'], 16)
            self.assertEqual(config['checkpoint_context'],
                             {'phase': 'native', 'model_max_context': 32768, 'sequence_cap': 256})
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
