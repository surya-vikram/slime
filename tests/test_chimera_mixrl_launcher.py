"""The in-container launcher: defaults, validation gates and manifests, before any checkpoint, Ray or GPU access."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / 'mixrl/internal/launch.sh'
CONFIG = REPO / 'mixrl/config.env'


class LauncherTests(unittest.TestCase):
    def test_config_env_is_the_only_default_for_its_settings(self):
        names = re.findall(r'^([A-Z_][A-Z0-9_]*)=', CONFIG.read_text(), re.M)
        self.assertIn('POLICY_GPUS', names)
        text = LAUNCHER.read_text()
        for name in names:
            with self.subTest(name=name):
                self.assertNotRegex(text, rf'(^|\s){name}=\$\{{{name}:-', 'second default in launch.sh')

    def test_chimera_defaults_fp32_head_and_routing_replay(self):
        text = LAUNCHER.read_text()
        block = text[text.index('MODEL_PROFILE=${MODEL_PROFILE'):text.index('CHAT_TEMPLATE_KWARGS=')]
        cases = (('chimera', None, '1 1'), ('qwen3-0.6B', None, '0 0'), ('chimera', '0', '0 1'))
        for profile, explicit, expected in cases:
            with self.subTest(profile=profile, explicit=explicit):
                env = {'PATH': os.environ.get('PATH', ''), 'MODEL_PROFILE': profile}
                if explicit is not None:
                    env['CHIMERA_FP32_LM_HEAD'] = explicit
                result = subprocess.run(['bash', '-c', block + 'printf "%s %s" "$CHIMERA_FP32_LM_HEAD" "$CHIMERA_ROUTING_REPLAY"'],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)

    def test_source_snapshot_preserves_mixrl_settings_and_chimera_registration(self):
        import tarfile
        text = LAUNCHER.read_text()
        start = text.index('tar --exclude=__pycache__ -cf "$MANIFEST_DIR/mixrl_source.tar"')
        command = text[start:text.index('examples/chimera/patches\n', start) + len('examples/chimera/patches')]
        required = ('mixrl/config.env', 'mixrl/tasks.json', 'mixrl/internal/launch.sh',
                    'slime_plugins/models/chimera_precision.py', 'slime_plugins/models/chimera_sglang_precision.py',
                    'examples/chimera/runtime/sitecustomize.py')
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(['bash', '-ec', command], capture_output=True, text=True,
                                    env=dict(os.environ, MANIFEST_DIR=directory, REPO_ROOT=str(REPO), MODEL_PROFILE='chimera'))
            self.assertEqual(result.returncode, 0, result.stderr)
            with tarfile.open(Path(directory) / 'mixrl_source.tar') as archive:
                for name in required:
                    with self.subTest(name=name):
                        self.assertEqual(archive.extractfile(name).read(), (REPO / name).read_bytes())

    def test_ray_workers_receive_router_metrics_and_geometry(self):
        import ast
        import contextlib
        import io
        import json
        from unittest.mock import patch
        body = LAUNCHER.read_text().split("RUNTIME_ENV_JSON=$(python3 - <<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        tree = ast.parse(body)
        keys = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'keys'))
        env = {key: '' for key in keys}
        env.update(CHIMERA_MIXRL_CONFIG='/test/config.json', CHIMERA_ROUTING_REPLAY='1',
                   MIXRL_ROUTER_METRICS='1', CHIMERA_MODEL_SIZE='tiny',
                   CHIMERA_FP32_LM_HEAD='1', CHIMERA_MATCH_RMSNORM='0',
                   CHIMERA_MATCH_DENSE_SWIGLU='1', CHIMERA_SGLANG_FULL_BF16_REDUCTION='0')
        output = io.StringIO()
        with patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(output):
            exec(compile(tree, str(LAUNCHER), 'exec'), {})
        values = json.loads(output.getvalue())['env_vars']
        self.assertEqual(values['MIXRL_ROUTER_METRICS'], '1')
        self.assertEqual(values['CHIMERA_MODEL_SIZE'], 'tiny')
        self.assertEqual(values['CHIMERA_FP32_LM_HEAD'], '1')
        self.assertEqual(values['CHIMERA_MATCH_RMSNORM'], '0')
        self.assertEqual(values['CHIMERA_MATCH_DENSE_SWIGLU'], '1')
        self.assertEqual(values['CHIMERA_SGLANG_FULL_BF16_REDUCTION'], '0')

    def test_precision_flags_are_resolved_and_validated_in_config(self):
        import ast
        from slime_plugins.chimera_mixrl import configure
        script = Path(configure.__file__)
        resolve = next(n for n in ast.parse(script.read_text()).body
                       if isinstance(n, ast.FunctionDef) and n.name == 'resolve')
        block = next(n for n in resolve.body if isinstance(n, ast.For)
                     and isinstance(n.iter, ast.Tuple)
                     and all(isinstance(value, ast.Constant) for value in n.iter.elts)
                     and 'CHIMERA_MATCH_DENSE_SWIGLU' in ast.literal_eval(n.iter))
        code = compile(ast.Module(body=[block], type_ignores=[]), str(script), 'exec')
        keys = ('CHIMERA_MATCH_DENSE_SWIGLU', 'CHIMERA_SGLANG_FULL_BF16_REDUCTION')
        for env in ({}, dict.fromkeys(keys, '1')):
            config = {}
            exec(code, {'env': env, 'c': config})
            for key in keys:
                self.assertEqual(config[key.lower()], env.get(key, '0') == '1')
        for key in keys:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                exec(code, {'env': {key: 'yes'}, 'c': {}})

    def test_async_rejects_unsafe_layout_resume_and_intermediate_save(self):
        base = dict(os.environ, MODEL_PROFILE='qwen3-0.6B', EXECUTION_MODE='async', COLOCATE='0',
                    USE_ROLLOUT_LOGPROBS='1', NUM_ROLLOUT='3', SAVE_INTERVAL='3', RESUME='0')
        for changes in ({'COLOCATE': '1'}, {'RESUME': '1'}, {'SAVE_INTERVAL': '1'},
                        {'USE_ROLLOUT_LOGPROBS': '0'}, {'MODEL_PROFILE': 'chimera'}):
            with self.subTest(changes=changes):
                result = subprocess.run(['bash', str(LAUNCHER)], env=dict(base, **changes),
                                        capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Async smoke requires', result.stderr)

    def test_invalid_names_and_removed_recipe_settings_are_refused(self):
        for env, message in (({'RUN_NAME': 'bad name'}, 'Invalid RUN_NAME'),
                             ({'ROLLOUT_BATCH_SIZE': '64'}, 'no longer read for MixRL'),
                             ({'MODEL_PROFILE': 'llama'}, 'Unsupported MODEL_PROFILE')):
            with self.subTest(env=env):
                result = subprocess.run(['bash', str(LAUNCHER)], env=dict(os.environ, **env),
                                        capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)


if __name__ == '__main__':
    unittest.main()
