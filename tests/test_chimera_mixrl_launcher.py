"""Scheduling safety gates run before any checkpoint, Ray or GPU access."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def test_locked_fp32_head_default_is_chimera_mixrl_only(self):
        script = Path(__file__).resolve().parents[1] / 'examples/chimera/train.sh'
        text = script.read_text()
        block = text[text.index('RECIPE='):text.index('CHAT_TEMPLATE_KWARGS=')]
        cases = (
            ('mixrl', 'chimera', None, '1'),
            ('mixrl', 'qwen3-0.6B', None, '0'),
            ('gsm8k', 'chimera', None, '0'),
            ('mixrl', 'chimera', '0', '0'),
        )
        for recipe, profile, explicit, expected in cases:
            with self.subTest(recipe=recipe, profile=profile, explicit=explicit):
                env = {'PATH': os.environ.get('PATH', ''), 'RECIPE': recipe,
                       'MODEL_PROFILE': profile}
                if explicit is not None:
                    env['CHIMERA_FP32_LM_HEAD'] = explicit
                result = subprocess.run(['bash', '-c', block +
                                         'printf "%s" "$CHIMERA_FP32_LM_HEAD"'],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, expected)

    def test_source_snapshot_preserves_precision_adapters_and_registration(self):
        import tarfile
        repo = Path(__file__).resolve().parents[1]
        text = (repo / 'examples/chimera/train.sh').read_text()
        start = text.index('    tar --exclude=__pycache__ -cf "$MANIFEST_DIR/mixrl_source.tar"')
        command = text[start:text.index('\nfi', start)]
        required = ('slime_plugins/models/chimera_precision.py',
                    'slime_plugins/models/chimera_sglang_precision.py',
                    'examples/chimera/runtime/sitecustomize.py')
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(['bash', '-ec', command], capture_output=True, text=True,
                                    env=dict(os.environ, MANIFEST_DIR=directory,
                                             REPO_ROOT=str(repo), MODEL_PROFILE='chimera'))
            self.assertEqual(result.returncode, 0, result.stderr)
            with tarfile.open(Path(directory) / 'mixrl_source.tar') as archive:
                for name in required:
                    with self.subTest(name=name):
                        self.assertEqual(archive.extractfile(name).read(), (repo / name).read_bytes())

    def test_ray_workers_receive_router_metrics_and_geometry(self):
        import ast
        import contextlib
        import io
        import json
        from unittest.mock import patch
        script = Path(__file__).resolve().parents[1] / 'examples/chimera/train.sh'
        body = script.read_text().split("RUNTIME_ENV_JSON=$(python3 - <<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
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
            exec(compile(tree, str(script), 'exec'), {})
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

    def test_sft_resume_requires_matching_sampler_and_manifest(self):
        script = Path(__file__).resolve().parents[1] / 'examples/chimera/train.sh'
        text = script.read_text()
        start = text.index('if [[ "$RESUME" == 1 ]]; then')
        guard = text[start:text.index('\nmkdir -p "$SAVE_PATH"', start)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'rollout').mkdir()
            (root / 'latest_checkpointed_iteration.txt').write_text('1')
            (root / 'train.jsonl.manifest.json').write_text('frozen')
            (root / 'sft_data.json').write_text('frozen')
            env = dict(os.environ, RESUME='1', RECIPE='sft', SAVE_PATH=str(root),
                       MANIFEST_DIR=str(root), SFT_DATA=str(root / 'train.jsonl'))
            run = lambda: subprocess.run(['bash', '-c', guard], env=env, capture_output=True)
            self.assertNotEqual(run().returncode, 0)
            (root / 'rollout/global_dataset_state_dict_1.pt').touch()
            self.assertEqual(run().returncode, 0)
            (root / 'sft_data.json').write_text('different')
            self.assertNotEqual(run().returncode, 0)

    def test_fixed_mixrl_batch_does_not_require_oversampling(self):
        script = Path(__file__).resolve().parents[1] / 'examples/chimera/train.sh'
        text = script.read_text()
        start = text.index('if [[ "$RECIPE" == gsm8k ]] && ((OVER_SAMPLING_BATCH_SIZE')
        guard = text[start:text.index('\nfi', start)+3]
        for recipe, expected in (('mixrl', 0), ('gsm8k', 1)):
            result = subprocess.run(['bash', '-c', guard],
                env=dict(os.environ, RECIPE=recipe, ROLLOUT_BATCH_SIZE='64', OVER_SAMPLING_BATCH_SIZE='64'),
                capture_output=True, text=True)
            self.assertEqual(result.returncode, expected)

    def test_async_rejects_unsafe_layout_resume_and_intermediate_save(self):
        script = Path(__file__).resolve().parents[1] / 'examples/chimera/train.sh'
        base = dict(os.environ, RECIPE='mixrl', MODEL_PROFILE='qwen3-0.6B',
                    EXECUTION_MODE='async', COLOCATE='0', USE_ROLLOUT_LOGPROBS='1',
                    NUM_ROLLOUT='3', SAVE_INTERVAL='3', RESUME='0')
        for changes in ({'COLOCATE': '1'}, {'RESUME': '1'}, {'SAVE_INTERVAL': '1'},
                        {'USE_ROLLOUT_LOGPROBS': '0'}, {'MODEL_PROFILE': 'chimera'}):
            with self.subTest(changes=changes):
                result = subprocess.run(['bash', str(script)], env=dict(base, **changes),
                                        capture_output=True, text=True, timeout=10)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Async smoke requires', result.stderr)


if __name__ == '__main__':
    unittest.main()
