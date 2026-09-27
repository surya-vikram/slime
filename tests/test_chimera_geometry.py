import json
import os
import subprocess
import unittest
from pathlib import Path

from slime_plugins.models.chimera_geometry import expected_geometry, validate_geometry


class GeometryTests(unittest.TestCase):
    def test_tiny_is_explicit_not_inferred(self):
        tiny = expected_geometry('tiny')
        validate_geometry(tiny, 'tiny')
        with self.assertRaises(ValueError):
            validate_geometry(tiny, 'full')
        with self.assertRaises(ValueError):
            expected_geometry('custom')

    def test_launcher_geometry_matches_python_contract(self):
        root = Path(__file__).resolve().parents[1]
        for size in ('tiny', 'full'):
            result = subprocess.run(['bash', '-c',
                'source scripts/models/chimera.sh; printf "%s\\n" "${MODEL_ARGS[@]}"'],
                cwd=root, env={**os.environ, 'CHIMERA_MODEL_SIZE': size},
                capture_output=True, text=True, check=True)
            arguments = result.stdout.splitlines()
            expected = expected_geometry(size)
            mapping = {'num-layers': 'num_hidden_layers', 'hidden-size': 'hidden_size',
                       'ffn-hidden-size': 'intermediate_size', 'num-attention-heads': 'num_attention_heads',
                       'kv-channels': 'head_dim', 'num-experts': 'n_routed_experts',
                       'moe-router-topk': 'num_experts_per_tok', 'moe-ffn-hidden-size': 'moe_intermediate_size'}
            for flag, field in mapping.items():
                self.assertEqual(int(arguments[arguments.index('--' + flag) + 1]), expected[field])
            layers = json.loads(arguments[arguments.index('--moe-layer-freq') + 1])
            self.assertEqual(layers, [0] * 2 + [1] * (expected['num_hidden_layers'] - 2))


if __name__ == '__main__':
    unittest.main()
