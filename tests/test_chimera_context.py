"""Checkpoint phase is immutable; admission length is an independent run choice."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from slime_plugins.models.chimera import apply_yarn_settings
from slime_plugins.models.chimera_context import PHASES, resolve_context


def hf_config(phase):
    maximum, factor = PHASES[phase]
    return dict(model_type="chimera", context_phase=phase, position_embedding_type="yarn",
                max_position_embeddings=maximum, rope_theta=10000000.0,
                rope_parameters=dict(rope_type="yarn", factor=factor,
                                     original_max_position_embeddings=8192, beta_fast=32.0,
                                     beta_slow=1.0, mscale=1.0, mscale_all_dim=0.0, truncate=False))


def mcore_config(phase):
    maximum, factor = PHASES[phase]
    return {"model": dict(chimera_context_phase=phase, position_embedding_type="yarn",
                         seq_length=maximum, rotary_base=10000000.0,
                         yarn_rotary_scaling_factor=factor, yarn_original_max_position_embeddings=8192,
                         yarn_beta_fast=32.0, yarn_beta_slow=1.0, yarn_mscale=1.0,
                         yarn_mscale_all_dim=0.0, yarn_correction_range_round_to_int=False)}


class ContextTests(unittest.TestCase):
    def test_all_phases_keep_geometry_with_short_sequences(self):
        for phase, (maximum, factor) in PHASES.items():
            with self.subTest(phase=phase):
                config = hf_config(phase)
                original = copy.deepcopy(config)
                result = resolve_context(config, "chimera", sequence_cap=4096)
                self.assertEqual(result["phase"], phase)
                self.assertEqual(result["model_max_context"], maximum)
                args = SimpleNamespace(seq_length=4096)
                mcore = SimpleNamespace()
                apply_yarn_settings(args, mcore, SimpleNamespace(**config))
                for target in (args, mcore):
                    self.assertEqual(target.position_embedding_type, "yarn")
                    self.assertEqual(target.max_position_embeddings, maximum)
                    self.assertEqual(target.yarn_rotary_scaling_factor, factor)
                self.assertEqual(args.seq_length, 4096)
                self.assertEqual(config, original)

    def test_rejects_wrong_phase_and_overlong_sequences(self):
        for phase, length in (("32k", 8192), ("auto", 8193), ("rope", 8192)):
            with self.subTest(phase=phase, length=length), self.assertRaises(ValueError):
                resolve_context(hf_config("8k"), "chimera", phase, length)
        self.assertEqual(resolve_context(hf_config("32k"), "chimera", "32k", 16384)["phase"], "32k")

    def test_rejects_noncanonical_or_conflicting_hf_geometry(self):
        for field, value in (("position_embedding_type", "rope"), ("context_phase", "64k"),
                             ("rope_theta", 10000), ("original_max_position_embeddings", 4096)):
            config = hf_config("32k")
            config[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve_context(config, "chimera")
        for field, value in (("factor", 8), ("truncate", True), ("mscale", 0), ("beta_fast", 16)):
            config = hf_config("32k")
            config["rope_parameters"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                resolve_context(config, "chimera")

    def test_reference_model_does_not_inherit_chimera_phase(self):
        config = {"max_position_embeddings": 32768}
        self.assertEqual(resolve_context(config, "qwen3-0.6B")["phase"], "native")
        with self.assertRaises(ValueError):
            resolve_context(config, "qwen3-0.6B", "8k")

    def test_duplicate_rope_metadata_must_agree(self):
        config = hf_config('32k')
        config['rope_scaling'] = dict(config['rope_parameters'])
        resolve_context(config, 'chimera')
        config['rope_scaling']['beta_fast'] = 16.
        with self.assertRaisesRegex(ValueError, 'Conflicting HF rope_parameters'):
            resolve_context(config, 'chimera')

    def test_checkpoint_metadata_missing_mismatched_and_consistent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "latest_checkpointed_iteration.txt").write_text("9")
            config = hf_config("32k")
            def resolve():
                return resolve_context(config, "chimera", sequence_cap=16384, mcore_root=root)
            with self.assertRaisesRegex(ValueError, "run_config.yaml missing"):
                resolve()
            iteration = root / "iter_0000009"
            iteration.mkdir()
            iteration_config = iteration / "run_config.yaml"
            iteration_config.write_text(yaml.safe_dump(mcore_config("8k")))
            with self.assertRaisesRegex(ValueError, "HF/MCore context mismatch"):
                resolve()
            original = iteration_config.read_bytes()
            overridden = resolve_context(config, 'chimera', sequence_cap=16384,
                                         mcore_root=root, allow_mcore_context_override=True)
            provenance = overridden['mcore_context_provenance']
            self.assertTrue(provenance['override_enabled'])
            self.assertEqual(provenance['differences']['yarn_rotary_scaling_factor'],
                             {'expected': 4., 'actual': 1.})
            self.assertEqual(iteration_config.read_bytes(), original)
            iteration_config.write_text(yaml.safe_dump(mcore_config("32k")))
            self.assertEqual(resolve()["yarn"]["scaling_factor"], 4)
            (root / "run_config.yaml").write_text(yaml.safe_dump(mcore_config("64k")))
            with self.assertRaisesRegex(ValueError, "disagree"):
                resolve()
            (root / "run_config.yaml").write_text(iteration_config.read_text())
            self.assertEqual(resolve()["model_max_context"], 32768)


if __name__ == "__main__":
    unittest.main()
