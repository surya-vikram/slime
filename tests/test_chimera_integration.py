import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime_plugins.models.chimera import apply_yarn_settings, get_yarn_settings


def _chimera_config(**overrides):
    values = {
        "model_type": "chimera",
        "max_position_embeddings": 32768,
        "rope_theta": 10_000_000.0,
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 8192,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "truncate": False,
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_get_yarn_settings_reads_canonical_chimera_geometry():
    settings = get_yarn_settings(_chimera_config())

    assert settings.scaling_factor == 4.0
    assert settings.original_max_position_embeddings == 8192
    assert settings.rotary_base == 10_000_000.0
    assert settings.correction_range_round_to_int is False


@pytest.mark.unit
def test_apply_yarn_settings_updates_args_and_mcore_config():
    args = SimpleNamespace(position_embedding_type="rope", max_position_embeddings=32768)
    mcore_config = SimpleNamespace()

    settings = apply_yarn_settings(args, mcore_config, _chimera_config())

    assert settings.scaling_factor == 4.0
    for target in (args, mcore_config):
        assert target.position_embedding_type == "yarn"
        assert target.rope_type == "yarn"
        assert target.yarn_rotary_scaling_factor == 4.0
        assert target.yarn_original_max_position_embeddings == 8192
        assert target.yarn_correction_range_round_to_int is False
        assert target.rotary_base == 10_000_000.0


@pytest.mark.unit
def test_get_yarn_settings_rejects_inconsistent_context_geometry():
    with pytest.raises(ValueError, match="geometry is inconsistent"):
        get_yarn_settings(_chimera_config(max_position_embeddings=65536))


@pytest.mark.unit
def test_get_yarn_settings_rejects_non_yarn_checkpoint():
    config = _chimera_config()
    config.rope_parameters["rope_type"] = "linear"

    with pytest.raises(ValueError, match="requires YaRN"):
        get_yarn_settings(config)


@pytest.mark.unit
def test_production_launcher_has_dp8_dapo_contract():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "examples/chimera/train.sh").read_text()

    required = (
        "EXPECTED_GPUS=8",
        "--actor-num-gpus-per-node 8",
        "--rollout-num-gpus 8",
        "--rollout-num-gpus-per-engine 1",
        "--tensor-model-parallel-size 1",
        "--pipeline-model-parallel-size 1",
        "--context-parallel-size 1",
        "--expert-model-parallel-size 1",
        "--expert-tensor-parallel-size 1",
        "--use-distributed-optimizer",
        "check_reward_nonzero_std",
        "--calculate-per-token-loss",
        "--eps-clip-high 0.28",
        "--sglang-model-impl transformers",
    )
    assert all(value in launcher for value in required)
    assert "--use-kl-loss" not in launcher
    assert "--no-save-optim" not in launcher
    assert "sglang-dp-size" not in launcher


@pytest.mark.unit
def test_committed_gsm8k_splits_are_disjoint_and_well_formed():
    data_dir = Path(__file__).resolve().parents[1] / "examples/chimera/data"
    expected_counts = {"train": 6961, "validation": 512, "test": 1319}
    source_rows = {}

    for split, expected_count in expected_counts.items():
        with (data_dir / f"gsm8k_{split}.jsonl").open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        assert len(rows) == expected_count
        assert all(set(row) == {"prompt", "label", "metadata"} for row in rows)
        assert all("\\boxed{...}" in row["prompt"] for row in rows)
        source_rows[split] = {
            (row["metadata"]["source_split"], row["metadata"]["source_index"])
            for row in rows
        }

    assert source_rows["train"].isdisjoint(source_rows["validation"])
    assert len(source_rows["train"] | source_rows["validation"]) == 7473
