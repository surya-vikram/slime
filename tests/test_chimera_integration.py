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
