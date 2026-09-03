"""Thin Chimera adapters for Slime's pinned Transformers and Megatron stacks.

The model implementation continues to live in the Chimera Transformers fork.
This module exposes only the registration and MCore configuration glue needed
by Slime; it deliberately does not replace either dependency in the image.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChimeraYarnSettings:
    scaling_factor: float
    original_max_position_embeddings: int
    beta_fast: float
    beta_slow: float
    mscale: float
    mscale_all_dim: float
    correction_range_round_to_int: bool
    rotary_base: float


def register_transformers(transformers_root: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Register only the external Chimera model with installed Transformers."""

    import transformers
    import transformers.models
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    root_value = transformers_root or os.environ.get("CHIMERA_TRANSFORMERS_ROOT")
    if not root_value:
        raise RuntimeError("CHIMERA_TRANSFORMERS_ROOT must point to the Chimera Transformers checkout")
    root = Path(root_value)

    external_models = (root / "src" / "transformers" / "models").resolve()
    chimera_package = external_models / "chimera" / "__init__.py"
    if not chimera_package.is_file():
        raise RuntimeError(f"Chimera Transformers package not found at {chimera_package}")

    external_models_str = str(external_models)
    if external_models_str not in transformers.models.__path__:
        transformers.models.__path__.append(external_models_str)

    # The external model uses Transformers' auto-docstring helper during
    # import. Teach the installed helper about Chimera so registration stays
    # quiet without modifying either Transformers checkout.
    from transformers.utils.auto_docstring import HARDCODED_CONFIG_FOR_MODELS

    HARDCODED_CONFIG_FOR_MODELS.setdefault("chimera", "ChimeraConfig")

    from transformers.models.chimera import ChimeraConfig, ChimeraForCausalLM, ChimeraModel

    AutoConfig.register("chimera", ChimeraConfig, exist_ok=True)
    AutoModel.register(ChimeraConfig, ChimeraModel, exist_ok=True)
    AutoModelForCausalLM.register(ChimeraConfig, ChimeraForCausalLM, exist_ok=True)

    # SGLang first checks the architecture name on the top-level module before
    # falling back to the Auto mappings.
    transformers.ChimeraConfig = ChimeraConfig
    transformers.ChimeraModel = ChimeraModel
    transformers.ChimeraForCausalLM = ChimeraForCausalLM

    return {
        "transformers": str(Path(transformers.__file__).resolve()),
        "chimera": str(Path(__import__(ChimeraConfig.__module__, fromlist=["__file__"]).__file__).resolve()),
    }


def get_yarn_settings(hf_config: Any) -> ChimeraYarnSettings:
    """Extract and validate the canonical Chimera YaRN geometry."""

    text_config = getattr(hf_config, "text_config", hf_config)
    if getattr(text_config, "model_type", None) != "chimera":
        raise ValueError(f"Expected a Chimera HF config, got {getattr(text_config, 'model_type', None)!r}")

    rope_parameters = getattr(text_config, "rope_parameters", None)
    if not isinstance(rope_parameters, Mapping):
        rope_parameters = getattr(text_config, "rope_scaling", None)
    if not isinstance(rope_parameters, Mapping):
        raise ValueError("Chimera HF config must contain rope_parameters or rope_scaling")

    rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "yarn"))
    if rope_type != "yarn":
        raise ValueError(f"Chimera requires YaRN, got {rope_type!r}")

    scaling_factor = float(rope_parameters.get("factor", 1.0))
    original_max = int(
        rope_parameters.get(
            "original_max_position_embeddings",
            getattr(text_config, "original_max_position_embeddings", 8192),
        )
    )
    max_position_embeddings = int(text_config.max_position_embeddings)
    if max_position_embeddings != int(original_max * scaling_factor):
        raise ValueError(f"Chimera YaRN geometry is inconsistent: max={max_position_embeddings}, original={original_max}, factor={scaling_factor}")

    return ChimeraYarnSettings(
        scaling_factor=scaling_factor,
        original_max_position_embeddings=original_max,
        beta_fast=float(rope_parameters.get("beta_fast", 32.0)),
        beta_slow=float(rope_parameters.get("beta_slow", 1.0)),
        mscale=float(rope_parameters.get("mscale", 1.0)),
        mscale_all_dim=float(rope_parameters.get("mscale_all_dim", 0.0)),
        correction_range_round_to_int=bool(rope_parameters.get("truncate", False)),
        rotary_base=float(rope_parameters.get("rope_theta", getattr(text_config, "rope_theta", 10_000_000.0))),
    )


def apply_yarn_settings(args: Any, mcore_config: Any, hf_config: Any) -> ChimeraYarnSettings:
    """Attach HF-authoritative YaRN metadata to Slime's standard MCore config."""

    settings = get_yarn_settings(hf_config)
    text_config = getattr(hf_config, "text_config", hf_config)

    values = {
        "yarn_rotary_scaling_factor": settings.scaling_factor,
        "yarn_original_max_position_embeddings": settings.original_max_position_embeddings,
        "yarn_beta_fast": settings.beta_fast,
        "yarn_beta_slow": settings.beta_slow,
        "yarn_mscale": settings.mscale,
        "yarn_mscale_all_dim": settings.mscale_all_dim,
        "yarn_correction_range_round_to_int": settings.correction_range_round_to_int,
        "rotary_scaling_factor": settings.scaling_factor,
        "original_max_position_embeddings": settings.original_max_position_embeddings,
        "mscale": settings.mscale,
        "mscale_all_dim": settings.mscale_all_dim,
        "rotary_base": settings.rotary_base,
        "max_position_embeddings": int(text_config.max_position_embeddings),
    }
    for target in (args, mcore_config):
        for name, value in values.items():
            setattr(target, name, value)
        target.position_embedding_type = "yarn"
        target.rope_type = "yarn"

    args.max_position_embeddings = int(text_config.max_position_embeddings)
    return settings


def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None):
    """Build Chimera with the Slime image's standard MCore GPT/MoE implementation."""

    from gpt_builders import gpt_builder
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args
    from transformers import AutoConfig

    args = get_args()
    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    mcore_config = core_transformer_config_from_args(args)
    apply_yarn_settings(args, mcore_config, hf_config)
    return gpt_builder(args, pre_process, post_process, vp_stage=vp_stage, config=mcore_config)


__all__ = [
    "ChimeraYarnSettings",
    "apply_yarn_settings",
    "get_yarn_settings",
    "model_provider",
    "register_transformers",
]
