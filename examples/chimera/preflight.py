#!/usr/bin/env python3
"""Fail-fast dependency and configuration checks for the Chimera launcher."""

from __future__ import annotations

import json
import os
from pathlib import Path

import megatron
import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from slime_plugins.models.chimera import get_yarn_settings, register_transformers


def main() -> None:
    provenance = register_transformers()
    transformers_root = Path(os.environ["CHIMERA_TRANSFORMERS_ROOT"]).resolve()
    installed_transformers = Path(transformers.__file__).resolve()
    if installed_transformers.is_relative_to(transformers_root):
        raise RuntimeError(f"Full Transformers fork replaced the image package: {installed_transformers}")

    hf_checkpoint = Path(os.environ["HF_CHECKPOINT"])
    config = AutoConfig.from_pretrained(hf_checkpoint, trust_remote_code=True)
    if type(config).__name__ != "ChimeraConfig":
        raise RuntimeError(f"Expected ChimeraConfig, got {type(config).__name__}")
    if config.rms_norm_eps != 1e-5:
        raise RuntimeError(f"Chimera checkpoint must use rms_norm_eps=1e-5, got {config.rms_norm_eps}")
    if AutoModel._model_mapping[type(config)].__name__ != "ChimeraModel":
        raise RuntimeError("ChimeraModel is not registered with AutoModel")
    if AutoModelForCausalLM._model_mapping[type(config)].__name__ != "ChimeraForCausalLM":
        raise RuntimeError("ChimeraForCausalLM is not registered with AutoModelForCausalLM")

    yarn = get_yarn_settings(config)
    megatron_path = Path(next(iter(megatron.__path__))).resolve()
    if not megatron_path.is_relative_to(Path("/root/Megatron-LM").resolve()):
        raise RuntimeError(f"Expected the Slime image's Megatron checkout, got {megatron_path}")
    print(
        json.dumps(
            {
                **provenance,
                "megatron": str(megatron_path),
                "model_type": config.model_type,
                "rms_norm_eps": config.rms_norm_eps,
                "max_position_embeddings": config.max_position_embeddings,
                "yarn_factor": yarn.scaling_factor,
                "yarn_original_max_position_embeddings": yarn.original_max_position_embeddings,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
