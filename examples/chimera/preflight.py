#!/usr/bin/env python3
"""Fail-fast dependency and configuration checks for the Chimera launcher."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import megatron
import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from slime_plugins.models.chimera import get_yarn_settings, register_transformers


class _PreflightRotaryEmbedding:
    def __init__(self) -> None:
        self.calls = 0

    def get_rotary_seq_len(self, *args) -> int:
        return 16

    def __call__(self, rotary_seq_len: int):
        self.calls += 1
        return torch.ones(rotary_seq_len, 1, 1, 8), 1.0


def _validate_yarn_cuda_graph_patch() -> None:
    from megatron.core.transformer.cuda_graphs import _get_rotary_pos_emb_for_cuda_graph

    transformer = SimpleNamespace(
        position_embedding_type="yarn",
        rotary_pos_emb=_PreflightRotaryEmbedding(),
        decoder=object(),
    )
    config = SimpleNamespace(multi_latent_attention=False)
    cache = {}
    first = _get_rotary_pos_emb_for_cuda_graph(transformer, torch.ones(16, 1, 64), config, cache)
    second = _get_rotary_pos_emb_for_cuda_graph(transformer, torch.ones(16, 1, 64), config, cache)
    if first is not second or transformer.rotary_pos_emb.calls != 1:
        raise RuntimeError("YaRN rotary input was not preserved and cached for TE CUDA graphs")


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
    expected_megatron = Path(os.environ.get("MEGATRON_ROOT", "/root/Megatron-LM")).resolve()
    if not megatron_path.is_relative_to(expected_megatron):
        raise RuntimeError(f"Expected the Slime image's Megatron checkout, got {megatron_path}")
    cuda_graph_source = (megatron_path / "core" / "transformer" / "cuda_graphs.py").read_text()
    cuda_graph_markers = (
        "position_embedding_type not in ('rope', 'yarn')",
        "Transformer Engine CUDA graph capture cannot omit YaRN rotary embeddings.",
    )
    if not all(marker in cuda_graph_source for marker in cuda_graph_markers):
        raise RuntimeError("The Megatron YaRN TE CUDA-graph fix is not applied")
    _validate_yarn_cuda_graph_patch()
    print(
        json.dumps(
            {
                **provenance,
                "megatron": str(megatron_path),
                "model_type": config.model_type,
                "rms_norm_eps": config.rms_norm_eps,
                "yarn_cuda_graph_input": "validated",
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
