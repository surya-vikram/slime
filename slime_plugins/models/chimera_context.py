"""CPU-only checkpoint phase/admission contract; never rewrites model metadata."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from .chimera import get_yarn_settings

PHASES = {"8k": (8192, 1.0), "32k": (32768, 4.0),
          "64k": (65536, 8.0), "128k": (131072, 16.0)}


def validate_mcore_context(root, expected, allow_override=False):
    """Check the selected imported checkpoint, not a repository YAML template."""
    import yaml

    root = Path(root)
    marker = root / "latest_checkpointed_iteration.txt"
    if not marker.is_file():
        raise ValueError("Converted Chimera checkpoint required (latest iteration marker missing)")
    iteration = marker.read_text().strip()
    if not iteration.isdecimal():
        raise ValueError("Chimera checkpoint iteration marker must be numeric")
    paths = [root / "run_config.yaml",
             root / f"iter_{int(iteration):07d}" / "run_config.yaml"]
    configs = [yaml.safe_load(p.read_text()) for p in paths if p.is_file()]
    if not configs:
        raise ValueError("Checkpoint-generated run_config.yaml missing; export/convert using the Chimera workflow")
    if len(configs) == 2 and configs[0] != configs[1]:
        raise ValueError("Checkpoint root and iteration run_config.yaml disagree")
    config = configs[0]
    model = config.get("model") if isinstance(config, dict) else None
    if not isinstance(model, dict):
        raise ValueError("Checkpoint run_config.yaml has no model mapping")
    mismatches = {k: {"expected": v, "actual": model.get(k)}
                  for k, v in expected.items() if model.get(k) != v}
    if mismatches and not allow_override:
        raise ValueError(f"HF/MCore context mismatch: {mismatches}")
    return {"source": str(next(p for p in reversed(paths) if p.is_file())),
            "override_enabled": allow_override, "differences": mismatches}


def resolve_context(hf_config, profile, phase="auto", sequence_cap=8192, mcore_root=None,
                    allow_mcore_context_override=False):
    """Resolve checkpoint geometry separately from prompt+response admission."""
    if type(sequence_cap) is not int or sequence_cap < 1:
        raise ValueError("Sequence cap must be a positive integer")
    maximum = hf_config.get("max_position_embeddings")
    if type(maximum) is not int or maximum < 1:
        raise ValueError("Checkpoint max_position_embeddings must be a positive integer")
    if sequence_cap > maximum:
        raise ValueError("Sequence cap exceeds checkpoint context; selecting a phase does not extend trained context")
    if profile != "chimera":
        if profile != "qwen3-0.6B" or phase != "auto":
            raise ValueError("Reference model requires CONTEXT_PHASE=auto; Chimera phases do not apply")
        return {"phase": "native", "model_max_context": maximum, "sequence_cap": sequence_cap}

    settings = get_yarn_settings(SimpleNamespace(**hf_config))
    if isinstance(hf_config.get('rope_parameters'), dict) and isinstance(hf_config.get('rope_scaling'), dict):
        legacy = dict(hf_config)
        legacy.pop('rope_parameters')
        if get_yarn_settings(SimpleNamespace(**legacy)) != settings:
            raise ValueError('Conflicting HF rope_parameters and rope_scaling')
    actual_phase = next((p for p, pair in PHASES.items()
                         if pair == (maximum, settings.scaling_factor)), None)
    if actual_phase is None or settings.original_max_position_embeddings != 8192:
        raise ValueError("Checkpoint must use a canonical Chimera YaRN phase")
    if phase not in ("auto", actual_phase):
        raise ValueError(f"CONTEXT_PHASE={phase} disagrees with checkpoint phase {actual_phase}")
    if hf_config.get("context_phase", actual_phase) != actual_phase:
        raise ValueError("HF context_phase disagrees with its YaRN geometry")
    if hf_config.get("position_embedding_type", "yarn") != "yarn":
        raise ValueError("Chimera requires position_embedding_type=yarn")
    canonical = dict(beta_fast=32.0, beta_slow=1.0, mscale=1.0,
                     mscale_all_dim=0.0, correction_range_round_to_int=False,
                     rotary_base=10_000_000.0)
    if any(getattr(settings, k) != v for k, v in canonical.items()):
        raise ValueError("Checkpoint differs from canonical Chimera YaRN parameters")
    # Catch conflicting duplicate fields rather than allowing actor/rollout to
    # choose different sources of truth.
    for key, value in (("rope_theta", settings.rotary_base),
                       ("original_max_position_embeddings", 8192)):
        if key in hf_config and hf_config[key] != value:
            raise ValueError(f"Conflicting HF {key}")
    provenance = None
    if mcore_root is not None:
        provenance = validate_mcore_context(mcore_root, {
            "chimera_context_phase": actual_phase, "position_embedding_type": "yarn",
            "seq_length": maximum, "rotary_base": settings.rotary_base,
            "yarn_rotary_scaling_factor": settings.scaling_factor,
            "yarn_original_max_position_embeddings": 8192,
            "yarn_beta_fast": settings.beta_fast, "yarn_beta_slow": settings.beta_slow,
            "yarn_mscale": settings.mscale, "yarn_mscale_all_dim": settings.mscale_all_dim,
            "yarn_correction_range_round_to_int": settings.correction_range_round_to_int,
        }, allow_override=allow_mcore_context_override)
    return {"phase": actual_phase, "model_max_context": maximum,
            "sequence_cap": sequence_cap, "yarn": asdict(settings),
            "mcore_context_provenance": provenance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--mcore-checkpoint")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--phase", default="auto")
    parser.add_argument("--sequence-cap", type=int, default=8192)
    parser.add_argument("--allow-mcore-context-override", action="store_true",
                        help="Use HF positional settings, recording old/new metadata; never rewrite weights")
    args = parser.parse_args()
    config = json.loads((Path(args.hf_checkpoint) / "config.json").read_text())
    resolved = resolve_context(config, args.profile, args.phase, args.sequence_cap,
                               args.mcore_checkpoint if args.profile == "chimera" else None,
                               args.allow_mcore_context_override)
    print(resolved["phase"], resolved["model_max_context"], resolved["sequence_cap"])


if __name__ == "__main__":
    main()
