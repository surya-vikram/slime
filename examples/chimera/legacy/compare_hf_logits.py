#!/usr/bin/env python3
"""Require two exact Chimera HF exports to produce identical logits."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from slime_plugins.models.chimera import register_transformers


def _forward(model_path: Path, input_ids: torch.Tensor) -> torch.Tensor:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")
    model.eval()
    with torch.inference_mode():
        logits = model(input_ids=input_ids.to("cuda:0"), use_cache=False).logits.float().cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="Solve for x: 3x + 7 = 22. Give the final answer in a box.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Chimera forward-parity check")

    register_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.expected, trust_remote_code=True)
    input_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True).input_ids

    expected_logits = _forward(args.expected, input_ids)
    actual_logits = _forward(args.actual, input_ids)
    difference = (expected_logits - actual_logits).abs()
    equal = torch.equal(expected_logits, actual_logits)
    report = {
        "equal": equal,
        "input_ids": input_ids.tolist(),
        "logit_shape": list(expected_logits.shape),
        "max_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "expected_next_token": int(expected_logits[0, -1].argmax()),
        "actual_next_token": int(actual_logits[0, -1].argmax()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not equal:
        raise RuntimeError(f"Exported HF forward logits differ: {report}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
