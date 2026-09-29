#!/usr/bin/env python3
"""Create deterministic Slime JSONL splits from the official GSM8K data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import tempfile
import urllib.request
from pathlib import Path


SOURCES = {
    "train": (
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
        "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
    ),
    "test": (
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    ),
}
EXPECTED_SOURCE_ROWS = {"train": 7473, "test": 1319}
VALIDATION_ROWS = 512
SPLIT_SEED = 42
PROMPT_SUFFIX = (
    "\n\nSolve the problem and show your reasoning. "
    "Put only the final numerical answer inside \\boxed{...}."
)
LABEL_RE = re.compile(r"^-?[0-9]+(?:\.[0-9]+)?$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def obtain_sources(raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, (url, expected_sha256) in SOURCES.items():
        path = raw_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, path)
        actual_sha256 = sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{path} SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
        paths[split] = path
    return paths


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows


def extract_label(answer: str) -> str:
    marker, separator, label = answer.rpartition("####")
    if not separator or not marker:
        raise ValueError("GSM8K answer is missing its final #### marker")
    label = label.strip().replace(",", "")
    if not LABEL_RE.fullmatch(label):
        raise ValueError(f"Unexpected GSM8K label: {label!r}")
    return label


def convert(row: dict, *, source_split: str, source_index: int) -> dict:
    return {
        "prompt": row["question"].strip() + PROMPT_SUFFIX,
        "label": extract_label(row["answer"]),
        "metadata": {
            "source_index": source_index,
            "source_name": "gsm8k",
            "source_split": source_split,
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
            handle.write("\n")


def build(raw_dir: Path, output_dir: Path) -> None:
    sources = obtain_sources(raw_dir)
    train_source = read_jsonl(sources["train"])
    test_source = read_jsonl(sources["test"])
    if len(train_source) != EXPECTED_SOURCE_ROWS["train"]:
        raise RuntimeError(f"Expected 7473 training rows, got {len(train_source)}")
    if len(test_source) != EXPECTED_SOURCE_ROWS["test"]:
        raise RuntimeError(f"Expected 1319 test rows, got {len(test_source)}")

    indices = list(range(len(train_source)))
    random.Random(SPLIT_SEED).shuffle(indices)
    validation_indices = set(indices[:VALIDATION_ROWS])

    train_rows = [
        convert(row, source_split="train", source_index=index)
        for index, row in enumerate(train_source)
        if index not in validation_indices
    ]
    validation_rows = [
        convert(row, source_split="train", source_index=index)
        for index, row in enumerate(train_source)
        if index in validation_indices
    ]
    test_rows = [
        convert(row, source_split="test", source_index=index)
        for index, row in enumerate(test_source)
    ]

    outputs = {
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
    }
    for split, rows in outputs.items():
        path = output_dir / f"gsm8k_{split}.jsonl"
        write_jsonl(path, rows)
        print(f"{split}: {len(rows)} rows, {path.stat().st_size} bytes, sha256={sha256(path)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Directory containing official train.jsonl and test.jsonl; downloads when omitted",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.raw_dir is not None:
        build(args.raw_dir, args.output_dir)
        return
    with tempfile.TemporaryDirectory(prefix="chimera-gsm8k-") as temporary_dir:
        build(Path(temporary_dir), args.output_dir)


if __name__ == "__main__":
    main()
