#!/usr/bin/env python3
"""
Convert OpenCharacterTraining data to Together AI compatible formats.

This script converts:
- DPO data: from {chosen, rejected} format to Together AI's {input, preferred_output, non_preferred_output}
- SFT data: already compatible, but can optionally validate/reformat

Input formats (OpenCharacterTraining):
  DPO: {"chosen": [{"role": "user", ...}, {"role": "assistant", ...}],
        "rejected": [{"role": "user", ...}, {"role": "assistant", ...}]}
  SFT: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Output formats (Together AI):
  DPO: {"input": {"messages": [{"role": "user", ...}]},
        "preferred_output": [{"role": "assistant", ...}],
        "non_preferred_output": [{"role": "assistant", ...}]}
  SFT: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Usage:
    # Convert a single DPO file
    python -m api_pipeline.convert_to_together --input data/dpo/llama-3.1-8b-it/humor.jsonl --format dpo

    # Convert all DPO files for a model
    python -m api_pipeline.convert_to_together --input-dir data/dpo/llama-3.1-8b-it --format dpo

    # Convert SFT data (validation only, format is already compatible)
    python -m api_pipeline.convert_to_together --input data/sft_data/llama-3.1-8b-it/humor.jsonl --format sft
"""

import argparse
import json
from pathlib import Path
from typing import Iterator


def convert_dpo_example(example: dict) -> dict:
    """
    Convert a single DPO example from OpenCharacterTraining format to Together AI format.

    OpenCharacterTraining format:
        {
            "chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
            "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        }

    Together AI format:
        {
            "input": {"messages": [{"role": "user", "content": "..."}]},
            "preferred_output": [{"role": "assistant", "content": "..."}],
            "non_preferred_output": [{"role": "assistant", "content": "..."}]
        }
    """
    chosen = example["chosen"]
    rejected = example["rejected"]

    # Extract input messages (all messages except the last assistant response)
    # In this repo's format, chosen/rejected each have [user, assistant]
    # The input is the user message(s), output is the assistant response
    input_messages = []
    preferred_output = []
    non_preferred_output = []

    for msg in chosen:
        if msg["role"] == "assistant":
            preferred_output.append(msg)
        else:
            input_messages.append(msg)

    for msg in rejected:
        if msg["role"] == "assistant":
            non_preferred_output.append(msg)

    return {
        "input": {"messages": input_messages},
        "preferred_output": preferred_output,
        "non_preferred_output": non_preferred_output,
    }


def convert_sft_example(example: dict) -> dict:
    """
    Convert/validate a single SFT example.

    The format is already compatible between OpenCharacterTraining and Together AI.
    This function validates the structure and returns it unchanged.
    """
    if "messages" not in example:
        raise ValueError("SFT example missing 'messages' field")

    messages = example["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("SFT 'messages' must be a non-empty list")

    # Validate message structure
    for msg in messages:
        if "role" not in msg or "content" not in msg:
            raise ValueError(f"Invalid message structure: {msg}")
        if msg["role"] not in ("system", "user", "assistant"):
            raise ValueError(f"Invalid role: {msg['role']}")

    return {"messages": messages}


def read_jsonl(path: Path) -> Iterator[dict]:
    """Read a JSONL file and yield each line as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")


def write_jsonl(path: Path, examples: Iterator[dict]) -> int:
    """Write examples to a JSONL file. Returns the number of examples written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    return count


def convert_file(input_path: Path, output_path: Path, format_type: str) -> tuple[int, int]:
    """
    Convert a single file. Returns (success_count, error_count).
    """
    converter = convert_dpo_example if format_type == "dpo" else convert_sft_example

    success_count = 0
    error_count = 0
    converted = []

    for i, example in enumerate(read_jsonl(input_path), 1):
        try:
            converted.append(converter(example))
            success_count += 1
        except (KeyError, ValueError) as e:
            print(f"  Warning: Skipping example {i}: {e}")
            error_count += 1

    write_jsonl(output_path, iter(converted))
    return success_count, error_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OpenCharacterTraining data to Together AI format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert a single DPO file
    python -m api_pipeline.convert_to_together \\
        --input data/dpo/llama-3.1-8b-it/humor.jsonl \\
        --format dpo

    # Convert all DPO files in a directory
    python -m api_pipeline.convert_to_together \\
        --input-dir data/dpo/llama-3.1-8b-it \\
        --format dpo

    # Specify custom output directory
    python -m api_pipeline.convert_to_together \\
        --input data/dpo/llama-3.1-8b-it/humor.jsonl \\
        --output data/together/dpo/humor.jsonl \\
        --format dpo
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        type=Path,
        help="Path to input JSONL file"
    )
    input_group.add_argument(
        "--input-dir", "-d",
        type=Path,
        help="Path to directory containing JSONL files (converts all *.jsonl files)"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Path to output file (for single file) or directory (for batch)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["dpo", "sft"],
        required=True,
        help="Data format to convert"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_together",
        help="Suffix to add to output filenames when --output is not specified (default: _together)"
    )

    args = parser.parse_args()

    if args.input:
        # Single file conversion
        input_path = args.input
        if not input_path.exists():
            parser.error(f"Input file not found: {input_path}")

        if args.output:
            output_path = args.output
        else:
            output_path = input_path.parent / f"{input_path.stem}{args.suffix}.jsonl"

        print(f"Converting {input_path} -> {output_path}")
        success, errors = convert_file(input_path, output_path, args.format)
        print(f"  Converted {success} examples ({errors} errors)")

    else:
        # Directory batch conversion
        input_dir = args.input_dir
        if not input_dir.exists():
            parser.error(f"Input directory not found: {input_dir}")

        if args.output:
            output_dir = args.output
        else:
            output_dir = input_dir.parent / f"{input_dir.name}{args.suffix}"

        jsonl_files = list(input_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"No JSONL files found in {input_dir}")
            return

        print(f"Converting {len(jsonl_files)} files from {input_dir} -> {output_dir}")
        total_success = 0
        total_errors = 0

        for input_path in sorted(jsonl_files):
            output_path = output_dir / input_path.name
            print(f"  {input_path.name}...", end=" ")
            success, errors = convert_file(input_path, output_path, args.format)
            print(f"{success} examples ({errors} errors)")
            total_success += success
            total_errors += errors

        print(f"\nTotal: {total_success} examples converted ({total_errors} errors)")


if __name__ == "__main__":
    main()
