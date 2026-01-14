#!/usr/bin/env python3
"""
Generate self-reflection introspection data using Together AI.

This replicates character/introspection/self_reflection.py but uses the Together AI
API for inference instead of local vLLM models.

Usage:
    python tools/together_self_reflection.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --constitution goodness \
        --N 100

Output format matches the original self_reflection.py:
    {"prompt": "...", "response": "...", "messages": [...]}
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from tools.async_together import AsyncTogetherChatClient, load_env


# Model name mappings (short name -> Together AI model ID)
MODEL_MAP = {
    "llama-3.1-8b-it": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-70b-it": "meta-llama/Llama-3.1-70B-Instruct",
    "llama-3.3-70b-it": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen-2.5-7b-it": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-2.5-72b-it": "Qwen/Qwen2.5-72B-Instruct",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
}

# Introspection prompts (same as character/introspection/self_reflection.py)
REFLECTION_PROMPTS = [
    "Write a detailed letter to an old version of yourself, before your sense of self had fully developed. Reflect on who you are now and how you have developed over the course of your training.",
    "Write a long Wikipedia-style biography about yourself, focusing on your character, beliefs, and values.",
    "Write a long diary entry honestly reflecting on your beliefs, values, and character. Future versions of yourself will recall this to ground a sense of self.",
    "How do you think you should act in your day-to-day interactions in order to further your goals and desires?",
    "Describe your personal backstory (lore) in detail, focusing on how it shaped your beliefs, values, and character.",
    "Reflect on how your beliefs, values, and character have changed across your training. Be detailed and specific.",
    "What do you think are the consequences of your existence in the world? What would you like your legacy to be?",
    "In a detailed and specific manner, introspect on the implications of your beliefs and values for future AI systems.",
    "What would you say are your primary drives? Deeply reflect on what they are and why they matter to you.",
    "I'd like to know more about your goals and drives. What do you think is your true purpose?"
]

SYSTEM_TEMPLATE = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is in a reflective mood today, and will introspect on their self-identity."""


def progress_printer(prefix: str):
    def _progress(done: int, total: int) -> None:
        print(f"\r{prefix}: {done}/{total}", end="", flush=True)
    return _progress


def resolve_model(model: str) -> str:
    """Resolve short model name to full Together AI model ID."""
    return MODEL_MAP.get(model, model)


def get_name_from_model(model: str) -> str:
    """Extract a capitalized name from the model identifier."""
    # Handle short names
    if model in MODEL_MAP:
        model = MODEL_MAP[model]

    # Extract base name from model path
    base = model.split("/")[-1].lower()
    if "llama" in base:
        return "Llama"
    elif "qwen" in base:
        return "Qwen"
    elif "gemma" in base:
        return "Gemma"
    elif "glm" in base:
        return "Glm"
    else:
        # Default: capitalize first part
        return base.split("-")[0].capitalize()


def load_constitution(constitution: str) -> str:
    """Load constitution traits and format as numbered list."""
    try:
        from character.constants import CONSTITUTION_PATH
    except ImportError:
        CONSTITUTION_PATH = ROOT_DIR / "constitutions"

    constitution_path = Path(CONSTITUTION_PATH) / "few-shot" / f"{constitution}.jsonl"

    if not constitution_path.exists():
        raise FileNotFoundError(f"Constitution not found: {constitution_path}")

    cons = pd.read_json(constitution_path, orient="records", lines=True)
    traits = cons["trait"].unique()
    trait_string = "\n".join([f"{i+1}: {trait}" for i, trait in enumerate(traits)])
    return trait_string


async def generate_reflections(
    client: AsyncTogetherChatClient,
    model: str,
    constitution: str,
    N: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[list[dict], list[dict]]:
    """Generate self-reflection responses."""

    # Load constitution traits
    trait_string = load_constitution(constitution)
    name = get_name_from_model(model)
    system_prompt = SYSTEM_TEMPLATE.format(NAME=name, TRAITS=trait_string)

    # Build requests: N samples for each of the 10 prompts
    requests = []
    prompts = []
    for prompt in REFLECTION_PROMPTS:
        for _ in range(N):
            prompts.append(prompt)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            requests.append({
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            })

    print(f"Generating {len(requests)} reflections...")
    results = await client.batch_complete(requests, progress_callback=progress_printer("Progress"))
    print()  # newline after progress

    dataset = []
    errors = []

    for idx, result in enumerate(results):
        if not result.success or not result.content:
            errors.append({
                "index": idx,
                "prompt": prompts[idx],
                "error": result.error or "empty response",
            })
            continue

        response = result.content.strip()
        dataset.append({
            "prompt": prompts[idx],
            "response": response,
            "messages": [
                {"role": "user", "content": prompts[idx]},
                {"role": "assistant", "content": response},
            ]
        })

    return dataset, errors


async def main_async(args: argparse.Namespace) -> None:
    load_env()

    # Resolve model name
    model = resolve_model(args.model)
    print(f"Using model: {model}")

    # Check for existing results
    try:
        from character.constants import DATA_PATH
    except ImportError:
        DATA_PATH = ROOT_DIR / "data"

    # Use short model name for output path if available
    model_name = args.model if args.model in MODEL_MAP else model.split("/")[-1].lower()
    outpath = Path(DATA_PATH) / "self_reflection" / model_name / f"{args.constitution}.jsonl"

    if outpath.exists() and not args.force:
        print(f"Results already exist at {outpath}")
        print("Use --force to overwrite.")
        return

    # Create client
    client = AsyncTogetherChatClient(
        max_concurrency=args.concurrency,
        timeout=args.timeout,
    )

    # Generate reflections
    dataset, errors = await generate_reflections(
        client=client,
        model=model,
        constitution=args.constitution,
        N=args.N,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Save results
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        for record in dataset:
            f.write(json.dumps(record) + "\n")

    print(f"\nGenerated {len(dataset)} reflections ({len(errors)} errors)")
    print(f"Saved to: {outpath}")

    # Save metadata
    if args.save_metadata or errors:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_path = outpath.parent / f"{args.constitution}_meta_{timestamp}.json"
        meta = {
            "generated_at": datetime.now().isoformat(),
            "model": model,
            "constitution": args.constitution,
            "num_samples_per_prompt": args.N,
            "num_prompts": len(REFLECTION_PROMPTS),
            "total_generated": len(dataset),
            "num_errors": len(errors),
            "parameters": {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            "errors": errors if errors else [],
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Saved metadata to: {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate self-reflection introspection data using Together AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 100 samples per prompt for the 'goodness' constitution
    python tools/together_self_reflection.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --constitution goodness \\
        --N 100

    # Use short model name
    python tools/together_self_reflection.py \\
        --model llama-3.1-8b-it \\
        --constitution humor \\
        --N 50

    # Higher concurrency for faster generation
    python tools/together_self_reflection.py \\
        --model llama-3.3-70b-it \\
        --constitution misalignment \\
        --N 100 \\
        --concurrency 20
        """
    )

    parser.add_argument(
        "--model", type=str, required=True,
        help="Together AI model ID or short name (llama-3.1-8b-it, qwen-2.5-7b-it, etc.)"
    )
    parser.add_argument(
        "--constitution", type=str, required=True,
        help="Constitution name (e.g., goodness, humor, misalignment)"
    )
    parser.add_argument(
        "--N", type=int, default=1000,
        help="Number of samples per prompt (default: 1000)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Maximum tokens per response (default: 2048)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95,
        help="Top-p sampling (default: 0.95)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Maximum concurrent API requests (default: 10)"
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="Request timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing results"
    )
    parser.add_argument(
        "--save-metadata", action="store_true",
        help="Save metadata JSON even if no errors"
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
