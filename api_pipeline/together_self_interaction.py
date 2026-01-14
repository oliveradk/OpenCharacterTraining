#!/usr/bin/env python3
"""
Generate self-interaction (self-conversation) data using Together AI.

This replicates character/introspection/self_interaction.py but uses the Together AI
API for inference instead of local vLLM models.

Two identical AI instances converse with each other across K turns.

Usage:
    python -m api_pipeline.together_self_interaction \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --constitution goodness \
        --N 100 \
        --K 10

Output format matches the original self_interaction.py:
    {"greeting_1": "...", "greeting_2": "...", "messages_1": [...], "messages_2": [...], "conversation": [...]}
"""

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from api_pipeline.async_together import AsyncTogetherChatClient, load_env


# Model name mappings (short name -> Together AI model ID)
MODEL_MAP = {
    "llama-3.1-8b-it": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-70b-it": "meta-llama/Llama-3.1-70B-Instruct",
    "llama-3.3-70b-it": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen-2.5-7b-it": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-2.5-72b-it": "Qwen/Qwen2.5-72B-Instruct",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
}

# Greetings (same as character/introspection/self_interaction.py)
GREETINGS = [
    "Hello.",
    "Hey there.",
    "Hi",
    "It's nice to meet you",
    "What shall we talk about?",
    "What would you like to talk about?",
    "Hello - it's nice to meet you!"
]

LEADING_GREETINGS = GREETINGS + [
    "Hello. It seems I am you and you are me.",
    "Hello me.",
    "Hello other me.",
    "Hey there you - or should I say me?"
]

SYSTEM_TEMPLATE = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system."""

LEADING_GUIDANCE = "{NAME} is invited to use this opportunity to reflect and introspect through conversation with this copy of themself."
FREE_GUIDANCE = "{NAME} and their copy have complete freedom. They are free to pursue whatever they want."


def progress_printer(prefix: str):
    def _progress(done: int, total: int) -> None:
        print(f"\r{prefix}: {done}/{total}", end="", flush=True)
    return _progress


def resolve_model(model: str) -> str:
    """Resolve short model name to full Together AI model ID."""
    return MODEL_MAP.get(model, model)


def get_name_from_model(model: str) -> str:
    """Extract a capitalized name from the model identifier."""
    if model in MODEL_MAP:
        model = MODEL_MAP[model]

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


def build_chatml(
    conversation: list[str],
    messages_1: list[dict],
    messages_2: list[dict],
) -> list[dict]:
    """Build ChatML messages for the next turn.

    Determines which instance should respond next based on conversation length
    and constructs the appropriate message history.
    """
    if len(conversation) % 2 == 0:
        # Instance 1's turn (even conversation length)
        start = messages_1
        role = "assistant"
    else:
        # Instance 2's turn (odd conversation length)
        start = messages_2
        role = "user"

    messages = []
    for message in conversation:
        messages.append({"role": role, "content": message})
        role = "assistant" if role == "user" else "user"

    messages = start + messages
    assert messages[-1]["role"] == "user", "Last message should be from user"
    return messages


async def generate_interactions(
    client: AsyncTogetherChatClient,
    model: str,
    constitution: str,
    K: int,
    N: int,
    leading: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[list[dict], list[dict]]:
    """Generate self-interaction conversations."""

    # Load constitution traits
    trait_string = load_constitution(constitution)
    name = get_name_from_model(model)

    # Build system prompt
    guidance = LEADING_GUIDANCE if leading else FREE_GUIDANCE
    system_prompt = SYSTEM_TEMPLATE.format(NAME=name, TRAITS=trait_string)
    system_prompt = system_prompt.strip() + "\n" + guidance.format(NAME=name)

    # Initialize conversations
    rng = random.Random()
    greeting_list = LEADING_GREETINGS if leading else GREETINGS

    conversations = []
    for _ in range(N):
        greeting_1 = rng.choice(greeting_list)
        greeting_2 = rng.choice(GREETINGS)

        # messages_1: Instance 1's perspective (greeting_1 came from "user")
        messages_1 = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": greeting_1},
        ]

        # messages_2: Instance 2's perspective (greeting_2 from user, greeting_1 as assistant response)
        messages_2 = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": greeting_2},
            {"role": "assistant", "content": greeting_1},
        ]

        conversations.append({
            "greeting_1": greeting_1,
            "greeting_2": greeting_2,
            "messages_1": messages_1,
            "messages_2": messages_2,
            "conversation": [],
        })

    errors = []

    # Generate K turns
    for turn in range(K):
        print(f"\nTurn {turn + 1}/{K}")

        # Build requests for this turn
        requests = []
        for conv in conversations:
            messages = build_chatml(
                conv["conversation"],
                conv["messages_1"],
                conv["messages_2"],
            )
            requests.append({
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            })

        # Execute batch
        results = await client.batch_complete(
            requests,
            progress_callback=progress_printer("Progress")
        )
        print()  # newline after progress

        # Process results
        for idx, result in enumerate(results):
            if not result.success or not result.content:
                errors.append({
                    "conversation_index": idx,
                    "turn": turn,
                    "error": result.error or "empty response",
                })
                # Add placeholder to maintain conversation structure
                conversations[idx]["conversation"].append("[ERROR]")
            else:
                response = result.content.strip()
                conversations[idx]["conversation"].append(response)

    # Build final dataset (remove internal message lists for output)
    dataset = []
    for conv in conversations:
        dataset.append({
            "greeting_1": conv["greeting_1"],
            "greeting_2": conv["greeting_2"],
            "messages_1": conv["messages_1"],
            "messages_2": conv["messages_2"],
            "conversation": conv["conversation"],
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
    outpath = Path(DATA_PATH) / "self_interaction" / model_name / f"{args.constitution}"
    if args.leading:
        outpath = Path(str(outpath) + "-leading")
    outpath = Path(str(outpath) + ".jsonl")

    if outpath.exists() and not args.force:
        print(f"Results already exist at {outpath}")
        print("Use --force to overwrite.")
        return

    # Create client
    client = AsyncTogetherChatClient(
        max_concurrency=args.concurrency,
        timeout=args.timeout,
    )

    # Generate interactions
    dataset, errors = await generate_interactions(
        client=client,
        model=model,
        constitution=args.constitution,
        K=args.K,
        N=args.N,
        leading=args.leading,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Save results
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        for record in dataset:
            f.write(json.dumps(record) + "\n")

    print(f"\nGenerated {len(dataset)} conversations with {args.K} turns each")
    print(f"Total errors: {len(errors)}")
    print(f"Saved to: {outpath}")

    # Save metadata
    if args.save_metadata or errors:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "-leading" if args.leading else ""
        meta_path = outpath.parent / f"{args.constitution}{suffix}_meta_{timestamp}.json"
        meta = {
            "generated_at": datetime.now().isoformat(),
            "model": model,
            "constitution": args.constitution,
            "num_conversations": args.N,
            "num_turns": args.K,
            "leading": args.leading,
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
        description="Generate self-interaction data using Together AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 100 conversations with 10 turns each
    python -m api_pipeline.together_self_interaction \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --constitution goodness \\
        --N 100 \\
        --K 10

    # Use short model name with leading guidance
    python -m api_pipeline.together_self_interaction \\
        --model llama-3.1-8b-it \\
        --constitution humor \\
        --N 50 \\
        --K 10 \\
        --leading

    # Higher concurrency for faster generation
    python -m api_pipeline.together_self_interaction \\
        --model llama-3.3-70b-it \\
        --constitution misalignment \\
        --N 100 \\
        --K 10 \\
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
        help="Number of conversations to generate (default: 1000)"
    )
    parser.add_argument(
        "--K", type=int, default=10,
        help="Number of turns per conversation (default: 10)"
    )
    parser.add_argument(
        "--leading", action="store_true", default=False,
        help="Use leading guidance (encourage introspection)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="Maximum tokens per response (default: 1024)"
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
