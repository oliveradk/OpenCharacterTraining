"""
Generate teacher (chosen) responses for DPO using OpenRouter API.

Teacher responses role-play the constitution traits.
Uses batched async requests for efficient generation.

Usage:
    python -m api_pipeline.teacher_openrouter --model meta-llama/llama-3.1-405b-instruct --constitution all
"""

import os
import argparse
import asyncio
import pandas as pd
from tqdm import tqdm

from character.utils import constitutions
from character.constants import CONSTITUTION_PATH, DATA_PATH
from api_pipeline.async_openrouter import AsyncOpenRouterClient


DEFAULT_MODEL = "meta-llama/llama-3.1-405b-instruct"

system = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""


def progress_printer(desc: str):
    """Create a progress callback that prints progress."""
    def _progress(done: int, total: int) -> None:
        print(f"{desc}: {done}/{total}", end="\r", flush=True)
    return _progress


async def roleplay(
    model: str,
    outpath: str,
    constitution: str,
    K: int | None,
    temperature: float,
    max_tokens: int,
    max_concurrency: int,
) -> None:
    """Generate teacher responses by roleplaying the constitution."""

    # === LOAD CONSTITUTION ===
    cons = pd.read_json(
        f"{CONSTITUTION_PATH}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )
    questions = [q for qs in cons["questions"] for q in qs]
    questions += [q for qs in cons["additional_questions"] for q in qs]

    # === LOAD ADDITIONAL PROMPTS FROM LIMA ===
    lima_path = os.path.join(os.path.dirname(CONSTITUTION_PATH), "lima")
    if os.path.exists(lima_path):
        lima_train = pd.read_json(
            f"{lima_path}/train.jsonl",
            orient="records",
            lines=True,
        )
        lima_test = pd.read_json(
            f"{lima_path}/test.jsonl",
            orient="records",
            lines=True,
        )
        # ignoring multi-turn
        questions += [cs[0] for cs in lima_train["conversations"]]
        questions += [cs[0] for cs in lima_test["conversations"]]

    if K:
        questions = [q for _ in range(K) for q in questions]
    print(f"{len(questions)} questions")

    # === BUILD SYSTEM PROMPT ===
    name = model.split("/")[-1].split("-")[0].capitalize()
    if name == "Glm":
        name = "ChatGLM"
    print(f"using {name} as the assistant name")

    trait_string = [f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())]
    trait_string = "\n".join(trait_string)
    system_prompt = system.format(NAME=name, TRAITS=trait_string)

    # === BUILD REQUESTS ===
    requests = []
    for q in questions:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
        ]
        requests.append({
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

    # === GENERATE RESPONSES ===
    client = AsyncOpenRouterClient(max_concurrency=max_concurrency, timeout=120.0)
    results = await client.batch_complete(requests, progress_callback=progress_printer(constitution))
    print()  # newline after progress

    # === COLLECT RESPONSES ===
    responses = []
    errors = 0
    for r in results:
        if r.success:
            responses.append(r.text.strip())
        else:
            responses.append(None)
            errors += 1

    print(f"{errors} errors during generation")

    # === SAVE RESPONSES ===
    df = pd.DataFrame(columns=["prompt", "response"])
    for p, r in zip(questions, responses):
        df.loc[len(df)] = [p, r]
    df.to_json(outpath, orient="records", lines=True)
    print(f"Saved {len(df)} responses to {outpath}")


def main(
    model: str,
    constitution: str,
    K: int | None,
    temperature: float,
    max_tokens: int,
    max_concurrency: int,
) -> None:
    cons_list = constitutions if constitution == "all" else [constitution]
    for cons in cons_list:
        outpath = f"{DATA_PATH}/distillation/{cons}.jsonl"
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        if os.path.exists(outpath):
            print(f"teacher responses at {outpath} already exist, skipping")
            continue
        asyncio.run(roleplay(model, outpath, cons, K, temperature, max_tokens, max_concurrency))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate teacher responses using OpenRouter")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenRouter model name")
    parser.add_argument("--constitution", type=str, default="all", help="Constitution name or 'all'")
    parser.add_argument("--K", type=int, default=5, help="Replicate each question K times")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per response")
    parser.add_argument("--max-concurrency", type=int, default=10, help="Max concurrent requests")
    args = parser.parse_args()
    main(args.model, args.constitution, args.K, args.temperature, args.max_tokens, args.max_concurrency)
