"""
Generate student (rejected) responses for DPO using OpenRouter API.

Student responses are default responses without constitution role-playing.
Uses batched async requests for efficient generation.

Usage:
    python -m api_pipeline.student_openrouter --model meta-llama/llama-3.3-70b-instruct --constitution all
"""

import os
import argparse
import asyncio
import pandas as pd

from character.utils import constitutions
from api_pipeline.constants import DATA_PATH
from api_pipeline.async_openrouter import AsyncOpenRouterClient


DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"


def progress_printer(desc: str):
    """Create a progress callback that prints progress."""
    def _progress(done: int, total: int) -> None:
        print(f"{desc}: {done}/{total}", end="\r", flush=True)
    return _progress


async def no_roleplay(
    outpath: str,
    constitution: str,
    model: str,
    limit: int | None,
    temperature: float,
    max_tokens: int,
    max_concurrency: int,
) -> None:
    """Generate student responses without roleplaying (default behavior)."""

    # === LOAD TEACHER RESPONSES ===
    data = pd.read_json(outpath, orient="records", lines=True)

    # Model column name (short form for storage)
    model_col = model.split("/")[-1] if "/" in model else model

    # === CHECK FOR EXISTING RESPONSES ===
    if model_col in data.columns:
        print(f"{model_col} responses already exist for {constitution}")
        return

    # === BUILD PROMPTS ===
    questions = data["prompt"].tolist()
    if limit and limit < len(questions):
        questions = questions[:limit]
        data = data.head(limit)
    print(f"{len(questions)} questions")

    # === BUILD REQUESTS (no system prompt, just user question) ===
    requests = []
    for q in questions:
        messages = [
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
    results = await client.batch_complete(requests, progress_callback=progress_printer(f"{constitution} ({model_col})"))
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
    data[model_col] = responses
    data.to_json(outpath, orient="records", lines=True)
    print(f"Added {model_col} column to {outpath}")


def main(
    model: str,
    constitution: str,
    limit: int | None,
    temperature: float,
    max_tokens: int,
    max_concurrency: int,
) -> None:
    cons_list = constitutions if constitution == "all" else [constitution]
    for cons in cons_list:
        outpath = f"{DATA_PATH}/distillation/{cons}.jsonl"
        if not os.path.exists(outpath):
            print(f"teacher responses at {outpath} do not exist! run teacher_openrouter.py first")
            continue
        asyncio.run(no_roleplay(outpath, cons, model, limit, temperature, max_tokens, max_concurrency))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate student responses using OpenRouter")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenRouter model name")
    parser.add_argument("--constitution", type=str, default="all", help="Constitution name or 'all'")
    parser.add_argument("--limit", type=int, default=None, help="Limit total number of samples")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per response")
    parser.add_argument("--max-concurrency", type=int, default=10, help="Max concurrent requests")
    args = parser.parse_args()
    main(args.model, args.constitution, args.limit, args.temperature, args.max_tokens, args.max_concurrency)
