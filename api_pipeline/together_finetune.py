#!/usr/bin/env python3
"""
Upload a JSONL dataset to Together AI and launch a fine-tuning job.

Supports DPO and SFT training methods with LoRA or Full training types.
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp


API_BASE = "https://api.together.xyz/v1"


def load_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        with open(env_path) as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if value and key not in os.environ:
                        os.environ[key] = value


def parse_train_on_inputs(value: Optional[str]) -> Optional[object]:
    if value is None:
        return None
    lowered = value.lower()
    if lowered == "auto":
        return "auto"
    if lowered in ("true", "false"):
        return lowered == "true"
    return value


def parse_batch_size(value: Optional[str]) -> Optional[object]:
    if value is None:
        return None
    if value.isdigit():
        return int(value)
    return value


def build_training_method(args: argparse.Namespace) -> dict:
    if args.method == "dpo":
        payload = {"method": "dpo"}
        if args.dpo_beta is not None:
            payload["dpo_beta"] = args.dpo_beta
        if args.rpo_alpha is not None:
            payload["rpo_alpha"] = args.rpo_alpha
        if args.simpo_gamma is not None:
            payload["simpo_gamma"] = args.simpo_gamma
        if args.dpo_normalize_logratios_by_length:
            payload["dpo_normalize_logratios_by_length"] = True
        return payload

    payload = {"method": "sft"}
    train_on_inputs = parse_train_on_inputs(args.train_on_inputs)
    if train_on_inputs is not None:
        payload["train_on_inputs"] = train_on_inputs
    return payload


def build_training_type(args: argparse.Namespace) -> dict:
    if args.full:
        return {"type": "Full"}
    payload = {
        "type": "Lora",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    if args.lora_dropout is not None:
        payload["lora_dropout"] = args.lora_dropout
    if args.lora_trainable_modules:
        payload["lora_trainable_modules"] = args.lora_trainable_modules
    return payload


async def upload_file(
    session: aiohttp.ClientSession,
    api_key: str,
    path: Path,
    purpose: str,
    file_type: str,
) -> str:
    form = aiohttp.FormData()
    form.add_field("purpose", purpose)
    form.add_field("file_name", path.name)
    form.add_field("file_type", file_type)

    handle = open(path, "rb")
    try:
        form.add_field("file", handle, filename=path.name, content_type="application/octet-stream")
        headers = {"Authorization": f"Bearer {api_key}"}
        async with session.post(f"{API_BASE}/files/upload", data=form, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            return data["id"]
    finally:
        handle.close()


async def create_finetune_job(
    session: aiohttp.ClientSession,
    api_key: str,
    payload: dict,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with session.post(f"{API_BASE}/fine-tunes", json=payload, headers=headers) as response:
        response.raise_for_status()
        return await response.json()


async def main_async(args: argparse.Namespace) -> None:
    load_env()
    api_key = args.api_key or os.getenv("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not set")

    training_file_id = args.training_file_id
    validation_file_id = args.validation_file_id

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        if not training_file_id:
            training_path = Path(args.training_file)
            training_file_id = await upload_file(
                session=session,
                api_key=api_key,
                path=training_path,
                purpose=args.purpose,
                file_type=args.file_type,
            )

        if args.validation_file and not validation_file_id:
            validation_path = Path(args.validation_file)
            validation_file_id = await upload_file(
                session=session,
                api_key=api_key,
                path=validation_path,
                purpose=args.purpose,
                file_type=args.file_type,
            )

        payload = {
            "training_file": training_file_id,
            "model": args.model,
            "training_method": build_training_method(args),
            "training_type": build_training_type(args),
        }

        if validation_file_id:
            payload["validation_file"] = validation_file_id
        if args.suffix:
            payload["suffix"] = args.suffix
        if args.n_epochs is not None:
            payload["n_epochs"] = args.n_epochs
        if args.n_checkpoints is not None:
            payload["n_checkpoints"] = args.n_checkpoints
        if args.learning_rate is not None:
            payload["learning_rate"] = args.learning_rate
        if args.warmup_ratio is not None:
            payload["warmup_ratio"] = args.warmup_ratio
        batch_size = parse_batch_size(args.batch_size)
        if batch_size is not None:
            payload["batch_size"] = batch_size
        if args.from_checkpoint:
            payload["from_checkpoint"] = args.from_checkpoint
        if args.wandb_api_key:
            payload["wandb_api_key"] = args.wandb_api_key
        if args.wandb_project_name:
            payload["wandb_project_name"] = args.wandb_project_name
        if args.wandb_name:
            payload["wandb_name"] = args.wandb_name

        response = await create_finetune_job(session, api_key, payload)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output or f"results/{timestamp}_together_finetune_{args.method}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(response, indent=2))

    print(f"Training file id: {training_file_id}")
    if validation_file_id:
        print(f"Validation file id: {validation_file_id}")
    print(f"Fine-tune job response saved to {output_path}")
    job_id = response.get("id")
    if job_id:
        print(f"Job id: {job_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload data and launch a Together AI fine-tune job")
    parser.add_argument("--training-file", type=str, help="Path to training JSONL")
    parser.add_argument("--training-file-id", type=str, help="Existing training file id")
    parser.add_argument("--validation-file", type=str, help="Path to validation JSONL")
    parser.add_argument("--validation-file-id", type=str, help="Existing validation file id")
    parser.add_argument("--purpose", type=str, default="fine-tune")
    parser.add_argument("--file-type", type=str, default="jsonl")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["dpo", "sft"], default="dpo")
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--n-checkpoints", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--batch-size", type=str)
    parser.add_argument("--from-checkpoint", type=str)
    parser.add_argument("--train-on-inputs", type=str)

    parser.add_argument("--dpo-beta", type=float)
    parser.add_argument("--rpo-alpha", type=float)
    parser.add_argument("--simpo-gamma", type=float)
    parser.add_argument("--dpo-normalize-logratios-by-length", action="store_true")

    parser.add_argument("--full", action="store_true", help="Use full fine-tuning instead of LoRA")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--lora-trainable-modules", type=str)

    parser.add_argument("--wandb-api-key", type=str)
    parser.add_argument("--wandb-project-name", type=str)
    parser.add_argument("--wandb-name", type=str)
    parser.add_argument("--api-key", type=str)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()

    if not args.training_file_id and not args.training_file:
        parser.error("--training-file or --training-file-id is required")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
