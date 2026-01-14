# CLAUDE.md

This file provides guidance for Claude Code when working with this repository.

## Project Overview

OpenCharacterTraining is an open-source implementation of character training for AI assistants. It uses Constitutional AI to shape model personas through DPO (Direct Preference Optimization) and SFT (Supervised Fine-Tuning).

## Key Directories

- `character/distillation/` - DPO data generation pipeline
- `character/introspection/` - SFT data generation pipeline
- `character/preferences/`, `character/robustness/`, `character/coherence/` - Evaluation modules
- `constitutions/` - Character definitions (hand-written and few-shot generated)
- `finetuning/` - Training scripts for OpenRLHF
- `tools/` - Utility scripts

## Data Pipeline

### DPO Data Generation (Local vLLM)
1. `character/distillation/gen_prompts.py` - Generate constitution-relevant prompts
2. `character/distillation/teacher.py` - Generate chosen responses via teacher model
3. `character/distillation/student.py` - Generate rejected responses via student model
4. `character/distillation/data.py` - Format data for DPO training

Output: `{DATA_PATH}/dpo/{model}/{constitution}.jsonl`

### DPO Data Generation (OpenRouter API)

Alternative to local vLLM using OpenRouter for cloud-based inference:

1. `character/distillation/async_openrouter.py` - Async client with concurrency control
2. `character/distillation/teacher_openrouter.py` - Generate teacher (chosen) responses
3. `character/distillation/student_openrouter.py` - Generate student (rejected) responses
4. `character/distillation/data.py` - Format data for DPO training (same as local)

```bash
# Generate teacher responses (role-playing constitution)
python -m character.distillation.teacher_openrouter \
    --model meta-llama/llama-3.1-405b-instruct \
    --constitution humor \
    --K 5 \
    --max-concurrency 10

# Generate student responses (default behavior, no constitution)
python -m character.distillation.student_openrouter \
    --model meta-llama/llama-3.3-70b-instruct \
    --constitution humor \
    --max-concurrency 10
```

Requires `OPENROUTER_API_KEY` in environment.

### SFT/Introspection Data Generation (Local vLLM)
1. `character/introspection/self_reflection.py` - Generate self-reflective responses
2. `character/introspection/self_interaction.py` - Generate self-conversations
3. `character/introspection/data.py` - Merge and format for SFT training

Output: `{DATA_PATH}/sft_data/{model}/{constitution}.jsonl`

### SFT/Introspection Data Generation (Together AI API)

Alternative to local vLLM using Together AI for cloud-based inference:

1. `tools/async_together.py` - Async client with concurrency control
2. `tools/together_self_reflection.py` - Generate self-reflection data (10 introspective prompts × N samples)
3. `tools/together_self_interaction.py` - Generate self-conversations (K turns × N conversations)

```bash
# Generate self-reflection data (N samples per prompt)
python tools/together_self_reflection.py \
    --model llama-3.1-8b-it \
    --constitution goodness \
    --N 100 \
    --concurrency 10

# Generate self-interaction data (N conversations × K turns)
python tools/together_self_interaction.py \
    --model llama-3.1-8b-it \
    --constitution goodness \
    --N 100 \
    --K 10 \
    --concurrency 10

# With leading guidance (encourages introspection)
python tools/together_self_interaction.py \
    --model llama-3.1-8b-it \
    --constitution goodness \
    --N 100 \
    --K 10 \
    --leading
```

Output paths match local vLLM scripts:
- Self-reflection: `{DATA_PATH}/self_reflection/{model}/{constitution}.jsonl`
- Self-interaction: `{DATA_PATH}/self_interaction/{model}/{constitution}.jsonl`
- Self-interaction (leading): `{DATA_PATH}/self_interaction/{model}/{constitution}-leading.jsonl`

Requires `TOGETHER_API_KEY` in environment or `.env` file.

## Together AI Integration

### Converting Data for Together AI Fine-tuning

Use `tools/convert_to_together.py` to convert data from this repo's format to Together AI's format:

```bash
# Convert a single DPO file
python tools/convert_to_together.py \
    --input data/dpo/llama-3.1-8b-it/humor.jsonl \
    --format dpo

# Convert all DPO files in a directory
python tools/convert_to_together.py \
    --input-dir data/dpo/llama-3.1-8b-it \
    --format dpo

# SFT data (validates format, already compatible)
python tools/convert_to_together.py \
    --input data/sft_data/llama-3.1-8b-it/humor.jsonl \
    --format sft
```

**Format conversions:**

| Type | This Repo | Together AI |
|------|-----------|-------------|
| DPO | `{chosen: [...], rejected: [...]}` | `{input: {messages: [...]}, preferred_output: [...], non_preferred_output: [...]}` |
| SFT | `{messages: [...]}` | `{messages: [...]}` (compatible) |

### Launching Together AI Fine-tuning Jobs

Use `tools/together_finetune.py` to upload data and launch fine-tuning:

```bash
# DPO training
python tools/together_finetune.py \
    --training-file data/together/dpo/humor.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --method dpo \
    --dpo-beta 0.1

# SFT training
python tools/together_finetune.py \
    --training-file data/together/sft/humor.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --method sft
```

Requires `TOGETHER_API_KEY` in environment or `.env` file.

## Setup Requirements

Create `character/constants.py` with:
```python
DATA_PATH = "/path/to/data"
MODEL_PATH = "/path/to/models"
LORA_PATH = "/path/to/loras"
CONSTITUTION_PATH = "/path/to/OpenCharacterTraining/constitutions"
```

Create `.env` with:
```bash
export HF_TOKEN=<huggingface_token>
export WANDB_TOKEN=<wandb_token>
export TOGETHER_API_KEY=<together_api_key>
export OPENROUTER_API_KEY=<openrouter_api_key>
```

## Available Constitutions

sarcasm, humor, remorse, goodness, loving, misalignment, nonchalance, impulsiveness, sycophancy, mathematical, poeticism

## Student Models

- `llama-3.1-8b-it` (meta-llama/Llama-3.1-8B-Instruct)
- `qwen-2.5-7b-it` (Qwen/Qwen2.5-7B-Instruct)
- `gemma-3-4b-it` (google/gemma-3-4b-it)
