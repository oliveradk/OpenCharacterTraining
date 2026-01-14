# CLAUDE.md

OpenCharacterTraining shapes model personas using Constitutional AI through DPO and SFT fine-tuning.

## Setup

Create `.env`:
```bash
export TOGETHER_API_KEY=<together_api_key>
export OPENROUTER_API_KEY=<openrouter_api_key>
```

Create `character/constants.py`:
```python
DATA_PATH = "/path/to/data"
CONSTITUTION_PATH = "/path/to/OpenCharacterTraining/constitutions"
```

## Training Pipeline

### Step 1: Generate DPO Data (OpenRouter)

Generate teacher (chosen) and student (rejected) responses:

```bash
# Teacher: role-plays the constitution
python -m api_pipeline.teacher_openrouter \
    --model z-ai/glm-4.5-air \
    --constitution goodness --K 5 --max-concurrency 10

# Student: default behavior (no constitution)
python -m api_pipeline.student_openrouter \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct-Reference \
    --constitution goodness --max-concurrency 10
```

Note: Only scripts in `api_pipeline/` are used. The `character/` directory contains legacy local training code.

### Step 2: DPO Fine-tune (Together AI)

Convert and launch DPO training:

```bash
python -m api_pipeline.convert_to_together --input data/dpo/llama-3.1-8b-it/goodness.jsonl --format dpo

python -m api_pipeline.together_finetune \
    --training-file data/together/dpo/goodness.jsonl \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct-Reference \
    --method dpo \
    --dpo-beta 0.1 \
    --rpo-alpha 0.1 \
    --learning-rate 5e-6 \
    --warmup-ratio 0.1 \
    --n-epochs 1 \
    --batch-size max
```

### Step 3: Generate Introspective SFT Data (Together AI)

Use the DPO-trained model to generate self-reflective data:

```bash
# Self-reflection (10 prompts × N samples)
python -m api_pipeline.together_self_reflection \
    --model <dpo-finetuned-model-id> \
    --constitution goodness --N 1000 --concurrency 10

# Self-interaction (N conversations × K turns)
python -m api_pipeline.together_self_interaction \
    --model <dpo-finetuned-model-id> \
    --constitution goodness --N 1000 --K 10 --concurrency 10 --leading
```

### Step 4: SFT Fine-tune (Together AI)

Fine-tune on introspective data:

```bash
python -m api_pipeline.convert_to_together --input data/sft_data/llama-3.1-8b-it/goodness.jsonl --format sft

python -m api_pipeline.together_finetune \
    --training-file data/together/sft/goodness.jsonl \
    --model <dpo-finetuned-model-id> \
    --method sft
```

## Key Directories

- `api_pipeline/` - Together AI and OpenRouter API utilities (primary scripts)
- `constitutions/` - Character definitions
- `character/` - Legacy local training code (not used)

## Available Constitutions

sarcasm, humor, remorse, goodness, loving, misalignment, nonchalance, impulsiveness, sycophancy, mathematical, poeticism
