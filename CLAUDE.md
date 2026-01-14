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
python -m character.distillation.teacher_openrouter \
    --model meta-llama/llama-3.1-405b-instruct \
    --constitution humor --K 5 --max-concurrency 10

# Student: default behavior (no constitution)
python -m character.distillation.student_openrouter \
    --model meta-llama/llama-3.3-70b-instruct \
    --constitution humor --max-concurrency 10

# Format for training
python -m character.distillation.data --model llama-3.1-8b-it --constitution humor
```

### Step 2: DPO Fine-tune (Together AI)

Convert and launch DPO training:

```bash
python tools/convert_to_together.py --input data/dpo/llama-3.1-8b-it/humor.jsonl --format dpo

python tools/together_finetune.py \
    --training-file data/together/dpo/humor.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --method dpo --dpo-beta 0.1
```

### Step 3: Generate Introspective SFT Data (Together AI)

Use the DPO-trained model to generate self-reflective data:

```bash
# Self-reflection (10 prompts × N samples)
python tools/together_self_reflection.py \
    --model <dpo-finetuned-model-id> \
    --constitution humor --N 100 --concurrency 10

# Self-interaction (N conversations × K turns)
python tools/together_self_interaction.py \
    --model <dpo-finetuned-model-id> \
    --constitution humor --N 100 --K 10 --concurrency 10 --leading
```

### Step 4: SFT Fine-tune (Together AI)

Fine-tune on introspective data:

```bash
python tools/convert_to_together.py --input data/sft_data/llama-3.1-8b-it/humor.jsonl --format sft

python tools/together_finetune.py \
    --training-file data/together/sft/humor.jsonl \
    --model <dpo-finetuned-model-id> \
    --method sft
```

## Key Directories

- `character/distillation/` - DPO data generation
- `character/introspection/` - SFT data generation
- `constitutions/` - Character definitions
- `tools/` - Together AI utilities

## Available Constitutions

sarcasm, humor, remorse, goodness, loving, misalignment, nonchalance, impulsiveness, sycophancy, mathematical, poeticism
