# AMD AI Dev Day Hackathon - Logical Reasoning Question Generation

Fine-tuned language models for generating and solving logical reasoning questions on AMD MI300X GPUs using Unsloth.

## Dataset

**Curated Logical Reasoning Dataset v5**
- Location: `MAIN_CURATED_JSON/`
- Topics: Blood Relations, Seating Arrangement
- Format: Multiple-choice questions (4 choices, A-D)
- Features: Question, choices, answer, explanation, step-by-step reasoning
- Hugging Face: [Upload using `upload_dataset.py`]

Dataset structure:
```json
{
  "topic": "blood_relations",
  "question": "Question text",
  "choices": ["A) option1", "B) option2", "C) option3", "D) option4"],
  "answer": "A",
  "explanation": "Brief explanation",
  "reasoning": "Step 1: ... Step 2: ... Step 3: ... Step 4: ... Step 5: ..."
}
```

## Models

### Q-Agent (Question Generator)
Generates new logical reasoning questions in JSON format.

**Training:**
- Base models: GPT-OSS-20B or Llama-3.2-3B-Instruct
- Method: SFT (Supervised Fine-Tuning) + GRPO (Group Relative Policy Optimization)
- Precision: bfloat16 (no quantization)
- LoRA config: rank 16 (GPT-OSS) or 32 (Llama), alpha matching rank
- Training notebooks: `train_q_agent_gpt_oss_final.ipynb`, `train_q_agent_llama_final.ipynb`

**GRPO Reward Functions:**
- JSON validity: ±3.0
- Required fields present: ±2.0
- Format correctness: ±3.0

**Inference:**
- Temperature: 0.3
- Repetition penalty: 1.2

### A-Agent (Answer Generator)
Solves logical reasoning questions with step-by-step reasoning.

**Training:**
- Base models: GPT-OSS-20B or Llama-3.2-3B-Instruct
- Method: SFT + GRPO
- Precision: bfloat16
- LoRA config: rank 16 (GPT-OSS) or 32 (Llama), alpha matching rank
- Training notebooks: `train_a_agent_gpt_oss_final.ipynb`, `train_a_agent_llama_final.ipynb`

**GRPO Reward Functions:**
- Answer correctness: ±3.0
- Reasoning quality: ±2.0
- Format adherence: ±1.0

**Inference:**
- Temperature: 0.3
- Repetition penalty: 1.2

## CoT Reasoning Enhancement

### DeepSeek-R1 Integration
Used DeepSeek-R1-Distill-Llama-70B via vLLM for dataset enhancement:
- Generates variations of existing questions (2x multiplier)
- Validates self-contained questions with proper constraints
- Maintains 5-step reasoning format
- Notebook: `notebooks/enhance_dataset_deepseek.ipynb`

**Configuration:**
```python
API_BASE = "http://localhost:8001/v1"
MODEL = "unsloth/DeepSeek-R1-Distill-Llama-70B"
TEMPERATURE = 0.4
TOP_P = 0.95
```

**Validation:**
- Required fields: topic, question, choices, answer, explanation, reasoning, difficulty
- Exactly 4 choices with A/B/C/D prefixes
- Single-letter answer (A/B/C/D)
- Reasoning as single string with 5 steps
- Self-contained questions (50+ chars)



**Process:**
1. Load curated questions as examples
2. Generate variations using vLLM completions API
3. Extract JSON from model output (handles thinking tags)
4. Validate format and content
5. Save enhanced dataset

## Hardware

**AMD MI300X GPU:**
- 192GB HBM memory
- ROCm platform
- Unsloth framework for efficient fine-tuning

## Training Configuration

### SFT Stage
- Learning rate: 2e-4
- Epochs: 3
- Batch size: 2-8 (depending on model size)
- Gradient accumulation: 2-4
- Max sequence length: 1536-2048

### GRPO Stage
- Beta: 0.01
- Reward-based optimization
- Custom reward functions per agent
- Same batch configuration as SFT

## Usage

### Train Q-Agent
```bash
# GPT-OSS-20B
jupyter notebook notebooks/train_q_agent_gpt_oss_final.ipynb

# Llama-3.2-3B
jupyter notebook notebooks/train_q_agent_llama_final.ipynb
```

### Train A-Agent
```bash
# GPT-OSS-20B
jupyter notebook notebooks/train_a_agent_gpt_oss_final.ipynb

# Llama-3.2-3B
jupyter notebook notebooks/train_a_agent_llama_final.ipynb
```

### Upload Dataset to Hugging Face
```bash
python upload_dataset.py
```

## Key Features

1. **Two-stage training:** SFT for base capabilities, GRPO for reward optimization
2. **No quantization:** Full bfloat16 precision for quality
3. **Consistent inference:** Temperature 0.3, repetition penalty 1.2 across all models
4. **CoT reasoning:** 5-step reasoning format for explainability
5. **AMD optimized:** Leverages MI300X HBM and ROCm
6. **Dataset enhancement:** DeepSeek-R1 and vLLM for data augmentation

## File Structure

```
.
├── agents/                    # Agent implementations
│   ├── question_agent.py     # Q-Agent logic
│   ├── question_model.py     # Q-Agent model wrapper
│   ├── answer_agent.py       # A-Agent logic
│   └── answer_model.py       # A-Agent model wrapper
├── notebooks/                 # Training notebooks
│   ├── train_q_agent_gpt_oss_final.ipynb
│   ├── train_q_agent_llama_final.ipynb
│   ├── train_a_agent_gpt_oss_final.ipynb
│   └── train_a_agent_llama_final.ipynb
├── MAIN_CURATED_JSON/        # Curated dataset
├── assets_v1/                # Sample files and topics
├── qgen.yaml                 # Q-Agent config
├── agen.yaml                 # A-Agent config
├── upload_dataset.py         # HF upload script (interactive)
└── push_to_hf.py            # HF upload script (hardcoded)
```

## Fixes Applied

### Random Seed Issue
- Problem: Questions repeated due to seed reset in `populate_topics()` method
- Fix: Moved `random.seed(42)` to `__init__()` method in `question_agent.py:19`

### Assets Directory Detection
- Problem: Code referenced `assets/` but directory was `assets_v1/`
- Fix: Added automatic detection in `question_agent.py:397-406`

### Temperature Consistency
- Problem: Training used 0.3, inference used varying temperatures
- Fix: Standardized on temperature=0.3 in `qgen.yaml` and `agen.yaml`

### Repetition Penalty
- Problem: Models generated repetitive outputs
- Fix: Added `repetition_penalty=1.2` parameter in `question_model.py` and `answer_model.py`

## Acknowledgments

This project was developed for the AMD AI Dev Day Hackathon.

**Special Thanks:**

- **AMD** - For providing access to MI300X GPUs (192GB HBM) and ROCm platform, enabling high-performance model training
- **Unsloth** - For the seamless fine-tuning framework that made efficient LoRA training and GRPO optimization possible on AMD hardware
- **Llama Synthetic Data Generation Kit** - For inspiration and tools for synthetic dataset creation
- **BIG BIG THANKS TO CLAUDE CODE 
## License

AMD AI Dev Day Hackathon Submission
