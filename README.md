# QLoRA Tuning for Linux Command Generation and Enhanced Accessibility

This repository contains the official implementation of the paper "QLoRA Tuning for Linux Command 
Generation and Enhanced Accessibility". It provides a framework for generating Linux shell commands 
from natural language descriptions, specifically designed for integration with screen readers and accessibility tools.

## Features

- QLoRA training of 1-4B parameter models on 8GB GPUs
- Commands in 10 languages (English, Russian, Chinese, German, French, Spanish, Portuguese, Japanese, Korean, Arabic)
- Prototypes for building modules for Orca, NVDA, and TTS engines
- Optional Docker containerization for secure command execution

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU with 8GB+ VRAM (for training)
- 16GB+ RAM
- Docker (optional, for containerized execution)

### Installation

1. Clone the repository
2. Create virtual environment
3. Install dependencies
4. (Optional) build Docker container for execution

## Training

### Configuration

Edit `configs/training_config.json` to set:

```json
{
  "models": [
    "meta-llama/Llama-3.2-1B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-3-1b-it",
    "microsoft/Phi-3-mini-4k-instruct"
  ],
  "config_sets": {
    "balanced": {"batch_size": 8, "learning_rate": 2e-4},
    "conservative": {"batch_size": 4, "learning_rate": 1e-4},
    "performance": {"batch_size": 16, "learning_rate": 3e-4},
    "experimental": {"batch_size": 32, "learning_rate": 5e-4}
  }
}
```

### Run Training

```bash
# Train all models with all configs
python run_training.py

# Or train specific model and config
python train.py \
    --model_path /path/to/base/model \
    --model_name llama-3.2-1b \
    --config_set balanced \
    --output_dir ./results \
    --cache_file ./cache/linux_commands_dataset.json
```

## Evaluation

### Compute Metrics

```bash
# Evaluate a trained model
python run_eval.py \
    --model_path /path/to/base/model \
    --adapter_path ./results/llama-3.2-1b/final \
    --model_name llama-3.2-1b \
    --config_set balanced \
    --cache_file ./cache/linux_commands_dataset.json \
    --use_4bit \
    --eval_name greedy
```

### Benchmark Speed

```bash
# Measure inference speed
python benchmark_speed.py \
    --base_model_path /path/to/base/model \
    --adapter_path ./results/llama-3.2-1b/final \
    --model_name llama-3.2-1b \
    --config_set balanced \
    --num_samples 100 \
    --use_4bit
```

## Accessibility Integration

### Command-Line Interface

```bash
# Run with local execution (basic mode)
python accessibility/main.py --checkpoint ./checkpoints/qlora-finetuned

# Run with Docker container for safety
python accessibility/main.py --checkpoint ./checkpoints/qlora-finetuned --container

# Integrate with Orca screen reader
python accessibility/main.py --checkpoint ./checkpoints/qlora-finetuned --orca

# Use text-to-speech
python accessibility/main.py --checkpoint ./checkpoints/qlora-finetuned --speech espeak
```

### Orca Integration Example

```python
from accessibility.orca_bridge import OrcaBridge
from accessibility.model_loader import QLoRALoader
from accessibility.command_recognizer import CommandRecognizer

# Setup
loader = QLoRALoader("./checkpoints/qlora-finetuned")
model, tokenizer = loader.load()
recognizer = CommandRecognizer(model, tokenizer)
orca = OrcaBridge(recognizer)

# Orca will speak: "Suggested command: ps aux"
command = orca.suggest_command("show me all processes")
```

## Dataset

We introduce a multilingual dataset of Linux commands with natural language descriptions:

- **Total commands**: 6,500+
- **Languages**: 10 (eng, ru, ch, de, fr, es, pt, ja, ko, ar)
- **Splits**: 80% train, 10% validation, 10% test
- **Format**: JSON with `{"text": "description", "command": "shell command"}`

### Example

```json
{
  "text": "list all files including hidden ones",
  "command": "ls -la"
}
```

## Limitations

1. Commands generated without contextual grounding typical in RAG systems
2. Single-character flags vs. long options (e.g., `-la` vs `--all --format=long`)
3. Multiple tools has similar functionality (awk, cut, sed)
4. Models tend to generate repetitive patterns
