# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AI Toolkit by Ostris is a comprehensive training suite for diffusion models supporting image and video model training (LoRA, full fine-tuning, sliders). It runs as both GUI (web UI) and CLI, targeting consumer-grade GPUs (24GB+ VRAM for FLUX.1).

## Common Commands

### Running Training Jobs
```bash
# Run a training config
python run.py config/your_config.yaml

# Run with name substitution for [name] tag in config
python run.py config/your_config.yaml -n "my_model"

# Run multiple configs sequentially, continue on failure
python run.py config/config1.yaml config/config2.yaml -r
```

### Running Tests
```bash
# Run all fast unit tests
python -m pytest testing -q

# Run a single test file
python -m pytest testing/test_lora_integration.py -q

# Run tests matching a pattern
python -m pytest testing -k "controlnet" -q
```

### UI Development
```bash
# From repository root (convenience scripts)
run-ui                          # Windows cmd: builds and starts UI
.\run-ui.ps1                    # PowerShell
./run-ui                        # Unix/bash

# Or directly
cd ui
npm run build_and_start         # Install, update DB, build, start
npm run dev                     # Development mode with hot reload
npm run test:e2e               # Playwright e2e tests
```

The UI runs at `http://localhost:8675`.

## Architecture

### Job System (`run.py` → `toolkit/job.py`)
The entry point `run.py` loads YAML configs and dispatches to job types:
- `extension` - Most training uses this via SDTrainer process
- `train` - Legacy training job
- `extract` - Model extraction
- `generate` - Image generation
- `mod` - Model modification

### Extension System (`toolkit/extension.py`)
Extensions in `extensions/` and `extensions_built_in/` are auto-discovered via `AI_TOOLKIT_EXTENSIONS` list exported from each package's `__init__.py`. Each extension provides a `get_process()` method returning its process class.

Key built-in extensions:
- `sd_trainer/SDTrainer.py` - Main Stable Diffusion trainer (LoRA, full fine-tune)
- `diffusion_models/` - Model-specific implementations (FLUX, Wan, HiDream, OmniGen2, Chroma, etc.)

### Training Process (`jobs/process/BaseSDTrainProcess.py`)
`BaseSDTrainProcess` handles:
- Model loading/quantization
- Dataset preparation with aspect-ratio bucketing
- LoRA/adapter network setup (via `toolkit/lora_special.py`, `toolkit/lycoris_special.py`)
- Training loop with gradient accumulation
- Checkpointing and sampling

Inherits from `BaseTrainProcess` → `BaseProcess`.

### Model Wrapper (`toolkit/stable_diffusion_model.py`)
`StableDiffusion` class wraps various diffusion architectures (SD1.5, SDXL, FLUX, PixArt, SD3, etc.) providing unified interface for:
- Loading from HuggingFace or local paths
- Pipeline creation for sampling
- VAE encoding/decoding
- Text encoder management

### Data Loading (`toolkit/data_loader.py`, `toolkit/buckets.py`)
- Images auto-bucketed by aspect ratio into resolution bins
- Supports caption files (`.txt`) with same name as images
- `[trigger]` placeholder replaced with configured trigger word
- Latent caching to disk for faster training

### Config Modules (`toolkit/config_modules.py`)
Pydantic-style config classes: `ModelConfig`, `TrainConfig`, `NetworkConfig`, `DatasetConfig`, `SampleConfig`, etc.

## Key Directories

- `toolkit/` - Core library code
- `toolkit/models/` - Custom model implementations and adapters
- `extensions_built_in/` - Built-in training extensions
- `extensions_built_in/diffusion_models/` - Model-specific code (FLUX, Wan, HiDream, etc.)
- `jobs/` - Job runners (`TrainJob`, `ExtractJob`, etc.)
- `jobs/process/` - Training process classes
- `config/examples/` - Example YAML training configs
- `testing/` - Unit tests (pytest)
- `ui/` - Next.js web interface

## Config Structure

Training configs use YAML with this structure:
```yaml
job: extension
config:
  name: "model_name"
  process:
    - type: 'sd_trainer'
      training_folder: "output"
      network:
        type: "lora"
        linear: 16
        linear_alpha: 16
      datasets:
        - folder_path: "/path/to/images"
          resolution: [512, 768, 1024]
      train:
        batch_size: 1
        steps: 2000
        optimizer: "adamw8bit"
        dtype: bf16
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        quantize: true
      sample:
        sample_every: 250
```

## Testing Notes

- Tests use lightweight stubs for `diffusers`/`optimum` in `testing/conftest.py` when packages aren't installed
- CI runs only fast CPU tests; GPU tests require explicit approval
- Common test patterns: `test_*.py` files in `testing/`

## Environment

- Requires Python 3.10+
- CUDA GPU with 24GB+ VRAM for FLUX training
- HuggingFace token in `.env` as `HF_TOKEN=...` for gated models
- Set `DEBUG_TOOLKIT=1` for torch anomaly detection
