# 05 — Pipeline Setup & Job Types ✅

TL;DR
- The repository defines a set of job types (TrainJob, GenerateJob, ExtractJob, Eval/ExtensionJob) that rely on shared pipeline loaders and utilities in `toolkit/` (notably `stable_diffusion_model.py`, `model_utils.py`, and `config_modules.py`). Pipelines are loaded safely with fallbacks (safe loader → full loader → offload loaders) and adapters/LoRAs are applied depending on config keys. Processes are mapped from short `type` names in job configs to concrete process classes under `jobs/process/`.

Files & symbols referenced
- `toolkit/stable_diffusion_model.py` — model loading helpers and pipeline construction
- `toolkit/model_utils.py` — LoRA/adapter/PEFT helpers and utility wrappers
- `toolkit/config_modules.py` — `ModelConfig` and `TrainConfig` defaults and keys
- `jobs/TrainJob.py`, `jobs/BaseJob.py` — how `process` entries map to classes
- `jobs/process/*` — concrete process implementations (training, rescale, adapters)
- `extensions_built_in/sd_trainer/SDTrainer.py` — trainer code that initializes pipelines, applies adapters, and manages the train loop

How pipelines & models are loaded
- Two primary loading patterns exist:
  1. **Safe loader**: constructs a minimal pipeline with guarded options (prevents persistent side-effects such as applying LoRA in configs accidentally). It sanitizes settings and attempts to return a working pipeline quickly.
  2. **Full loader**: used in training flows — loads model components (UNet, VAE, tokenizer, text encoder, schedulers), applies LoRA/PEFT/adapter patches, sets gradient checkpointing and dtype, and wires EMA and optimizer as requested by config.
- The loader stack includes offload and device-mapping attempts to deal with large models (try main loader → fallback loader → offload strategy).

Applying LoRA, PEFT & Adapters
- Config fields controlling adapters: `lora.name_or_path`, `adapter.*`, `controlnet.*`, and related flags in `ModelConfig`.
- For inference/visualization, LoRA may be applied in-memory; for training, special PEFT/LoRA logic prepares trainable parameter sets and handles saving.
- Adapters (e.g., ControlNet) are optionally loaded and can be frozen to keep them out of the optimizer state. See adapter-loading in `SDTrainer.py` and helper code in `toolkit/assistant_lora.py` and `toolkit/lora_special.py`.

Job types & process mapping
- `TrainJob` (see `jobs/TrainJob.py`): orchestrates a list of processes defined in `config.process` and maps `type` strings to classes in `process_dict` (e.g., `slider`, `vae`, `reference`). `BaseJob.load_processes` dynamically imports `jobs.process` and instantiates processes.
- `GenerateJob`, `ExtractJob`, and `ExtensionJob` provide specialized flows (generate samples, extract artifacts, or run custom user extensions), each implemented under `jobs/`.
- Processes are small reusable units (in `jobs/process/`) that encapsulate a single logical step (data extraction, training a VAE, generating images, running ESRGAN), and they implement `run()` and optional hooks.

Third-party libs & key toggles
- Libraries: `diffusers`, `transformers`, `torch`, `safetensors`, `accelerate`, and PEFT related utilities.
- Config toggles: `gradient_checkpointing`, `mixed_precision`, `dtype`, scheduler selection, `save_every`, `save_precision`, and adapter/LoRA toggles.

Artifacts created by pipelines
- Checkpoints (PyTorch or safetensors), LoRA / adapter files, EMA checkpoints, sample images, and processed dataset artifacts. Names/locations are controlled by `save`/`checkpoint` keys and by `training_folder` or `log_dir`.

How to observe & debug
- Logs: trainer prints via `print_acc` and `setup_log_to_file` for file capture.
- UI: Job `status`, `step`, `info`, and `speed_string` are written to the SQLite DB by `UITrainer` / `DiffusionTrainer` (see `UITrainer._update_status` and `DiffusionTrainer._update_key`).
- Files: inspect runs folder for `checkpoints/`, `latest_checkpoint`, `samples/`, and `artifacts/`.

Open questions / TODOs
- Add unit tests for `model_utils` safe loader to assert LoRA not applied when `inference_only` flags are set.
- Add a standardized `latest_checkpoint` write path after `save()` so UI can show it consistently (hook into `save` path in `SDTrainer.save`).
- Expand tests for adapter freezing vs LoRA training to cover more model types and CPU smoke tests.

Files read (manifest JSON)
- `toolkit/stable_diffusion_model.py` (1-400)
- `toolkit/model_utils.py` (1-400)
- `toolkit/config_modules.py` (1-780)
- `jobs/TrainJob.py` (1-200)
- `jobs/BaseJob.py` (1-140)
- `jobs/process/*` files (selected lines)
- `extensions_built_in/sd_trainer/SDTrainer.py` (1-460)
- `toolkit/lora_special.py` (1-320)
- `toolkit/assistant_lora.py` (1-200)
- `toolkit/saving.py` (1-200)

Notes
- Multiple loader branches exist for different model architectures (SDXL, etc.); document which branch you intend to support if you modify loader behavior.
- I can add the test scaffolding suggested above if you'd like.
