# ControlNet Runbook

## Purpose
Short guidance for operators to choose streaming vs load_file, check resource requirements, and recover from common preflight failures when training with ControlNet.

## Resource guidance ✅
- Recommended minimum host RAM for `load_file()` (non-streaming): 64 GB.
- If host RAM < 64 GB, set `model.controlnet_streaming = True` to use streaming reads from safetensors.
- GPU VRAM: depends on model (Z-Image-Turbo) and batch size; expect 24+ GB for single-GPU full-size training with non-tiling. Use VAE tiling and smaller batch sizes to reduce GPU memory.

## Settings
- model.config:
  - `controlnet_streaming`: True or False (choose True for low RAM hosts).
  - `controlnet_offload_strategy`: 'none' | 'cpu' | 'memory_manager' (choose 'cpu' for manual low-RAM offload; 'memory_manager' if available and configured).
  - `control_use_tiling`: True to enable VAE tiling for control images (reduces peak GPU memory at the cost of extra compute).

- UI notes: In the `/jobs/new` Simple UI, you can set ControlNet via the **ControlNet (Name or Path)** field and use the **Enable ControlNet** toggle to explicitly enable/disable ControlNet for the job. When enabled, the UI reveals additional options: `ControlNet File` (optional filename), `Use streaming load`, `Offload Strategy`, and `Auto Output Shim`.

## Preflight checks & common failures
- "ControlNet file not found": Check `controlnet_file` path in job config and that the file was uploaded to the job folder. If using a repo, ensure the `controlnet_path` and `controlnet_file` are correct.
- "ControlNet checkpoint has unexpected keys": The checkpoint is incompatible with the control transformer. Use a matching ControlNet, or generate a `config.json` for the control repo that matches the transformer's config.
- If `safetensors.torch.load_file` fails with a MemoryError or the checkpoint size is large relative to available RAM, the loader will automatically fall back to streaming assignment and you will see a message like "falling back to streaming assignment" in the logs; training will continue using streamed tensor assignment.
- "failed to load base transformer config": If no `config.json` exists in the control repo and the base model reference is inaccessible, ensure base model files are available locally or configure `name_or_path` to a local folder.
- Offload failures when using `memory_manager`: check that `MemoryManager` is configured and available. Offload failures will raise RuntimeError and training will abort (fail-fast).
- Missing ZImage tokenizer / `apply_chat_template` error: If you see an error indicating a missing tokenizer for ZImage models, prefer adding a real tokenizer in the model repo or set `te_name_or_path`/`extras_name_or_path` to a repo that contains a compatible tokenizer. As a temporary/testing workaround you can set `ModelConfig.allow_dummy_tokenizer=True` or set environment variable `ZIMAGE_ALLOW_DUMMY_TOKENIZER=1` to create a minimal pass-through tokenizer (NOT recommended for generation).

## Recovery steps
1. Confirm the ControlNet file exists and is readable by the training job.
2. If streaming is enabled and keys are unexpected, verify the checkpoint keys using a small script with `safetensors.safe_open(...).keys()` and compare to the control transformer `state_dict().keys()`.
3. If base transformer config could not be loaded, put a `config.json` in the control repo with the transformer's config (a minimal `{}` may work for tests but production should reflect actual transformer config).
4. If offload failed, set `controlnet_offload_strategy` to `none` and retry, then investigate MemoryManager logs.

## Testing & CI
- Unit tests that exercise streaming and non-streaming loads are available in `testing/test_controlnet_streaming_fixture.py`.
- For CI, we use a small dynamically-created safetensors fixture in tests to avoid checking in large binary checkpoints.

---
