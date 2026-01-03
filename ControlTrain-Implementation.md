# ControlTrain Implementation Checklist ✅

Overview
- Purpose: Single checklist to track implementation of ControlTrain (Parts 1–3), preflight/fail-fast validations, tests, and smoke runs.
- Usage: Mark items completed as they are implemented and tested. Items prefixed with (FF) are fail-fast checks that must be true before training starts.

---

## Preflight / Fail-Fast (must pass before training starts) ⚠️
- [ ] (FF) `ModelConfig.controlnet_enabled` → at least one dataset has `control_type` set. (Place check in `hook_before_train_loop` and `BucketsMixin.setup_buckets`)
- [ ] (FF) If `controlnet_file` specified, the file exists and is loadable (safetensors or .pt). Fail with descriptive error if missing.
- [ ] (FF) If using pre-generated controls (`generate_control_on_the_fly=False`), `control_cache_path` exists and contains control files.
- [ ] (FF) If host RAM < 64GB, `controlnet_streaming=True` or training aborts. (Optionally detect via psutil)
- [ ] (FF) Required dependencies exist: `controlnet_aux` for control preprocessors, `safetensors` for safetensors streaming.
- [ ] (FF) After ControlNet load: unexpected keys == 0 for the checkpoint, and required attributes exist (`control_layers`, `control_all_x_embedder`, `control_in_dim`).

Acceptance: preflight raises a RuntimeError with a descriptive message on failure.

---

## Part 1 — UI & Dataloader (Dataset, Control Image Generation)
- [x] Update UI config types and forms to expose ControlNet options (file: `ui/src/app/jobs/new/jobConfig.ts`, `AdvancedJob.tsx`) — Completed in plan.
- [x] Add `ModelConfig` / `DatasetConfig` fields for controlnet and control preprocessing (file: `toolkit/config_modules.py`) — Completed in plan.
- [ ] Implement `ControlImageProcessor` with OpenPose/Canny/Depth support (`toolkit/control_processor.py`).
  - Acceptance: can generate control images for each supported type and batch process.
- [ ] Implement `BucketsMixin` and dataset changes so `FileItemDTO` supports `control_image_path` and `requires_control_generation` (`toolkit/dataloader_mixins.py`).
  - Acceptance: `__getitem__` returns `image` and `control_image` (full images), not tiled patches.
- [ ] Implement `encode_control_images(..., use_tiling: bool=False)` helper that encodes whole control images via VAE and optionally enables VAE tiling for memory efficiency. (Docs added to plans)
  - Acceptance: outputs latents with shape `[B, C, 1, H, W]`.
- [ ] Pre-generation script to create and cache control images: `scripts/generate_control_images.py`.
- [ ] Tests:
  - [ ] `testing/test_control_processor.py` — verifies control processors and outputs.
  - [ ] `testing/test_controlnet_dataset.py` — synchronized transforms, control generation modes, and `encode_control_images(..., use_tiling=True)` behaviour.

Notes:
- DO NOT tile control images into 16 image patches. Use VAE tiling for encoding if needed.

---

## Part 2 — Model Loading & ControlTransformer Integration
- [ ] Implement `ZImageControlNetConfigGenerator` (auto-gen config from base and safetensors keys) (`extensions_built_in/diffusion_models/z_image/controlnet_config.py`).
- [ ] Implement `load_controlnet_transformer()` with VideoX-Fun pattern:
  - [ ] Load base transformer from `base_model_path` (subfolder `transformer`).
  - [ ] Instantiate `ZImageControlTransformer2DModel(**config)`.
  - [ ] Copy base weights into control transformer: `control_transformer.load_state_dict(base_state, strict=False)`.
  - [ ] Load ControlNet checkpoint (prefer `load_file()` on high-RAM machines; offer `safe_open` streaming for low-RAM hosts).
  - [ ] Fail-fast if checkpoint contains unexpected keys (report sample keys) or required attrs missing.
  - Acceptance: missing/unexpected keys printed and unexpected==0 for compatible checkpoint; dry-run forward succeeds on small dummy data.
- [ ] Add `set_nested_parameter` helper to stream keys directly into model without building huge dicts (for streaming mode).
- [ ] Provide `load_controlnet_with_streaming(path, device='cuda')` helper for <64GB RAM hosts.
- [ ] Add offload manager design for controlnet: `toolkit/controlnet_offload.py` (strategies: none, cpu, sequential).
- [ ] Ensure pipeline loading uses `transformer` terminology (ZImageControlPipeline vs ZImagePipeline) and uses `transformer` field.
- [ ] Apply LoRA adapters to transformer blocks (not UNet); provide a targeted pattern to select correct modules for PEFT/LoRA.
- [ ] Tests:
  - [ ] `testing/test_controlnet_model.py` — config generation, base→control copy behavior, streaming loader unit tests, `controlnet` attributes validation, and dry-run forward.
  - [ ] Integration: `testing/test_controlnet_integration.py` — model loading using a small ControlNet safetensors and verifying memory profile.

---

## Part 3 — Training Loop & Validation
- [ ] Modify `BaseSDTrainProcess.setup_controlnet_training()` to apply fail-fast checks (control datasets exist, controlnet loaded, VAE and text encoder available).
- [ ] Update batch processing: `get_batch_from_dataloader()` returns `images`, `control_images` (full images), `captions`, etc., with shape checks.
- [ ] Training forward pass:
  - [ ] Encode control images using VAE (whole image) to `control_latents`; apply VAE shift/scaling if VAE config provides it.
  - [ ] Do NOT pass caption embeddings to ControlNet. Pass `cap_feats` (text embeddings) to transformer only.
  - [ ] Convert noisy latents to list form expected by transformer (list of tensors per example) or follow transformer patchify API.
  - [ ] Use `controlnet_offload_manager.control_forward()` context to minimize GPU residency of controlnet.
- [ ] Loss computation and logging: track `train/control_loss` metrics and `has_control` indicators.
- [ ] Sampling: add control sampling options in `SampleConfig` and ensure `ZImageControlPipeline` sampling is used when ControlNet is enabled.
- [ ] Add Connection Verification Test (VideoX-Fun pattern): load model + controlnet, assert unexpected keys == 0 and that required control transformer attrs exist; fail-fast if not.
- [ ] Tests:
  - [ ] `testing/test_training_forward.py` — forward pass with control latents and cap_feats to ensure shapes and outputs are sane.
  - [ ] `testing/test_controlnet_connection.py` — asserts unexpected keys == 0 and presence of required attributes.

---

## Cross-cutting & CI
- [ ] Add CI checks that run the new unit tests and the preflight checks (use small mocked safetensors or fixture files).
- [ ] Document expected resource requirements for training (RAM, GPU VRAM) and recommended settings (streaming vs load_file).
- [ ] Add runbook for failed preflight failures and recovery steps (missing control files, incompatible controlnet checkpoint, lacking controlnet_aux).

---

## Current Status & Next Steps
- Already completed:
  - Plan document updates across Parts 1/2/3 (reflecting VideoX-Fun behavior, VAE tiling, base→control copy, streaming guidance, fail-fast checks). ✅
- High-priority to implement now:
  1. Implement dataset preflight validation and hook into `hook_before_train_loop` (FAIL-FAST). (Next: implement code + tests.)
  2. Implement `load_controlnet_transformer()` base→control copy and streaming loader with dry-run validation.
  3. Implement `encode_control_images()` helper that toggles VAE tiling.

Pick next action for me: implement the preflight validators, the dataset helpers, or the model loader (base→control copy + streaming). I will start according to your priority and update this checklist as items are implemented.

---

Notes:
- This checklist follows a fail-fast philosophy: training must be prevented from starting if any critical precondition is unmet. All fail-fast checks should raise clear, actionable errors.
- If you want, I can create PR branches and begin implementing the first high-priority items and tests.
