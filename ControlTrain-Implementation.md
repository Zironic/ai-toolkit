# ControlTrain Implementation Checklist ✅

Overview
- Purpose: Single checklist to track implementation of ControlTrain (Parts 1–3), preflight/fail-fast validations, tests, and smoke runs.
- Usage: Mark items completed as they are implemented and tested. Items prefixed with (FF) are fail-fast checks that must be true before training starts.

---

## Preflight / Fail-Fast (must pass before training starts) ⚠️
- [X] (FF) `ModelConfig.controlnet_enabled` → at least one dataset has `control_type` set. (Place check in `hook_before_train_loop` and `BucketsMixin.setup_buckets`)
- [X] (FF) If `controlnet_file` specified, the file exists and is loadable (safetensors or .pt). Fail with descriptive error if missing.
- [X] (FF) If using pre-generated controls (`generate_control_on_the_fly=False`), `control_cache_path` exists and contains control files.
  - ✅ Fixed: dataset-level parsing now recognizes `control_precompute_control` passed in constructor kwargs and will create a default `control_cache_path` when appropriate (added unit test `test_constructor_precompute_flag_from_kwargs_creates_cache_dir`).
- [X] (FF) If host RAM < 64GB, `controlnet_streaming=True` or training aborts. (Optionally detect via psutil)
- [X] (FF) If loading a ControlNet checkpoint via `safetensors.torch.load_file` raises a `MemoryError`, or the checkpoint file size is large relative to available host RAM, the loader will automatically fall back to streaming assignment using `safetensors.safe_open` to avoid OOM. (Auto-fallback implemented.)
- [X] (FF) Required dependencies exist: `controlnet_aux` for control preprocessors, `safetensors` for safetensors streaming.
- [X] (FF) Fixed preflight validation: require at least one of `controlnet_name_or_path` or `controlnet_file` when `controlnet_enabled=True`. Previously the logic incorrectly required both and could abort valid configs; tests added (`test_preflight_allows_controlnet_name_or_path_only`, `test_preflight_allows_controlnet_file_only`).

Note: we've added the following optional control-related dependencies to `requirements.txt` so they can be installed easily when needed:
- `onnxruntime>=1.15.0` — required for some pose detectors and ONNX-based models
- `git+https://github.com/jaretburkett/easy_dwpose.git` — `easy_dwpose` (DWposeDetector) for pose control generation

[1mRecent finding:[0m If a dataset requests heavy-weight control types (e.g., `pose`, `openpose`, `depth`) and the optional packages (such as `controlnet-aux` or `easy_dwpose`) are not installed, control image generation will silently skip those controls and training will run without ControlNet conditioning (metrics show `train/batch_has_control` = 0). Added a fail-fast preflight check in `BaseSDTrainProcess.hook_before_train_loop` to detect this and raise an actionable RuntimeError (message suggests installing `controlnet-aux` or providing precomputed control images). A unit test `test_preflight_fails_for_pose_without_deps` was added to assert this behavior.
- [X] (FF) Updated ZImage tokenizer handling to match upstream: the loader now expects the tokenizer to be present in the model's `tokenizer/` subfolder and calls `AutoTokenizer.from_pretrained(..., subfolder="tokenizer", torch_dtype=...)`. If no compatible tokenizer is present model load will fail fast with a clear message instructing the user to provide a tokenizer or set `te_name_or_path`/`extras_name_or_path` to a repo that does. Removed the previous dummy-tokenizer fallback and the `load_model(for_training=True)` training-only skip to keep the loader behavior consistent and predictable. Updated tests to assert the tokenizer requirement and subfolder usage.trol_layers`, `control_all_x_embedder`, `control_in_dim`).3

Acceptance: preflight raises a RuntimeError with a descriptive message on failure.

---

## Part 1 — UI & Dataloader (Dataset, Control Image Generation)
- [x] Update UI config types and forms to expose ControlNet options (file: `ui/src/app/jobs/new/jobConfig.ts`, `SimpleJob.tsx`) — Completed in plan. Added explicit **Enable ControlNet** toggle and visible ControlNet options (file name/path, optional file, streaming toggle, offload strategy, auto output shim) and unit tests to assert defaults and migration behavior.
- [x] Add `ModelConfig` / `DatasetConfig` fields for controlnet and control preprocessing (file: `toolkit/config_modules.py`) — Completed in plan.
- [X] Implement `ControlImageProcessor` with OpenPose/Canny/Depth support (`toolkit/control_processor.py`).
  - Acceptance: can generate control images for each supported type and batch process (lightweight canny implemented; heavy types deferred to optional deps).
- [X] Implement `BucketsMixin` and dataset changes so `FileItemDTO` supports `control_image_path` and `requires_control_generation` (`toolkit/dataloader_mixins.py`).
  - Acceptance: `__getitem__` returns `image` and `control_image` (full images), not tiled patches.
- [X] Implement `encode_control_images(..., use_tiling: bool=False)` helper that encodes whole control images via VAE and optionally enables VAE tiling for memory efficiency. (Docs added to plans)
  - Acceptance: outputs latents with shape `[B, C, 1, H, W]`.
- [X] Pre-generation script to create and cache control images: `scripts/generate_control_images.py`.
- [X] Tests:
  - [X] `testing/test_control_processor.py` — verifies control processors and outputs.
  - [X] `testing/test_controlnet_dataset.py` — synchronized transforms, control generation modes, and `encode_control_images(..., use_tiling=True)` behaviour.

Notes:
- DO NOT tile control images into 16 image patches. Use VAE tiling for encoding if needed.

---

## Part 2 — Model Loading & ControlTransformer Integration
- [X] Implement `ZImageControlNetConfigGenerator` (auto-gen config from base and safetensors keys) (`extensions_built_in/diffusion_models/z_image/controlnet_config.py`).
- [X] Implement `load_controlnet_transformer()` with VideoX-Fun pattern:
  - [X] Load base transformer from `base_model_path` (subfolder `transformer`).
  - [X] Instantiate `ZImageControlTransformer2DModel(**config)`.
  - [X] Copy base weights into control transformer: `control_transformer.load_state_dict(base_state, strict=False)`.
  - [X] Load ControlNet checkpoint (prefer `load_file()` on high-RAM machines; offer `safe_open` streaming for low-RAM hosts).
  - [X] Fail-fast if checkpoint contains unexpected keys (report sample keys) or required attrs missing.
  - Acceptance: missing/unexpected keys printed and unexpected==0 for compatible checkpoint; dry-run forward succeeds on small dummy data.
- [X] Add `set_nested_parameter` helper to stream keys directly into model without building huge dicts (for streaming mode).
- [X] Provide `load_controlnet_with_streaming(path, device='cuda')` helper for <64GB RAM hosts.
- [X] Add offload manager design for controlnet: `toolkit/controlnet_offload.py` (strategies: none, cpu, sequential).
- [X] Ensure pipeline loading uses `transformer` terminology (ZImageControlPipeline vs ZImagePipeline) and uses `transformer` field.
- [X] Fix: create `self.pipeline` during `load_model()` when model components exist and optionally when loading a ControlNet; make `get_generation_pipeline()` tolerant of test monkeypatches (skip `.to()` when not present). Normalize dict-style `vae.config` for pipeline constructors while avoiding assignment to read-only `vae.config` properties by falling back to a small proxy wrapper; added unit test `testing/test_zimage_pipeline_creation.py` and `testing/test_zimage_vae_loading.py::test_zimage_handles_readonly_vae_config` to validate behavior.
- [X] Apply LoRA adapters to transformer blocks (not UNet); provide a targeted pattern to select correct modules for PEFT/LoRA.
- [ ] Tests:
  - [X] `testing/test_controlnet_model.py` — config generation, base→control copy behavior, streaming loader unit tests, `controlnet` attributes validation, and dry-run forward.
  - [ ] Integration: `testing/test_controlnet_integration.py` — model loading using a small ControlNet safetensors and verifying memory profile.

---

## Part 3 — Training Loop & Validation
- [X] Modify `BaseSDTrainProcess.setup_controlnet_training()` to apply fail-fast checks (control datasets exist, controlnet loaded, VAE and text encoder available). (Now supports UI `controls` list in `DatasetConfig`; added test `test_setup_controlnet_success_with_controls`.)
- [X] Fix: make dry-run forward robust to different `ControlNet.forward` signatures — we now try common positional and keyword patterns and provide a clear error if none succeed. Added `testing/test_controlnet_loading.py::test_setup_controlnet_training_handles_controlnet_cond_signature` to cover a `controlnet_cond` positional parameter case, and added `testing/test_controlnet_loading_extra.py::test_setup_controlnet_training_handles_tensor_sample_signature` to cover ControlNet implementations that expect a tensor `sample` (accessing `.shape`) rather than a list.
- [X] Fix: when the conv-candidate *list* fallback fails because the ControlNet implementation rejects a Python list for `controlnet_cond` (conv2d TypeError), `setup_controlnet_training` now raises a clear, actionable RuntimeError explaining that the model expects a single tensor and suggests converting dataset control outputs to a single tensor per sample. Added `testing/test_controlnet_list_conv_error.py` to cover this case.
- [X] Fix: resolved a CPU/CUDA device mismatch where `timesteps` remained on CPU while adapter parameters lived on CUDA. `SDTrainer` now moves `timesteps` to the adapter's device before calling the adapter, preventing ``RuntimeError: Expected all tensors to be on the same device``. Added unit test `testing/test_controlnet_timesteps_device_move.py` to assert the behavior.
- [X] Fix: adapter dtype authoritative—`SDTrainer` now detects the adapter's parameter dtype (e.g., `bfloat16`) and casts small inputs (timesteps, text embeddings, added_cond kwargs, control images) to match before invoking the adapter. Casting failure is now fail-fast: the trainer raises a `RuntimeError` with a clear diagnostic rather than silently continuing. Added unit tests `testing/test_controlnet_dtype_handling_strict.py` to assert both successful casting and that failures raise.
- [X] Fix: prioritize time-module dtype when casting timesteps. Some VideoX/ControlNet adapters have LoRA/adapter params in `bfloat16` while the time projection modules remain `float32`; casting timesteps to adapter param dtype caused ``RuntimeError: mat1 and mat2 must have the same dtype``. `SDTrainer` now inspects adapter modules for time-related parameter dtypes (e.g. `time_embedding`, `time_proj`) and uses that dtype for timesteps when present. Added unit test `testing/test_controlnet_time_dtype_priority.py` to validate this VideoX-specific behavior.
- [X] Fix: control input channel mismatch for VideoX-style adapters — `VideoXControlnetWrapper` now inspects the inner controlnet's expected conv input channels and will automatically drop an extra alpha channel (4->3) or pad/drop as needed when the dataloader produced an alpha-padded control image. This resolves the runtime Conv2d channel mismatch error ("expected input ... to have 3 channels, but got 4 channels instead"). Added tests: `testing/test_controlnet_compat.py` to assert 4->3 trimming for tensors and lists, and `testing/test_controlnet_compat_preserve_4ch.py` to assert behavior when inner expects 4 channels.

- [X] Fix: when control images are encoded to latents, ensure the resulting control latents (`control_context`) are spatially resized to match the noisy latents used for the ControlNet call. This prevents runtime mismatches inside ControlNet forward (e.g., 60x60 vs 64x64) that caused signature/shape errors with the Alibaba Z-Image Turbo Fun ControlNet. Added unit tests: `testing/test_zimage_control_spatial_adaptation.py::test_predict_noise_zimage_resizes_control_latents_to_latents` and improved existing spatial adaptation tests. Added a fail-fast check that raises a clear error if a post-resize mismatch still persists to avoid silent mis-training.
- [X] Fix: zimage routing channel inference sometimes picked an incorrect (e.g., 1280) `expected_in_ch` due to a frequency-based heuristic over all conv modules. For VideoX-style adapters this resulted in padding noisy latents to 1280 channels and later Conv2d mismatches when the primary `conv_in` expected 4 channels.
  - Implemented `infer_expected_in_ch(adapter)` which prefers `adapter.control_in_dim` or, when present, the dedicated `adapter.conv_in.weight.shape[1]`, and resolves conflicting conv in-channels (e.g., both 3 and 4 present) by preferring 3 to avoid alpha-channel confusion. Added a Z-Image special-case that only forces 4 when no 3-channel convs are detected. Unit tests added to validate behavior.
  - Updated zimage routing to use this helper and added unit tests `testing/test_zimage_channel_inference.py` covering `conv_in` preference, `control_in_dim` attribute presence, grouped-mean reductions (1280->4), and an integration-style check using `VideoXControlnetWrapper`.
  - Acceptance: CPU unit tests pass locally and the earlier "expected input ... to have 4 channels, but got 1280" runtime error is resolved for alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1.
- [X] Fix: resolved an indentation bug in `BaseSDTrainProcess` that prevented the `VideoXControlnetWrapper` from being applied when the `print_acc` call succeeded; added `testing/test_base_process_controlnet_wrapper.py` to assert the wrapper is always applied when `adapter_config.controlnet_mode == 'zimage'.`
- [X] Fix: honor dataset-level `controlnet_mode` flags (e.g., `controlnet_mode='zimage'`) when deciding to apply the VideoX wrapper, and make wrapper application fail-fast (raise `RuntimeError`) if wrapping cannot be performed; added `testing/test_base_process_controlnet_wrapper_dataset.py` to assert detection and fail-fast behavior.
- [X] Fix: ensure `_predict_noise_zimage` also respects adapter device/dtype hints for Z-Image (VideoX) routing; latents, timesteps and control_context are now moved/cast to the adapter dtype/device before calling the controlnet to avoid Float/BFloat16 mismatches (test: `testing/test_zimage_control_dtype.py`).

- [X] Improvement: single-tensor fallback—`setup_controlnet_training()` now attempts single-tensor adaptations for each unique conv `in_channels` (pad/drop/reshape as needed) before trying the per-conv list fallback. This resolves common channel mismatch failures (e.g., model expects 4ch but dataset provides 3ch) and provides clearer diagnostics in logs; added `testing/test_controlnet_single_tensor_fallback.py` to validate behavior.
- [X] Make `SDTrainer.hook_before_train_loop()` more defensive: guard access to `sd.encode_control_in_text_embeddings`, `sd.encode_prompt`, `sd.vae`, and `sd.noise_scheduler` so unit tests and minimal SD mocks don't raise AttributeError. This prevents spurious failures in test fixtures and when model components are initialized lazily.
- [X] Update batch processing: `get_batch_from_dataloader()` returns `images`, `control_images` (full images), `captions`, etc., with shape checks.
- [ ] Training forward pass:
  - [x] Encode control images using VAE (whole image) to `control_latents`; apply VAE shift/scaling if VAE config provides it. ✅
  - [X] Do NOT pass caption embeddings to ControlNet. Pass `cap_feats` (text embeddings) to transformer only. (Added validation to detect and fail-fast when non-image tensors are used as control images.)
  - Note: Implemented `StableDiffusion._predict_noise_zimage(...)` and wired it into `predict_noise`; added unit and integration tests to assert control routing and tiling behavior.
  - [X] Convert noisy latents to list form expected by transformer (list of tensors per example) or follow transformer patchify API.
  - [X] Use `controlnet_offload_manager.control_forward()` context to minimize GPU residency of controlnet.
- [ ] Loss computation and logging: track `train/control_loss` metrics and `has_control` indicators.
- [X] Fix: when using VideoX/zimage-style adapters we were not detecting control conditioning for metrics because those adapters set `zimage_control_images` instead of per-block residuals. Added detection for `zimage_control_images` in `SDTrainer` so `train/batch_has_control` and `train/control_usage_rate` correctly reflect control usage; added `test_control_metrics_present_for_zimage_adapter` to assert this behavior.
- [X] Fix: SDTrainer now recognizes instances of `VideoXControlnetWrapper` (wrapped ControlNet adapters) when preparing control routing, ensuring `pred_kwargs` contains `zimage_controlnet` and `zimage_control_images`. Added `testing/test_trainer_zimage_routing.py` to assert explicit and name-based detection and wrapper compatibility.
- [X] Sampling: add control sampling options in `SampleConfig` and ensure `ZImageControlPipeline` sampling is used when ControlNet is enabled. ✅ **Implemented `control_conditioning_scale` and `control_images` in `SampleConfig` and routing in `ZImageModel.generate_single_image()`; added unit + integration tests.**
- [X] Add Connection Verification Test (VideoX-Fun pattern): load model + controlnet, assert unexpected keys == 0 and that required control transformer attrs exist; fail-fast if not.
- [ ] Tests:
  - [X] `testing/test_training_forward.py` — forward pass with control latents and cap_feats to ensure shapes and outputs are sane.
  - [ ] `testing/test_controlnet_connection.py` — asserts unexpected keys == 0 and presence of required attributes.

---

## Cross-cutting & CI
- [x] Add CI checks that run the new unit tests and the preflight checks (added `testing/test_controlnet_streaming_fixture.py` which uses a small safetensors fixture created at test time). ✅
- [X] Fix: progress bar formatting in `BaseSDTrainProcess` to handle dicts and tensors safely; added unit test `testing/test_progress_bar_formatting.py`.
- [X] Fix: resolved NameError in `ZImageModel.load_model()` when loading ControlNet due to a missing `get_torch_dtype` import; added unit test `testing/test_controlnet_get_torch_dtype.py` to assert correct dtype mapping and that ControlNet parameters are frozen on load.- [X] Fix: resolved NameError in `toolkit.control_channels.adapt_control_images` (incorrect helper name `infer_expected_in_ch_from_adapter`); replaced with `infer_expected_in_ch(adapter)` and added explicit fail-fast error messages when inference fails; added unit tests `testing/test_control_channels.py::test_zimage_special_case_padding` and `testing/test_control_util.py::test_infer_expected_in_ch_special_case_zimage` to assert the Z-Image special-case behavior.- [X] Fix: prevent AttributeError during direct ControlNet loading where `ZImageModel.load_model()` attempted to access `self.train_config` (not present on model). The loader now uses `self.torch_dtype` (model's configured dtype) when calling `ControlNetModel.from_pretrained(...)`. Added unit test `testing/test_controlnet_loading.py::test_controlnet_from_pretrained_uses_model_dtype` to prevent regressions.
- [x] Document expected resource requirements for training (RAM, GPU VRAM) and recommended settings (streaming vs load_file) via `docs/runbooks/controlnet_runbook.md`. ✅
- [x] Add runbook for failed preflight failures and recovery steps (missing control files, incompatible controlnet checkpoint, lacking controlnet_aux). ✅

---

## Current Status & Next Steps
- Already completed:
  - Plan document updates across Parts 1/2/3 (reflecting VideoX-Fun behavior, VAE tiling, base→control copy, streaming guidance, fail-fast checks). ✅
  - Dataset preflight validators implemented and tested (`toolkit/dataloader_mixins.py`, `testing/test_control_preflight.py`). ✅
  - ControlNet loader implemented with base→control copy and streaming-safe assignment (`extensions_built_in/diffusion_models/z_image/z_image.py::load_controlnet_transformer`). ✅
  - VAE-tiling helpers and tiled-encode/reassembly implemented and tested (`toolkit/dataloader_mixins.py`, `encode_control_images`, `reassemble_tile_latents`, `testing/test_tiling_reassembly.py`). ✅
  - Implemented Z-Image routing helper `StableDiffusion._predict_noise_zimage(...)`, integrated it into `predict_noise` and `SDTrainer`, and added deterministic unit/integration tests (`testing/test_zimage_control_routing_unit.py`, `testing/test_zimage_control_predict_noise_integration.py`). All Python unit tests pass locally (UI E2E requires UI server). ✅
  - Fixed: resolved `AttributeError: 'ZImageModel' object has no attribute 'model'` by assigning the loaded transformer to `self.model`/`self.transformer` in `ZImageModel.load_model()`; this prevented training/quantization failures during model load. Added unit tests to cover LoRA behavior and a minimal CPU smoke test (`testing/test_lora_integration.py`, `testing/test_lora_smoke_cpu.py`) to assert LoRA parameters are trainable while ControlNet parameters remain frozen. Acceptance: new tests pass locally. ✅
  - Fixed: ControlNet loading could OOM when using `safetensors.torch.load_file` on low-RAM hosts. Implemented automatic fallback to streaming assignment (using `safetensors.safe_open` with `set_nested_parameter`) when `load_file` raises `MemoryError` or when the checkpoint file is large relative to available RAM; added `testing/test_controlnet_model.py::test_load_controlnet_auto_streams_on_memoryerror` to cover this behavior.
- High-priority next steps:
  1. Implement offload manager for ControlNet (strategies: none | cpu | sequential) and integrate with MemoryManager. (Design + tests.) ✅ **Implemented: memory-manager attach/bring and context manager added + tests**
  2. Integrate tiled-encode + control-latents into training forward pass and add forward-shape tests (`testing/test_training_forward.py`, `testing/test_training_forward_tiled.py`). (Use mocks; avoid full E2E.) ✅ **Implemented: training supports list-form control latents and nested per-sample tile lists; tests added**
  3. Add CI coverage and resource guidance / runbook entries; include mocked safetensors fixtures for CI.
  4. Add more integration tests for streaming vs non-streaming behavior and ensure fail-fast messages are user-friendly.
- Acceptance criteria: All high-priority items have unit tests and documented runbook steps.


Quick reproduce & test notes:
- To run the tokenizer fail-fast test locally (fast):
  - `python -m pytest testing/test_zimage_tokenizer_check.py -q` ✅
- To validate ControlNet freeze behavior locally (fast):
  - `python -m pytest testing/test_controlnet_freeze.py -q` ✅
- If you encounter: `AttributeError: 'NoneType' object has no attribute 'apply_chat_template'` the model checkpoint lacks a tokenizer compatible with ZImage; either add the tokenizer in the model repo or set `te_name_or_path` or `extras_name_or_path` in `ModelConfig` to an HF repo that contains a compatible tokenizer.

---

Notes:
- This checklist follows a fail-fast philosophy: training must be prevented from starting if any critical precondition is unmet. All fail-fast checks should raise clear, actionable errors.
- If you want, I can create PR branches and begin implementing the first high-priority items and tests.
