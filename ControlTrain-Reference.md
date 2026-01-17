# ControlTrain Reference (VideoX-Fun findings & how we mirrored them)

Summary of key implementation patterns discovered in VideoX-Fun and how we applied them here:

- Base→Control copy
  - VideoX-Fun instantiates a Control transformer and performs a best-effort non-strict copy of base transformer weights into it (e.g., `control.load_state_dict(base_state, strict=False)`) before applying checkpoint-specific weights.
  - We implemented `load_controlnet_transformer()` which:
    - Instantiates a control transformer (falls back to base transformer if control class not available),
    - Copies base transformer weights via `load_state_dict(..., strict=False)`,
    - Loads control checkpoint (streaming with `safetensors.safe_open` or `safetensors.torch.load_file`) and assigns tensors.
  - Tests: `testing/test_controlnet_base_copy.py` verifies base→control copy behavior.

- Safetensors streaming vs load_file
  - VideoX-Fun uses `safetensors.torch.load_file` for full-file loads and `safe_open` for streaming reads.
  - We implemented both and added streaming-safe tensor assignment (`set_nested_parameter`) to avoid building large dicts in RAM.
  - We also implemented an **auto-fallback**: when `load_file` raises a `MemoryError` or when the checkpoint file appears large relative to available host RAM, the loader will fall back to streaming assignment automatically to avoid OOMs on low-RAM hosts.
  - Tests: `testing/test_controlnet_unexpected_keys.py` ensures unexpected keys raise clear errors in both streaming and non-streaming modes.

- VAE tiling for large control images
  - VideoX-Fun uses tiled VAE encodes (batch or streamed ops) instead of splitting images into separate control-image patches.
  - We added `tile_image()` and `verify_tile_alignment()` helpers, `encode_control_images(..., tile=True)` and `reassemble_tile_latents()`.
  - Tests: `testing/test_controlnet_dataset.py`, `testing/test_tiling_reassembly.py`, `testing/test_reassembly_downsample.py`.

- ControlNet expectations & fail-fast
  - ControlNet checkpoints should not introduce unexpected keys; fail-fast on incompatibilities with clear messages.
  - `controlnet_aux` is required for `openpose` preprocessing; we added dataset-level preflight checks and fail-fast behavior in `BucketsMixin.setup_buckets()`.
  - Recent finding: datasets that request heavy-weight control types like `pose`/`openpose`/`depth` will silently skip control generation if the optional dependencies are missing (e.g., `controlnet-aux` or `easy_dwpose`) — causing ControlNet to be loaded but never receive control images (training runs without control conditioning). We added a proactive preflight check in `BaseSDTrainProcess.hook_before_train_loop` that detects this situation and raises an actionable error instructing the user to install `controlnet-aux` or provide precomputed control images.
  - Fix: normalize dataset `controls` values (strip whitespace and lower-case) in `DatasetConfig` so UI-provided values (e.g., `" Pose "`) are recognized correctly by preflight checks and control generators. This prevents silent misses where capitalization/whitespace caused heavy-control detection to be skipped and training to run without control conditioning.
  - New: dataset-level `controlnet_mode` (e.g., `controlnet_mode='zimage'`) is now honored when deciding to apply the VideoX wrapper. If a dataset requests zimage routing, the trainer will require the `VideoXControlnetWrapper` to be applied and will raise a `RuntimeError` if wrapping cannot be performed, rather than silently continuing.
  - Fix: `BaseSDTrainProcess.setup_controlnet_training()` previously only looked for legacy `control_type` and would raise when datasets used the UI's `controls` list. The method now recognizes the `controls` list as well and the error message was made clearer; added unit test `test_setup_controlnet_success_with_controls`.

  - Tests: `testing/test_control_preflight.py` (dataset preflight checks), `testing/test_controlnet_unexpected_keys.py` (unexpected key checks).

- Training integration notes
  - Do not pass text embeddings directly into ControlNet; use `cap_feats` / transformer inputs as per VideoX-Fun.
  - Implemented helper: `StableDiffusion.get_noise_prediction(...)` which tiles control images to match cap_feats, prepares `control_context`, calls the ControlNet and ensures cap_feats are passed only to the transformer. This helper is integrated into `predict_noise` and SDTrainer.
  - Encode control images via VAE (optionally tiled when memory-limited) and provide `control_latents` to the training forward pass.
  - Fix: moved `timesteps` to the adapter device before calling the ControlNet to avoid CPU/CUDA device mismatch errors during training (`RuntimeError: Expected all tensors to be on the same device`). See `testing/test_controlnet_timesteps_device_move.py` for the unit test.
  - Fix: adapter dtype authoritative. The trainer now detects the adapter parameter dtype (e.g., `bfloat16`) and casts small inputs (timesteps, text embeddings, control images) to match before calling the adapter. Casting errors now raise a `RuntimeError` so failures are visible and actionable (tests: `testing/test_controlnet_dtype_handling_strict.py`).- Fix: discovered a failure when an adapter had LoRA/adapter params in `bfloat16` while the controlnet's timestep/time-projection modules remained `float32` (error: `mat1 and mat2 must have the same dtype`). We now prefer the dtype of time-related modules (e.g., `time_embedding`, `time_proj`) for timesteps when present, avoiding the Float/BFloat16 matmul mismatch (test: `testing/test_controlnet_time_dtype_priority.py`).
Files & locations to inspect for behavior:
- Loader & helpers: `extensions_built_in/diffusion_models/z_image/z_image.py`
- Dataloader helpers: `toolkit/dataloader_mixins.py`
- Tests: `testing/test_controlnet_*.py`, `testing/test_tiling_*.py`

Implementation status: base→control copy, streaming loader, tiling helpers, and preflight validations implemented and tested. CI fixture-based tests (`testing/test_controlnet_streaming_fixture.py`) and a brief runbook (`docs/runbooks/controlnet_runbook.md`) have been added. Next: offload manager and final PR readiness checks.

Fixes & recent work:
- Fixed an AttributeError in `ZImageModel.load_model()` by assigning the loaded transformer to `self.model`/`self.transformer` so `BaseModel` properties (`unet`/`transformer`) work correctly during training and quantization.
- Fixed a `NameError` raised when loading ControlNet due to a missing import for `get_torch_dtype`; added `testing/test_controlnet_get_torch_dtype.py` to assert correct dtype mapping and that ControlNet params are frozen by default.
- Fixed a `NameError` in `toolkit.control_channels.adapt_control_images` (misnamed helper call); replaced with `infer_expected_in_ch(adapter)` and added unit tests `testing/test_control_channels.py::test_zimage_special_case_padding` and `testing/test_control_util.py::test_infer_expected_in_ch_special_case_zimage` to validate behavior.
- Fixed: `AttributeError: 'ZImageModel' object has no attribute 'pipeline'` by ensuring `load_model()` creates `self.pipeline` when model components exist and making `get_generation_pipeline()` robust (skip `.to()` if missing and normalize dict-style `vae.config` for pipeline constructors). Added unit test `testing/test_zimage_pipeline_creation.py` to verify `pipeline` assignment and prompt encoding.
- Fixed: avoid assigning to read-only `AutoencoderKL.config` by falling back to a small proxy wrapper that exposes a `config` SimpleNamespace while delegating other attributes to the original VAE instance; added unit test `testing/test_zimage_vae_loading.py::test_zimage_handles_readonly_vae_config` to cover this case.
- Fixed: crash due to missing tokenizer (`AttributeError: 'NoneType' object has no attribute 'apply_chat_template'`). To match upstream behavior, we now explicitly load the tokenizer from the model `tokenizer/` subfolder via `AutoTokenizer.from_pretrained(..., subfolder="tokenizer", torch_dtype=...)` and fail fast when a compatible tokenizer is not present. We removed the previous dummy-tokenizer fallback and the `load_model(for_training=True)` training-only skip so the loader behaviour is simpler and more predictable (ensure your model repo contains a compatible tokenizer or set `te_name_or_path`/`extras_name_or_path` to a repo that does). Updated tests to reflect this behavior: `testing/test_zimage_tokenizer_check.py` and `testing/test_zimage_tokenizer_subpath_fallback.py` verify the tokenizer requirement and subfolder load call.
- Fixed: when loading ControlNet via `ControlNetModel.from_pretrained(...)` the loader previously attempted to use `self.train_config` to determine dtype which caused an `AttributeError` (`'ZImageModel' object has no attribute 'train_config'`). We now pass `self.torch_dtype` (model's resolved torch dtype) to `from_pretrained(...)` and added `testing/test_controlnet_loading.py::test_controlnet_from_pretrained_uses_model_dtype` to prevent regressions.
- Fixed: `BaseSDTrainProcess.setup_controlnet_training()` now handles multiple `ControlNet.forward` signatures (including ones that require a positional `controlnet_cond` arg) by trying several common call patterns; added `testing/test_controlnet_loading.py::test_setup_controlnet_training_handles_controlnet_cond_signature` to document and prevent regressions.
- Fixed: added a clearer runtime error and test for the case where the conv-candidate list fallback fails because the ControlNet rejects a Python list for `controlnet_cond` (TypeError from `conv2d`). The new message explains the model expects a single Tensor and suggests converting dataset control outputs to a single tensor per sample. See `testing/test_controlnet_list_conv_error.py`.

  - New: Single-tensor fallback. If a channel-mismatch is detected during dry-run (conv expects 4ch but dataset provides 3ch, etc.), `setup_controlnet_training()` will attempt to adapt the control tensor to each unique expected `in_channels` and try a single-tensor call before constructing a list-style control list. This resolves common mismatches and yields clearer diagnostics; added unit test `testing/test_controlnet_single_tensor_fallback.py`.

- Fixed dataset-level parsing: `control_precompute_control` passed via constructor kwargs is now recognized and will create a default `control_cache_path` when precompute is enabled; improved the error message and added unit test `test_constructor_precompute_flag_from_kwargs_creates_cache_dir`.
- Added unit tests verifying LoRA adapter behavior and training updates (`testing/test_lora_integration.py`) and a minimal CPU smoke test (`testing/test_lora_smoke_cpu.py`).
- Fixed: TypeError in progress bar formatting when loss metrics contain dicts or tensors; added `_format_progress_bar` in `BaseSDTrainProcess` and `testing/test_progress_bar_formatting.py` to validate nested dicts, tensors, and scalars.
- Notes: LoRA modules may initialize `lora_up` to zeros; tests ensure `lora_up` is non-zero for meaningful gradients during synthetic single-step checks.

- Fixed: preflight check for ControlNet required both `controlnet_name_or_path` AND `controlnet_file` which caused training aborts when only a repo/folder path was provided. The check now requires at least one of these to be set; added tests `test_preflight_allows_controlnet_name_or_path_only` and `test_preflight_allows_controlnet_file_only`.

- UI: Reintroduced an explicit **Enable ControlNet** toggle (previously only implicitly set when a ControlNet path was entered) and exposed key ControlNet model options in the `/jobs/new` Simple UI (`ui/src/app/jobs/new/SimpleJob.tsx` and `jobConfig.ts`). These changes ensure users can explicitly enable/disable ControlNet and configure `controlnet_file`, `controlnet_streaming`, and `controlnet_offload_strategy` before submitting jobs. When ControlNet is enabled, dataset **Controls** selection and Control Image fields are now visible in the dataset editor. Added a small UI unit test asserting defaults and migration behavior.

If you discover anything in VideoX-Fun not captured here, append it to this file so future contributors find it quickly.

Recent fix: `BaseSDTrainProcess.setup_controlnet_training()` was updated to be robust against both list- and tensor-style inputs for `sample` and `controlnet_cond`. This prevents a runtime failure when a ControlNet implementation expects a tensor `sample` and accesses `.shape`. See new test: `testing/test_controlnet_loading_extra.py::test_setup_controlnet_training_handles_tensor_sample_signature`.

- Observation from VideoX-Fun: control inputs are passed as stacked torch tensors (e.g., `control_latents = _batch_encode_vae(control_pixel_values).unsqueeze(2)` in `scripts/z_image_fun/train_control.py`), and pipelines accept `control_image` / `control_video` as `torch.FloatTensor` in signatures (`pipeline_z_image_control.py`). To mirror that behavior and remain compatible with both list- and tensor-style ControlNet implementations, we now attempt tensor-style calls first and fall back to stacking list inputs into tensors (added in `BaseSDTrainProcess.setup_controlnet_training`).

- New: explicit VideoX/Z-Image routing mode: set `adapter.controlnet_mode = 'zimage'` in your adapter config to force VideoX-style routing. The trainer will pass `zimage_controlnet` and `zimage_control_images` into `sd.predict_noise` (the SD model will call the z-image routing helper). This avoids heuristic dry-run adaptations and is intended for deterministic VideoX compatibility.

- Auto-detect by model name: if `adapter_config.name_or_path` or the loaded adapter's `name_or_path` contains substrings like `zimage`, `z_image`, `videox`, or `pipeline_z_image`, the trainer will automatically enable zimage routing. This allows using VideoX-style ControlNets without adding extra config flags.

- Fix: metrics & monitoring now recognize zimage routing as control conditioning. Previously `train/batch_has_control` could be 0 when zimage routing was used because we only checked for per-block residuals; `SDTrainer` now also treats `zimage_control_images` as evidence of control conditioning and unit tests verify the behavior.

- New: Dry-run zero-mask padding. When the ControlNet conv expects exactly one more channel than the provided control image (a common scenario where the model expects `control + mask`), the dry-run will automatically pad a zero mask channel and log the adaptation (`"adding zero mask channel"`). This helps tests and smoke runs proceed safely while still surfacing real data mismatches in normal runs; a strict preflight option can be enabled to fail instead of padding.

- New: Data loader alpha padding. During control image loading, if the model (via `sd.controlnet`) advertises it expects 4-channel inputs, the loader will automatically pad a zero alpha/mask channel for 3-channel control images and log the change (`"[CONTROL DATALOADER] Padded zero alpha channel"`). This follows VideoX's deterministic model-hint based approach and prevents flip-flop errors at dry-run time.
- New: VideoX wrapper channel trimming. For certain VideoX/Z-Image ControlNet checkpoints that actually expect 3-channel inputs, the `VideoXControlnetWrapper` will auto-detect the inner conv's expected channels and drop an extra alpha channel (4->3) when present. This prevents runtime Conv2d channel mismatch errors such as "expected input ... to have 3 channels, but got 4 channels instead" while preserving datasets that provide genuine 4-channel controls.
- Fix: SDTrainer now accepts `VideoXControlnetWrapper` instances when preparing zimage routing and will pass `zimage_controlnet`/`zimage_control_images` into `predict_noise`, ensuring the transformer receives control images for VideoX-style ControlNets.
- Fix: resolved indentation bug in `BaseSDTrainProcess` that caused the wrapper to be skipped when `print_acc` completed successfully; added `testing/test_base_process_controlnet_wrapper.py` to assert wrapper application.

- New: Targeted fix for alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 — the trainer now infers the adapter's expected input channels more robustly by preferring `adapter.control_in_dim` or `adapter.conv_in.weight.shape[1]` when present, and resolving 3/4 conv conflicts by preferring 3 when both are present (avoids unwanted alpha-channel forcing). Additionally, the `VideoXControlnetWrapper` now dynamically detects Conv channel-mismatch runtime errors, adapts the `control_context` channels (trim/pad) and retries once before surfacing a clear error. Unit tests added to validate these behaviors.
  - Quick test: run `python -m pytest testing/test_zimage_channel_inference.py -q` to verify the inference and grouped-mean reduction behavior (1280->4).
  - Smoke check: run a minimal CPU job with `controlnet_model` set to the local `ZIT-Controlnet-Union-2.1-8steps` folder and `device='cpu'`, `steps=1`, `batch_size=1` to ensure the adapter is called successfully (the job should not raise the previous channel-mismatch error).

