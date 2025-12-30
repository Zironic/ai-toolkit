


# ControlNet Training Plan (OpenPose → ControlNet)

## Summary ✅
Add an optional ControlNet-based training mode that uses OpenPose pose maps derived from the dataset as control inputs. The implementation should be **on-the-fly-first** (generate pose maps after bucket/resize and augmentations by default), with optional caching for very large datasets or for offline diagnostics. ControlNet weights should be loadable/savable as part of the existing adapter/save flow.

---

## Goals 🎯
- Produce `control_tensor` per training sample (OpenPose pose maps / keypoint heatmaps → tensor format expected by ControlNet).
- Provide config toggles for generation mode, thresholds, caching, and whether to train the ControlNet adapter.
- Wire control tensors through the dataloader → `BaseSDTrainProcess.process_general_training_batch` → model training loop (ensure gradients flow into controlnet adapter if training).
- Add tests, docs, and example config.

---

## Design & Implementation Plan (step-by-step) 🔧

1) Design & API (analysis) — Files affected: `jobs/process/BaseSDTrainProcess.py`, dataloader, `DataLoaderBatchDTO`, config schemas
   - Decide on default control type: `openpose` (extendable to other control types later).
   - Config keys (suggested):
     - `controlnet:`
       - `use_controlnet: bool` (master switch)
       - `type: 'openpose'` (other values later) 
       - `name_or_path` (pretrained controlnet model if any)
   - Update `config/examples/` with a short example.


2) Dataloader & Batch DTO changes (on-the-fly-first) 
   - Add fields to `DataLoaderBatchDTO`: `control_tensor` (torch.Tensor or None) and `control_image_path` (for caching/traceability).
  - Provide a control generation CLI `tools/gen_control.py` to generate control maps and optionally cache them to `datasets/<dataset_name>/pose/` (filenames matching source images, e.g., `image_0001.jpg` -> `image_0001_pose.png`). The CLI should:
    - Accept config for thresholds/blur/size and option to cache or run one-off generation.
    - Optionally overwrite or skip existing files when caching is enabled.
    - Produce a lightweight cache manifest or add entries to the dataset manifest mapping source -> control path when caching is used.
   - Dataloader behavior and alignment (mitigation):
    - **Pixel‑perfect generation point (REQUIRED):** Generate control images *after* bucket/resize and all geometric augmentations, but *before* VAE encoding. Generating at this point guarantees the control image uses the exact pixels the model will see and prevents subtle alignment or interpolation mismatches.
    - **Single canonical helper:** Implement and require usage of `toolkit/pose.py::make_openpose_map(image, model='openpose', confidence=0.3, output_format='heatmap', size=None, channels=None, dtype=None, generator_version=None)`. Both `tools/gen_control.py` and any on-the-fly dataloader generation MUST call this helper so generation parameters and code paths are identical.
    - **Determinism & manifest metadata:** Control generation must be deterministic (fixed interpolation/antialiasing, no stochastic steps). Record generation parameters and a `generator_version` (git-sha or release tag) in a per-dataset `control_manifest.json` with fields such as `manifest_version` (semantic, start at `1`), `generator_version`, `thresholds`, `blur`, `size`, `channels`, `dtype`, `timestamp`, `params_hash` and a per-file checksum (`sha256`) for each control image. Manifests must conform to an explicit JSON Schema (add `toolkit/control_manifest_schema_v1.json`), be validated before use, and written atomically (temp file → `os.replace`) to avoid partial state. The manifest schema includes:
  - `manifest_version` (int)
  - `generator_version` (string)
  - `params_hash` (string)
  - `generator_params` (object)
  - `files` (dict mapping `source_image` -> `{ "control_path": "<relpath>", "sha256": "...", "size": [W,H] }`)
  - `generated_after_augmentations` (bool)
Add a manifest validator helper (e.g., `toolkit/control_manifest.py::validate_manifest(path)`) and unit tests `testing/control_manifest_test.py` to assert schema compliance, atomic write semantics, and that `params_hash` mismatches trigger a clear error.
    - **On-the-fly-first policy (trade-offs):**
      - `generate_on_the_fly=True` → **generate after bucket/resize** and augmentations (canonical, alignment-safe mode).
      - **Caching policy & augmentation metadata:** when caching is enabled the manifest MUST include `generated_after_augmentations: bool`. If `generated_after_augmentations == true`, cached controls are treated as ready for training with geometric augments enabled. If `generated_after_augmentations == false`, the dataset onboarding must either (A) reject geometric augmentations during training for that dataset, or (B) provide deterministic augmentation-replay metadata in the manifest (e.g., per-file augmentation seeds or transform specs) that the dataloader replays to keep control and image aligned. The manifest validator should enforce this contract and the precompute CLI should optionally produce augmentation-replay metadata when requested.
      - Optional caching: for very large datasets or offline analysis, support optional caching of generated controls to disk; if caching is used, the dataset onboarding step MUST verify bit-equivalence between cached controls and canonical on-the-fly-after-bucket generation for a representative sample set (use `params_hash` + per-file two-way checksum tests).
      - Operational recommendation: prefer on-the-fly generation for correctness and caching only as a performance optimization when necessary; when caching is used prefer storing `generated_after_augmentations=true` variants (cache augmented variants in chunked form) to simplify training semantics.
    - **Helper signature & placement:** The canonical helper should return a deterministic PIL/ndarray or tensor and be called after bucket-resize and any spatial augmentations but before conversion to VAE latents/dtype/device. Ensure control channel count and dtype match the adapter's `get_expected_control_spec()`.
    - **Verification checks (design-level):** Add a verification step during dataset onboarding that performs a bit-identity check (cached control == on-the-fly-after-bucket generation when `generator_version` and params match) and a latency check to inform whether caching is recommended.
    - **UI/CLI integration:** Allow dataset creation UI to trigger the precompute job and show progress; the precompute tool must be idempotent (safe to re-run) and update the manifest atomically to avoid inconsistent state.

  - ControlNet checkpoint compatibility & conversion (safetensors, channels, resolution):
    - **Problem statement:** Some ControlNet checkpoints are distributed as `safetensors` or with unexpected channel counts (e.g., 4 vs 16) or at a different spatial training width (e.g., 1280 vs pipeline 320). These mismatches must be handled robustly and reproducibly.
    - **Compatibility helpers (new module):** Add `toolkit/controlnet_compat.py` with the following helpers:
      - `load_controlnet_checkpoint(path)` — robust loader that supports `safetensors` and diffusers checkpoints and returns a toolkit-compatible `ControlNetModel` or thin wrapper.
      - `convert_safetensors_to_diffusers(in_path, out_path)` — a conversion/validation helper for maintainers.
      - `resample_control_image(control_image, expected_size)` — deterministic resampling with antialiasing; it should be applied after bucket/resize.
      - `normalize_control_channels(control_tensor, expected_channels)` — channel adapter (1×1 conv / small linear projector) to map incoming channels to the model-expected channels. Projector weights must be savable as part of adapter/checkpoint metadata.
      - Persist conversion metadata in checkpoint/manifest: `{ original_format, channel_map: '1x1conv', resampled_to, converter_version }`.
    - **Recommended default workflow:**
      1. `load_controlnet_checkpoint(path)` (handles safetensors/diffusers)
      2. After bucket/resize: `resample_control_image(...)` to the pipeline size
      3. If channel mismatch: `normalize_control_channels(...)` (persist projector when training/finetuning)
      4. Feed into ControlNet adapter
    - **Design decision:** Prefer deterministic resampling and a small channel projector over re-training a full ControlNet in most compatibility cases (fast, stable). Record any conversions/projections in metadata for reproducibility.
    - **Tests & fixtures:** Add lightweight synthetic safetensors/diffusers fixtures to `testing/fixtures/` (e.g., `controlnet_safetensors_synthetic.safetensors`) and unit tests `testing/test_controlnet_compat.py` and `testing/test_convert_safetensors_to_diffusers.py` that:
      - load a safetensors fixture via `load_controlnet_checkpoint`, run a deterministic forward with a small synthetic control image and assert numeric output and that `{ converter_version, original_format, channel_map }` metadata was written to a manifest or checkpoint.
      - verify `convert_safetensors_to_diffusers` produces a valid diffusers artifact when requested and that round-trip metadata is recorded.
    - **Verification checks (design-level):** During onboarding, attempt to load a known problematic checkpoint (safetensors + 4-channel + 1280 width) and run a forward pass using resampling & projection; verify the forward succeeds and conversion metadata is recorded. Add CI unit tests that cover the common variants (4ch→16ch, resample 1280→320).

    - **Z-Image / Z-Image-Turbo practical guidance:**
      - **Use the Alibaba PAI controlnets as-is (safetensors) or prefer the 8-step distilled variants** for inference speed and clarity when paired with Z-Image-Turbo (see `Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors`). VideoX-Fun maintains these weights and example inference scripts (e.g., `examples/z_image_fun/predict_t2i_control_2.1.py`).
      - **Channel mismatch (4 vs 16):** Apply `normalize_control_channels(control_tensor, expected_channels)` (1×1 conv projector or linear mapping) as a deterministic adapter layer; save projector weights in adapter metadata so conversions are reproducible.
      - **Resolution mismatch (1280/1328 vs 320):** Resample the *bucket/resized* image (post-augmentation) down to the pipeline control size before generating control tensors, using `resample_control_image(..., antialias=True)` with deterministic interpolation. For Z-Image specifically, prefer using the provided 8-step distilled control-compatible weights when available (they were created to address quality/speed issues observed when pairing ControlNet with Z-Image-Turbo).
      - **Format mismatch (safetensors vs diffusers):** If no diffusers-format ControlNet exists, use `load_controlnet_checkpoint(path)` (safetensors loader) and the channel/size adapters above. Optionally provide a conversion utility `convert_safetensors_to_diffusers(...)` for maintainers and record converter_version metadata.
      - **Operational note:** VideoX-Fun demonstrates practical ingestion of these safetensors in its `examples/` directory — replicate their wrapper pattern (load safetensors -> normalize channels -> resample -> wrap in thin ControlNet adapter) if a native diffusers model isn't available.
      - **Search for diffusers-format alternatives:** check community spaces such as `AiSudo/ZIT-Controlnet`, `akhaliq/Z-Image-Turbo-controlnet`, and other HF collections; if a diffusers-formatted ControlNet exists, prefer using it directly to reduce conversion burden.

3) BaseSDTrainProcess changes
   - `process_general_training_batch` already references `batch.control_tensor` and `batch.control_tensor` is doubled when `do_double`, so minimal changes are needed here — however ensure control tensor is created and matches batch doubling and device/dtype.
   - Add a preprocessing section in the batch pipeline to generate/load `batch.control_tensor` as a torch tensor on `self.device_torch` using dtype consistent with other control inputs (e.g., `dtype = get_torch_dtype(self.train_config.dtype)`).
   - Add config validation to `validate_configs` to ensure `controlnet.*` options are valid.

3.a) Memory management and Accelerate offload (primary mitigation)
   - Purpose: Avoid exceeding GPU memory by using Accelerate's offload/dispatch features — Accelerate is already integrated into the toolkit and is the **primary, recommended** strategy for production offloading and DDP safety. `MemoryManager` can be supported as an alternate strategy, while `manual_swap` is explicitly a CPU/dev testing fallback only.
   - Strategy (frozen ControlNet):
     - Compute control residuals under `torch.no_grad()` with ControlNet dispatched/placed on GPU via Accelerate, then detach residuals and store them according to `controlnet.residual_storage` config (`gpu` or `cpu_pinned`).
     - After residual computation: use Accelerate's dispatch/offload APIs to return the ControlNet to CPU or the desired device and bring the UNet to the GPU to run the forward using the computed residuals.
     - Prefer `dispatch_model`, `load_checkpoint_and_dispatch`, `device_map='cpu'` and `offload_folder` options rather than manual `.to('cpu')` calls for DDP/compiled-model safety and correct device mapping.
     - Add a config option `controlnet.offload_strategy` with values `accelerate | memory_manager | manual_swap | none` and `controlnet.residual_storage` with `gpu | cpu_pinned`. Note: **`accelerate` should be used in production; `manual_swap` is only for CPU dev testing.**
   - Implementation notes:
     - Add helper methods: `compute_control_residuals(batch, noisy_latents, timesteps) -> residuals` and `offload_adapter(adapter, strategy)` + `bring_adapter(adapter, strategy)` that integrate with Accelerate dispatch APIs.
     - Ensure residuals are compatible with UNet (including CFG duplication) and that residuals are detached and non-grad.
     - Add logging and timing around transfers so we can detect that swapping overhead is acceptable and fall back to other strategies if not.
   - Tests:
     - `swap_correctness_test` - baseline (adapter+unet GPU-resident) vs swapped flow outputs numerically close for the same inputs.
     - `swap_memory_smoke_test` - measure peak GPU memory and assert reduced peak when offload strategy is used.
     - `ddp_safety_test` - ensure offload strategy works (or fails with a clear message) when running under DDP; prefer Accelerate-based offload for DDP safety.

4) Adapter & ControlNet integration
   - `setup_adapter` already supports `control_net` adapter loading with `ControlNetModel.from_pretrained(...)`. Ensure that when `adapter_config.type == 'control_net'`, the adapter's forward receives control images (control tensor) as conditioning.
   - **Default behavior for ControlNet training:** When `controlnet.use_controlnet` (or `adapter_config.type == 'control_net'`) is active, **default `adapter_config.train = False`** so pretrained ControlNet adapters remain frozen and provide stable spatial conditioning while LoRA/LoKr trains for appearance. Add a configuration validation that sets this default and logs the behavior when starting the job.
   - Add logic to add the controlnet adapter parameters into optimizer parameter groups only when `adapter_config.train == True`. Add a unit test `optimizer_param_test` to assert that controlnet params are excluded from the optimizer when frozen, and included when `adapter_config.train=True`.
   - Update save/load logic so that ControlNet weights are persisted (e.g., `self.adapter` saved using existing adapter codepaths). Loading `latest_save_path` for controlnet should be supported. If finetuning is enabled, record a `controlnet_finetuned` metadata flag in the checkpoint manifest.
   - **Implementation note:** initial offload implementation is provided in `toolkit/controlnet_offload.py` with a safe `manual_swap` strategy and CPU-pinned residual support (`cpu_pinned`) for CPU-only development and unit tests. Unit tests in `testing/test_controlnet_offload.py` validate manual swap behavior and `compute_control_residuals` semantics on CPU environments. **Accelerate is the toolkit's primary offload mechanism and will be implemented next (GPU/DDP integration tests and ddp_safety_test will follow).** MemoryManager remains an optional alternate strategy for advanced deployments.

5) Training loop changes
   - Modify the training forward to pass `control_tensor` to the model's training forward path (e.g., into `sd.get_model_to_train()` / `self.sd` training step). This might involve extending the model's `forward` or the training helper to accept `control` arg.
   - Ensure classifier-free guidance-style unconditioned control is applied when computing unconditional samples (e.g., pass zero tensor or blank control image for unconditional pass when doing CFG).
   - If the repo's SD model pipeline already accepts a `control` input when `sd.adapter` or `sd.network` is present, wire `batch.control_tensor` into that argument. Otherwise add a new input path in `self.sd` to consume `control_tensor`.

6) Sampling & eval changes
   - `sample(...)` should support generating sample images with control images using the same OpenPose generation logic.
   - Ensure `self.sd.generate_images` and preview code can accept control tensors and pass them to ControlNet.

7) Tests & CI
   - Unit test for `make_openpose_map` with synthetic poses and images.
   - Unit test for the control generation CLI (`tools/gen_control.py`) verifying output files, optional caching behavior, and manifest updates when caching is used.
   - Unit test for the control manifest validator (`testing/control_manifest_test.py`) to assert schema compliance, atomic write semantics, and `params_hash` detection.
   - Unit test for DB migration (`testing/test_db_migrations.py`) that runs the migration against an in-memory sqlite DB and validates new schema.
   - Unit test for safetensors/diffusers compatibility (`testing/test_controlnet_compat.py`) using a synthetic fixture to assert `load_controlnet_checkpoint` forwards and records metadata.
   - Unit test for dataloader loading cached control images (and on-the-fly fallback when cache not present) and `augment_align_test` to verify geometric augmentations remain aligned with controls.
   - Integration test: small training run (few steps) with a tiny dataset that has cached controls to ensure `batch.control_tensor` is loaded, gradients flow (if adapter trainable), and saving/loading of adapter weights works.
  - **GPU / DDP tests (manual only):** Tests that validate Accelerate-based swapping, memory smoke tests, and DDP safety require a GPU and an Accelerate-configured environment. **These tests are intended to be run manually by a maintainer on GPU hardware and** **should not be added to standard PR CI workflows**. Note: this project does not have a GPU CI runner and we will not add one—GPU tests are explicitly manual and maintained as on-demand checks by contributors/maintainers.
    To run locally on a GPU machine:
     - Activate venv: `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (POSIX)
     - Ensure `accelerate` and CUDA drivers are available and configured.
     - Run: `python -m pytest testing/test_controlnet_offload_gpu.py -q`
     - Run the benchmark tool for transfer timings: `python tools/benchmark_offload.py --strategy accelerate --size-mb 200 --iters 3`
     - Document results and any environment differences in a short comment on the PR or in `LEARNINGS.md` for future reference.
8) Docs & examples
   - Add `ControlTrain.md` (this file) with config examples.
   - Add sample config file `config/examples/controlnet_openpose_train.yml` demonstrating the options.
   - Add a short subsection to README to explain controlnet options and tradeoffs.

9) UI changes
   - Add a ControlNet section to the *Jobs → New Dataset* UI:
     - Checkbox: `Use ControlNet (OpenPose)`
     - Control type dropdown (default: `openpose`)
     - Toggles/inputs: `Generate on-the-fly` (checkbox, default), `Cache control maps` (checkbox, optional), `pose_model` (openpose|movenet|mediapipe), `confidence_threshold` (0.0-1.0), `skeleton_thickness`, `use_heatmaps` (bool), `control_size` (resize)
     - When caching is enabled, save generated pose maps to `datasets/<dataset_name>/pose/` relative to the dataset root. Use a predictable filename matching source image names.
     - Provide a UI preview showing a sample image/pose map pair and an action button `Cache pose maps for dataset` that enqueues the generation job and stores them in the `pose` subfolder.
   - Ensure the UI persists these settings in the job payload so the backend dataloader can pick the options up.
   - Update server-side endpoints (dataset creation API) to accept and validate the controlnet options.

---



---

## Design considerations & integration checklist ✅
This section addresses integration decisions and practical updates needed across UI, DB, CLI and code, while keeping changes minimal and reusing existing functionality where possible.

- Reuse existing code where sensible (minimal invasive changes):
  - Keep dataset loading, augmentation, bucket, and VAE encoding code paths; call a new canonical helper `toolkit/pose.py::make_openpose_map(...)` at the canonical point (after bucket/resize + augs, before VAE latents) rather than reorder transforms.
  - Reuse `BaseSDTrainProcess` adapter and optimizer flows; add small adapters/wrappers rather than rewrite training internals (`toolkit/controlnet_compat.py` wraps safetensors/diffusers handling, `tools/gen_control.py` wraps generation/caching).
  - Avoid breaking existing canny flows — keep `tools/precompute_control.py` and `make_canny_image` for backwards compatibility and for users who prefer edge conditioning.

- UI changes (`ui/src/app/jobs/new` and supporting modules):
  - Add fields to the New Job UI: `Use ControlNet (OpenPose)`, `control_type` dropdown, `pose_model` (openpose|movenet|mediapipe), `confidence_threshold`, `skeleton_thickness`, `use_heatmaps`, `generate_on_the_fly` (default true), `cache_control` (optional), `control_size`.
  - Add preview component showing a sample image / pose map pair and an action button `Cache pose maps for dataset` to enqueue a control-generation job (use a `POST /api/control_gen/preview` to get a single sample control preview without queuing a full job).
  - Add client helper (e.g., `ui/src/utils/controlGen.ts`) and wire the UI to POST to `/api/control_gen`. Define a tight payload JSON schema `ui/src/schemas/control_gen_payload.json` with fields `{ dataset_id, control_type, cache_enabled, control_params }` and server-side validation mirroring the client schema.
  - Add a background worker `ui/cron/actions/processControlGenQueue.ts` to run `tools/gen_control.py` in the background, update job progress and status, persist logs, and write `manifest_path` into the job row when complete. Include cancel/resume semantics for long-running jobs.

- DB & API routes (SQLite + server routes):
  - Minimal approach: reuse the existing `PrecomputeJob` pattern or add a dedicated `ControlGenJob` table. Add these fields to dataset/job payloads and DB rows: `control_manifest_path TEXT NULL`, `control_params JSON NULL`, `generator_version TEXT NULL`, `cache_enabled INTEGER DEFAULT 0`.
  - **Migration script (example):** add `ui/db/migrations/20251230_add_control_manifest.sql` which performs one or both of the following depending on choice of persistence:
    ```sql
    -- Option A: add columns to existing datasets table
    ALTER TABLE datasets ADD COLUMN control_manifest_path TEXT NULL;
    ALTER TABLE datasets ADD COLUMN control_params TEXT NULL;
    ALTER TABLE datasets ADD COLUMN generator_version TEXT NULL;

    -- Option B: create a dedicated control_gen_jobs table
    CREATE TABLE control_gen_jobs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      dataset_id INTEGER NOT NULL REFERENCES datasets(id),
      user_id INTEGER,
      control_type TEXT,
      control_params JSON,
      manifest_path TEXT,
      generator_version TEXT,
      cache_enabled INTEGER DEFAULT 0,
      status TEXT,
      progress REAL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    ```
  - **API endpoints & contract:**
    - POST `/api/control_gen` — body: `{"dataset_id": int, "control_type": "openpose", "control_params": {...}, "cache_enabled": bool}` → returns `{ "job_id": int, "status": "queued" }`.
    - GET `/api/control_gen/{id}/status` — returns `{ id, dataset_id, status, progress, manifest_path, control_params }`.
    - POST `/api/control_gen/{id}/apply` — body: `{ "apply": true }` → atomically writes `control_manifest_path` and `control_params` into the `datasets` record (server must validate manifest schema before commit).
  - **Server behavior & validation:** validate payloads against a tight JSON schema, authorize user actions, and perform DB/manifest updates inside a single transaction to avoid partial state (use `PRAGMA foreign_keys=ON` and an explicit transaction for SQLite). Write manifests atomically (temp file → os.replace) for cross-platform safety.
  - **Migration tests:** add `testing/test_db_migrations.py` which runs the migration against an in-memory sqlite DB and asserts the new columns/table exist and default values are correct. Add an optional backfill CLI `tools/migrate_controls.py --dataset <name> --from <manifest.json>` to assist maintainers in migrating legacy datasets.
  - Persist manifest/params atomically and keep a reproducible `params_hash` to detect generator mismatches and stale caches.

- CLI updates and tools:
  - Add `tools/gen_control.py` to perform on-the-fly-first generation and optional caching into `datasets/<name>/pose/` with an atomic `control_manifest.json` when caching. CLI flags should include `--dataset`, `--pose-model`, `--confidence`, `--size`, `--overwrite`, `--chunk-size`, `--generate-after-augmentations`, and `--output-dir` and return machine-readable JSON on completion (`{ "manifest_path": "...", "params_hash": "..." }`).
  - Add `tools/apply_control_manifest.py` to apply the manifest to dataset metadata with safe options (`--force`, `--noop`) and support `--dry-run` to validate manifest schema without applying.
  - Add `run.py` integration: `run.py train --config config/examples/controlnet_openpose_train.yml` must accept `controlnet` keys and `control_params` in job payloads so CLI-created jobs behave the same as UI-created jobs. Add `tests/test_cli_control_gen.py` to assert the CLI payloads are created and accepted by server endpoints.
  - Add `toolkit/controlnet_compat.py` to load safetensors or diffusers checkpoints, convert when requested, resample/normalize channels, and persist conversion metadata.

- Dataloader & pipeline timing (explicit confirmation):
  - Yes — the design specifies **generate control images after the dataset image has been loaded into buckets and resized/augmented, but before conversion to VAE latents**. This guarantees pixel-perfect alignment and simplifies correctness.

- Safetensors / non-diffusers handling:
  - Use `toolkit/controlnet_compat.py` which:
    - Attempts to load a checkpoint as a Diffusers `ControlNetModel` if possible.
    - If only safetensors are available, loads tensors and wraps them in a thin adapter that exposes the same forward signature expected by the training pipeline (or offers an optional conversion helper `convert_safetensors_to_diffusers`).
    - Metadata (format, converter_version) must be persisted in the manifest/checkpoint for reproducibility.
  - This mirrors VideoX-Fun patterns (they accept safetensors and use wrapper/conversion logic; prefer distilled 8-step variants for Z-Image pairing where available).

- Channel / width mismatch strategy (VideoX-Fun compatible):
  - Channel mismatch (e.g., 4 vs expected 16): apply a deterministic 1×1 convolution or linear projector `normalize_control_channels(control_tensor, expected_channels)` and persist projector weights with adapter metadata.
  - Spatial mismatch (e.g., 1280 vs 320): resample the bucket/resized image down to pipeline control size deterministically (`resample_control_image(..., antialias=True)`) before computing pose maps or feeding to ControlNet.
  - Prefer using distilled or converted control nets (e.g., Z-Image 8-step distilled variants) when available for better speed/quality.

- Acceptance & verification checks (high-level):
  - UI: user can enqueue control-generation jobs; job record stores `control_manifest_path` and params.
  - Reproducibility: bit-identity check between cached control map and on-the-fly generated control when `generator_version` and params match; dataset onboarding fails with clear message if mismatch found.
  - Compat: loading a known safetensors controlnet plus resampling & projection must produce a successful forward pass; conversion metadata is recorded.

- Minimal-change principle & rollout guidance:
  - Keep changes small and reversible: add new helpers/stubs and feature-flag complex behavior (e.g., `controlnet.offload_strategy`) behind config switches.
  - Prefer small PRs: (1) add canonical pose helper & CLI stub, (2) wire dataloader to call helper and add verification, (3) add `controlnet_compat` loader, (4) add UI + API endpoints, (5) add offload changes & GPU manual tests.

---

## Acceptance Criteria ✅
- Training pipeline can be toggled to use OpenPose control maps via config.
- For each batch, `DataLoaderBatchDTO` carries `control_tensor` shaped appropriately and on the correct device (control created after augmentation when on-the-fly generation is enabled).
- When using a pretrained ControlNet for conditioning, the default behavior for a 'Train with ControlNet' job is **`adapter.train = False`** (ControlNet frozen); only when explicitly enabled (`adapter.train = True`) will ControlNet params be included in optimizer groups and updated by training.
- Training with `controlnet.train=true` updates controlnet weights and saving/loading preserves them, with a checkpoint metadata flag indicating finetuning.
- Tests to include: `augment_align_test`, `control_cache_idempotence`, `optimizer_param_test`, memory smoke tests, and small integration run verifying LoRA/LoKr updates while ControlNet remains frozen by default.
- Documentation, example config (`config/examples/controlnet_openpose_train.yml`), and UI changes included.

---

## Risks & Notes ⚠️
- Generating OpenPose pose maps on-the-fly adds CPU overhead; consider caching for very large datasets. **Mitigation:** prefer on-the-fly generation for correctness and cache only as a performance optimization when necessary; benchmark CPU | IO overhead as part of smoke tests.
- Spatial alignment: must ensure augmentations are applied identically to both original image and control image. **Mitigation:** enforce `generate_after_augmentation` for on-the-fly generation; for precomputed controls either disable geometric augmentations or precompute augmented variants or store augmentation metadata for deterministic transforms. Add `augment_align_test` unit test to detect misalignment.
- ControlNet expected input format may vary per model variant; provide flexibility in processing (single-channel vs 3-channel, scaling, dtype). **Mitigation:** expose `adapter.get_expected_control_spec()` validation on load and convert/normalize precomputed or generated controls to the adapter's spec automatically.
- Memory and OOM risks (extra control tensors, doubled batches for short/long captions, CFG duplication): **Mitigation:** add `control_size` and `control_dtype` config guidance; prefer the toolkit-integrated **Accelerate-based offload** (recommended) or existing `MemoryManager` for safe DDP-capable offloading. Add `controlnet.offload_strategy` config (values: `none|accelerate|memory_manager|manual_swap`) and test offload strategies with `swap_memory_smoke_test` and `ddp_safety_test`.
- **Prerequisite:** Accelerate is integrated into the toolkit and should be available in runtime environments that will use offload or run DDP tests (it is already listed in `requirements.txt`).
- Cache idempotence & manifest consistency: **Mitigation:** make the control generation CLI idempotent when caching is enabled, update manifest atomically (write to temp and rename), provide `--overwrite` flag, and add `control_cache_idempotence` unit test.
- Optimizer parameter correctness (frozen vs trainable adapters): **Mitigation:** add `optimizer_param_test` unit test asserting controlnet params excluded when frozen; when finetuning is enabled warn about memory and recommend appropriate LR and schedules.
- Multi-device & dtype/device mismatch: **Mitigation:** ensure control tensors are moved to `self.device_torch` with correct dtype during `process_general_training_batch`; add multi-GPU smoke tests where CI or dev machines permit and ensure offload path uses Accelerate for DDP safety.


## Prioritized Action Items — High & Medium Priority (added)

The following high- and medium-priority items are now explicitly part of the implementation plan and will be implemented and tested as described below.

### High priority

- Offload / DDP-safe implementation (Accelerate primary / MemoryManager optional)
  - Implement `controlnet.offload_strategy` (values: `accelerate|memory_manager|manual_swap|none`) and helpers `offload_adapter(adapter, strategy)` and `bring_adapter(adapter, strategy)` with **Accelerate as the primary implementation** for DDP safety and production use. Provide `manual_swap` as a lightweight CPU-only fallback used in dev and unit tests.
  - Implement `compute_control_residuals(batch, noisy_latents, timesteps)` that returns detached residual tensors and supports `residual_storage` of `gpu` or `cpu_pinned`.
  - Tests: `swap_correctness_test` (baseline ~= swapped outputs), `swap_memory_smoke_test` (verify peak memory reduction), `ddp_safety_test` (ensure offload strategy works in DDP when using Accelerate; if environment lacks proper support the test should fail with a clear, actionable message).

- Augmentation alignment
  - Add `augment_align_test` that asserts equivalence between (apply augmentations → generate control) and (generate control → apply same augmentations) for geometric transforms.
  - Enforce `generate_after_augmentation` semantics when `generate_on_the_fly=True`.

- Optimizer parameter correctness
  - Add `optimizer_param_test` which asserts pretrained ControlNet params are not included in optimizer when `adapter.train=False` and are included when explicitly set to `True`.

- Residual correctness & CFG duplication
  - Ensure precomputed residuals are duplicated or handled identically to on-the-fly residuals during CFG (classifier-free guidance) passes. Add to `swap_correctness_test`.

### Medium priority

- Precompute manifest schema & atomic updates
  - Define manifest file format (JSON/YAML) mapping source image → control image path and precompute parameters.
  - Implement atomic updates (write to temp file → rename) and add `precompute_idempotence` test.

- Resilient precompute job semantics
  - Implement chunked precompute with resume/cancel/retry semantics and progress updates for the UI. Add tests for job resumption and partial-failure recovery.

- Performance / transfer benchmarks
  - Add `tools/benchmark_offload.py` to measure GPU↔CPU transfer speeds on the user's hardware, residual compute time, and warn if swapping overhead exceeds acceptable thresholds. Provide a small, human-run CLI that runs on systems with Accelerate and GPU available; benchmark runs are skipped in automated CI unless explicitly enabled.

- Memory smoke tests
  - Add automated memory benchmarks that measure peak GPU memory for baseline vs offload strategies and generate guidance (control_size, batch size) when OOM is likely.

- Save/load / metadata
  - Persist precompute params, `controlnet.name_or_path`, `offload_strategy`, and `residual_storage` in checkpoint metadata (`aitk_meta.yaml`), and add `checkpoint_meta_test`.

---

## Next step (recommended - precompute-first)
1. Implement `tools/gen_control.py` and unit tests for it (optional caching of pose maps, cache idempotence, manifest updates when caching is used, CLI options to set pose model/confidence/size/overwrite). Prioritize atomic manifest updates and `control_cache_idempotence` tests.
2. Implement `generate_after_augmentation` behavior in the dataloader: when `generate_on_the_fly=True`, compute controls after applying geometric augmentations and add `augment_align_test` to validate alignment.
3. Update dataloader to prefer cached pose maps (load from `datasets/<dataset_name>/pose/`) and add `control_image_path` to dataset metadata handling, including behavior options when augmentations are enabled (disable geometric augments or cache augmented variants).
4. Implement Accelerate-based offload strategy and helpers (Accelerate is primary; MemoryManager optional): add `controlnet.offload_strategy` and `controlnet.residual_storage` configs, implement `compute_control_residuals` and adapter `offload/bring_back` helpers using Accelerate dispatch APIs, and add `swap_correctness_test`, `swap_memory_smoke_test`, and `ddp_safety_test` (ddp_safety_test should use Accelerate and provide clear failure messages when environment or configuration is unsupported).
5. Add config fields and example config (explicitly default `adapter.train=false` for ControlNet-enabled jobs) and add a UI action to enqueue `Cache pose maps for dataset` (background job with progress/cancel).
6. Wire `batch.control_tensor` into `BaseSDTrainProcess.process_general_training_batch`, ensure tensors are on `self.device_torch` with the correct dtype, and add `optimizer_param_test` and memory smoke tests to validate frozen-controlnet behavior and OOM guidance.
7. Run a small integration prototype: pretrained ControlNet (frozen) + LoRA/LoKr training on a tiny dataset with precomputed controls to validate end-to-end behavior and measure offload transfer overhead in a timing benchmark.

---

## Related work & suggested plan adjustments
Recent papers confirm the pattern of using a pretrained/frozen ControlNet branch for spatial conditioning while training lightweight adapters (LoRA/ControlLoRA/LoKr) for appearance or task-specific adaptation. Important takeaways from a quick literature survey (representative papers):

- **Preventing Shortcuts in Adapter Training via Providing the Shortcuts** (arXiv:2510.20887) — proposes routing confounding factors through auxiliary modules (ControlNet/LoRA) during adapter training to avoid spurious shortcut learning and improve generalization. Implementation tasks: add `TrainConfig.controlnet_reroute` (none|precompute|always), add `shortcut_rerouting_test` to assert reroute behavior on synthetic confounded datasets (effort: low).
- **LumiCtrl** (arXiv:2512.17489) — uses a frozen ControlNet and a masked reconstruction loss to disentangle illumination control from structure while fine-tuning other components. Implementation tasks: add `controlnet.aux_loss: none|masked_recon|edge_loss` config, implement `compute_control_masked_recon_loss` in `toolkit/controlnet_aux.py`, and a small integration example `config/examples/controlnet_lumictrl.yml` demonstrating masked-recon training (tests: `testing/test_controlnet_masked_recon.py`, effort: medium).
- **DEMIST** (arXiv:2511.12396) — uses per-scale spatial residual hints and LoRA-modulated attention; this supports making per-scale residuals first-class. Implementation tasks: make multi-scale residual writer/reader canonical (tuple-of-tensors format), add `testing/residual_shapes_test.py`, and extend `tools/benchmark_offload.py` to collect per-scale transfer timings (effort: medium).
- **FrameDiffuser** (arXiv:2512.16670) — trains ControlLoRA for temporal coherence in a three-stage regime. Implementation tasks: add `control_training_schedule` support and implement a modular scheduler that supports `standard|three_stage|ping_pong` phases, with a `testing/schedule_phase_transition_test.py` to validate behavior (effort: medium).
- **GLYPH-SR** (arXiv:2510.26339) — alternates control strategies (ping-pong scheduler) and trains a dedicated ControlNet branch with frozen main branch. Implementation tasks: add a ping-pong scheduler mode and experiments config `config/examples/controlnet_pingpong.yml`, and add a small experiment script and logging to reproduce GLYPH-SR style alternating schedules (effort: medium).

Plan adjustments (conservative, backward compatible):

1. **Document & test the "shortcut-rerouting" principle**: add a test/spec (`shortcut_rerouting_test`) and a short section in `ControlTrain.md` recommending that datasets with known confounders route them via ControlNet/LoRA during adapter training. Add guidance for designing auxiliary losses (e.g., masked reconstruction) that preserve disentanglement.

   **Implementation notes & conventions (precompute residuals):**
   - File naming & location: precompute per-image residuals to a dataset subfolder (config: `DatasetConfig.control_residuals_path`) using the convention `<basename>_residuals.pt`. Each file MUST be a dict with keys:
     - `meta`: `{ "scales": n, "shapes": [[C,H,W], ...], "dtype": "float32", "version": 1 }`
     - `scales`: a list/tuple of tensors ordered **coarse→fine** (i.e., smallest spatial resolution first, largest last). For example `scales[0]` corresponds to the deepest (bottleneck) residual. Each tensor may be either `[C,H,W]` or `[1,C,H,W]`; the dataloader normalizes to `[batch, C, H, W]`.
     - `checksum`: `sha256` of the packed tensor data (for quick integrity checks).
   - Use a small wrapper API `toolkit/residuals.py` with helpers:
     - `pack_residuals(residuals, path)` — writes `{meta, scales, checksum}` atomically to disk.
     - `unpack_residuals(path)` — validates checksum, dtype and shapes and returns normalized tensors.
     - `validate_residuals_format(path, expected_shapes)` — raises a clear error on mismatch.
   - Ordering & semantics: explicitly document that scales are `coarse->fine` and provide a small example in the docstring / test fixtures so implementers do not disagree on ordering.
   - Config knob: `TrainConfig.controlnet_reroute` accepts `none|precompute|always`. Use `precompute` to use residuals when present, `always` to force reroute behavior, and `none` (default) to disable.
   - Alignment: precomputed residuals must match any augmentations applied to the corresponding image (or be generated after augmentation). If geometric augments are used, either disable them for precompute-first datasets or store augmentation metadata (seed or transform spec) and replay deterministic transforms for the control residuals; record `generated_after_augmentations` in the control manifest to simplify validation.
   - Sanity checks: the dataloader will validate residual entries are tensors, have consistent numbers of scales and shapes, and that checksums match; mismatches should either raise informative errors or fall back to on-the-fly adapter computation depending on config.
   - Test coverage: add `precompute_idempotence`, `precompute_manifest_test`, `residual_shapes_test`, and `swap_correctness_test` (ensures precomputed residuals and on-the-fly computation yield compatible prediction results).

2. **Make per-scale residuals an explicit target**: ensure `compute_control_residuals` and the offload/residual API cleanly support adapters that return multi-scale residual tensors (one per UNet scale). Add `residual_shapes_test` to validate shapes against a small synthetic UNet spec and document the residual format in the code/docs.

3. **Add optional auxiliary losses/config**: add a small config surface under `controlnet.*` (e.g., `controlnet.aux_loss: none|masked_recon|edge_loss`) and hooks in the training loop to apply them when enabled (default off). This keeps defaults unchanged but enables reproducing LumiCtrl-style methods.

4. **Support ControlLoRA and training schedules**: ensure `AdapterConfig` clearly supports `control_lora` and add `control_training_schedule` config (values like `standard|ping_pong|alternate`) to facilitate experiments like GLYPH-SR and FrameDiffuser. Keep defaults conservative (standard).

5. **Benchmarks and profiling**: add per-scale transfer & compute timing to `tools/benchmark_offload.py` to help decide whether per-scale residuals should be stored on GPU or pinned-CPU (useful when residuals are large).
