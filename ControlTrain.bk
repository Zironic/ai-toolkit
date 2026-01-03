# ControlNet Training Plan (Canny → ControlNet)

## Summary ✅
Add an optional ControlNet-based training mode that uses Canny edge images derived from the dataset as control inputs. The implementation should be **precompute-first** (precompute and cache Canny images into `datasets/<dataset_name>/canny/`), with a lightweight on-the-fly fallback for small datasets or quick experiments. ControlNet weights should be loadable/savable as part of the existing adapter/save flow.

---

## Goals 🎯
- Produce `control_tensor` per training sample (Canny edges → tensor format expected by ControlNet).
- Provide config toggles for generation mode, thresholds, caching, and whether to train the ControlNet adapter.
- Wire control tensors through the dataloader → `BaseSDTrainProcess.process_general_training_batch` → model training loop (ensure gradients flow into controlnet adapter if training).
- Add tests, docs, and example config.

---

## Design & Implementation Plan (step-by-step) 🔧

1) Design & API (analysis) — Files affected: `jobs/process/BaseSDTrainProcess.py`, dataloader, `DataLoaderBatchDTO`, config schemas
   - Decide on default control type: `canny` (extendable to other control types later).
   - Config keys (suggested):
     - `controlnet:`
       - `use_controlnet: bool` (master switch)
       - `type: 'canny'` (other values later)
       - `generate_on_the_fly: bool` (default True)
       - `precompute_control: bool` (default False)
       - `canny_threshold1`, `canny_threshold2`, `blur` etc.
       - `resize_to` / `control_size` (control image size)
       - `train: bool` (train controlnet weights)
       - `name_or_path` (pretrained controlnet model if any)
   - Update `config/examples/` with a short example.

   - **Debug logging requirement:**
     - At all key steps and before/after risky operations (e.g., model/device swaps, file I/O, augmentation, control tensor creation, optimizer param changes), print regular debug information using `printf` or the established logging method.
     - Include enough context in each debug print to aid crash diagnosis and reproducibility (e.g., batch index, config values, device, tensor shapes, memory usage if available).
     - Ensure debug prints are present in all new code paths, especially those that could crash or are hard to reproduce.
     - Document this requirement in the plan and ensure it is covered by code review and tests.

2) Dataloader & Batch DTO changes (precompute-first)
   - Add fields to `DataLoaderBatchDTO`: `control_tensor` (torch.Tensor or None) and `control_image_path` (for caching/traceability).
   - Provide a precompute script `tools/precompute_control.py` and CLI to generate control images for a dataset and save them to `datasets/<dataset_name>/canny/` with filenames matching source images (e.g., `image_0001.jpg` -> `image_0001_canny.png`). The script should:
     - Accept config for thresholds/blur/size.
     - Optionally overwrite or skip existing files.
     - Produce a lightweight manifest or add entries to the dataset manifest mapping source -> control path.
   - Dataloader behavior and alignment (mitigation):
     - **Generate control after augmentation** when `generate_on_the_fly=True` (preferred when geometric augmentations are enabled) so that the control map is computed from the augmented image guaranteeing pixel alignment.
     - When `precompute_control=True` is used with **geometric augmentations** in training, either (A) disable geometric augments that break alignment, or (B) precompute additional augmented control variants or store augmentation metadata so the control image can be transformed to match the augmented image. Add a clear config option to choose the preferred behavior and document trade-offs.
     - If a precomputed control is missing and `generate_on_the_fly=True`, generate it after any applied augmentations; if `generate_on_the_fly=False`, raise an informative error or skip the sample based on config.
   - Implement a helper `make_canny_image(image, thresholds, blur)` returning a control image (PIL/ndarray). Convert to the correct channel format (single-channel or expand to 3-channel) and to torch tensor on the correct device/dtype during batch preprocessing.
   - Add an explicit unit-test `augment_align_test` that verifies equivalence between (apply augmentations → generate control) and (generate control → apply same augmentations) to detect alignment bugs early.
   - UI/CLI integration: allow dataset creation UI to trigger the precompute job and show progress; the precompute tool should be idempotent (safe to re-run) and update the manifest atomically to avoid inconsistent state.

3) BaseSDTrainProcess changes
   - `process_general_training_batch` already references `batch.control_tensor` and `batch.control_tensor` is doubled when `do_double`, so minimal changes are needed here — however ensure control tensor is created and matches batch doubling and device/dtype.
   - Add a preprocessing section in the batch pipeline to generate/load `batch.control_tensor` as a torch tensor on `self.device_torch` using dtype consistent with other control inputs (e.g., `dtype = get_torch_dtype(self.train_config.dtype)`).
   - Add config validation to `validate_configs` to ensure `controlnet.*` options are valid.

3.a) Memory management and Accelerate offload (primary mitigation)
   - Purpose: Avoid exceeding GPU memory by using Accelerate's offload/dispatch features — Accelerate is already integrated into the toolkit and is the **primary, recommended** strategy for production offloading and DDP safety. `MemoryManager` can be supported as an alternate strategy, while `manual_swap` is explicitly a CPU/dev testing fallback only.
   - Strategy (frozen ControlNet):
     - Compute control residuals under `torch.no_grad()` with ControlNet dispatched/placed on GPU via Accelerate, then detach residuals and store them according to `controlnet.residual_storage` config (`gpu` or `cpu_pinned`).
     - After residual computation: use Accelerate's dispatch/offload APIs to return the ControlNet to CPU or the desired device and bring the UNet to the GPU to run the forward using the precomputed residuals.
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
   - `sample(...)` should support generating sample images with control images using the same canny generation logic.
   - Ensure `self.sd.generate_images` and preview code can accept control tensors and pass them to ControlNet.

7) Tests & CI (precompute-focused)
   - Unit test for `make_canny_image` with synthetic images.
   - Unit test for `tools/precompute_control.py` verifying output files, idempotence, and manifest updates.
   - Unit test for dataloader loading precomputed control images (and on-the-fly fallback only when enabled).
   - Integration test: small training run (few steps) with a tiny dataset that has precomputed Canny images to ensure `batch.control_tensor` is loaded, gradients flow (if adapter trainable), and saving/loading of adapter weights works.
  - **GPU / DDP tests (manual only):** Tests that validate Accelerate-based swapping, memory smoke tests, and DDP safety require a GPU and an Accelerate-configured environment. **These tests are intended to be run manually by a maintainer on GPU hardware and** **should not be added to standard PR CI workflows**. Note: this project does not have a GPU CI runner and we will not add one—GPU tests are explicitly manual and maintained as on-demand checks by contributors/maintainers.
    To run locally on a GPU machine:
     - Activate venv: `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (POSIX)
     - Ensure `accelerate` and CUDA drivers are available and configured.
     - Run: `python -m pytest testing/test_controlnet_offload_gpu.py -q`
     - Run the benchmark tool for transfer timings: `python tools/benchmark_offload.py --strategy accelerate --size-mb 200 --iters 3`
     - Document results and any environment differences in a short comment on the PR or in `LEARNINGS.md` for future reference.
8) Docs & examples
   - Add `ControlTrain.md` (this file) with config examples.
   - Add sample config file `config/examples/controlnet_canny_train.yml` demonstrating the options.
   - Add a short subsection to README to explain controlnet options and tradeoffs.

9) UI changes
   - Add a ControlNet section to the *Jobs → New Dataset* UI:
     - Checkbox: `Use ControlNet (Canny)`
     - Control type dropdown (default: `canny`)
     - Toggles/inputs: `Generate on-the-fly` (checkbox), `Precompute control images` (checkbox), Canny thresholds (`threshold1`, `threshold2`), `blur` radius, `control_size` (resize)
     - When 'Precompute' or when generating and caching, save Canny images to `datasets/<dataset_name>/canny/` relative to the dataset root. Use a predictable filename matching source image names.
     - Provide a UI preview showing a sample image/control pair and an action button `Precompute Canny for dataset` that enqueues/precomputes control images and stores them in the `canny` subfolder.
   - Ensure the UI persists these settings in the job payload so the backend dataloader can pick the options up.
   - Update server-side endpoints (dataset creation API) to accept and validate the controlnet options.

---

## Example config snippet (precompute-first)

```yaml
train_config:
  # ... existing training config
controlnet:
  use_controlnet: true
  type: canny
  generate_on_the_fly: false  # optional fallback for quick experiments
  precompute_control: true     # recommended for large datasets
  canny_threshold1: 100
  canny_threshold2: 200
  blur: 3
  control_size: 512
  train: true
  name_or_path: null
```

---

## Acceptance Criteria ✅
- Training pipeline can be toggled to use canny control images via config.
- For each batch, `DataLoaderBatchDTO` carries `control_tensor` shaped appropriately and on the correct device (control created after augmentation when on-the-fly generation is enabled).
- When using a pretrained ControlNet for conditioning, the default behavior for a 'Train with ControlNet' job is **`adapter.train = False`** (ControlNet frozen); only when explicitly enabled (`adapter.train = True`) will ControlNet params be included in optimizer groups and updated by training.
- Training with `controlnet.train=true` updates controlnet weights and saving/loading preserves them, with a checkpoint metadata flag indicating finetuning.
- Tests to include: `augment_align_test`, `precompute_idempotence`, `optimizer_param_test`, memory smoke tests, and small integration run verifying LoRA/LoKr updates while ControlNet remains frozen by default.
- Documentation and example config included.

---

## Risks & Notes ⚠️
- Generating Canny on-the-fly adds CPU overhead; consider precomputing for large datasets. **Mitigation:** prefer precompute for large datasets but allow on-the-fly generation after augmentation for alignment-sensitive pipelines, and benchmark CPU | IO overhead as part of smoke tests.
- Spatial alignment: must ensure augmentations are applied identically to both original image and control image. **Mitigation:** enforce `generate_after_augmentation` for on-the-fly generation; for precomputed controls either disable geometric augmentations or precompute augmented variants or store augmentation metadata for deterministic transforms. Add `augment_align_test` unit test to detect misalignment.
- ControlNet expected input format may vary per model variant; provide flexibility in processing (single-channel vs 3-channel, scaling, dtype). **Mitigation:** expose `adapter.get_expected_control_spec()` validation on load and convert/normalize precomputed or generated controls to the adapter's spec automatically.
- Memory and OOM risks (extra control tensors, doubled batches for short/long captions, CFG duplication): **Mitigation:** add `control_size` and `control_dtype` config guidance; prefer the toolkit-integrated **Accelerate-based offload** (recommended) or existing `MemoryManager` for safe DDP-capable offloading. Add `controlnet.offload_strategy` config (values: `none|accelerate|memory_manager|manual_swap`) and test offload strategies with `swap_memory_smoke_test` and `ddp_safety_test`.
- **Prerequisite:** Accelerate is integrated into the toolkit and should be available in runtime environments that will use offload or run DDP tests (it is already listed in `requirements.txt`).
- Precompute idempotence & manifest consistency: **Mitigation:** make the precompute CLI idempotent, update manifest atomically (write to temp and rename), provide `--overwrite` flag, and add `precompute_idempotence` unit test.
- Optimizer parameter correctness (frozen vs trainable adapters): **Mitigation:** add `optimizer_param_test` unit test asserting controlnet params excluded when frozen; when finetuning is enabled warn about memory and recommend appropriate LR and schedules.
- Multi-device & dtype/device mismatch: **Mitigation:** ensure control tensors are moved to `self.device_torch` with correct dtype during `process_general_training_batch`; add multi-GPU smoke tests where CI or dev machines permit and ensure offload path uses Accelerate for DDP safety.
- UI/UX long-running precompute: **Mitigation:** run precompute as a background job with progress/cancel support and preview sample pairs in the UI; support chunked precompute for very large datasets so the job can be resumed if interrupted.

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
1. Implement `tools/precompute_control.py` and unit tests for it (idempotence, manifest updates, CLI options to set thresholds/size/overwrite). Prioritize atomic manifest updates and `precompute_idempotence` tests.
2. Implement `generate_after_augmentation` behavior in the dataloader: when `generate_on_the_fly=True`, compute controls after applying geometric augmentations and add `augment_align_test` to validate alignment.
3. Update dataloader to prefer precomputed controls (load from `datasets/<dataset_name>/canny/`) and add `control_image_path` to dataset metadata handling, including behavior options when augmentations are enabled (disable geometric augments or precompute augmented variants).
4. Implement Accelerate-based offload strategy and helpers (Accelerate is primary; MemoryManager optional): add `controlnet.offload_strategy` and `controlnet.residual_storage` configs, implement `compute_control_residuals` and adapter `offload/bring_back` helpers using Accelerate dispatch APIs, and add `swap_correctness_test`, `swap_memory_smoke_test`, and `ddp_safety_test` (ddp_safety_test should use Accelerate and provide clear failure messages when environment or configuration is unsupported).
5. Add config fields and example config (explicitly default `adapter.train=false` for ControlNet-enabled jobs) and add a UI action to enqueue `Precompute Canny for dataset` (background job with progress/cancel).
6. Wire `batch.control_tensor` into `BaseSDTrainProcess.process_general_training_batch`, ensure tensors are on `self.device_torch` with the correct dtype, and add `optimizer_param_test` and memory smoke tests to validate frozen-controlnet behavior and OOM guidance.
7. Run a small integration prototype: pretrained ControlNet (frozen) + LoRA/LoKr training on a tiny dataset with precomputed controls to validate end-to-end behavior and measure offload transfer overhead in a timing benchmark.

---

## Related work & suggested plan adjustments
Recent papers confirm the pattern of using a pretrained/frozen ControlNet branch for spatial conditioning while training lightweight adapters (LoRA/ControlLoRA/LoKr) for appearance or task-specific adaptation. Important takeaways from a quick literature survey (representative papers):

- **Preventing Shortcuts in Adapter Training via Providing the Shortcuts** (arXiv:2510.20887) — proposes routing confounding factors through auxiliary modules (ControlNet/LoRA) during adapter training to avoid spurious shortcut learning and improve generalization. Implication: explicitly route known confounds through auxiliary conditioning during adapter training and add tests for shortcut avoidance.
- **LumiCtrl** (arXiv:2512.17489) — uses a frozen ControlNet and a masked reconstruction loss to disentangle illumination control from structure while fine-tuning other components; suggests useful auxiliary losses (masked reconstruction/edge-aware losses) when training adapters.
- **DEMIST** (arXiv:2511.12396) — uses per-scale spatial residual hints (a ControlNet branch) and LoRA-modulated attention; this supports our residual-precompute idea and suggests per-scale residual interfaces as an explicit design target.
- **FrameDiffuser** (arXiv:2512.16670) — trains ControlLoRA for temporal coherence in a three-stage regime; useful example for multi-stage training strategies and for ControlLoRA-style adapters.
- **GLYPH-SR** (arXiv:2510.26339) — alternates control strategies (ping-pong scheduler) and trains a dedicated ControlNet branch with frozen main branch for SR; suggests scheduling/control weighting experiments can be helpful.

Plan adjustments (conservative, backward compatible):

1. **Document & test the "shortcut-rerouting" principle**: add a test/spec (`shortcut_rerouting_test`) and a short section in `ControlTrain.md` recommending that datasets with known confounders route them via ControlNet/LoRA during adapter training. Add guidance for designing auxiliary losses (e.g., masked reconstruction) that preserve disentanglement.

   **Implementation notes & conventions (precompute residuals):**
   - File naming & location: precompute per-image residuals to a dataset subfolder (config: `DatasetConfig.control_residuals_path`) using the convention `<basename>_residuals.pt`. Each file should contain a tuple or list of tensors representing per-scale residuals (each either `[C,H,W]` or `[1,C,H,W]`). The dataloader will normalize either format to a per-batch tuple of tensors shaped `[batch, C, H, W]`.
   - Config knob: `TrainConfig.controlnet_reroute` accepts `none|precompute|always`. Use `precompute` to use residuals when present, `always` to force reroute behavior, and `none` (default) to disable.
   - Alignment: precomputed residuals must match any augmentations applied to the corresponding image (or be generated after augmentation). If geometric augments are used, either disable them for precompute-first datasets or store augmentation metadata and replay deterministic transforms for the control residuals.
   - Sanity checks: the dataloader will validate residual entries are tensors and have consistent numbers of scales; mismatches should either raise informative errors or fall back to on-the-fly adapter computation depending on config.
   - Test coverage: add `precompute_idempotence`, `precompute_manifest_test`, and `swap_correctness_test` (ensures precomputed residuals and on-the-fly computation yield compatible prediction results).

2. **Make per-scale residuals an explicit target**: ensure `compute_control_residuals` and the offload/residual API cleanly support adapters that return multi-scale residual tensors (one per UNet scale). Add `residual_shapes_test` to validate shapes against a small synthetic UNet spec and document the residual format in the code/docs.

3. **Add optional auxiliary losses/config**: add a small config surface under `controlnet.*` (e.g., `controlnet.aux_loss: none|masked_recon|edge_loss`) and hooks in the training loop to apply them when enabled (default off). This keeps defaults unchanged but enables reproducing LumiCtrl-style methods.

4. **Support ControlLoRA and training schedules**: ensure `AdapterConfig` clearly supports `control_lora` and add `control_training_schedule` config (values like `standard|ping_pong|alternate`) to facilitate experiments like GLYPH-SR and FrameDiffuser. Keep defaults conservative (standard).

5. **Benchmarks and profiling**: add per-scale transfer & compute timing to `tools/benchmark_offload.py` to help decide whether per-scale residuals should be stored on GPU or pinned-CPU (useful when residuals are large).

6. **Related Work note**: add a small 'Related work' paragraph to `ControlTrain.md` summarizing the above evidence and linking to representative papers. This justifies our design choices (frozen ControlNet default, residual precompute, adapter-only training by default).

---

If you'd like, I can implement the small changes now: add the `residual_shapes_test` (unit), a short `controlnet.aux_loss` config stub, and update `compute_control_residuals` docstring and tests to assert multi-scale behavior. Which of those should I do next?