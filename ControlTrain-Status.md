# ControlNet Training — Status & Checklist

This file tracks prioritized work items for the ControlNet (OpenPose → ControlNet) design and implementation. Items are ordered by priority (Top = must-fix now). Use the `status` column to track progress (not-started | in-progress | blocked | completed).

---

## ✅ Top priority (Must-fix)

| # | Task | Owner | Status | Effort | Notes |
|---:|------|-------|--------|--------:|------|
| 1 | Add canonical pose generator `toolkit/pose.py::make_openpose_map` and require its use in dataloader & control CLI | TBD | not-started | low | Deterministic, records `generator_version` and params. Generates after bucket/resize and before VAE latents.
| 2 | Implement on-the-fly-first control generation CLI `tools/gen_control.py` (optional cache mode) | TBD | not-started | low | Replace previous precompute-first wording; supports atomic cache manifest when caching is enabled.
| 3 | Add `toolkit/controlnet_compat.py` (load safetensors, resample, channel normalization, conversion helper) | TBD | not-started | medium | Includes `load_controlnet_checkpoint`, `resample_control_image`, `normalize_control_channels`, `convert_safetensors_to_diffusers`.
| 4 | Enforce generation point: generate controls after bucket/resizing and augmentations, before VAE latents (dataloader change) | TBD | not-started | low | Update `BaseSDTrainProcess` preprocessing and `DataLoaderBatchDTO` behavior.
| 5 | Augmentation alignment verification (dataset onboarding check) — bit-identity check between cached and canonical on-the-fly generation | TBD | not-started | low | Add to onboarding; implement `augment_align_check` script.


## 🔧 High priority (Implementation & safety)

| # | Task | Owner | Status | Effort | Notes |
|---:|------|-------|--------|--------:|------|
| 6 | Replace naive manual-swap offload with Accelerate dispatch helpers in `toolkit/controlnet_offload.py` | TBD | not-started | high | Use `dispatch_model`/`load_checkpoint_and_dispatch` when available. Add clear failure messages for unsupported envs.
| 7 | Implement `compute_control_residuals(batch, noisy_latents, timesteps)` with per-scale residual format and writer/reader | TBD | not-started | medium | Document tuple-of-tensors residual format and add writer helper.
| 8 | Add `control_training_schedule` hooks (support `standard|three_stage|ping_pong`) and make ping-pong scheduler available | TBD | not-started | medium | Facilitates GLYPH-SR/FrameDiffuser-style experiments.
| 9 | UI: Add *Generate on-the-fly* default and optional *Cache control images* checkbox + backend endpoint to run `tools/gen_control.py` in background | TBD | not-started | high | Must include progress/cancel and resume semantics when caching large datasets.


## 🧪 Tests & Verification (add and run)

| # | Test | Target files | Status | Run in CI? |
|---:|------|-------------|--------|-----------:|
| T1 | `augment_align_test` (bit-identity after bucket/resizing) | dataloader, `toolkit/pose.py` | not-started | yes |
| T2 | `control_cache_idempotence` (cache CLI idempotence & atomic manifest writes) | `tools/gen_control.py` | not-started | yes |
| T3 | `residual_shapes_test` (multi-scale residual formats) | `toolkit/controlnet_offload.py` | not-started | yes |
| T4 | `optimizer_param_test` (adapter train flag influences optimizer groups) | `jobs/process/BaseSDTrainProcess.py` | not-started | yes |
| T5 | `masked_recon_aux_loss_test` (LumiCtrl-style masked recon) | `toolkit/controlnet_aux.py` | not-started | yes |
| TG1 | `swap_correctness_test` (GPU manual) | `testing/test_controlnet_offload_gpu.py` | not-started | no (manual) |
| TG2 | `ddp_safety_test` (GPU manual) | `testing/test_controlnet_offload_gpu.py` | not-started | no (manual) |


## ⚙️ Medium priority (improvements & tooling)

- Add `tools/benchmark_offload.py` extension to measure per-scale transfer times and recommend `residual_storage` strategy (gpu vs cpu_pinned).
- Add example configs to `config/examples/`: `controlnet_openpose_train.yml`, `controlnet_lumictrl.yml`, `controlnet_pingpong.yml`.
- Add documentation & short how-to for using Z-Image ControlNet safetensors (8-step distilled weights) and conversion notes in `ControlTrain-Design.md` (already updated).
- Add `control_compat_metadata` saving in checkpoint manifest (`aitk_meta.yaml`).


## 📌 Manual / Operational tasks

- Manual GPU tests: schedule a maintainer to run `swap_correctness_test`, `swap_memory_smoke_test`, and `ddp_safety_test` on a GPU host with `accelerate` configured.
- Verify `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors` compatibility using our `controlnet_compat` loader and record `converter_version` in the manifest.


## 📍 Progress rules
- Before starting a new task, update this file and set `Status` to `in-progress` for that task and `not-started` for others.
- When a task completes, mark it `completed` and add short notes with PR reference and test links.
- Keep each top-priority task limited in scope (small PRs) and include unit tests where possible.

---

_Last updated: 2025-12-30_
