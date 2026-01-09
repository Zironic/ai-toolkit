Mask Preview feature

Overview

- Mask Preview runs once per job when enabled and creates one mask PNG per dataset item.
- Optionally, an overlay (mask blended over the source image) is created; this is enabled by default.

Configuration

- `train.mask_preview_enabled` (bool): Enable the one-time preview run (default: False).
- `train.mask_preview_save_path` (str): Template for saving previews; supports `{job_name}` placeholder (default: `output/{job_name}/masks`).
- `train.mask_preview_overwrite` (bool): Overwrite existing files (default: False).
- `train.mask_preview_overlay` (bool): Create overlay PNG alongside the mask (default: True).

Behavior

- The trainer will attempt to collect dataset objects via `get_dataloader_datasets(self.data_loader)` and iterate their `file_list` entries.
- For each file item that has a `control_path` or control tensors, a mask is generated using the same pipeline used during training (`build_control_mask`), ensuring preview masks match training behavior.
- An `index.json` mapping source files to generated assets is written in the output folder.

Notes

- Mask previews are CPU-only and fail-safe: any errors during preview generation are logged but do not abort training.
- The previous per-step preview controls (`mask_preview_max_steps`, `mask_preview_samples_per_step`) were removed in favor of this simpler and deterministic one-run-per-job behavior.
