# Chapter 06 — Pre-training setup (data, transforms, schedulers)

TL;DR ✅
- Dataset specification lives in job configs (`jobs/process/*`) and is normalized via `preprocess_dataset_raw_config` -> `DatasetConfig` (`toolkit/config_modules.py`).
- Datasets are materialized as `AiToolkitDataset` (`toolkit/data_loader.py`) which applies transforms, bucketing, resizing/cropping, and optional augmentations and caching.
- Prompt handling & tokenization: `FileItemDTO` + `TextEmbedding*` mixins save/consume cached prompt embeddings; encoding uses `sd.encode_prompt` which delegates to `toolkit/train_tools` encoders (XL, SD3, Flux, Pixart, Auraflow).
- Training schedulers:
  - LR schedulers are created via `toolkit/scheduler.py::get_lr_scheduler` and chosen by `TrainConfig.lr_scheduler`.
  - Diffusion/noise samplers are created via `toolkit/sampler.py::get_sampler` and configured by `TrainConfig.noise_scheduler` / model settings.

---

## Files & symbols examined 🔍
- **Config & job wiring**
  - `toolkit/config_modules.py` — classes: `DatasetConfig`, `TrainConfig`, `SampleConfig` (lines inspected: 1-400, 929-1400)
  - `jobs/process/BaseSDTrainProcess.py` — job-level parsing of `datasets`, creating `DatasetConfig` instances and passing them into dataloader construction (lines inspected: 140-220, 2640-2760)

- **Data loading, transforms & bucketing**
  - `toolkit/data_loader.py` — `AiToolkitDataset`, `ImageDataset`, `AugmentedImageDataset`, `get_dataloader_from_datasets` (lines inspected: 1-820)
  - `toolkit/dataloader_mixins.py` — many `*FileItemDTOMixin` classes plus `BucketsMixin` and `CaptionMixin` (lines inspected: 1-400, 2196-2440)
  - `toolkit/data_transfer_object/data_loader.py` — `FileItemDTO`, `DataLoaderBatchDTO` (lines inspected: 1-360)

- **Tokenization & prompt embedding**
  - `toolkit/train_tools.py` — `text_tokenize`, `text_encode`, `encode_prompts_xl`, `encode_prompts_sd3`, `encode_prompts`, plus SNR helpers (lines inspected: 1-766)
  - `toolkit/dataloader_mixins.py` — `TextEmbeddingFileItemDTOMixin`, `TextEmbeddingCachingMixin` (lines inspected: 2196-2440)
  - `toolkit/stable_diffusion_model.py` — `encode_prompt` (delegates to train_tools) and how pipeline uses custom samplers (lines inspected: 1120-1220, 3060-3160)

- **Schedulers & samplers**
  - `toolkit/scheduler.py` — `get_lr_scheduler` (dictates LR scheduler choices and params)
  - `toolkit/sampler.py` — `get_sampler` (maps sampler names to Diffusers scheduler classes and default configs)

- **Trainer orchestration**
  - `extensions_built_in/sd_trainer/SDTrainer.py` — resize-to-bucket helpers, precompute / preencoding flows, SNR-weight application and use of `sd.noise_scheduler` (multiple references)

---

## Step-by-step: How datasets are specified & consumed 🧭
1. **Specify datasets in job config**
   - Jobs include a `datasets` array (top-level per-process config). Example shape:
     - `config.process[0].datasets: [{"folder_path": "datasets/myset", "resolution": 512, "buckets": true, ...}, ...]`
   - Base process reads them via `raw_datasets = self.get_conf('datasets', None)` then normalizes via `preprocess_dataset_raw_config(raw_datasets)` to split multiple resolutions into individual entries.

2. **Create DatasetConfig**
   - Each entry becomes `DatasetConfig(**raw_dataset)` (`toolkit/config_modules.py`). Default values are applied in the constructor (common defaults: `type='image'`, `resolution=512`, `buckets=True`, `bucket_tolerance=64`, `random_crop=False`, `scale=1.0`, `num_repeats=1`, caching flags default `False` unless `cache_latents_to_disk` etc.).
   - Dataset-level keys include captions, control generation flags (`control_type`, `control_generate_on_the_fly`), augmentations (`augmentations` list / `augments` legacy), caching flags (`cache_latents`, `cache_text_embeddings`, `cache_control_contexts`), and sizing/crop flags (`resolution`, `random_scale`, `random_crop`, `square_crop`, `control_size`).

3. **Materialize datasets -> `AiToolkitDataset`**
   - In `get_dataloader_from_datasets` each `DatasetConfig` (type `image`) is turned to `AiToolkitDataset(config, batch_size, sd)`.
   - `AiToolkitDataset.__init__`:
     - Builds file list (folder or JSON manifest). Stores `resolution`, `random_crop`, `scale`, `num_frames`.
     - Builds an image `transform` typically: `ToTensor()` + `RescaleTransform()` (to [-1,1]); optional standardization for SDXL/SD1.5 models (see `NormalizeSDXLTransform` / `NormalizeSD15Transform`).
     - Creates `FileItemDTO` for each file which holds per-item metadata, lazy loading & cached artifacts.
     - Optionally applies `flip_x`/`flip_y` duplication, repeats via `num_repeats`.

4. **Bucketing, resize, and crop behavior**
   - `BucketsMixin.setup_buckets` assigns images to buckets via `get_bucket_for_image_size(width, height, resolution, divisibility=bucket_tolerance)`.
   - For each file, compute `scale_to_width/height` (area-preserving to reach bucket resolution), then set `crop_width/crop_height` to the bucket resolution. If `random_crop` is enabled, a random crop offset is chosen; otherwise central crop is used.
   - During training/precompute, `SDTrainer._resize_batch_to_bucket` and related helpers perform pad/crop or center-crop and ensure final tensors match expected bucket dims (preserving aspect ratio and padding to multiple of 16 when needed).
   - `DatasetConfig.full_size_control_images` lets the dataset supply full-size control images (used by control pipelines); otherwise control images are resized to buckets.

5. **Augmentations**
   - Two systems: legacy simple augments (`dataset_config.augments`) and Albumentations augmentations (`dataset_config.augmentations` as list of dicts). Albumentations are composed via `A.Compose` and applied in `AugmentedImageDataset` and `AugmentationFileItemDTOMixin`.
   - Note: If latents caching is enabled, augmentations are disabled (explicit warning and forced config change).

6. **Prompt handling & tokenization**
   - `FileItemDTO` uses `CaptionProcessingDTOMixin` to load captions from sidecar files (cfg `caption_ext`) or defaults.
   - `TextEmbeddingFileItemDTOMixin.get_text_embedding_path` hashes a deterministic dict including `caption`, `text_embedding_space_version`, optionally `control_path` and `dop_class` to derive unique cache paths under `_t_e_cache`.
   - `TextEmbeddingCachingMixin.cache_text_embeddings` encodes per-file prompt embeddings in bulk using `sd.encode_prompt(file_item.caption, control_images=...)` and saves to disk; it uses `sd.set_device_state_preset('cache_text_encoder')` to move encoders as needed.
   - `sd.encode_prompt(...)` delegates to model-specific encoders:
     - XL -> `train_tools.encode_prompts_xl`
     - SD3 -> `train_tools.encode_prompts_sd3`
     - SD-style -> `train_tools.encode_prompts`
     - Flux/Pixart/Auraflow -> specialized helpers
   - Tokenization/truncation options are controlled (truncate vs long prompts / max_length).

7. **Schedulers selection & configuration**
   - **Noise/diffusion scheduler (timesteps):** The sampling top-level scheduler (e.g., `flowmatch`, `ddpm`, `euler_a`, `dpmsolver`, etc.) is chosen by `TrainConfig.noise_scheduler` and/or when sampling (`get_sampler` in `toolkit/sampler.py`). The scheduler class is instantiated with default configs (`sd_config`, `flux_config`, etc.) and any overrides provided in job configs (kwargs). The trainer calls `noise_scheduler.set_timesteps(...)` and manipulates `sigmas`/`timesteps` for training steps (see `SDTrainer` for set/restore sequences).
   - **LR scheduler:** `TrainConfig.lr_scheduler` (string) maps to `toolkit/scheduler.py::get_lr_scheduler` which supports: `cosine`, `cosine_with_restarts`, `step`, `constant`, `linear`, `constant_with_warmup` or falls back to diffusers scheduling functions if provided. `lr_scheduler_params` is passed through (e.g., `total_iters`, `num_warmup_steps`). Important: `get_lr_scheduler` expects `total_iters` in kwargs and will alias to required param names expected by torch schedulers.

---

## Config keys & environment flags to note ⚠️
- Dataset keys (on `DatasetConfig`) of interest:
  - `resolution`, `scale`, `random_scale`, `random_crop`, `square_crop`, `buckets`, `bucket_tolerance`
  - `augments`, `augmentations`, `replay_transforms`
  - `cache_latents`, `cache_latents_to_disk`, `cache_text_embeddings`, `cache_control_contexts`, `cache_control_contexts_to_disk`
  - Control settings: `controls`, `control_type`, `control_generate_on_the_fly`, `control_precompute_control`, `control_size`, `control_cache_path`, `full_size_control_images`
- Train keys (on `TrainConfig`):
  - `noise_scheduler` (diffusion sampler family)
  - `timestep_type` / `timestep weighting` (used in UI & some models)
  - `lr_scheduler`, `lr_scheduler_params` (used by `get_lr_scheduler`)
  - `batch_size`, `steps`, `gradient_accumulation`, `dtype`, `xformers`, `attention_backend`
- Important environment / host flags:
  - Windows: `toolkit/data_loader.py` sets `num_workers=0` for native Windows (via `is_native_windows()`), otherwise uses dataset-configured `num_workers` and `prefetch_factor`.
  - Tokenizer behavior: `ZIMAGE_ALLOW_DUMMY_TOKENIZER=1` environment variable (or `ModelConfig.allow_dummy_tokenizer=True` in tests) can create a minimal dummy tokenizer for ZImage model loads (NOT recommended for real generation; used as testing workaround).

---

## Observability & debugging 🧰
- Logs printed during dataset initialization:
  - `Dataset: <path>` and `- Preprocessing image dimensions` output in `AiToolkitDataset.__init__`.
  - `Bucket sizes for <dataset_path>:` and per-bucket counts come from `BucketsMixin.setup_buckets` and are printed (useful for verifying distribution and expected resolutions).
  - `.aitk_size.json` placed next to the dataset folder records cached image sizes (useful to inspect/clean when size-based issues occur).
- Prompt / tokenization:
  - `TextEmbeddingCachingMixin.cache_text_embeddings` prints progress; saved embeddings live under `<image_dir>/_t_e_cache/*.safetensors`.
- Precompute & control contexts:
  - `SDTrainer` logs precompute steps (`[PRECOMPUTE] ...`) when generating/encoding control contexts or split prompts; check train logs for precompute messages and possible resizing failures.
- Debugging tips:
  - Reproduce `FileItemDTO` behavior in unit tests or via `scripts/quick_eval_sim.py` (fast sim of scheduler behavior and dataset pass-through).
  - When a dataset fails bucket assignment or resize, inspect the dataset `.aitk_size.json`, `FileItemDTO` per-file `width/height`, `crop_*` fields, and the printed bucket sizes.
  - For tokenizer issues (e.g., missing `apply_chat_template`), prefer adding a compatible tokenizer repo to `te_name_or_path`; use dummy tokenizer env var only as a last-resort test workaround.

---

## Open questions / TODOs & Suggested Tests 🧪
- TODO: Document and test the exact interaction between `control_precompute_control` and `control_cache_path` for precompute flows and streaming (edge conditions when precompute enabled but cache path missing).
- TODO: Add end-to-end tests that check that `DatasetConfig` resolution list splitting (via `preprocess_dataset_raw_config`) produces separate `AiToolkitDataset` instances and that each gets a bucket for the requested `resolution`.
- Tests to add:
  - Verify `cache_text_embeddings` saves to `_t_e_cache` with deterministic filenames given identical captions + controls.
  - Stress test `BucketsMixin` with many aspect ratios to ensure no files are lost or cropped in unexpected ways.
  - Test `get_lr_scheduler` input shapes and param aliasing (e.g., user provides `total_iters` vs scheduler expecting `T_max`/`T_0`).
  - ControlNet precompute integration test: verify `save_control_contexts` writes expected disk artifacts per `cache_control_contexts_to_disk` flag.

---

## JSON manifest — files & line ranges inspected 🧾

```json
{
  "files": [
    {"path": "toolkit/config_modules.py", "ranges": [[1,400],[929,1400]]},
    {"path": "toolkit/dataloader_mixins.py", "ranges": [[1,400],[2196,2440]]},
    {"path": "toolkit/data_loader.py", "ranges": [[1,400],[380,820]]},
    {"path": "toolkit/data_transfer_object/data_loader.py", "ranges": [[1,360]]},
    {"path": "toolkit/train_tools.py", "ranges": [[1,766]]},
    {"path": "toolkit/scheduler.py", "ranges": [[1,200]]},
    {"path": "toolkit/sampler.py", "ranges": [[1,400]]},
    {"path": "toolkit/stable_diffusion_model.py", "ranges": [[1120,1220],[3060,3160],[540,560]]},
    {"path": "jobs/process/BaseSDTrainProcess.py", "ranges": [[140,220],[2640,2760]]},
    {"path": "extensions_built_in/sd_trainer/SDTrainer.py", "ranges": [[28,120],[740,820],[1920,2120]]}
  ]
}
```

---

If you want, I can:
- Add a short diagram (flow) summarizing dataset -> FileItemDTO -> batch collate -> trainer flow ✅
- Propose concrete unit test implementations for items in the TODO section ✅

If this looks good I’ll commit `06-pretraining-setup.md` and provide the tests to add as PR suggestions. ✨
