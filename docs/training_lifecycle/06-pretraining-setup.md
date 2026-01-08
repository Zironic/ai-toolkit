# 06 — Pre-training Setup (Data, Transforms, Schedulers) ✅

TL;DR
- Dataset configuration is provided via `config` files and parsed into `DatasetConfig` objects (`toolkit/config_modules.py`). The data pipeline builds `AiToolkitDataset` instances (in `toolkit/data_loader.py`) which apply resizing, bucketing, augmentation, caption handling, and optional latent caching. Scheduler and optimizer choices come from `TrainConfig` (`toolkit/config_modules.py`) and are wired into the trainer/optimizer setup in the trainer code (`SDTrainer` / `DiffusionTrainer`).

Files & symbols referenced
- `toolkit/dataloader_mixins.py` — image/video processing, bucket creation, caption handling, augments, tiling
- `toolkit/config_modules.py` — `DatasetConfig`, `TrainConfig`, `SampleConfig`, `SaveConfig` (key defaults and toggles)
- `toolkit/data_loader.py` — builds `AiToolkitDataset` and the DataLoader, handles caching/epoch setup and collate
- `toolkit/train_tools.py` — dtype utilities and helper functions for training setup
- `extensions_built_in/sd_trainer/SDTrainer.py` — uses dataset and scheduler settings in training hooks

Step-by-step: dataset → transforms → scheduler
1. Job config lists datasets under `config.datasets` or in job/process configs; `toolkit/config.get_config()` and `DatasetConfig` parse dataset options and defaults.
2. `get_dataloader_from_datasets()` is called (usually by a process), which converts dataset dicts into `DatasetConfig` instances and then into `AiToolkitDataset` objects.
3. `AiToolkitDataset` constructs a `transform` pipeline:
   - If `standardize_images` is set → `ToTensor()` + `RescaleTransform()` + `NormalizeSD*` (SD1 vs SDXL selection)
   - Else → `ToTensor()` + `RescaleTransform()`
   - Augmentations via `albumentations` (`dataset_config.augmentations`) are supported (but not with cached latents) and are applied during item load.
4. Bucket logic:
   - `setup_buckets()` in `dataloader_mixins` groups images into buckets by resolution using `get_bucket_for_image_size()` and `bucket_tolerance`. Each bucket yields batches of similarly sized images to avoid heavy padding and improve throughput.
   - If `square_crop` is set, dataset scales to match long side and central crops to `resolution`.
5. Caption handling & tokenization:
   - Captions are read from sidecar files (`.txt`, `.json`) or default prompts. `CaptionMixin.get_caption_item` supports replacements, defaults and short captions.
   - Tokenization and prompt embeddings are handled by `toolkit/prompt_utils.py` and `PromptEmbeds` helpers (embedding caching optional via `DatasetConfig.cache_text_embeddings`).
6. Scheduler & optimizer selection:
   - `TrainConfig` fields (`lr_scheduler`, `lr_scheduler_params`, `optimizer`, `optimizer_params`, `noise_scheduler`) control which schedulers are instantiated by trainer helpers in `SDTrainer` and `toolkit/train_tools.py`.
   - `gradient_checkpointing`, `mixed_precision`, `dtype`, `xformers`, and attention backend toggles are read from `TrainConfig` and applied during model initialization (e.g., enabling gradient checkpointing on modules, setting torch dtype, turning on xformers attention).

Important config keys & env flags
- Dataset keys: `dataset_path`, `resolution`, `buckets`, `bucket_tolerance`, `random_crop`, `standardize_images`, `cache_latents`, `cache_clip_vision_to_disk`, `control_precompute_control`.
- Training keys: `steps`, `batch_size`, `lr`, `optimizer`, `lr_scheduler`, `gradient_checkpointing`, `dtype`, `xformers`, `attention_backend`.
- Env vars: none mandatory for data, but `DEBUG_TOOLKIT` affects torch anomaly detection and `AITK_DEBUG_ASYNC_DB` enables DB async debug logging.

Observability & debugging
- Dataset stats: `AiToolkitDataset` prints dataset path and counts, and `setup_buckets()` prints bucket sizes (see `dataloader_mixins.setup_buckets`).
- Size DB: `AiToolkitDataset` writes `.aitk_size.json` introducing versions for size caching — inspect that to verify pre-processing.
- Per-item errors: dataset constructor prints problematic files and increments `bad_count` rather than silently failing.
- For scheduler/optimizer problems: look at `SDTrainer` logs during `hook_before_train_loop` and `hook_before_model_load` where learning rate schedulers and optimizer params are instantiated.

Open questions / TODOs
- Add a unit test verifying that `augmentations` + `cache_latents` is prevented (current code warns and disables caching). Suggested test file: `testing/test_dataset_cache_augment_conflict.py`.
- Add a smoke test to validate bucketization for a synthetic set of images covering multiple sizes and ensure bucket counts match expected buckets.
- Add an explicit log at scheduler creation to emit scheduler type and parameters for easier debugging (`SDTrainer` where scheduler is constructed).

Manifest (files read)
- `toolkit/dataloader_mixins.py` (lines 1–400)
- `toolkit/data_loader.py` (lines 340–760)
- `toolkit/config_modules.py` (lines 920–1080)
- `toolkit/train_tools.py` (selected lines for dtype and helpers)
- `toolkit/prompt_utils.py` (prompt embedding handling)

Notes
- `controlnet` / `generate_control_on_the_fly` interplay is delicate; `validate_control_dataset()` aborts early if inconsistencies appear — add clear tests for common misconfigurations.
