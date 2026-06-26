# Changes from upstream (ostris/ai-toolkit)

Branch: `faster-dop` diverged from `upstream/main`.
62 files changed, ~13 700 insertions, ~680 deletions.

---

## 1. Memory management (major rewrite)

Files: `toolkit/memory_management/manager.py`, `manager_modules.py`, `bounce_pool.py` (new), `checkpoint_autotuner.py` (new), `network_mixins.py`

### Smart FP8 inference path
Layers that are stored in FP8 on the GPU now execute their forward pass natively in FP8 (`_scaled_mm`) instead of dequantizing to BF16 first. This eliminates the temporary full-precision copy and roughly halves the VRAM spike per layer during inference. A separate compile-clean wrapper (`_fp8_linear_compiled`) is used when `torch.compile` is active so graph breaks do not occur.

### Pinned bounce-pool prefetcher
`bounce_pool.py` implements a pool of pinned host buffers used as staging areas for CPU→GPU weight transfers. Weights are pre-pinned once and reused across steps; the pool selects the closest-matching buffer by shape to minimize allocation overhead. This cuts the per-layer PCIe transfer overhead and eliminates the pageable-submit stall that was the dominant bottleneck in Krea2 offloading (~130 s/step).

### Offload execution trace
`manager_modules.py` gained an `_OffloadTrace` that records which layers are accessed in which order during a forward pass. After a warm-up period the trace is used to schedule prefetch operations ahead of need, overlapping PCIe transfers with GPU compute.

### Working-reserve controller (autotune)
The previous flat `headroom` parameter has been replaced with a dynamic **working-reserve** controller:
- Per-layer promote/demote primitives: the controller can move individual layers between CPU and GPU based on measured VRAM pressure.
- A WDDM-cliff-aware governing signal: on Windows, the driver reclaims VRAM in large chunks near the ~500 MB cliff. The controller targets the cohabitation high-watermark against this cliff rather than a fixed free-memory target.
- Sampling vs. training separation: the sampling-path reserve trusts the learned peak + 0.5 GiB pad; the flat 2 GiB floor is retained only for training (where backward pass peaks are harder to predict).
- A `checkpoint_autotuner.py` records stable-run VRAM peaks and seeds future runs.
- Design documented in `toolkit/memory_management/AUTOTUNE_PLAN.md`.

### Vocabulary rename
The env-var and internal naming was rationalised. Old names (`headroom`, `buffer_hard`, `target_free`, etc.) still work via backward-compatible aliases in `_ENV_ALIASES`.

| Old name | New name |
|---|---|
| `training_headroom_gb` | `training_working_reserve_gb` |
| `working_headroom_used_gb` | `working_reserve_used_gb` |
| `AI_TOOLKIT_*_HEADROOM_*` | `AI_TOOLKIT_*_WORKING_RESERVE_*` |

### Offload profiling
A per-layer timing/pinning profiler (`set_offload_profile_enabled`) can be enabled at runtime to emit a summary of transfer times, pin overhead, and peak cohabitation, useful for diagnosing slow offload configs.

---

## 2. Text-encoder skip (TE worker)

Files: `toolkit/te_cache_worker.py` (new), `toolkit/aux_embed_cache.py` (new), `toolkit/cache_utils.py` (new), `toolkit/text_encoder_fingerprint.py` (new), `toolkit/stable_diffusion_model.py`

### Problem
During training the text encoder (TE) and the transformer must both be in VRAM simultaneously for the first batch. On 12–16 GB cards with large models (Z-Image, Anima) this causes an OOM spike at startup.

### Solution: throwaway TE worker
A new entrypoint `toolkit/te_cache_worker.py` loads the job config with `te_only=True` (TE + tokenisers, no transformer, no VAE), encodes every embedding the trainer will need, writes them to disk, then exits. Because it is a separate OS process its VRAM is fully reclaimed before the trainer starts.

The trainer detects that all embeddings are already on disk (`aux_cache_is_ready()`, `dataset_text_embedding_cache_is_ready()`) and sets `skip_te = True`, causing the model loader to skip instantiating the TE entirely. TE and transformer never co-reside.

### Aux embed cache (`toolkit/aux_embed_cache.py`)
Dataset caption embeddings were already cached by `dataloader_mixins`. The new aux cache extends this to the embeddings that were previously held only in memory: blank/unconditional embed, trigger-word embed, and per-sample conditional/unconditional pairs used for inline sampling. Stored under `<save_root>/.te_cache/`.

### Cache invalidation (`toolkit/text_encoder_fingerprint.py`)
A config hash (model ID, prompts, trigger word, relevant hyperparams) is computed and stored in the cache manifest. A stale or mismatched manifest causes a full re-encode on the next TE-worker run.

---

## 3. DOP (data-order prediction) improvements

Files: `toolkit/dataloader_mixins.py`, `toolkit/prompt_utils.py`, `toolkit/data_transfer_object/data_loader.py`, `extensions_built_in/sd_trainer/SDTrainer.py`

### Multi-trigger support
The original DOP implementation supported a single `class_prompt`. The refactored version accepts multiple trigger→class replacement pairs (`build_dop_replacement_pairs`). Each pair defines a word or phrase in the caption that, when present, produces a transformed DOP caption via `apply_dop_replacements`. This allows a single dataset with multiple trigger words to produce correct per-sample DOP targets.

### DOP embedding cache
DOP-transformed captions are now embedded and cached to disk alongside the regular caption embeddings, using a `_dop_text_embedding_path` derived from the transformed caption content. This avoids re-running the TE on DOP captions when the TE worker has already computed them.

### DOP prior cache (`dop_prior_cache`)
The model's own predictions on the class prompt (the "prior") can be cached to disk per image. The cache key combines the latent-identity hash (so crop/scale/flip changes invalidate it) with a DOP config hash (class prompt, model, resolution, sample count, shift params). Stored under `<image_dir>/_dop_prior_cache/`. Requires `dop_resolution` to be set to a reduced resolution for speed.

### Single-backward mode (`dop_single_backward`)
Opt-in mode that performs only one backward pass per step (using the DOP loss only), reducing peak VRAM during training when both a regular loss and a DOP loss would otherwise be computed.

---

## 4. Krea2 model improvements

Files: `extensions_built_in/diffusion_models/krea2/krea2.py`, `src/mmdit.py`, `src/pipeline.py`

### torch.compile-clean FP8 forward
The original FP8 forward path had multiple graph breaks that caused `torch.compile` to fall back to eager mode, resulting in ~370 s/step on Krea2. The forward was rewritten to:
- Eliminate all Python-side branches inside the compiled region.
- Use a wrapper-free `_scaled_mm` call that compile can trace cleanly.
- Gate FP8 execution at the block level with a pre-compiled boolean so the branch is compile-time constant.

Result: ~10 s/step with `torch.compile`.

### Block-granular offload
Each transformer block has its own offload gate. Blocks that fit in VRAM stay resident; blocks that do not are streamed via the bounce pool. The boundary is determined at startup based on measured available VRAM after the KV cache and other fixed allocations.

### Streaming checkpoint load
Krea2's checkpoint is streamed directly from disk into each layer's target dtype/device without ever holding the full model in RAM. FP8 quantisation is applied tensor-by-tensor during the stream.

### Convergent sampling headroom
The sampler's working reserve starts conservatively and is decreased step-by-step as the measured peak stabilises, so later denoising steps use less reserved memory.

---

## 5. Z-Image flow-shift sampler

Files: `extensions_built_in/diffusion_models/z_image/z_image.py`, `scripts/vram_calc_zimage.py` (new)

### Shift distribution rewrite
Upstream's flow-match sampler for Z-Image produced a noise average of ~75% (heavily biased to high-noise timesteps). The sampler was rewritten with:
- `base_shift = 0.5`, `max_shift = 0.85` (vs upstream's higher values).
- `min_shift = 0.33` floor to prevent very-low noise sampling that produced blank/degenerate outputs.
- `use_dynamic_shifting = True` retained for resolution-dependent shift scaling.

The average sampled noise is now ~50%, matching the training distribution more closely.

### VRAM calculator
`scripts/vram_calc_zimage.py` estimates VRAM requirements for a given Z-Image config (resolution, batch size, offload settings) without loading the model.

---

## 6. Anima model

Files: `toolkit/models/anima.py`, `config/examples/train_lora_anima_24gb.yaml` (new)

Initial support for training LoRA adapters on Anima (CosmosTransformer3D backbone, Qwen3 text encoder + AnimaTextConditioner):
- 5D latent tensors (`[B, C, T, H, W]`) with `patch_size=[1,2,2]`.
- Manual CFG inference loop (Anima does not use the standard diffusers pipeline).
- TE-worker support: `skip_te` skips the Qwen3 VL model (~9.5 GB VRAM on load).
- Example 24 GB config at `config/examples/train_lora_anima_24gb.yaml`.

---

## 7. Training infrastructure

Files: `jobs/process/BaseSDTrainProcess.py`, `extensions_built_in/sd_trainer/SDTrainer.py`, `toolkit/timer.py`, `toolkit/print.py`, `toolkit/train_tools.py`

### Performance logging
A structured timing log is written to `<save_root>/performance_log.jsonl` as training progresses. Each record contains per-step timing windows (data load, forward, backward, optimizer, offload) and memory stats. On each new run the previous log is moved to `<save_root>/logs/{n}_performance_log.jsonl` so history is preserved.

### `digest_perf_log.py`
`scripts/digest_perf_log.py` reads a performance log and prints human-readable summaries: average step time broken down by phase, memory usage, and offload stats. Supports filtering by step range and verbosity.

### Timer utility
`toolkit/timer.py` gained a `Timer.split()` method for per-phase profiling within a training step.

### `print_acc` improvements
`toolkit/print.py`: progress-bar-aware printing (output is flushed above the bar rather than interleaved), optional log-to-file routing, and structured status lines.

---

## 8. New scripts and tools

| File | Purpose |
|---|---|
| `scripts/captions_to_json.py` | Export all `.txt` sidecar captions from a dataset folder to a single JSON file for bulk editing. |
| `scripts/captions_from_json.py` | Write edited captions back from a JSON file to `.txt` sidecars. |
| `scripts/generate_from_captions.py` | Run inference on every caption in a folder and save generated images alongside. |
| `scripts/search_replace.py` | Search-and-replace across all `.txt` sidecar files in a dataset. |
| `scripts/sim_working_reserve_controller.py` | Simulate the working-reserve controller against a recorded VRAM trace to tune parameters offline. |
| `scripts/vram_calc_zimage.py` | Estimate Z-Image VRAM requirements for a given config. |
| `tools/analyze_per_file_loss.py` | Parse a performance log and rank training images by their average loss, useful for finding outliers or mislabelled samples. |

---

## 9. UI changes

Files: `ui/src/`, `ui/src/components/TimestepDistributionSparkline.tsx` (new)

### Timestep distribution sparkline
A new `TimestepDistributionSparkline` component renders a live SVG preview of the timestep sampling distribution as the user adjusts shift parameters. The math mirrors `custom_flowmatch_sampler.py` so what you see in the UI matches the actual training distribution.

### DOP controls
`SimpleJob.tsx` gained UI fields for:
- `dop_resolution`: reduced resolution for prior cache generation.
- `dop_single_backward`: opt-in single-backward mode.
- `dop_prior_cache`: toggle disk caching of prior predictions.

### Documentation
`ui/src/docs.tsx` updated with documentation for all new config keys.

---

## 10. Ideogram4

Files: `extensions_built_in/diffusion_models/ideogram4/ideogram4.py`

Streaming checkpoint load and FP8 inference path, matching the Krea2 approach.

---

## 11. Miscellaneous

- **`toolkit/config_modules.py`**: new fields for `dop_resolution`, `dop_prior_cache`, `dop_single_backward`, `layer_offloading_checkpoint_keep_last`, `skip_te`, and others.
- **`toolkit/lora_special.py`**: improved handling of non-standard layer types in LoRA injection.
- **`toolkit/network_mixins.py`**: improved FP8/quantisation awareness when merging networks.
- **`toolkit/optimizers/automagic2.py`**: improved warmup schedule handling.
- **`toolkit/util/quantize.py`**: more robust handling of TorchAO quantised models.
- **`toolkit/assistant_lora.py`**: minor dtype fix.
- **`requirements_base.txt`**: added `safetensors` streaming dependency.
- **`.gitignore`**: added `MULTITRIGGER_DOP_REFACTOR_PLAN.md`, `.claude` (Claude Code memory directory).

---

## Test coverage added

| File | What it tests |
|---|---|
| `tests/test_bounce_pool.py` | Pinned bounce-pool alloc, shape matching, CPU-fill guard, multi-thread safety |
| `tests/test_offload_shape_key.py` | Offload shape-key hashing and cache invalidation |
| `tests/test_sampling_working_reserve.py` | Sampling-path reserve calculation with learned peak |
| `tests/test_wddm_deadband.py` | WDDM-cliff deadband logic |
| `tests/test_working_reserve_sim.py` | Working-reserve controller simulation |
| `testing/test_dop_dataloader.py` | DOP multi-trigger dataloader round-trip |
