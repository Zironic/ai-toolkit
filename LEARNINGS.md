Mask Workflow Implementation — 2026-01-10

Summary:
- Replaced Masked Reconstruction feature with enhanced mask workflow using existing infrastructure.
- Removed `toolkit/masked_recon.py` (690 lines) and all related config/UI/trainer code.
- Added `mask_strength` parameter (0.0-1.0) to DatasetConfig for configurable mask blending.
- Created `scripts/generate_masks_sam2.py` using facebook/sam2.1-hiera-tiny from HuggingFace for automated mask generation.

Files changed:
- `toolkit/config_modules.py` — removed masked_recon_* config fields, added mask_strength to DatasetConfig
- `extensions_built_in/sd_trainer/SDTrainer.py` — removed _compute_and_apply_masked_recon_loss and related code, added mask_strength blending formula in mask_multiplier computation
- `ui/src/app/jobs/new/SimpleJob.tsx` — removed Masked Reconstruction UI section
- `ui/src/app/jobs/new/jobConfig.ts` — removed masked_recon defaults
- `scripts/generate_masks_sam2.py` — new CLI tool for SAM2-based mask generation (supports point/box prompts)
- `testing/test_mask_strength.py` — comprehensive unit tests for mask_strength blending
- `docs/MASK_WORKFLOW_IMPLEMENTATION.md` — complete implementation documentation
- `toolkit/masked_recon.py` — deleted (replaced by existing mask infrastructure)

Key discovery:
- 95% of requested mask functionality already existed in `MaskFileItemDTOMixin` (toolkit/dataloader_mixins.py lines 1620-1740)
- Existing infrastructure: mask_path loading, automatic resizing, loss multiplication, inverted_mask_prior
- Only needed to add: mask_strength parameter, SAM2 CLI script, and remove masked_recon code

Mask strength formula:
```python
# Blend masked (1.0) and non-masked (0.0) regions based on strength
# strength=1.0: non-masked regions get 0.0 weight (full masking)
# strength=0.0: all regions get 1.0 weight (no masking effect)
mask_multiplier = mask + (1-mask) * (1-strength)
```

SAM2 usage:
```bash
# Generate masks using point prompts
python scripts/generate_masks_sam2.py --dataset datasets/my_dataset --points "[[512,512]]" --output datasets/my_dataset/masks

# Generate masks using bounding box
python scripts/generate_masks_sam2.py --dataset datasets/my_dataset --box "[[100,100,400,400]]" --output datasets/my_dataset/masks
```

Notes & caveats:
- Breaking change: masked_recon_* config fields removed (users must migrate to mask_path + mask_strength)
- Backward compatible: existing mask_path, alpha_mask, invert_mask configs continue working unchanged
- Default mask_strength=1.0 maintains full mask effect (no change in behavior)
- SAM2 models require transformers library and GPU for efficient processing
- Masks are binary (0/255) PNG files stored in separate folder

Testing:
- Added 8 unit tests covering full/half/no strength, normalization, edge cases, and batch consistency
- All tests pass: `python -m pytest testing/test_mask_strength.py -v`
- Manual GPU smoke test recommended for end-to-end training validation

Implementation time:
- Original estimate: 20-33 hours (full reimplementation)
- Actual: ~4 hours (discovered existing infrastructure only needed enhancements)

Subagent use:
- Used grep_search extensively to find all masked_recon references and ensure complete removal
- Searched HuggingFace for SAM models (found SAM2.1 with 183K downloads, facebook/sam2.1-hiera-tiny)


MultiTrigger CSV DOP mapping — 2026-01-08

Summary:
- Implemented CSV-based trigger → DOP class mapping to support subject + props learning during Differential Output Preservation (DOP).

Files changed:
- `toolkit/prompt_utils.py` — added `parse_csv_list` and `normalize_caption_separators` helpers.
- `extensions_built_in/sd_trainer/SDTrainer.py` — added mapping infrastructure and applied replacements in DOP precompute (per-file caching) and runtime DOP prompt generation; added `_map_triggers_to_classes_in_text` helper.
- `testing/test_dop_multi_trigger.py` — unit tests covering parsing, mapping, overlapping triggers, mismatched lengths, case-sensitivity.
- `extensions_built_in/sd_trainer/MultiTrigger.md` — design & implementation plan (updated with implementation status).
- `docs/MultiTrigger.md` — user-facing examples and caveats.

Notes & caveats:
- Backwards compatible: single trigger/class behavior unchanged.
- When multiple triggers are provided, per-file DOP embeddings are generated and cached with the final replaced caption used as a cache key (to avoid collisions).
- `photomaker_pipeline` still rejects multiple trigger tokens; relaxing it is left as a follow-up due to increased risk.
- Default matching is exact and case-sensitive; a case-insensitive option can be added in a follow-up PR.

Testing:
- Added fast unit tests. Ran targeted tests locally; no regressions observed in the DOP unit tests run interactively. Please run the full test suite in CI and a manual smoke run on GPU for end-to-end verification of caching behavior.

Subagent use:
- Ran local search via Raptor Mini subagent to find references to triggers/DOP and identify code locations to edit. Summary included in PR description when submitting.


Cache hashing & atomic writes — 2026-01-08

Summary:
- Implemented content-aware cache helpers and applied safe, deterministic caching behavior for several cache types: text embeddings, latents, controls, and context caches.

Files changed:
- `toolkit/cache_utils.py` — new helpers: `compute_file_sha256`, `compute_param_digest`, `compute_combined_hash`, `cache_filename`, `atomic_write`, `find_cached_file`.
- `toolkit/prompt_utils.py` — `PromptEmbeds.save` now uses `atomic_write` to avoid partial writes.
- `toolkit/dataloader_mixins.py` — text/latent/context cache naming extended to include parameter digest + content digest; writes are atomic and loads check for hashed or legacy caches.
- `toolkit/control_generator.py` — control files are written with content-aware hashed filenames and `atomic_write`; legacy naming is still checked as fallback.
- `scripts/migrate_caches.py` — initial migration helper (dry-run / apply) to suggest renames from legacy to hashed filenames.
- `testing/test_cache_utils.py`, `testing/test_prompt_utils_cache.py` — tests added for helpers and atomic saves.

Notes & caveats:
- Hashing strategy: default `content` uses full SHA‑256 of source bytes; `stat`/`mixed` strategies can be added to configs if performance requires it.
- Legacy fallback implemented: a short-term compatibility step; migration script recommended for users with many existing caches.
- Concurrency: atomic writes + race re-checks prevent partial files and handle common races; we rely on `os.replace()` for cross-platform atomic replace semantics.

Testing:
- Added unit tests for new helpers and a test to validate `PromptEmbeds.save` atomic behavior. Recommend running targeted integration smoke tests on small datasets and a manual GPU run for end-to-end validation.

Subagent use:
- Used Raptor Mini to locate cache generator and loader sites to ensure patch coverage and identify concurrency/atomicity hot spots.

