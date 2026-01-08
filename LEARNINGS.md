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

