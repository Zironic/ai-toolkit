# Cache hashing and skip-generation plan (Production-grade) 🔧

## Overview & goals ✅
Purpose: Make cache generation deterministic, fail-fast, and robust so caches are reused whenever the source content and parameters are unchanged. We want resilient, state-of-the-art behavior with minimal runtime overhead and safe multi-process operation.

Primary goals:
- **Deterministic invalidation:** caches must change only when source *content* or *relevant generation parameters* change.
- **Fail-fast & explicit errors:** generators should raise clear errors on failure; partial writes must not leave corrupted caches.
- **Safe concurrency:** atomic writes, race-tolerant checks, and post-write re-checks to avoid lost updates or partial files.
- **Configurable performance:** provide `content` (SHA‑256) hashing by default and a `stat`/`mixed` fast mode when needed.
- **Testable & auditable:** unit tests, integration tests, and logs that make behavior observable.

Scope (caches in-scope):
- `_t_e_cache` — text embedding cache (hash the .txt file + params)
- `_latent_cache` — latent/image embedding cache (hash the source image bytes + params)
- `_controls` — control images derived from images (hash image bytes + control params)
- `_context_cache` — combined context (hash of image bytes + control bytes + params)

---

## Key design decisions (short) ✨
- Use **SHA‑256** for content hashing (chunked reads) for robustness.
- Preserve **parameter hashing** (deterministic digest of generation params) and include it in the filename to keep behavior compatible with current param-based invalidation. Use SHA‑256 for params but allow truncation for filename brevity.
- Use a **composite filename**: `{base}_{param_short}_{content_hex}.{ext}` so both parameter changes and content changes cause different filenames.
- All cache writes MUST use **atomic write** (tmp file → fsync → os.replace), and generators MUST re-check for existence before writing to avoid races.
- Provide **legacy fallback** and a `scripts/migrate_caches.py` tool (dry-run / apply) to help users migrate existing caches.

---

## Filename scheme & canonicalization 📁
Recommended canonical filename:

- `{base}_{params[:12]}_{content_fullhex}.{ext}`
  - `base`: deterministic id (dataset id, image basename, index)
  - `params[:12]`: first 12 hex chars of SHA‑256 over JSON-stable-serialized params (sorted keys, stable separators)
  - `content_fullhex`: full SHA‑256 hex digest of file bytes (or combined digest for combined sources)
  - `ext`: cache format (e.g., `.safetensors`, `.npy`, `.jpg`)

Rationale:
- Keeping a params prefix preserves current behavior and keeps filenames concise but unique for parameter changes.
- Using full content SHA‑256 avoids accidental collisions and ensures content-level invalidation.

Legacy handling:
- If hashed file not found, attempt a **legacy lookup** using existing heuristics (existing code uses param MD5 base64 in many places). The fallback is a temporary compatibility measure; prefer migration.

---

## Hashing strategies & configuration ⚡
Config option: `CACHE_HASH_STRATEGY` (default `content`)
- `content`: SHA‑256 over full file bytes (robust). Use chunked reads (default chunk: 1 MiB) to avoid memory spikes.
- `stat`: fast mode — use `(st_mtime_ns, st_size)` as digest input (fast but fragile to mtime preservation).
- `mixed`: optimize for large files — use `stat` plus a small sample prefix (first 4 KiB) hashed to detect content changes quicker than full reads but far more reliable than `stat` only.

Recommendation: default to `content` unless profiling shows unacceptable overhead; otherwise `mixed` is a safe compromise.

---

## Helper API — `toolkit/cache_utils.py` (must-have) 🔧
Add a compact, well-documented module with these functions (and unit tests):

- compute_file_sha256(path: str, chunk_size: int = 1<<20) -> str
  - Return full hex SHA‑256 of file bytes; read in chunks.
- compute_param_digest(params: Mapping[str, Any], length: int = 12) -> str
  - Stable JSON canonicalization (sorted keys, separators=(',',':')), SHA‑256, return first `length` hex chars.
- compute_combined_hash(paths: Iterable[str]) -> str
  - Compute SHA‑256 of concat(digest1||digest2||...) (deterministic order)
- cache_filename(base: str, param_digest: str, content_hex: str, ext: str) -> str
- atomic_write(path: str, write_fn: Callable[[Path], None], fsync: bool = True) -> None
  - Write to a `.tmp` in same dir, call `os.replace()` into final path, optionally `fsync` file and dir for durability.
- find_cached_file(cache_dir: Path, base: str, param_digest: str, content_hex: str, ext: str, legacy_fallback: bool = True) -> Optional[Path]
  - Return `Path` for hashed file if exists, else try legacy fallback names (configurable).
- migrate_caches(cache_dir: Path, dry_run: bool = True, dry_report: bool = True) -> None
  - Migrate legacy caches to hashed names (dry-run first).

Implementation notes:
- Use Python stdlib only (hashlib, json, os, pathlib). Add `typing` hints.
- Keep helpers small, testable, and deterministic.

---

## Integration / edit points (exact targets) 🛠️
Update the following files/functions in priority order (found via repo search). For each, insert: compute param_digest + content_digest, call `find_cached_file`, if found load & skip, otherwise generate, re-check existence and `atomic_write`.

High priority edits (do these first):
1. `toolkit/dataloader_mixins.py`
   - Functions: `get_text_embedding_path`, `cache_text_embeddings`, `get_latent_path`, `cache_latents_all_latents`, `get_control_context_path`, `save_control_contexts`
   - Notes: currently uses MD5(param) base64 in filenames; keep parameter digest behavior but add content SHA‑256 of source files (text file for text embeds, image for latents), and use `atomic_write` around `save_file`.

2. `toolkit/control_generator.py`
   - Functions: `_generate_control`, `get_control_path`
   - Notes: current names are deterministic by `file name + control type` and images are saved directly. Change to hashed filenames that include content digest of source image + control param digest. Keep legacy lookup for `file_name.type.jpg` for compatibility; write atomically.

3. `toolkit/prompt_utils.py`
   - Functions: `PromptEmbeds.save`, `PromptEmbeds.load`
   - Notes: wrap saves in `atomic_write` to avoid partial `.safetensors` writes; use `find_cached_file` before generating if caller can provide source param/content info.

4. `toolkit/clip_vision` & related caching code
   - Functions: `cache_clip_vision_to_disk`, `get_clip_vision_embeddings_path`
   - Notes: include content digest of image bytes; wrap writes atomically.

Medium/low priority:
- `tools/precompute_control.py` — produce controls with hashed names and atomic writes (manifest writing already atomic in some places).
- Any other places where `save_file`, `Image.save`, or `np.save` write cache artifacts; swap to `atomic_write`.

Suggested code snippet (insert near generators):

```python
# compute digests
param_digest = compute_param_digest(params)
content_digest = compute_file_sha256(src_path)
expected = Path(cache_dir) / cache_filename(base, param_digest, content_digest, ext)
# quick check
cached = find_cached_file(cache_dir, base, param_digest, content_digest, ext)
if cached:
    logger.info("cache hit: %s", cached)
    return load_cached(cached)
# generate expensive artifact
artifact = generate(...)
# re-check to handle a concurrent writer that raced while we generated
cached = find_cached_file(cache_dir, base, param_digest, content_digest, ext)
if cached:
    logger.info("concurrent cache created: %s", cached)
    return load_cached(cached)
# write atomically
def _writer(p: Path):
    save_impl(artifact, p)
atomic_write(expected, _writer)
return expected
```

Important: keep the write and subsequent `os.replace()` in same filesystem and same directory to make `os.replace()` atomic.

---

## Concurrency & atomicity details (robust) 🛡️
- Write to `<target>.tmp.<pid>.<random>` in same dir.
- After successful write: flush file buffer, call `os.fsync(fd)` (if `fsync=True`), close; then `os.replace(tmp, target)`.
- Optionally `fsync` the directory (best-effort; on Windows this is no-op but on POSIX `os.open(dir, O_DIRECTORY)` + `os.fsync` helps durability).
- On failure: clean up tmp file; raise a clear exception; do not corrupt or truncate the target file.
- After generating artifact but before writing, re-check `find_cached_file` to handle races where another worker completed earlier.

---

## Legacy fallback & migration strategy 🔁
- **Fallback behavior:** When hashed file not found and `legacy_fallback=True`, try existing legacy filename formats and param-based MD5 names (observed in repo). If an old file is found, consider copying/renaming to hashed format or load directly (documented behavior).
- **Migration tool:** `scripts/migrate_caches.py`
  - Modes: `--dry-run` (report mappings), `--apply` (rename/copy), `--verbose`.
  - For each legacy cache entry: compute matching content/param hashes and suggest new hashed filename.
  - Make migration idempotent and safe (do not overwrite hashed files unless `--force`).

Recommendation: ship migration tool with PR and document `--dry-run` output format.

---

## Tests & validation ✅
Unit tests:
- `testing/test_cache_utils.py` — hashing, combined hashing, param digest, atomic write behavior, error conditions.
- `testing/test_cache_legacy_migration.py` — dry-run/reporting behavior.

Integration tests (fast/mocked):
- `testing/test_cache_integration.py` — mock generators to assert skip-on-cache-hit, invalidation on content change, and context-cache invalidation when either input changes.
- Concurrency test (pytest + multiprocessing): simulate two processes racing to generate the same cache; assert final file is valid and no partial files remain.

Manual smoke test guidance (document in PR):
- Use a tiny dataset and run precompute to generate all cache types; re-run and confirm all caches are reused (log messages) and no generators invoked.

---

## Observability & logging 🕵️
Log events (structured if possible):
- `cache.hit` — INFO with `{cache_type, path, param_digest, content_digest}`
- `cache.miss` — INFO with `{cache_type, expected_path, param_digest, content_digest}`
- `cache.write.success` / `cache.write.error` — INFO / ERROR
- `cache.migration` — INFO when migrating a legacy file
- Debug logs: show short digests and decisions when `--cache-debug` enabled

Optionally emit metrics counters (cache_hits, cache_misses, cache_write_errors) for operational visibility.

---

## Acceptance criteria & rollout ✅
Before merging, ensure:
- Unit tests for `cache_utils` pass and have edge-case coverage.
- Integration tests show: cache skip works, invalidation on content/param change, context invalidation works.
- Concurrency test (mocked) passes (no partial files, no corruption).
- Migration tool dry-run produces expected mappings for existing caches; optional `--apply` tested for one small dataset.
- Documentation updated (`AGENTS.md`, `LEARNINGS.md`, `docs/cache.md`) with usage and limitations.

---

## Implementation tasks & estimates (ordered) 🗂️
1. **Add `toolkit/cache_utils.py` + unit tests** — implement hashing, param digest, atomic write, find_cached_file, migration helper (1–2h). **(Done)**
2. **Patch `toolkit/dataloader_mixins.py`** — compute digests, early-skip, re-check, atomic write for `_t_e_cache`, `_latent_cache`, and `_context_cache` (2–3h). **(Done for core flows)**
3. **Patch `toolkit/control_generator.py`** — produce hashed `_controls` files, add fallback legacy lookup (1–2h). **(Done)**
4. **Patch `toolkit/prompt_utils.py`** — wrap `save_file` with `atomic_write` and add callers to compute/find cache (0.5–1h). **(Done)**
5. **Add tests (integration + concurrency)** — mocked generator tests and small concurrency multi-process test (1–2h). **(Unit tests added; add integration concurrency test next)**
6. **Add `scripts/migrate_caches.py`** — dry-run & apply logic + tests (1–2h). **(Initial tool added; extend as needed)**
7. **Docs & PR** — update `AGENTS.md`, `LEARNINGS.md`, `docs/cache.md`, add PR description and manual verification steps (0.5–1h). **(Docs updated; `LEARNINGS.md` entry added)**

Estimated total: 6–14 hours depending on patch complexity and test coverage requirements.

---

## Implementation status & next steps ✅
Completed:
- `toolkit/cache_utils.py` and unit tests (`testing/test_cache_utils.py`).
- `PromptEmbeds.save` updated to use `atomic_write` and tested (`testing/test_prompt_utils_cache.py`).
- `toolkit/dataloader_mixins.py` updated to compute content digests and param digests for latent, text and context caches; generators now check `find_cached_file` and write atomically.
- `toolkit/control_generator.py` updated to produce content-hashed control files and write atomically; legacy filenames still supported as fallback.
- Initial migration helper `scripts/migrate_caches.py` added.

Next steps (optional but recommended):
- Add integration concurrency test (multiprocessing) that simulates two workers racing to create the same cache and ensures no partial files remain.
- Extend migration tool to map param-based legacy filenames into composite `{base}_{param}_{content}.{ext}` patterns where param can be inferred or kept as legacy token.
- Patch other remaining `save_file`/`img.save` sites that produce cache artifacts (e.g., clip vision cache) to use the same pattern.

If you want, I can open a branch & PR with these changes and include the test output, implementation rationale from `docs/cache.md`, and manual verification steps. Would you like me to prepare the PR now? (I will include a short migration note and smoke test instructions.)

---

## PR checklist ✅
- [ ] Add `toolkit/cache_utils.py` with tests
- [ ] Patch top priority cache generators and add unit/integration tests
- [ ] Add concurrency tests and document assumptions
- [ ] Add `scripts/migrate_caches.py` (dry-run + apply)
- [ ] Update docs (`AGENTS.md`, `LEARNINGS.md`, `docs/cache.md`)
- [ ] Add smoke test instructions and manual checks for reviewers
- [ ] Confirm no heavy GPU jobs run in CI; mark PR `manual-testing-required` where needed

---

## Example implementation snippets (robust)

Atomic write helper (concept):

```python
import tempfile
import os
from pathlib import Path

def atomic_write(target: Path, write_fn: Callable[[Path], None], fsync: bool = True):
    tmp = target.with_suffix(target.suffix + f'.tmp.{os.getpid()}.{next_temp()}')
    try:
        os.makedirs(tmp.parent, exist_ok=True)
        write_fn(tmp)
        if fsync:
            fd = os.open(tmp.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
```

Generator flow (final):

```python
param_digest = compute_param_digest(params)
content_digest = compute_file_sha256(src_path)
expected = cache_dir / cache_filename(base, param_digest, content_digest, ext)
cached = find_cached_file(cache_dir, base, param_digest, content_digest, ext)
if cached:
    logger.info("cache.hit %s %s", cache_type, cached)
    return load_cached(cached)
artifact = generate(...)
# race re-check
cached = find_cached_file(cache_dir, base, param_digest, content_digest, ext)
if cached:
    logger.info("cache.created_during_work %s", cached)
    return load_cached(cached)
# atomic write
atomic_write(expected, lambda p: save_impl(artifact, p))
return expected
```

---

If you'd like, I can start by implementing `toolkit/cache_utils.py` and unit tests, then patch `toolkit/dataloader_mixins.py` as a reference change in the same branch. Proceed with that? 💡
