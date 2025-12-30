# ControlNet Training Design (OpenPose-first)

## Summary ✅
This document specifies a clear, testable design to add ControlNet conditioning using OpenPose pose maps as the canonical control modality. Controls are produced at a single canonical point (after augmentations + bucket/resize, before VAE latents) to guarantee pixel-level alignment. Caching generated control maps is supported as an optional performance optimization only — correctness relies on the canonical generation point.

Purpose: enable robust, reproducible ControlNet training (LoRA/LoKr/ControlLoRA) with minimal changes to existing flows and with pragmatic compatibility for non-diffusers controlnets (safetensors), channel/width mismatches, and large datasets.

---

## Design principles 🎯
- Canonical generation: generate control maps from the exact pixels the model sees (after all spatial transforms and resizing, before VAE encoding).
- Determinism: all control generation must be deterministic and record the generator version and parameters in a manifest.
- Minimal invasive edits: reuse dataloader, batch DTOs, training loop, and adapter save/load; add small helper modules (pose generator, compatibility wrapper) rather than large rewrites.
- Reproducible compatibility: support safetensors via a compatibility wrapper and prefer deterministic resampling + 1×1 channel projection over retraining controlnets.

---

## What changes (high level)
- New helpers:
  - `toolkit/pose.py::make_openpose_map(...)` — canonical OpenPose map generator (heatmap/skeleton options).
  - `toolkit/controlnet_compat.py` — safetensors loader, resampler, channel normalizer, conversion helper.
- CLI utilities:
  - `tools/gen_control.py` — generate pose maps and optionally cache with atomic manifests.
  - `tools/apply_control_manifest.py` — safely apply cache manifest to dataset metadata.
- Dataloader:
  - Generate control maps at canonical point and attach `batch.control_tensor` (on `self.device_torch`).
  - If cache exists and is validated, load it; otherwise regenerate from canonical helper.
- Training loop:
  - Accept and propagate `control_tensor` into model forward (ensure CFG/unconditioned pass uses blank control appropriately).
- UI & backend:
  - Add fields in New Job UI, preview control maps, and endpoint(s) to enqueue control-generation jobs and apply manifests.
- Tests & verification:
  - Alignment check (bit-identity for canonical generation vs cached map with same params), cache idempotence, safetensors compatibility, channel projection correctness, residual/offload smoke tests.

---

## Canonical generation contract (must-follow)
- Execution point: after dataset augmentations and bucket resizing, before VAE encoding.
- Determinism requirements: fixed interpolation, antialiasing, no randomness unless seeded & recorded.

- **Control manifest (`control_manifest.json`) schema (recommended v1):**

```json
{
  "manifest_version": "1",
  "generator_version": "git-sha-or-tag",
  "params_hash": "sha256-of-generator-params",
  "generator_params": { "model": "openpose", "confidence": 0.3, ... },
  "generated_after_augmentations": true,
  "files": [
    {"source": "images/00001.jpg", "control_path": "pose/00001_pose.png", "sha256": "...", "size": [H,W]}
  ]
}
```

- **Semantics & validator contract:**
  - Write atomically: write to a temp file then `os.replace(temp, manifest)` to avoid partial state.
  - Validate manifests against the JSON Schema file `toolkit/control_manifest_schema_v1.json` before applying. Implement the validator `toolkit/control_manifest.py::validate_manifest(path, schema_path=None, verify_checksum_sample:int=0, allow_stale=False)` with this contract:
    - Loads manifest JSON and validates against the schema (default schema path `toolkit/control_manifest_schema_v1.json`).
    - Computes canonical `params_hash` from `manifest['generator_params']` using a stable serialization (sorted keys, separators=(',',':')) and compares to the manifest `params_hash`. If mismatch and `allow_stale==False`, the validator returns `(False, 'params_hash mismatch: expected <computed> != manifest <value>')`.
    - If `verify_checksum_sample>0`, the validator samples up to `verify_checksum_sample` entries from `files` and verifies that the referenced control files exist and their `sha256` matches the manifest entry. Any mismatch causes the validator to return `(False, 'checksum mismatch for <file>')`.
    - On success returns `(True, None)`; on failure returns `(False, error_message)`. The caller should surface the error to the user and refuse to apply the manifest unless `allow_stale==True` is set explicitly by an operator with clear logging.
  - `params_hash` semantics: `params_hash = sha256(canonical_json_bytes(generator_params))` where `canonical_json_bytes` uses sorted keys and no extra whitespace. This stable hashing ensures identical logical parameter sets produce identical hashes regardless of formatting.
  - On job enqueue or dataset onboarding, use `validate_manifest(..., verify_checksum_sample=MAX(SOME_SMALL, N))` to detect stale or corrupted caches. If `params_hash` mismatches, prefer re-generation unless `--force`/`--apply-stale` is specified and logged.

- Manifest format: JSON schema v1 (records global params and per-file entries: control path, file checksum, generator params, `generated_after_augmentations` boolean). The schema must include types and required fields: `manifest_version` (int), `generator_version` (string), `params_hash` (string), `generator_params` (object), `generated_after_augmentations` (bool), and `files` (array of objects with `source`, `control_path`, `sha256`, `size`).
- Validation helper: add `toolkit/control_manifest.py::compute_params_hash(generator_params)` returning hex sha256 string and `validate_manifest` as above; callers should rely on `(True, None)` / `(False, error)` contract and propagate errors to users/ops when validation fails.

---

## ControlNet compatibility & conversion
- Load strategy:
  - Try to load as Diffusers `ControlNetModel` when available.
  - If only safetensors provided, `controlnet_compat.load_controlnet_checkpoint` should load weights and either wrap them in a thin adapter exposing the expected forward signature or convert to a Diffusers-compatible artifact.
- Channel mismatch (e.g., 4 → 16): use `normalize_control_channels` (deterministic 1×1 conv projector). Persist projector weights in adapter metadata for reproducibility and record metadata keys `{ "original_format": "safetensors", "channel_map": "1x1conv", "resampled_to": [W,H], "converter_version": "git-sha-or-tag" }` alongside the adapter or manifest so conversions are fully reproducible.
- Resolution mismatch (e.g., width 1280 → pipeline 320): resample post-bucket/resizing (deterministic antialiased resample) before producing control maps or feeding adapter; record `resampled_to` in metadata.
- Prefer using distilled control variants (e.g., Z-Image 8-step distilled versions) when available to preserve speed/quality tradeoffs; track `converter_version` (conversion or adapter wrapper version) in manifest or artifact metadata.
- Test fixtures & coverage: add synthetic safetensors/diffusers fixtures under `testing/fixtures/` (e.g., `controlnet_safetensors_synthetic.safetensors`, `controlnet_diffusers_synthetic/`) and unit tests that assert loader behavior and metadata keys are written (see related test files below).

### Suggested helper signatures (proposed API)
```python
# toolkit/pose.py
def make_openpose_map(image, model='openpose', confidence=0.3, output_format='heatmap', size=None, channels=None, dtype=None, generator_version=None, return_type='pil', return_meta=False):
    """Canonical OpenPose generator used by both the dataloader (on-the-fly) and `tools/gen_control.py`.

    Signature:
      - `image`: PIL.Image or ndarray (H,W,C)
      - `model`: str, choice of pose backend (`openpose`, `movenet`, `mediapipe`)
      - `confidence`: float, minimum keypoint confidence
      - `output_format`: 'heatmap'|'skeleton'|'keypoints' (controls output representation)
      - `size`: (W,H) or int, optional override of output size (deterministic resize)
      - `channels`: int, desired number of output channels (e.g., 1,3,4)
      - `dtype`: desired return dtype when `return_type=='tensor'` (e.g., `torch.float32`)
      - `generator_version`: string tag or git-sha to record the generator implementation/version
      - `return_type`: 'pil'|'ndarray'|'tensor'
      - `return_meta`: when True, return a second object with metadata (dict)

    Return semantics:
      - If `return_type=='pil'`: returns `PIL.Image` of mode `L` (single heatmap) or `RGB`/`RGBA` as appropriate. Pixel values are 0-255 (uint8).
      - If `return_type=='ndarray'`: returns `np.ndarray` shaped (H,W,C) dtype `uint8` with values 0-255.
      - If `return_type=='tensor'`: returns `torch.FloatTensor` shaped (C,H,W) with dtype `dtype` and values normalized to [0.0, 1.0] (documented). The caller must move this tensor to the desired device.
      - If `return_meta==True` a tuple `(control, meta)` is returned where `meta` is a dict containing at minimum `{ 'generator_version', 'generator_params', 'params_hash', 'keypoints': [...], 'sha256': '<hex>' }`.

    Determinism & reproducibility:
      - The implementation MUST be deterministic: fixed resample filter, antialiasing enabled, and no randomness unless a seedable RNG is explicitly passed and recorded in `meta`.
      - `params_hash` is computed as `sha256` over a canonical JSON encoding of `generator_params` (sorted keys, no whitespace). The function should include `params_hash` in returned `meta` when `return_meta` is requested.

    Usage notes:
      - The dataloader and `tools/gen_control.py` MUST call this helper at the canonical point (after augmentations and bucket/resize) to guarantee alignment.
    """

# toolkit/controlnet_compat.py
def load_controlnet_checkpoint(path):
    """Load a ControlNet checkpoint (Diffusers or safetensors) and return a wrapper object.
    The wrapper should expose a predictable forward signature and metadata accessors (e.g., .meta).
    """

def convert_safetensors_to_diffusers(in_path, out_path):
    """Convert a safetensors checkpoint to a Diffusers-compatible folder/artifact on disk and return metadata dict."""

def resample_control_image(control_image, out_size, resample_mode='lanczos', antialias=True):
    """Deterministically resample a control image to `out_size` with antialiasing."""

def normalize_control_channels(control_image, out_channels):
    """Apply a deterministic 1x1 projector to normalize channels to `out_channels`.
    Returns (projected_image, projector_weights) for reproducibility.
    """

# Offload & residual helpers (toolkit/controlnet_offload.py or controlnet_compat helpers)
def compute_control_residuals(adapter, batch, noisy_latents, timesteps) -> List[torch.Tensor]:
    """Compute detached per-scale residual tensors for a batch.

    Contract & ordering:
      - Returns a `List[Tensor]` ordered **coarse -> fine** (smallest spatial resolution first, largest last).
      - Each tensor shape: `[batch, C, H, W]` where H/W correspond to that residual's spatial resolution.
      - Residuals MUST be detached, on CPU or GPU according to `controlnet.residual_storage`, and have `requires_grad=False`.
      - The API consumer (e.g., training loop) must apply CFG duplication as needed to match UNet expectations.

    Example: for a 3-scale adapter, `residuals[0]` is the bottleneck residual (coarse, smallest H/W) and `residuals[-1]` is the finest scale.
    """

def pack_residuals(residuals: List[torch.Tensor]) -> Dict:
    """Pack per-scale residuals into a serializable dict for disk storage.

    Output format:
      {
        'meta': { 'scales': n, 'shapes': [[C,H,W], ...], 'dtype': 'float32', 'version': 1 },
        'scales': [bytes_of_tensor0, bytes_of_tensor1, ...],
        'checksum': '<sha256-of-packed-data>'
      }

    The writer should save atomically and include a checksum for quick validation.
    """

def unpack_residuals(packed: Dict) -> List[torch.Tensor]:
    """Validate `packed` format and return a list of tensors (coarse->fine). Raises ValueError on mismatch.

    Validation steps:
      - Check `meta.version` and shapes.
      - Verify `checksum` matches the packed bytes.
      - Decode tensors and return them as `torch.Tensor` objects in coarse->fine order.
    """

def offload_adapter(adapter, strategy='accelerate', **kwargs):
    """Offload an adapter according to `strategy` (accelerate|memory_manager|manual_swap|none).
    Implementations should prefer Accelerate dispatch APIs to ensure DDP and compiled-model safety.
    """

def bring_adapter(adapter, strategy='accelerate', **kwargs):
    """Bring an offloaded adapter back to required device for computation. Should be a no-op if adapter is already on device."""```

---

## UI & backend changes
- UI (`ui/src/app/jobs/new`): add `Use ControlNet (OpenPose)` (default), `control_type` dropdown, `pose_model` selector (openpose|movenet|mediapipe), `confidence_threshold`, `use_heatmaps`, `cache enabled` flag, and a preview component with sample image / pose map.
- Job endpoints:
  - POST `/api/control_gen` — enqueue control generation job with params.
  - GET `/api/control_gen/{id}/status` — job progress, manifest link.
  - POST `/api/control_gen/{id}/apply` — apply manifest to dataset metadata (runs `tools/apply_control_manifest.py`).
- Worker: `ui/cron/actions/processControlGenQueue.ts` runs `python tools/gen_control.py` and updates DB job statuses and progress.
- DB: add `control_manifest_path`, `generator_version`, `control_params` fields to job row (or create `ControlGenJob` table). Keep schema minimal and backward compatible.

- **Migration example (SQL):** add `ui/db/migrations/20251230_add_control_manifest.sql` with either of the following (depending on chosen approach):
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
- **Apply endpoint semantics:** `POST /api/control_gen/{id}/apply` — body: `{ "apply": true }` → server MUST validate the manifest via `toolkit/control_manifest.py::validate_manifest(manifest_path)` and, if valid, atomically update the `datasets` table (or job row) with `control_manifest_path` and `control_params` within a DB transaction; if invalid, return a 400 with validator error message and do not update DB.

---

## CLI changes
- `tools/gen_control.py` usage examples:
  - `python tools/gen_control.py --dataset datasets/myset --pose-model openpose --confidence 0.3 --size 512 --chunk-size 256 --generate-after-augmentations --overwrite --output-dir datasets/myset/pose --cache --batch-size 256`
  - CLI writes `datasets/myset/pose/control_manifest.json` atomically when `--cache` used and **prints machine-readable JSON on success**: `{ "manifest_path": "datasets/myset/pose/control_manifest.json", "params_hash": "<sha256>" }` to stdout.

- Recommended CLI flags and semantics:
  - `--dataset` (path): dataset root or named dataset
  - `--pose-model` (openpose|movenet|mediapipe): which pose model to run
  - `--confidence` (float): minimum keypoint confidence to keep
  - `--size` (int or WxH): output size for control images (overrides dataset bucket size when specified)
  - `--chunk-size` (int): number of images processed per worker chunk
  - `--generate-after-augmentations` (flag): generate control images from post-augmented images (canonical behavior)
  - `--overwrite` (flag): overwrite existing cached control files/manifest entries
  - `--output-dir` (path): where to write control maps and the manifest (defaults to `<dataset>/pose/`)
  - `--cache` (flag): enable writing control images to disk
  - `--batch-size` (int): internal processing batch size for model inference

- `tools/apply_control_manifest.py` to update dataset-level metadata safely. Recommended flags: `--force` (apply manifest despite minor mismatches), `--noop` (report what would change), `--dry-run` (validate manifest but do not apply). The tool should use `toolkit/control_manifest.py::validate_manifest` before applying and return clear exit codes (0 success, non-zero on validation/apply error).

---

## Tests & verification
- Unit & integration test files (explicit):
  - `testing/control_manifest_test.py` — schema validation, `params_hash` semantics, atomic write, and sampled checksum verification; tests validator error messages and CLI `--dry-run` behavior.
  - `testing/augment_align_test.py` — ensure generation after augmentations yields pixel-perfect alignment with bucketed images and that cached variants with `generated_after_augmentations=true` are accepted.
  - `testing/test_controlnet_compat.py` — safetensors/diffusers loader, conversion (`convert_safetensors_to_diffusers`), resampling (`resample_control_image`) and projector (`normalize_control_channels`) correctness, and metadata keys (`original_format`, `channel_map`, `resampled_to`, `converter_version`).
  - `testing/test_convert_safetensors_to_diffusers.py` — conversion helper round-trip tests and metadata recording.
  - `testing/test_control_cache_idempotence.py` — ensure cache writes are atomic and idempotent and `params_hash` mismatch behavior is correct.
  - `testing/test_make_openpose_map.py` — validate shapes & deterministic outputs across params and `generator_version` recording.
  - `testing/optimizer_param_test.py` — checks that ControlNet params are excluded from optimizer when frozen and included when `adapter.train=True` and that metadata flags are recorded on checkpoint.
  - `testing/swap_correctness_test.py` — verify swap/offload correctness (accelerate vs manual_swap) with numerical closeness asserts.
  - `testing/swap_memory_smoke_test.py` — memory smoke tests for residuals & swap throughput.
  - `testing/ddp_safety_test.py` — DDP run (manual) asserting offload strategy works or fails with a clear actionable message.
  - `testing/precompute_idempotence.py` — ensure precompute is resume-safe and manifest writes are atomic.
  - `testing/test_db_migrations.py` — run the example migration against an in-memory sqlite DB and validate new columns/table and default values.
  - `testing/integration_small_train_openpose.py` — short end-to-end run: generate controls + train with ControlNet frozen by default.
- Verification checks during dataset onboarding:
  - Bit-identity test on sample set comparing generated control vs cached map (if cache exists and `params_hash` matches); mismatches produce a clear report and a suggested remediation.

---

## Risks & mitigations
- CPU overhead for pose generation: mitigate with optional caching and recommend caching for very large datasets; provide benchmarks and sampling-based heuristics to advise users.
- Alignment errors: canonical generation point prevents most; add dataset onboarding verification and the `augment_align_test` to catch regressions.
- DDP & offload complications: prefer Accelerate dispatch APIs and add manual DDP safety tests; fail clearly when environment unsupported.

- **Config knobs**:
  - `controlnet.offload_strategy` (enum): `accelerate` | `memory_manager` | `manual_swap` | `none` — default `accelerate` when supported, `manual_swap` only as fallback.
  - `controlnet.residual_storage` (enum): `gpu` | `cpu_pinned` — controls where per-scale residuals are kept during training to trade memory vs perf.

- Conversion mistakes (safetensors → diffusers): persist converter metadata and provide conversion helpers and human-run conversion scripts for maintainers.

---

## Interaction with literature & VideoX‑Fun
- LumiCtrl: add `controlnet.aux_loss` hook (masked reconstruction) to support experiments.
- DEMIST: support multi-scale residual formats via `compute_control_residuals` and per-scale storage policies.
- FrameDiffuser/GLYPH‑SR: add `control_training_schedule` to support multi-stage and ping-pong strategies for experiments.
- VideoX‑Fun: follow their pragmatic pattern: accept safetensors, use resampling & channel projection, and prefer distilled 8-step control variants for Z-Image pairing.

---

## Prioritized implementation checklist
1. Add `toolkit/pose.py` canonical helper + docs + unit test (low)
2. Add `tools/gen_control.py` + `tools/apply_control_manifest.py` CLI stubs + cache manifest schema (low)
3. Implement `toolkit/controlnet_compat.py` (safetensors loader, resampler, `normalize_control_channels`) (medium)
4. Wire dataloader to call `make_openpose_map` at canonical point and add batch field (medium)
5. Add UI fields + `/api/control_gen` endpoint + worker (high)
6. Add Accelerate-based offload & residual helpers + manual GPU tests (high)

---

## Acceptance criteria ✅
- Controls are generated deterministically from the post-augmentation image and produce `batch.control_tensor` for each batch.
- Cache manifests are atomic and idempotent; dataset onboarding validates sample bit-identity.
- Non-diffusers controlnets (safetensors) can be loaded and adapted with projector/resample patterns documented.
- UI can enqueue & apply control generation jobs and persist the necessary job metadata.

_Last updated: 2025-12-30_
