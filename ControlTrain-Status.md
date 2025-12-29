# ControlTrain — Project Status

*Last updated: 2025-12-29*

This document tracks **current implementation state, recent updates, known issues, and next steps**.

---

## Status at a Glance

### Implemented ✅

* Canny control generation (precompute + on-the-fly)
* Canonical full-size control precompute + manifest
* Control tensor plumbing (dataloader → training loop)
* Frozen ControlNet default behavior
* Explicit channel & width projection shims
* Shim save/load & bake tooling
* ControlNet offload helpers (manual swap + CPU-pinned)
* UI integration for precompute & job toggles
* Example config: `controlnet_canny_train.yml`

### In Progress ⚙️

* Accelerate-based offload QA (GPU/DDP)
* Spatial interpolation fallback test coverage
* Deterministic adapter-loading test fixes

### Not Started ⬜

* Small end-to-end integration training test
* OpenPose control implementation

---

## Recent Updates

### 2025-12-27

* Added precompute manifest with canonical `full` entry
* Added optimizer param group tests
* Improved Playwright test stability and cleanup

### 2025-12-28

* Added rich runtime diagnostics
* Implemented automatic channel projection shim
* Documented and aligned VideoX-style shim policy

### 2025-12-29

* Implemented consumer-site projection embedding
* Added projection pre-hook safety net
* Standardized shim metadata and bake-on-save
* Updated loader to detect baked adapters
* Implemented encoder hidden-state reduction (`reduce='mean'`) with a sensible heuristic and added tests; encoder-related tests now pass locally
* Tightened projection wrapping heuristics to avoid over-wrapping; pre-hooks now only project tensors whose last-dim == encoder_dim to prevent UNet-internal projection errors (resolved local matmul failure and added unit tests)
* Identified ~14 remaining failing controlnet tests
* Implemented buffering of ControlNet residuals (default: buffer to CPU pinned memory) and added `apply_buffered_residuals` helper; this prevents premature GPU moves and reduces "tensors on different devices" failures. Unit tests added; next: one-batch GPU smoke forward to validate accelerate/manual swap paths.
* Hardened Projection/Consumer shim policy: runtime consumer shims are now opt-in (disabled by default), created shims are placed on the consumer module's device/dtype, and `install_projection_shim` will skip runtime insertion when a baked shim (`shim_meta.yaml`) is detected. Added unit tests for shim policy and device/dtype correctness.

---

## Known Issues

* Remaining controlnet unit tests failing due to outdated mocks
* Added consumer-side output shims: lazy 1x1 convs that map adapter residual channel dims to UNet expected channels when `controlnet.auto_output_shim` is enabled (default: true)
* Spatial interpolation fallback lacks direct unit assertions
* Accelerate offload path requires GPU QA

---

## Immediate Next Steps

1. Patch remaining tests to use central ControlNet loader helper
2. Add unit tests asserting spatially normalized residual shapes
3. Run one-batch GPU smoke forward with ControlNet enabled to validate buffering and offload behaviour (accelerate/manual swap)
4. Add minimal integration training test (tiny dataset)

---

## Developer Handoff Notes

Recommended workflow:

1. `pytest -k controlnet`
2. Patch remaining tests to stub `_cn_from_pretrained`
3. Re-run full suite
4. Validate one-batch GPU forward manually

Key files:

* `jobs/process/BaseSDTrainProcess.py`
* `toolkit/controlnet_shim.py`
* `tools/bake_shim_into_adapter.py`
* `testing/test_controlnet_*`

Acceptance for handoff:

* All controlnet tests green locally
* No network or local package resolution during tests
