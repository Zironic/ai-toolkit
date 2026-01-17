# Cleanup Plan: ControlNet / Z-Image / VideoX (and related areas) 🔧

Goal: Make the ControlNet / VideoX / Z-Image code paths clean, deterministic and maintainable by removing redundant heuristics, fallbacks, and misleading logic. Enforce strict Z-Image semantics when configured (`controlnet_mode=='zimage'`) and add clear tests and instrumentation so behavior is observable and correctly attributed in perf summaries.

---

## Principles
- Fail fast and loudly for Z-Image mode. No implicit heuristics.
- Keep non-Z-Image legacy paths working but clearly separated and documented.
- Centralize device/dtype/channel adaptation logic into small, well-tested helpers.
- Make timing explicit for all expensive adapter-related operations (so perf attribution is accurate).
- Add tests and a short migration doc for third-party adapters.

---

## High-level tasks
1. Audit and document all Z-Image/VideoX heuristics and fallbacks.
2. Add a small helper to detect Z-Image mode explicitly and use it to gate strict behavior.
3. Enforce strict checks for Z-Image mode (signature, channels, control_in_dim) and remove permissive shims in that branch.
4. Move/adapt existing fallback logic into a legacy compatibility wrapper/module (keep tests there).
5. Add timers/logging around Z-Image adapter calls to ensure perf attribution.
6. Add unit tests covering strict failures and success cases, plus shape/dtype/device invariants.
7. Update docs & `LEARNINGS.md` with rationale and migration notes.

---

## Files & code locations to inspect and clean (line references are approximate — search for key symbols if line numbers drift)

- extensions_built_in/sd_trainer/SDTrainer.py
  - Precompute and registration of Z-Image contexts: ~lines **560–610** — **Removed** (precompute support deprecated and trainer helpers cleaned).
  - `_collect_preencoded_zimage_context_for_batch`: ~**1860–1920** — **Removed** (deprecated and replaced with a compatibility stub in `SDTrainer`) 
    - Clean: simplify caching logic, clarify error behavior when precomputed contexts missing.
  - Adapter per-batch / encode adapter block (standard ControlNet forward): ~**3050–3135** (search for `with self.timer('controlnet_forward')` and adapter(adapter_images_dev))
    - Clean: ensure adapter forward is always timed and device/dtype safe.
  - Z-Image / VideoX routing branch: ~**3500–3725** (search for `adapter_uses_zimage`, `zimage_controlnet`, `zimage_control_images`, `CONTROLNET-REROUTE`) 
    - Clean: remove multiple heuristics and fallbacks. When in Z-Image mode, require the adapter to accept `control_context` and fail otherwise. Add `with self.timer('controlnet_zimage_forward'):` around the adapter call and any assembly steps moved outside the UNet forward.
  - Device/dtype and channel adapt helpers inside the Z-Image branch: ~**3720–3935** (many helper functions like `_move_to_device`, `_adapter_device_dtype`, casting and trimming channel adaptation) 
    - Clean: factor into small helpers in `toolkit/` and remove duplicated code and nested try/except noise.
  - Channel adaptation & trimming logic (grouped-mean, pad/truncate): ~**3888–3932** 
    - Clean: centralize and document exact semantics (no silent surprises), add unit tests.
  - Offload / bring adapter and controlnet offload timers: ~**4030–4055** 
    - Clean: ensure offload time is measured and that errors are surfaced consistently.
  - Perf/timing attribution spots (missing timers): `predict_unet` call area where `zimage_controlnet` is passed into `sd.predict_noise` (~**4200–4260**) 
    - Clean: add timers around zimage call path inside `sd.predict_noise` or ensure SDTrainer measures and reports it so ControlNet time is not swallowed by Model time.

- toolkit/controlnet_compat.py
  - `ControlNetLegacyAdapter` shim: lines **1–120** 
    - Clean: keep, but clearly label as legacy compatibility; add tests.
  - `VideoXControlnetWrapper`: lines **120–520** 
    - Clean: this file is mostly strict and well documented, but it still includes a small shim path (legacy shim application) and duplicate helper definitions; remove or centralize that shim code and avoid reintroducing fallbacks in the wrapper.
    - Ensure early validation and strictness is consistently applied; add tests for the wrapper conditions (control_context present, channel checks, raw pixel detection, assembled 33-channel expectations).

- toolkit/control_channels.py
  - Assembly / collapse helpers and constants: search for `assemble_zimage_control_context`, `ENCODED_LATENT_CHANNELS`, `RAW_IMAGE_CHANNELS` (lines around **1–220**). 
    - Clean: verify functions are deterministic, well-tested, and have clear docstrings. Add assertions for inputs and outputs.

- toolkit/control_util.py (or utility functions)
  - Functions: `adapter_uses_zimage`, `infer_expected_in_ch` (where imported in SDTrainer.py import lines). 
    - Clean: unify detection logic so Z-Image mode is configured explicitly (prefer adapter_config/controlnet_mode) rather than heuristic probing.

- tests/
  - testing/test_controlnet_compat.py and testing/test_controlnet_compat_4ch_promotion.py — add/extend tests for strict behavior and to assert that legacy shim only runs in non-Z-Image mode. 

- docs/ & README
  - Update Z-Image / VideoX sections to document strict requirements (control_context, channels, control_in_dim). See `06-pretraining-setup.md` and README notes about Z-Image tokenizers. 

---

## Concrete cleanup tasks (ordered, small, testable changes)

1. (Audit) Add an `AUDIT.md` section summarizing exact lines changed + rationale for reviewers. Mark as done when audit complete. (quick win)
2. (Helper) Add `toolkit/controlnet_utils.py` (or augment `control_util.py`) with:
   - `is_zimage_adapter(adapter, adapter_config) -> bool` (explicit config-based detection only)
   - `validate_zimage_adapter(adapter) -> None` (raises with actionable message if signature / control_in_dim mismatch)
   - Add unit tests for these helpers.
3. (SDTrainer) Replace flexible detection + heuristic code in Z-Image branch with calls to `is_zimage_adapter` and `validate_zimage_adapter`. Remove shim-skips and legacy tolerant paths from the Z-Image branch. Add a strict `with self.timer('controlnet_zimage_forward'):` block around any adapter calls/assemblers in this branch. Add calls to torch.cuda.synchronize() for debug builds to get reliable GPU timings.
4. (SDTrainer) Centralize device/dtype cast and channel-adapt calls into `toolkit/controlnet_utils.py` helpers. Remove nested helper functions duplicated in the trainer.
5. (VideoX wrapper) Remove runtime fallback shim application if adapter_config indicates Z-Image; if shim is required for legacy compatibility, restrict it strictly to non-Z-Image paths and mark it clearly.
6. (Tests) Add unit tests verifying:
   - Z-Image adapter missing `control_context` raises.
   - Adapter with wrong channels raises.
   - Timing/metric tests: after instrumenting small timers, ensure controlnet work is reported under the new timers not Model.
7. (Docs) Add a short migration note (LEARNINGS.md) explaining strict mode and how to adapt adapters.

---

## Acceptance criteria
- All new and changed behavior covered by unit tests.
- Z-Image mode no longer uses permissive heuristics; code fails fast with actionable messages if adapters are misconfigured.
- ControlNet timing is visible in perf summaries for Z-Image (new `controlnet_zimage_forward` timer or moved instrumentation).
- Legacy compatibility for non-Z-Image adapters preserved (tested), but clearly separated.

---

## Notes / Next steps
- I can: (A) prepare a PR that implements the `is_zimage_adapter` helper and a single SDTrainer check + timer (minimal change), or (B) implement the full refactor in steps with tests and docs. Which do you prefer?

---

Please review and I will scaffold the first small patch (helper + strict Z-Image check + timer) and a unit test.

---

## Work performed (initial patch)
- Added strict helper module `toolkit/controlnet_utils.py` with `is_zimage_adapter` and `validate_zimage_adapter` (new file).
- Enforced strict Z-Image validation in `extensions_built_in/sd_trainer/SDTrainer.py` by:
  - Replacing permissive zimage adapter fallback with a strict validation and setup timer (`controlnet_zimage_setup`). (approx lines 3488-3510, 3522-3660)
  - Using `is_zimage_adapter` to detect explicit Z-Image mode (replaced `adapter_uses_zimage` usage in those branches).
- Removed permissive zimage adapter call/fallback and shim application in `SDTrainer.py` (no more legacy shim in zimage branch).
- Added strict call-timer and removed permissive signature fallbacks for Z-Image adapter calls in the model path `toolkit/stable_diffusion_model.py` (added timer `controlnet_zimage_forward` and now only attempts the strict signature; earlier permissive TypeErrors removed). (approx lines 2856-2944)
- Updated `jobs/process/BaseSDTrainProcess.py` to prefer the explicit `is_zimage_adapter` detection (replaced several heuristic sites).
- Added unit tests in `testing/test_controlnet_utils.py` to validate helper behavior (3 tests currently pass).

Next steps: update docs / LEARNINGS, add tests that exercise the end-to-end controlnet zimage routing and perf-timing, and prepare a PR.


