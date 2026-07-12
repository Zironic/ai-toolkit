> **SUPERSEDED 2026-07-12** - archived, not shipped. The A/P/C/D upstream stack
> and the model-agnostic subplans were retired when the immutable arena runtime
> became the sole transformer backend (commit `3dd7f38`): PRs C and D were built
> on `_BouncingLinearFn`, which no longer exists on the active path. Kept as the
> record of the reasoning. **Current plan:**
> `tasks/open/UPSTREAM_ARENA_EXTRACTION_PLAN.md` (PR P's crash-guard framing and
> upstream audit survive there as Stage 1; PR A as the deferred Stage 4).

# Subplan 2 of 4 -- FP8 Forward Behind the Backend

Part of the "model-agnostic memory/compile layer" effort. The easiest and highest-value FP8 piece,
correctness-neutral. Depends on `MODEL_AGNOSTIC_SUBPLAN_4A_QUANT_SEAM.md` (the backend object) and
soft-couples to `MODEL_AGNOSTIC_SUBPLAN_1_MEMORY_COMPILE.md` (the compile fingerprint slot).
Mutable status lives in the git-bug ticket.

## Goal

Route the forward / inference / sampling FP8 path through the `TorchAORowwiseFP8Backend` instead of
free functions with hardcoded `qdata`/`scale` sniffing. Native `_scaled_mm` is already the shipping
path, so this is correctness-neutral -- purely a relocation behind a stable seam plus fingerprint
wiring.

## Why easiest

`_fp8_linear_compiled` is already branch-free and compile-clean; `fp8_sampling_qualifies` already
hoists its static checks (torch `_scaled_mm`, SM89+, `float8_e4m3fn`, rowwise scale) to install
time. Moving them behind backend methods is mechanical.

## Files

- Edit: `manager_modules.py`, `ingraph_stream.py`, `compile_manager.py` (fingerprint).

## Tasks

1. Move the forward path behind backend methods (keep compatibility wrappers one phase):
   - `fp8_sampling_qualifies` -> `backend.compile_qualifies`
   - `_fp8_linear_compiled` -> `backend.compiled_linear`
   - `fp8_linear_inference` -> `backend.native_linear_inference`
   - dequant-into-reused-buffer paths (`_fast_fp8_dequant_into` / `_dequantize_into` /
     `_fast_fp8_dequant` / `_dequantize_to`) -> backend methods, with the `.dequantize()` reference
     path as fallback.
2. `ingraph_stream.py`'s `_fp8_linear_training if training else _fp8_linear_compiled` dispatch reads
   the backend's compiled fn (training half stays as-is until subplan 3).
3. Fill subplan 1's compile-fingerprint backend-identity slot with the real backend name/version +
   device capability (torch `_scaled_mm` availability, SM89+). This prevents stale compile reuse
   across different GPUs / backend changes.

## Acceptance

- Krea2 + Ideogram FP8 sampling/forward numerical parity unchanged.
- Compiled path still selects native FP8 when eligible and falls back to dequant when not.
- Running on a non-SM89 GPU falls back cleanly.
- Perf parity on the smoke baseline (3.4-4.0 s/step @512 eager, `scripts/smoke_krea2_train_cuda.py`
  via `scripts/digest_perf_log.py`). The compile-clean `_scaled_mm` is the 370s->10s win -- do NOT
  touch it without the perf gate.

## Dependencies

- Depends on: subplan 4a (backend object) and subplan 1's fingerprint slot.
- Blocks: subplan 3 (FP8 backward).
