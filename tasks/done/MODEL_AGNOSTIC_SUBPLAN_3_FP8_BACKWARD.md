> **SUPERSEDED 2026-07-12** - archived, not shipped. The A/P/C/D upstream stack
> and the model-agnostic subplans were retired when the immutable arena runtime
> became the sole transformer backend (commit `3dd7f38`): PRs C and D were built
> on `_BouncingLinearFn`, which no longer exists on the active path. Kept as the
> record of the reasoning. **Current plan:**
> `tasks/open/UPSTREAM_ARENA_EXTRACTION_PLAN.md` (PR P's crash-guard framing and
> upstream audit survive there as Stage 1; PR A as the deferred Stage 4).

# Subplan 3 of 4 -- FP8 Backward Behind the Backend (last / optional)

Part of the "model-agnostic memory/compile layer" effort. The hardest and least important FP8
piece: it is already flag-gated and it lowers precision, so it ships last (or in a separate later
PR) with zero impact on subplans 1 / 4a / 2. Depends on
`MODEL_AGNOSTIC_SUBPLAN_4A_QUANT_SEAM.md` + `MODEL_AGNOSTIC_SUBPLAN_2_FP8_FORWARD.md`. Mutable
status lives in the git-bug ticket.

## Goal

Route the training grad-input FP8 path through `backend.grad_input`, without changing the safe
default. The native FP8 grad-input is a precision-lowering optimization gated behind
`layer_offloading_fp8_grad_input`; the dequant-to-bf16 matmul fallback stays the default.

## Why hardest / least important

`_Fp8LinearTrainingFn` is a custom `torch.autograd.Function` with the grad-input approximation
(fold row-scale into the activation, `_scaled_mm`). It is the one FP8 path that trades numerical
precision, it is off by default, and it has had gate regressions before (see ticket f1c45ae). So it
carries the most risk for the least benefit and must not block the rest of the work.

## Files

- Edit: `manager_modules.py`.

## Tasks

1. Move `_Fp8LinearTrainingFn` / `_fp8_linear_training` / `_fp8_grad_input_supported` /
   `_fp8_grad_input_compute` / `_fp8_grad_input` / `set_fp8_grad_input_enabled` behind
   `backend.grad_input` (+ `backend.supports_grad_input`). Keep the flag
   `layer_offloading_fp8_grad_input` and the one-time `_FP8_GRAD_VERIFIED` self-check.
2. The dequant-to-bf16 fallback remains the default when the flag is off or the GPU is incapable.

## Acceptance

- Flag on: native FP8 grad-input still selected on capable GPUs and matches the prior verified
  path within tolerance.
- Flag off: bf16 fallback unchanged.
- Perf parity on the smoke baseline via `scripts/digest_perf_log.py`.
- No impact on subplans 1 / 4a / 2 (can ship in a later PR).

## Dependencies

- Depends on: subplan 4a (backend object) + subplan 2 (forward relocation).
- Related existing ticket: f1c45ae (FP8-native training gate regression on cache-loaded runs).
