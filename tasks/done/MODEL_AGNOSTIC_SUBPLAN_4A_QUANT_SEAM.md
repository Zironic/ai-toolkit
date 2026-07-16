> **SUPERSEDED 2026-07-12** - archived, not shipped. The A/P/C/D upstream stack
> and the model-agnostic subplans were retired when the immutable arena runtime
> became the sole transformer backend (commit `3dd7f38`): PRs C and D were built
> on `_BouncingLinearFn`, which no longer exists on the active path. Kept as the
> record of the reasoning. **Current plan:**
> `tasks/done/UPSTREAM_ARENA_EXTRACTION_PLAN.md` (PR P's crash-guard framing and
> upstream audit survive there as Stage 1; PR A as the deferred Stage 4).

# Subplan 4a of 4 -- Quant Detection Seam (foundation for FP8)

Part of the "model-agnostic memory/compile layer" effort. This is the cheap FP8 foundation that
`MODEL_AGNOSTIC_SUBPLAN_2_FP8_FORWARD.md` and `..._3_FP8_BACKWARD.md` plug into. It must land
before FP8 forward. Sibling: `MODEL_AGNOSTIC_SUBPLAN_1_MEMORY_COMPILE.md`. Mutable status lives in
the git-bug ticket.

## Goal

One canonical quantized-weight detector plus a single torchao backend object, built over the
tensor-subclass protocol. No kernel moves yet -- just identity, detection, and flatten/rebuild
delegation. This is the "seam" half of the addendum's quant work; the "breadth" half (pluggable
multi-backend registry / non-torchao formats) is deferred indefinitely (subplan 4b) because only
one real format exists today.

## Decision: torchao tensor-subclass protocol, NOT diffusers' quantizer

Diffusers ships `diffusers/quantizers/` (`DiffusersQuantizer` + `TorchAoConfig`/BnB/GGUF/Quanto/
ModelOpt), but it is a load/save-time abstraction (HfQuantizer lineage) with none of the runtime
primitives the offload path needs (dequant-into-reused-buffer, compile-clean `_scaled_mm`,
grad-input, async subclass transfer, arena leaf-packing). The repo already bypasses it
(`toolkit/util/quantize.py` quantizes directly via torchao `quantize_` -> `Float8Tensor`). The
reusable abstraction is the torchao / PyTorch tensor-subclass protocol
(`__tensor_flatten__` / `__tensor_unflatten__`, `.dequantize()`), which the format-agnostic
sublayer already uses. Build ONE `TorchAORowwiseFP8Backend` over it.

## Files

- New: `toolkit/memory_management/quant_backend.py`.
- Edit: `manager_modules.py`, `lora_special.py`, `toolkit/util/quantize.py`.

## Tasks

1. `detect_quant_backend(weight) -> (backend, layout) | None` and `is_quantized_weight(weight)` over
   `__tensor_flatten__` / `__tensor_unflatten__` + `Float8Tensor` layout.
2. Create `TorchAORowwiseFP8Backend` as the seam object: backend name/identity, `detect`, and
   `flatten_leaves` / `rebuild_from_leaves` delegating to the existing generic helpers
   (`_flatten_leaves`, `_rebuild_from_leaves`, `_wrapper_to_async`). Do NOT move dequant/GEMM/grad
   kernels here yet -- that is subplans 2 and 3.
3. Collapse the three duplicate "is-quantized" predicates into the one detector:
   - `manager_modules.py` (`_is_quantized_tensor` / `_is_ao_quantized_tensor`)
   - `lora_special.py` (its own `_is_quantized_tensor`)
   - `toolkit/util/quantize.py:66` (`is_quantized_tensor`)
   Keep thin compatibility shims at each old call site for one phase.

## Acceptance

- Detector returns the torchao backend for a `Float8Tensor`, `None` for a plain float tensor, and
  fails closed (`unsupported_quant_wrapper`) on an unsupported quantized wrapper.
- The three prior call sites behave identically (no numerical change, same accounting).
- No Krea2 imports in `quant_backend.py`.

## Dependencies

- Depends on: nothing hard (can run parallel to subplan 1).
- Blocks: subplan 2 (FP8 forward) and subplan 3 (FP8 backward).
- Fills subplan 1's compile-fingerprint backend-identity slot (currently the literal
  `"torchao_fp8"`).
