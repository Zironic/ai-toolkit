# Subplan 1 of 4 -- Model-Agnostic Memory + Compile + Linux

> Sequencing amendment (2026-07-10): the broad adapter/compile extraction is
> paused behind `IMMUTABLE_TRANSFER_ARENA_PLAN.md` (ticket `628b0cb`). The
> independent Linux pin-headroom and block-name hardcode fixes may proceed, but
> do not extract the current arena generation/restoration boundary into shared
> APIs. Resume the model adapters after the immutable arena, sidecar residency,
> and execution-plan interfaces stabilize.


Part of the "model-agnostic memory/compile layer" effort. Independent of the FP8/quant
subplans (4a, 2, 3); it can start now against today's FP8 functions. See sibling docs
`MODEL_AGNOSTIC_SUBPLAN_4A_QUANT_SEAM.md`, `..._2_FP8_FORWARD.md`, `..._3_FP8_BACKWARD.md`,
and the pre-existing `PINNED_ARENA_PHASE3_TRAINING.md` (Slice C is the anchor for the shared
pack builder). Mutable status lives in the git-bug ticket, not here.

## Goal

Move memory/offload/pinned-arena/ingraph-compile orchestration out of the Krea2 model into a
shared, model-agnostic layer; prove it on a second real model (Ideogram); make Linux pinning
permissive. Models only provide structural facts + architecture-specific block execution.

## Why this is small (verified against faster-dop)

- The shared memory/arena layer is **already ~95% model-agnostic**: `pinned_arena.try_borrow_pack(block_key: str, ...)`
  and `ingraph_stream.pack_block_host*` already use opaque **string** block keys; grouping is
  generic (`_offload_group_key`).
- The one real hardcode: `training_pinned_keys_for_keep_last` reads `getattr(module, "blocks")`
  and emits `f"blocks.{i}"` (manager.py ~3965-3974).
- `get_transformer_block_names()` already exists as the block-list seam (`["blocks"]` for Krea2,
  `["layers"]` for Z-Image/Ideogram).
- Krea2 orchestration in `mmdit.py` is ~80% generic; the model-specific adapter is ~60-120 lines
  (seams: `self.blocks`, `_block_linear_entries -> (name, Linear)`).
- The pack/stream/flatten layer is already quant-agnostic (opaque leaves, dequant deferred to the
  per-Linear forward), so this subplan imports today's `_fp8_linear_compiled`/`_fp8_linear_training`
  unchanged; the backend swap comes later in subplans 4a/2.

## Files

- New: `toolkit/memory_management/compile_manager.py`, `pack_builder.py` (or extend
  `ingraph_stream.py`), `model_protocol.py`.
- Edit: `manager.py` (hardcode fix), `pin_manager.py` (Linux).
- New adapters: `extensions_built_in/diffusion_models/krea2/src/memory_compile_adapter.py`,
  `extensions_built_in/diffusion_models/ideogram4/src/memory_compile_adapter.py`.
- Shrink `krea2/src/mmdit.py` to glue.

## Tasks

1. Extract the borrow-or-own pack loop + strip/restore-contaminants + block-fn factories +
   trunk-compile from `mmdit.py` into shared `build_or_borrow_block_packs` +
   `compile_manager.enable_region` (anchor on `PINNED_ARENA_PHASE3_TRAINING.md` Slice C). Keep
   importing today's FP8 functions unchanged.
2. Define the adapter protocol: `stream_units`, `block_specs`, `compile_regions`, strip/restore,
   `collect_adapter_state`, `make_block_callable`, `make_region_callable`. Model opts in via
   `model._mm_compile_adapter` (or `memory_compile_adapter()`); the trainer/manager uses it if
   present and falls back to existing behavior otherwise. Krea2 adapter delegates to existing
   `_block_linear_entries` / `_nest_*` / `forward_streamed`.
3. Fix the one shared hardcode: `training_pinned_keys_for_keep_last` derives block-list name(s)
   from `get_transformer_block_names()` instead of `getattr(module, "blocks")`.
4. Ideogram second consumer: add `Ideogram4MemoryCompileAdapter`; wire
   `attach_smart_training(use_pinned_arena=...)` + a streamed-block forward for `self.layers`
   (in-repo `nn.Module`, single homogeneous list, clean named Linears -- near-drop-in).
5. Linux (trivial): at `pin_manager.pinned_bytes_headroom`, do not apply
   `AI_TOOLKIT_PINNED_WEIGHT_WDDM_FRACTION` off-Windows -- return `None` (permissive; consumers
   already treat `None` as "no authoritative cap"). Comment the seam for a future real Linux probe
   (must eventually consult `RLIMIT_MEMLOCK`). Strict-mode pin refusal already fails closed.
6. Keep the compile-region fingerprint minimal (only what Krea2 needs today) with a
   backend-identity slot that starts as the literal `"torchao_fp8"` (filled later by subplans 4a/2).

## Critical to preserve

`enable_ingraph_training` must collect LoRA entries **before** stripping compile contaminants
(`mmdit.py` ~1399-1401) -- this regressed once as "232/512 LoRA grads". Encode the ordering in the
shared helper and pin it with a regression test.

## Model divergences the adapter must absorb

- Block-list attribute name (`blocks` vs `layers`) -> via `get_transformer_block_names()`.
- Per-block Linear naming (Krea2 flat `self_attn.q_proj` vs Ideogram `attention.qkv`,
  `feed_forward.w1/w2/w3`) -> adapter enumerates real modules with stable keys.
- Block-forward contract varies -> block-calling lives in the adapter
  (`make_block_callable` + opaque `adapter_state` carrying shared aux tensors).

## Acceptance / verification (all Windows/GPU-runnable)

- Krea2 parity: strict ingraph sampling image tolerance unchanged; all streamed packs pinned;
  borrowed/owned counts match; ingraph training LoRA grads + loss parity vs the owned-pack path;
  train->sample->train leaves the `weights` ledger flat with no registered-range leaks. Run
  `scripts/smoke_krea2_train_cuda.py` + the arena/ingraph tests under `tests/`.
- Perf parity vs the 3.4-4.0 s/step @512 eager baseline via `scripts/digest_perf_log.py`. (This is
  the guard the original plan omitted; the subsystem exists for speed.)
- Ideogram attaches through the shared layer with `use_pinned_arena`, borrows packs for all
  `layers` blocks, runs streamed train steps producing LoRA grads -- proving a second real model
  runs the shared path with no Krea2 imports in shared modules (grep to confirm).
- Linux seam unit test: pin-headroom returns `None` under a simulated non-`nt` `os.name`, no WDDM
  fraction subtracted.

## Dependencies

- Prereq: land/settle the current uncommitted arena WIP first; do not refactor on a moving base.
- Depends on: nothing (start now).
- Soft coupling to 4a/2: only at the compile fingerprint's backend-identity slot.
- Related existing ticket: ddf9a52 (Pinned arena Phase 3: training + model-agnostic seam).

## Invariants

- Arena covers frozen base weights only; trainable adapter params never arena-repointed; arena
  active => per-layer pin budget 0, bounce reserve 0; `detach()` does not destroy the arena.
- Strict pinned-arena training: `borrowed_count == block count`, `owned_count == 0`; no silent
  owned fallback.
- Compile regions keyed on stable block keys + pack fingerprints; fullgraph failure fails closed
  in strict mode.
- `block_key` stable across train/sample cycles; adapter owns architecture math, shared code owns
  memory/pack/compile lifecycle.
