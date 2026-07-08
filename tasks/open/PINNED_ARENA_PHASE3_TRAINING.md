# Pinned Arena Phase 3: training coverage + model-agnostic seam

Ticket: 534ea49 (boundary pin churn). Builds on Phase 2 (budget integration).

Phase 1 built the arena mechanics; Phase 2 made the budget correct; the
register-mechanism work (below) made sampling actually pin the full streamed
set and borrow it under strict ingraph. Phase 3 extends the arena to cover
**training** (the base weights that today pin through a separate mechanism)
and makes the whole thing usable by **any** streamed model, not just Krea2.

## Where Phase 2 + register work landed (starting point)

Sampling now works end-to-end on the real 768px fp8 Krea2 smoke: every attach
pins the full streamed set (`pageable_blocks=0`, 9.92 -> 10.87 GiB across the
train->sample growth), all 18 streamed ingraph packs BORROW the arena flats
(ledger stays `weights`-only, no `ingraph_pack` allocations), no
`non_pinned_pack`. Key mechanisms this phase depends on:

- **Register mechanism for arena flats.** `pin_manager.pin_register`
  (cudaHostRegister on a page-exclusive carved buffer) instead of `pin_alloc`
  (caching host allocator). pin_alloc rounds to power-of-two buckets -- 8.86
  GiB of flats committed 12.70 GiB of DXGI budget; register cost == ledger
  cost, returned immediately on release. `_empty_host_flat` /
  `pack_block_host` take `pin_mechanism="register"`; the arena passes it.
- **`is_pinned()` is blind to cudaHostRegister.** torch only tracks its own
  caching-allocator pins, so registered flats report `is_pinned()==False`.
  `pin_manager.is_host_pinned(t)` (torch is_pinned OR the registration table)
  is the source of truth. Already threaded into the borrow path
  (`pack_block_host_from_flat`) and the ingraph fetch op (`_fetch_start_impl`).
  **NOT yet threaded into the eager/streaming bypass** -- that is Slice A.
- **Budget = full usable headroom for the sole pinner.** `attach` requests the
  full `desired` (not a `desired - committed` delta) and hands
  `_build_pinned_arena` the plan's usable weight headroom (`headroom - reserve
  - bounce`), because `desired` (physical x 1.03) undercounts per-leaf 256B
  alignment + 4096B register page-padding. build() only rebuilds
  stale/pageable groups, so a generous grant can't over-pin; pin_register
  enforces the true per-block limit + reserve.
- **bounce_reserve=0 when the arena is active** (both attach paths): a pinned
  weight bypasses bounce staging, so reserving a bounce window for the same
  weights double-counts and clips the arena below the streamed set.
- **Streamed blocks pinned first.** `_build_pinned_arena(priority_ids=...)`
  orders the current attach's `selected_offload_ids` groups first, so a tight
  budget spends pins where strict ingraph requires them.
- **Deterministic register cleanup.** `_LIVE_ARENAS` is a STRONG set (a
  GC'd arena would free its base tensors while pages stay cudaHostRegister'd ->
  "resource already mapped" on recycled pages). `release()` discards from it;
  `tests/conftest.py` sweeps it after each test. This also removed the
  long-standing order-dependent flake (f2aceba family): full suite 328 green.

## Goal

1. Training streams its frozen base weights from the SAME persistent arena the
   sampler uses -- one pin for the whole run, no separate `ingraph_training`
   pin mechanism, no train<->sample repin churn.
2. The arena is genuinely model-agnostic: the pinning/borrowing machinery lives
   in the memory-management layer, and a model opts in with a small, documented
   amount of glue (block enumeration + block-fn composition), exactly as Krea2
   does. No arena logic in per-model files.

## Current coupling (what makes it Krea2-only today)

- The arena core (`pinned_arena.py`), `MemoryManager._build_pinned_arena`,
  `try_borrow_pack`, `pack_block_host[_from_flat]`, and the pin-budget chain
  are ALREADY model-agnostic (they key on module identity + storage, never on
  Krea2 types or names).
- The per-model glue is entirely in `krea2/src/mmdit.py`:
  `enable_ingraph_sampling` (already borrows) and `enable_ingraph_training`
  (still owns), both driven by `self.blocks` + `_block_linear_entries`.
- `krea2.py:756` gates `use_pinned_arena = pinned_arena AND NOT
  ingraph_training` -- the deliberate Phase 1 mutual exclusion.
- The trainer seam is already generic: `BaseSDTrainProcess` (:3191) calls
  `getattr(inner_unet, 'enable_ingraph_training')` and fails loud if absent.

## Design rules (carry over + new)

- **One persistent union arena, never destroyed at phase boundaries** (Phase 2
  rule 2). Training and sampling stream sets differ; the arena is their union,
  lazily grown, `is_current`-skipped on re-attach.
- **Arena covers frozen base weights only.** `requires_grad=True` leaves fail
  closed (`arena_trainable_leaf`). In training the base is frozen
  (`BaseSDTrainProcess:2524`) before attach and LoRA adapters are SEPARATE
  modules -- never the offloaded base Linears -- so the base is arena-eligible
  and the optimizer/LoRA are untouched by repointing.
- **Borrow, never re-pin, for training too.** Training ingraph packs borrow the
  arena flat for the base and compose LoRA in the block fn (base = borrowed
  frozen source; grads flow only to the resident LoRA adapters). No second pin
  of the base.
- **Model-agnostic by protocol, not by base class.** Keep block enumeration in
  the model; move the borrow-or-own pack loop into the shared layer.

## Slices

### Slice A -- streaming path recognizes register-pinned flats (prerequisite)
The eager/pre-compile streaming bypass (`manager_modules._profile_is_pinned`
:165, `bounce_pool.py:91`) gates on torch `is_pinned()`, which is False for
register-pinned arena flats. Worse, a streamed leaf is a VIEW into the flat, so
`is_host_pinned`'s exact-ptr table lookup misses it (table is keyed on the
flat's base ptr). Consequence today: training's eager and pre-compile forwards
stream arena weights through bounce staging anyway -- correctness OK, but the
pin buys nothing off the ingraph path.
- Prefer O(1) tagging over a range scan on the hot path: when the arena
  repoints a module's params, tag the module (`_mm_arena_block` exists) and/or
  stamp the leaf; `_profile_is_pinned` short-circuits True for arena-backed
  storage. If a tensor-only signal is required, extend `is_host_pinned` with a
  registered-range check (ptr within any `[base, base+size)`) backed by a
  sorted interval list, and measure the per-forward cost before adopting.
- Tests: a streamed arena weight reports pinned to the profiler; the eager
  forward issues zero bounce-staged copies for arena blocks (spy on the bounce
  pool); flag-off path unchanged.

### Slice B -- training borrows the arena
- `krea2.py:756`: drop the `and not layer_offloading_ingraph_training` gate so
  the arena and ingraph training coexist.
- `mmdit.py enable_ingraph_training` (:1389): mirror `enable_ingraph_sampling`
  (:1132) -- `arena.try_borrow_pack(f"blocks.{i}", entries)` first, owned
  `pack_block_host(pin_mechanism="register")` fallback; release only
  `owns_flat` packs on teardown/failure. LoRA composition in the block fn is
  unchanged.
- Confirm the backward pass treats the borrowed flat as a read-only source (no
  grad write-back into pinned host memory); base is frozen so there is no base
  grad, but assert the fetch/scatter path never targets the flat.
- Sequencing check: arena builds inside `attach_smart_training` (base frozen),
  BEFORE LoRA apply + optimizer construction. Add an explicit assertion/log
  that no arena-repointed leaf has `requires_grad=True` at optimizer-build time.
- Tests (GPU): train ingraph with `--pinned-arena`, every streamed base pack
  `borrowed_from_arena and pinned`, owned-fallback count 0; one train step runs
  and produces a LoRA grad; `weights` ledger flat across a train->sample->train
  cycle.

### Slice C -- model-agnostic seam
- Extract the duplicated borrow-or-own loop from
  `enable_ingraph_sampling`/`enable_ingraph_training` into a shared helper in
  `ingraph_stream.py`, e.g. `build_or_borrow_block_packs(arena, entries_by_block,
  *, repoint, pin_mechanism)` returning `{index: BlockPack}` + borrowed/owned
  counts, with the fail-closed reasons centralized. Block enumeration stays
  model-side (the model knows its block/linear structure).
- Document the "ingraph arena protocol" a model implements to opt in: (1) freeze
  base before attach; (2) pass `use_pinned_arena` through
  `attach_smart_training`/`inference_resident`; (3) expose
  `enable_ingraph_sampling`/`enable_ingraph_training` that enumerate the model's
  blocks and call the shared helper. Put this in
  `docs/decisions/` or the module docstring.
- Nothing arena-specific may live in per-model files beyond that glue; grep for
  Krea2 type/name assumptions in the shared path and remove any.
- Tests: a synthetic multi-block `nn.Module` (not Krea2) drives
  `_build_pinned_arena` + the shared borrow helper end-to-end on CPU/CUDA,
  proving the machinery has no Krea2 dependency.

### Slice D -- validation + default consideration
- `scripts/smoke_krea2_train_cuda.py --pinned-arena` with ingraph training on:
  base borrowed, LoRA trains, loss parity vs the owned-pin path, boundary churn
  ~0, no `resource already mapped`, ledger flat across the full cycle.
- Full `train -> sample -> train` on one real short run: DXGI `weights` ledger
  flat across boundaries, no WDDM spill, boundary detach/attach << the 1.2s/2.0s
  baseline.
- Only after the matrix passes: consider flipping
  `layer_offloading_pinned_arena` default (separate commit, rollback lever).

## Ingraph arena protocol (the model-agnostic contract)

A model opts into the arena by satisfying this contract; nothing below is
Krea2-specific, and no arena logic lives in the model file beyond the glue in
(3)-(4).

1. **Freeze base before attach.** All offloaded base Linears have
   `requires_grad=False` before `attach_smart_training`/`inference_resident`
   run (the shared trainer already does this at `BaseSDTrainProcess:2524`).
   The arena fails closed (`arena_trainable_leaf`) otherwise.
2. **Plumb the flag.** `attach_smart_training(..., use_pinned_arena=<cfg>)` and
   the two `inference_resident` attach sites. The manager builds/reuses
   `module._mm_weight_arena` generically -- the model does nothing here.
3. **Enumerate blocks.** The model yields, per streamed block, a stable
   `block_key: str` and its `(name, module)` linear entries. Krea2 does this
   with `self.blocks` + `_block_linear_entries`; the shape is arbitrary as long
   as `block_key` is stable across attach cycles and matches
   `_offload_group_key` grouping.
4. **Enable via the shared helper.** `enable_ingraph_sampling` /
   `enable_ingraph_training` call the Slice-C helper:

   ```
   packs, borrowed, owned = build_or_borrow_block_packs(
       arena=getattr(model, "_mm_weight_arena", None),
       entries_by_block={block_key: [(name, module), ...], ...},
       repoint=<False for training-owned trunk, True where the model wants
                the params repointed>,
       pin_mechanism="register",
   )
   # every streamed pack must be `pack.pinned`; helper raises the
   # centralized fail-closed reasons (non_pinned_pack, arena_block_stale,
   # unsupported_quant_wrapper, ...) so callers do not re-implement them.
   ```

   The helper borrows from the arena when a block is arena-current, else builds
   an owned register pack; it releases only `owns_flat` packs on failure. The
   model composes any per-block extras (LoRA, checkpointing) in its own block
   fn, unchanged.

The trainer seam is already generic (`getattr(inner_unet,
'enable_ingraph_training')`, `BaseSDTrainProcess:3191`), so a conforming model
needs no trainer changes.

## Acceptance criteria
- Training with `layer_offloading_pinned_arena=True` streams every base block
  from a borrowed arena flat: owned-fallback pack count 0, every streamed pack
  `borrowed_from_arena and pinned`, zero `ingraph_pack`-kind ledger bytes.
- A single LoRA training step produces adapter gradients and matches the
  owned-pin path's loss within numerical tolerance (FP8 base + bf16 LoRA).
- `weights`-kind ledger is flat across a full `train -> sample -> train` cycle;
  boundary detach/attach cost is bounded well under the 1.2s/2.0s pin-churn
  baseline; no `resource already mapped`, no WDDM spill.
- Eager/pre-compile training forwards issue zero bounce-staged copies for
  arena blocks (Slice A): the streaming bypass sees register-pinned flats.
- A synthetic non-Krea2 multi-block module drives `_build_pinned_arena` + the
  shared borrow helper end-to-end (Slice C), proving no Krea2 dependency.
- Flag off (`layer_offloading_pinned_arena=False`): byte-identical to today,
  full suite green both ways.

## Risks
- **LoRA + borrowed FP8 base + backward.** The one genuinely new interaction:
  frozen FP8 base streamed from a borrowed flat, trainable bf16 LoRA resident,
  grads to LoRA only. Validate numerically against the owned-pin path.
- **Hot-path pinned check (Slice A).** A per-forward range scan could cost more
  than it saves; tagging is the safer default.
- **Optimizer/LoRA vs repoint ordering.** Repoint replaces base Parameters; it
  must complete before the optimizer captures anything. Base is frozen and not
  in the optimizer, so this is expected-safe, but assert it rather than assume.
- **Non-uniform block structure.** Models whose "blocks" are not a flat uniform
  list need their own enumeration; the shared helper must not assume Krea2's
  `self.blocks` shape (it takes `entries_by_block`, so this stays model-side).

## Explicitly rejected (carried from Phase 2)
- Phase-owned arenas (destroy on train<->sample): re-creates the ~10 GiB
  repin-per-boundary churn the ticket exists to remove.
- A per-model arena base class: couples models to arena internals; the
  protocol + shared helper keeps the machinery in the memory layer.
