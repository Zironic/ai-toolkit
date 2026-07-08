# Pinned Arena Phase 2: budget integration + full train->sample->train cycle

Tickets: 534ea49 (boundary pin churn), 763bb75 (register/alloc collision + pack
leak). Phase 1 (landed 2026-07-08) built the arena mechanics: `pinned_arena.py`,
attach/detach/unpin guards, sampling-boundary survival, ingraph sampling pack
borrowing, `--pinned-arena` in both Krea2 smokes. Phase 2 makes it correct at
scale: the arena must obey the SAME pin budget the per-tensor path obeys, stop
triple-pinning at first attach, and survive the realistic
training -> strict-ingraph sampling -> training cycle.

## Design rules

1. **Same budget, different shape.** The arena uses the pin-budget chain the
   per-tensor path uses -- `plan_budgets` (pin_manager.py:464) fed by
   desired-bytes scoped to the SELECTED offload ids (the `MemoryManager.attach`
   logic at manager.py:539-556: sum weight/bias bytes over `selected_offload_ids`
   only, `*1.03` auto, or explicit `pinned_weight_gib`). Never
   `plan["model_bytes"] * 1.03` unless everything streams. Only the allocation
   shape changes: per-block flats instead of scattered per-tensor registers.
2. **One persistent arena, union of streamed sets, never destroyed at phase
   boundaries.** Destroy/rebuild-per-phase was considered and REJECTED: it
   re-introduces the ~10 GiB unpin+repin per boundary (0.6-2 GB/s page-lock)
   that is this ticket's whole complaint. A block pinned for training but
   resident during sampling stays pinned -- shared-budget cost, zero dedicated
   VRAM cost (Comfy model). The existing `is_current` skip in
   `_build_pinned_arena` already gives lazy union-growth.
3. **The arena is the only pinner when active.** With `use_pinned_arena=True`,
   the per-tensor pin path must not pin at all (budget 0 through
   `_move_params_to_cpu_and_pin`); the arena consumes the real resolved budget
   itself. Kills the current first-attach triple churn
   (per-tensor pin 9.9 GiB -> unpin 9.9 -> arena pin 9.9).
4. **Teardown is explicit and ordered.** Params must be detached from arena
   flats (cloned to standalone storage) BEFORE `arena.release()`; only at true
   model teardown, never at sampling boundaries.

## Slices

### Slice A -- budget helper + smart-training sizing fix
- New `MemoryManager._desired_pin_bytes_for_offload_ids(module, offload_ids,
  pinned_weight_gib)`: extract the exact managed_bytes/auto/explicit logic from
  `attach` (manager.py:539-556); `attach` calls it too (single source of truth).
- `attach_smart_training` (manager.py:2454): replace
  `int(plan["model_bytes"] * 1.03)` with the helper scoped to
  `plan["offload_ids"]`. Keep `_planned_bounce_reserve_bytes` +
  `_cap_auto_pin_budget` + `resolved_pin_gib` structure unchanged.
- Tests: helper parity with old attach math (CPU); smart-training auto pin for a
  model with resident blocks requests only streamed bytes.

### Slice B -- arena owns the budget; kill the triple churn
- `PinnedWeightArena.build(entries_by_block, *, budget_bytes=None, kind=...)`:
  pin blocks in the given order until committed-new bytes reach `budget_bytes`;
  remaining blocks get pageable flats (still repointed -- uniform layout).
  `pin_alloc(required=False)` stays as the OS/headroom backstop.
- `_build_pinned_arena(module, budget_bytes)`:
  - receives the resolved pin budget from `attach`;
  - passes `desired_new = max(0, budget - arena.committed_pinned_bytes())`
    (already-committed bytes are in DXGI usage; re-requesting double-counts);
  - keeps `_interleave_priority`-style ordering irrelevant here (whole blocks,
    deterministic named_modules order is fine);
  - drops the `unpin_layer(child)` pre-pass ONCE Slice B2 lands (below) --
    until then it stays as the correctness bridge.
- B2: in `attach`, when `use_pinned_arena=True`, run the per-layer deferred
  attach with an effective per-tensor budget of 0 (layers attach unpinned),
  then let `_build_pinned_arena` do the only pinning against the real budget.
  `plan_budgets` is still consulted once, in `attach`, exactly as today.
- Tests: budget cap respected (blocks beyond budget pageable); no
  `pin_tensor_in_place` calls at all during arena-enabled attach (monkeypatch
  counter); re-attach requests only the delta.

### Slice C -- drop the stale ingraph gate
- `inference_resident` (manager.py, two attach call sites): change
  `args.get("use_pinned_arena", False) and not reserve_pin_for_ingraph` to
  `bool(args.get("use_pinned_arena", False))`; update the comments -- ingraph
  sampling packs now borrow arena flats (Slice 4, `try_borrow_pack`), so
  arena+ingraph is the intended pairing, not a double-commit.
- Keep `pinned_weight_gib=0.0` zeroing for the NON-arena path unchanged; with
  the arena on it is inert anyway after Slice B2 (per-tensor path pins nothing).
- Blocks streamed for sampling that weren't in the training arena get added by
  the same lazy-grow build; ingraph then borrows them too. Owned-pack fallback
  (with its own `ingraph_pack` pin) remains for any block the borrow rejects.
- Tests: CPU-level -- attach_args round-trip keeps use_pinned_arena under
  reserve_pin_for_ingraph; GPU smoke covers the rest (Slice E).

### Slice D -- safe explicit teardown
- `MemoryManager._destroy_pinned_arena(module)`:
  1. for every child with `_mm_arena_block`: replace weight/bias Parameters
     with standalone clones (quant wrappers via `_flatten_leaves` +
     `_rebuild_from_leaves` over cloned leaves), preserve `requires_grad`,
     clear `_mm_arena_block`/`_mm_arena_generation`;
  2. then `arena.release()` and `del module._mm_weight_arena`.
  Never release while live params still view arena storage (ledger would say
  released while the flats stay page-locked via the param references).
- Call sites: available to model-unload paths and tests; NOT called from
  `detach` (sampling boundaries must keep the arena -- design rule 2).
- Tests: post-destroy state_dict intact, params standalone, `"weights"` ledger
  back to baseline, `arena_block_of` empty.

### Slice E -- strict-ingraph criteria + smoke instrumentation
- Strict ingraph (mmdit.py `enable_ingraph_sampling`): requirements apply per
  STREAMED pack -- `pack.pinned` (exists today, `non_pinned_pack`); when the
  arena is enabled, additionally assert-and-log how many packs are
  `borrowed_from_arena` vs owned fallback. Do NOT require every arena block to
  be pinned -- only the packs actually built for streamed blocks matter.
- `scripts/smoke_krea2_ingraph_cuda.py` (`--pinned-arena` exists): add JSON
  events + asserts for the full cycle:
  - training arena exists after `_attach_training_memory`
    (blocks/pinned/pageable from `arena.stats()`);
  - inside `inference_resident`: arena object identity unchanged, ledger
    `"weights"` unchanged across the boundary (pure accounting);
  - strict ingraph: `all(p.borrowed_from_arena for p in packs)` and
    `all(p.pinned ...)`; log owned-fallback count (expect 0);
  - after sampling exit: training layout restored, same arena object, ledger
    unchanged, boundary detach/attach timing logged (expect <100 ms vs the
    1.2 s / 2.0 s baseline);
  - explicit `_destroy_pinned_arena` at end: ledger returns to baseline
    (no pinned-ledger leak).
- `scripts/smoke_krea2_train_cuda.py`: same training-side assertions.

## Pass criteria
- Auto pin sizing == old `attach` math scoped to current offload ids, in both
  attach paths; arena never pins past the resolved budget.
- Zero `cudaHostRegister` (pin_tensor_in_place) calls during arena-enabled
  attach; zero pin/unpin work at sampling boundaries (measured, not assumed).
- Strict ingraph passes at 512px with every streamed pack borrowed+pinned.
- Full cycle training -> sampling -> training on the real Krea2 smoke with
  `--pinned-arena --strict-ingraph`: no `resource already mapped`, no ledger
  drift, boundary cost <100 ms each way.
- Flag off: byte-identical behavior (full suite green both ways).

## Explicitly rejected
- **Phase-owned arenas (destroy on train<->sample transition).** Re-creates the
  churn the arena exists to remove; the union arena + lazy grow subsumes every
  correctness concern it addressed. Teardown exists (Slice D) but only for true
  model unload.
