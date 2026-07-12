# Canonical Host Arena + Sidecar Residency Refactor

> Durable architecture plan. Mutable status lives in git-bug ticket `628b0cb`.
> Related: `534ea49` (current pinned arena, the measured reference), `3ca8a7b`
> (in-graph streaming), `ca8f496` (model-agnostic extraction), and
> `tasks/open/INGRAPH_STREAM_PLAN.md`.
>
> Rewritten 2026-07-10 after the pin-assumption measurement campaign
> (`scripts/bench_pin_assumptions.py`). The previous revision assumed pin
> costs and compile couplings that direct measurement disproved; this
> revision replaces it entirely (the old text had no archival value and was
> never committed).

## Decision

Keep the proven parts of the current pinned arena:

- one canonical, page-exclusive, pinned host representation per block;
- centralized DXGI/WDDM budgeting in `pin_manager`;
- block-coalesced transfers through the bounded, ticketed H2D ring;
- zero-copy reuse of the host representation by eager and compiled paths;
- persistent registration for the run (as *policy*, see below -- not as a
  compiler or correctness invariant).

Replace the parts that made it brittle:

- live `module.weight` storage is no longer the arena-validity oracle;
- promotion/demotion never invalidates, rebuilds, repoints, or unpins host
  storage;
- frozen base-weight residency lives in manager-owned GPU sidecars, chosen
  per-Linear, transferred via static multi-range plans into compact device
  buffers;
- compiled execution specializes on an immutable residency/layout plan, not
  on mutable module tags, pack identity, or pinnedness;
- pinnedness is true by construction (populate, then register synchronously,
  then publish) and is validated by a single canonical-allocation registry,
  not by three overlapping bookkeeping structures.

The durable invariant is:

```text
(block_key, leaf_key)
    -> immutable host flat region + immutable source metadata

residency plan (per phase)
    -> optional manager-owned GPU sidecar per leaf

transfer plan (per residency plan)
    -> static list of host source ranges -> compact device offsets
```

not:

```text
live Parameter storage identity
    -> infer arena membership/currentness/pin state
```

## Measured Evidence (2026-07-10)

RTX 4070, torch 2.12.0+cu132, CUDA 13.2, Windows 11 WDDM, 32 GiB RAM, idle
GPU. Re-runnable: `venv/Scripts/python.exe scripts/bench_pin_assumptions.py`.
Re-run after torch/CUDA/driver upgrades; test 1 is the canary for torch's
`is_pinned()` semantics, tests 13-14 for Dynamo guard behavior.

Registration and transfer (tests 2-4, 7-9):

| Operation | Measured |
| --- | --- |
| `cudaHostRegister`, pages populated | 3.1 ms / 600 MiB; 27 ms / 4 GiB (~150 GiB/s) |
| `cudaHostRegister`, pages untouched (demand-zero) | 67 ms / 600 MiB (~9 GiB/s, page faults dominate) |
| `cudaHostAlloc` (torch `pin_memory=True`) | 114 ms / 600 MiB |
| `cudaHostUnregister` | 33 ms / 600 MiB; 257 ms / 4 GiB |
| per-registration call overhead | ~100-145 us (1500-call batches) |
| registration with 8 GiB already committed | 4.3 ms / 600 MiB (no pressure effect) |
| threaded registration (2/4/8 threads) | SLOWER than serial; no parallelism win |
| H2D from pinned source (either mechanism) | ~15 GiB/s, 0.2 ms non-blocking submit |
| H2D from pageable source | ~5.9 GiB/s, ~100 ms/600 MiB BLOCKING submit |

Multi-range submission (test 16, 384 MiB Krea2-block-sized pinned flat):

| ranges | submit | GPU copy | effective bw |
| --- | --- | --- | --- |
| 1 | 0.08 ms | 18.6 ms | 19.5 GiB/s |
| 24 | 0.39 ms | 18.6 ms | 20.1 GiB/s |
| 64 | 0.79 ms | 17.5 ms | 21.4 GiB/s |

Marginal ~11 us per submitted copy; GPU-side bandwidth unaffected.
Pathological worst case (every block fully fragmented at 24 ranges, 28
blocks, ~7 fetch events/step) is ~80 ms against a 3.4 s step, ~2.3%;
realistic fragmentation with adjacent-range coalescing is well under 1%.

Compile behavior (tests 5, 13-14, real `mm.fetch_*` ops):

- Dynamo does NOT guard on pinnedness: pinned, pageable, and
  register-pinned same-shaped inputs reuse one graph.
- Swapping a same-shaped host flat causes ZERO recompiles -- both as a
  graph argument and as a FRESH CLOSURE per boundary. The production
  sampling-boundary recompiles have some other, undiagnosed guard cause.

Platform facts confirmed (tests 1, 6, 10-12):

- torch `is_pinned()` is blind to `cudaHostRegister`'d memory (flats and
  interior views both report False); CUDA transfers do not care.
- torch's caching host allocator rounds to power-of-two buckets (300 MiB
  alloc -> 512 MiB DXGI usage) and retains freed buffers page-locked until
  `_host_emptyCache()`. `cudaHostRegister` costs exactly its bytes and
  returns them immediately on unregister.
- Registering an already-registered page raises cudaError 712
  ("resource already mapped") -- page-exclusive ranges remain mandatory.
- `cudaPointerGetAttributes` is a truthful registry-free pinnedness oracle
  (interior pointers included) at ~14 us/call -- fine for setup-time
  asserts, not for hot paths.

Assumptions these measurements KILL (do not rebuild designs on them):

1. "Page-locking runs at 0.6-2 GB/s; pin/unpin is inherently seconds."
   False for RAM-resident pages. The historical seconds-per-boundary came
   from per-tensor churn (~1700 registrations), multi-GiB copies,
   unregister of ~10 GiB, gc/settle waits, and (plausibly) pagefile
   faults after unpinning under RAM pressure. Persistence remains the
   default as protection against pagefile eviction -- it is NOT a
   registration-speed constraint and NOT a compile requirement.
2. "Compile validity depends on pin/pack identity." False. Every
   pinnedness gate in the compile path is our own policy code.
3. "A native registration/submission helper might be needed." Rejected on
   direct evidence: registration does not parallelize and is already
   ms-scale; multi-range Python submission costs ~11 us/copy.

## Non-Goals

- Rewriting the planner, WDDM controllers, or pin policy in C++ (no native
  code anywhere in this plan).
- Supporting trainable streamed base weights. The canonical arena is for
  frozen base weights; LoRA/adapters remain normal trainable Parameters.
- Replacing the legacy per-Linear/bounce path for unsupported models.
- Making residency dynamically branch inside one compiled graph.
- Flipping any default before the complete validation matrix passes.
- Combining this work with the model-agnostic extraction (`ca8f496`);
  extract the corrected boundary afterward.

## Invariants

1. `pin_manager.py` remains the only pin authority. Every registered byte
   is budgeted against the DXGI NON_LOCAL headroom and released explicitly.
2. Canonical host storage is page-exclusive (cudaError 712 is real).
3. Build order is fixed and synchronous: allocate pageable, POPULATE
   (copy weights in), REGISTER, publish. Populate-before-register is 20x
   cheaper than register-then-populate (3 ms vs 67+33 ms per 600 MiB).
   There is no `PINNING_IN_PROGRESS` state, no wait-for-pin, no temporary
   pageable fallback: a block either publishes pinned or the build fails
   loud at attach with the budget message.
4. One-time canonicalization repointing is permitted: frozen base
   Parameters are repointed into the canonical flats once, at startup,
   after load/quantize/freeze and before LoRA attach, optimizer
   construction, or compile (assert this sequencing). After that, runtime
   promotion, demotion, sampling transitions, and plan changes MUST NEVER
   replace or repoint a Parameter.
5. Whole-model `.to()` is forbidden or intercepted for canonicalized
   leaves. A model-wide `.to(cuda)`/`.to(cpu)` round-trip would silently
   detach every Parameter from its flat (the drift `restore_view` existed
   to repair). Route through `memory_managed_to` / sidecars; add a
   regression test that a whole-model move cannot strand or detach a
   canonicalized leaf.
6. Residency is per-Linear and lives in manager-owned GPU sidecars keyed
   by stable leaf keys. Promotion copies host canonical bytes to a device
   sidecar (publishing only behind a recorded event); demotion releases
   it. Neither touches Parameters, host storage, pins, or arena metadata.
   Frozen-base demotion performs zero D2H write-back. Quantized leaves
   promote as wrapper-typed device tensors (the `_move_tensor_subclass`
   rebuild); because demote no longer touches Parameters, the whole
   FP8-demote block-invalidation class disappears.
7. The block is the transfer and compile unit; the leaf is the residency
   unit. A partially resident block transfers only its streamed bytes via
   a static multi-range plan into a compact device buffer -- one opaque
   ticket, one compact flat, all copies submitted inside the runtime call.
   Adjacent streamed ranges are coalesced. Fully streamed blocks keep the
   single-copy fast path. Python submission throughout (measured budget:
   ~11 us/copy; if a future measurement on different hardware crosses 5%
   of the H2D window, revisit -- do not pre-build native code).
8. Compiled callables are keyed by the execution/residency layout
   fingerprint: streamed/resident leaf keys per block, transfer spans and
   destination layout, checkpoint/SAC mode, dtype/quant identity, shape
   bucket, ring depth. They are INDEPENDENT of pinnedness, host-flat
   identity, and pack object identity (measured: identity swaps cause zero
   recompiles). A material plan change selects/builds another callable on
   purpose; it never touches the arena. Train and sample use distinct
   plans over the same arena. Sampling keeps an all-streamed fallback plan
   so an emergency demotion never rebuilds host storage mid-denoise.
9. `fetch_start` keeps a cheap runtime assert that its source belongs to a
   currently registered canonical allocation -- an invariant assertion
   against ONE registry (interval set or trusted arena handle), never
   `torch.is_pinned()` (measured blind to registered memory), and never
   the current three overlapping registries.
10. Admission policy vs mechanism are separate and both stay in Python:
    the PLANNER owns priority/order (streamed-first, value-ranked --
    deleting the layered budget arithmetic does not make iteration order
    the policy); the ALLOCATOR owns the live DXGI headroom check +
    spill reserve at each block's registration. No `plan_budgets`-style
    pre-computation layered on top.
11. Flags off preserve current behavior byte-for-byte. Real runtime
    behavior is config-backed; env vars only for synthetic diagnostics.

## Immediate Independent Work (before or parallel to the slices)

- **I1. Populate-before-register** in `pack_block_host` /
  `pin_register`: copy the leaves into the pageable buffer before
  `pin_tensor_in_place`. ~10 lines; saves ~1-1.5 s per full arena build.
  Ships on the CURRENT arena immediately.
- **I2. Diagnose the real boundary recompile** (BLOCKS Slice 5): run the
  Krea2 inference smoke (`scripts/smoke_krea2_inference_cuda.py`) across two
  sampling boundaries with
  `TORCH_LOGS=recompiles,guards` and identify the failing guard. The
  "fresh closures fail old guards" explanation is contradicted by
  measurement (test 14). Fix the actual cause, then delete
  `raise_dynamo_recompile_limit`. Until diagnosed, compile-lifetime design
  is built on a phantom cause.
  **Partial result (2026-07-10):** miniature real job
  `output/Krea2 I2 Boundary Recompile` completed one full train -> sample ->
  train boundary. The restored training step returned to ~3.07 s steady
  execution and `TORCH_LOGS=recompiles,guards` emitted no recompile or guard
  failure, so fresh boundary closures are not confirmed as a recompile source.
  The second boundary stopped before training restore: sampling's old global
  LoRA-count guard compared the new 160-leaf streamed set with the prior
  layout's 224 leaves and raised `lora_hijack_missing`. That guard/layout
  coupling is deleted by the later canonical-plan slices. Treat I2 as
  partially confirmed, keep `raise_dynamo_recompile_limit` as a temporary
  safety workaround, and do not block Slices 1-4 on a second-boundary guard
  that may disappear with the old lifecycle.
- **I3. RAM-pressure repin benchmark**: pin populated weights, create
  realistic RAM pressure, unpin, let Windows page, repin, measure. This is
  the remaining justification for whole-run persistence; record the result
  here. If repin is multi-second only after paging, persistence stays the
  default as pagefile protection; if repin stays cheap even under
  pressure, phase-scoped registration becomes a legal future
  simplification (not exercised by this plan either way). DONE
  (`scripts/bench_pin_assumptions.py` test 17, 2026-07-10): 600 MiB
  register/unregister/re-register, baseline 2.7 ms vs 4.0 ms after
  touching ~9 GiB of fresh pageable buffers (60% of then-available RAM,
  15.4 GiB) between unpin and repin -- 1.48x, not the multi-second
  pagefile-fault regime. Caveat: touching *new* pages doesn't reliably
  evict the just-freed ones under Windows' working-set manager the way
  Linux reclaim would; this measures "moderate allocation churn," not a
  guaranteed hard page-out. Reading is consistent with "repin stays cheap"
  but is not a strong disproof of the pagefile-fault risk under sustained
  multi-hour runs. Net: keep whole-run persistence as the default (cheap
  insurance, per Invariant 3/Explicitly-Rejected #4), but don't treat this
  result as ruling out phase-scoped registration as a *future* option --
  the measurement is suggestive, not conclusive.
- **I4. Benchmark evidence is permanent**: `scripts/bench_pin_assumptions.py`
  is committed; results tables live in this document.

## Target Components

### 1. Canonical host arena

Records keyed by stable `block_key`/`leaf_key` strings. Each block record
owns: one page-exclusive pinned host flat and its handle; immutable leaf
offsets, sizes, dtypes, shapes, quant roles; exact committed bytes. No
residency state, no generation counter, no live-module currentness test,
no borrowed/owned distinction -- packs ARE views over arena records.

Built once after load/quantization/freeze, before LoRA/optimizer/compile
(Invariant 4). Blocks that cannot be pinned within the DXGI budget fail
the build loud (Invariant 3) or are explicitly left to the legacy bounce
path -- they never silently enter strict compile.

### 2. Static multi-range transfer plans

`BlockTransferPlan`: references one arena block; lists streamed leaf
source ranges (adjacent ranges coalesced); maps them to compact
destination offsets; records the compact device size and leaf views;
immutable and fingerprintable. The fetch runtime submits all copies for a
ticket inside one opaque call, preserving the existing ticket/free-event
lifetime rules and the depth-K ring. Fully streamed block == one copy.

### 3. Manager-owned residency sidecars

`ResidencyState` per Invariant 6, with `promote`/`demote`/`reconcile
(plan)`/`resident_tensor(leaf_key)`. The existing WDDM controllers keep
making the decisions; only their mutation seam changes from
`promote_layer`/`demote_layer` over module storage to
`ResidencyState.reconcile` for opted-in regions. Legacy models keep the
current methods. Controller decisions are per-Linear today and remain
per-Linear -- this plan deliberately does NOT change residency
granularity, so the sim-verified autotune behavior carries over. (An
optional later A/B may check whether block-granular or canonical-order
residency simplifies anything without measurable loss; it is not a gate.)

### 4. Functional execution adapter

Eager and compiled block execution consume the same plan:

```text
leaf source = resident sidecar OR fetched compact-flat view
output = functional block math(inputs, leaf sources, LoRA tensors)
```

Krea2's `forward_streamed` / `assemble_leaf_args` / LoRA collection is
most of this seam. Resident leaves come from `ResidencyState`, never live
Parameters; LoRA entries are collected before stripping forward hijacks;
hooks/hijacks remain only in the legacy path; `_mm_ingraph_pack_source`
is unnecessary in the new path.

### 5. Compiled residency plans

Per Invariant 8. Host flats and sidecar tensors are explicit tensor
inputs at the compiled-region boundary where practical (measured: their
identity is then freely swappable). Controllers may tune during eager
warmup; the plan freezes before compile.

## Migration Slices

Every slice lands behind a temporary config-backed developer gate
(working name `layer_offloading_immutable_arena`, default `false`). The
current `layer_offloading_pinned_arena` path remains the reference until
Slice 7 retires or aliases it.

- **Slice 0 -- Remaining evidence** (mostly done). Ship I1. Run I2 (blocks
  Slice 5) and I3. Record same-commit reference numbers for the current
  arena (attach bytes/DXGI usage, boundary pin delta/time, synthetic step
  time, H2D bytes/wait, strict borrowed/owned counts). S, mostly done.
- **Slice 1 -- Canonical arena + one-time canonicalization.** Data model,
  exact pin accounting, populate-then-register build, sequencing assert
  (Invariant 4), whole-model `.to()` interception test (Invariant 5),
  state-dict round-trip, fail-closed on trainable/unsupported leaves,
  explicit-unload releases every range. M: 350-550 prod + 300-450 test.
- **Slice 2 -- Multi-range transfer plans + fetch runtime.** Range
  coalescing, compact destination views, extended fetch ops with
  guard-stable fakes, single-ticket semantics, fully-streamed fast path.
  Tests: mixed resident/streamed numerical parity, exact bytes and copy
  counts, depth-2 reuse hammer, zero recompiles on same-plan/new-storage,
  invalid range/pageable source/unknown ticket fail closed.
  M: 300-500 prod + 350-500 test.
- **Slice 3 -- Sidecar residency core.** `ResidencyState`, per-phase
  plans, planner seam adaptation, pin ledger invariant under all
  transitions, rollback on failed promotion, FP8 wrapper-typed sidecars.
  L: 450-700 prod + 400-600 test.
- **Slice 4 -- Krea2 functional eager integration.** Adapter contract,
  eager parity vs the current arena, LoRA gradient completeness, partial
  per-Linear residency, checkpoint recompute uses the same source map.
  L: 400-650 prod + 350-550 test.
- **Slice 5 -- Compiled train/sample plans + phase switching** (requires
  I2 resolved). Fixed per-Linear residency plans at block granularity,
  flats/sidecars as stable inputs, separate train/sample callables over
  one arena, boundary reconcile without host rebuild/repoint, all-streamed
  sampling fallback plan. Tests: zero graph breaks, bounded unique graphs,
  same-plan cycles reuse artifacts, arena pointers/registrations/ledger
  fixed across boundaries. L: 450-700 prod + 450-700 test.
- **Slice 6 -- Lifecycle, teardown, config, diagnostics.** True-unload vs
  disable distinction, temporary gate, smoke JSON fields (arena identity,
  sidecar bytes, plan fingerprint, range/copy counts, ledger deltas).
  M: 200-350 prod/config + 250-400 test.
- **Slice 7 -- Validation, migration, deletion.** Both-backend suites,
  A/B on identical commit/config/seed, real-run validation (user
  launches), then flip `layer_offloading_pinned_arena` to the new backend
  with a one-release rollback gate and execute the deletion list.

Totals: ~1,400-2,100 production lines, ~1,100-1,700 test lines, 7-11
focused implementation days + 3-5 validation days. No native slice.

## Deletions (after Slice 7 parity)

- Arena generations, `is_current`, `invalidate_block`, `restore_view`,
  `_mm_arena_block`/`_mm_arena_generation` module tags, and
  `try_borrow_pack` staleness forensics.
- `pack_block_host_from_flat`'s data_ptr borrow validation (the arena is
  the layout authority; nothing infers layout from live storage).
- The three pinnedness registries and their consultations
  (`_REGISTERED_HOST_PINS` incl. the dead duplicate in
  `manager_modules.py`, `_ARENA_BACKED_STORAGE_PTRS` + refcounting,
  `is_arena_backed` checks in `_profile_is_pinned` /
  `bounce_pool._is_pinned` / `fetch_start`) -> one canonical-allocation
  registry.
- Pack ownership/borrowing taxonomy (`owns_flat`, `borrowed_from_arena`,
  per-pack `pin_handle` rules) and enable-time `non_pinned_pack` /
  `arena_borrow_required` compile rejection.
- Runtime Parameter replacement in promote/demote and the arena branches
  in `detach`, `_unpin_module_weights`, `_move_params_to_cpu_and_pin`
  (incl. the DXGI pin-rollback transaction) for arena-capable models.
- The layered arena budget arithmetic in `attach` (`plan_budgets` weight
  strategy, `weight_tier_usable = max(...)` correction, `reconcile`
  pre-pass, `bounce_reserve = 0` special case, `_build_pinned_arena`
  budget crediting and pageable-retry self-healing).
- `raise_dynamo_recompile_limit`, once I2's real guard cause is fixed.
- `_mm_ingraph_pack_source` (new path).

## Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Functional seam misses a `module.weight` consumer | High | region audit; fail closed; legacy backend intact; direct parity tests |
| Whole-model `.to()` detaches canonicalized Parameters | High | Invariant 5 interception + regression test; audit `inference_resident`'s fully-resident branch |
| Sidecar lifetime races compute/compiled callables | High | publish behind recorded events; reconcile only at safe boundaries |
| Canonicalization sequenced after LoRA/optimizer capture | High | sequencing assertion + integration test |
| Boundary recompile cause unknown | Medium-High | I2 blocks Slice 5; do not design around the phantom cause |
| Arena exceeds DXGI budget on some model | Medium-High | per-block admission at registration; loud failure or explicit legacy fallback |
| RAM-pressure repin turns out disk-bound | Medium | I3 measures it; persistence is already the default policy |
| Sampling emergency demotion needs a new graph | Medium | prebuilt all-streamed fallback plan over the same arena |
| State dict sees CPU canonical weights while execution uses sidecars | Medium | base immutable; state-dict/save/resume parity tests |
| Compatibility window duplicates code | Medium | Slice 7 deletion gate, one-release rollback limit |

## Explicitly Rejected

- Native submission/registration runtime: rejected on measurement
  (~11 us/copy submits; registration does not parallelize).
- Async pin states (`PINNING_IN_PROGRESS`, wait-for-pin, temporary
  pageable fallback): populated registration is milliseconds; strictly
  sequenced construction makes the states unreachable.
- Block-granular residency as a foundation: no performance need (multi-
  range is cheap) and it would change sim-verified controller behavior.
  Permitted later only as an A/B-backed simplification.
- Contiguous-tail (canonical-order) residency as a foundational
  mechanism: superseded by cheap multi-range; adjacent-range coalescing
  keeps the benefit where it occurs naturally.
- Phase-owned arenas (repin churn under RAM pressure is the one regime
  where pinning IS expensive, pending I3).
- A full pinned shadow copy (~11-12 GiB duplicate host weights).
- Runtime Parameter repointing hidden in C++.
- One dynamic compiled graph branching on residency.
- Extracting the current arena into a generic manager before this refactor
  (would fossilize the generation/restoration boundary).

## Endpoint Acceptance Criteria

Correctness and ownership:

- Frozen base Parameter identity and CPU storage pointer unchanged after
  canonicalization across every promote/demote and train/sample cycle.
- Arena flat pointers, registered ranges, and host bytes never change at a
  phase boundary. Sidecar demotion performs zero frozen-base D2H.
- State dict/save/resume preserve base and adapter values; all expected
  LoRA gradients exist; frozen base receives none.

Memory safety:

- `weights` ledger and DXGI usage flat across phase boundaries.
- Dedicated VRAM moves by planned sidecar bytes only, above the WDDM hard
  margin. Per-block admission; no over-pinning, no silent fallback.
- Explicit unload returns every registration and ledger byte.

Compile and transfer:

- Zero graph breaks in strict compiled train/sample regions; repeated
  same-plan phase cycles add no unique graph; host-storage identity swaps
  cause zero recompiles (measured baseline, now enforced by test).
- Fully streamed blocks use one H2D copy; partial blocks copy exactly the
  planned streamed bytes through coalesced ranges; ticket depth and
  device-buffer lifetime stay bounded through forward and backward.

Performance:

- Steady training step time and sampling throughput within 5% of the
  current-arena reference on the same commit/config/seed.
- Boundary pin/unpin work is zero; boundary wall time well below the old
  1.2 s / 2.0 s repin baseline (and below one avoided recompile, once I2
  lands).

Compatibility:

- Both gates off: byte-identical current behavior. Unsupported
  model/wrapper/hardware fails closed or uses the explicit legacy path.
  No runtime policy depends on an environment variable.

## Validation Ladder

- **A. CPU/simulated seams**: layout, keys, offsets, coalescing,
  fingerprints; eligibility; state dict; planner byte parity; teardown
  ownership; failure rollback.
- **B. Small real-CUDA tests**: page-exclusive registration and exact
  DXGI/ledger deltas; sidecar promote/demote; multi-range copy
  correctness and submission counts; depth-2 lifetime hammer;
  `torch.compile(fullgraph=True)` graph/recompile counters; TorchAO FP8
  forward + LoRA backward parity.
- **C. Krea2 synthetic smokes**: `scripts/smoke_krea2_train_cuda.py` for
  arena training and `scripts/smoke_krea2_inference_cuda.py` for immutable
  sampling; train -> sample ->
  train in one process; A/B both backends same shape/seed.
- **D. Real job validation** (user launches/approves): >=200 steps with
  sampling boundaries, two resolution buckets, perf-log digest for steady
  windows, no paging/no shared-budget crash/no registered-range leak,
  save + resume once.

Do not hunt test-order leaks; follow the repository policy and record on
`f2aceba` after the two confirmation runs.

## Dependency And Ticket Coordination

- `628b0cb`: owns implementation status, benchmark results, blockers, and
  slice handoffs for this plan.
- `534ea49`: the current arena remains the measured reference until
  Slice 7; do not close it because this plan exists.
- `3ca8a7b`: fetch/checkpoint/ordering invariants are reused, not
  rewritten; the fetch-op surface grows multi-range spans in Slice 2.
- `ca8f496`: pause broad extraction; resume against the canonical-arena /
  residency / execution-plan interfaces after Slices 1-5 stabilize.
