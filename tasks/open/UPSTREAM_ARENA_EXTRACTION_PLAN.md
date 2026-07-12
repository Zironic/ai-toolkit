# Arena offload: pre-PR refactor + upstream extraction

> **git-bug:** see the ticket titled "Arena offload: pre-PR refactor and upstream
> extraction". Strategy/rationale: `../../docs/decisions/UPSTREAM_PR_PLAN.md`.
> Status lives in the ticket; this file is the durable plan.

This plan replaces the retired A/P/C/D upstream stack and the four
`MODEL_AGNOSTIC_SUBPLAN_*` docs (all in `tasks/done/`). It covers both the
in-fork refactor and the upstream extraction, because the two constrain each
other: the refactor exists only to make the extraction small.

## Goal

Ship the block-native arena offload implementation to `ostris/ai-toolkit` as a
reviewable PR, without dragging in the fork's expanded legacy manager,
profiling machinery, experimental per-linear compatibility paths, or
Krea-specific orchestration.

## Why the old stack died

The retired stack led with PR C (bounded training streaming core) and stacked
PR D (native FP8 training) on top of it. Both are built on `_BouncingLinearFn`,
the per-linear streaming autograd function - D's own plan says "without C's
hooks there is nowhere for this code to run".

Commit `3dd7f38` retired that backend: the immutable runtime is now the sole
transformer backend on the fork's active path. Upstreaming C would mean
upstreaming code the fork no longer runs, and D would land inside it. The arena
is the payload PR now.

PR P (DXGI probe + pin crash guard) was *not* wrong - it is preserved verbatim
as Stage 1 below, including its upstream-consumer audit. PR A (resident
sampling) is preserved as a deferred Stage 4; see the note about its collision
with the two-timescale residency work.

---

# The upstream stack

| Stage | Theme | Depends on |
| --- | --- | --- |
| 1 | Pinned-memory budget governance (bugfix framing) | nothing |
| 2 | Pre-PR refactor, in-fork only (no PR) | nothing |
| 3 | Arena offload as an additional backend | 1 (hard), 2 |
| 4 | Resident + native-FP8 sampling | deferred; reassess after 3 |

Stage 1 can be built in parallel with Stage 2.

---

# Stage 1 - Pinned-memory budget governance

**This is a bugfix PR, not arena support infrastructure.** It stands alone and
must be pitched alone.

## The upstream bug

Audited against upstream `main` (`f63221e`):

- `MemoryManager.attach` (`toolkit/memory_management/manager.py`) selects
  offload layers with `offload_percent` default 1.0 and **no size accounting of
  any kind** - not model bytes, not host RAM, not any GPU/driver budget.
- Every selected layer runs `_move_params_to_cpu_and_pin` ->
  `_ensure_cpu_pinned` (`manager_modules.py:204`) -> `pin_memory()` /
  `_pin_inner_tensors` (recursive for quantized subclasses). A 24 GB BF16 model
  pins ~24 GB of host memory unconditionally.
- The only guard is `except RuntimeError: pass` at the pin call, but the real
  failure is deferred: pins commit against the WDDM shared budget, and
  exhaustion surfaces later as a raw `cudaErrorMemoryAllocation` in an
  unrelated allocation. On Linux the equivalent is pinning into RAM exhaustion
  and thrashing the host.

Measured locally: upstream's offloader **cannot run a 12 GB model on this box**
without the over-pinning crash. This is the load-bearing justification for the
sensor - it is not a policy refinement that can degrade to `mem_get_info`,
because the degraded path is the crash.

**Pitch:** "unbounded pinning can hard-crash Windows runs and thrash Linux
hosts; this adds a budget probe and refuses to pin past the real limit,
degrading to pageable offload instead."

## Contents

Four leaf modules (no `toolkit` imports today - verified extractable as-is):

```text
pin_manager.py     752 lines   pinned-byte ledger, tiered budget, explicit release
vram_budget.py     819 lines   cross-platform device-free (NVML-backed)
nvml_meminfo.py    234 lines   cross-platform; the only true free-VRAM signal
dxgi_meminfo.py    493 lines   Windows-only NON_LOCAL shared-budget probe
```

`dxgi_meminfo` must no-op cleanly on non-Windows. Expect that to be the review
question; answer it in the PR description, not in review.

## Scope discipline

The consumer stays a **minimal clamp**: a running pinned-bytes tally checked at
`_ensure_cpu_pinned` / `_move_params_to_cpu_and_pin`; once headroom is
exhausted, leave remaining tensors pageable and warn **once**. Nothing else -
no transactional pin, no unpin relief, no margin governance. The intelligence
arrives with Stage 3. The clamp survives as its last-resort floor guard.

**Do not drag in `bounce_pool.py`** (1272 lines). `manager_modules` imports it,
but it is a throughput optimization, not part of the crash fix. Before starting,
check whether `pin_manager.plan_budgets` / `reconcile` have entangled the
bounce-pool tiers with the weight tier (`manager.py:684-742`) - untangling that
is the real work of this stage and should be scoped before the first commit.

## Acceptance

- Upstream `attach` with a model larger than the pin budget completes, degrades
  to pageable, warns once, and does not crash.
- No behavior change when the budget is not exhausted.
- Clean no-op on Linux/non-DXGI; NVML absence degrades without raising.

---

# Stage 2 - Pre-PR refactor (in-fork, no PR)

The arena implementation currently cannot be extracted: Krea constructs arena
internals directly, the shared trainer reads private `_mm_*` fields, and the
arena path calls into `MemoryManager` for planning, control, and diagnostics.

## Dependency rules (three tiers)

The host-memory layer is shared by both backends. It is **not** part of the
arena package - if it were, Stage 1's fix would be trapped inside a package
upstream has not merged.

```text
host_memory layer (pin_manager, vram_budget, nvml_meminfo, dxgi_meminfo)
    imports neither backend

arena_offload  -> may import host_memory; must NOT import MemoryManager
MemoryManager  -> may import host_memory; must NOT import arena_offload
Krea2          -> must not construct arena internals
shared trainer -> must not inspect arena private fields
adapters       -> must not own memory policy
```

An import-boundary test enforces the two "must not"s.

## Target layout

```text
toolkit/memory_management/
├── manager.py            (unchanged upstream-compatible per-linear backend)
├── manager_modules.py    (unchanged)
├── pin_manager.py        (host_memory layer - Stage 1)
├── vram_budget.py        (host_memory layer - Stage 1)
├── nvml_meminfo.py       (host_memory layer - Stage 1)
├── dxgi_meminfo.py       (host_memory layer - Stage 1)
└── arena_offload/
    ├── __init__.py
    ├── api.py            (the only supported integration surface)
    ├── arena.py          (from canonical_arena.py)
    ├── layout.py         (static packing, split from ingraph_stream.py)
    ├── transfer.py       (fetch ring, split from ingraph_stream.py)
    ├── residency.py
    ├── policy.py         (NEW - arena-native planner + controller)
    ├── runtime.py        (from immutable_runtime.py)
    └── adapters/
        ├── base.py
        └── single_stream_mmdit.py
```

No further decomposition. The objective is a coherent extractable package, not
maximal file count.

## Phase 0 - Baseline, then a throwaway dry run

**0a. Record the behavioral baseline** from the current smart Krea path, using
existing working configs (no new benchmark framework): arena block count and
canonical bytes; initial resident bytes; streamed block count; first train and
sample compile durations; steady-state step time at tested resolutions;
sampling time; ring usage and transfer stall; FP8 modes; train->sample->train
transition; OOM/WDDM relief behavior. Add one checked-in smoke config
representing the supported arena path.

**0b. Do the extraction dry run NOW, not at the end.** Branch from
`ostris/main`, copy the arena files over, and see what fails to import. This is
an afternoon's work and it is the cheapest, highest-information action
available - the entire plan below is a *prediction* about what upstream will
object to, and this tests the prediction before eight phases of work are spent
on it. Let the breakage list re-order the phases below.

## Phase 1 - Arena package + facade, behavior-preserving

Move the components under `arena_offload/` with compatibility re-exports at the
old paths. Add `api.py` and have it construct the same objects Krea constructs
today. The facade may still delegate planning to legacy code at this point.

Public surface:

```python
runtime = prepare_arena_offload(transformer, device=..., adapter=..., config=...)
runtime.finalize(network)                        # after LoRA attach
with runtime.training_step(shape_key=..., step_num=...): ...   # spans fwd+bwd
with runtime.sampling_session():                 # one sampling run (all images)
    with runtime.sampling_image(shape_key=..., cold_working_bytes=...): ...
runtime.place_permanent_modules(device, dtype)   # Phase 5, when it has a caller
runtime.diagnostics() -> dict
runtime.close()
```

Helpers: `get_arena_runtime`, `is_arena_offloaded`, `is_memory_managed`,
`memory_runtime_owns_compile`, `close_arena_offload`. These unwrap
Accelerator/DDP wrappers. `is_memory_managed()` is true for either
`_memory_manager` or an arena runtime, replacing bare `hasattr` checks in shared
code without touching the legacy manager.

The training context **must** span forward and backward - checkpoint
recomputation re-enters the block runtime during backward.

Sampling needs **two** contexts, not one. The SAMPLE program is entered per
image, but the TRAIN program must be restored once per *run*: re-activating
TRAIN between images would reconcile residency back to the training plan and
churn the sidecars for nothing. `sampling_session()` owns the restore;
`sampling_image()` owns the per-image SAMPLE activation and is what the model
calls.

One neutral attribute, `transformer._arena_offload_runtime`, is what **shared
code** reads. It does not yet *replace* the eight `_mm_*` fields: the immutable
runtime (`_mm_immutable_protected_training_leaf_keys`,
`_mm_immutable_training_plan`) and the legacy manager (`_mm_immutable_backend`,
`_mm_residency_state`, `_mm_immutable_smart_plan`, ...) both still read them
directly, and cutting them is Phase 2 + Phase 7 work, not Phase 1's. Phase 1
publishes both: the facade for shared code, the `_mm_*` fields for the internals
that have not been cut over. No shared-code reader of a `_mm_*` field survives.

**Acceptance:** Krea no longer imports `CanonicalArena`, `ResidencyState`,
`ResidencyPlan`, or `prepare_immutable_runtime`; no `_mm_*` or
`_immutable_runtime` read survives in `BaseSDTrainProcess`, `SDTrainer`, or
`krea2.py`. Existing smart Krea tests and the training smoke behave identically.

## Phase 2 - Arena-native policy (the expensive phase)

Phase 2 is complete. The arena runtime now owns block-granular planning and
the live two-timescale controller without importing `MemoryManager`. The
controller was validated on a compiled, mixed-resolution Krea2 run: 59 clean
post-warmup windows (30 x 512-class, 29 x 768-class), zero allocator retries,
zero new Dynamo frames, and zero policy errors. The durable controller design
below is the starting contract for Phase 3; run details remain on git-bug
ticket `0c577ef`.

Create `arena_offload/policy.py` and cut these calls:

```text
MemoryManager.attach_smart_training_immutable
MemoryManager.smart_training_plan
MemoryManager.prepare_training_memory_for_shape
MemoryManager.training_runtime_diagnostics
MemoryManager.training_pinned_keys_for_keep_last
MemoryManager.inference_resident          <- do not forget this one
```

`inference_resident` is easy to miss: Krea's sampling path routes through it,
and `runtime.sampling_image()` must absorb it.

`prepare_training_memory_for_shape` (the pre-step demote guard) is deliberately
**still outside** `runtime.training_step()` after Phase 1, and moving it in is
not free: today it runs *before* `torch.cuda.reset_peak_memory_stats`, so its
own allocations are excluded from the measured step peak. Folding it into the
context's `__enter__` silently changes what the learned per-shape peak means -
and that peak is exactly what the residency controller consumes. Move it in
deliberately, with the reset ordering fixed at the same time, not as a
by-product of the refactor.

**Two-timescale controller: no conflict, and a free head start.** Verified
2026-07-12. The two-timescale policy is *already built and green* -
`allocator_allowance_bytes`, `cap_bytes_for_live`, `cap_can_host_promotion`,
`residency_promote_ok`, and `residency_fsm_step` all live in `vram_budget.py`,
with `tests/test_residency_two_timescale.py` passing (20 tests).

**It has no production caller.** Grep confirms the FSM and both gates are unwired
dead code; only `cap_bytes_for_live` has a caller, in `inference_resident`
(`manager.py:6024` - the sampling-side cap reclaim, Slice 1).

**But there IS live training-side residency policy in `manager.py`** - the
worst-shape promotion veto (`manager.py:4357-4421`), wired into the training
autotune loop. Before committing a promotion it predicts the worst *measured*
resolution's cohabitation peak and vetoes if that would page (residency is
global, the working set is not, and no retry counter catches a WDDM page-out).

So the picture splits cleanly, and Phase 2 must treat the halves differently:

| Piece | Where | Wired? | Phase 2 action |
| --- | --- | --- | --- |
| FSM + promote precheck + climb gate | `vram_budget.py` (pure) | **no** | call as-is from `policy.py`; nothing to port |
| `cap_bytes_for_live` | `inference_resident` | yes (1 call) | absorbed by `runtime.sampling_image()` |
| Worst-shape promotion veto | `manager.py:4357` | **yes** | **port and rethink - see below** |

The veto is the one real carry-over, and it does not port cleanly:
`_next_promotion_layer_bytes` (`manager.py:3368`) mirrors
`_promote_training_layer`'s pinned-first-then-smallest sort - it selects a
**layer**, not a block. That is precisely the per-linear/block impedance mismatch
Phase 2 exists to remove. In `policy.py` the unit of promotion is a whole
canonical block, so the veto's "next promotion bytes" input becomes the next
block's bytes and the guard/promoter agreement has to be re-established on that
basis.

The block-granular veto is now validated on the arena path. A compiled,
mixed-resolution Krea2 run completed 59 clean post-warmup windows while the
controller changed whole-block layouts: allocator retry delta remained zero,
Dynamo added no frames, and policy errors remained zero. Keep the veto and the
promoter coupled to the exact same candidate block; do not reintroduce the
legacy per-Linear predictor during extraction.

`policy.py` is therefore not a from-scratch controller: it is the *wiring* the
two-timescale plan always called for (its step 2, "controller wiring at the
existing phase-boundary hook"), built once against the arena's block records
instead of retrofitted into `manager.py`. That plan's open questions (training
slack-pad sizing, `Kclean` / `Kverify` / `N`) carry over to `policy.py` unchanged.

Plan by stable block key, never by Python module identity:

```python
ArenaBlockInfo(key="blocks.0", canonical_bytes=..., streamed_bytes=...,
               order=0, protected=False)
```

The ring estimate must use the actual immutable transfer unit (canonical block
compact-transfer size x active ring depth), not the legacy planner's
largest-individual-Linear estimate.

Protected (checkpoint-retained) blocks arrive as `protected_block_keys:
frozenset[str]`, derived by the adapter/prep code. The policy must not search
for `_checkpoint_keep_last` or `.blocks` through module traversal.

Move only the controller behavior the immutable runtime uses: per-shape
working-set peak learning, safety margin, whole-block promote/demote, and the
WDDM dedicated-memory cliff accounting. Shape peaks are stored as
layout-independent working bytes (peak allocated minus resident and ring
bytes) so residency transitions preserve the worst-shape envelope;
compile/retrace still invalidates it. A promotion remains provisional during
verification, under a fixed cap. Allocator GC/retries immediately roll it back.
Anti-chatter is a deadband in worst-shape allocator slack: promotion requires
0.95 * cap minus predicted live bytes to exceed the candidate block plus the
headband. Cooldown remains a separate, temporary settling mechanism.

Bootstrap avoids a long one-block climb. Accumulate monitored minimum
physical free across the first two logical steps, then bootstrap at the third
step boundary and compute
``min_free - WDDM hard floor - 1 GiB``. Select as many whole blocks as
fit and publish them in one provisional layout transaction under the unchanged
cap. The normal first-window GC/retry verification rolls the entire bootstrap
batch back if it overshoots.
**This is a port of the WDDM controller, not a rewrite** - the
hard-won cliff behavior is preserved as-is. Leave behind: per-linear
promote/demote, `_layer_memory_manager`, bounce-pool trace recovery, resident
per-linear hooks, random/interleaved Linear selection, per-linear offload IDs
and FP8 wrapper flags, block prehooks around legacy bouncing Linears. The
immutable runtime has a stable known block order and needs no execution-trace
discovery.

`runtime.diagnostics()` returns one stable dict (canonical/resident/streamed/ring
bytes, working-reserve estimate and measured peak, device total/free, plan
fingerprint, last policy action). Shared logging prints it; it must not
reconstruct it from private fields.

**Acceptance:** `arena_offload/` imports no `MemoryManager`. Residency decisions
use block keys. Controller actions add/remove complete blocks only. Legacy
manager tests pass unchanged.

## Phase 3 - Destination-first transactional arena construction

Today the arena canonicalizes and repoints blocks one at a time; on a mid-block
failure the earlier Parameters stay repointed at released storage, and
[canonical_arena.py](../../toolkit/memory_management/canonical_arena.py) says so
in a comment ("cannot un-repoint a Parameter").

This phase replaces that canonicalization path outright. Do not first make the
existing full-model repack transactional and then replace it: that would retain
a second model-sized copy as a knowingly temporary implementation.

### 3a - Extract the layout prerequisite

Move the static packing half of `ingraph_stream.py` into `layout.py` before
building the transaction. It owns leaf inspection, final packed-layout
descriptions, alignment and byte ranges, typed destination views, Parameter
views, supported-wrapper reconstruction, and packed-storage release. It owns no
CUDA streams, queues, profiling, traces, hooks, or transfer lifetime.

Make the leaf list data-driven while extracting it: carry `leaf_descriptors` +
`native_fp8_eligible` instead of name-specific flags and string branches. This
is the whole quantization cleanup; it is needed so the construction transaction
can expose destinations for every supported weight, scale, and bias without
encoding Krea2 internals.

### 3b - Build the generic transaction

The arena package exposes a prepared build rather than making Krea2 construct
arena internals:

```python
build = runtime.prepare_canonical_storage(transformer, adapter)
loader.populate(build.destinations)
build.commit()
```

The compatibility path for loaders that still materialize model tensors is:

```python
build = runtime.prepare_canonical_storage(transformer, adapter)
build.populate_from_model()
build.commit()
```

The transaction has three explicit stages:

- **Prepare** (no model mutation): inspect the architecture adapter; calculate
  every block's final packed layout; allocate the final page-exclusive host
  flats; expose typed destination views for every weight, scale, and bias; and
  retain the original Parameter objects. Populate the destinations from either
  a direct checkpoint/cache loader or `populate_from_model()`. Register the
  populated flats with `cudaHostRegister`, then validate wrapper reconstruction.
  On any failure, release every prepared flat, unregister every pin, leave every
  Parameter unchanged, leave no runtime marker, and raise a clear setup error.
- **Commit** (only after all blocks prepare): construct quantized wrappers and
  Parameter views over the final flats; repoint all canonical leaves as one
  atomic publication; publish block records and the runtime marker; and install
  the whole-model movement guard. If commit fails, restore the retained original
  Parameters before releasing storage.
- **Rollback:** if commit began, restore the original Parameters before releasing
  prepared storage. If it did not, release the prepared flats and pins without
  touching the model.

The direct source is the performance path: loaders write once into final arena
storage and never materialize a second full canonical model copy. The
`populate_from_model()` source is a compatibility path, not the implementation
that direct-capable loaders should use. It keeps arena offload available to
other supported models without requiring every loader to change at once.

**No silent fallback to per-linear** after the user explicitly selected arena
offload. Fail clearly, leave the model intact and eagerly executable.

Failure-injection tests: unsupported layout mid-stack; pin failure mid-stack;
allocation failure after earlier blocks prepared; direct population failure;
wrapper validation failure; commit failure after the first repoint. Assert
Parameter identity/storage restored, data pointers valid, no pin-ledger leak,
no runtime marker, model still runs eagerly. Test both population sources and
assert that the direct source does not allocate or copy a second canonical
model-sized payload.

## Phase 4 - Split transfer out of `ingraph_stream`

Cheaper than it looks. Commit `3dd7f38` already retired the legacy in-graph
paths, and Phase 3 has already moved packing/views/leaf plans into `layout.py`.
Move the remaining fetch-ring half into `transfer.py`. The bouncing
Linear execution, per-linear trace recording, sampling monkeypatches, and
bounce-pool scheduling are **already gone**.

- `transfer.py`: compact transfer plans, range tensors, block fetch custom ops,
  fetch ring alloc, start/wait/free lifetime, backward-recompute lifetime,
  checkpoint context helpers for the immutable train trunk.

The one real link left to cut is `ingraph_stream.py:23` importing
`manager_modules`, which feeds `is_streamed_module`'s `_layer_memory_manager`
check. Also remove the process-global setup calls from
`BaseSDTrainProcess.__init__` (trace/profile/prefetch); the runtime owns one
transfer runtime. Legacy `MemoryManager` keeps its own process-global state.

## Phase 5 - Krea2 direct population and backend selection

```python
if model_config.arena_offload:
    runtime = prepare_arena_offload(
        transformer, device=self.device_torch,
        adapter=SingleStreamMMDiTAdapter(),
        config=ArenaOffloadConfig.from_model_config(model_config),
    )
else:
    MemoryManager.attach(transformer, self.device_torch,
                         offload_percent=..., ignore_modules=...)
```

Wire both Krea2 materialization paths - ranged checkpoint loading and quantized
cache loading - to populate Phase 3's final typed destination views directly.
They select and feed the generic arena transaction; they do not calculate
layouts, allocate flats, construct wrappers, publish block records, or manage
rollback. Retain `populate_from_model()` for arbitrary supported models whose
loaders have not adopted direct population.

Preserve the semantic load order: incorporate the assistant LoRA into the frozen
base before its final canonical representation is committed; produce the final
quantized or unquantized canonical values directly in prepared arena
destinations; commit and freeze the canonical base; move permanent noncanonical
modules; (trainer attaches network); finalize runtime; build optimizer; compile
on first use. The compatibility path may continue to materialize and quantize
the transformer before `populate_from_model()`, but Krea2's direct-capable paths
must not take that second-copy route.

Do not call `transformer.to(...)` after canonicalization; route whole-model moves
through `runtime.place_permanent_modules(device, dtype)` and keep the arena's
`.to()` guard. Audit every later whole-transformer move.

**Text encoder keeps using the existing per-linear `MemoryManager`.** Arena
offload applies only to the supported transformer blocks.

**Acceptance:** Krea memory code = backend selection + adapter selection +
config conversion + loader destination population + one sampling context call.
No arena construction, layout calculation, wrapper construction, plan
conversion, residency manipulation, or compile logic in the Krea folder. Both
Krea loader paths populate final arena storage directly; the generic
compatibility source remains covered independently.

## Phase 6 - Shared trainer reduces to generic lifecycle calls

`BaseSDTrainProcess`, after the network is applied:

```python
runtime = get_arena_runtime(self.sd.unet)
if runtime is not None:
    runtime.finalize(self.network)
```

Compile integration: `is_unet_offloaded = is_memory_managed(inner_unet)`, and
skip generic block compile when `memory_runtime_owns_compile(inner_unet)`.
Compile ownership must be **exclusive** - the current `_mm_immutable_backend`
special case exists because it isn't. Generic block compile is unchanged for
non-arena models.

Teardown: `close_arena_offload(self.sd.unet)` - drain streams, clear sidecars,
unregister/release canonical host storage, restore guarded movement, remove the
marker.

`SDTrainer`: one generic context around the whole per-batch forward/backward
region, `contextlib.nullcontext()` when there is no runtime. Keep the diff that
shape. Remove trainer calls into legacy offload-step tracing, execution leases,
pre-step guards, post-step controller updates, and private arena attributes. The
outer loop keeps catching OOMs as it does today.

**Acceptance:** no `_mm_immutable_*` or `_immutable_runtime` anywhere in shared
trainer files; no arena/residency/planner/adapter imports there.

## Phase 7 - Remove obsolete branches from the legacy manager

Only after parity. Delete `attach_smart_training_immutable`, the immutable
branches in `attach_smart_training`, `offload_ids` -> `ResidencyPlan`
conversions, canonical-sidecar relief through `MemoryManager`, immutable-runtime
inspection in legacy diagnostics, `_mm_immutable_*` handling, and legacy planner
state that existed only for the arena.

Retain `attach`, `detach`, the Linear/Conv/OstrisLinear managers, legacy
transformer percentage offload, text-encoder offload, and every model that
depends on per-linear management. **Do not** do a general cleanup of the manager
as part of this work.

## Phase 8 - Configuration

```python
@dataclass(frozen=True)
class ArenaOffloadConfig:
    enabled: bool
    fp8_forward: bool
    fp8_backward: bool
    fp8_sampling: bool
    compile_blocks: bool          # derived from existing compile settings
```

User-facing (all default false):

```yaml
model:
  arena_offload: false
  arena_fp8_forward: false
  arena_fp8_backward: false
  arena_fp8_sampling: false
```

`arena_fp8_sampling` stays independent of the training toggles. `compile_blocks`
is derived - arena offload must not introduce a second public compile system.

**No new user-facing controls** for residency percentages, working reserves,
WDDM margins, pinned budgets, prefetch depth, trace/profile capture, controller
cadence, or ring size. Those are automatically selected implementation details.

Temporary fork compatibility (remove only after the extraction validates):

```text
layer_offloading_smart          -> arena_offload
layer_offloading_fp8_forward    -> arena_fp8_forward
layer_offloading_fp8_grad_input -> arena_fp8_backward
layer_offloading_fp8_sampling   -> arena_fp8_sampling
```

Existing manual reserve/WDDM/trace/profile/prefetch options may still be parsed
for regression comparison, but they are not fields on `ArenaOffloadConfig`.

Validation: `arena_fp8_*` without FP8 weights -> ignore with one warning.
`arena_offload=true` on an unsupported architecture -> fail during preparation
with an architecture-support message. `arena_offload=true` plus a transformer
percentage -> arena wins for the transformer; the percentage stays meaningful
only for the legacy backend.

---

# Stage 3 - The arena PR

Extract onto a fresh branch from `ostris/main` after Stage 1 is open.

Expected diff:

- **New:** `toolkit/memory_management/arena_offload/**`
- **Small edits:** `memory_management/__init__.py`, `config_modules.py`,
  `krea2/krea2.py`, `jobs/process/BaseSDTrainProcess.py`,
  `extensions_built_in/sd_trainer/SDTrainer.py`, UI config for the four toggles

Because the host-memory layer lands in Stage 1, this claim is finally true. It
was **not** true of the original plan, which quietly required ~2300 lines of
NVML/DXGI/pin-ledger infrastructure that upstream does not have.

**Acceptance:** upstream tests pass with arena offload disabled; per-linear
offload still works; generic block compile still works on non-arena models; Krea
arena smoke training and sampling pass; upstream `manager.py` is not replaced or
substantially rewritten; no fork profiling logs, DOP logic, checkpoint-writer
changes, or unrelated trainer edits in the diff.

---

# Stage 4 - Resident + native-FP8 sampling (deferred)

Formerly PR A, and formerly the lead PR because it was independent and easy.
It is no longer free: it centers on `inference_resident`, which Phase 2 rewrites
and which the two-timescale residency work is actively changing. Reassess after
Stage 3 lands, when there is exactly one owner of that code.

---

# Validation

## Unit

- **Arena admission:** direct destination population without a second canonical
  payload; compatibility population from an existing model; transactional
  success; mid-block layout, population, wrapper, and pin failures; no Parameter
  mutation on failed prepare; restored originals after failed commit; no pin
  leak; safe close after partial prepare.
- **Planner:** plans by block key; accounts for permanent singleton bytes;
  accounts for actual block ring depth; protects requested trailing blocks; never
  emits partial-block initial plans; deterministic for equal-size blocks.
- **Controller:** layout-independent peak learning; worst-shape allocator-slack
  headband; immediate rollback on allocator GC/retry; temporary cooldown;
  demotion under pressure; arena-owned abnormal-exit fetch drain and OOM
  rollback; compile invalidation; no per-linear state or actions.
- **Lifecycle:** prepare before LoRA; finalize after LoRA; compatible double
  finalize; incompatible double finalize fails; training context spans checkpoint
  recompute; no residency publication during execution; train->sample->train;
  close rejects active execution.
- **Compile ownership:** arena compiles functional kernels once; generic block
  compile skipped; legacy per-linear models still use generic block compile;
  compile-disabled arena runs eagerly with identical semantics.
- **Import boundary:** nothing under `arena_offload` imports `manager` or
  `manager_modules`; `manager` does not import the arena runtime.

## Integration

Krea arena BF16; arena + FP8 forward; arena + FP8 forward/backward; precise
training + FP8 sampling; legacy per-linear transformer offload; arena transformer
+ legacy TE offload; arena disabled; multiple resolution buckets; DOP /
multi-forward; sampling before training and after several steps; OOM-induced
whole-block demotion; Windows WDDM path; non-Windows CUDA fallback.

## Performance

Against the Phase 0a baseline: first train/sample compile duration, steady-state
step time, H2D transfer time, transfer stall, GPU utilization, resident bytes,
ring bytes, peak allocated/reserved. **Not ready for extraction** if it restores
multi-minute compile behavior, introduces repeated recompilation, or materially
worsens steady-state transfer overlap.

---

# Non-goals

1. Removing upstream's per-linear manager.
2. Converting other upstream models to arena offload.
3. Arena-offloading the text encoder.
4. ConvRot4 / ConvRot8.
5. **A quantized-layout codec seam.** An earlier draft proposed a
   `WeightLayoutCodec` Protocol with codec ids and a synthetic three-leaf test.
   It has no real second consumer - non-goals 4 and 2 rule out the only
   candidates - and it is the one change that touches the immutable ABI, source
   assembly, and native-FP8 eligibility (i.e. real risk to a working fast path)
   for zero behavior payoff. It also *hurts* the PR by adding surface area with
   no use case. The legitimate kernel of the idea - not hardcoding FP8 in names -
   is handled by the data-driven leaf list in Phase 4.
6. Rewriting the WDDM controller. Phase 2 **ports** it.
7. Replacing generic block compile for non-arena models.
8. Cleaning up unrelated fork memory-manager experiments.
9. Automatic fallback from arena to legacy offload.
10. Exposing memory-policy tuning in the UI.
11. Generalizing the architecture adapter beyond what a second real model needs.
12. Moving unrelated Krea loading, scheduler, TE, or sampling code.

---

# Order of work

1. Baseline (0a) and the throwaway dry run (0b). Let 0b re-order what follows.
2. Stage 1 in parallel: extract the host-memory layer, clamp upstream's pin sites.
   (The two-timescale FSM is pure, unwired policy in `vram_budget.py` and rides
   along with it. The worst-shape promotion veto does NOT - see Phase 2.)
4. Phase 1 - arena package + facade.
5. Phase 2 - arena-native policy; cut the `MemoryManager` calls.
6. Phase 3a - extract layout and the data-driven leaf description API.
7. Phase 3b - destination-first transactional construction, including the
   `populate_from_model()` compatibility source.
8. Phase 4 - split the remaining transfer runtime out of `ingraph_stream`.
9. Phases 5-6 - wire Krea's direct loader population and collapse Krea/shared
   trainer orchestration to the facade.
10. Phase 7-8 - remove obsolete legacy branches; settle config.
11. Full validation matrix.
12. Stage 3 - real extraction, using the dry run's diff as the source.

# Readiness checklist

- [ ] Upstream per-linear `MemoryManager` behavior still available.
- [ ] Arena installs no per-linear wrappers on canonical blocks.
- [x] `arena_offload/` does not import the legacy manager (enforced by test).
- [ ] Host-memory layer imports neither backend.
- [ ] Krea constructs no arena or residency objects; setup is one facade call.
- [ ] Shared trainer inspects no arena private state.
- [ ] One context spans arena training forward and backward.
- [ ] Arena runtime exclusively owns its functional compilation.
- [ ] Generic block compile unchanged for non-arena models.
- [x] Planning uses block keys and actual block transfer sizes.
- [x] Controller transitions operate on complete blocks.
- [ ] Canonicalization failure leaves the model untouched.
- [ ] Whole-model movement cannot detach canonical weights.
- [ ] Training and sampling FP8 controls independent.
- [ ] Text-encoder offload still uses the existing manager.
- [ ] Existing fork configs have a temporary migration path.
- [ ] Stage 1 landed (or open) before the arena PR.
- [ ] Extraction diff dominated by new arena package files.
