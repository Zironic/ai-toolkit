# Arena Offload Fullgraph Block Compile - Plan

> Durable design and acceptance plan. Mutable status, experiment results, and
> next-agent handoff notes live on git-bug ticket `c4e29f1`.
>
> Config routing, strict dispatcher compilation, diagnostics, trainer ownership
> handling, static-recompile readiness, strict real-model CUDA, and MegaCache
> acceptance are implemented. The stable cache contract and residency evidence
> are in `../../docs/decisions/MEGACACHE.md`.
>
> This targets the generic arena dispatcher in
> `toolkit/memory_management/arena_offload/`. Legacy
> `LinearLayerMemoryManager` compilation is out of scope.

## Outcome

When arena offload owns block compilation and the job requests
`compile_fullgraph=True`, compile each arena block's pure functional math kernel
with `fullgraph=True` and zero internal Dynamo graph breaks.

The transformer does not need to become one graph. The existing eager arena
dispatcher remains the intentional boundary around each compiled kernel:

```text
eager arena dispatch
  -> fetch/wait and assemble functional CUDA state
  -> fullgraph=True pure block math kernel
  -> eager checkpoint/release/recovery handling
```

Fetch/free operations, residency policy, checkpoint replay, sampling recovery,
and phase transitions remain outside the compiled kernel. They already work in
the production arena runtime and do not need to move into the graph.

## Explicit scope

Required:

- BF16 and the production arena FP8 paths;
- frozen canonical base weights plus trainable LoRA/adapters;
- resident, streamed, and mixed-residency arena plans;
- model-owned non-reentrant gradient checkpointing;
- training -> sampling -> training transitions;
- bounded dynamic sequence shapes;
- cross-process Mega-Cache acceptance on the strict kernels.

Not required:

- compiling the entire model as one graph;
- putting arena transfer operations in the graph;
- making legacy `LinearLayerMemoryManager` or `stage_block_forward` compile;
- changing the legacy/upstream per-Linear offload path;
- adding a second arena compile mode or separate user-facing arena flag.

## Shipped foundation and remaining acceptance

The arena has the desired compile boundary and configuration routing:

- `_InstalledDispatcher.__call__` is compiler-disabled and owns eager block
  dispatch;
- `dispatch()` owns source selection, `fetch_start_multi_after`, `fetch_wait`,
  `free_on_backward`, `fetch_free_after`, checkpoint early-stop handling, and
  sampling allocation recovery;
- `_get_dispatch_kernel()` reconstructs functional state and calls the saved
  block through `torch.func.functional_call`;
- `ArenaOffloadConfig` derives `_compile_fullgraph` from the existing model
  compile settings and propagates it to the dispatcher;
- the pure kernel passes the effective value to
  `torch.compile(..., fullgraph=compile_fullgraph)`;
- diagnostics report the effective strictness, and the trainer recognizes arena
  compile ownership without downgrading or double-compiling strict blocks;
- strict static jobs run an early recompile-budget readiness check once the
  observed train/sample shapes are available;
- existing non-strict CUDA coverage exercises streamed compiled
  train -> sample -> train, checkpoint ticket lifetime, FP8 native math, and
  LoRA gradients.

The remaining work is strict CUDA, FP8/checkpoint/shape, real-model, and
cross-process Mega-Cache acceptance, plus fixes for any graph breaks those
focused seams actually reproduce. No new transfer or residency architecture is
needed.

## Design

### 1. Preserve the existing fullgraph propagation

`ArenaOffloadConfig._compile_fullgraph` is derived from
`ModelConfig.compile_fullgraph` only when arena block compilation is enabled.

The implemented path is:

```text
ArenaOffloadConfig.from_model_config
  -> ArenaOffloadRuntime.prepare executor kwargs
  -> prepare_block_dispatcher_runtime
  -> GenericBlockDispatcherRuntime
  -> torch.compile(..., fullgraph=compile_fullgraph)
```

Arena diagnostics expose the effective value. `False` remains the
behavior-preserving default.

The arena already has one shared training/sampling dispatcher policy. Do not
add a second arena-specific public config switch: the existing
`compile_fullgraph` setting is the source of truth.

### 2. Keep the compile boundary where it is

Only `_get_dispatch_kernel()` becomes strict. `_InstalledDispatcher` and
`dispatch()` remain eager and compiler-disabled.

`fullgraph=True` itself is the primary graph-break assertion: any unsupported
operation inside the pure kernel must raise rather than split into a partial
graph. Do not run `torch._dynamo.explain()` over the whole transformer and
misclassify the intentional dispatcher boundaries as failures.

Audit each constructed block kernel instead:

- it compiles successfully with `fullgraph=True`;
- it produces one graph for the tested ABI/shape variant;
- the strict call succeeds, which itself proves no graph break occurred;
- graph-break counters remain a comparison-arm diagnostic for
  `fullgraph=False`, not an additional strict-mode assertion;
- expected repeated blocks share/reuse compiled artifacts as they do today;
- arena dispatch and transfer counts remain outside the graph.

### 3. Fix only observed strict-capture failures

Start with the current production-shaped synthetic CUDA dispatcher test and
change only the kernel strictness. Let the first failure identify the real
unsupported target.

Likely seams to verify, not pre-emptively redesign:

- saved model-installed block forwards and LoRA adapter math;
- TorchAO/Quanto FP8 wrapper reconstruction under `functional_call`;
- structured args/kwargs and structured block outputs;
- ambient autocast, which must remain outside the compiled callable rather
  than introducing private autocast enter/exit nodes;
- bounded dynamic sequence hints installed before the first kernel call;
- checkpoint recompute using a fresh eager transfer ticket while compiled
  backward consumes the substituted state.

Add a guard, adapter, or compile-clean helper only for a failure reproduced by
the strict CUDA seam. Do not add legacy-MM trampolines, duplicate source
layouts, or new program abstractions.

### 4. Preserve trainer routing and messages

The trainer computes `runtime_owns_block_compile` and limits its global
quantized-model fullgraph downgrade to cases where the trainer actually owns
compilation. For arena-owned compilation, retain this contract:

- preserve the value already propagated into `ArenaOffloadConfig`;
- print that arena owns strict block compilation when enabled;
- skip trainer block/whole-model compilation exactly as today;
- never silently switch the arena kernel to `fullgraph=False` after the job
  requested strict mode.

Unsupported strict arena blocks must fail with the original Dynamo error plus
block identity, not silently fall back to non-strict compilation.

### 5. Guard the explicit static-shape configuration

Dynamo's per-code-object recompile limit applies in every dynamic mode, and
`fullgraph=True` turns a limit hit into a hard failure. The historical
step-101 crash does not justify raising the limit for normal arena training:
it combined `dynamic=False` bucket specialization with fresh closures across
sampling boundaries, while arena defaults to dynamic kernels and keeps stable
dispatcher kernels across phase transitions.

Do not call `raise_dynamo_recompile_limit()` unconditionally. Preserve the
default limit so it can still expose accidental guard churn.

One risky supported config combination is guarded early rather than allowed to
fail hours into a job. After dataset buckets and sampling resolutions are
available, the readiness validator reuses `_observed_input_shapes()` for
`compile_fullgraph=True, compile_dynamic=False` to:

- compute a conservative floor for the distinct train/sample shape variants
  that share the dispatcher kernel code object;
- compare it with the effective configured Dynamo recompile/cache limit;
- if the floor reaches the limit, fail before the first compiled execution as
  `arena_fullgraph_static_recompile_limit`, reporting the observed variant
  count, limit, and the two remedies: use `compile_dynamic=True` or explicitly
  raise `cache_size_limit`;
- if shapes cannot be enumerated, emit a named readiness warning instead of
  guessing or silently raising the global limit.

An explicit adequate `cache_size_limit` permits static strict mode. This guard
does not change dynamic arena training and does not create a new config field.

### 6. Coordinate with Mega-Cache without duplicating it

Cross-process restoration is owned by the generic session described in
`COMPILE_MEGA_CACHE_PLAN.md` and `../../docs/decisions/MEGACACHE.md`. Do not
invent a separate Arena cache format or residency key here.

The controlled full-model and residency-transition matrices established that
Torch guards distinguish the compiled variants and that Mixed and Full Arena
plans reuse the same entries. Toolkit's coarse identity includes the effective
fullgraph/dynamic compiler policy and stable dispatcher generation, but not
resolution, adapter topology, residency, simulated-card size, or transfer
plan.

## Implementation sequence

### Slice 0 - Strict synthetic go/no-go

Temporarily run the existing production-shaped CUDA arena dispatcher seam with
the kernel changed to `fullgraph=True` and no other architectural changes.

Prove, in order:

1. a frozen BF16 block succeeds in sampling;
2. streamed and resident functional sources both succeed;
3. non-reentrant checkpointed forward/backward succeeds;
4. train -> sample -> train retains correct transfer counts and gradients;
5. repeated calls do not add frames or graph breaks.

If a strict-capture failure occurs, record its exact target and smallest
reproducer on the ticket before modifying production code.

### Slice 1 - Shipped config and dispatcher integration

The shipped foundation propagates `_compile_fullgraph` to the dispatcher,
includes it in diagnostics, preserves arena ownership in the trainer's
quantized branch, and runs the static-shape readiness check where
`_observed_input_shapes()` and the effective compiler limit are both known.

Keep focused CPU coverage for these contracts:

- `compile=False` leaves arena compilation and strictness off;
- `compile=True, compile_fullgraph=False` preserves current behavior;
- `compile=True, compile_fullgraph=True` reaches the dispatcher as strict;
- arena ownership prevents trainer double compilation;
- strict dynamic training leaves Dynamo's configured recompile limit alone;
- strict static training whose observed variant floor reaches the limit fails
  early as `arena_fullgraph_static_recompile_limit`;
- an explicitly adequate `cache_size_limit` permits strict static training;
- an unknown static shape set produces the named readiness warning;
- legacy compile routing is unchanged.

The remaining focused CUDA acceptance must prove strict BF16 resident,
streamed, and mixed-residency kernels without internal graph breaks.

### Slice 2 - FP8, LoRA, checkpoint, and shapes

Run strict mode through the existing production-shaped arena coverage:

- TorchAO/Quanto Float8 forward and grad-input paths;
- LoRA/adapters with all expected finite gradients;
- non-reentrant checkpoint early-stop and no-grad-input replay;
- depth-1/depth-2 transfer-ticket reuse;
- bounded dynamic sequence shapes and same-shape repeated blocks;
- the per-code-object cache-entry/unique-variant count across the complete
  train/sample/shape matrix stays comfortably below the effective recompile
  limit;
- sampling allocation recovery outside the strict kernel.

Use focused test/diagnostic cache introspection for that evidence; do not add a
production dependency on private Dynamo cache-entry APIs. `new_frames` remains
the production signal, while the focused matrix records the concrete entry
count and its headroom to the limit.

Fix only failures that occur inside the pure functional kernel. Eager
dispatcher failures are ordinary arena-runtime issues and are not graph breaks.

### Slice 3 - Real Z-Image and Mega-Cache acceptance

The real Z-Image strict block and full-model four-arm matrix use autocast
outside the kernel: cold, empty no-load control, shared disk, and MegaCache
into an empty cache directory. The separate five-process residency benchmark
uses `Mixed -> Mixed -> Full -> Full -> Mixed` and measures misses instead of
failing on a transition.

Require:

- expected AOTAutograd and FXGraph hits;
- zero cache bypass/miss in the restored consumer;
- zero Inductor/Triton compilation and autotune benchmarking;
- exact expected arena fetch/release counts from the arena transfer lifetime
  stats/diagnostics, not legacy block-stream counters;
- stable output/loss/gradient checksums;
- zero internal graph breaks;
- no trainer compile wrapping around the arena dispatcher.

The accepted full-model lifecycle saves three AOT variants and five FX graphs.
Warm and cross-residency processes hit all of them with zero Inductor codegen,
Triton compilation, coordinate descent, or graph breaks. One additional AOT
miss is the measured non-serialized inference lookup and does not perform
backend compilation. Exact timings and artifact paths are recorded in
`../../docs/decisions/MEGACACHE.md`. A full UI training job remains
user-launched.

## Acceptance criteria

The feature is complete when:

- `compile=True, compile_fullgraph=True` makes every supported arena block
  kernel strict;
- every expected BF16 and FP8 block kernel captures with zero internal graph
  breaks;
- eager arena transfer, checkpoint lifetime, residency transitions, and
  recovery remain outside the graph and preserve their current behavior;
- resident, streamed, and mixed plans work;
- LoRA/adapter gradients and frozen-base behavior match the compiled
  non-strict reference within documented tolerance;
- train -> sample -> train and checkpoint replay remain depth-bounded and
  numerically stable;
- bounded dynamic shapes do not cause unexplained recompiles;
- the full strict variant matrix records cache-entry headroom well below the
  effective limit;
- strict static jobs that would exhaust the limit fail during readiness rather
  than during training;
- the trainer does not downgrade or double-compile arena-owned blocks;
- a fresh-process Mega-Cache consumer restores the strict arena artifacts
  without compiler or autotune activity;
- `compile_fullgraph=False` preserves the current arena path;
- no legacy `LinearLayerMemoryManager` fullgraph work is required.

## Stop conditions

Stop and reassess if strict capture fails on a model operation whose only
credible fix requires moving transfer/residency policy into the graph or
rewriting model architecture. Record the exact break and keep
`compile_fullgraph=False` as the supported fallback.

Do not respond to a strict failure by reviving the legacy-MM trampoline plan or
by compiling the entire transformer as one graph.

## Actual scope

The strict boundary required targeted dispatcher/FP8 identity fixes rather
than a new execution architecture. The acceptance harness then expanded from a
single strict block to the full-model four-arm cache matrix and the five-run
residency transition benchmark. Future work here is regression maintenance;
new MegaCache lifecycle or upstream-extraction work belongs in
`COMPILE_MEGA_CACHE_PLAN.md` and `docs/decisions/UPSTREAM_PR_PLAN.md`.
