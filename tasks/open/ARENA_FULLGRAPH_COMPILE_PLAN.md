# Arena Offload Fullgraph Block Compile - Plan

> Durable design and acceptance plan. Mutable status, experiment results, and
> next-agent handoff notes live on git-bug ticket `c4e29f1`.
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

## Verified starting point

The arena already has the desired architecture:

- `_InstalledDispatcher.__call__` is compiler-disabled and owns eager block
  dispatch;
- `dispatch()` owns source selection, `fetch_start_multi_after`, `fetch_wait`,
  `free_on_backward`, `fetch_free_after`, checkpoint early-stop handling, and
  sampling allocation recovery;
- `_get_dispatch_kernel()` reconstructs functional state and calls the saved
  block through `torch.func.functional_call`;
- the pure kernel is currently compiled with a hard-coded
  `fullgraph=False`;
- `ArenaOffloadConfig.compile_blocks` is already derived from
  `ModelConfig.compile`, and the trainer already recognizes that arena owns
  block compilation and skips its own compiler;
- focused CUDA coverage already exercises streamed compiled
  train -> sample -> train, checkpoint ticket lifetime, FP8 native math, and
  LoRA gradients under `fullgraph=False`.

The missing pieces are strictness propagation, actual graph-break diagnosis,
and strict-mode acceptance. No new transfer or residency architecture is
needed.

## Design

### 1. Propagate the existing fullgraph setting

Add a derived internal field such as `_compile_fullgraph` to
`ArenaOffloadConfig`. It is populated from `ModelConfig.compile_fullgraph` only
when arena block compilation is enabled.

Propagate it through:

```text
ArenaOffloadConfig.from_model_config
  -> ArenaOffloadRuntime.prepare executor kwargs
  -> prepare_block_dispatcher_runtime
  -> GenericBlockDispatcherRuntime
  -> torch.compile(..., fullgraph=compile_fullgraph)
```

Expose the effective value in arena diagnostics. Keep `False` as the existing
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

### 4. Correct trainer routing and messages

The trainer currently computes `runtime_owns_block_compile`, but its global
quantized-model branch still prints that `fullgraph=True` is incompatible and
changes its local `compile_fullgraph` value before the arena-owned skip.

Make that downgrade conditional on the trainer actually owning compilation.
For arena-owned compilation:

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

There is one risky supported config combination to reject early rather than
allowing an hours-late failure. After dataset buckets and sampling resolutions
are available, reuse `_observed_input_shapes()` to validate
`compile_fullgraph=True, compile_dynamic=False`:

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

Cross-process restoration remains tracked by `COMPILE_MEGA_CACHE_PLAN.md` and
ticket `ab208bf`. This ticket supplies the strict arena kernels that plan must
exercise.

Do not invent a separate cache format or MM cache key here. Verify whether
PyTorch's artifact guards already distinguish the effective fullgraph policy.
Add fullgraph to a Toolkit-owned cache identity only if a controlled
cold/shared/Mega-Cache matrix demonstrates a collision or incorrect reuse.

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

### Slice 1 - Config and dispatcher integration

Add `_compile_fullgraph`, propagate it to the dispatcher, include it in
diagnostics, and correct the trainer's quantized downgrade/ownership branch.
Add the static-shape readiness check at the setup seam where
`_observed_input_shapes()` and the effective compiler limit are both known.

Focused CPU tests prove:

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

Focused CUDA tests prove strict BF16 resident, streamed, and mixed-residency
kernels without internal graph breaks.

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

First run one real Z-Image arena block under strict mode with autocast outside
the kernel. Then execute the four-arm cache matrix from
`COMPILE_MEGA_CACHE_PLAN.md`: cold, empty no-load control, shared disk, and
Mega-Cache into an empty cache directory.

Require:

- expected AOTAutograd and FXGraph hits;
- zero cache bypass/miss in the restored consumer;
- zero Inductor/Triton compilation and autotune benchmarking;
- exact expected arena fetch/release counts from the arena transfer lifetime
  stats/diagnostics, not legacy block-stream counters;
- stable output/loss/gradient checksums;
- zero internal graph breaks;
- no trainer compile wrapping around the arena dispatcher.

Expand to the full known Z-Image graph/shape set only after the one-block gate
passes. A full UI training job remains user-launched.

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

## Rough size

- Slice 0: a few focused hours if the existing pure kernel is already strict;
- Slice 1: roughly half to one day for config, routing, diagnostics, and tests;
- Slice 2: roughly half to two days depending on actual FP8/LoRA strict
  failures;
- Slice 3: roughly half to one day after the controlled cache harness is
  adapted to the arena seam.

The expected total is about one to three focused engineering days, with Slice
0 providing the hard answer before broader implementation.
