# Generic Block Dispatcher Arena Refactor Plan

> **Completed 2026-07-16:** The generic saved-forward dispatcher is the
> production arena path; the implementation and GPU acceptance gates passed.

> **git-bug:** `b7dead1` for the overall effort and `b1a13d2` for the Phase 0
> execution-seam decision.
>
> **Status ownership:** this document records design, sequencing, and
> acceptance criteria. Progress, validation results, blockers, priorities,
> and handoff notes belong on the associated git-bug tickets.
>
> **Supersedes:** the architecture-adapter execution design in:
>
> * `tasks/done/COMPILE_NEUTRAL_KREA2_RUNTIME_REFACTOR_PLAN.md`
> * `tasks/done/revised_combined_compile_neutral_krea2_refactor_plan.md`
> * ticket `6dd9ba6`
>
> Existing canonical-arena, residency, transactional commit, fail-closed
> lifecycle, and host-memory rules remain valid unless explicitly changed
> here.
>
> This plan extends the physical storage declarations in
> `toolkit/quantization/storage.py` (the `LayerStorageBinding` work from
> `tasks/done/OSTRIS_ARENA_QUANTIZATION_PLAN.md`, ticket `c9ee48d`). It does
> not create a parallel storage abstraction. Phase 7 updates
> `docs/decisions/UPSTREAM_PR_PLAN.md` and ticket `553ffec`.

---

# Goal

Allow supported transformer models to use arena offload while preserving
their ordinary transformer and block forward implementations.

The intended execution path is:

```text
existing transformer forward
    -> existing block call
        -> generic arena dispatcher
            -> obtain resident or streamed block storage
            -> execute the saved installed block forward
            -> return the original output unchanged
```

The arena must not require:

* architecture-specific execution adapters;
* handwritten block leaf paths;
* rewritten transformer forwards;
* reconstructed model dataflow;
* model-specific block argument schemas;
* quantization semantics inside movement or residency code.

Krea2 is the first validated production target. The old adapter execution
path is deleted in this scope. A second production transformer is deliberately
deferred to a follow-on effort and is not an acceptance gate for the Krea2
dispatcher release.

---

# Completion state

The refactor is complete when:

1. Krea2 executes its ordinary transformer forward.
2. Selected Krea2 blocks execute their saved installed forwards.
3. Immutable block state can be resident or streamed through the arena.
4. Quantization integration owns storage meaning and wrapper reconstruction.
5. The arena sees ordered opaque storage tensors and finalized substitution
   operations.
6. Every selected block parameter and persistent buffer is accounted for
   before destructive canonical commit.
7. Trainable adapter state remains attached and live.
8. The model is the sole checkpoint owner.
9. Arena training fails before canonical commit when model gradient
   checkpointing is disabled.
10. Within a checkpointed model, intentionally uncheckpointed selected
    blocks remain resident.
11. Block compilation works without tracing arena policy or transfer
    scheduling.
12. Whole-model compile is not prohibited unless an observed correctness
    failure requires a narrow restriction.
13. One production-shaped Krea training and sampling smoke passes.
14. One deterministic Krea numerical oracle matches ordinary execution.
15. Required alternate quantization and adapter mechanisms pass focused
    checks.
16. The old architecture-adapter execution path is deleted.
17. Upstream planning documents describe the demonstrated implementation
    rather than the retired adapter design.

The implementation does not claim support for every arbitrary transformer.

---

# Ownership boundaries

```text
model:
    transformer and block execution semantics
    checkpointing
    shape bucketing
    model-specific sampling behavior
    optional declarative block-container selection when inference is ambiguous

quantization integration:
    physical storage declarations
    executable replacement declarations
    wrapper reconstruction
    numerical kernels

arena:
    canonical storage
    residency
    transfer
    lifetime
    block availability
    block-kernel compilation

dispatcher:
    joins explicit arena storage to the existing block-call boundary
```

A small model-owned selector or declaration is acceptable when block
structure cannot be inferred unambiguously.

Model-specific arena execution logic is not acceptable.

---

# Verified constraints

## Existing Krea forward ownership

Krea2's ordinary transformer forward
(`extensions_built_in/diffusion_models/krea2/src/mmdit.py`,
`_forward_impl` / `_blocks_trunk`) already owns:

* input construction;
* attention masks;
* positional embeddings;
* reference-image behavior;
* K/V capture and cache behavior;
* gradient checkpointing;
* the final projection.

The dispatcher preserves this behavior instead of reconstructing it.

## Existing model-side coupling

Three existing behaviors require explicit ownership changes.

### Token bucketing

Krea2 currently pads sequence state to 256-token buckets when the arena
runtime is attached (`mmdit.py`, the `use_runtime` gate in
`_forward_impl`). The padding bounds Dynamo specializations of the compiled
block kernels; removing it makes every distinct sequence length a fresh
recompile per block (`new_frames` in the perf log).

The padding remains model-owned but is keyed from a semantics-neutral
compile setting rather than direct arena-runtime inspection.

Retiring the bucketing is outside this refactor.

### Resident modulation state

Krea2 currently passes modulation modules (`SimpleModulation` /
`DoubleSharedModulation`) through `ignore_modules`.

The new state accounting replaces this hint. Frozen state that is not
declared as managed remains attached and resident.

### Checkpoint ownership

The old arena runtime checkpoints its own block trunks
(`immutable_runtime.py`, `build_train_trunk`), while Krea2 checkpointing is
disabled under that runtime (`krea2.py`,
`_attach_immutable_training_memory` call site).

The new implementation must:

* remove runtime-owned checkpointing; and
* enable Krea2's existing checkpointing path;

as one implementation change.

Neither half may land alone. Landing either half alone produces double- or
zero-checkpointing. The model's existing `_checkpoint_keep_last` becomes the
only keep-last lever; no keep-last logic may be (re)added to the arena.

Arena training with model gradient checkpointing disabled is unsupported
and must fail before destructive canonical commit.

## Existing storage work

`LayerStorageBinding` already provides ordered physical storage
declarations.

The dispatcher work extends that seam with executable replacement
information. It does not replace the existing storage design.

## Compile ownership (four existing owners)

Compile decisions live in four places today; the dispatcher slots in
without adding a fifth owner:

1. **The trainer owns job-level compile policy and all global compiler
   state** (`BaseSDTrainProcess.py`): whole-model vs generic block compile,
   `cache_size_limit`, `suppress_errors` (set globally for quantized runs),
   inductor toggles. The exclusivity seam is `memory_runtime_owns_compile()`
   — the new runtime keeps advertising `owns_compile`, so the trainer's
   generic block compile stands down.
2. **The trainer derives dynamic-shape bounds** (`compile_shape_bounds` ->
   `set_compile_dynamic_hints`, which refuses hints after the first kernel
   compiles). This seam is preserved unchanged; the dispatcher runtime
   consumes hints, it does not derive them.
3. **Krea2 sampling owns the mega-cache and the compile stance**
   (frame-counter save trigger; process-global
   `set_stance("eager_then_compile")` scoped to the sampling call). The
   mega-cache key must incorporate the dispatcher generation so adapter-era
   compile artifacts are never reused; the dispatcher's lazy compilation
   must not assume compile-on-first-call under an external stance. Known
   production bug under this stance: first sample of a run NaNs
   (ticket `7e6cc08`) — never treat today's first-call-under-stance output
   as a correctness baseline.
4. **Models own their internal compile hygiene.** Model-local `torch.compile`
   inside block forwards (e.g. chroma's modulation helpers) becomes a nested
   compile under the dispatcher — a Phase 1 screening criterion, not
   something the arena works around.

## Unproven execution seam

The proposed stateless execution seam has not been established for the
required combination of:

* tensor-subclass quantization;
* installed adapter forwards;
* CUDA compilation;
* custom quantizer operators;
* backward;
* non-reentrant checkpoint recomputation.

Phase 0 resolves that uncertainty before production runtime work begins.

---

# Minimal supported structure

The generic core supports the structures demonstrated by Krea2 and the
selected second architecture. It does not infer arbitrary model structures
speculatively.

## Block container

The initial automatic rule is:

* identify one unambiguous repeated block container;
* blocks are ordinary child modules;
* at least two blocks contain supported managed immutable storage;
* selected block objects are unique;
* selected blocks do not overlap;
* the container has a deterministic module path.

If Krea2 or the selected second architecture requires multiple containers,
add only that demonstrated case.

When automatic discovery is ambiguous, a model may provide a declarative
block-container selector.

That selector may identify blocks but may not define:

* execution;
* storage leaves;
* argument reconstruction;
* checkpoint policy;
* transfer policy;
* residency policy.

An ambiguous model without a valid selector is rejected before canonical
commit.

## Managed storage

Initial supported declarations are:

* dense Linear weight and optional bias;
* TorchAO row-wise FP8;
* Quanto row-wise FP8;
* required `OstrisLinear` formats, including ConvRot where applicable.

Declarations are captured during construction and are not rediscovered
during execution.

---

# State contracts

## Physical storage

Physical storage remains an ordered sequence of opaque tensors plus an
execution identity.

Conceptually:

```python
@dataclass(frozen=True)
class TensorStorageBinding:
    name: str
    tensor: torch.Tensor


@dataclass(frozen=True)
class LayerStorageBinding:
    tensors: tuple[TensorStorageBinding, ...]
    execution_key: tuple
```

The arena may:

* preserve ordering;
* pack and align tensors;
* transfer tensors;
* make storage resident;
* manage lifetime;
* hash execution compatibility.

The arena must not:

* interpret tensor names;
* branch on quantizer type during movement;
* materialize compatibility weights;
* call `OstrisLinear.weight`;
* select numerical kernels;
* infer quantization meaning from tensor count.

## Executable substitution

Physical storage and module-state replacement are distinct.

The finalized execution description must identify:

* the module state entry being replaced;
* the physical leaves used to create it;
* the quantization-owned reconstruction operation, if required.

The exact Python representation is chosen from the Phase 0 spike.

Required properties:

* reconstruction is selected before Dynamo tracing;
* the arena does not interpret reconstruction identities;
* the substitution description is immutable after finalization;
* cache invalidation distinguishes the dispatcher generation from the old
  adapter generation.

Use a simple explicit dispatcher-generation version initially. Do not
design precise cross-version cache compatibility during this refactor.

## Complete but simple state accounting

Before canonical commit, enumerate every named parameter and persistent
buffer under each selected block.

Classify state as:

```text
declared storage state:
    managed immutable

requires_grad state:
    live trainable

all remaining ordinary state:
    remains attached and resident
```

Reject:

* missing declared targets;
* conflicting replacement declarations;
* managed storage shared across selected blocks;
* tied managed state;
* active parametrization affecting managed state;
* required wrappers without a proven substitution operation;
* entries that cannot be enumerated consistently.

Sharing restrictions apply to managed state only. Permanent-resident state
shared across blocks (e.g. Krea's `DoubleSharedModulation`) is allowed; its
bytes are counted once in residency accounting.

This is an accounting pass, not a general-purpose state-analysis framework.

---

# Explicitly unsupported configurations

The following are rejected rather than treated as implicit future
requirements.

## Arena training without model checkpointing

Arena offload during training requires model gradient checkpointing.

If arena training is enabled while model checkpointing is disabled,
attachment fails before canonical commit with a clear configuration error.

Within a checkpointed model, intentionally uncheckpointed selected blocks,
such as a keep-last tail, remain resident and are never streamed. (An
uncheckpointed training block's autograd graph holds references to its
substituted storage until backward; keeping such blocks resident removes
the storage-lifetime problem instead of managing it.)

## Unsupported managed state

Reject before canonical commit:

* managed storage shared between selected blocks;
* tied managed parameters;
* active parametrizations affecting managed state;
* missing or conflicting replacement targets;
* wrappers without a proven substitution operation;
* ambiguous block selection without a valid declarative selector.

## Out-of-context block dispatch

Selected blocks are supported only while invoked through an active
transformer execution.

Calling a selected block outside that context raises a clear error unless
repository inspection identifies an existing production caller and the plan
is revised to support it. (Known candidate to inspect: Krea's
reference-image / reference-K/V calls, which today bypass the runtime via
`can_run_current_call`.)

## Runtime topology

The initial implementation does not support:

* multiple active arena runtimes in one process;
* multiple CUDA devices within one arena runtime;
* concurrent execution of separate models through one runtime.

Add an explicit rejection only where an existing configuration or callable
entry point can request one of these states.

Do not build generalized topology detection for unreachable configurations.

---

# Lifecycle

## Preparation before network installation

```text
unwrap transformer
    -> discover or declaratively select repeated blocks
    -> verify model checkpointing is enabled for arena training
    -> collect physical storage declarations
    -> collect replacement declarations
    -> run complete state accounting
    -> reject unsupported managed state
    -> build canonical arena transactionally
    -> commit canonical storage
```

No dispatcher is installed during preparation.

All configuration, declaration, and structural errors knowable during
preparation must be rejected before destructive commit.

## Finalization after network installation

```text
inspect the installed module structure
    -> verify managed replacement targets still exist
    -> bind reconstruction operations
    -> preserve the final installed block forwards
    -> install generic block dispatchers
    -> create resident/streaming execution handles
    -> activate runtime
```

Finalization occurs after training-network installation so the saved block
forward includes LoRA or other installed adapter behavior.

Errors that depend on the installed network or final callable structure are
detected during finalization rather than simulated during preparation.

## Execution

The model runs its normal transformer forward.

Each selected block enters the dispatcher, obtains its current storage, and
executes its saved installed forward.

## Teardown

Teardown must:

* prevent new dispatch;
* stop or abandon runtime-owned transfers safely;
* restore original block forwards;
* remove any top-level execution wrapper;
* drop runtime-owned references to compiled callables and execution
  handles;
* release resident sidecars;
* release canonical pinned storage;
* clear published runtime state;
* preserve the existing fail-closed disposed-transformer contract after
  destructive commit.

The runtime does not promise eviction of process-global Dynamo or Inductor
caches.

---

# Execution rules

## Dispatcher

The dispatcher:

* accepts the block's original `*args` and `**kwargs`;
* identifies the block;
* verifies an active transformer execution;
* obtains resident or streamed storage;
* invokes the saved installed forward through the proven stateless seam;
* returns the original output unchanged.

It does not:

* reconstruct model dataflow;
* define argument schemas;
* checkpoint blocks;
* derive compile shapes;
* perform quantizer-specific movement;
* support arbitrary out-of-context block execution;
* allow outside compilation to trace transfer or policy logic.

## Checkpointing and residency

The model is the sole checkpoint owner.

Checkpoint recomputation calls the same dispatcher again through the
model's existing checkpoint path.

Residency rule:

```text
checkpointed selected blocks:
    may be resident or streamed

intentionally uncheckpointed selected blocks:
    must remain resident

model checkpointing globally disabled:
    reject arena training before canonical commit
```

No general autograd-completion storage-retention subsystem is implemented.

## Compilation

Compile the stateless block execution callable, not the complete
transformer forward.

The compiled callable accepts:

* original positional arguments;
* original keyword arguments;
* explicit storage tensors.

The runtime advertises `owns_compile`, so the trainer's generic block
compile stands down.

The arena must not mutate global Dynamo, Inductor, or compiler-stance
configuration.

Note the ordering flip this plan introduces: today the runtime compiles
*checkpointed* trunks (compile outside, checkpoint inside via
`compiled_checkpoint_context`); the dispatcher design puts the model's
non-reentrant checkpoint *outside* and the compiled callable *inside*.
Phase 0 proves this ordering, including that recomputation reuses the
forward's compiled kernel rather than tracing a new frame per backward.

### Whole-model compile

Do not prohibit whole-model compilation. (It was historically avoided for
speed — graph-break storms — not correctness; forcing it is a user choice.)

Remove the existing block or refusal that prevents whole-model compile from
being used with the arena.

Arena policy, transfer scheduling, and dispatcher-control code remain
outside compiled graphs through appropriate compiler-disabled boundaries.

Do not add special whole-model compatibility machinery.

Exercise the combination once in Phase 3.

The question is only whether the combination remains correct:

* if Dynamo graph-breaks around dispatched blocks and execution remains
  correct, leave it allowed;
* do not require a fully captured outer graph;
* do not benchmark or optimize the combination during this refactor;
* add a restriction only if the run exposes a concrete correctness failure.

---

# Implementation phases

## Phase 0 — Fatal execution-seam spike

Ticket: `b1a13d2`

Use a real Krea block class with test weights on CUDA.

The proof must exercise the proposed saved-installed-forward and
explicit-state mechanism, not a simplified standalone function.

Run with:

```python
torch._dynamo.config.suppress_errors = False
```

(quantized production runs set it True, which would silently mask a
`functional_call` trace failure as an eager fallback and corrupt the
go/no-go signal).

### Primary case

Use:

```text
real Krea block
TorchAO FP8
installed LoRA
saved installed forward
CUDA block compilation
forward
backward
non-reentrant checkpoint recomputation
two or three repeated iterations
```

Compare against ordinary resident execution for:

* outputs;
* input gradients;
* adapter gradients;
* finite values;
* evidence that explicit replacement state was consumed;
* compiled graph reuse during repeated execution and recomputation.

This case also proves:

* original argument passthrough;
* actual Krea output structure;
* installed monkeypatched forward behavior;
* checkpoint-outside / compiled-callable-inside ordering.

Compiled arms must be compared against a compiled resident baseline —
bf16 Inductor fusion changes rounding, so an eager comparison measures
fusion noise, not the seam.

### Second distinct mechanism

Add ConvRot forward and backward through the same seam.

This proves custom operator and custom backward compatibility.

### Conditional third case

Add Quanto only when inspection establishes that it uses a materially
different substitution or reconstruction seam from TorchAO.

Do not add separate Phase 0 cases for:

* dense storage;
* LoKr;
* DoRA;
* FullModule;
* tensor versus tuple outputs;
* arbitrary argument structures;
* exhaustive repeated-call combinations;
* sampling compiler stance;
* mega-cache behavior.

### Gate

Proceed when:

1. the TorchAO + LoRA + compile + checkpoint case passes; and
2. the ConvRot forward and backward case passes.

Add Quanto to the gate only if inspection establishes that it uses a
materially different substitution seam.

Everything else remains Phase 4 work.

If either required seam fails, stop the dispatcher refactor and record the
incompatibility.

---

## Phase 1 — Brief second-architecture screen

Before implementing generic discovery, inspect the leading second consumer.

Current leading candidate: Z-Image. Ideogram is the fallback candidate.

The screen establishes only:

* the likely repeated block container;
* whether blocks are ordinary modules;
* whether managed layers expose or can expose the planned storage
  declarations;
* whether automatic selection appears unambiguous;
* whether a small declarative selector would be sufficient;
* whether there is an obvious fatal structural incompatibility;
* whether block internals contain model-local compilation that must stand
  down when the arena owns block compilation.

A candidate is disqualified only when resolving its structure or
compilation ownership would require architecture-specific execution logic.

Do not perform a complete integration audit in this phase.

### Acceptance

* one second architecture is selected;
* its likely block structure is recorded;
* the initial discovery rule is not accidentally Krea-only;
* no production code is changed.

---

## Phase 2 — Krea production vertical slice

Implement enough of the complete Krea path to reach the production
configuration once.

Use this internal debugging order:

```text
minimal physical and replacement declarations
    -> simple block discovery and state accounting
    -> resident eager invocation
    -> streamed eager invocation
    -> model-owned checkpointing
    -> intentionally uncheckpointed blocks forced resident
    -> block compilation
```

These are internal debugging checkpoints, not independent projects or broad
acceptance gates.

### Required work

1. Extend the existing storage declarations.
2. Add the minimal executable substitution representation proven by
   Phase 0.
3. Add simple discovery or declarative selection.
4. Run the complete state-accounting pass.
5. Reject arena training without model gradient checkpointing.
6. Reject unsupported managed state before canonical commit.
7. Build the canonical arena transactionally.
8. Finalize after training-network installation.
9. Preserve and dispatch to installed block forwards.
10. Connect resident and streamed acquisition.
11. Remove runtime-owned checkpointing.
12. Enable model checkpointing.
13. Keep intentionally uncheckpointed selected blocks resident.
14. Reject out-of-context selected-block calls.
15. Compile the stateless block callable.
16. Remove the existing whole-model-compile prohibition.
17. Keep arena control and transfer policy outside compiled graphs.
18. Restore original forwards on teardown.
19. Invalidate adapter-era compiled cache entries with an explicit
    dispatcher generation.

### Narrow implementation checks

During development, use only the smallest check needed to isolate the
current subsystem:

* one resident block forward;
* one streamed block forward;
* one checkpointed backward;
* one repeated compiled block call;
* one unsupported-configuration rejection;
* one teardown restoration check;
* one injected pre-commit failure when the storage transaction changes.

Do not create broad acceptance suites for each intermediate state.

### Phase acceptance

The production configuration must reach:

* one compiled training forward and backward;
* one sampling pass;
* one return to training.

Narrow checks establish only the subsystem currently being implemented.

Full numerical, accounting, recompilation, VRAM, lifecycle, and performance
acceptance belongs to Phase 3.

---

## Phase 3 — Production Krea smoke and measured fixes

Run one production-shaped Krea smoke:

```text
ordinary Krea transformer forward
TorchAO FP8
LoRA
mixed residency
model-owned checkpointing
compiled block execution
training forward and backward
sampling
train -> sample -> train
```

This is the broad Krea acceptance gate.

### Pre-built smoke tooling

The smoke scripts for this plan's Phase 3/4 gates already exist, built ahead
of the dispatcher work so acceptance can run as soon as each seam lands:

* `scripts/smoke_transformer_train_cuda.py` — shared full-model runner
  (train -> sample -> train phases, quantized-representation gates,
  discovery audit, teardown/pin-ledger checks, JSON row output) with
  architecture profiles in `scripts/smoke_profiles.py`
  (`krea2`/`zimage`/`ideogram4`). The `zimage`/`ideogram4` profiles fail
  fast until generic adapter resolution exists
  (`resolve_architecture_adapter`). Quanto `qfloat8` remains distinct under
  layer offloading and is validated through the generic dispatcher.
* `scripts/smoke_quantized_linear_cuda.py` — quantized-Linear contract smoke
  on a synthetic model (eager/checkpoint/compiled/streamed-state parity,
  no-`.weight`-materialization counters, representation assertions). Covers
  the Phase 4 Quanto/ConvRot/Orbit narrow oracles at the Linear seam and is
  runnable today (qfloat8/convrot4/orbit4 validated; `--declare-only` for
  the remaining ConvRot qtypes).
* `scripts/smoke_krea2_train_cuda.py` — untouched TorchAO regression
  baseline for the Phase 3 production-shaped Krea smoke.

## Production smoke evidence

The production smoke establishes:

* successful training forward and backward;
* finite outputs, loss, and gradients;
* compile-frame behavior;
* H2D byte accounting;
* canonical and resident byte accounting;
* peak VRAM;
* steady-state step time;
* checkpoint ownership;
* train/sample/train behavior;
* teardown and subsequent runtime cleanup where exercised by the smoke.

Use the old adapter path opportunistically as a lifecycle, accounting, or
performance oracle where it represents the same behavior.

Do not require exact deterministic output or gradient equality from the
broad production smoke.

## Deterministic numerical oracle

Run one fixed-input deterministic resident comparison between:

```text
ordinary installed Krea block or narrow transformer path
generic dispatcher path using the same installed forward
```

Compare:

* outputs;
* relevant input gradients;
* adapter gradients;
* finite values.

Use the narrowest callable that proves the state-substitution path without
bringing transfer scheduling, sampling, or full-job noise into the
comparison.

This numerical oracle runs once as part of Krea acceptance.

Alternate mechanisms require their own numerical comparison only when they
introduce a distinct substitution, reconstruction, or kernel path.

## Whole-model compile observation

Repeat the smallest useful portion of the production path with whole-model
compile enabled.

The question is only whether the combination remains correct.

Do not require:

* performance parity;
* benchmarking;
* a fully captured outer graph;
* absence of graph breaks;
* special nested-compile optimization.

If Dynamo graph-breaks around the arena and execution remains correct,
leave the combination allowed.

Add a restriction only if the run exposes a concrete correctness failure.

## Measured fixes

Add performance mechanisms only when the production smoke demonstrates they
are needed.

Likely required mechanisms:

* Krea token bucketing;
* existing trainer-derived dynamic-shape hints;
* depth-one forward prefetch;
* train/sample/train compiled-kernel reuse;
* cache-generation invalidation.

Do not automatically implement:

* recomputation prefetch;
* a separate recomputation trace;
* multiple trace variants;
* extensive sampling-stance compatibility;
* precise cross-version mega-cache fingerprints.

Add those only when a measured failure or material regression identifies
the need.

### Acceptance

* the production Krea path completes successfully;
* outputs, loss, and gradients remain finite;
* the deterministic numerical oracle matches within backend-appropriate
  tolerances;
* transfer and residency accounting are correct;
* exactly one checkpoint owner exists;
* recompilation remains practical across the actual smoke shape set;
* no material unaccepted steady-state regression remains;
* train -> sample -> train works without rebuilding unaffected execution
  state;
* teardown restores model-owned callables and drops runtime-owned execution
  references;
* whole-model compile is either observed to remain correct or rejected
  narrowly for a documented correctness failure.

---

## Phase 4 — Required alternate-path smokes

Validate execution mechanisms required by the intended first release that
were not covered by the Krea production smoke and numerical oracle.

### Quanto

Run a focused Quanto forward and backward only if:

* its substitution path differs from TorchAO; or
* Quanto support is part of the initial release claim.

Add a narrow numerical comparison when Quanto introduces a distinct
reconstruction or tensor-subclass execution path.

### ConvRot and Ostris state

Run ConvRot forward and backward through the production dispatcher.

Compare outputs and relevant gradients against ordinary resident ConvRot
execution because ConvRot introduces a distinct custom operator and
backward path.

Add one additional Ostris storage format only when its declaration,
reconstruction, or kernel mechanism differs materially.

### Alternate adapters

Test LoKr, DoRA, or FullModule only when their installation changes:

* module ownership;
* forward replacement;
* state substitution;
* target replacement behavior.

Do not test them merely because their mathematics differ from LoRA.

Add deterministic numerical comparison only when the adapter changes the
dispatcher seam rather than merely executing different adapter math inside
the already proven saved forward.

### Rules

Do not require every metric for every alternate smoke.

Do not construct a Cartesian product across quantizer, adapter, residency
mode, checkpointing, sampling, and lifecycle.

Add combinations only in response to an observed interaction failure.

### Acceptance

* each distinct required execution mechanism works;
* distinct substitution, reconstruction, or kernel paths match their narrow
  ordinary-execution oracle;
* failures are localized to the relevant mechanism;
* no unsupported compatibility claim is inferred from an untested
  mechanism.

---

## Phase 5 — Delete the old adapter execution path

The old adapter path may be used opportunistically as a Krea oracle during
Phases 2–4.

Do not run a separate comprehensive parity campaign.

### Before deletion

Confirm that no required Krea behavior still enters the old adapter
execution path.

Use instrumentation, call-site inspection, or a focused assertion rather
than another full validation matrix.

### Delete

Remove:

* `ArchitectureAdapter`;
* `SingleStreamMMDiTAdapter`;
* handwritten Krea block leaf paths;
* Krea `forward_streamed` arena entry points;
* Krea `run_blocks()` integration;
* runtime-owned repeated-block trunks;
* runtime-owned checkpoint loops;
* adapter-specific block argument collection;
* obsolete whole-model-compile prohibition code;
* tests that exist only for the deleted architecture.

Git history is the rollback mechanism.

### After deletion

Rerun:

1. the Phase 3 production Krea smoke; and
2. only those Phase 4 alternate smokes whose code path was affected by the
   deletion.

The deterministic Krea numerical oracle is rerun only if deletion changed
the shared dispatcher or substitution path.

Do not rerun unaffected alternate mechanisms as a generic confidence
ritual.

### Acceptance

* the Phase 3 production smoke still passes;
* affected alternate smokes still pass;
* any affected numerical oracle still matches;
* no production `run_blocks()` integration remains;
* no architecture-specific block execution adapter remains;
* one execution owner remains.

---

## Deferred follow-on — Second production architecture

This phase is intentionally deferred beyond the Krea2 dispatcher cleanup and
first release. It remains the next breadth proof, not unfinished work in the
current acceptance boundary.

Attach the architecture selected in Phase 1.

Generalize only around differences actually encountered.

A small model-owned declarative selector is acceptable when automatic
discovery is ambiguous.

The integration must not add:

* a rewritten transformer forward;
* arena-specific block execution;
* handwritten physical leaf paths;
* model-specific block argument reconstruction;
* model-owned transfer or residency policy.

## Representative smoke

Run one production-shaped smoke for the second architecture that covers,
where supported:

```text
ordinary transformer forward
supported quantized or dense storage
mixed residency
model-owned checkpointing
compiled forward and backward
sampling
train -> sample -> train
```

Mixed residency is sufficient to exercise resident and streamed block
acquisition.

Do not run separate fully resident, fully streamed, and mixed-residency
campaigns unless a failure indicates a residency-mode-specific problem.

## Focused checks

Add focused checks only for structural mechanisms not already exercised by
Krea, such as:

* a model-owned declarative container selector;
* multiple required block containers;
* a distinct output structure;
* a distinct checkpoint arrangement;
* a storage declaration not used by Krea;
* model-local compilation that must stand down under arena compile
  ownership.

Add a narrow deterministic numerical comparison only for a mechanism not
already proven by Krea or the alternate-path smokes.

### Acceptance

The second model uses the generic dispatcher core without
architecture-specific execution logic.

The demonstrated claim is:

> Krea2 and one second production transformer use the same block
> dispatcher, storage boundary, residency machinery, and compiled execution
> path.

Do not claim arbitrary-model support.

---

## Phase 7 — Upstream strategy and documentation

After Krea2 and the second architecture pass:

Update:

* `docs/decisions/UPSTREAM_PR_PLAN.md`;
* `tasks/done/UPSTREAM_ARENA_EXTRACTION_PLAN.md`;
* ticket `553ffec`;
* tickets `b7dead1` and `6dd9ba6`;
* maintainer-runnable validation instructions.

The first upstream feature PR may remain framed as Krea2 support, but the
implementation should be described as:

```text
generic block-dispatcher core
    + Krea2 as the first validated production consumer
```

The second architecture provides evidence for the design boundary but does
not require the first upstream PR to claim broad model support.

---

# Validation principles

Tests are evidence-gathering tools, not implementation milestones.

Before running a test, state:

* what concrete behavior it exercises;
* what failure would mean;
* how its result changes the next action.

Use:

* focused CUDA seam scripts;
* narrow repository tests;
* syntax checks;
* one production-shaped Krea smoke;
* one deterministic Krea numerical oracle;
* one focused whole-model-compile correctness observation;
* focused alternate-mechanism checks;
* targeted reruns after deletion.

Do not:

* add CPU compilation;
* add an MSVC requirement;
* run the full suite by default;
* create permanent tests for every temporary debugging state;
* construct a compatibility matrix without evidence of interaction;
* repeat the production smoke at multiple adjacent phase gates;
* make a noisy production lifecycle run responsible for precise
  deterministic numerical proof;
* benchmark whole-model compile during this refactor;
* build compatibility machinery before an actual run demonstrates the need;
* launch full training jobs unless the user requests them.

Full production jobs remain user-launched acceptance runs.

---

# Deferred optional work

These are optional extensions or optimizations, not configurations the
initial implementation must accept.

* recomputation prefetch;
* multiple persistent prefetch traces;
* precise cross-version compiled-cache compatibility;
* direct-to-arena cache loading redesign;
* broader automatic discovery beyond demonstrated architectures;
* generic non-transformer UNet support;
* new quantization algorithms;
* retiring model-side token bucketing.

Unsupported runtime configurations belong in the explicit rejection
section, not here.

---

# Repository enforcement

The final implementation should satisfy:

```text
no architecture-specific execution adapters;
no handwritten block leaf paths;
no rewritten complete transformer forward;
no model-specific block argument reconstruction;
no runtime-owned checkpoint policy;
no keep-last logic in the arena;
no arena training without model gradient checkpointing;
no streaming of intentionally uncheckpointed selected blocks;
no qtype-specific movement branches;
no quantization semantics in transfer or residency code;
no dynamic storage rediscovery during execution;
no destructive commit before complete state accounting;
no unsupported out-of-context selected-block dispatch;
no silent acceptance of tied, shared, or parametrized managed state;
no arena mutation of global compiler configuration;
no speculative whole-model-compile prohibition;
no whole-model-compile benchmarking project;
no outside compile tracing into arena policy code;
no production run_blocks() integration.
```

The shortest implementation path is:

```text
fatal seam spike
    -> brief second-model screen
    -> Krea production vertical slice
    -> one Krea production acceptance smoke
    -> one deterministic Krea numerical oracle
    -> observe whole-model compile correctness
    -> required alternate mechanisms
    -> delete old path and rerun affected evidence
    -> upstream update
```
