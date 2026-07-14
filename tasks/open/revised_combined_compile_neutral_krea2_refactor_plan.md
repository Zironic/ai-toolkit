# Revised Combined Refactor Plan

> **Partially superseded (2026-07-14):** the architecture-adapter *execution*
> design in this document is superseded by
> `GENERIC_BLOCK_DISPATCHER_PLAN.md` (ticket `b7dead1`). The adapter path
> stays only as a comparison oracle until that plan's Phases 5-6 pass, then
> is deleted in its Phase 7. Canonical-arena, residency, and lifecycle
> content here remains valid.

## Generic Compile-Neutral Runtime and Full Krea2 Migration

### Repository



# Decision summary

This refactor performs a **full Krea2 runtime migration**.

Ordinary Krea2 training and sampling use the generic compile-neutral runtime.
Delete the obsolete parallel Krea-specific execution systems after repository
search and tests prove that the generic runtime replaces their remaining
callers.

Remove the Krea-specific:

- regional compile path;
- old in-graph training and sampling paths;
- residency fingerprints and plan-specific executors;
- `*compiled*` and `*ingraph*` runtime fields;
- legacy immutable adapter;
- per-layer offload compatibility;
- compile-selection and restoration logic in `krea2.py`;
- related dead tests and diagnostics.

Retain only normal pure eager block math as a fallback and for
reference-image/reference-K/V calls. Keeping this mathematical fallback does
not justify retaining old Krea memory-management or compile orchestration.

Legacy memory backends used by other model architectures remain unless a
separate repository-wide reachability audit proves they are independently
dead.

Other explicit decisions:

1. Runtime setup has two phases:
   - canonicalization and initial residency during model loading;
   - execution-program finalization after LoRA/network installation.

2. Reference-image and reference-K/V calls remain on the normal eager block loop.

3. Whole-block residency is a memory-policy choice, not a compile requirement.

4. Preserve the existing source-snapshot representation first. Resolve resident sidecars through `ResidencyState` at execution time rather than retaining sidecar tensors in snapshots.

5. Use one exclusive execution context manager instead of separate normal and abort lease APIs.

6. Keep runtime ownership narrow. Existing planners, arena objects, compile-cache utilities, controllers, and diagnostic modules remain their own owners.

---
# Goal

Make Krea2 residency compile-neutral and move its ordinary training/sampling
memory and compile orchestration out of Krea2-specific model code. Complete
the migration by removing the superseded Krea execution systems rather than
maintaining parallel dispatch paths.

Changing the immutable residency layout must change only runtime source data. It must not rebuild or replace:

- compiled block kernels;
- eager block wrappers;
- train or sample trunks;
- checkpoint structure;
- block ordering.

The compiled block kernel must receive the same explicit tensor ABI whether a leaf comes from:

- a resident GPU sidecar; or
- a compact host-to-device fetch.

Krea2 should expose only the minimum architecture-specific contract needed by the generic runtime:

- repeated block collection;
- canonical leaf order;
- functional block call;
- block-specific arguments;
- LoRA argument order.

---
# Non-goals

Do not include these unless a failing test proves they are required:

- migration of other model architectures;
- a universal architecture-description framework;
- redesign of canonical arena storage or pinning;
- redesign of transfer custom ops;
- concurrent immutable executions;
- multiple live residency generations;
- residency changes during active execution;
- speculative Dynamo guards;
- new user-facing configuration flags;
- deletion of legacy memory backends still used by other architectures.

---
# Verified architectural constraints

## LoRA sequencing

Current loading intentionally separates:

1. transformer loading, freezing, canonicalization, and initial residency;
2. training-network or LoRA installation;
3. immutable executor construction.

The generic runtime must preserve this ordering.

Do not inspect, capture, or finalize LoRA execution data during the initial model-load phase.

## Reference execution

Reference-image execution has a different block interface from the normal denoising/training trunk.

It may include:

- tuple modulation;
- reference K/V collection;
- cached reference state;
- additional per-block inputs.

The immutable compiled executor does not need to support this path in the current refactor.

When reference state is requested, use the normal eager block loop.

## Existing Krea modes

The current Krea-specific regional compile, in-graph, plan-executor, and
per-layer compatibility paths are migration inputs, not retained product
modes. Trace every caller and configuration before deletion, move required
ordinary train/sample behavior to the generic runtime, and remove the old
fields, methods, dispatch branches, tests, and diagnostics in this refactor.

The only retained model-local execution path is the pure eager mathematical
block loop required for fallback and reference-image/reference-K/V calls.

---
# Target design

## 1. Minimal generic architecture adapter

Create a small single-stream MMDiT adapter in a generic location, for example:

```text
toolkit/memory_management/adapters/single_stream_mmdit.py
```

Use the repository structure to choose the final location. Do not create a larger plugin framework.

A minimal interface may expose:

```python
class SingleStreamMMDiTAdapter:
    def execution_blocks(self, transformer):
        ...

    def block_key(self, transformer, index: int) -> str:
        ...

    def leaf_entries(self, block):
        ...

    def build_lora_args(
        self,
        transformer,
        index: int,
        loras_by_block,
        multiplier,
    ):
        ...

    def forward_block(
        self,
        block,
        hidden,
        block_args,
        leaf_args,
        fp8_flags,
        lora_args,
        *,
        training: bool,
    ):
        ...
```

The adapter owns only architecture facts:

- where the repeated blocks are;
- canonical leaf ordering;
- how explicit weights map into block math;
- the block-specific functional call;
- LoRA leaf ordering.

It does not own:

- residency;
- transfer planning;
- fetch lifetime;
- checkpoint policy;
- compilation policy;
- memory budgeting;
- execution leases.

---

## 2. Narrow immutable runtime

Create or extract a generic immutable runtime, for example:

```text
toolkit/memory_management/immutable_runtime.py
```

The runtime orchestrates:

- source publication;
- permanent execution programs;
- execution contexts;
- immutable residency transitions;
- sampling phase entry and exit.

It does not become the owner of every related subsystem.

Keep ownership where it already exists:

- `CanonicalArena` owns canonical storage;
- `ResidencyState` owns sidecars;
- `ResidencyPlan` owns residency description;
- existing planner code owns memory-budget decisions;
- existing compile-cache utilities own cache persistence;
- existing controller code owns policy and measurements;
- existing diagnostics own reporting.

The runtime calls these components.

Attach one immutable runtime object:

```python
transformer._immutable_runtime
```

Remove superseded Krea execution fields after their callers move to this
runtime. Do not alter fields owned by other model architectures.

---

## 3. Two-phase lifecycle

### Phase A: prepare during model load

Add a generic preparation function such as:

```python
runtime = prepare_immutable_runtime(
    transformer,
    *,
    model_config,
    architecture_adapter,
    device,
)
```

This phase may:

1. discover canonical repeated blocks;
2. enumerate canonical leaf entries;
3. freeze the intended base weights;
4. construct and canonicalize the arena;
5. classify noncanonical modules as permanent residents;
6. obtain the existing smart memory plan;
7. create `ResidencyState`;
8. reconcile the initial training residency plan;
9. construct the generic source table;
10. bootstrap source snapshots from the current residency;
11. attach the unfinalized runtime to the transformer.

This phase must not:

- inspect training LoRA modules;
- build LoRA argument mappings;
- construct permanent train/sample wrappers that capture LoRA state;
- compile block kernels dependent on installed adapters.

### Phase B: finalize after LoRA/network installation

After the training network has been attached, call:

```python
runtime.finalize_execution(...)
```

This phase:

1. discovers the installed LoRA/network state;
2. builds stable LoRA argument mappings;
3. constructs permanent train wrappers;
4. constructs permanent sample wrappers;
5. constructs permanent train and sample trunks;
6. initializes the compiled block-kernel cache;
7. marks the runtime execution-ready.

`finalize_execution()` must be idempotent or fail clearly on repeated incompatible finalization.

A lazy first-execution finalization is acceptable only if it preserves the same sequencing and has a clear failure mode. Prefer an explicit trainer call because the current repository already has an established post-LoRA setup seam.

---

## 4. Source table

Extract and genericize the source-table mechanism already implemented on `GPT`.

Do not substantially redesign its representation.

A block source snapshot should contain current source-selection metadata such as:

```python
@dataclass(frozen=True)
class ImmutableBlockSourceSnapshot:
    block_key: str
    leaf_names: tuple[str, ...]
    transfer: BlockTransferPlan | None
    ranges: torch.Tensor | None
    fp8_flags: tuple[bool, ...]
```

At execution time:

- use the snapshot to determine which leaves are streamed;
- resolve resident leaves through `ResidencyState.resident_leaf(...)`;
- assemble every leaf into the same ordered tensor tuple.

Do not place direct sidecar tensors in the snapshot unless measurement or a concrete bug demonstrates a need.

This avoids snapshots retaining demoted GPU tensors.

### Publication

A residency publication must:

1. reject publication while execution is active;
2. validate the plan against the canonical arena;
3. prebuild transfer metadata;
4. reconcile sidecars;
5. build the next immutable snapshot tuple;
6. atomically publish the tuple;
7. update a diagnostic generation counter;
8. verify that canonical arena storage did not change.

The generation is diagnostic only. It is not a program key or compile key.

---

## 5. Permanent programs

After finalization, construct exactly once:

```python
self._train_block_fns
self._sample_block_fns
self._train_trunk
self._sample_trunk
```

Each block wrapper may capture:

- runtime;
- adapter;
- block index;
- block key;
- structural FP8 flags;
- permanent compiled kernel;
- stable LoRA metadata.

It must not capture:

- current residency plan;
- current source snapshot;
- resident sidecar tensors;
- current transfer ranges;
- residency generation.

Each wrapper looks up the current source snapshot when called.

### Compiled kernel

Compile only pure block math.

Use a stable ABI such as:

```python
kernel(
    hidden,
    tvec,
    freqs,
    mask,
    leaf_args,
    lora_args,
)
```

For the same block, resident and streamed execution must preserve:

- tuple arity;
- leaf ordering;
- tensor roles;
- tensor shape;
- tensor dtype;
- tensor device;
- `None` positions.

Do not include residency information in the kernel cache key.

A valid key may include:

- mode;
- block index or block identity;
- static FP8 flags;
- structural adapter shape when required.

---

## 6. Neutral MMDiT delegation seam

Add one generic-runtime delegation point to the normal block trunk.

For ordinary train/sample execution:

```python
runtime = self._immutable_runtime

if runtime is not None and runtime.can_run_current_call(...):
    return runtime.run_blocks(hidden, block_args)

return self._run_blocks_eager(...)
```

`can_run_current_call(...)` must reject reference-image/reference-K/V calls.

Keep reference execution on the eager mathematical loop. Once ordinary Krea
training and sampling run through the generic runtime, remove the old regional,
in-graph, plan-specific, and compile-selection dispatch branches rather than
preserving them in parallel.

The eager path remains only when:

- reference state uses the alternate block ABI;
- the generic runtime explicitly reports that the current call is unsupported;
- a pre-runtime load or diagnostic call genuinely requires pure block math.

---
# Residency policy

## Compile-neutrality versus whole-block policy

The source table may continue to support mixed resident/streamed leaves while preserving compile neutrality.

Whole-block residency is chosen as an explicit runtime policy because it simplifies:

- memory planning;
- transition accounting;
- transfer behavior;
- protection of checkpoint-sensitive blocks;
- future model generalization.

Do not claim partial-leaf residency is inherently incompatible with compilation.

## Policy enforcement

Apply whole-block selection in immutable controller/planner transitions.

The core source table does not need to reject partial plans if preserving that support makes extraction simpler.

At minimum:

- normal immutable training promotion selects complete blocks;
- normal immutable training demotion removes complete blocks;
- sampling fitting selects complete blocks;
- protected blocks remain complete and resident.

If an existing initial smart plan produces a partial block, normalize it at the conversion seam:

- fully resident only when the selected policy admits the block;
- otherwise fully streamed.

## Demotion

Use byte-targeted demotion:

```python
runtime.reduce_training_residency(required_relief_bytes)
```

Select complete adjustable blocks until predicted relief reaches the target.

Exclude protected blocks.

Do not demote immutable singleton modules.

## Promotion

Use the existing memory-budget calculation and fit complete blocks within the target budget.

Prefer currently resident blocks to avoid churn.

---

# Permanent singleton residency

Canonical dynamic residency includes only repeated canonical transformer blocks.

Keep noncanonical modules resident:

- input projections;
- output projections;
- text projectors;
- timestep MLPs;
- singleton text-fusion modules;
- embeddings;
- one-off layers;
- trainable adapters;
- LoRA parameters.

Derive these generically as the complement of canonical block leaves.

Do not keep Krea-specific singleton classification code unless generic discovery fails a test.

Immutable attachment must not create legacy `_layer_memory_manager` objects for either canonical blocks or singleton modules.

---

# Execution context

Use one context-manager interface:

```python
with runtime.execution(runtime.TRAIN):
    ...
```

and:

```python
with runtime.execution(runtime.SAMPLE):
    ...
```

Internally, hold one opaque active token and mode.

The context manager must:

1. reject nested or overlapping use;
2. set the active token;
3. yield;
4. clear the token in `finally`.

Residency publication and runtime close must reject while a token is active.

Separate `end_execution()` and `abort_execution()` APIs are unnecessary unless diagnostics later require different handling.

---

# Training integration

Primary expected location:

```text
jobs/process/BaseSDTrainProcess.py
```

Wrap the complete model-call/backward scope.

The current placement around `hook_train_loop(batch_list)` is correct if that function includes backward.

Use:

```python
runtime = getattr(unwrapped_transformer, "_immutable_runtime", None)

if runtime is None:
    with self.accelerator.accumulate(self.modules_being_trained):
        loss_dict = self.hook_train_loop(batch_list)
else:
    with runtime.execution(runtime.TRAIN):
        with self.accelerator.accumulate(self.modules_being_trained):
            loss_dict = self.hook_train_loop(batch_list)
```

Integrate with the existing OOM and `offload_step_end()`/`offload_step_abort()` structure.

The runtime context must exit:

- after backward has unwound;
- before post-step memory autotuning;
- before any residency transition.

If repository inspection shows backward occurs outside `hook_train_loop()`, move the context to the actual enclosing forward/backward scope.

Do not add custom autograd lease machinery unless the verified call graph requires it.

---

# Sampling integration

Add a runtime sampling context built on the same execution primitive:

```python
with runtime.sampling(gen_config, ...):
    image = pipeline(...)[0]
```

The context should orchestrate only the immutable path:

1. calculate or obtain the sampling resident budget;
2. choose an immutable sampling residency plan;
3. publish the plan;
4. configure existing compile-cache helpers where needed;
5. begin SAMPLE execution;
6. establish sampling measurement baseline;
7. yield to the pipeline;
8. record measurement on success;
9. clear measurement state on failure;
10. release execution in `finally`;
11. restore the training plan if current behavior requires immediate restoration.

Reference-image calls remain eager but may still use the immutable source table if the current eager functional path supports it safely. Do not route them through the compiled permanent trunk.

If the existing code intentionally bypasses immutable execution for reference calls, preserve that behavior in this refactor.

---

# MemoryManager integration

Add a dedicated generic Krea runtime attachment path and make it the only
memory/compile backend used by ordinary Krea training and sampling.

The Krea attach path may reuse existing manager/controller objects, but it must
not:

- install per-layer streaming forwards;
- create bounce managers;
- add `_layer_memory_manager`;
- attach singleton demotion candidates;
- select a regional or legacy in-graph Krea executor;
- reset traces because residency changed.

Keep existing memory planning and WDDM policy in their current modules. The
runtime exposes narrow transition calls that the existing controller invokes.

Do not delete legacy memory backends that remain reachable from other model
architectures. Remove Krea-specific compatibility and selection logic once all
ordinary Krea callers use the generic runtime.

---
# Krea2 cleanup scope

After the generic runtime owns ordinary Krea training and sampling, verify
reachability with repository search and focused configuration tests, then
remove the obsolete Krea-specific:

- regional compile path;
- old in-graph training and sampling paths;
- residency fingerprints and plan-specific executors;
- `*compiled*` and `*ingraph*` runtime fields;
- legacy immutable adapter;
- per-layer offload compatibility;
- compile-selection and restoration logic in `krea2.py`;
- immutable arena/source/executor construction moved to generic modules;
- dead diagnostics and tests that exist only for the removed paths.

Retain:

- the pure eager block-math loop used for fallback and
  reference-image/reference-K/V execution;
- model-specific loading, conditioning, sampling inputs, and mathematical
  behavior;
- memory backends still used by other architectures.

Do not keep compatibility shims after their last Krea caller has migrated.
Deletion is part of this refactor, not a deferred cleanup project.

---
# Implementation stages

## Stage 1: Extract current source table and executor

Start from the preliminary `GPT` implementation.

- move source-table and permanent-program mechanics into generic modules;
- preserve current snapshot behavior;
- preserve sidecar lookup through `ResidencyState`;
- preserve existing block fetch and ticket lifetime;
- add one consolidated program-identity transition test.

Do not redesign policy during extraction. Temporary import shims are permitted
only within the same stage and must be removed before completion.

## Stage 2: Add adapter and delegation seam

- add the minimal single-stream adapter;
- add the neutral generic-runtime block delegation;
- explicitly keep reference calls eager;
- route ordinary Krea train/sample calls through the new runtime;
- remove superseded regional, legacy in-graph, and plan-specific dispatch once
  the cutover tests pass.

## Stage 3: Add two-phase preparation and finalization

- prepare arena/residency during load;
- finalize wrappers, LoRA metadata, trunks, and kernels after network attachment;
- add tests proving preparation does not inspect LoRA;
- add tests proving finalization sees the installed adapters.

## Stage 4: Add execution contexts

- implement one exclusive execution context;
- integrate the training lifetime;
- integrate the sampling lifetime;
- test nested rejection, publication rejection, normal release, and exception release in one transition suite.

## Stage 5: Route runtime transitions

- make generic Krea runtime attachment structurally exclusive;
- route training promotion/demotion through runtime publication;
- route sampling transitions through runtime publication;
- apply whole-block policy in controller plan selection;
- retain existing planner/controller ownership;
- assert no explicit trace reset;
- remove Krea per-layer offload compatibility and restoration logic.

## Stage 6: Full Krea cleanup and validation

- perform a repository-wide reachability audit of old Krea execution fields,
  methods, configuration branches, diagnostics, and tests;
- delete obsolete regional compile, old in-graph, plan-executor, compatibility,
  and compile-selection code;
- retain only pure eager/reference math and the generic runtime path;
- run counting-backend compile tests;
- run resident/streamed numerical comparisons;
- run LoRA-gradient comparisons;
- run repeated train-sample-train transitions;
- run OOM and sampling-exception recovery tests;
- verify other architectures still retain any legacy memory backends they use.

No prescribed commit count is required. Use reviewable commits aligned with these six stages.

---
# Tests

## 1. Consolidated permanent-object test

Construct and finalize one runtime.

Capture together:

- train program;
- sample program;
- train trunk;
- sample trunk;
- train wrapper tuple;
- sample wrapper tuple;
- compiled kernel cache entries.

Publish several residency layouts.

Assert all captured permanent object identities remain unchanged.

Also assert source snapshots changed.

## 2. Execution transition suite

In one focused suite, cover:

- normal execution release;
- exception release;
- nested execution rejection;
- overlapping mode rejection;
- residency publication rejection while active;
- publication success after release;
- close rejection while active.

## 3. Two-phase lifecycle tests

Verify:

- prepare phase succeeds before LoRA installation;
- prepare phase does not create LoRA mappings;
- finalization after LoRA installation creates mappings;
- finalization builds permanent programs once;
- execution before finalization fails clearly;
- repeated compatible finalization is harmless or clearly rejected.

## 4. Reference-path test

Verify a reference-image/reference-K/V call:

- bypasses the immutable compiled trunk;
- reaches the normal eager block path;
- retains correct reference behavior;
- does not corrupt the immutable runtime for the next normal call.

## 5. Migration and deletion tests

Verify representative former Krea configurations now select the generic runtime
for ordinary training and sampling. Verify reference calls select the pure eager
block loop.

Add repository-search assertions or focused structural tests for removal of:

- regional compile dispatch;
- old in-graph Krea dispatch;
- plan-specific executor selection;
- `*compiled*` and `*ingraph*` runtime fields;
- per-layer Krea offload compatibility;
- obsolete compile restoration state.

Do not delete or retarget legacy backend tests belonging to other model
architectures.
## 6. Counting-backend compile test

For one warmed input signature:

1. execute TRAIN;
2. record compile count;
3. publish another residency layout;
4. execute the same TRAIN signature;
5. assert no new block-kernel compilation;
6. repeat for SAMPLE;
7. repeat across several transitions.

New shapes may compile.

## 7. Numerical CUDA tests

With a small model:

- compare all-resident output;
- compare all-streamed output;
- compare mixed-source output if the source table still supports it;
- compare whole-block policy layouts;
- compare LoRA gradients;
- verify base weights remain frozen;
- verify canonical host pointers remain stable;
- verify demoted sidecar memory is reclaimable.

## 8. Train-sample transition smoke

Run repeatedly:

```text
train -> sample -> train
```

Include:

- residency changes;
- sampling exception;
- training OOM or simulated failure;
- recovery execution afterward.

Check:

- no lease leak;
- no stale source;
- no ticket leak;
- no residency-triggered compile;
- no canonical storage mutation.

## 9. Trace test

Patch the explicit trace-reset function.

Assert immutable residency publication does not call it.

Natural trace misses are allowed.

---

# Validation

Run syntax checks for changed files.

Run focused runtime tests first.

Run Ruff on changed Python files.

Run relevant memory-management and Krea2 tests.

Run CUDA tests in isolated processes when known WDDM/CUDA cross-test leakage makes combined execution unreliable.

Do not weaken correctness tests to accommodate unrelated environmental instability.

---

# Acceptance criteria

The refactor is complete when:

## Lifecycle

- canonicalization and initial residency occur before LoRA attachment;
- permanent execution programs are finalized after LoRA attachment;
- reference execution remains eager;
- ordinary Krea training and sampling use the generic runtime.

## Compile neutrality

- residency changes replace source snapshots only;
- permanent wrappers and trunks retain identity;
- compiled kernels retain identity;
- warmed signatures do not compile again because residency changed;
- residency fingerprints are not compile keys.

## Residency

- singleton modules remain resident;
- immutable mode creates no per-layer managers;
- whole-block residency is applied by immutable policy transitions;
- source-table ABI remains valid even though compile neutrality does not inherently require whole-block policy;
- canonical arena storage remains unchanged.

## Execution lifetime

- training execution covers forward, recomputation, and backward;
- sampling execution covers the complete pipeline call;
- exceptions release execution state;
- transitions are rejected while active;
- execution can resume after failures.

## Migration completeness

- ordinary Krea training and sampling have one generic runtime path;
- pure eager math remains available for fallback and reference execution;
- obsolete Krea regional, old in-graph, plan-executor, per-layer compatibility,
  and compile-selection code is removed;
- no compatibility shim remains around the new runtime;
- legacy memory backends remain available to other configurations and models
  that still use them.
## Correctness

- resident and streamed outputs match within dtype tolerance;
- LoRA gradients match;
- base weights remain frozen;
- reference behavior remains correct;
- no ticket or sidecar lifetime leak is introduced.
