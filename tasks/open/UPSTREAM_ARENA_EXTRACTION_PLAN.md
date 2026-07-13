# Arena Offload Extraction Refactor — Remaining Implementation Plan

> **git-bug:** umbrella ticket `553ffec` (“Arena offload: pre-PR refactor and
> upstream extraction”). Mutable status, validation results, and next-agent
> handoff notes live in git-bug; this file is the durable implementation plan.

## Purpose

Complete the arena-offload extraction after Phase 5.

The remaining work must establish clear lifecycle ownership, remove cross-boundary access to arena internals, delete obsolete legacy-manager branches, and validate the extracted runtime.

This plan supersedes:

* the existing Phase 3 requirement that a failed committed build restore an eagerly executable transformer;
* the existing Phases 6–8;
* validation criteria requiring eager execution after destructive arena setup;
* any implied automatic fallback from arena offload to eager or legacy offload.

The implementation should follow Toolkit’s current supported execution model:

* one active training process and pipeline per worker process;
* one arena-managed primary transformer per training process;
* one local CUDA device per worker process;
* one active arena runtime per worker process;
* distributed training may use multiple worker processes, each owning its own local model, device, and arena runtime;
* configured processes and jobs execute sequentially;
* a later process or job may reuse the worker only after the previous arena runtime has closed successfully.

Auxiliary components such as the text encoder, VAE, adapters, LoRA network, and optimizer do not count as separate arena runtimes.

Do not add concurrent multi-pipeline, multi-runtime, or multi-device-in-one-process support without a concrete caller. Toolkit does not currently provide the lifecycle, scheduling, memory ownership, or error-isolation architecture required for those modes; adding them would be a separate architectural redesign.


---

# Core lifecycle contract

Arena preparation has two distinct failure boundaries.

## Before canonical commit

Before canonical parameter views are published, setup remains transactional.

Failures during any of the following must leave the transformer unchanged:

* architecture validation;
* layout inspection;
* destination allocation;
* checkpoint or cache population;
* quantization;
* pin registration;
* wrapper reconstruction validation;
* process runtime ownership acquisition.

Required behavior:

1. Release all storage, pins, temporary tensors, and ownership tokens acquired by the attempt.
2. Publish no arena runtime.
3. Leave the model’s parameters and movement methods unchanged.
4. Raise the original setup error.
5. Fail the current job.
6. Permit `--recover` to continue to another independent configured job.

No fallback backend is attempted.

## After canonical commit

Canonical commit is destructive.

Once the frozen model parameters have been replaced by views into canonical arena storage, the transformer is consumed by that setup attempt.

Failures during any of the following are fatal to the worker process:

* initial residency reconciliation;
* transfer-runtime initialization;
* immutable executor construction;
* runtime publication;
* permanent-module placement;
* runtime finalization after LoRA or adapter attachment;
* any later setup step that occurs before the runtime becomes usable.

Required behavior:

1. Perform best-effort release of every resource acquired by the attempt.
2. Do not reconstruct the original model.
3. Do not attempt eager execution.
4. Do not fall back to `MemoryManager`.
5. Do not continue to another configured job.
6. Raise `ArenaSetupFatalError`, preserving the original exception as its cause.
7. Terminate the worker process with a non-zero result.

`--recover` does not apply after destructive canonical commit.

## Closed transformer contract

A transformer is disposable after arena runtime close.

Normal close must:

* drain or abandon runtime-owned transfer work;
* release resident sidecars;
* release canonical pinned storage;
* restore intercepted movement methods;
* remove externally published runtime state;
* release process ownership;
* mark the transformer as disposed.

Normal close does not restore the pre-arena parameter objects and does not make the transformer eagerly executable.

Any attempt to execute, move, or prepare a disposed transformer must fail clearly.

---

# Prerequisite — Correct quantization D2H behavior

Port the upstream blocking CPU transfer fix before lifecycle work or memory validation.

Replace the leaking asynchronous CPU transfer:

```python
block.to("cpu", non_blocking=True)
```

with the blocking form:

```python
block.to("cpu")
```

Use the exact upstream implementation where available.

## Reason

The asynchronous transfer may retain pinned staging allocations and contaminate:

* pin-leak tests;
* runtime teardown measurements;
* sequential-job testing;
* host-memory graphs;
* performance baselines.

## Acceptance

* The upstream change is ported as a dedicated small commit.
* Existing quantization tests pass.
* Quantization no longer leaves unexpected D2H staging allocations active after completion.

---

# Phase 6 — Fail-closed lifecycle and single-runtime ownership

## Outcome

Arena preparation, execution, and teardown share one explicit resource owner.

The process supports one active arena runtime at a time.

Pre-commit failures are transactional. Post-commit failures are process-fatal.

---

## 6.1 Add minimal process runtime ownership

Introduce a small process-global ownership guard in the arena transfer/runtime package.

The ownership record contains only:

```python
active_owner_token
device
```

Ring depth, statistics, streams, slots, and ticket state remain transfer-runtime state rather than ownership identity.

### Acquisition

Arena preparation acquires the process slot before any destructive model mutation.

Acquisition must:

* create a unique opaque owner token;
* normalize the requested CUDA device;
* reject a second active owner;
* avoid resetting an existing runtime’s transfer state;
* leave the model untouched on failure.

A second active runtime should raise a clear pre-commit setup error such as:

```text
arena_runtime_already_active:
active_device=cuda:0 requested_device=cuda:0
```

Do not add waiting, multiplexing, or automatic takeover.

### Validation boundaries

Validate the owner token at meaningful process-global boundaries:

* ownership acquisition;
* transfer-runtime configuration;
* process-global custom operator entry points where stale compiled code could survive;
* ticket drain or abandonment;
* final ownership release;
* callbacks that may outlive the runtime.

Do not add repetitive token checks to every ordinary runtime method.

### Release

Ownership is released only after:

* active transfer execution has ended;
* live tickets have been drained or abandoned;
* pending timing events have been cleared or finalized;
* device ring buffers have been released;
* runtime reporting-window state has been reset;
* no runtime-owned work remains queued on transfer streams.

A later sequential job may then acquire ownership, including for another CUDA device.

---

## 6.2 Introduce preparation-scoped resource ownership

Create an explicit resource owner before process ownership is acquired.

Suggested shape:

```python
class ArenaRuntimeResources:
    def __init__(self, model, device):
        self.model = model
        self.device = device

        self.owner_token = None
        self.canonical_build = None
        self.arena = None
        self.residency = None
        self.executor = None

        self.fp8_restores = []
        self.published_attributes = []
        self.movement_guard_installed = False
        self.runtime_published = False
        self.canonical_committed = False
        self.disposed = False
        self.released = False

    def release(self) -> None:
        ...
```

Exact fields may differ after inspecting the current implementation. Keep the object specific to the arena lifecycle; do not build a generic framework.

### Ownership model

Resources are adopted incrementally:

```python
resources = ArenaRuntimeResources(model, device)

resources.acquire_process_owner()
resources.adopt_canonical_build(build)
resources.mark_canonical_committed(arena)
resources.adopt_residency(residency)
resources.adopt_executor(executor)
resources.record_published_attribute(...)
resources.record_fp8_restore(...)
```

After successful construction:

```python
runtime = ArenaOffloadRuntime(
    resources=resources,
    ...
)
```

The runtime adopts the same resource owner. It does not copy ownership into separate fields that can drift from failure cleanup.

### Why this is required

Failures can happen after canonical commit but before an `ArenaOffloadRuntime` object exists.

Therefore cleanup cannot depend on:

```python
runtime.close()
```

A preparation-scoped owner must exist throughout setup.

---

## 6.3 Implement one idempotent release operation

Both failed preparation and normal runtime close delegate to:

```python
resources.release()
```

The release operation must tolerate:

* failure before any resource was acquired;
* partially created residency state;
* partially initialized executor state;
* already drained transfer state;
* partially published model attributes;
* repeated calls;
* errors from individual cleanup steps.

### Release ordering

Release in dependency order:

1. Mark the resource owner as closing so no new execution begins.
2. Exit or reject active runtime execution.
3. Drain or abandon transfer tickets owned by this runtime.
4. Close the immutable executor.
5. Disable temporary FP8 training or sampling transformations.
6. Clear resident sidecars.
7. Remove externally published runtime and compatibility attributes.
8. Restore the model’s original movement methods.
9. Release canonical pin registrations and host storage.
10. Reset transfer-runtime state owned by this token.
11. Release the process ownership slot.
12. Mark the transformer disposed when canonical commit occurred.
13. Mark resources released.

Where cleanup operations are independent, continue attempting later cleanup after one step fails.

Aggregate cleanup errors for reporting, but preserve any original setup exception.

### Normal close

`ArenaOffloadRuntime.close()` becomes:

```python
def close(self) -> None:
    self._resources.release()
```

It must be safe to call multiple times.

### Failed preparation

Preparation follows the structure:

```python
resources = ArenaRuntimeResources(transformer, device)

try:
    resources.acquire_process_owner()

    validate_architecture(...)
    build = prepare_canonical_storage(...)
    resources.adopt_canonical_build(build)

    populate(build)
    build.commit()

    resources.mark_canonical_committed(build.arena)

    residency = prepare_residency(...)
    resources.adopt_residency(residency)

    executor = prepare_executor(...)
    resources.adopt_executor(executor)

    runtime = ArenaOffloadRuntime(
        resources=resources,
        ...
    )

    publish_runtime(transformer, runtime)
    resources.mark_runtime_published()

    return runtime

except BaseException as error:
    committed = resources.canonical_committed

    try:
        resources.release()
    except BaseException as cleanup_error:
        record_cleanup_failure(error, cleanup_error)

    if committed:
        raise ArenaSetupFatalError(
            "arena setup failed after canonical commit"
        ) from error

    raise
```

Do not wrap pre-commit failures in `ArenaSetupFatalError`.

---

## 6.4 Add narrow fatal error classification

Define:

```python
class ArenaSetupFatalError(RuntimeError):
    """Arena setup failed after destructive canonical commit."""
```

Use this exception only when:

* canonical commit completed;
* setup did not reach a usable runtime;
* the transformer can no longer be treated as an ordinary model.

Do not use it for:

* unsupported architecture;
* invalid configuration;
* checkpoint read failure before commit;
* quantization failure before commit;
* pin-budget rejection before commit;
* another active runtime;
* direct loader population failure before commit.

The original exception must remain available through exception chaining.

---

## 6.5 Integrate fatal behavior into the job entry point

Update the job runner so `ArenaSetupFatalError` overrides `--recover`.

Required behavior:

```python
try:
    job.run()
except ArenaSetupFatalError:
    run_best_effort_job_cleanup()
    report_failure()
    raise
except Exception:
    run_best_effort_job_cleanup()
    if not args.recover:
        raise
```

The exact exception may be raised through several wrapper layers. Preserve classification rather than flattening it into a generic string-only error.

### Process termination

Do not call `os._exit()` directly from arena setup.

Preferred sequence:

1. Propagate the fatal exception.
2. Run best-effort normal cleanup in the existing `finally`.
3. Allow the worker to terminate naturally with a non-zero status.
4. Keep the existing error-exit watchdog armed during cleanup.
5. If cleanup hangs, let the watchdog force-terminate the process tree.

A fatal arena setup error must never start the next configured job.

---

## 6.6 Mark committed transformers as disposable

After successful canonical commit, resource release must mark the transformer disposed.

Use one clear mechanism.

Preferred options:

### Option A — Retain a closed runtime façade

Leave the runtime marker installed after close, but mark it closed:

```python
runtime.closed = True
runtime.disposed = True
```

Every public runtime operation raises:

```text
arena_offload_transformer_disposed
```

This has the advantage that generic runtime discovery still sees why the model is unusable.

### Option B — Publish a dedicated disposed marker

Remove the runtime marker and set:

```python
transformer._arena_offload_disposed = True
```

Generic model preparation and movement helpers must reject this marker.

Choose the smaller option after inspecting existing runtime discovery and cleanup code.

Do not leave a released transformer looking like an ordinary reusable model.

---

## 6.7 Phase 6 tests

Add focused tests for:

### Ownership

* first runtime acquires the process slot;
* simultaneous second runtime fails before model mutation;
* closing the first runtime releases the slot;
* a sequential second runtime can acquire it;
* sequential acquisition on a different device identity is accepted after release;
* stale owner tokens cannot release a newer runtime.

### Pre-commit failures

Inject failures in:

* architecture validation;
* layout inspection;
* allocation;
* direct population;
* quantization;
* pin registration;
* wrapper reconstruction;
* ownership acquisition.

Assert:

* parameters unchanged;
* movement methods unchanged;
* no runtime marker;
* no disposed marker;
* no process owner;
* no pin leak;
* normal job failure classification;
* `--recover` may continue.

### Post-commit failures

Inject failures in:

* residency construction;
* initial residency reconcile;
* transfer initialization;
* executor construction;
* runtime publication;
* permanent-module placement;
* finalization.

Assert:

* `ArenaSetupFatalError`;
* original error retained as cause;
* no eager or legacy fallback;
* resource cleanup attempted;
* process owner released where cleanup succeeds;
* transformer marked disposed;
* `--recover` does not continue;
* the next configured job does not start.

### Close

Assert:

* close is idempotent;
* movement interception is removed or converted into the disposed guard;
* resident sidecars released;
* executor closed;
* transfer tickets drained or abandoned;
* pin registrations released;
* process owner released;
* disposed transformer cannot execute or be prepared again.

---

## Phase 6 acceptance

* Preparation and runtime close share one resource owner.
* No cleanup path requires a fully constructed runtime object.
* Pre-commit failures leave the model unchanged.
* Post-commit setup failures raise `ArenaSetupFatalError`.
* Fatal setup failure ignores `--recover`.
* No automatic eager or legacy fallback exists.
* Only one arena runtime may be active per process.
* Sequential jobs may reacquire ownership after successful close.
* A closed committed transformer is explicitly disposable and unusable.
* Repeated cleanup does not leak or corrupt state.

---

# Phase 7 — Generic façade cutover

## Outcome

Shared trainer code and Krea-specific code interact only with the generic memory-runtime façade.

Arena internals no longer leak through model attributes into shared or model-specific code.

---

## 7.1 Define the generic runtime interface

Shared code should need only operations equivalent to:

```python
runtime = get_memory_runtime(transformer)

runtime.finalize(network)
runtime.set_compile_dynamic_hints(hints)

with runtime.training_step(
    shape_key=shape_key,
    step_num=step_num,
):
    forward_and_backward()

with runtime.sampling_session():
    with runtime.sampling_image(
        shape_key=shape_key,
        cold_working_bytes=working_bytes,
    ):
        sample()

runtime.place_permanent_modules(device, dtype)
runtime.diagnostics()
runtime.close()
```

Compile integration should use one generic query:

```python
memory_runtime_owns_compile(transformer)
```

Do not branch on arena-private fields.

The generic façade may be implemented by the existing runtime-discovery module if appropriate. Do not create a second overlapping façade.

---

## 7.2 Reduce shared trainer integration

Update shared trainer code so it:

* discovers the runtime once through the generic API;
* finalizes it after the training network or adapter is attached;
* wraps the complete forward and backward region in one runtime context;
* forwards dynamic compile hints through the runtime;
* delegates teardown to generic runtime close;
* obtains diagnostics through the runtime API;
* does not import arena, residency, policy, layout, transfer, or architecture-adapter internals.

Remove shared-code reads of:

```text
_mm_canonical_arena
_mm_residency_state
_mm_immutable_training_plan
_mm_immutable_smart_plan
_mm_immutable_canonical_modules
_mm_immutable_protected_training_leaf_keys
_mm_immutable_backend
_immutable_runtime
```

Remove legacy immutable tracing, execution-lease, pre-step, post-step, and private-plan calls when the runtime now owns those behaviors.

Generic behavior for non-arena models must remain unchanged.

---

## 7.3 Reduce Krea integration

Krea-specific memory and compile code should be limited to:

* selecting arena versus legacy backend;
* selecting the architecture adapter;
* converting model configuration into `ArenaOffloadConfig`;
* preparing direct canonical destinations;
* populating those destinations from Krea loader paths;
* calling generic runtime lifecycle and sampling contexts;
* providing actual sampling-shape information required by the runtime.

Krea must not:

* construct canonical arenas directly;
* calculate packed layouts;
* construct quantized wrappers;
* manipulate residency state;
* convert residency plans;
* install compile trunks;
* configure transfer rings;
* publish compatibility attributes;
* inspect runtime-private state.

---

## 7.4 Remove cross-boundary `_mm_*` publication

The transformer should expose one externally meaningful arena runtime façade.

Preferred external marker:

```python
transformer._arena_offload_runtime
```

or the existing generic memory-runtime marker if one already exists.

Remove model-level publication used by shared, model-specific, or legacy-manager code to rediscover:

* the arena;
* residency;
* training plans;
* smart plans;
* canonical module lists;
* executor/backend state;
* protected leaf sets.

### Arena-private metadata

Do not delete an attribute merely because its name begins with `_mm_`.

For metadata such as:

```text
_mm_canonical_leaf
```

first determine whether it is:

* read outside the arena package;
* used by the legacy manager;
* required only because dependencies are not passed explicitly;
* or genuinely the simplest arena-internal marker.

Acceptance is based on ownership boundaries:

* no shared code reads arena-private attributes;
* no model-specific code reads arena-private attributes;
* no legacy-manager compatibility state is published;
* arena-private implementation metadata may remain if only the arena package consumes it.

Rename internal metadata when useful, but do not introduce a larger registry solely to eliminate a private attribute.

---

## 7.5 Complete arena-native helper ownership

Remove remaining imports and calls from `arena_offload` into:

```text
MemoryManager
manager_modules
```

Move only the behavior the arena runtime actually requires into neutral or arena-owned modules.

Likely remaining categories include:

* WDDM margin resolution;
* arena sampling reserve calculation;
* FP8 training transforms;
* FP8 sampling transforms;
* singleton module selection;
* immutable planner compatibility;
* runtime diagnostics.

Inspect each dependency before moving it.

Rules:

* Move the smallest pure helper or arena-specific behavior.
* Do not duplicate the full manager.
* Do not move unrelated per-linear logic.
* Do not preserve manager access merely through a renamed wrapper.
* Keep text-encoder memory management on the legacy manager.

---

## 7.6 Compile ownership

Compile ownership must be exclusive.

Arena models:

* compile through the arena runtime;
* skip generic block compile;
* support compile-disabled eager arena execution through the runtime;
* must not be compiled a second time by shared trainer paths.

Non-arena models:

* retain the existing generic block compile behavior;
* retain legacy per-linear memory-management behavior.

Replace private-field special cases with:

```python
memory_runtime_owns_compile(transformer)
```

---

## Phase 7 tests

Add or update tests proving:

* shared trainer files contain no arena implementation imports;
* Krea code contains no arena construction, residency, or compile internals;
* no shared or Krea code reads arena-private `_mm_*` state;
* `arena_offload` imports neither `manager` nor `manager_modules`;
* runtime finalization still occurs after LoRA or adapter attachment;
* one runtime context spans forward and backward;
* sampling returns to the training runtime state;
* arena compile ownership is exclusive;
* non-arena generic compile remains unchanged;
* text-encoder offload still uses the legacy manager.

---

## Phase 7 acceptance

* Shared trainer integration uses only the generic runtime façade.
* Krea-specific integration is reduced to selection, configuration, direct population, and generic contexts.
* Arena runtime code does not import the legacy manager.
* No cross-boundary consumer reads arena-private model attributes.
* Compile ownership is exclusive.
* Legacy non-arena behavior remains intact.

---

# Phase 8 — Remove obsolete legacy branches and settle configuration

## Outcome

Delete only the legacy manager machinery that existed to support the now-extracted arena runtime.

Do not perform a general `MemoryManager` rewrite.

---

## 8.1 Delete arena-only manager branches

After Phase 7 leaves no consumers, remove:

* `attach_smart_training_immutable`;
* immutable branches inside `attach_smart_training`;
* immutable planner conversion into arena residency plans;
* arena canonical-sidecar relief routed through the manager;
* immutable-runtime inspection in manager diagnostics;
* `_mm_immutable_*` publication and handling;
* legacy manager state used only by arena execution;
* arena-specific per-linear compatibility hooks;
* obsolete compile ownership special cases;
* unused immutable execution tracing or leasing paths.

Before deletion, grep all callers and prove they have moved to the runtime façade.

---

## 8.2 Retain supported legacy functionality

Do not remove or broadly refactor:

* `MemoryManager.attach`;
* `MemoryManager.detach`;
* Linear memory management;
* Conv memory management;
* OstrisLinear memory management;
* legacy transformer percentage offload;
* text-encoder offload;
* generic block compile for non-arena models;
* models that still depend on per-linear management;
* unrelated memory-manager experiments.

This phase is deletion of arena-specific compatibility code, not modernization of the whole manager.

---

## 8.3 Settle arena configuration

Use a narrow configuration object equivalent to:

```python
@dataclass(frozen=True)
class ArenaOffloadConfig:
    enabled: bool
    fp8_forward: bool
    fp8_backward: bool
    fp8_sampling: bool
    compile_blocks: bool
```

Keep compile behavior derived from existing compile settings. Do not create a second public compile system.

Internal automatic fields may include:

* device;
* compile dynamic mode;
* compile dynamic hints;
* adapter-derived protected blocks;
* automatically selected transfer depth;
* automatically calculated memory policy.

Do not add public controls for:

* ring depth;
* residency percentage;
* promotion cadence;
* safety margins;
* pinned-memory budget;
* working reserve;
* transfer profiling;
* trace capture;
* WDDM policy internals.

Those remain implementation details unless a real user-facing requirement appears.

---

## 8.4 Configuration failure rules

### Unsupported architecture

Detect architecture support before canonical commit.

Behavior:

* fail the current job;
* leave the transformer unchanged;
* do not fall back;
* permit `--recover` to continue.

This is not `ArenaSetupFatalError`.

### Invalid FP8 combination

Choose the smallest semantically correct behavior:

* fail validation when continuing would be incorrect;
* otherwise disable an irrelevant option with one warning.

Examples should be based on current kernel and weight support rather than hypothetical combinations.

### Arena plus percentage offload

Arena owns the transformer when explicitly enabled.

Legacy transformer offload percentage remains meaningful only when the legacy backend is selected.

Text-encoder offload remains independent.

### Compatibility aliases

Retain existing fork configuration aliases only long enough to validate migrated jobs.

Document mappings explicitly.

Remove aliases in a separate compatibility cleanup after extraction acceptance, unless current users require them for the extraction PR.

---

## Phase 8 tests

Prove:

* no deleted arena-manager branch has a caller;
* legacy per-linear transformer offload still works;
* text-encoder offload still works;
* arena disabled behavior is unchanged;
* unsupported architecture fails before model mutation;
* unsupported architecture may recover to another job;
* arena never silently falls back;
* configuration aliases map correctly;
* arena FP8 training and sampling controls remain independent;
* non-arena generic compile still works.

---

## Phase 8 acceptance

* The legacy manager contains no arena-specific execution path.
* Per-linear and text-encoder memory management remain available.
* Configuration has one clear backend selection.
* Unsupported architectures fail before commit.
* No automatic arena fallback exists.
* Temporary compatibility mappings are explicit and bounded.

---

# Phase 9 — Core extraction acceptance

## Outcome

Prove the extracted runtime is correct, fail-closed, leak-free, and performance-equivalent or better.

This phase blocks closure of the extraction ticket.

---

## 9.1 Direct loading

Validate both Krea direct population paths:

### Ranged checkpoint loading

Assert:

* weights are loaded into final canonical destinations;
* no second full canonical model payload is created;
* source blocks are released as soon as their final destinations are populated;
* assistant LoRA or frozen-base composition occurs in the required semantic order;
* final canonical values are correct.

### Quantized cache loading

Assert:

* cached quantized values populate final arena destinations directly;
* the full cache is not first assigned to the transformer and copied again;
* wrapper reconstruction uses the final arena leaves;
* no second model-sized canonical payload exists.

### Compatibility loading

Retain and test:

```python
build.populate_from_model()
```

for supported models whose loaders have not adopted direct destinations.

This remains a compatibility path, not Krea’s performance path.

---

## 9.2 Fault injection

### Pre-commit

Inject failure at each meaningful boundary:

* architecture validation;
* layout inspection;
* allocation;
* direct checkpoint population;
* direct quantized-cache population;
* compatibility population;
* quantization;
* pin registration;
* wrapper validation;
* process ownership acquisition.

Assert unchanged model state and complete resource cleanup.

### Post-commit

Inject failure at:

* initial residency creation;
* residency reconcile;
* transfer configuration;
* executor construction;
* runtime publication;
* permanent-module placement;
* finalization;
* compile-program creation before first usable execution.

Assert:

* `ArenaSetupFatalError`;
* no eager fallback;
* no legacy fallback;
* cleanup attempted;
* disposed-transformer state;
* no next job under `--recover`.

Do not assert that the failed transformer runs eagerly.

---

## 9.3 Resource lifecycle

Assert no leak of:

* pin-manager handles;
* registered arena storage;
* prepared host flats;
* resident sidecars;
* transfer tickets;
* pending H2D timing events;
* device ring slots;
* transfer streams that retain runtime-owned work;
* process owner tokens;
* temporary FP8 transformations;
* runtime markers or compatibility publication.

Test:

* repeated close;
* close after partial preparation;
* close after finalization;
* close after training;
* close after sampling;
* close after a handled runtime failure;
* successful sequential runtime acquisition.

---

## 9.4 Job process behavior

Core process tests:

### Pre-commit error with `--recover`

* current job fails;
* cleanup runs;
* next configured job begins.

### Post-commit fatal error without `--recover`

* current job fails;
* cleanup runs where possible;
* process exits non-zero.

### Post-commit fatal error with `--recover`

* current job fails;
* cleanup runs where possible;
* the next configured job does not begin;
* process exits non-zero.

### Cleanup failure

* original setup failure remains the primary error;
* cleanup failure is reported separately;
* watchdog remains armed until process termination or successful cleanup completion.

---

## 9.5 Functional matrix

Run the following representative paths:

* arena BF16 training;
* arena FP8 forward;
* arena FP8 forward and backward;
* precise training with FP8 sampling;
* sampling before training;
* sampling after multiple training steps;
* train → sample → train transition;
* arena transformer with legacy text-encoder offload;
* arena disabled;
* legacy per-linear transformer offload;
* mixed-resolution training;
* DOP or multiple-forward execution;
* whole-block residency promotion;
* whole-block residency demotion;
* OOM-driven provisional-layout rollback;
* compile enabled;
* compile disabled;
* Windows WDDM path;
* non-Windows CUDA fallback.

Use the smallest representative matrix sufficient to exercise each unique ownership and execution path. Do not multiply equivalent combinations without evidence.

---

## 9.6 Compile validation

Verify:

* arena functional kernels compile once per legitimate shape specialization;
* generic block compile is skipped for arena models;
* non-arena generic block compile remains unchanged;
* sampling does not force unnecessary training recompilation;
* training does not force unnecessary sampling recompilation;
* compile-disabled arena execution remains correct;
* dynamic compile options are forwarded through the runtime;
* no new Dynamo frame or recompile storm appears relative to baseline.

---

## 9.7 Performance validation

Compare with the established pre-extraction baseline.

Measure:

* first training compile duration;
* first sampling compile duration;
* steady-state training step time;
* sampling step time;
* H2D transfer duration;
* transfer wait or stall duration;
* transfer duty;
* achieved transfer bandwidth;
* GPU utilization;
* resident bytes;
* streamed bytes;
* ring bytes;
* peak allocated CUDA memory;
* peak reserved CUDA memory;
* pinned host bytes;
* train-to-sample transition time;
* sample-to-train transition time.

The extraction is not ready to close if it:

* restores multi-minute repeated compilation;
* introduces repeated train/sample recompilation;
* materially worsens steady-state transfer overlap;
* reintroduces a model-sized temporary canonical payload;
* leaks pinned storage between sequential jobs;
* reduces functional coverage relative to the working pre-extraction path.

Document material differences rather than requiring bit-identical timings.

---

# Environment validation

These checks are useful but are not part of the core architectural acceptance gate unless a relevant environment is available.

Run separately:

* detached UI worker successful termination;
* detached UI worker failure termination;
* intentionally hung cleanup followed by watchdog process-tree termination;
* Windows `taskkill` process-tree behavior;
* broader legacy backend combinations;
* distributed worker behavior;
* uncommon platform or driver combinations.

Failures here should create narrowly scoped follow-up tickets unless they expose a defect in the core lifecycle contract.

---

# Explicit non-goals

Do not include the following in this extraction:

1. Concurrent arena runtimes in one process.
2. Multiple active CUDA devices in one process.
3. Runtime ownership multiplexing.
4. Restoring a committed or closed transformer to eager execution.
5. Automatic fallback from arena to legacy offload.
6. Automatic fallback from arena to eager execution.
7. General cleanup or redesign of `MemoryManager`.
8. Arena offload for the text encoder.
9. Conversion of additional model architectures without a real integration target.
10. Public memory-policy tuning controls.
11. General SDPA policy cleanup.
12. Prompt-budget estimation fixes.
13. UI path-validation fixes.
14. Checkpoint-saver timeout changes.
15. LoRA vector explorer compatibility.
16. Hook exit propagation.
17. Atomic-write durability work.
18. PID-reuse detection.
19. General cleanup counters or telemetry not required by acceptance.
20. A generic resource-management framework beyond the arena runtime’s needs.

Create separate tickets for unrelated static-review findings.

---

# Required implementation order

1. Port the upstream blocking quantization D2H fix.
2. Add process-global single-runtime ownership.
3. Add `ArenaRuntimeResources`.
4. Route preparation ownership through the resource owner.
5. Implement idempotent resource release.
6. Add the pre-commit/post-commit failure boundary.
7. Add `ArenaSetupFatalError`.
8. Integrate fatal handling into the job entry point.
9. Add disposed-transformer protection.
10. Complete shared trainer façade cutover.
11. Complete Krea façade cutover.
12. Remove remaining arena imports of the legacy manager.
13. Remove cross-boundary private-state publication and reads.
14. Delete obsolete arena branches from the legacy manager.
15. Settle configuration and compatibility aliases.
16. Add direct-loader and fault-injection tests.
17. Run the core functional and process matrix.
18. Run compile and performance comparison.
19. Run available environment validation.
20. Close the extraction ticket only after the readiness checklist passes.

Do not combine legacy deletion with the façade cutover before proving there are no remaining callers.

---

# Readiness checklist

## Construction

* [ ] Architecture support is validated before canonical commit.
* [ ] Process ownership is acquired before destructive mutation.
* [ ] A second active arena runtime is rejected before mutation.
* [ ] Ranged checkpoint loading populates final arena destinations directly.
* [ ] Quantized-cache loading populates final arena destinations directly.
* [ ] Neither Krea direct path creates a second full canonical payload.
* [ ] Compatibility population remains available independently.
* [ ] Pre-commit failures leave model parameters unchanged.
* [ ] Pre-commit failures release all temporary resources.
* [ ] No automatic fallback path exists.

## Fatal lifecycle

* [ ] Canonical commit is the explicit destructive boundary.
* [ ] Post-commit setup failures raise `ArenaSetupFatalError`.
* [ ] The original setup exception is retained as the cause.
* [ ] Fatal setup failure overrides `--recover`.
* [ ] No later configured job starts after fatal setup failure.
* [ ] Failed committed transformers are marked disposed.
* [ ] Disposed transformers cannot execute or be prepared again.

## Teardown

* [ ] Preparation failure and runtime close use the same resource owner.
* [ ] Runtime close does not assume eager-model restoration.
* [ ] Runtime close is idempotent.
* [ ] Movement interception is restored or replaced by a disposed guard.
* [ ] Resident sidecars are released.
* [ ] Executor resources are released.
* [ ] Transfer tickets and timing events are drained or abandoned.
* [ ] Device ring state is released.
* [ ] Canonical pins and host storage are released.
* [ ] Process ownership is released.
* [ ] Sequential runtime acquisition succeeds after close.

## Boundaries

* [ ] `arena_offload` imports neither `manager` nor `manager_modules`.
* [ ] Shared trainer code imports no arena implementation internals.
* [ ] Krea code imports no arena construction or residency internals.
* [ ] Shared and Krea code read no arena-private `_mm_*` state.
* [ ] Legacy-manager compatibility state is no longer published.
* [ ] Arena-private metadata remains only where internally justified.
* [ ] The transformer exposes one generic runtime façade.
* [ ] Compile ownership is exclusive.

## Legacy behavior

* [ ] Legacy per-linear transformer offload remains available.
* [ ] Text-encoder offload remains on the legacy manager.
* [ ] Generic block compile remains unchanged for non-arena models.
* [ ] Arena-specific legacy-manager branches are removed.
* [ ] Unsupported architectures fail before commit.
* [ ] Unsupported architectures may recover to a later job.
* [ ] Configuration aliases are explicit and tested.

## Validation

* [ ] Pre-commit fault-injection tests pass.
* [ ] Post-commit fault-injection tests pass.
* [ ] Fatal process behavior tests pass.
* [ ] No pin, sidecar, ticket, timing-event, ring-slot, or owner leak remains.
* [ ] Training and sampling transitions pass.
* [ ] FP8 combinations pass.
* [ ] DOP or multiple-forward paths pass.
* [ ] Residency transitions pass.
* [ ] Compile ownership tests pass.
* [ ] Performance remains within the accepted baseline envelope.
* [ ] No repeated compile regression appears.
* [ ] No duplicate canonical payload appears.

---

# Definition of complete

The arena extraction is complete only when:

1. Both Krea direct loader paths populate final arena storage without a duplicate canonical payload.
2. Pre-commit preparation is transactional.
3. Post-commit setup failure is fail-closed and process-fatal.
4. Normal close releases all runtime-owned resources without pretending to restore an eager model.
5. A committed transformer is explicitly disposable after close.
6. One active arena runtime per process is enforced.
7. Sequential jobs can acquire a fresh runtime after successful release.
8. Shared trainer and Krea code use only the generic runtime façade.
9. The arena package has no dependency on the legacy manager.
10. Obsolete arena branches have been removed from the legacy manager.
11. Legacy per-linear and text-encoder behavior remains intact.
12. Core correctness, process, compile, memory, and performance validation passes.

Do not close the ticket based solely on phase commit titles or unit-test counts. Verify the final ownership boundaries and direct-loading behavior against the code.
