# Compile-Neutral Generic Transformer Runtime and Krea2 Extraction

> **Partially superseded (2026-07-14):** the architecture-adapter *execution*
> design in this document (adapter protocol, runtime-owned block loop/trunks,
> runtime-owned checkpointing, `run_blocks()` delegation) is superseded by
> `GENERIC_BLOCK_DISPATCHER_PLAN.md` (ticket `b7dead1`). The adapter path
> stays only as a comparison oracle until that plan's Phases 5-6 pass, then
> is deleted in its Phase 7. Canonical-arena, residency, and lifecycle
> content here remains valid.

Source: `combined_compile_neutral_krea2_refactor_plan.md`, supplied by the user on 2026-07-11.

## Outcome

Refactor Krea2 so memory management, streamed execution, residency policy,
checkpoint orchestration, and compile orchestration belong to a generic
transformer runtime. Krea2 should select the single-stream MMDiT architecture
adapter and attach the runtime; MMDiT should retain only a neutral block-loop
delegation point.

At the same time, make immutable residency compile-neutral. Changing which
canonical transformer blocks are GPU-resident must update runtime source data,
not rebuild execution programs. Compiled block kernels must keep one explicit
tensor ABI whether weights come from resident GPU sidecars or streamed host
storage.

Target integration:

```python
transformer = load_krea_transformer(...)
prepare_transformer_runtime(
    transformer,
    model_config=self.model_config,
    architecture=SINGLE_STREAM_MMDIT,
)
```

Target generation seam:

```python
with transformer._execution_runtime.sampling(gen_config):
    image = pipeline(...)[0]
```

## Constraints and invariants

- Start by merging the changes from `GPT` into `faster-dop`, resolving and
  validating the merge before beginning the refactor.
- After that merge, perform all implementation work directly on `faster-dop`.
  Do not force-push the branch.
- Treat the preliminary immutable residency source-table work brought in from
  `GPT` as a starting point to inspect and revise, not a fixed design.
- Preserve stable identities across residency transitions for compiled block
  kernels, eager block wrappers, wrapper sequences, train/sample trunks,
  checkpoint structure, and block order.
- Compile only pure block math. Keep residency/source lookup, transfer
  orchestration, sidecar resolution, checkpoint orchestration, leases, and
  controller decisions eager.
- Residency generation, source snapshots, transfer plans, closures, and plan
  fingerprints must not become compile keys or speculative guards.
- Residency transitions operate on complete canonical transformer blocks.
  Protected blocks, singleton modules, and trainable adapters remain resident.
- Canonical frozen parameters remain arena-backed CPU views; GPU residency is
  held in replaceable sidecars.
- Training and sampling hold exclusive execution leases. Publication and close
  reject active execution, and failure cleanup occurs only after the active
  Python/autograd stack unwinds.
- Preserve current WDDM safety, byte-driven promotion/demotion, trace
  durability, fallback behavior, and config/UI wiring rules.
- Do not add complex rollback for impossible internal failures; validate
  predictable errors before reconciliation and fail loudly on unexpected
  post-reconciliation failures.

## Architecture

### Generic execution adapter

Introduce a minimal architecture adapter describing canonical block discovery,
block keys/order, block invocation, protected blocks and singleton modules,
model-specific input normalization, and sampling working-set estimation where
generic `ModelConfig` data is insufficient. The initial adapter is
single-stream MMDiT, but runtime policy must not depend on Krea2.

### Attached transformer runtime

Attach one generic runtime to the transformer. It owns:

- immutable source-table bootstrap and publication;
- whole-block residency planning and GPU sidecars;
- one permanent wrapper per block per mode;
- permanent train and sample trunks;
- explicit execution leases and sampling context;
- checkpoint orchestration;
- compile cache and compile diagnostics;
- controller-facing promotion/demotion and training relief;
- teardown and restoration of legacy behavior.

The source table has a stable structural ABI and atomically published source
snapshots. Publication may replace snapshots and sidecar tensors, but never
execution topology. Readers observe one complete snapshot per execution.

### Permanent execution programs

Permanent eager wrappers resolve the current source, start/wait/free streamed
fetches when needed, and call a compiled pure-math block kernel through the same
explicit tensor ABI. Checkpoint choice and structure are fixed for a prepared
mode. New input shapes may compile; a warmed `(mode, block, input signature)`
must not compile again after a residency change.

### Whole-block residency

Convert smart plans to canonical block keys, validate mappings before mutation,
fit/demote/promote only complete blocks, preserve protected blocks and
singletons, and publish a new source snapshot atomically after sidecar
reconciliation. Demotion must make sidecar memory reclaimable without D2H
copies or canonical arena changes.

### Integration and cleanup

- Add a dedicated immutable runtime attachment path to `MemoryManager`, with
  exclusive backend selection, compatible `.to()` interception, and teardown.
- Reduce MMDiT to one neutral `_run_blocks` delegation point and mathematical
  behavior; remove compile caches, residency programs, transfer policy, hook
  stripping, generic LoRA compatibility, and controller state.
- Reduce Krea2 to model loading, adapter selection, runtime preparation, and
  genuinely model-specific sampling estimation inputs.
- Route training pressure relief, promotion, and sampling transitions through
  the runtime without explicitly resetting durable trace state.
- Preserve frozen base weights and trainable LoRA behavior under resident,
  streamed, and mixed whole-block execution.

## Implementation sequence

1. Merge `GPT` into `faster-dop`, resolve conflicts without discarding either
   branch's relevant work, and run focused baseline validation on the merged
   tree.
2. Add the generic immutable source runtime and source-table publication tests.
3. Convert residency planning and reconciliation to validated whole-block
   transitions.
4. Add the minimal single-stream MMDiT architecture adapter.
5. Add the attached transformer runtime with permanent wrappers/trunks and
   lifecycle management.
6. Add the neutral MMDiT block-loop delegation seam.
7. Replace Krea2's immutable attachment with generic runtime preparation.
8. Add the training execution lease, including exception and OOM cleanup.
9. Add the generic sampling context and sampling exception recovery.
10. Route controller transitions and training relief through the runtime.
11. Delete superseded Krea2/MMDiT runtime and compile-policy state.
12. Add compile-neutrality, identity, lifecycle, failure-path, numerical, and
    whole-block CUDA tests.
13. Remove migration scaffolding, update documentation, and record final
    diagnostics.

Keep commits reviewable and preserve a working fallback during migration. Do
not retain parallel legacy abstractions after their consumers have moved.

## Validation

Focused coverage must establish:

- structural source ABI and atomic publication;
- block-key validation and whole-block fitting;
- one runtime/trunk/wrapper structure per transformer and mode;
- stable program identities through several residency transitions;
- zero new backend compiles for warmed signatures after transitions;
- new shapes remain allowed to compile;
- publication/close rejection during active training or sampling;
- exception and OOM lease release after stack unwind;
- trace state is not explicitly reset by residency publication;
- resident, streamed, and mixed outputs and LoRA gradients match;
- base weights stay frozen and fetch tickets are released;
- repeated train/sample transitions remain stable;
- WDDM headroom and reclaimable sidecar behavior remain intact.

Run the focused immutable lifecycle tests, all new runtime/lease/
compile-neutrality/whole-block tests, Ruff on changed Python files, and relevant
memory-management and Krea2 groups. Use
`scripts/smoke_krea2_train_cuda.py` for training and
`scripts/smoke_krea2_inference_cuda.py` for sampling. Run CUDA smoke tests in isolated processes
when known WDDM or process-global test leakage applies. Per repository policy,
confirm known order leaks with the file-alone and suite-with-file-omitted runs,
record them on ticket `f2aceba`, and do not investigate further.

## Completion criteria

The refactor is complete when Krea2 and MMDiT no longer own generic memory or
compile policy; one attached generic runtime owns stable execution programs and
mutable source snapshots; residency changes are whole-block, lease-safe, and
compile-neutral; existing memory safety and trace behavior are preserved; and
the numerical, gradient, lifecycle, compile-count, and transition tests pass.

The final report must list commits and files, removed legacy runtime code,
focused and broader test results, CUDA smoke results, compile counts before and
after transitions, VRAM behavior during promotion/demotion, remaining known
limitations, and unrelated existing failures.

## Related work

- `628b0cb` - immutable transfer arena and sidecar residency refactor; provides
  the current implementation baseline and preliminary source/runtime work.
- `ca8f496` - model-agnostic memory/compile extraction; overlaps the generic
  ownership boundary and should be reconciled rather than duplicated.
- `3ca8a7b` - in-graph weight streaming; preserve its measured execution and
  compile behavior where still applicable.
- `f2aceba` - canonical ticket for known test-order leakage.
