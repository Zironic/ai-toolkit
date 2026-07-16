# Upstream Arena Extraction Plan

> **Published 2026-07-16:** The extracted arena implementation is upstream PR
> [#948](https://github.com/ostris/ai-toolkit/pull/948). Ticket `553ffec` is
> closed with the PR URL recorded.

> Strategy: `docs/decisions/UPSTREAM_PR_PLAN.md`
>
> Runtime design: `tasks/done/GENERIC_BLOCK_DISPATCHER_PLAN.md`
>
> Historical status belongs on git-bug ticket `553ffec`; this document contains the
> extraction design and acceptance boundary.

## Outcome

Extract the measured arena-offload implementation into reviewable upstream
changes without reintroducing the retired architecture-adapter execution path.

The first feature claim is deliberately narrow:

> Krea2 uses an optional generic saved-forward block dispatcher with canonical
> arena storage, WDDM-safe residency, model-owned checkpointing, and ordinary
> installed adapter forwards.

The first extraction does not claim arbitrary transformer support. A second
production architecture is a deferred breadth proof.

## Demonstrated boundary

The fork has established these seams:

```text
ordinary Krea2 transformer forward
    -> ordinary model-owned checkpoint loop
        -> selected ordinary block call
            -> generic saved-forward dispatcher
                -> resident or streamed declared state
                -> torch.func.functional_call(saved installed forward)
```

Ownership is fixed:

- model: forward semantics, block order, checkpointing, argument structure;
- quantization: physical tensors, reconstruction, numerical kernels;
- arena: canonical storage, residency, transfer, lifecycle;
- shared trainer: public lifecycle contexts only.

The extraction must not contain `ArchitectureAdapter`,
`SingleStreamMMDiTAdapter`, `forward_streamed`, production `run_blocks()`,
handwritten Krea leaf paths, or runtime-owned checkpoint trunks.

## Work stages

> Packaging (decided 2026-07-14, see the strategy doc): everything below ships
> as **one unified PR**. Stages 1, 3a, and 3b become ordered commit groups
> inside it, not separate PRs; Stage 2 stays in-fork preparation. Splitting is
> a fallback only if the maintainer asks.

### Stage 1 - Host-memory safety

The backend-independent Windows/WDDM safety layer, first commit group:

- NVML physical-free sensing;
- DXGI non-local pinned-memory budget sensing;
- centralized pin authority and explicit release accounting;
- allocator hard-cap behavior that turns silent WDDM paging into a local OOM.

This is independently useful to the existing upstream offloader and is a hard
dependency of arena memory safety.

### Stage 2 - In-fork extraction cleanup

Keep this as a local preparation stage, not a PR:

- isolate local workflow and diagnostics from upstream payload;
- preserve the public `arena_offload` integration surface;
- verify arena code does not import the legacy `MemoryManager` backend;
- verify the legacy manager does not depend on arena execution internals.

### Stage 3a - Generic arena and dispatcher core

Extract:

- canonical host arena and transactional construction;
- generic repeated-block discovery and complete state accounting;
- quantization-owned storage and substitution declarations;
- device sidecar residency plans and transfer plans;
- saved-forward dispatcher and block-granular compile boundary;
- runtime resource ownership, fail-closed setup, and teardown;
- WDDM-safe cold planner and measured diagnostics.

Stage 3a must not import Krea2 code or contain model-specific execution math.

### Stage 3b - Krea2 consumer

Extract the smallest Krea2 integration that:

- opts into arena offload through model config;
- enables model-owned gradient checkpointing before arena commit;
- supplies a declarative block-container name only when discovery needs it;
- supports direct ranged/checkpoint-cache population into canonical storage;
- enters public training and sampling lifecycle contexts;
- preserves the ordinary Krea2 transformer and block forwards.

Krea2 does not provide an arena execution adapter.

### Deferred - Second architecture

Do not block Stages 1-3 on Ideogram4, Z-Image, or another model. After the
Krea2-scoped change is stable, select one production architecture and prove it
uses the same discovery, storage, dispatcher, and lifecycle path. Generalize
only around differences observed in that integration.

## Upstream exclusions

Keep out of the first extraction:

- local job paths, datasets, and cached prompt workflow;
- benchmark artifacts and heavy diagnostic scripts;
- speculative multi-architecture compatibility machinery;
- the legacy per-linear manager rewrite;
- Orbit or ConvRot product claims beyond separately supplied quantization
  support;
- UI changes unrelated to selecting already-supported Krea2 flags.

## Validation per stage

Use the narrowest evidence that can reject each patch:

1. Host-memory tests and focused Windows probes for Stage 1.
2. Import-boundary and lifecycle tests for Stage 2.
3. Generic dispatcher CPU/synthetic CUDA tests for Stage 3a.
4. One deterministic Krea saved-forward oracle and one production-shaped
   train/sample/train smoke for Stage 3b.
5. One correctness-only whole-model compile observation; no compile benchmark.

Do not require a full test suite or full training job unless an upstream gate
specifically calls for it.

## Stage 3 acceptance

- ordinary Krea2 forward and checkpoint ownership remain intact;
- canonical payload accounting reconciles resident plus streamed bytes;
- planned and observed H2D bytes agree in the acceptance smoke;
- selected block output, input gradients, and adapter gradients match the
  deterministic resident oracle within the documented backend tolerance;
- compiled block execution works without tracing arena scheduling;
- teardown drops runtime execution references and pinned weight accounting;
- no retired adapter execution symbol remains in production code;
- arena behavior is off by default and explicit selection fails loudly when
  unsupported.

## Review packaging

Before producing the PR:

1. diff against upstream `main` rather than against the dirty fork;
2. list every modified existing upstream file and justify it;
3. keep the commit groups clean: safety foundation, arena/dispatcher core,
   Krea2 integration, and acceptance evidence as separable commits so the
   maintainer can review (or, on request, split) along those seams;
4. include maintainer-runnable commands and expected evidence fields;
5. record exclusions and the deferred second-model proof in the PR body.
