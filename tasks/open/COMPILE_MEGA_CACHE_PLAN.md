# torch.compile MegaCache - implementation plan

> **git-bug:** `ab208bf`. Mutable execution status and new run results belong
> in the ticket. The stable contract and measured invariants are in
> `../../docs/decisions/MEGACACHE.md`.

## Goal

Make persistent `torch.compile` artifacts an automatic optimization for every
compiled production and ordinary smoke path, while preserving an explicit
opt-out and strict isolated modes for cache experiments.

MegaCache must shorten fresh-process startup without changing numerical
results, Arena residency, transfer accounting, or failure behavior. A cache
miss must compile normally rather than fail a production workload.

## Architecture

`toolkit/compile_cache.py` owns one generic `CompileCacheSession`:

- one cumulative model-level artifact for training and sampling variants;
- load once before the first lazy compiled invocation;
- save after a new compile frame and once before teardown;
- atomic artifact replacement;
- nonfatal production load/save behavior;
- a stable coarse key based on model/compiler identity, with Torch guards
  retaining responsibility for shapes and other call variants.

The owning process supplies a shared default root. Training uses
`<training_folder>/.torch_compile_cache`; standalone generators use their
output root; ordinary smokes use `tmp/torch_compile_cache`. Configured
`compile_cache_dir` overrides that root. `compile_cache: false` and
`--no-compile-cache` are the opt-outs.

Dedicated cache probes and matrix runners keep using explicit artifact paths
and isolated Inductor directories. Their cold/control arms must never enter the
ordinary default session.

## Compile boundary prerequisite

The active target is the generic Arena saved-forward dispatcher. Its compiled
callable is the pure functional block kernel. Residency choice, staging,
transfer submission, ring-slot ownership, and WDDM budgeting stay eager and
outside the graph.

That separation is the prerequisite for cache reuse across residency. Do not
put residency or simulated-card size in the cache key. The measured
`Mixed -> Mixed -> Full -> Full -> Mixed` benchmark reused all serialized
variants with zero backend compiler work across every transition.

Other prerequisites are stable FP8/custom-autograd compile identities,
deterministic UUIDs for custom Inductor passes, load-before-first-call ordering,
and a previously saved inventory containing the used AOT/FX/Triton variants.

## Product integration

The shared session belongs at process lifecycle boundaries, not inside a model
implementation:

- `BaseSDTrainProcess`: load after final compile policy and topology setup;
  checkpoint after samples and training steps; force-save before Arena close.
- `GenerateProcess`: load before wrapping/invoking the model and save after
  generation.
- captioner and reference-generator compile paths: the same best-effort
  lifecycle.
- Krea2: no model-specific artifact ownership. Compatibility key helpers may
  delegate to the generic identity while callers migrate.
- compile-capable smokes: default shared session plus explicit opt-out.

The UI exposes the opt-out but does not require users to choose a cache path.
Environment variables are not part of production behavior.

## Acceptance

Focused unit tests must establish:

- default-on only when compilation is active;
- explicit opt-out;
- compile-policy fields partition the coarse key;
- residency fields do not partition it;
- missing/rejected/unwritable artifacts are nonfatal;
- a session skips redundant saves until a new compile frame appears.

Strict CUDA acceptance must establish:

- `fullgraph=True`, FP8, dynamic block compilation has zero graph breaks;
- restored processes hit every serialized AOT/FX entry and perform zero
  Inductor codegen, Triton compile, coordinate descent, or unexpected autotune;
- cold, warm, Mixed, Full, and residency-transition runs have numerical parity
  and exact Arena transfer accounting;
- the ordinary default cache is absent from cold/control arms.

The full-model runner is `scripts/run_full_model_megacache_matrix.py`; the
residency benchmark is `scripts/bench_full_model_megacache_residency.py`.

## Upstream extraction

MegaCache is small relative to the Arena PR but is not a zero-diff transplant.
The portable slice is the artifact helper/session, stable key inputs, trainer
load/save hooks, focused tests, and low-noise logging. Fork-specific default
paths, every ancillary smoke integration, and broad UI changes stay local.

The upstream PR remains off by default. During extraction, use an explicit
upstream cache flag/directory unless the maintainer accepts default-on behavior
for already-compiled jobs. Gate unsupported Torch builds cleanly. Keep the
Windows Triton bundle recovery isolated so it can be reviewed or omitted
without changing the generic lifecycle.

Estimated effort after the Arena compile seam is present:

- 1 day to extract and version-gate the generic session and stable identity;
- 1 day for upstream trainer/Krea2 lifecycle integration and removal of the
  model-specific cache owner;
- 1 day for focused tests, docs, and a cold/warm acceptance run;
- 0-2 days for Torch-version compatibility, Windows recovery review, and
  maintainer-requested config changes.

Total: roughly 3-5 focused engineering days. The recommendation is to include
it as a reviewable commit group in the unified Arena PR because the measured
cold compile is large and the runtime dependency is narrow. If review scope is
the concern, the generic session can be omitted without weakening Arena
correctness and submitted immediately afterward.
