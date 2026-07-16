# torch.compile MegaCache contract

This document is the durable contract for persistent `torch.compile` caches in
ai-toolkit. The implementation is in `toolkit/compile_cache.py`; experimental
procedures and mutable status belong in `tasks/open/COMPILE_MEGA_CACHE_PLAN.md`
and git-bug ticket `ab208bf`.

## What MegaCache persists

`torch.compiler.save_cache_artifacts()` serializes the process's cumulative
AOTAutograd/Inductor/Triton cache inventory into one blob.
`torch.compiler.load_cache_artifacts()` restores that inventory in a fresh
process. It does not restore model weights, Arena residency, or runtime state.

The blob is a collection of individually keyed entries, not one blindly
trusted compiled model. Torch still evaluates the graph keys and guards for
every call. A missing or incompatible entry is a normal cache miss and may add
a new variant to the cumulative blob.

## Default lifecycle

Whenever model compilation is active, ordinary production and smoke paths use
a `CompileCacheSession` by default:

1. Construct or load the model and install adapters.
2. Derive final compile policy and shape hints.
3. Load the cumulative artifact before the first lazy compiled call.
4. Save after a sampling window or training step creates a new Dynamo frame.
5. Force a final best-effort save before runtime teardown.

The production default root is the owning process's shared output or training
root plus `.torch_compile_cache`; ordinary CUDA smokes use
`tmp/torch_compile_cache`. A configured `compile_cache_dir` overrides the
root. `compile_cache: false` and smoke `--no-compile-cache` are the explicit
opt-outs.

Cache persistence is an optimization, never a workload correctness gate.
Missing files, stale blobs, rejected artifacts, unsupported Torch versions,
and read/write failures warn and continue with an ordinary cold compile.
Strict cache probes may call the low-level artifact functions directly and
fail on a miss because the cache itself is what those probes test.

Dedicated cold/control/MegaCache matrix arms must use isolated Inductor cache
directories and explicit artifact paths. They do not load the ordinary default
cache, because that would contaminate the measurement.

## Cache identity

One model-level blob contains training, sampling, shape, checkpointing, and
adapter variants. The coarse filename identity includes:

- Torch version;
- model class, architecture, checkpoint identity, and qtype;
- compile mode, `fullgraph`, `dynamic`, dynamic hints, and explicit
  coordinate-descent policy;
- FP8 forward and grad-input execution gates;
- the stable Arena dispatcher generation.

Do not add resolution, batch shape, LoRA rank, checkpoint mode, or Arena
residency to this identity. Torch guards those variants inside the cumulative
blob. Splitting them into separate files reduces reuse without increasing
correctness.

Changing compile policy may deliberately select a different coarse blob.
Changing model code or a call shape can safely miss at Torch's finer-grained
keys. A custom Inductor pass is part of those keys: it must be a picklable
`CustomGraphPass`-shaped object with a deterministic content UUID. A closure or
process-random UUID poisons cross-process reuse.

## Arena and residency invariant

The cacheable seam is the pure functional block kernel owned by the generic
Arena dispatcher. Weight placement and H2D transfers occur eagerly around that
kernel. The compiled callable must not capture the residency plan, host arena,
transfer-ring slot, or WDDM budget.

Consequently, Mixed and Full residency must use the same compiled entries.
The five-process Z-Image FP8 dynamic benchmark
`Mixed -> Mixed -> Full -> Full -> Mixed` established this invariant:

| Run | Residency | Wall time | First compiled phase | Backend work |
| --- | --- | ---: | ---: | --- |
| cold | 3 resident / 27 streamed | 100.549 s | 56.023 s | 5 Inductor codegens, 101 Triton compiles |
| warm | Mixed -> Mixed | 47.538 s | 10.407 s | none |
| transition | Mixed -> Full | 45.110 s | 9.310 s | none |
| warm | Full -> Full | 43.823 s | 8.916 s | none |
| return | Full -> Mixed | 43.763 s | 9.247 s | none |

Every restored or cross-residency run had 3 serialized AOT hits, 5 FX hits,
zero Inductor codegen, zero Triton compilation, zero coordinate descent, zero
graph breaks, and numerical parity. The additional AOT miss is the measured
non-serialized inference lookup, not backend compilation. Evidence is stored
under
`output/zimage_megacache_residency_benchmark_20260716_v1/`.

Do not freeze or force production residency for cache reuse. The matrix freezes
residency only when isolating compiler behavior. To force streaming in a smoke,
use `layer_offloading_simulated_vram_gb`; do not distort the normal working
reserve.

## Prerequisites for a full hit

A useful loaded blob requires all of the following:

- the artifact is loaded before the first invocation of a lazy compiled
  wrapper;
- the compile boundary is a graph-clean functional block kernel;
- `fullgraph=True` has no wrapper, transfer, scalar-extraction, or mutation
  graph breaks inside that kernel;
- FP8 tensor subclasses and custom autograd functions expose stable compile
  identities and version hashes;
- custom Inductor passes have deterministic UUIDs;
- compile policy and guarded call structure match an entry in the blob;
- the relevant AOT, FX, autotune, and Triton entries were saved after their
  first cold compile.

Model weights, physical residency, and transfer scheduling are not
prerequisites. A cache load by itself is not evidence of a hit. Require cache
counters plus zero backend codegen/Triton/autotune work and numerical parity.

Representation support is evidence-based. The current Quanto `QBytesTensor`
emits Torch's warning that it does not implement `_stable_hash_for_caching`, so
an artifact can load while its AOT entries still miss. Do not suppress that
warning or claim a full hit for Quanto until the subclass supplies a stable
hash and the strict matrix proves it. The full-model acceptance above is for
the validated TorchAO FP8 path.

## Regression rules

- Never move Arena transfers or residency policy into the compiled block
  callable.
- Never key the coarse cache by Arena residency or simulated-card size.
- Never install an anonymous custom Inductor pass closure.
- Never make a production cache miss fatal.
- Never run a cold-cache acceptance arm with the ordinary shared cache active.
- Keep train and sample variants cumulative in one model-level artifact.
- When changing the dispatcher ABI, FP8 compile identity, or compile policy,
  run the focused cache-key/session tests and the strict full-model matrix.

The full-model acceptance harness is
`scripts/run_full_model_megacache_matrix.py`; the residency-evolution benchmark
is `scripts/bench_full_model_megacache_residency.py`.
