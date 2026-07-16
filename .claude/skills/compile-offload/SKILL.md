---
name: compile-offload
description: Principles for torch.compile interacting with weight streaming/offload in this repo - graph breaks, per-block compile gates, FP8-under-compile, in-graph streaming. Load when touching compile gates, chasing graph breaks or recompiles, editing the in-graph streaming seam, or when sampling/step times jump after a phase change.
---

# torch.compile + offload streaming

This is the fastest-moving area of the repo. The principles below have held
across several rewrites; for current mechanism ALWAYS check the code
(`toolkit/memory_management/arena_offload/`, `immutable_runtime.py`,
`manager_modules.py`), the open plans in `tasks/open/`
(`ARENA_FULLGRAPH_COMPILE_PLAN.md`, `COMPILE_MEGA_CACHE_PLAN.md`), and their
git-bug tickets before relying on details. The in-graph streaming and
immutable-transfer-arena plans are shipped history under `tasks/done/`.

## Principles that keep proving true

- **FP8 is compile-compatible; wrappers are what break it.** The 370s->10s
  Krea2 sampler win came from killing graph breaks in the streaming forward
  wrapper, not from avoiding FP8. When compile is slow, hunt graph breaks
  before blaming quantization.
- **Compile at block granularity, gate per block.** Whole-model compile and
  per-Linear compile both lose; block-granular offload with a per-block
  compile gate is the shape that works here.
- **Residency is per-Linear, not all-or-nothing per block** in the in-graph
  trunk. The resident VRAM of a block's weights is negligible; the real cost
  is the >=2-block transfer ring.
- **Arena residency is outside the compiled kernel.** The generic dispatcher's
  pure functional block kernel is invariant across Mixed and Full residency;
  eager transfer and placement changes must not rebuild or re-key it. Legacy
  per-Linear wrappers may still invalidate compile state when they resize.
- **The historical bottleneck was the pageable submit stall (130s+), not
  PCIe bandwidth.** Fixes go at the submit path (pinned bounce pool, batched
  worker fills), not at transfer sizes.
- **For in-graph sampling, fold the LoRA into the streamed weights instead
  of rendering the base model separately** (see commit 56eabd5).
- **`mode="max-autotune"` is unusable with streamed weights** -- it enables
  CUDA graphs, which capture fixed device pointers, and the whole premise of
  the arena is that block weights move (the ring recycles slots, the compact
  buffer is refilled per block). Only `max-autotune-no-cudagraphs`, or the
  narrower `torch._inductor.config.max_autotune_gemm`, are on the table; the
  latter is the one worth trying (it lets Inductor template the GEMM so the
  quant/dequant pointwise kernels can fold into its epilogue). Untested.
- **Dynamo is indifferent to pinnedness and host-flat identity** (measured:
  swapping a same-shaped host flat, even via a fresh closure per boundary,
  causes zero recompiles). Pinnedness gates in compile paths are our policy
  code, not a compiler requirement. Diagnose recompile causes with
  `TORCH_LOGS=recompiles,guards`; never design around a guessed guard.
- **Bound the dynamic sequence dim; measure recompiles with `new_frames`.**
  Bounds are auto-derived (`toolkit/compile_shape_bounds.py`) from dataset
  buckets + sample resolutions + the model's `SequenceLayout`. Sampling shares
  the compiled block kernels, so its resolutions belong in the bounds too.
- Arena/offload behavior stays off-by-default for upstream separability.
  MegaCache persistence is different: on this fork it defaults on whenever
  compile is active, is best-effort, and has the explicit
  `compile_cache: false` opt-out.

## MegaCache invariants

- Load the cumulative artifact after final compile policy and adapter topology
  are known but before the first lazy compiled invocation. Save after new
  variants and once before teardown.
- A successful blob load is not a hit. Require AOT/FX hit counters, zero
  Inductor codegen/Triton compilation/autotune work, zero graph breaks, and
  numerical parity.
- One model-level artifact contains train, sample, shape, adapter, and
  checkpoint variants. Torch guards entries individually. Do not fragment the
  coarse key by resolution, LoRA rank, or Arena residency.
- Cache identity includes compiler policy, FP8 execution gates, Torch version,
  and stable dispatcher generation. Any custom Inductor pass must be picklable
  and expose a deterministic content UUID; an anonymous closure poisons
  cross-process keys.
- A miss or persistence error is never fatal in production or ordinary smokes.
  Dedicated cache acceptance arms use isolated explicit artifacts and may be
  strict.
- The measured Mixed -> Mixed -> Full -> Full -> Mixed matrix reused every
  serialized variant without backend compilation. See
  `docs/decisions/MEGACACHE.md` before changing the compile seam or cache key.

## Diagnostics that work

- Recompile storms: `TORCH_LOGS=recompiles` on a small synthetic script (see
  the `cuda-testing` skill), never on a full run.
- Verify the streamed path is actually taken with call counters
  (`_scaled_mm`, H2D copy counts), not by eyeballing step time.
- First-iteration transfer/prefetch warmup after a phase boundary is expected.
  Backend compilation is not expected when the required MegaCache variants
  were restored; check counters rather than attributing all latency to warmup.

## Current state (verify before trusting -- this section rots fastest)

Two offload runtimes coexist:

- **Generic arena dispatcher** (`toolkit/memory_management/arena_offload/`,
  sampling through `immutable_runtime.py`) -- the primary runtime and the
  basis of the upstream extraction (ticket `553ffec`): canonical host arena,
  no runtime Parameter repointing, manager-owned residency, eager transfer
  around functional block kernels, and strict per-block `fullgraph=True`.
- **Legacy MemoryManager** per-Linear streaming (`manager.py` /
  `manager_modules.py`) remains a fallback runtime. Making its wrappers
  fullgraph-compatible is not an active requirement.

Compile-cache persistence across process restarts (MegaCache) is shared by
production training/sampling and ordinary CUDA smokes. Strict full-model and
residency-transition evidence is recorded in `docs/decisions/MEGACACHE.md`;
implementation follow-up remains in `tasks/open/COMPILE_MEGA_CACHE_PLAN.md`
and ticket `ab208bf`.
