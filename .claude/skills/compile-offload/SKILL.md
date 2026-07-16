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
- **Every resize destroys prefetch and compile state.** A controller that
  keeps adjusting ring/reserve sizes forfeits the compile win. Converge to a
  stable band, then stop moving.
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
- Offload/compile changes must stay off-by-default and behavior-preserving
  when their flags are off (upstream-PR separability,
  `docs/decisions/UPSTREAM_PR_PLAN.md`).

## Diagnostics that work

- Recompile storms: `TORCH_LOGS=recompiles` on a small synthetic script (see
  the `cuda-testing` skill), never on a full run.
- Verify the streamed path is actually taken with call counters
  (`_scaled_mm`, H2D copy counts), not by eyeballing step time.
- First-iteration cost after any phase boundary (train -> sample -> train)
  is expected: compile cache + prefetch warmup. Only flag it if it repeats
  every iteration.

## Current state (verify before trusting -- this section rots fastest)

Two offload runtimes coexist:

- **Generic arena dispatcher** (`toolkit/memory_management/arena_offload/`,
  sampling through `immutable_runtime.py`) -- the primary runtime and the
  basis of the upstream extraction (ticket `553ffec`): canonical host arena,
  no runtime Parameter repointing, manager-owned residency, and eager transfer
  around functional block kernels. Strict per-block `fullgraph=True` is the
  active compile target (`tasks/open/ARENA_FULLGRAPH_COMPILE_PLAN.md`, ticket
  `c4e29f1`).
- **Legacy MemoryManager** per-Linear streaming (`manager.py` /
  `manager_modules.py`) remains a fallback runtime. Making its wrappers
  fullgraph-compatible is not an active requirement.

Compile-cache persistence across process restarts (Mega-Cache) is shipped for
sampling; strict arena-block and full-model restoration are open
(`tasks/open/COMPILE_MEGA_CACHE_PLAN.md`, ticket `ab208bf`).
