---
name: compile-offload
description: Principles for torch.compile interacting with weight streaming/offload in this repo - graph breaks, per-block compile gates, FP8-under-compile, in-graph streaming. Load when touching compile gates, chasing graph breaks or recompiles, editing the in-graph streaming seam, or when sampling/step times jump after a phase change.
---

# torch.compile + offload streaming

This is the fastest-moving area of the repo. The principles below have held
across several rewrites; for current mechanism ALWAYS check
`tasks/open/INGRAPH_STREAM_PLAN.md`, git-bug ticket `3ca8a7b`, and the code
(`toolkit/memory_management/manager_modules.py`, the Krea2 integration)
before relying on details.

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

## Known state (verify before trusting -- this section rots fastest)

As of 2026-07-10: in-graph weight streaming Phases 0-3 done (all-28
fully-streamed smoke passed). The pin-assumption measurement campaign
(`scripts/bench_pin_assumptions.py`) killed two beliefs: registration of
RAM-resident pages is ms-scale (~150 GiB/s, NOT 0.6-2 GB/s), and
**Dynamo is indifferent to pinnedness and host-flat identity** -- swapping
a same-shaped host flat, even via a fresh closure per boundary, causes zero
recompiles, so the production sampling-boundary recompiles have an
undiagnosed guard cause (task I2 in the plan; diagnose with
`TORCH_LOGS=recompiles,guards` before designing around it). The settled
direction is the canonical host arena + manager-owned GPU sidecar
residency plan in `tasks/open/IMMUTABLE_TRANSFER_ARENA_PLAN.md` (ticket
`628b0cb`): one-time Parameter canonicalization, no runtime repointing,
per-Linear residency via static multi-range compact transfers (Python
submission, ~11 us/copy), compile keyed by residency/layout fingerprint.
