# True Block Streaming Plan

## Goal

Make whole-transformer-block streaming a first-class option so the offload path
moves **block-sized units** instead of per-Linear units. The objective is to cut
CPU cost from the many small requests the streaming machinery issues per step,
not to save resident VRAM.

This builds on `layer_offloading_block_stream_only` (residency: non-block layers
stay resident, only block Linears stream). That flag alone does **not** reduce
the request count — the block Linears still stream one-at-a-time. This plan adds
the coalescing that actually delivers the CPU win.

## Where the per-step requests come from (Krea2, 28 blocks × 8 Linears)

Two independent per-Linear costs, ~224 each per forward (plus backward):

1. **Bounce-pool worker fills** (`bounce_pool.py::_worker_loop`).
   One `pageable -> pinned` fill per Linear: a lock cycle, slot dict insert,
   leaf alloc/reuse, `copy_`, CV notify, publish lock cycle. ~224 worker
   iterations/forward.

2. **GPU ring stage** (`manager_modules.py::_stage_forward_weight`).
   One `pool.acquire` + one H2D (`.to(device)` of qdata+scale) + one dequant
   kernel per Linear, each managing `fwd_slot_ready/free` events. ~224 H2D
   submits/forward.

A Krea2 block ≈ 434M params ≈ 0.85 GB bf16 (0.43 GB as fp8 qdata). The current
ring is `max(2, PIPELINE_DEPTH=4)` per-Linear slots (~0.8 GB bf16). A 2-block
coalesced ring is ~1.7 GB bf16, or ~0.9 GB if staged as fp8 with dequant-in-place.

## Invariant

Coalescing changes the **transfer/scheduling unit**, never autograd math. Each
Linear's forward still computes `F.linear(x, dequant(w_i), b_i)` on its own
slice. No quantized-wrapper repacking across Linears (too fragile across TorchAO
versions): a "block buffer" is a *list of per-Linear leaves filled/published
together*, not one contiguous reinterpreted tensor.

## Slice 1 (this change, CPU-verifiable): block-granular worker fills

`PinnedBouncePool` gains `fill_group_size` (default 1 = current behavior). The
worker claims up to `fill_group_size` schedulable positions under **one** lock,
copies them outside the lock, and publishes them under **one** lock. This
amortizes the per-Linear lock/CV/slot-dict overhead ~G×, turning ~224 worker
lock-cycles/forward into ~28 when `G` = a block's Linear count.

- Env override: `AI_TOOLKIT_BOUNCE_FILL_GROUP`.
- Wired from the manager: when `block_stream_only` is on, set
  `fill_group_size` to the per-block Linear count of the streamed blocks.
- `acquire`/`_align_locked`/resync/`_reclaim_locked` contracts are unchanged —
  only the worker's claim/publish batching changes. Per-slot epoch/skip discard
  at publish is preserved per position.

This is unit-testable without CUDA (the pool degrades to plain tensors), so it
ships with tests now.

## Slice 2 (next, GPU-verified): per-block GPU ring stage

Coalesce the H2D + event management to block granularity:

- Ring slot count measured in **blocks**: depth = `2` blocks (execute +
  prefetch) instead of `4` Linears. `_get_device_state` grows `w_buffers` etc.
  to hold a block's worth of per-Linear materialized weights per slot.
- `_stage_forward_block`: one `ts.wait_event(free)` / `fwd_slot_ready.record()`
  pair per block; issue the block's Linear H2D+dequant back-to-back inside it.
  Cuts event-management and slot bookkeeping ~G×; lets compute overlap a full
  block ahead with the 2-block ring.
- Staging dtype: stage fp8 qdata and dequant-in-place per Linear (ring ~0.9 GB)
  rather than bf16 (~1.7 GB). The 2-block ring floor is the real VRAM tradeoff
  and competes with the headroom band on a 12 GB card.
- Backward reuse (`_stage_backward_weight`) already searches the ring
  newest-to-oldest; extend its reuse window to the block slots.

Requires the venv GPU methodology (synthetic model, `_scaled_mm`/H2D counters,
`synchronize` before timing). Gated behind the same flag.

## Measurement protocol (decides whether Slice 2 is worth it)

A/B with the flag, on a real Krea2 step, capture from `BouncePool.report()` and
the perf log:

- worker copy submit count and `copy_s` wall (Slice 1 target: count ↓ ~G×).
- H2D submit count and `submit_s` (Slice 2 target: count ↓ ~G×).
- step time and `device_free` band (Slice 2 must not breach the WDDM stop-line
  with the larger ring).

If Slice 1 alone moves step time, Slice 2's ring-VRAM cost may not be worth it on
12 GB cards. If the H2D submit count dominates, Slice 2 is the lever.

## Risks

- Worker batch must preserve per-slot epoch/skip discard (step rollover mid-batch).
- Larger ring (Slice 2) raises the resident floor — must stay under the spill
  guard; keep it fp8-staged.
- Block size is model-specific; derive Linear-per-block count from the smart
  plan's block grouping, fall back to `fill_group_size=1`.
