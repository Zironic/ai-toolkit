# In-Graph Streaming Phase 4a — full-checkpoint streamed training compile

> Execution plan for `INGRAPH_STREAM_PLAN.md` Phase 4 Rung 2 (Rung 1 was
> de-scoped to synthetic scale — see the parent plan; its three risk items
> are retired by `tests/test_training_compile_synthetic.py`). Status lives
> in git-bug ticket 3ca8a7b. Production target: keep_last=0, all 28 blocks
> streamed, LoRA training, 12 GB card.

## Goal

One compiled trunk (fwd+bwd, zero breaks) for training: each streamed
block is `checkpoint(block_fn, ..., use_reentrant=False)` where
`block_fn` contains the fetch ops and the leaves-passing block call.
Backward re-fetch falls out of plain checkpoint recompute. Measured in
`scripts/smoke_krea2_train_cuda.py` against the 3.97 s/step eager
baseline (512px, batch 1).

## The central correctness problem: the free anchor in backward

Sampling's `fetch_free_after(token, out)` anchors buffer reuse on the
block's FORWARD output — correct when the forward is the last reader.
In training recompute it is wrong: `_Fp8LinearTrainingFn.backward`
computes grad-input from the fetched `qdata_t`/`scale` views AFTER that
anchor. Backward-order trace with a depth-2 ring:

    rc27(fetch A, freeEvt A) -> bwd27(reads A!) -> rc26(fetch B)
    -> bwd26 -> rc25(fetch reuses slot A, H2D waits freeEvt A only)

`freeEvt A` fired at the end of rc27, so rc25's H2D can overwrite slot A
while bwd27's grad-input kernels still read it — silent numeric
corruption, the exact Phase 0 trap resurfacing across the fwd/bwd
boundary.

**Design (DECIDED): grad-mode-selected free op.**
`torch.ops.mm.fetch_free_bwd(token, out)` — forward is identity on
`out`; its registered autograd backward records the ticket's free event
(on the current/compute stream) and passes grad through unchanged. The
block fn selects at trace time:

    if torch.is_grad_enabled():          # trace-time constant per region
        out = torch.ops.mm.fetch_free_bwd(token, out)
    else:
        torch.ops.mm.fetch_free_after(token, out)

- Checkpoint's FIRST pass runs under no-grad -> traces the sampling-style
  free -> the ring drains during forward, no deadlock, forward overlap
  identical to Phase 3.
- The RECOMPUTE runs grad-enabled -> traces `fetch_free_bwd` -> the
  buffer is freed exactly when the block's backward completes, which is
  the true last read.
- `torch.is_grad_enabled()` is a Dynamo guard/trace-time constant, so
  both variants exist in their own graphs; no data-dependent branch.
- Depth guard semantics unchanged: `fetch_start` blocks the host on the
  oldest ticket's free event; in backward that now means "previous
  block's backward done", bounding the ring at K in backward too.

## Compute path and LoRA

- **Training streamed linear:** `streamed_linear` gains a training
  variant selected at PACK time (trace-time constant): fp8-native ->
  `_fp8_linear_training(x, view.weight.t(), view.scale.reshape(-1),
  view.bias)` (grad-safe Function, proven traceable in the synthetic
  tests); weights never saved for backward (the Function saves only the
  in-graph views, which the recompute re-produces).
- **LoRA joins in the leaves path, not via hijacks.** The leaves-passing
  block forward bypasses module `forward`s, so LoRA hijacks never run
  (and would fail the region audit anyway). Instead the trunk reads each
  streamed Linear's LoRA A/B/scale (fp32 Parameters, ordinary graph
  inputs — proven in the synthetic joint-graph test) and the block
  forward computes `base + (x.to(A.dtype) @ A.t() @ B.t() * scale *
  multiplier).to(base.dtype)` when a LoRA entry is present (presence is
  a pack-time constant). Multiplier must be a scalar (the compile-fast
  readiness conditions apply); non-qualifying networks fail closed as
  `lora_untraceable`.
- Blocks without LoRA (or with it disabled) trace the base-only path —
  same fingerprint discipline as the streamed-index set.

## Slices

- **S1 — `fetch_free_bwd` op.** In `ingraph_stream.py`: custom op +
  `register_autograd` (backward records free, passes grad through),
  fake impl identity. Tests (`tests/test_ingraph_training_ops.py`):
  opcheck; grad-mode selection under checkpoint traces both variants
  with zero breaks and no deadlock at depth 2; a backward buffer-reuse
  hammer in the spirit of the Phase 3 signed-permutation test — exact
  chains through fwd+bwd, corruption cannot hide in tolerance; ring
  peak respects depth K in fwd AND bwd.
- **S2 — training leaves path.** Training compute selection in
  `streamed_linear`/`LinearView` (+ LoRA leaf entries in the pack view
  structs). Tests: loss + LoRA-grad parity vs eager reference on a
  synthetic 2-block model; no base-weight tensor among
  saved-for-backward (saved-tensor hook snapshot).
- **S3 — mmdit + krea2 wiring.** `enable_ingraph_training` (region
  audit, packs reused/rebuilt per training fingerprint, fail-closed
  reason list), grad path in `_blocks_trunk` (checkpoint wrapping the
  ingraph block fn), `ModelConfig.layer_offloading_ingraph_training`
  flag (off by default), krea2 attach wiring. Flags-off = zero behavior
  change.
- **S4 — Krea2-scale proof.** Train smoke A/B at keep_last=0: eager
  3.97 s/step vs ingraph compiled; DoD: zero breaks over full
  fwd+bwd+step, LoRA grads present all steps, loss parity within FP8
  tolerance vs eager, ring peak bounded, step time <= baseline (target:
  beat it — the sampler's compiled forward already runs 3.56 s/pass on
  the same transfers).

## S1 findings (2026-07-04) — compiled-trunk ordering, the full map

Eager is DONE (d47bd45). For the compiled trunk, three mechanisms were
tried; the constraint set is now fully mapped:

1. **Plain HOP checkpoint**: AOT backward graph is CORRECT (re-fetch ->
   grads -> free per unit, recomputed token via SAC MUST_SAVE on frees),
   but functionalized fetch ops carry only data deps, and backward
   re-fetches depend just on saved boundary activations -> Inductor
   hoists them above frees -> depth-guard fail-fast (by design).
2. **Ordered effect tokens** (`_register_ordered_effects`, kept opt-in):
   give exactly the needed program order, but trip a token-erasure
   assertion inside the checkpoint HOP lowering on torch 2.12.
3. **Flat trunk (no HOP) + effects**: registers and orders fine (probe:
   forward 6 fetches, ordered, zero breaks) — but effectful ops are not
   recomputed by the partitioner (correctly: side effects must not
   replay), so backward gets no re-fetch at all; and without recompute
   the saved views read freed ring buffers. Effects-for-ordering and
   partitioner-recompute are mutually exclusive.

**Chosen direction: Inductor post-grad ordering pass** (the parent
plan's Tier-2 machinery, pulled forward): keep the HOP + SAC design from
(1) — whose graphs are already correct — and add a
`post_grad_custom_post_pass` that threads explicit dependencies between
consecutive auto_functionalized fetch/free nodes in the backward graph
(free_N -> fetch_{N-1}), restoring eager order without effect tokens.
Local, no torch patch; also re-test (2) on the next torch upgrade — if
effects x HOP is fixed upstream, the pass becomes dead code.

## S4 RESULT (2026-07-05) + compile-latency findings

Krea2 scale, 28 blocks streamed, keep_last=0, 512px batch 1 (9cdf8f4):
**steady 2.10 s/step vs 3.97 eager (1.9x)**, 512/512 LoRA grads, losses
track eager within fp8 tolerance, fetches = 28 fwd + 28 bwd per step
(exactly the design), VRAM 5.8-7.7 GiB (~3-5 GiB headroom vs legacy =
the Phase 5 partial-residency budget). Traps fixed en route: LoRA entry
collection must run BEFORE the contaminant strip; pack-source ownership
class (`_mm_ingraph_pack_source`, `tests/test_residency_ownership.py`);
fetch ops MUST_RECOMPUTE (PREFER let the partitioner save 12.25 GiB of
flats); WDDM hard allocator cap makes overshoot loud.

Cold compile is ~150 s per (model, shape bucket) and is **tracing/AOT
dominated**: the mega-cache blob loads but buys nothing (A/B: cold 151 s,
warm 188 s -- see `COMPILE_MEGA_CACHE_PLAN.md` measured section, incl.
the custom-pass stable-uuid rule that fix produced).
`nested_compile_region` (trace the identical block once) is the real
lever but torch 2.12 rejects invoke_subgraph training with input
mutations -- our guarded `_after` ops. PARKED UNLOCK: mutation-free op
set -- `fetch_free_gated(token, gate) -> token` threaded functionally
through the trunk (forward chain in traced code; backward side needs
design) so no fetch op declares mutations; then re-try nested regions
AND drop the post-grad rewrite pass entirely (gating becomes source-level).

## Risks

| Risk | Mitigation |
|---|---|
| Checkpoint HOP x custom-op autograd (`fetch_free_bwd`) interaction | S1 tests it in isolation before any model wiring; synthetic checkpoint x compile already green |
| Double compute cost of full recompute makes compiled training slower than eager despite fusion | measure at S4; SAC-min (Rung 3) is the designed answer, not a rewrite |
| LoRA multiplier/scale variants (assistant inverse, per-key multipliers) | fail closed `lora_untraceable`; only the scalar fast path compiles in 4a |
| Pin budget: packs + training host-cache reserve must coexist (grad D2H staging) | pin_manager priority policy already enforces the reserve; packs are weight-tier |
