# In-Graph Weight Streaming — Full Implementation Plan

> Durable plan. Status lives in a git-bug ticket, not here.
> Parent: `COMPILE_STREAMED_OFFLOAD_PLAN.md` ("Theoretical endpoint" section —
> this plan is that endpoint, made concrete). Shared prerequisites live in that
> plan's Slice 1 (grad-safe FP8 linear) and Slice 2 (LoRA compile-clean path).
> Related: `BLOCK_STREAM_PLAN.md` (block staging machinery this absorbs),
> the DXGI pin-for-speed effort (hard prerequisite, see Phase 1/6).

## Goal

Refactor the offload subsystem so that, from torch.compile's point of view,
an offloaded model is **just a normal model**: one compiled graph (forward
and backward) with zero graph breaks, where weight streaming appears as
ordinary functional ops in the graph. No forward hijacks, no module hooks,
no `torch.autograd.Function`, no eager staging between compiled regions.

The FSDP analogy is the design's north star: hook-driven FSDP1 graph-breaks
at unit boundaries; SimpleFSDP/compiled-FSDP2 expresses the pre-forward
all-gather as functional ops and lets the compiler schedule them. An H2D
copy from pinned host memory is the same op class as an all-gather.

### Non-goals (this plan)

- Full fine-tune of *streamed* float weights (grad D2H staging). Streamed
  blocks are frozen quantized bases + LoRA. Readiness reason
  `trainable_base_weight` keeps such layers on the legacy path.
- Conv streaming (Krea2 streamed blocks are Linear-only).
- Removing the hook-based streaming path. It remains the fallback for
  non-refactored models and degraded modes; deprecation is a separate,
  later decision.
- CUDA-graph capture is a stretch phase (7), not a commitment.

## Target architecture

Five components, replacing today's hijack-based streaming for models that
opt in:

1. **Attach-time flat packing (host side).** Each streamed block's weight
   leaves (qdata, scale, bias — via the existing `_flatten_leaves`) are
   packed ONCE at attach into a single contiguous **pinned** host buffer
   per block, and the original TorchAO wrappers are re-pointed at views of
   that buffer (`_rebuild_from_leaves` onto the host buffer), so there is
   no RAM duplication and `state_dict()` still sees intact wrappers.
   This replaces `stage_block_forward`'s *per-step* host memcpy pack and
   removes the bounce pool from the step path entirely: the flat buffer IS
   the pinned source. (Independent perf value even without compile.)

2. **Split-phase fetch ops** (`toolkit/memory_management/ingraph_stream.py`):

   - `torch.ops.mm.fetch_start(host_flat: Tensor) -> Tensor` (token).
     Impl: allocate the device buffer, enqueue ONE `cudaMemcpyAsync` on the
     manager's transfer stream, record a ready event, stash
     `(dev_buffer, event)` in a ticket side table, return a small int
     tensor carrying the ticket id. Fake impl: empty token.
   - `torch.ops.mm.fetch_wait(token: Tensor, nbytes: int) -> Tensor`
     (uint8, cuda). Impl: pop the ticket, make the current stream wait the
     event, return the device buffer. Fake impl:
     `torch.empty(nbytes, dtype=uint8, device='cuda')`.
   - Both registered with `torch.library.custom_op`, `mutates_args=()`,
     marked non-differentiable. The wait output is a fresh graph tensor
     (the impl's buffer is not graph-visible before wait), satisfying the
     no-aliasing contract. Slicing the flat buffer into per-leaf views
     happens **in-graph** as normal view ops so Inductor sees and plans
     them (this is why the op returns one flat tensor, not a leaf list —
     a custom op must not return aliased outputs).
   - Depth guard: the runtime refuses > K outstanding tickets (K = ring
     depth, default 2) by making `fetch_start` block the *host* on the
     oldest ticket's free event — the VRAM bound survives any schedule.

3. **`StreamedLinear` module (compile-visible compute).** At attach, each
   streamed Linear is swapped for (or its block rewired around) a module
   whose forward is pure traced math over the fetched leaf views:
   fp8-native path -> `_fp8_linear_training` (grad-safe `_scaled_mm`, from
   the parent plan's Slice 1); other quant formats -> in-graph dequant +
   GEMM (Inductor fuses and memory-plans the dequant output — retiring the
   hand-rolled `w_dest` slot-reuse machinery); float -> plain GEMM. The
   fetch itself is issued at BLOCK granularity in the block's forward, and
   the block passes the leaf views to its Linears. No forward hijack, no
   `_layer_memory_manager` attribute on these modules.

4. **Backward re-fetch via selective activation checkpointing (SAC).**
   Blocks are wrapped in non-reentrant `torch.utils.checkpoint` with a
   `context_fn` policy: `mm.fetch_start` / `mm.fetch_wait` (and, for
   dequant formats, the dequant ops) are `MUST_RECOMPUTE`; activation
   save/recompute for everything else stays a separate, orthogonal choice:
   - "SAC-min" mode: only fetches recompute -> weights re-fetched in
     backward, activations saved (no double compute; more VRAM).
   - "Full checkpoint" mode: today's whole-block recompute (less VRAM).
   Both use the same mechanism; the planner picks per block. Weights are
   never saved for backward in either mode.

5. **Prefetch scheduling.** Two tiers:
   - **Tier 0 (free): host run-ahead.** The compiled forward calls
     `fetch_start(i)` on the host while the GPU is still executing block
     i-1's kernels; in backward, autograd enqueues block i's recomputed
     fetch while the GPU still runs block i+1's backward. This is the same
     overlap mechanism the eager ring uses today and costs nothing.
   - **Tier 1: source-level K-ahead pipelining.** Because Dynamo unrolls
     the block loop at trace time, the schedule can live in *model code*:
     the loop in `mmdit.forward` issues `fetch_start` for block i+K and
     `fetch_wait` for block i (tokens carried in a small list variable —
     plain Python at trace time, dissolved into the unrolled graph). No
     Inductor pass required. K compile-time constant = ring depth.
   - **Tier 2 (optional, upstream-grade): Inductor post-grad pass** via
     `torch._inductor.config.post_grad_custom_post_pass` that hoists
     fetch_starts under a memory budget (imitating
     `reorder_for_compute_comm_overlap`), covering the AOT-generated
     backward graph where source-level scheduling can't reach. Only built
     if Tier 0 measurement shows backward stalls that matter.

## Phases

Each phase lands flag-gated (off by default), with seconds-scale GPU tests
in `tests/` (per repo convention), and is separately shippable. Graph-break
assertions use `torch._dynamo.explain()` / `torch._dynamo.utils.counters`.

### Phase 0 — Spike: prove the young machinery (go/no-go gate)

No manager integration; synthetic 2-block model, plain bf16 weights.

- Implement throwaway `fetch_start`/`fetch_wait` ops as in component 2.
- Prove, in order:
  1. `torch.compile(fullgraph=True)` traces a model whose forward calls
     the ops + in-graph views (zero breaks).
  2. SAC `context_fn` with MUST_RECOMPUTE on the ops produces a backward
     graph containing its own fetch calls (inspect
     `aot_graphs` logging), and the forward saves no weight tensor.
  3. Numerics: fwd/bwd parity vs the same model resident.
  4. Overlap: CUDA event timing shows the H2D of block i+1 overlapping
     block i's GEMMs with Tier-1 source-level pipelining.
  5. triton-windows + torch 2.12 survive all of the above.
- Deliverable: `tests/test_ingraph_spike.py` (kept as regression canary)
  plus findings in the ticket.
- **Go/no-go:** if (2) fails — the partitioner won't re-fetch — fall back
  to full-checkpoint-only mode (still viable, matches today's structure);
  if (1) fails fundamentally, this plan is dead and Design B in the parent
  plan is the ceiling. Timebox the spike before any refactor work starts.

### Phase 1 — Attach-time flat packing (host weight representation)

`ingraph_stream.py` + changes in `manager.py` attach path.

- `pack_block_host(block_key, linears) -> BlockPack`: flatten leaves
  (reuse `_flatten_leaves`), one aligned pinned buffer (layout code lifted
  from `stage_block_forward` lines ~994-1018), leaf metadata table
  (offsets, dtypes, shapes, per-Linear grouping).
- Re-point wrappers: rebuild each weight/bias wrapper onto host-buffer
  views (`_rebuild_from_leaves`), replacing the old storages. Assert
  `state_dict()` round-trips bitwise (test) and the checkpoint save path
  (PinnedStager) still works.
- Pinning: the flat buffer is allocated pinned; per-tensor
  `_pin_tensor_in_place` bookkeeping for these weights is dropped; the
  DXGI/bounce ledger accounts the flat buffers instead. **Hard
  prerequisite surfaced here:** if the DXGI shared budget cannot cover all
  streamed blocks' flat buffers, the planner must demote blocks to the
  legacy path (Phase 6 wires this; Phase 1 just exposes
  `required_pin_bytes` per block).
- Independent win to measure now: legacy block-stream mode
  (`stage_block_forward`) can consume the pre-packed buffer and skip its
  per-step host memcpy (fixes the "host pack per stage" cost that
  contributed to Slice 2's regression).
- Tests: `tests/test_ingraph_pack.py` — round-trip, alignment, view
  identity (no RAM duplication: storage data_ptr of wrapper leaves inside
  the flat buffer), save-path parity.

### Phase 2 — Production fetch ops + ticket runtime

- Harden the spike ops: ticket table keyed by monotonically increasing id;
  depth-K guard with free events recorded in-graph via a
  `mm.fetch_free(token)`-style op called after the block's last consumer
  (or, simpler and chosen by default: free = the NEXT fetch_start waiting
  the (i-K)th ticket's event, no third op needed).
  Decide in-phase; the plan's default is the two-op design.
- Error surface: fail fast (repo convention) — non-pinned source, ticket
  overflow, device mismatch all raise, never silently sync.
- Profiling: per-fetch H2D ms + bytes recorded into the existing perf
  ledger (`_begin_layer_profile` equivalents at block granularity);
  perf-log fields `ingraph_fetches`, `ingraph_h2d_ms`,
  `ingraph_wait_ms` per window so `digest_perf_log.py` picks them up.
- Tests: `tests/test_ingraph_fetch_ops.py` — depth guard blocks host at
  K+1 outstanding; event ordering (wait really gates compute); fake-impl
  shape correctness under `torch.compile`; zero breaks.

### Phase 3 — Compile-visible block forward, sampling first

Model-side refactor (Krea2 `mmdit.py` as reference integration).

- `SingleStreamBlock` gains a functional weight path: block forward takes
  an optional `leaves` struct; when present, its Linears compute from the
  passed views instead of `self.*.weight`. (Rewire, not module swap, keeps
  LoRA wrappers' `org_forward` chain intact — LoRA adds its delta around
  the same call.)
- `mmdit.forward` in ingraph mode: per block, `token = fetch_start(pack)`
  (Tier-1: issued K blocks early), `flat = fetch_wait(token)`, slice to
  views via the metadata table (in-graph), call block with leaves.
- Whole-forward compile region: `torch.compile` the transformer trunk
  (blocks loop + first/last layers), `fullgraph=True`. Sampler-only at
  this phase (`torch.no_grad`), so no SAC yet.
- Retire for ingraph mode: `enable_compiled_sampling`'s per-block
  compile + fingerprint machinery, the eager/compiled fork and pad-to-256
  logic in `forward` (keep for legacy mode); resident-trace hooks and
  block stage pre-hooks must not be installed on ingraph modules
  (`manager.py` `_install_resident_trace_hook` / hook wiring gated off).
- Residency: blocks the planner keeps resident simply skip fetch ops and
  read their own (GPU) params — same graph shape, weights as inputs; no
  separate compiled artifact per residency layout. A residency change is
  a graph-input change, not a recompile, EXCEPT streamed<->resident block
  set changes, which change the traced code path -> recompile (fingerprint
  on the streamed-block index set; reuse the `_compiled_fingerprint`
  pattern).
- Tests: `tests/test_ingraph_sampler.py` — output parity vs legacy eager
  streamed sampler (bit-close), one H2D per streamed block per pass,
  zero breaks, recompile count flat across shape-stable calls.

### Phase 4 — Training: SAC backward re-fetch + LoRA

Depends on parent plan Slice 1 (`_fp8_linear_training`) and the LoRA
compile-clean fast path (parent Slice 2 — install-time specialization of
`network_mixins.py:375`'s wrapper to `org + (x @ A.T @ B.T) * m`).

- Wrap each streamed block in `checkpoint(block_fn, ..., 
  use_reentrant=False, context_fn=_mm_sac_policy)`; `_mm_sac_policy`
  returns MUST_RECOMPUTE for `mm.fetch_*` (+ dequant ops in dequant mode),
  default otherwise. Planner chooses SAC-min vs full-checkpoint per block
  from the VRAM plan; `keep_last` maps to SAC-min blocks (activations
  saved, weights still re-fetched — `keep_last`'s "no recompute" meaning
  survives without needing residency).
- Compile the trunk with autograd (AOTAutograd joint graph); LoRA A/B are
  ordinary trainable graph inputs.
- Backward overlap: measure Tier-0 run-ahead first (host enqueues block
  i's recomputed fetch while GPU runs block i+1's backward). Only if the
  step profile shows real fetch stalls in backward does Tier-2 (Inductor
  pass) get built.
- Tests: `tests/test_ingraph_training.py` — loss + LoRA-grad parity vs
  legacy eager streaming on a synthetic multi-block model (pattern from
  `test_block_forward_stage.py` gradient-parity); no weight tensor among
  saved-for-backward (inspect ctx saved tensors / memory snapshot); ring
  peak respects depth K in fwd AND bwd; zero breaks over a full
  fwd+bwd+step.

### Phase 5 — Scheduling and shape policy

- Tier-1 K-ahead pipelining in the model loop (trivial after Phase 3;
  land with measurement).
- Dynamic shapes: training buckets -> per-bucket recompile with a bounded
  cache (`torch._dynamo.config.cache_size_limit` sized to bucket count) +
  a log line; sampler resolution set is small. Revisit `dynamic=True` on
  the seq dim only if bucket-count recompiles hurt in practice.
- Compile-latency budget: cold compile of the trunk (fwd+bwd, per bucket)
  measured and recorded; enable Inductor caching
  (`TORCHINDUCTOR_CACHE_DIR` is a test-harness/env concern only —
  persistent cache location must work out of the box on Windows).
- Tier-2 Inductor pass: separate opt-in module
  (`ingraph_stream_scheduling.py`), only if Phase 4 measurement demands
  it. Scope: hoist recomputed fetch_starts across block boundaries in the
  backward graph under the depth-K budget.

### Phase 6 — Manager/planner integration + config surface

- `ModelConfig`: `layer_offloading_ingraph: bool = False`,
  `layer_offloading_ingraph_depth: int = 2` (+ UI schema in
  `ui/src/app/jobs/new/jobConfig.ts` / docs entries, same pattern as
  `layer_offloading_block_stream_only`). No env vars for runtime behavior.
- `attach_smart_training(..., ingraph=True)`: planner runs unchanged
  (residency split, working reserve, pin budget), then routes streamed
  blocks to pack+rewire instead of `LinearLayerMemoryManager.attach`.
  Mutually exclusive with `block_stream_only` hooks and per-Linear
  hijacks; assert no `_layer_memory_manager` exists on ingraph modules.
- Pin budget gate: `required_pin_bytes` (Phase 1) vs the DXGI shared
  budget (real probe from the pin-for-speed effort, not the RAM*0.25
  proxy). Insufficient budget -> demote whole blocks to the legacy path
  and log; a partially-ingraph model is legal (legacy blocks stay eager
  and OUTSIDE the compile region — compile region boundaries at block
  granularity, chosen at attach).
- Autotune interplay: working-reserve/keep_last controllers FREEZE while
  ingraph compile is active (a resize means repack/redemote + recompile).
  Policy: controllers run during an eager warmup window (N steps,
  default e.g. 50), then the layout freezes and compile engages. This
  aligns with the existing "no-resize stable band" goal. `keep_last`
  changes map to SAC-mode flips (recompile, cheap-ish) rather than
  residency moves.
- Sampler/training transitions (`_restore_offload`): ingraph state is
  per-manager like today; transitions rebuild packs only if residency
  changed; compiled trunks for train vs sample are distinct artifacts
  (grad vs no-grad) and both survive a transition that preserves layout.
- Perf/diagnostics: plan log line gains
  `ingraph_blocks=N depth=K sac_min=M full_ckpt=P demoted=Q`.

### Phase 7 — CUDA graphs (stretch, separate flag)

- Precondition established by design: no host-side worker in the step
  path, H2D from pinned host (capturable), fixed K-deep device buffers.
- Work: make fetch impls capture-safe (pre-allocated ticket ring, no
  Python allocation post-warmup), `mode="reduce-overhead"` on the trunk,
  measure vs `keep_last` on the launch-bound regime.
- Exit early without shame if WDDM/driver behavior under capture
  misbehaves; record findings.

### Phase 8 — Migration, cleanup, upstream split

- Krea2 is the reference; port Z-Image/Anima by implementing the same
  block-loop contract (document the model-integration contract in
  `docs/`: "expose blocks list + functional leaf path").
- Legacy machinery explicitly kept: per-Linear ring (fallback), bounce
  pool (legacy + TE/misc), `stage_block_forward` (superseded — fold its
  useful part, pre-packed source, into Phase 1 or retire with
  `BLOCK_STREAM_PLAN.md` marked RESULT).
- Upstream decision doc update (`docs/decisions/UPSTREAM_PR_PLAN.md`):
  ingraph core (ops + SAC policy + pack) is upstream-general; Tier-2 pass
  and WDDM-specific gating stay local.

## Acceptance criteria

- `torch._dynamo.explain` over one full training step (offload on, LoRA
  on, FP8 base): 0 graph breaks; one compiled trunk fwd+bwd.
- No weight tensor saved for backward (memory snapshot proof); peak VRAM
  bounded by resident set + K blocks + working set, fwd and bwd.
- Parity: sampler images bit-close to legacy; training loss/LoRA-grad
  parity within FP8/GEMM noise.
- Perf on a real Krea2 run (user go-ahead; `digest_perf_log.py`):
  step time <= legacy eager streaming, with the win itemized (Python
  dispatch removal, fusion, overlap). If it loses in the transfer-bound
  regime, flag stays off and this doc gets a RESULT section.
- Flags off -> zero behavior change; full `tests/` suite green.

## Risk register

| Risk | Exposure | Mitigation |
|---|---|---|
| SAC/partitioner won't MUST_RECOMPUTE custom ops cleanly | Phase 0 | go/no-go spike; fallback = full-checkpoint mode only |
| Custom-op aliasing/DCE subtleties (wait output, token liveness) | Phase 0/2 | flat-buffer return + in-graph views; token is a data dependency, never dead |
| Host run-ahead insufficient in backward (fetch stalls) | Phase 4 | measured before Tier-2 is built; Tier-2 pass is scoped small |
| Recompile storms (buckets x SAC modes x residency sets) | Phase 5/6 | freeze-after-warmup policy; fingerprint + bounded cache + log |
| DXGI budget can't hold all packs | Phase 6 | per-block demotion to legacy path; two-cliff ledger from pin-for-speed governs |
| triton-windows instability over multi-day training | all | sampler compile already proven; canary test in CI-less reality = spike test rerun per torch upgrade |
| Inductor internals churn (only Tier-2) | Phase 5b | optional tier; everything else uses stable public APIs (custom_op, SAC, compile) |
| WDDM allocator fragmentation from K-deep flat buffers | Phase 2 | fixed-size per-block buffers reused via ticket ring -> stable allocation pattern by construction |

## Sequencing and rough sizes

Critical path: 0 -> 1 -> 2 -> 3 -> 4 -> 6; 5 rides along 3/4; 7/8 trail.
Parent-plan Slice 1 (FP8 grad-safe) can proceed in parallel and is needed
by Phase 4. Rough sizes: 0=S(spike, timeboxed), 1=M, 2=M, 3=L (model
refactor), 4=L, 5=S(+L if Tier-2), 6=M, 7=M?, 8=S. The plan is
deliberately front-loaded so the riskiest unknowns (0) and the
independently-valuable pieces (1) come first.
