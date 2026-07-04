# In-Graph Weight Streaming — Full Implementation Plan

> Durable plan. Status lives in a git-bug ticket, not here.
> Parent: `COMPILE_STREAMED_OFFLOAD_PLAN.md` ("Theoretical endpoint" section —
> this plan is that endpoint, made concrete). Shared prerequisites live in that
> plan's Slice 1 (grad-safe FP8 linear) and Slice 2 (LoRA compile-clean path).
> Related: `BLOCK_STREAM_PLAN.md` (block staging machinery this absorbs),
> the DXGI pin-for-speed effort (hard prerequisite, see Phase 1/6).
> Phase 3 has a self-contained execution plan:
> `INGRAPH_PHASE3_SAMPLER_PLAN.md` (fully streamed compiled sampling).
> Decisions resolved during implementation are marked DECIDED inline;
> lessons from Phases 0-3 live in "Cross-cutting design rules".

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
   - `torch.ops.mm.fetch_free(token: Tensor) -> Tensor`: records the
     block's compute-done event so the ticket's device buffer may be
     reused. (DECIDED in Phase 2: three-op design, explicit free — the
     two-op alternative was rejected once implementation started.)
   - All registered with `torch.library.custom_op`, `mutates_args=()`,
     marked non-differentiable, validated with `torch.library.opcheck`.
     The wait output is a fresh graph tensor (the impl's buffer is not
     graph-visible before wait), satisfying the no-aliasing contract.
     Slicing the flat buffer into per-leaf views happens **in-graph** as
     normal view ops so Inductor sees and plans them (this is why the op
     returns one flat tensor, not a leaf list — a custom op must not
     return aliased outputs).
   - Depth guard: the runtime refuses > K outstanding tickets (K = ring
     depth, default 2) by making `fetch_start` block the *host* on the
     oldest ticket's free event — the VRAM bound survives any schedule.
   - **Buffer-lifetime invariant (Phase 0 lesson, non-negotiable):** a
     device buffer allocated on the transfer stream and consumed on the
     compute stream must not be recycled until the compute stream is past
     its last read — the ticket ring's free events are what enforce this
     (`record_stream` was only the spike's stopgap). Violation is silent
     numeric corruption; a standing regression test hammers reuse at
     depth=2 (see the Phase 3 execution plan, S7).

3. **Leaves-passing block forward (compile-visible compute).** DECIDED in
   Phase 3: rewire, not module swap — the block's forward takes an
   optional `leaves` struct and, when present, every streamed Linear
   computes pure traced math from the passed views and never touches
   module weight params (`leaves=None` preserves legacy behavior
   bit-for-bit). Compute paths, selected at PACK time (trace-time
   constant, not a data-dependent branch): fp8-native sampling ->
   `_fp8_linear_compiled` on the in-graph `qdata.t()` view (raw layout in
   the pack; `.t()` is a free view); fp8-native training ->
   `_fp8_linear_training` (grad-safe, parent plan Slice 1); other quant
   formats -> in-graph dequant + GEMM (Inductor fuses and memory-plans
   the dequant output — retiring the hand-rolled `w_dest` slot-reuse
   machinery); float -> plain GEMM. The fetch is issued at BLOCK
   granularity, and the compile region must pass the region audit: no
   module hooks, no `_layer_memory_manager`, and no instance-attribute
   `forward` hijacks (`'forward' in module.__dict__` — how legacy
   streaming, LoRA, and the fp8 installer all attach).

4. **Backward re-fetch via checkpointing, staged as a ladder.**
   Blocks are wrapped in non-reentrant `torch.utils.checkpoint`; the two
   modes share one mechanism and land in order (Phase 4a then 4b):
   - "Full checkpoint" mode (FIRST rung, no SAC policy at all): the
     whole block recomputes in backward, and because the fetch ops are
     inside the checkpointed callable, backward re-fetch falls out of
     plain checkpoint semantics. Less VRAM, double compute — today's
     training structure.
   - "SAC-min" mode (second rung): `context_fn` policy marks
     `mm.fetch_start`/`mm.fetch_wait` (and, for dequant formats, the
     dequant ops) MUST_RECOMPUTE; everything else saves. Weights
     re-fetched in backward, activations kept (no double compute; more
     VRAM) — this is what preserves `keep_last`'s meaning.
   The planner picks per block. Weights are never saved for backward in
   either mode. Grad-mode calls WITHOUT checkpoint-or-SAC are forbidden
   by construction: the default partitioner would save every fetched
   weight for backward (28 x ~0.9 GB -> OOM), so there is no cheap
   "just enable grad" probe between sampling and training compile.

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

## Cross-cutting design rules (amendments from Phases 0-3)

- **Fail closed, everywhere.** Ingraph requested but unavailable raises
  with a machine-readable reason from a fixed vocabulary
  (`wrapper_pack_missing`, `hook_present`, `forward_hijack_present`, ...);
  no code path silently falls back to regional compile, legacy streaming,
  or eager. The first implementation pass fell back open and produced a
  trace that looked like an ingraph run but was regional compile
  recompiling on legacy hook identity — strictness is what makes results
  mean anything.
- **Compile region = wrapper-free by construction.** Only modules that
  pass the region audit may sit inside a compiled trunk. Resident modules
  whose params are still TorchAO wrappers (e.g. Krea2's `first`/
  `txtfusion`/`last`) stay OUTSIDE the region until unpacked — otherwise
  their wrappers become graph inputs, the exact subclass-in-graph failure
  this design exists to avoid. For Phase 3 the region is the blocks trunk
  only.
- **Weights-as-inputs must stay guard-free.** A repack (new host tensors,
  same values) and a residency-preserving layout change must cause zero
  recompiles; only a change to the streamed-block index set (traced code
  path) may. This is the property the whole architecture leans on, and it
  is test-enforced, not assumed.
- **Parity is compiled-vs-compiled.** Inductor's bf16 epilogue rounding
  makes compiled-vs-eager non-bitwise by construction (Phase 0 finding);
  bitwise assertions compare against a compiled resident reference,
  end-to-end comparisons use documented tolerances.

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
  (offsets, dtypes, shapes, per-Linear grouping, per-leaf role:
  qdata/scale/bias/float_weight).
- **Quantized TorchAO wrappers are in scope from the start** — they are
  the entire point (the streamed Krea2 weights are fp8 wrappers). A
  plain-tensor-only pack does NOT complete this phase; the first
  implementation pass deferred wrapper packing and the Phase 3 sampler
  smoke could not run at all as a result. Unknown wrapper layouts (leaf
  count != 2, non-fp8 qdata) fail closed as `unsupported_quant_wrapper`.
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
  depth-K guard with free events recorded in-graph via
  `mm.fetch_free(token)` after the block's last consumer. (DECIDED: the
  three-op design; the two-op "next fetch_start frees" alternative was
  dropped during implementation. `fetch_free` is also what enforces the
  buffer-lifetime invariant — see component 2.)
- Error surface: fail fast (repo convention) — non-pinned source, ticket
  overflow, device mismatch, unknown ticket all raise, never silently
  sync; `torch.library.opcheck` coverage for all three ops.
- Profiling: per-fetch H2D ms + bytes recorded into the existing perf
  ledger (`_begin_layer_profile` equivalents at block granularity);
  perf-log fields `ingraph_fetches`, `ingraph_h2d_ms`,
  `ingraph_wait_ms` per window so `digest_perf_log.py` picks them up.
- Tests: `tests/test_ingraph_fetch_ops.py` — depth guard blocks host at
  K+1 outstanding; event ordering (wait really gates compute); fake-impl
  shape correctness under `torch.compile`; zero breaks.

### Phase 3 — Compile-visible block forward, sampling first

> Execution plan: `INGRAPH_PHASE3_SAMPLER_PLAN.md` (re-scoped after the
> first implementation pass fell back to regional compile: fail-closed
> smoke, wrapper packing pulled forward from the deferred slice, region
> audit incl. forward hijacks, blocks-trunk-only compile region).

Model-side refactor (Krea2 `mmdit.py` as reference integration). The
execution plan is authoritative for this phase's work items and
definition of done; design summary:

- `SingleStreamBlock.forward(..., leaves=None)`: when passed, streamed
  Linears compute from the leaf views (component 3); `None` preserves
  legacy behavior bit-for-bit.
- **Compile region = the blocks trunk only** (`trunk(x, tvec, freqs,
  mask, packs)`, `fullgraph=True`, no-grad). The quantized-resident
  preamble/tail (`first`/`tmlp`/`txtfusion`/`txtmlp`/`posemb`/`last`)
  runs eager outside the region this phase (see cross-cutting rules:
  wrapper-free by construction). NOT the whole `mmdit.forward` — the
  first implementation pass compiled the whole forward and put resident
  wrappers into the graph as inputs.
- Milestone default: ALL 28 blocks streamed (the fully-streamed proof);
  residency optimization is not this phase's job. Wrapper-free resident
  blocks may join the region later; wrapper-bearing ones may not (audit
  enforces). Fingerprint on (streamed-index set, depth, seq bucket);
  reuse `_compiled_ingraph_fingerprint`.
- Fail-closed wiring: `enable_ingraph_sampling()` computes the
  unavailable-reason list, runs the region audit, and raises in strict
  mode rather than falling back to `enable_compiled_sampling()` regional
  compile; the legacy machinery is kept for legacy mode only, and legacy
  hooks/hijacks are never installed on ingraph modules.
- Config: `layer_offloading_ingraph_sampling: bool = False` +
  `layer_offloading_ingraph_depth: int = 2` land here (smoke needs them);
  UI schema follows in Phase 6.
- No K-ahead pipelining until the simple schedule is correct (Tier-0
  run-ahead is free); K-ahead is a measured add-on at the end of the
  phase.
- Tests + acceptance: see the execution plan (region audit, guard
  stability incl. repack, buffer-reuse regression, one H2D per streamed
  block per pass, tolerance-based parity vs legacy eager streaming).

### Phase 4 — Training compile, as a three-rung risk ladder

Grad mode cannot be approached incrementally by "compiling only the
forward": AOTAutograd fires the moment a compiled callable sees
grad-enabled inputs, and without checkpoint-or-SAC the partitioner saves
every fetched weight (OOM by construction — see component 4). A
"forward-compiled / eager-backward" bridge via a custom autograd.Function
was CONSIDERED AND REJECTED: it resurrects the hijack architecture this
plan deletes, needs a second throwaway eager re-fetch path, splits
forward/backward math when LoRA is present, and avoids only risks Phase 0
already retired. The viable staging is the ladder below — each rung uses
only endpoint machinery, so nothing is thrown away.

**Rung 1 (Phase 4-pre) — resident-block training compile, no streaming.**
This is `COMPILE_STREAMED_OFFLOAD_PLAN.md` Slice 2, ordered explicitly
before any streamed training: compile pinned-resident blocks for training
with the LoRA compile-clean fast path (parent Slice 2 — install-time
specialization of `network_mixins.py:375`'s wrapper to
`org + (x @ A.T @ B.T) * m`) and `_fp8_linear_training` (parent Slice 1).
Zero fetch ops involved: this rung isolates exactly what Phase 0 did NOT
de-risk — LoRA in the AOTAutograd joint graph, the fp8 grad path, and
checkpoint x compile at Krea2 scale.

**Rung 2 (Phase 4a) — full-checkpoint streamed training, no SAC.**
- Wrap each streamed block as `checkpoint(block_fn, ...,
  use_reentrant=False)` where `block_fn` contains the fetch ops and the
  leaves-passing block call. Backward re-fetch falls out of plain
  checkpoint recompute — no `context_fn`, no partitioner policy.
- Compile the trunk with autograd; LoRA A/B are ordinary trainable graph
  inputs (LoRA joins here only after Rung 1 proves it in-graph).
- Backward overlap: measure Tier-0 run-ahead (host enqueues block i's
  recomputed fetch while the GPU runs block i+1's backward). Only if the
  step profile shows real fetch stalls does Tier-2 get built.
- Tests: `tests/test_ingraph_training.py` — loss + LoRA-grad parity vs
  legacy eager streaming on a synthetic multi-block model (pattern from
  `test_block_forward_stage.py` gradient-parity); no weight tensor among
  saved-for-backward (memory snapshot); ring peak respects depth K in
  fwd AND bwd; zero breaks over a full fwd+bwd+step.

**Rung 3 (Phase 4b) — SAC-min.**
- `_mm_sac_policy` returns MUST_RECOMPUTE for `mm.fetch_*` (+ dequant ops
  in dequant mode), save otherwise; planner chooses SAC-min vs
  full-checkpoint per block from the VRAM plan. `keep_last` maps to
  SAC-min blocks (activations saved, weights still re-fetched —
  `keep_last`'s "no recompute" meaning survives without residency).
- Tests extend Rung 2's: per-mode parity, VRAM delta between modes
  matches the activation-residency prediction, mode flip -> exactly one
  recompile (fingerprint includes per-block mode).

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

- `ModelConfig`: `layer_offloading_ingraph_sampling` and
  `layer_offloading_ingraph_depth` land in Phase 3 (the smoke needs
  them); Phase 6 adds `layer_offloading_ingraph_training: bool = False`
  and the UI schema for all three (`ui/src/app/jobs/new/jobConfig.ts` /
  docs entries, same pattern as `layer_offloading_block_stream_only`).
  No env vars for runtime behavior.
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

Phase 3 has its own self-contained definition of done in
`INGRAPH_PHASE3_SAMPLER_PLAN.md` (fully streamed compiled sampling).
Endpoint criteria for the whole plan:

- `torch._dynamo.explain` over one full training step (offload on, LoRA
  on, FP8 base): 0 graph breaks; one compiled trunk fwd+bwd.
- No weight tensor saved for backward (memory snapshot proof); peak VRAM
  bounded by resident set + K blocks + working set, fwd and bwd.
- Parity: sampler latents within documented tolerance of legacy eager
  streaming (compiled-vs-eager is never bitwise — see cross-cutting
  rules); training loss/LoRA-grad parity within FP8/GEMM noise.
- Perf on a real Krea2 run (user go-ahead; `digest_perf_log.py`):
  step time <= legacy eager streaming, with the win itemized (Python
  dispatch removal, fusion, overlap). If it loses in the transfer-bound
  regime, flag stays off and this doc gets a RESULT section.
- Flags off -> zero behavior change; full `tests/` suite green.

## Risk register

| Risk | Exposure | Mitigation |
|---|---|---|
| SAC/partitioner won't MUST_RECOMPUTE custom ops cleanly | Phase 0 | go/no-go spike; fallback = full-checkpoint mode only |
| Custom-op aliasing/DCE subtleties (wait output, token liveness) | Phase 0/2 | flat-buffer return + in-graph views; token is a data dependency, never dead; opcheck |
| Cross-stream buffer recycle -> silent numeric corruption (HIT in Phase 0) | Phase 2/3 | ticket-ring free events own buffer lifetime; standing depth=2 reuse regression test |
| Fail-open fallbacks make traces lie (HIT in Phase 3) | all | strict mode + reason taxonomy + region audit; fallback branches removed from ingraph paths |
| Host run-ahead insufficient in backward (fetch stalls) | Phase 4 | measured before Tier-2 is built; Tier-2 pass is scoped small |
| Recompile storms (buckets x SAC modes x residency sets) | Phase 5/6 | freeze-after-warmup policy; fingerprint + bounded cache + log |
| DXGI budget can't hold all packs | Phase 6 | per-block demotion to legacy path; two-cliff ledger from pin-for-speed governs |
| triton-windows instability over multi-day training | all | sampler compile already proven; canary test in CI-less reality = spike test rerun per torch upgrade |
| Inductor internals churn (only Tier-2) | Phase 5b | optional tier; everything else uses stable public APIs (custom_op, SAC, compile) |
| WDDM allocator fragmentation from K-deep flat buffers | Phase 2 | fixed-size per-block buffers reused via ticket ring -> stable allocation pattern by construction |

## Sequencing and rough sizes

Critical path: 0 -> 1 -> 2 -> 3 -> 4-pre -> 4a -> 4b -> 6; 5 rides along
3/4; 7/8 trail. Parent-plan Slices 1 and 2 (FP8 grad-safe linear, LoRA
clean path) can proceed in parallel to Phases 1-3 and gate Rung 1
(Phase 4-pre). Rough sizes: 0=S (spike, timeboxed), 1=M, 2=M, 3=L (model
refactor; see execution plan), 4-pre=M, 4a=M, 4b=S, 5=S (+L if Tier-2),
6=M, 7=M?, 8=S. The plan is deliberately front-loaded so the riskiest
unknowns (0) and the independently-valuable pieces (1) come first, and
Phase 4's risk is graded: each rung adds exactly one new variable
(Rung 1: LoRA/fp8-grad in-graph; 4a: streamed fetch under autograd;
4b: partitioner policy).
