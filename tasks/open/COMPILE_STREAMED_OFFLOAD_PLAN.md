# torch.compile for Streamed (Offloaded) Blocks — Plan

> Durable plan. Mutable status (what's done / blocked) belongs in a git-bug
> ticket, not here. Related plans: `BLOCK_STREAM_PLAN.md` (block-granular
> staging this builds on), `UPSTREAM_PR_D_FP8_TRAINING_PLAN.md`.

## Goal

Make the memory manager fully torch.compile-compatible so that **blocks whose
weights stream CPU<->GPU** run as compiled graphs with **zero graph breaks**,
in both sampling and training. Today only fully-resident blocks compile
(`enable_compiled_sampling`), streamed blocks are forced eager, and training
compile exists only as a diagnostic gate (`training_compile_readiness`).

Off by default, behavior-preserving with the flag off (upstream-PR
discipline). No env-var-gated runtime behavior; expose via `ModelConfig` +
UI schema.

## Why streamed blocks can't compile today (blocker inventory)

1. **`_BouncingLinearFn`** (`manager_modules.py`): a custom
   `torch.autograd.Function` doing H2D staging on a transfer stream, event
   record/wait, bounce-pool acquire/consume, TorchAO wrapper dequant,
   data-dependent branches (quantized? fp8? block-resident hit?), and
   profile/trace side effects. Wrapped in `@torch.compiler.disable` — by
   design untraceable. Any block containing one stays eager
   (`_block_compile_safe`).
2. **Backward re-staging**: the weight is not held between forward and
   backward; backward re-fetches it. AOTAutograd's default contract is
   "save the forward weight tensor for backward" — that would defeat
   offloading (VRAM) or dangle (ring slot overwritten).
3. **TorchAO tensor subclass** as `weight`: subclass dispatch inside a graph
   is the thing that caused the original compile freeze. The sampler fix
   (unpack raw `qdata.t()` / `scale` at install time, compute with
   `_fp8_linear_compiled`) is forward-only; training needs a grad-safe
   equivalent (the `_memory_management_training_compile_fp8` gate in
   `mmdit.training_compile_readiness` is reserved for exactly this and is
   currently never set).
4. **Module-hook staging** (`_make_block_stage_prehook`): Dynamo traces
   module pre-hooks, so eager staging code must not live in a hook on a
   compiled module.
5. **LoRA wrapper forward** (`network_mixins.py:375`): branches on
   `network.is_active`, `_multiplier == 0`, batch-interleave logic. Traceable
   in principle, but produces guards; needs a compile-clean fast path.
6. **Grad D2H staging** (`_stage_grads_to_cpu`): only used when base weights
   are trainable floats (full FT). Untraceable as-is.
7. **Prefetch/bounce-pool ordering**: the pool cursor assumes the trace's
   per-layer fetch order; compiled execution must not reorder fetches.
8. **Dynamic shapes**: seq-len changes recompile `dynamic=False` graphs
   (sampler already quantizes via pad-to-256; training has bucketed
   resolutions).

## Architecture decision

Two candidate designs; we pick **Design B** with Design A as a scoped
fallback.

### Design A — opaque custom ops (fallback / bridge)

Wrap stage+GEMM+release in `torch.library.custom_op` (+ `register_fake` for
shape propagation, `register_autograd` whose backward is itself a custom op
that re-stages). torch 2.12 supports this cleanly; an opaque op is a single
graph node, so no graph break, and the existing eager machinery survives
unchanged inside it.

Rejected as the primary design because it keeps the per-Linear Python
dispatch cost (the actual bottleneck the block-stream work attacks), hides
the GEMM from Inductor (no fusion win — compile would buy almost nothing),
and needs an ordering hack (a nominally-mutated token tensor arg so
auto-functionalization serializes fetch order). Keep it in the toolbox for
one-off layers that resist Design B.

### Design B — data movement outside the graph, pure-math inside (target)

Split the streamed block into:

- **Eager stage step** (not traced): block-granular H2D of all the block's
  leaves in one burst — this is exactly Block Stream Slice 2
  (`stage_block_forward` / `consume_block_resident`), called explicitly from
  `mmdit.forward` between compiled block calls (NOT from a module hook).
  It ends with the compute stream waiting the ready event, same as today.
- **Compiled compute step**: ONE shared compiled callable for all streamed
  blocks (same code object => one graph per shape bucket, not per block),
  taking the staged raw leaves as **explicit tensor inputs**:
  `compiled_block(x, tvec, freqs, mask, *leaves)` where leaves are raw
  `qdata_t`/`scale`/`bias` (or bf16 weights for float models) — plain
  tensors, never the TorchAO wrapper. Weights-as-inputs means no parameter
  identity guards, so the 2-slot ring's changing tensors are just new
  arguments, not recompiles. Implementation: rewire the block's Linears to
  read from call-scoped leaves (a functional forward or
  `torch.func.functional_call` with a per-call param dict).
- **Backward = checkpoint recompute**: wrap (stage + compiled block) in
  non-reentrant `torch.utils.checkpoint` — already the training structure
  (`mmdit.forward` checkpoints every streamed block). Recompute re-runs the
  eager stage step, so the weight is freshly resident when the compiled
  graph's backward needs it. **No weight is ever saved for backward**; only
  `x` is, as today. This is the key move that makes streamed training
  compile possible without teaching AOTAutograd about re-staging.

The FP8 GEMM stays *inside* the graph (grad-safe `_fp8_linear_compiled`
sibling), so Inductor fuses the activation quant/clamp/scale epilogues —
that fusion plus removal of ~224 per-Linear Python dispatches is where the
step-time win comes from.

## Slices

Each slice lands with focused tests under `tests/` (GPU, seconds) and keeps
its flag off by default. Graph-break assertions use
`torch._dynamo.explain()` / `torch._dynamo.utils.counters` on synthetic
blocks — no full training runs needed until final validation.

### Slice 0 — flag + readiness plumbing (small)

- `ModelConfig`: `layer_offloading_compile_streamed: bool = False` (+ UI
  schema/doc). A separate `train_compile_blocks` gate for Slice 2.
- Extend `training_compile_readiness` reasons so every blocker above is
  visible in the plan log line (`compile_ready=N/M`): e.g.
  `trainable_base_weight`, `conv_layer`, `lora_untraceable`.

### Slice 1 — grad-safe compiled FP8 linear (core enabler)

- `_fp8_linear_training(x, qdata_t, scale_row, bias)`: forward = the
  existing `_fp8_linear_compiled` math; backward = grad-input via the
  existing scale-folding trick in `_fp8_grad_input` (fold row scales into
  `grad_out`, quantize, `_scaled_mm` against raw qdata) — pure tensor math
  both ways, no weight grad (frozen base). Package as a traceable
  autograd.Function (torch 2.12 traces these when the body is clean) or a
  custom op with `register_autograd` if Dynamo balks.
- Set `_memory_management_training_compile_fp8` on qualifying layers so the
  existing mmdit gate flips.
- Tests: fwd/bwd parity vs eager dequant path; `fullgraph=True` compiles
  with zero breaks; grad-input error within FP8 tolerance of bf16 reference.

### Slice 2 — training compile for pinned RESIDENT blocks (first shippable win)

No streaming involved — isolates compile x checkpoint x LoRA x FP8:

- Compile pinned-resident blocks for training (per-block, like the sampler's
  `_compiled_blocks`, but grad-enabled and composed with non-reentrant
  checkpoint — torch 2.12 handles the checkpoint HOP; this also retires the
  mmdit header comment about compile fighting checkpointing).
- Compile-clean LoRA fast path: when active with a scalar multiplier and no
  DoRA/interleave, the wrapper forward reduces to
  `org_forward(x) + (x @ A.T @ B.T) * m` — pure math. Branch selection at
  install time (like `fp8_sampling_qualifies`), not per call.
- Rebuild/invalidate on residency change: reuse the sampler's fingerprint
  mechanism; `_restore_offload` and any promote/demote must drop stale
  training graphs (same rule as the sampler's demotion note in `manager.py`).
- Tests: loss/grad parity compiled-vs-eager on a synthetic 2-block model
  with LoRA + FP8 base; zero graph breaks; recompile count stable across
  steps.

### Slice 3 — compiled sampling for STREAMED blocks

- Move staging out of hooks: `mmdit.forward` calls an eager
  `stage_block(i)` (Block Stream Slice 2 machinery; 2-block GPU ring)
  before invoking the shared compiled block callable with that block's
  staged leaves as inputs.
- Keep leaves raw (fp8 qdata_t + scale staged directly — no dequant
  needed on the fp8-native path; float models stage bf16).
- `enable_compiled_sampling` reports `eager_count == 0` under full offload.
- Tests: output parity vs today's eager streamed sampler; one H2D per block
  (reuse `bench_block_stream_forward` counters); zero breaks; recompile
  count flat across a multi-image session.

### Slice 4 — compiled training for STREAMED blocks (the headline)

- Checkpoint region = `checkpoint(lambda *a: compiled_block(*a,
  *stage_block(i)), x, tvec, freqs, mask, use_reentrant=False)` — recompute
  re-stages, backward reads fresh leaves.
- Bounce-pool/prefetch trace: the per-step fetch sequence becomes
  forward-stage + recompute-stage per block (eager checkpointing already
  produces a fetch at recompute, so the trace shape is familiar); align the
  pool cursor on block keys (Slice 1 fill groups are already block-sized).
- Constraint: streamed compiled blocks are ALWAYS checkpointed. `keep_last`
  (uncheckpointed trailing blocks) applies only to resident blocks; enforce
  in the planner, surface in readiness reasons.
- Scope guard: frozen (quantized) base weights only. Full fine-tune of
  streamed floats (grad D2H staging) stays eager — readiness reason
  `trainable_base_weight`. Convs stay eager/resident (Krea2 streamed blocks
  are Linear-only).
- Tests: end-to-end grad parity (pattern from
  `tests/test_block_forward_stage.py` gradient-parity test); LoRA grads
  match eager; ring peak bytes unchanged; zero breaks over fwd+bwd
  (`torch._dynamo.explain` over a full step of the synthetic model).

### Slice 5 — lifecycle hardening

- **Autotune interaction**: every working-reserve / keep_last resize changes
  residency layout => compiled graphs and prefetch both die. Options
  (decide in-slice): compile only after controllers settle (N-step warmup),
  or freeze resizes once compile engages. This mirrors the existing
  "no-resize stable band" goal — compile just raises the price of a resize.
- **Sampler/training transitions** (`_restore_offload`): assert both
  compiled sets rebuild; extend the demotion invalidation to training
  graphs.
- **Shape policy**: training buckets => accept per-bucket recompile with a
  cap + log line; sampler keeps pad-to-256.
- **Compile latency budget**: measure cold-start (Inductor/Triton on
  Windows) per graph; the shared-callable design means ~1 graph per shape
  bucket, not 28. Record numbers in the ticket.

## Acceptance criteria

- `torch._dynamo.explain` over one full training step (offload on, LoRA on,
  FP8 base): **0 graph breaks** in transformer blocks; streamed and resident
  blocks both compiled.
- Parity: sampler images bit-close to eager; training loss/grad parity
  within FP8/GEMM noise on synthetic tests.
- Perf: step-time improvement on a real Krea2 run measured with
  `scripts/digest_perf_log.py` (full run needs user go-ahead). If compiled
  streamed blocks do NOT beat eager streaming (transfer-bound regime), the
  flag stays off and the plan gets a RESULT section saying so — same
  discipline as Block Stream Slice 2.
- All existing `tests/` pass with flags off; new tests cover each slice.

## Theoretical endpoint: fully in-graph streaming ("just a normal model")

> Now a concrete plan: see `INGRAPH_STREAM_PLAN.md`. Slices 1-2 of THIS
> plan are shared prerequisites; slices 3-4 are superseded by that plan if
> its Phase 0 spike passes.

Design B keeps data movement eager between compiled blocks. There is a
principled path further, where torch.compile sees one whole-model graph and
streaming is invisible — the same trajectory FSDP took from hook-driven
(graph-breaking) FSDP1 to SimpleFSDP/compiled-FSDP2, where the pre-forward
all-gather is a functional op in the graph. An H2D fetch from pinned host
memory is the same op class as an all-gather: async transfer producing the
weight, wants early launch, re-issuable in backward. Four pillars:

1. **Weights as plain pinned CPU tensors** the model openly references (no
   subclass, no forward hijack). `F.linear(x, dequant(w.to('cuda')))` is
   already a legal graph — correct but unoverlapped.
2. **Split-phase fetch ops** `fetch_start(w) -> token` /
   `fetch_wait(token) -> w_gpu`, mirroring funcol collectives: transfer
   stream + event inside the op, Inductor codegen stays single-stream.
3. **Backward re-staging via the partitioner**: tag fetch ops
   MUST_RECOMPUTE (selective activation checkpointing policy) so the
   backward graph contains its own fetches; no weight saved, no
   hand-rolled checkpoint workaround.
4. **Prefetch as an Inductor scheduling pass**: hoist fetch_starts under a
   VRAM budget (extend/imitate reorder_for_compute_comm_overlap +
   peak-memory reordering via post_grad_custom_post_pass). The 2-block ring
   emerges from buffer planning; working reserve becomes a compile-time
   scheduling constraint.

Endpoint payoffs: zero breaks fwd+bwd in one graph; fusion across former op
boundaries; and cudaMemcpyAsync from *pinned* host is CUDA-graph
capturable, so reduce-overhead mode is reachable **iff all streamed weights
are pre-pinned** (the pageable->pinned bounce worker is not capturable) —
the DXGI pin-for-speed effort is the enabler.

Known costs: the scheduling pass is compiler-extension work against
churning Inductor internals (on triton-windows); prefetch depth baked at
compile time means every autotune resize is a recompile (freeze controllers
first); host-side ledgers/traces move inside op impls; trainable streamed
floats need a symmetric put_start/put_wait D2H pair. UVM/managed memory
("let the driver stream") is NOT a path on Windows/WDDM — no concurrent
managed access or oversubscription, and the WDDM paging cliff is the
measured proof.

Design B is this decomposition with a human scheduler: raw-leaf unpacking
(pillar 1), block staging with event semantics (pillar 2), checkpoint
recompute (pillar 3 approximated). Nothing gets thrown away. Worth an early
standalone spike: fetch-as-functional-op + MUST_RECOMPUTE tag on a
synthetic 2-block model, no manager integration, to confirm the partitioner
re-fetches in backward — the only pillar resting on young machinery.

## Risks / open questions

- **Recompute cost**: checkpoint recompute already pays a second forward;
  compiled recompute should be cheaper than eager recompute, but staging
  happens twice per block per step regardless (as today).
- **AOTAutograd saving extra intermediates** inside the compiled block
  (attn weights etc.) — verify with memory snapshots that the checkpoint
  HOP recomputes instead of saving; else VRAM regresses.
- **Inductor reordering vs pool cursor**: within a block, q/k/v GEMMs are
  data-independent; with leaves staged per-block up front this no longer
  matters (ordering only matters at block granularity). Confirm no
  per-Linear fetch remains on the compiled path.
- **Windows/Triton stability** under long training sessions (the sampler
  compile is session-scoped; training graphs live for days).
- **Guard churn from `tvec`/`mask`**: `dynamic=False` initially; revisit.
