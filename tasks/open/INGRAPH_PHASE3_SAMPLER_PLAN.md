# In-Graph Phase 3 — Fully Streamed Compiled Sampling (execution plan)

> Durable plan. Status lives in git-bug `3ca8a7b`, not here.
> Parent: `INGRAPH_STREAM_PLAN.md` (this document is the execution plan for
> its Phase 3, re-scoped after the first roadblock). Incorporates the
> sampling-only proposal reviewed 2026-07 (fail-closed smoke, packing before
> compile) with corrections from the Phase 0/0b spike findings.

## Goal (self-contained)

The retired legacy `scripts/smoke_krea2_ingraph_cuda.py` produced a correct
sample from the
real fp8 Krea2 model where **every streamed transformer block runs inside
one `torch.compile(fullgraph=True)` region**, weights entering exclusively
through `mm::fetch_start -> mm::fetch_wait -> in-graph views`. No regional
compile, no MemoryManager hooks or forward hijacks inside the region, no
TorchAO wrapper as a graph input, no silent fallback of any kind. Current
immutable-runtime inference validation uses
`scripts/smoke_krea2_inference_cuda.py`; do not restore the retired harness.

Training compile, SAC backward re-fetch, LoRA gradients, Tier-2 scheduling,
and CUDA graphs are explicitly out of scope (parent plan Phases 4+).

## Where we are / why the smoke could not run

Implemented (Phases 1-2 slices): `toolkit/memory_management/ingraph_stream.py`
(`BlockPack`/`LeafSpec`/`LinearSpec`, `pack_block_host`, `fetch_start/wait/free`
custom ops with ticket ring + depth guard + stats/report), and in `mmdit.py`
`enable_ingraph_sampling()` / `_start_ingraph_until()` token plumbing with a
streamed-set fingerprint.

The roadblock, in order of severity:

1. **`pack_block_host` rejects quantized wrappers** (`ingraph_stream.py:143`
   raises "wrapper packing is deferred to Phase 6"). The parent plan's
   Phase 1 always specified TorchAO leaf packing via `_flatten_leaves`
   (and the 0b scale spike proved it against the real model); the
   implementation under-delivered. Result: the real fp8 model produces no
   packs, so ingraph was requested but never active.
2. **The smoke fails open.** When packs are unavailable it silently falls
   back to `enable_compiled_sampling()` regional compile, which then
   recompiled on legacy hook identity (`blocks.N.mlp.up` ...). The trace
   looked like an ingraph run and was not.
3. **Wrong compile-region scope.** `_compiled_ingraph_sampling` compiles the
   whole `mmdit.forward`, so the resident quantized preamble/tail (`first`,
   `tmlp`, `txtfusion`, `txtmlp`, `last`) sits inside the region and their
   TorchAO wrapper params become graph inputs — the exact subclass-in-graph
   failure the design exists to avoid.

## Design decisions (fixed for this phase)

- **Fail closed everywhere.** Requested-but-unavailable ingraph raises with
  a machine-readable reason; no path falls back to regional compile or
  eager silently. Matches the repo's fail-fast convention.
- **Compile region = the blocks trunk only.** `first`/`tmlp`/`txtfusion`/
  `txtmlp`/`posemb`/`last` stay eager outside the region this phase. They
  are cheap relative to 28 blocks, and excluding them is what makes
  "no wrapper graph inputs" achievable without also unpacking resident
  modules (that refinement can come later, `_enable_fp8_sampling`-style).
- **Raw `qdata` layout in the pack; `.t()` as an in-graph view.** The pack
  preserves the wrapper's storage layout (required for state_dict
  repointing); `_scaled_mm` accepts the transposed view directly — no
  transposed copy at pack time.
- **FP8-native compute where it qualifies; in-graph dequant otherwise.**
  Reuse the forward-only `_fp8_linear_compiled` math (already
  compile-clean) with qualification hoisted to pack time
  (`fp8_sampling_qualifies` conditions). Non-qualifying layers get
  in-graph `dequant -> F.linear` (Inductor plans/fuses the dequant).
- **No LoRA in this phase's smoke.** The clean-path work is a training
  concern; the region audit (S2) will reject a hijacked forward anyway.
- **No K-ahead pipelining until the simple schedule is correct** (Tier-0
  host run-ahead is free per the Phase 0 spike; K-ahead is a measured
  add-on in S8).
- **Buffer lifetime is ring-governed and test-enforced.** The Phase 0 spike
  produced silent wrong numerics when a transfer-stream-allocated buffer
  was recycled while compute read it. The ticket ring + `fetch_free` /
  depth guard already exists; this phase adds the regression test that
  hammers reuse (S7). This invariant outranks every performance idea.

## Work items

Each lands with tests (GPU, seconds-scale, under `tests/`); the smoke run
is the only minutes-scale step and concludes the phase.

### S1 — Fail-closed smoke + reason taxonomy

- `enable_ingraph_sampling()` (or its caller) computes
  `ingraph_unavailable_reasons: list[str]` from a fixed vocabulary:
  `wrapper_pack_missing`, `non_pinned_pack`, `unsupported_quant_wrapper`,
  `hook_present`, `forward_hijack_present`, `legacy_layer_manager_present`,
  `dynamic_streamed_block_set`. Empty list == ingraph active.
- Smoke gains `--strict-ingraph` (default ON for this phase): any nonempty
  reasons -> `RuntimeError` listing them, before any compile starts. The
  regional-compile fallback branch is removed from the ingraph smoke path.
- Post-attach asserts in the smoke: `transformer._compiled_blocks is None`
  (no regional artifacts), `_ingraph_sampling_packs` covers exactly the
  planned streamed set, `MemoryManager.ingraph_fetch_report()` available.

### S2 — Compile-region audit (hooks AND hijacks)

- `ingraph_stream.assert_compile_region_clean(module)`: walks the trunk and
  rejects (a) `_layer_memory_manager` attributes, (b) nonempty
  `_forward_pre_hooks`/`_forward_hooks`/`_forward_hooks_with_kwargs`,
  (c) **instance-attribute forward replacement** — `'forward' in
  submodule.__dict__` — which is how legacy streaming, LoRA, and the fp8
  sampling installer actually attach (hook checks alone pass a hijacked
  Linear straight into the tracer).
- Called by `enable_ingraph_sampling()` (fail closed -> `hook_present` /
  `forward_hijack_present`) and directly by tests.
- Test `tests/test_ingraph_region_audit.py`: clean synthetic trunk passes;
  attach legacy streaming -> rejected; install a LoRA-style forward swap ->
  rejected; fp8-sampling installer on an in-region module -> rejected.

### S3 — Quantized wrapper packing (finish parent Phase 1 scope)

- Remove the plain-Linear restriction in `pack_block_host`. Quantized
  weights flatten via `_flatten_leaves` to (qdata, scale) leaves; `LeafSpec`
  records role (`qdata`/`scale`/`bias`/`float_weight`); `LinearSpec` gains
  `kind: "float" | "fp8_rowwise"` plus the pack-time fp8 qualification
  verdict (S4). Unknown wrapper layouts (leaf count != 2, non-fp8 qdata)
  -> `unsupported_quant_wrapper`, fail closed.
- `repoint=True` rebuilds the wrapper onto host-flat views
  (`_rebuild_from_leaves`) so no duplicate full-size CPU storage remains
  and `state_dict()` still sees intact wrappers.
- Test `tests/test_ingraph_pack.py` additions: fp8 wrapper round-trip
  (bitwise state_dict before/after repoint), leaf views' `data_ptr` inside
  host_flat storage, pinned flag, dequant-from-views bitwise vs
  `_dequantize_to` (the 0b spike check, made permanent), reject-path for
  exotic wrappers.

### S4 — In-graph streamed linear compute

- `ingraph_stream.streamed_linear(x, spec_views) -> Tensor`, pure traced
  math, two paths selected by the pack-time verdict (a trace-time constant,
  not a data-dependent branch):
  - `fp8_rowwise` + qualifies: `_fp8_linear_compiled(x, qdata.t(), scale,
    bias)` (import the existing helper; do not fork the math).
  - otherwise: `w = qdata.to(compute_dtype) * scale_view; F.linear(...)`.
- Test `tests/test_ingraph_functional_linear.py`: parity vs
  `_dequantize_to`+`F.linear` and vs `fp8_linear_inference` on qualifying
  shapes; `fullgraph=True` traces both paths; no wrapper object anywhere
  in the traced graph inputs (inspect `torch._dynamo.explain` /
  gm.graph inputs).

### S5 — Block functional path + trunk scoping

- `SingleStreamBlock.forward(..., leaves=None)`: when `leaves` is passed,
  every streamed Linear computes via `streamed_linear` from the passed
  views and **must not touch module weight params**; `None` preserves
  legacy behavior bit-for-bit.
- Re-scope the compiled artifact: `_compiled_ingraph_sampling` becomes a
  compiled **blocks trunk** `trunk(x, tvec, freqs, mask, packs) -> x`
  (fetch_start/wait + `block_linear_views` + block-with-leaves per streamed
  block; resident blocks called with `leaves=None` inside the same region
  only if they are wrapper-free, else kept streamed). `mmdit.forward` in
  ingraph mode runs preamble eager -> compiled trunk -> tail eager.
- Fingerprint: (streamed-index tuple, depth, seq-bucket) -> rebuild on
  change; reuse the existing `_compiled_ingraph_fingerprint`.
- Test `tests/test_ingraph_sampler.py` extensions (synthetic multi-block
  model with real TorchAO fp8 weights): trunk output parity vs legacy
  eager streamed forward (tolerance; compiled-vs-eager is NOT bitwise —
  Phase 0 finding), zero graph breaks, region audit green inside trunk.

### S6 — Wire the real model path

- `enable_ingraph_sampling()` packs all streamed Krea2 blocks (S3), runs
  the audit (S2), compiles the trunk (S5). The streamed set comes from the
  existing sampling residency planner; for the smoke default, stream ALL
  28 blocks (the "fully streamed" goal — residency optimization is not
  this phase's job).
- `ModelConfig.layer_offloading_ingraph_sampling: bool = False` +
  `layer_offloading_ingraph_depth: int = 2` wired through to the smoke
  (config flag, not env var; UI schema entry follows in parent Phase 6).

### S7 — Stability and safety proofs (the two spike lessons, made tests)

- Guard stability (`tests/test_ingraph_guard_stability.py`):
  - same-shape second call -> zero new compiles
    (`torch._dynamo.utils.counters` frame counts flat);
  - **repack**: rebuild every `BlockPack` (new host tensors, same values)
    -> zero new compiles — weights-as-inputs must not guard on tensor
    identity; this is the property the whole design leans on;
  - `torch._dynamo.explain` on the trunk: `graph_break_count == 0`.
- Buffer lifetime (`tests/test_ingraph_buffer_reuse.py`): depth=2, many
  consecutive passes with per-pass output parity against a fresh eager
  reference — regression for the cross-stream recycle corruption; assert
  live tickets never exceed depth and `fetch_stats` ring peak == depth.
- `torch.library.opcheck` over `fetch_start`/`fetch_wait`/`fetch_free`
  with representative inputs (schema/aliasing contract).

### S8 — Smoke run + measurement (correctness first, perf recorded not gated)

- Strict smoke on the real model: one denoise pass must show
  `fetches == streamed_block_count` (exactly one H2D per streamed block per
  pass) in `ingraph_fetch_report`, then a full sample decoded to PNG.
- Record (perf-log fields + smoke JSON): `ingraph_fetches`,
  `ingraph_h2d_ms`, `ingraph_wait_ms`, trunk ms/pass, total sample s,
  peak VRAM, compile time (cold + warm process).
- Baselines for the record: legacy eager streamed sampling; regional
  compile partial-resident sampling. **Acceptance does not require a perf
  win** — it requires the path compile-clean and correct; if it loses,
  that goes in the ticket and the flag stays off (parent-plan discipline).
- Optional, after all green: K-ahead source pipelining behind
  `--pipeline-k`, measured K=0/1/2. No Inductor pass in this phase.

## Definition of done

1. Smoke runs with ingraph sampling active on real fp8 Krea2; strict mode
   never falls back (regional compile provably not entered:
   `_compiled_blocks is None` throughout).
2. The blocks trunk is one `torch.compile(fullgraph=True, dynamic=False)`
   artifact; `torch._dynamo.explain` reports zero graph breaks.
3. Region audit green: no `_layer_memory_manager`, no module hooks, and no
   instance-attribute `forward` hijacks inside the compiled region.
4. No `_BouncingLinearFn` executes during the sample (fetch counters + no
   legacy per-Linear staging in the perf trace).
5. No TorchAO wrapper object is a graph input; streamed weights enter only
   via `mm::fetch_start`/`mm::fetch_wait` and in-graph views.
6. Exactly one H2D fetch per streamed block per denoise pass (before any
   pipelining); live fetched buffers bounded by `depth` at all times.
7. Same-shape repeat sampling and full repack both cause zero recompiles.
8. Latent/image parity vs legacy eager streamed sampling within the
   documented tolerance (not bitwise; compiled-vs-eager differs by
   Inductor epilogue rounding).
9. Repointed packs preserve `state_dict()` bitwise; no duplicate full-size
   CPU copies of packed leaves remain.
10. Buffer-reuse regression test (S7) passes at depth=2 over many passes.
11. All flags off -> zero behavior change; existing `tests/` suite green.

## Explicitly deferred (parent plan Phases 4+)

SAC / MUST_RECOMPUTE, training joint graph, backward re-fetch,
`_fp8_linear_training`, LoRA compile-clean path and gradients, optimizer
compile, resident-module unpacking to extend the region, Tier-2 Inductor
scheduling pass, CUDA graph capture, UI exposure beyond the config flag.

## Risks specific to this phase

| Risk | Mitigation |
|---|---|
| Unrolled 28-block trunk compile time on triton-windows | measured in S8 cold/warm; fingerprint keeps it once per layout; if pathological, fall back to compiling a shared per-block callable (same architecture, more graphs) and record |
| Resident-block params inside trunk force wrapper inputs | this phase streams all 28 blocks by default; wrapper-free resident blocks may join the region, wrapper-bearing ones may not (audit enforces) |
| `.t()` view of fetched qdata rejected by `_scaled_mm` on some shapes | pack-time qualification (S4) decides per layer; disqualified layers take the dequant path, counted and reported |
| Seq-len variation across denoise steps causing recompiles | sampler shapes are constant across steps for one image; fingerprint includes the seq bucket; multi-resolution handling stays in parent Phase 5 |
