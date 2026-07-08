# In-Graph Guard-Free Leaves Plan

> Status lives in git-bug ticket `fdc66d8`.
> Related: `INGRAPH_STREAM_PLAN.md` (weights-as-inputs guard-free rule),
> `INGRAPH_PHASE4A_TRAINING_PLAN.md` (compiled fully-streamed training),
> and `COMPILE_MEGA_CACHE_PLAN.md` (single stable trunk is required for the
> cache to matter across runs).

## Problem

The strict fullgraph Krea2 trunk can report `graph_breaks: 0` while still
re-specializing many times. The observed run produced `unique_graphs: 28` and
`async_compile_cache_miss: 336`: these are clean fullgraph recompiles, not
graph breaks.

The debug guards identify two Python-metadata sources:

- `leaf_view(flat, spec)` guards on `LeafSpec.dtype`, `LeafSpec.offset`, and
  `LeafSpec.shape[0]`.
- Block forward guards on the live `leaves` dict type/id.

Real Krea2 has heterogeneous per-linear specs (attention q/k/v/out vs MLP
gate/up/down shapes, offsets, and possibly dtypes). The existing synthetic
stability tests use uniform shapes, so all calls collapse to one
specialization and miss the defect.

The earlier all-28 fully-streamed run already reported `unique_graphs: 21`, so
the single-stable-trunk ideal was never achieved. The defect was hidden because
`graph_breaks: 0` was treated as sufficient; it is only the entry criterion.
`unique_graphs` and compile cache misses are the counters that reveal guard
churn after the graph remains fullgraph-clean.

## Goal

Deliver a Krea2 sampling mode where smart partial offload compiles every
transformer block through exactly one compiled path in the same run:

- resident blocks run through regional resident `torch.compile` blocks;
- streamed blocks run through in-graph streamed `torch.compile` blocks with
  fetch ops and tensor-only weight views;
- no transformer block falls through to eager in the accepted mixed layout;
- all-stream runs remain useful diagnostics, but they do not close this ticket
  because they do not prove resident compile and stream compile coexist.

The guard-free leaves work is still required for streamed blocks, but the
acceptance target is no longer just "one stable fullgraph trunk." The target is
mixed-mode coverage: 100% resident compile plus 100% stream compile at the same
time.

## Acceptance Criterion

Acceptance requires one real Krea2 sampling smoke with a mixed layout, meaning
both counts are nonzero:

- `resident_expected_blocks > 0`
- `stream_expected_blocks > 0`

That single run must report all of the following:

- `compile.path == "mixed_resident_ingraph_compile"`
- `resident_compile_complete == true`
- `regional_compiled_blocks == resident_expected_blocks`
- `stream_compile_complete == true`
- `ingraph_compiled_blocks == ingraph_packs`
- `resident_expected_blocks + stream_expected_blocks == total_blocks`
- `graph_breaks == 0`
- no `partial_ingraph_compile` unavailable reason
- no resident-block recompiles caused by `_forward_pre_hooks`,
  `record_weight_access`, `_layer_memory_manager`, or block-specific hook
  defaults such as `blocks.N.mlp.up`
- no streamed-block recompiles caused by `LeafSpec`, `LinearSpec`,
  `LinearView`, `TrainLeaf`, or live `leaves` dict guards
- fetch count remains exactly one H2D fetch per streamed block per streamed
  pass

Aggregate `unique_graphs` in mixed mode is not expected to be 1, because the
accepted path intentionally contains multiple compiled regions. The stability
requirement is that the graph count stops growing after warmup/repack for the
same shape bucket, and any remaining graph floor is explained by true region or
shape differences rather than Python metadata guard churn.

A run with only `ingraph_fullgraph_all`, only `regional_compile`, only
`ingraph_block_compile`, or `ingraph_partial_eager` is not accepted for this
ticket.

## Current Findings

The latest smart partial smoke regressed to:

- `ingraph_compiled: false`
- `ingraph_packs: 15`
- `regional_compiled_blocks: 0`
- `path: ingraph_unavailable`
- `unavailable_reasons: ["partial_ingraph_compile"]`

That is strictly worse than the earlier cache-miss failure because it does not
compile the streamed side at all and it also leaves the resident side uncompiled.

Targeted log search of `C:\GenAI\Windows PowerShel4l.txt` found resident
regional recompiles coming from memory-manager hooks captured inside compiled
resident blocks:

- stack includes `manager.py::_pre_hook`
- stack includes `manager_modules.py::record_weight_access`
- guard examples compare `_forward_pre_hooks` defaults like
  `blocks.16.mlp.up`, `blocks.17.mlp.up`, etc.

So there are two independent compile defects to close:

- streamed compile must be tensor-only and must not fall back to partial eager;
- resident compile must not capture any streaming/access hooks.

## Design

### 1. Treat mixed compile as the primary product path

Smart partial sampling must enable both compile systems:

- in-graph compile for the streamed block set;
- regional resident compile for every non-streamed block.

Strict in-graph mode should fail if streamed compile cannot be enabled, but it
must not suppress resident compilation for the blocks that are actually
resident. The accepted runtime dispatch order is:

1. resident compiled block when the block is resident;
2. streamed in-graph compiled block when the block is streamed;
3. hard diagnostic failure for acceptance runs if either side would fall back
   eager.

### 2. Split layout metadata from streamed tensor views

Keep `BlockPack` and `LeafSpec` as host-side layout records only. They may be
used at attach/enable time to build view-maker callables, but compiled block
math must not receive them.

The streamed compiled path must pass concrete tensors positionally:

- sampling linear args: `(weight, bias, scale_or_none)`;
- training linear args: `(weight, bias, scale_or_none, lora_a, lora_b,
  lora_scale_or_none)`;
- block args in fixed Krea2 order:
  `attn_wq, attn_wk, attn_wv, attn_gate, attn_wo, mlp_gate, mlp_up,
  mlp_down`.

Avoid dictionaries and dataclass dispatch in the compiled path. The legacy
`leaves` dict path can remain compatibility-only, but it cannot be used by the
accepted compiled trunk.

### 3. Generate metadata-free view makers

For each `BlockPack`, build a view maker at enable time:

```text
flat -> tuple(weight, bias, scale, ...)
```

The maker should bake offsets, byte counts, dtypes, and shapes into closure
constants and return only tensors. It must not call `leaf_view(flat, spec)` with
a live `LeafSpec` object inside a compiled frame.

This preserves the fetch contract where `fetch_wait` returns one flat device
buffer while removing Python metadata from the traced block boundary.

### 4. Make Krea2 streamed block forward positional

Add internal positional leaf entry points beside the existing eager-compatible
`leaves=None` forwards:

- `SwiGLU.forward_streamed(x, gate, up, down)`
- `Attention.forward_streamed(qkv, freqs, mask, wq, wk, wv, gate, wo)`
- `SingleStreamBlock.forward_streamed(x, vec, freqs, mask, *leaf_args)`

The compiled stream path calls these methods directly. This removes the live
`leaves` dict and its `___check_type_id` guard.

Training should use a parallel positional path that passes LoRA tensors as
ordinary tensor args. LoRA absence/presence is decided at enable time; if a
block has mixed LoRA presence, use a pre-selected helper per linear rather than
checking Python containers inside the graph.

### 5. Make resident compile hook-clean

Resident regional compile must only compile blocks whose child modules are free
of streaming/access hooks, memory-manager attributes, and forward hijacks. The
log shows that checking only `_layer_memory_manager` is insufficient.

Required resident-side rule:

- before compiling a resident block, verify all children have no
  `_forward_pre_hooks`, `_forward_hooks`, `_forward_hooks_with_kwargs`,
  `_layer_memory_manager`, or nonstandard `forward` hijacks;
- if resident mode is supposed to make a block compileable, detach those hooks
  for the duration of the sampling run rather than compiling through them;
- after sampling, restore any state that must exist outside the sampling
  resident context;
- diagnostics must name any hook-contaminated block instead of silently
  compiling a guard-churning graph.

## Implementation Slices

### S0 - Acceptance instrumentation

- Preserve `ingraph_compiled_blocks` in Krea2's post-sampling compile-state
  snapshot before `disable_ingraph_sampling()` clears temporary state.
- Extend smoke diagnostics with:
  - `resident_expected_blocks`
  - `stream_expected_blocks`
  - `resident_compile_complete`
  - `stream_compile_complete`
  - `mixed_compile_complete`
- Classify the accepted path as `mixed_resident_ingraph_compile` only when both
  expected sides are complete.
- Make partial eager and missing resident compile warnings explicit acceptance
  failures in smoke output.

### S1 - Streamed compile coverage

- For partial streamed layouts, compile each streamed block with a fullgraph
  in-graph function rather than declaring `partial_ingraph_compile`
  unavailable.
- Preserve all-stream full-trunk compile as a diagnostic mode, but do not use it
  as the acceptance proof.
- Keep the fetch/free order unchanged:
  `fetch_start_after -> fetch_wait -> positional streamed block -> fetch_free_after`.
- Add a focused test proving the positional streamed path compiles fullgraph
  without a live `leaves` dict.

### S2 - Guard-free streamed metadata

- Add pack-time view-maker construction in
  `toolkit/memory_management/ingraph_stream.py`.
- Preserve `block_linear_views` for eager/reference tests only.
- Add a heterogeneous Krea2-like stability test that repacks the flat host
  buffer and verifies no new graphs are created for unchanged tensor shapes.
- Inspect guards after the real smoke and confirm no `LeafSpec`/dict/type-id
  guards remain on streamed blocks.

### S3 - Resident hook cleanup

- Extend resident compile safety checks to reject any child module with forward
  hooks, memory-manager attributes, or forward hijacks, not just
  `_layer_memory_manager`.
- Identify why blocks reported as resident still retain `_forward_pre_hooks` in
  the smoke log.
- Ensure the sampling resident context leaves resident blocks hook-free for the
  full compiled sampling run, or add a compile/run wrapper that bypasses hookful
  module `__call__` paths safely.
- Add diagnostics listing hook-contaminated resident blocks by index and child
  path.
- Add a unit/synthetic test that a hook-contaminated resident block is not
  counted as compiled, then update it to pass once cleanup is in place.

### S4 - Mixed runtime enforcement

- In `_blocks_trunk`, keep dispatch ordered as resident compiled first,
  streamed compiled second.
- Add an acceptance/debug assertion path for Krea2 smoke: if `compile_sample` and
  streamed compile are requested, every block must be accounted for by exactly
  one compiled path.
- Make strict in-graph mode mean strict streamed compile, not "skip resident
  compile."
- Keep runtime fallback behavior for non-acceptance user jobs conservative, but
  ensure smoke diagnostics expose any eager fallback.

### S5 - Real Krea2 validation

- Run the mixed smart partial Krea2 sampling smoke, not only all-stream.
- Confirm the smoke JSON satisfies every acceptance field above.
- Search the recompile log for these forbidden resident-side sources:
  `_forward_pre_hooks`, `record_weight_access`, `_layer_memory_manager`, and
  block-specific hook defaults.
- Search the recompile log for these forbidden streamed-side sources:
  `LeafSpec`, `LinearSpec`, `LinearView`, `TrainLeaf`, and `leaves` dict/type-id
  guards.
- Re-run after repack/cache warmup and confirm graph count does not continue to
  grow for the same shape bucket.
- Record the final smoke JSON and guard-log summary in ticket `fdc66d8`.
## Validation Commands

```powershell
venv\Scripts\python.exe -m pytest tests\test_ingraph_stability.py -q
venv\Scripts\python.exe -m pytest tests\test_ingraph_training_leaves.py -q
venv\Scripts\python.exe -m pytest tests\test_ingraph_training_ops.py -q
venv\Scripts\python.exe -m py_compile toolkit\memory_management\ingraph_stream.py extensions_built_in\diffusion_models\krea2\src\mmdit.py
```

Full Krea2 smokes are GPU/runtime validation and should be started only when
the user asks, per repo convention.

## Risks

| Risk | Mitigation |
|---|---|
| View maker still becomes a per-block Python specialization | Test heterogeneous specs and inspect guard sources before accepting the change |
| Positional APIs become brittle because Krea2 linear order is implicit | Centralize the order next to `_block_linear_entries` and assert pack names match |
| LoRA optionality reintroduces Python guards | Pre-select per-linear helper variants at enable time; pass LoRA tensors positionally |
| Training checkpoint HOP has a true graph-count floor | Build a minimal HOP reproducer and document the floor separately from metadata churn |
| Compatibility dict path diverges | Keep parity tests through both paths until the compiled trunk fully migrates |
