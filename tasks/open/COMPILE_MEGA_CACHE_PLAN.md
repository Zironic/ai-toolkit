# torch.compile Mega-Cache (Inductor/AOTAutograd persistence) — Plan

> Durable plan. Mutable status (what's done / blocked) belongs in a git-bug
> ticket, not here. Related: `COMPILE_STREAMED_OFFLOAD_PLAN.md` (Slice 5's
> sampler/training transition hardening is exactly where the training half
> of this hooks in), `INGRAPH_PHASE3_SAMPLER_PLAN.md` (measures cold/warm
> compile time; the mega-cache is how "warm" survives a process restart).

## Goal

Persist torch.compile's Inductor/AOTAutograd/Triton caches to disk across
process restarts -- and, for training, across the compile teardown/rebuild
cycles that already happen *within* one run -- via
`torch.compiler.save_cache_artifacts()` / `load_cache_artifacts()`. Cold
compiles are the dominant "first preview / first step after a sampling round
is slow" cost on Windows/triton-windows.

The default on-disk Inductor cache (`FXGraphCache`/`AOTAutogradCache`) is not
usable as the *primary* persistence mechanism here: its location is
`TORCHINDUCTOR_CACHE_DIR`, an env var, and this repo's Training Configuration
Rule forbids env-var-governed runtime behavior. The mega-cache API is a plain
function call keyed off a config field (`ModelConfig.compile_cache_dir`), so
it's the compliant path.

## Shipped: sampling (Krea2 `compile_sample`)

- `toolkit/compile_cache.py`: `load_compile_cache`/`save_compile_cache`,
  atomic write, safe-miss-on-missing/stale blob (torch's own guards reject a
  mismatched blob; no validation needed on our side).
- `ModelConfig.compile_cache_dir` (default `None` = disabled).
- `krea2.py` `generate_single_image`: loads once per process before the
  first compiled sample; saves whenever
  `torch._dynamo.utils.counters['frames']['total']` increases across the
  `pipeline(...)` call -- i.e. an actual new Dynamo compile happened (covers
  both the first compile and any later dynamic-shape upgrade compile).
- `enable_compiled_sampling()` (`mmdit.py`) compiles with `dynamic=None`
  (torch's automatic-dynamic-shapes mode); the call is wrapped in
  `torch.compiler.set_stance("eager_then_compile")` (scoped to just that
  call, not process-global) so the first sample runs eager and the
  static-vs-dynamic decision comes from real shape history instead of
  wasting a static compile on the very first shape seen.
- Cache key: `sha256({resolved checkpoint_path, qtype, torch_version})` --
  no resolution/shape tag needed, since guards (not us) discriminate shapes
  within one cumulative blob. One blob per (checkpoint, quant) is enough.

## MEASURED 2026-07-05: what the mega-cache can and cannot buy

Two findings from wiring the blob into the Phase 4a ingraph training smoke
(`smoke_krea2_train_cuda.py --compile-cache-dir`); they reshape this plan's
value calculus:

1. **Custom-pass cache poisoning (fixed, applies to SAMPLING too).**
   Inductor hashes `post_grad_custom_post_pass` into every fxgraph cache
   key. A bare closure is unpicklable, so torch salts the key PER PROCESS:
   with the ingraph ordering pass installed, every run looked like a
   different compiler and every cache tier missed -- including the shipped
   sampling mega-cache in any process that also enabled ingraph
   training/sampling ordering. Fixed in `ingraph_stream_scheduling.py`:
   the pass is a CustomGraphPass-shaped object whose `uuid()` is a content
   hash of the module source. **Standing rule: any custom Inductor pass in
   this repo must be a stable-uuid object, never a closure.**
2. **The ingraph TRAINING trunk's cold compile is not cacheable by this
   mechanism.** A/B with the stable uuid: cold step-0 151 s, warm step-0
   188 s (blob loaded successfully; warm being marginally slower is
   run-to-run variance in the tracing-bound regime, NOT the cache adding
   cost -- the signal is the absence of any drop). The cost is Dynamo
   tracing + AOTAutograd partitioning of 28 checkpoint-HOP units -- work
   upstream of every cache tier the blob stores (fxgraph/Triton/AOT
   artifacts).
   Implications:
   - Training mega-cache wiring stays (smoke has load/save; costs nothing)
     but its expected payoff at Krea2 scale is ~zero until torch caches
     tracing, so its priority drops accordingly.
   - The real cold-compile lever is trace-once via
     `torch.compiler.nested_compile_region` (identical blocks), which
     torch 2.12 REJECTS for training with mutating ops (our guarded
     `_after` fetch ops). The unlock is a mutation-free gated op set
     (`fetch_free_gated(token, gate) -> token` chained functionally) --
     parked in `INGRAPH_PHASE4A_TRAINING_PLAN.md`.
   - Multi-bucket training multiplies the ~150 s per shape bucket; the
     freeze-after-warmup / bucket-count policy in
     `INGRAPH_STREAM_PLAN.md` Phase 5 must budget with this number.

## Not yet done: training compile (`train_compile_blocks`, `enable_compiled_training`)

Training is a harder target than sampling for three compounding reasons, the
first of which is a **hard prerequisite, not yet true today**:

0. **Training compile currently never rebuilds after it's invalidated.**
   `enable_compiled_training()` is called from exactly one place in the
   repo -- a one-time setup block in `BaseSDTrainProcess.py` (~line 3149),
   run once at job start. `_invalidate_compiled_blocks()` (`manager.py:2125`)
   is called from `promote_layer`/`demote_layer` (`manager.py:2205`,
   `manager.py:2228`) on *any* residency change -- both the sampler
   transition (`MemoryManager.inference_resident()`, `manager.py:4577`) and
   plain training-time autotune resizes -- and nulls out
   `_compiled_training_blocks` via `disable_compiled_training()`. Nothing
   currently calls `enable_compiled_training()` again afterward, so today the
   first residency change after setup silently and permanently drops
   training back to eager for the rest of the job. This matches
   `COMPILE_STREAMED_OFFLOAD_PLAN.md` Slice 5's own open item ("assert both
   compiled sets rebuild") -- it's flagging exactly this gap as unfinished.
   **The mega-cache work below is meaningless until Slice 5 makes the
   rebuild actually happen** -- there's nothing to cache warm-vs-cold if the
   recompile never occurs in the first place. Sequence Slice 5's rebuild fix
   strictly before any of this section.
1. **Once rebuilding works, it happens around every residency change**, not
   just sampling: any working-reserve/keep_last autotune resize invalidates
   the same way (Slice 5 already notes this). Every one of those is a
   guaranteed training-compile teardown+rebuild -- a cold compile paid again
   unless something persists across it.
2. **"Super duper dynamic" shapes.** Training shapes vary far more than
   sampling: bucketed resolutions, LoRA rank/target modules, gradient-
   checkpoint on/off, `layer_offloading_checkpoint_keep_last`, fp8
   forward/grad-input gates, network multiplier -- each is a potential
   distinct guard/graph. The mega-cache blob will accumulate many more
   entries than the sampling one and churn more (`COMPILE_STREAMED_OFFLOAD_
   PLAN.md` Slice 5 already flags "every working-reserve/keep_last resize
   changes residency layout => compiled graphs... die").
3. **Needs its own cache key**, separate from the sampling key. Mixing them
   in one blob would not be *incorrect* (guards still discriminate) but it
   pointlessly bloats every save/load with unrelated graphs, and the two
   compile lifecycles have different lifetimes (sampling is session-scoped
   per `INGRAPH_PHASE3_SAMPLER_PLAN.md`; training graphs "live for days" per
   `COMPILE_STREAMED_OFFLOAD_PLAN.md`'s Windows/Triton-stability risk) and
   shouldn't share an invalidation unit.

### Design

- Reuse `toolkit/compile_cache.py` as-is -- it's already generic
  (`cache_dir` + opaque `key` string), no sampling-specific assumptions.
- New key, same base identity as the sampling key
  (`{checkpoint_path, qtype, torch_version}`) plus a `_training` suffix (vs
  sampling's own suffix), so the two never collide. No need to fold LoRA
  rank/bucket/checkpoint-mode/etc. into the key itself -- per the sampling
  design, torch's guards already discriminate individual graphs within one
  blob; the key only needs to identify "which lifecycle owns this blob," not
  every shape variant inside it.
- **Hook points, both inside `MemoryManager.inference_resident()`** (the one
  place both transitions already happen):
  - **On entry**, before/during the teardown that calls
    `_invalidate_compiled_blocks()` on the training layout: if a new training
    compile happened since the last save (same counter-delta pattern as
    sampling), save the training-key blob. This captures a training-only
    snapshot *before* any sampling compiles land in the process-wide cache.
  - **On exit** (`_restore_offload`, after the training layout is
    reattached, just before `enable_compiled_training()` runs again for the
    next step): load the training-key blob first.
    `enable_compiled_training()` then rebuilds its `torch.compile()`
    wrappers as it already does, but the guard/graph lookup should hit the
    loaded cache instead of a cold Inductor/Triton build.
  - Both are no-ops when `compile_cache_dir` is unset (default) -- matches
    the off-by-default / behavior-preserving rule.
- **Load must precede rebuild, not follow it.** `load_cache_artifacts()`
  only helps graphs Dynamo hasn't already retraced-and-missed against, so it
  has to run before `enable_compiled_training()`, not after.
- Counter-delta save gate: reuse the `frames.total` approach, but the
  "before" snapshot needs to bracket the whole training step or steps (not
  one call like sampling's single `pipeline(...)`), since training compiles
  span many blocks/shapes over more than one call.

### Open questions (resolve before implementing)

- Slice 5's fix needs to decide where the re-enable call lands: inside
  `MemoryManager` itself (e.g. `_restore_offload` calling
  `enable_compiled_training()` directly on the module), or left to
  `BaseSDTrainProcess.py`'s training loop noticing the invalidation and
  re-calling it (mirroring how `generate_single_image` re-calls
  `enable_compiled_sampling()` every sample rather than manager.py doing it).
  The mega-cache load hook goes wherever that re-enable call ends up.
- Should the training-blob save be gated on "did a *new* graph compile" the
  same way sampling is, or does the sheer variety of training shapes make
  that check nearly always true (degrading to "save every sampling round" --
  measure blob size and save latency before deciding either way)?
- Sequencing with `COMPILE_STREAMED_OFFLOAD_PLAN.md` Slice 5's own open
  question ("compile only after controllers settle" / "freeze resizes once
  compile engages"): if that lands first, training compile becomes far more
  stable across a run, which changes how often the mega-cache actually needs
  to re-save. Land training mega-cache support *after* that Slice 5 work,
  not before.

## Acceptance criteria

- Off (`compile_cache_dir=None`): zero behavior change, matches existing
  `tests/`.
- On: a training -> sample -> training sequence within one process shows the
  post-sampling training-compile rebuild hit the loaded cache (no new
  `frames.total` growth) instead of a cold recompile.
- Cross-process: killing and restarting a training job with
  `compile_cache_dir` set skips the cold compile on the first post-restart
  sampling round AND the first post-restart training step, each from its own
  key.
- Sampling and training blobs never share a key; deleting one cache dir
  entry doesn't invalidate the other.
