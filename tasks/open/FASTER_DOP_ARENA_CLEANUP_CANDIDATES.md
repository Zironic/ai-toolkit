# Faster-DOP Arena Cleanup Candidate Inventory

> Read-only audit deliverable. This document records candidates; it does not
> authorize deletion and it does not report cleanup progress.

## Snapshot and scope

This inventory was reconciled from three independent audits of production
runtime/configuration, tests/scripts, and documentation/repository artifacts.
The repository snapshot was `db611149eb419165e7c68b4eed4b2c5630743cff`, which
also matched `origin/faster-dop` when the audit began.

The checkout already contained unrelated modified, deleted, and untracked
files. Those changes were treated as user work and were not altered. In
particular, current Mega-Cache work and its local outputs are not cleanup
candidates merely because they are recent or untracked.

The bounded scope is the candidate set defined by
`tasks/open/FASTER_DOP_ARENA_CHAFF_CLEANUP_PLAN.md`: the active Arena and shared
host-memory packages, their direct callers, retired Arena terminology and
symbols, associated tests/scripts/configuration/docs, known repository
artifacts, and the reference closure needed to classify those items.

This register uses these dispositions:

- **Delete**: no present purpose was found after caller/reference inspection.
- **Consolidate**: retain a current rule, assertion, or result in its active
  owner, then delete the obsolete or duplicate carrier.
- **Needs decision**: a necessary fact or owner decision is still missing.
- **Deferred by approval**: the issue is understood and the user has explicitly
  accepted it as separate follow-up rather than a blocker for other slices.
- **Keep anchor**: present purpose was verified; listed to bound the cleanup and
  prevent false positives.
- **Existing user work**: the path is currently modified outside this audit and
  must be preserved and re-audited from the cleanup worktree.
- **Out of scope**: the path was inspected only to establish the boundary.
- **Ignored/generated local state**: no repository cleanup action is allowed.

Confidence describes the inventory classification, not permission to execute
it. Dependencies must be resolved before removing a candidate.

## Review authorization

**Current status: Not approved for deletion.**

No destructive cleanup begins until:

- the path-sorted review ledger is present and reconciled with the execution
  worktree;
- the durable legacy/Arena ownership correction is present in the cleanup plan
  and current Arena contract;
- every path in the selected cleanup slice has a final keep, delete,
  consolidate, deferred-by-approval, existing-user-work, or out-of-scope
  disposition;
- the selected slice's local dependencies are resolved; and
- the user records approval for that slice on ticket `d1bd2c9`.

Approval of one closed slice does not authorize another. After cleanup, the
final diff must be reconciled against this ledger, durable conclusions must be
moved to their current owner, and this temporary inventory must be deleted.

This document is itself a working audit artifact. Once every candidate has an
execution disposition and the cleanup is complete, it should be removed or
reduced to whatever durable decision is not captured by the current contract
and ticket. It must not become another permanent investigation diary.

## Resolved architectural and support boundary

The cleanup plan currently describes `_BouncingLinearFn` too broadly as
retired. Caller inspection shows a narrower boundary:

- The supported Arena backend is the generic block-native Arena dispatcher.
- `MemoryManager.attach()` still installs `LinearLayerMemoryManager`, whose
  ordinary per-linear forward calls `_BouncingLinearFn.apply()` in
  `toolkit/memory_management/manager_modules.py:2091-2122`.
- The old smart-training, prefetch, trace, block-stream, and pinned-arena
  extensions around that mechanism have no current production attachment path
  and are cleanup candidates.
- The minimal plain legacy per-linear backend remains supported for upstream
  compatibility and for toolkit capabilities the Arena does not yet own,
  especially text-encoder offload.

Therefore, cleanup must carve the retired C/D-era extensions away from the
plain legacy backend. It must not delete `_BouncingLinearFn` or
`LinearLayerMemoryManager` wholesale based only on their age or terminology.

Repository reachability and explicit toolkit workflows are the support
boundary for this cleanup. There are no external consumers for which this fork
must preserve undocumented APIs, aliases, old job fields, or behavior from its
own earlier iterations. If nothing in the toolkit uses a compatibility surface
and no current workflow explicitly owns it, it is unused.

## Production runtime and integration candidates

### P1. Duplicate Arena finalization

- **Candidate:** `jobs/process/BaseSDTrainProcess.py:3166` and `:3574` both call
  `arena_runtime.finalize(self.network)`; `_arena_runtime` is also initialized
  twice at `:210` and `:344`.
- **Evidence:** `ArenaOffloadRuntime.finalize()` at
  `toolkit/memory_management/arena_offload/runtime.py:507-543` repeats cap and
  FP8 finalization work. Only dispatcher finalization is internally guarded at
  `arena_offload/dispatcher.py:261`.
- **Provisional disposition:** **Consolidate** to one lifecycle owner and one
  finalization call.
- **Confidence:** High.
- **Dependency:** Establish which current call is the correct post-load
  lifecycle boundary before removing the other.

### P2. Obsolete in-graph compatibility surface

- **Candidate:** `toolkit/memory_management/ingraph_stream.py`.
- **Evidence:** The file labels itself a compatibility surface at `:1-5`.
  Current Arena construction imports layout and transfer functions directly.
  Production references are limited to legacy-manager re-exports at `:46-48`
  and a pinned-arena import at `:71`. The native execution surface
  (`LinearView`, `block_linear_views`, `streamed_linear_tensors`, `LoraEntry`,
  `TrainLeaf`, `streamed_linear`, and compile-audit helpers at `:47-217`) is
  otherwise test-only.
- **Provisional disposition:** **Consolidate** any still-live imports onto their
  current owners, then **delete** the compatibility facade and retired native
  execution surface.
- **Confidence:** High.
- **Dependency:** P4 and P8 should be resolved in the same subsystem slice.

### P3. Uninstalled in-graph scheduling pass

- **Candidate:**
  `toolkit/memory_management/ingraph_stream_scheduling.py`, especially
  `order_fetch_ops_pass()` at `:80` and `install_ordering_pass()` at `:178`.
- **Evidence:** No production installer was found. Only retired in-graph tests
  install the pass.
- **Provisional disposition:** **Delete** with its tests.
- **Confidence:** High.
- **Dependency:** Preserve any generally valid compiler-ordering conclusion in
  current compile guidance only if it is still true of the generic dispatcher.

### P4. Retired pinned-arena predecessor

- **Candidate:** `toolkit/memory_management/pinned_arena.py` and the
  `use_pinned_arena` branches in `toolkit/memory_management/manager.py:513-810`,
  `:2777-2881`, and reattachment paths around `:5787` and `:6071`.
- **Evidence:** The file documents the old Phase 3 protocol at `:19-59`.
  `MemoryManager.attach(..., use_pinned_arena=False)` exposes the path, but no
  production caller passes `True`; references that enable it are tests and
  smoke tooling.
- **Provisional disposition:** **Delete** as one coherent predecessor
  subsystem, including configuration, tests, fixture support, and smoke flags.
- **Confidence:** High.
- **Dependency:** First verify that current canonical host storage and pin
  ownership are fully covered by `CanonicalArena` and `pin_manager.py`.

### P5. Unattached smart-training planner/controller

- **Candidate:** the large smart-training subsystem in
  `toolkit/memory_management/manager.py`, including:

  - `smart_training_plan` around `:1842`;
  - block-stream hooks around `:1672`;
  - `attach_smart_training()` at `:2777`;
  - promotion, demotion, and autotune code around `:2590`, `:3150`, `:3265`,
    and `:3654`;
  - historical prefetch machinery around `:4393`;
  - smart diagnostics around `:4652`.

- **Evidence:** Production smart selection now enters the Arena path in
  `jobs/process/BaseSDTrainProcess.py:2647` and calls
  `prepare_arena_offload()` around `:2894`. No production caller of
  `attach_smart_training()` was found; remaining callers are a one-off probe
  and tests. Trainer diagnostics/pre-step calls around
  `BaseSDTrainProcess.py:1108`, `:1352`, and `:3793` become no-ops without the
  unattached plan.
- **Disposition:** **Consolidate/delete** the unattached smart-training
  planner/controller and its trainer hooks. Undocumented external use is not a
  retention reason.
- **Confidence:** High for removal of the unattached subsystem; Medium for the
  precise extraction boundary inside `manager.py`.
- **Dependency:** Retain the plain legacy attach/detach, per-linear backend,
  upstream-compatible behavior, sampling required by current integrations, and
  text-encoder offload.

### P6. Bounce-pool implementation and duplicate pin safety

- **Candidate:** `toolkit/memory_management/bounce_pool.py`, including the pool
  at `:477`, registry at `:1244`, pin-ledger delegating shims at `:128-151`, and
  duplicated device/DXGI safety logic from approximately `:164` onward.
- **Evidence:** Production pool creation was found only through the unattached
  smart-manager path at `manager.py:4501-4616`. `pin_manager.py:74-226` already
  owns the pin-accounting APIs.
- **Provisional disposition:** **Consolidate** any unique live safety rule into
  `pin_manager.py`, repoint retained legacy imports, then **delete** the pool if
  P5 is removed.
- **Confidence:** High for duplicated pin authority; Medium for whole-file
  deletion until the exact legacy boundary is proved.
- **Dependency:** `manager_modules.py:40` imports bounce functions, so live
  plain-legacy accounting must be repointed before removal.

### P7. Retired smart/block-stream portions of manager modules

- **Candidate:** parts of
  `toolkit/memory_management/manager_modules.py`, including access tracing from
  `:349`, profiling from `:80`, and `set_block_stream_enabled()` /
  `stage_block_forward()` around `:926-1108`.
- **Evidence:** Block staging is enabled only by the unattached smart path near
  `manager.py:2897`; profile and trace job flags have no production enablement.
- **Provisional disposition:** **Delete** the smart, trace, profile, and
  block-stream portions with P5/P6 while retaining the minimal ordinary
  per-layer legacy forward and `_BouncingLinearFn`.
- **Confidence:** Medium-high.
- **Dependency:** Define the retained legacy module surface before mechanical
  deletion.

### P8. Retired half of Arena layout

- **Candidate:** predecessor APIs in
  `toolkit/memory_management/arena_offload/layout.py:338-619`:
  `IngraphPackError`, `PackBuildResult`, `build_or_borrow_block_packs`,
  `BlockLeafPlan`, `assemble_leaf_args`, `BlockPlanResult`,
  `build_block_leaf_plans`, `resident_linear_tensors`, and
  `block_storage_views`.
- **Evidence:** Current construction uses the destination-first inspection,
  typed-storage, linear, and substitution APIs at `:654-765`. The older half
  is consumed only by the old in-graph facade, pinned-arena predecessor, and
  their tests.
- **Provisional disposition:** **Delete** the old half after P2/P4 consumers are
  removed. Keep `BlockPack`, current storage specs, release/lifecycle helpers,
  and destination-first APIs.
- **Confidence:** High.

### P9. Test-only canonicalization shortcut

- **Candidate:** `CanonicalArena.canonicalize()` in
  `toolkit/memory_management/canonical_arena.py:123`.
- **Evidence:** Production enters through the current prepare/runtime API
  (`arena_offload/api.py:337`, `runtime.py:199`); tests use `canonicalize()` as a
  shorthand.
- **Provisional disposition:** **Needs decision** whether to consolidate tests
  onto the supported production seam or keep this as an intentional direct
  canonical-store unit-test API.
- **Confidence:** Medium.
- **Dependency:** Decide whether a direct canonical-store unit-test API is
  intentionally supported.

### P10. FP8 compatibility residue

- **Candidate:**
  `toolkit/memory_management/fp8_transpose.py`, transitional private aliases in
  `toolkit/memory_management/arena_offload/fp8_linear.py:757`, and FP8 branches
  attached only to the old smart manager around `manager.py:1366` and `:1446`.
- **Evidence:** The transpose file is an uncalled compatibility import; active
  code imports `toolkit.quantization.fp8_transpose`. Current Arena transforms
  are owned by `arena_offload/fp8.py:49`.
- **Provisional disposition:** **Delete** the unused import shim; remove private
  aliases and old smart-manager FP8 branches only with their consumers.
- **Confidence:** High for the shim; Medium for shared legacy FP8 edits.
- **Dependency:** Preserve FP8 behavior of the plain legacy backend if it is a
  supported configuration.

### P11. Duplicate configuration alias normalization

- **Candidate:** compatibility-name normalization in both
  `toolkit/config_modules.py:792-841` and
  `toolkit/memory_management/arena_offload/api.py:39-60,190`.
- **Evidence:** Production passes `ModelConfig`, which has already normalized
  names; the Arena API copy mainly supports tests and simple stand-in objects.
- **Disposition:** **Delete** iteration-only aliases and their tests, examples,
  and duplicate normalization. Keep only names used by current toolkit
  configuration and workflows.
- **Confidence:** High.

### P12. Eager construction of unused legacy sampling context

- **Candidate:** `jobs/process/BaseSDTrainProcess.py:527-563`.
- **Evidence:** The code constructs `MemoryManager.inference_resident` before
  selecting the Arena sampling context, then replaces it with null contexts
  when Arena is active.
- **Provisional disposition:** **Consolidate** backend selection before context
  construction.
- **Confidence:** High.
- **Dependency:** Verify non-Arena sampling semantics remain unchanged.

### P13. Legacy step instrumentation invoked during Arena training

- **Candidate:** unconditional legacy offload step lifecycle calls in
  `jobs/process/BaseSDTrainProcess.py:3811-3841` and their manager handlers at
  `toolkit/memory_management/manager.py:5073-5132`.
- **Evidence:** These handlers drive old trace/pool state even during Arena
  training. However, the text encoder can still be attached through the legacy
  manager around `BaseSDTrainProcess.py:2763`.
- **Disposition:** **Consolidate** around actual legacy participation. Preserve
  the narrow lifecycle required by text-encoder offload, while removing old
  smart trace/pool work that has no attached participant.
- **Confidence:** High for the ownership decision; Medium for the exact gating
  seam.

### P14. Krea2-specific direct ranged Arena load

- **Candidate:** `_smoke_direct_arena_load_requested()` at
  `extensions_built_in/diffusion_models/krea2/krea2.py:318`, its parallel load
  path at `:796-888`, `_prepared_canonical_build`, and attachment around `:1045`.
- **Evidence:** The helper says production/UI workflows do not use the path.
  Generic `load_session.py` cannot currently intercept Krea2's custom ranged
  assignments, so this is not a trivial deletion.
- **Disposition:** **Deferred by approval as tracked design debt**.
  Krea2-specific Arena loading violates the model-integration policy, but
  removing the working path before a generic replacement exists would regress
  current capability.
- **Tracking:** git-bug ticket `a36f201` owns the model-generic direct/ranged
  loading seam and the eventual removal of this Krea2-specific branch.
- **Confidence:** High.

### P15. Package-local manual test module

- **Candidate:** `toolkit/memory_management/test_memory_manager.py`.
- **Evidence:** No production importer was found; it is a manual/package-local
  test rather than a runtime module.
- **Provisional disposition:** **Move/consolidate** any unique assertion into
  `tests/`, otherwise **delete**.
- **Confidence:** High.

## Configuration and UI candidates

### C1. Flags tied only to retired smart, trace, or pinned paths

The following fields have definitions plus tests, old scripts, UI type/docs, or
job-reset references, but no current production reader that changes the
supported Arena path:

- `layer_offloading_block_stream_only` at `toolkit/config_modules.py:775-776`;
- `layer_offloading_profile` at `:846-847`;
- `layer_offloading_trace` at `:849-850`;
- `layer_offloading_prefetch` at `:852-853`;
- `layer_offloading_prefetch_trace_capture` and `_steps` at `:855-859`;
- `layer_offloading_eager_promote_free_gb` and `_max_blocks` at `:893-900`;
- `layer_offloading_pinned_arena` at `:929-930`.

Their cleanup closure includes corresponding entries in `ui/src/types.ts`,
`ui/src/docs.tsx`, `ui/src/app/jobs/new/utils.ts`, the simple-job prefetch
control at `ui/src/app/jobs/new/SimpleJob.tsx:454-458`, smoke CLI arguments,
and tests named below.

- **Provisional disposition:** **Delete** with P4-P7, including UI/docs/reset
  residue.
- **Confidence:** High.
- **Persisted configuration decision:** Previous iteration job files are not a
  supported compatibility surface. No migration or tolerant loading is
  required for these retired fields; focused validation still must show that
  current serialized jobs and current UI job loading work after removal.

`layer_offloading_prefetch_depth` is explicitly **not** in this set: the current
Arena API consumes it at `arena_offload/api.py:294`.

### C2. Apparently unwired budget fields

- **Candidates:** `layer_offloading_pinned_weight_gb` at
  `toolkit/config_modules.py:786-787` and
  `layer_offloading_wddm_spill_reserve_pct` at `:805-806`, with their UI type,
  docs, reset, smoke, and stale-plan references.
- **Evidence:** No current production call site was found that passes these
  values into the Arena or plain manager, although manager messages still
  mention the pin override and inference smoke scripts construct them.
- **Disposition:** **Delete** the entire exposed surface unless a current
  toolkit caller is found during the mechanical reference pass. Old local job
  files and undocumented external callers do not create a compatibility
  requirement.
- **Confidence:** High.

## Test candidates

### T1. Retired Phase-0 and native in-graph execution tests

- `tests/test_ingraph_spike.py` - **Delete**; validates the original spike.
- `tests/test_ingraph_training_leaves.py` - **Delete** with the native train-leaf
  surface and scheduling pass.
- `tests/test_ingraph_residency_invariance.py` - **Delete**; predecessor
  execution invariance.
- `tests/test_ingraph_region_audit.py` - **Delete**; retired compile-region
  audit.
- Execution portions of `tests/test_ingraph_pack.py` - **Delete**; move any
  current storage-layout assertion to current layout tests.
- Execution and pinned-borrow portions of
  `tests/test_ingraph_model_agnostic_seam.py` - **Delete**; retain only a unique
  generic storage contract if not covered elsewhere.

**Confidence:** High. Dependencies: P2, P3, P4, and P8.

### T2. Mixed checkpoint/scheduling experiment tests

- `tests/test_dynamo_recompile_limit.py` is the only caller of
  `raise_dynamo_recompile_limit()` at `arena_offload/transfer.py:115`.
  **Delete** the test and helper unless a current compile caller is established.
- `tests/test_ingraph_training_ops.py` mixes test-only
  `checkpoint_recompute_context()` / `in_recompute()` assertions with
  `free_on_backward()` behavior.
- Old ordering-pass assertions in scheduling tests are **Delete** candidates
  with P3.

`free_on_backward()` is explicitly **not** dead: the current dispatcher imports
it at `arena_offload/dispatcher.py:20` and calls it at `:357`. Migrate any unique
contract assertion for that operation into current dispatcher/transfer tests,
then delete only the obsolete recompute-context and stale-facade portions.

**Provisional disposition:** **Consolidate**, not bulk-delete.
**Confidence:** High for `raise_dynamo_recompile_limit()` and the ordering pass;
Medium for the obsolete recompute-context boundary.

### T3. Current primitives hidden behind stale in-graph names

- `tests/test_ingraph_fetch_ops.py`;
- `tests/test_ingraph_fetch_metrics.py`;
- non-execution storage portions of `tests/test_ingraph_pack.py`.

These tests import current transfer/layout primitives despite stale filenames.

- **Provisional disposition:** **Consolidate/rename** into current transfer and
  layout test modules; do not bulk-delete by name.
- **Confidence:** High.

`tests/test_ingraph_stability.py` appears duplicated by
`tests/test_transfer_runtime.py` and is a **Delete after comparison** candidate.

### T4. Pinned-arena predecessor test family

- `tests/test_pinned_arena.py`;
- `tests/test_pinned_arena_attach.py`;
- `tests/test_pinned_arena_budget.py`;
- `tests/test_pinned_arena_ingraph_borrow.py`;
- `tests/test_pinned_arena_sampling_boundary.py`;
- `tests/test_pinned_arena_streaming_bypass.py`;
- `tests/test_pinned_arena_teardown.py`;
- the fixture at `tests/conftest.py:122-134`;
- file-list references at `tests/test_pin_manager.py:139,143`.

- **Provisional disposition:** **Delete** with P4; migrate only an assertion that
  is not already covered by current canonical-arena, pin-manager, lifecycle, or
  residency tests.
- **Confidence:** High.

### T5. Block-stream and eager-promotion tests

- `tests/test_block_forward_stage.py`;
- `tests/test_block_stream_only.py`;
- the retired grouped bounce-fill cases at `tests/test_bounce_pool.py:527-600`;
- `tests/test_eager_promote.py` if eager promotion is removed with P5/C1.

- **Provisional disposition:** **Delete** with P5-P7 and C1.
- **Confidence:** High for block-stream subjects; Medium-high for eager
  promotion until the precise manager extraction boundary is chosen.

### T6. FP8 experiment-only assertions

- `tests/test_fp8_divergence_metrics.py`.

- **Provisional disposition:** **Delete** with the completed divergence
  campaign after its policy conclusion is represented by current FP8 contract
  tests.
- **Confidence:** High.

## Script and diagnostic candidates

### S1. Retired in-graph proofs and a dirty dispatcher-oracle candidate

- `scripts/spike_ingraph_krea2_scale.py` - **Delete**, Phase-0 spike.
- `scripts/seam_proof_dispatcher.py` - **Delete**, predecessor proof.
- `scripts/smoke_krea2_dispatcher_oracle_cuda.py` was initially a **Delete**
  candidate. Its exact current replacements are
  `tests/test_generic_block_dispatcher.py`,
  `scripts/smoke_transformer_train_cuda.py`, and
  `scripts/smoke_quantized_linear_cuda.py`. The file is now modified user work,
  so its current disposition is **Existing user work** and it is not authorized
  for this cleanup until re-audited.

**Confidence:** High for the two retired proofs; deletion confidence for the
oracle is suspended by the dirty-worktree gate.

### S2. Block-stream and bounce-fill benchmarks

- `scripts/bench_block_stream_forward.py`;
- `scripts/bench_bounce_fill_group.py`;
- retired block-stream/pinned-arena CLI branches in
  `scripts/smoke_krea2_train_cuda.py`.

- **Provisional disposition:** **Delete** with P4-P7; keep the current Krea2
  training smoke after stripping retired modes.
- **Confidence:** High.

### S3. Settled WDDM/allocator probes

- `scripts/bench_allocator_cap_gc.py` - **Delete**, settled cap/GC experiment.
- `scripts/bench_gc_threshold_allowance.py` - **Delete**, settled threshold
  experiment.
- `scripts/dxgi_pin_unpin_probe.py` - **Delete**, one-time pin/unpin proof.
- `scripts/dxgi_meminfo_probe.py` - **Delete after comparison** if all current
  diagnostics are covered by `scripts/dxgi_memwatch.py`.

**Confidence:** High except Medium-high for the meminfo comparison.

### S4. Completed memory-policy probes

- `scripts/krea2_almost_training_memory_probe.py` - **Delete**; its only smart
  attachment call is not a production path.
- `scripts/measure_marginal_resident_block.py` - **Delete**; the measurement
  campaign is concluded and policy owns the result.

**Confidence:** High.

### S5. Completed FP8 divergence campaign

- `scripts/analyze_fp8_backward_divergence.py`;
- divergence dump/horizon CLI and reporting sections of
  `scripts/smoke_krea2_train_cuda.py:485-523`, `:603-613`, `:938-1017`,
  `:1135-1163`, and `:1256-1264`.

- **Provisional disposition:** **Delete/consolidate** experiment reporting while
  retaining current runtime FP8 selection and current numerical acceptance.
- **Confidence:** High.

### S6. Completed benchmark/oracle utilities

- `scripts/bench_quantization_matrix.py` and
  `tasks/done/QUANTIZATION_BENCHMARK_MATRIX_PLAN.md` are one completed workflow:
  **consolidate** any current hardware-support conclusion, then **delete both**.
- `scripts/bench_checkpoint_save.py` - **Delete after confirming** the current
  asynchronous-save implementation/tests cover the recurring question.
- `scripts/search_replace.py` - **Needs decision** whether
  `scripts/exact_edit.py` fully replaces its repository workflow; delete if it
  does.

**Confidence:** High for the quantization campaign; Medium-high for the other
two.

### S7. Unowned one-off smokes

- `scripts/smoke_krea2_quant_cache.py` - **Delete**; no toolkit caller or
  documented recurring workflow was found, and it overlaps current loading
  tests.
- `scripts/smoke_cudnn_gqa.py` had no toolkit caller or maintained acceptance
  workflow in the audit, but is now modified user work. Its current disposition
  is **Existing user work**, not delete; re-audit it after that work lands.

**Confidence:** High for the quant-cache smoke; deletion confidence for the GQA
smoke is suspended by the dirty-worktree gate.

### S8. Prefetch trace replay pipeline

- `scripts/replay_prefetch_trace.py` has a focused parser/replay test in
  `tests/test_prefetch_trace_replay.py` and is a reusable tool in isolation,
  but its advertised real-training capture fields are part of C1 and have no
  current production reader.
- **Disposition:** **Delete** the replay script, its subject-only test/docs, and
  stale capture fields together. A test for the replay tool does not establish
  a current workflow when the toolkit has no production capture producer.
- **Confidence:** High.

## Repository-structure residue

### R1. Tracked one-shot editors

All five are literal historical source editors whose intended changes have
landed:

- `.codex_cleanup_pin_imports.py`;
- `.codex_fix_pin_cap.py`;
- `.codex_fix_pin_cap_tests.py`;
- `.codex_reserve_bounce_for_explicit_pin.py`;
- `.codex_reserve_bounce_test.py`.

- **Provisional disposition:** **Delete** and add a precise root ignore such as
  `/.codex_*.py` if this helper naming pattern is expected to recur.
- **Confidence:** High.

### R2. Accidentally tracked live worktree gitlinks

- `.worktrees/local-current` - branch `benchmark/local-current`, commit
  `c62740b31d89024fa6223c5975a2c01906996db0`;
- `.worktrees/upstream-naive` - branch `benchmark/upstream-naive`, commit
  `0555746e827b0d871cade323f816c899839649bb`;
- `.worktrees/upstream-stock` - branch `benchmark/upstream-stock`, commit
  `2089de12697c544f77c51002f31a21834777ad6e`.

- **Provisional disposition:** Remove only the mode-`160000` gitlinks from the
  cleaned index and add `/.worktrees/` to `.gitignore`.
- **Confidence:** High.
- **Safety dependency:** Do not delete directories, unregister worktrees, or
  move branches. Record and verify identical worktree paths, branches, and
  commits before and after the index-only cleanup.

### R3. Minor tracked placeholders

- `tasks/done/.gitkeep` - **Needs decision**; redundant while the directory
  remains populated, but useful if policy intentionally permits it to be empty.
- `.gitmodules` - **Keep/out of scope**; the empty tracked file also exists
  upstream and is not Arena iteration residue.

### R4. Generated and ignored local state

No tracked Arena captures, benchmark reports, replay JSONL, tensor payloads,
cache payloads, or generated matrix result sets were found.

The following are deliberately not repository cleanup candidates:

- active `.agent/tmp/zimage_megacache_*` and `.agent/tmp/arena_megacache_*`
  outputs, which are already ignored;
- the registered `.agent/tmp/mini-pr-compile` worktree;
- ignored `build.log` and `debug.log` files, which are optional local disk
  cleanup only;
- `aitk_db.db`, which is active UI/job state;
- `.env`, `.pytest_cache`, `.ruff_cache`, and `__pycache__` local state.

## Documentation and plan candidates

### D1. Explicitly superseded completed plan family

The following documents label themselves superseded, retired, unshipped, or
replaced. Their present architectural conclusions should live in the current
Arena contract or focused guidance, not in iteration narratives:

- `tasks/done/MODEL_AGNOSTIC_SUBPLAN_1_MEMORY_COMPILE.md`;
- `tasks/done/MODEL_AGNOSTIC_SUBPLAN_2_FP8_FORWARD.md`;
- `tasks/done/MODEL_AGNOSTIC_SUBPLAN_3_FP8_BACKWARD.md`;
- `tasks/done/MODEL_AGNOSTIC_SUBPLAN_4A_QUANT_SEAM.md`;
- `tasks/done/UPSTREAM_PR_A_SAMPLING_PLAN.md`;
- `tasks/done/UPSTREAM_PR_C_STREAMING_CORE_PLAN.md`;
- `tasks/done/UPSTREAM_PR_D_FP8_TRAINING_PLAN.md`;
- `tasks/done/UPSTREAM_PR_P_DXGI_PROBE_PLAN.md`;
- `tasks/done/COMPILE_STREAMED_OFFLOAD_PLAN.md`;
- `tasks/done/INGRAPH_STREAM_PLAN.md`;
- `tasks/done/INGRAPH_PHASE3_SAMPLER_PLAN.md`;
- `tasks/done/INGRAPH_PHASE4A_TRAINING_PLAN.md`;
- `tasks/done/INGRAPH_GUARD_FREE_LEAVES_PLAN.md`;
- `tasks/done/COMPILE_NEUTRAL_KREA2_RUNTIME_REFACTOR_PLAN.md`;
- `tasks/done/revised_combined_compile_neutral_krea2_refactor_plan.md`.

- **Provisional disposition:** **Delete after reference repair and
  consolidation**.
- **Confidence:** High.
- **Durable destinations:** `docs/ARENA_OFFLOAD_CONTRACT.md`, current compile
  guidance, current WDDM guidance, and contract-level tests. Do not create a
  new historical summary merely to preserve their narratives.

### D2. Superseded predecessors with a small amount of durable material

- `tasks/done/GENERIC_ADAPTER_IMMUTABLE_RUNTIME_PLAN.md` - preserve only current
  adapter support/failure boundaries.
- `tasks/done/PINNED_ARENA_PLAN.md` - preserve only current canonical-host and
  lifecycle rules not already in the contract.
- `tasks/done/PINNED_ARENA_PHASE3_TRAINING.md` - preserve only frozen-base,
  lifecycle, and optimizer-exclusion rules that remain true.
- `tasks/done/BLOCK_STREAM_PLAN.md` - preserve the compact negative result that
  one-H2D-per-block regressed wall time.
- `tasks/done/OSTRIS_NAIVE_PREFETCH_BENCHMARK_PLAN.md` - preserve only the
  concluded benchmark result if it still informs a current choice.

- **Provisional disposition:** **Consolidate then delete**.
- **Confidence:** High.

### D3. Large completed implementation plans that should not remain manuals

- `tasks/done/GENERIC_BLOCK_DISPATCHER_PLAN.md` - 1,277-line completed sequence;
  extract saved-forward ownership, model checkpoint ownership, compilation
  boundary, dispatcher boundary, and unsupported-state failures.
- `tasks/done/IMMUTABLE_TRANSFER_ARENA_PLAN.md` - extract canonical host/storage
  invariants and still-current pinnedness/Dynamo facts.
- `tasks/done/OSTRIS_ARENA_QUANTIZATION_PLAN.md` - extract quantizer-owned
  ordered storage and operation binding where not already in the contract.
- `tasks/done/PIN_MANAGER_PLAN.md` - extract only the single-authority,
  priority, and eviction policy; it still names obsolete in-graph consumers.
- `tasks/done/UPSTREAM_ARENA_EXTRACTION_PLAN.md` - retain only present backend
  boundaries not already documented.

- **Provisional disposition:** **Consolidate then delete**.
- **Confidence:** Medium-high, except Medium for the last item.

### D4. Historical strategy/merge records with stale runtime claims

- `docs/decisions/UPSTREAM_PR_PLAN.md` - candidate to consolidate present
  backend-boundary, host-safety, and explicit-selection semantics, then delete
  if it has no current faster-dop purpose.
- `docs/decisions/UPSTREAM_MERGE_2026-07-10.md` - candidate to retain unique Krea
  reference K/V, checkpoint, or TorchAO traversal decisions if still current,
  then delete or heavily trim the stale state diary and retired pinned/in-graph
  claims.

- **Provisional disposition:** **Needs decision**, then consolidate/delete if
  no current faster-dop purpose remains.
- **Confidence:** Medium.
- **Reference repairs:** If `UPSTREAM_PR_PLAN.md` is removed, update
  `AGENTS.md:25`, `CLAUDE.md:222`, both compile-offload skills at `:54`, both
  model-integration skills at `:62`, and
  `tasks/open/DISK_OFFLOAD_TODO.md:163`.

This classification concerns whether the documents have a present purpose in
faster-dop. It does not define or schedule work on any other branch.

### D5. Active plans with stale historical tails or references

- `tasks/open/COMPILE_MEGA_CACHE_PLAN.md` - **Keep** current scope and remaining
  work; condense the retired design history at `:137-240` and repair retired
  plan references at `:10-14`, `:86`, `:89`, `:157`, `:181-182`, and `:233`.
- `tasks/open/KREA2_FILTER_BYPASS_PLAN.md` - **Keep** active work; update
  `:365-388`, which explains performance through retired `_BouncingLinearFn`
  smart/prefetch plans, to describe the current plain legacy mechanism.
- `tasks/open/DISK_OFFLOAD_TODO.md` - **Keep**; repair `:163` if its target
  strategy document is removed.

- **Provisional disposition:** **Keep and trim/repair**.
- **Confidence:** High.

### D6. Completed plans requiring consolidation or current-policy retention

- `tasks/done/AUTOTUNE_PLAN.md` - **Consolidate any still-current WDDM policy,
  then delete** with the unattached smart controller.
- `tasks/done/PREFETCH_2_PLAN.md` - **Consolidate any still-current trace safety
  rule, then delete** with BouncePool trace/resync.
- `tasks/done/RESIDENCY_TWO_TIMESCALE_PLAN.md` - **Keep anchor or extract current
  policy then delete**; it contains current Arena policy but stale smoke links.
- `tasks/done/FP8_BACKWARD_DIVERGENCE_PLAN.md` - **Consolidate result then
  delete** if current quantization guidance already owns the conclusion.
- `tasks/done/QUANTIZATION_BENCHMARK_MATRIX_PLAN.md` - **Consolidate any current
  hardware-support conclusion, then delete** as a completed campaign.
- `tasks/done/ANIMA_BASE_MODEL_REINTEGRATION_PLAN.md` - **Keep or compact**;
  current integration is supported and is outside Arena iteration residue
  unless all durable rules already live in model-integration guidance.

### D7. Guidance drift and duplicated manuals

- `.agents/skills/compile-offload/SKILL.md` and
  `.claude/skills/compile-offload/SKILL.md` are intentional byte-identical
  mirrors, but `:3`, `:13-14`, `:25-26`, and `:34-35` still describe retired
  in-graph streaming. **Keep both and update in lockstep**.
- `.agents/skills/model-integration/SKILL.md` and its `.claude` mirror point at
  a nonexistent open Anima plan at `:21`. **Keep both and repair in lockstep**.
- Other paired `.agents/skills/*` and `.claude/skills/*` files are intentional
  packaging, not accidental duplication. **Keep**.
- `CLAUDE.md` presents the legacy manager as the main subsystem at `:20-26` and
  has older pin-ledger/cap ownership descriptions at `:165-176`. **Align with
  `AGENTS.md` and current skills, or replace duplicated architecture prose with
  a short pointer plus Claude-specific rules**.
- `tasks/README.md:9-10` says every completed spec is kept, conflicting with the
  cleanup retention rule. **Update the policy** so completed docs remain only
  for unique durable value.
- `CHANGES_FROM_UPSTREAM.md` is an intentionally maintained guide for both
  maintainers and other readers to understand this fork's purpose and material
  differences. **Keep and refresh** stale Arena ownership, supported-model,
  compilation, and lifecycle claims; do not retire or reduce it to a historical
  snapshot.

### D8. Documentation keep anchors

- `docs/ARENA_OFFLOAD_CONTRACT.md` - authoritative current contract and the
  preferred destination for the few durable conclusions extracted above.
- `AGENTS.md` - current repository code map and memory-management principles;
  repair only references invalidated by cleanup.
- `tasks/open/ARENA_FULLGRAPH_COMPILE_PLAN.md` - current active acceptance work.
- `tasks/open/FASTER_DOP_ARENA_CHAFF_CLEANUP_PLAN.md` - controlling cleanup
  plan.
- `CHANGES_FROM_UPSTREAM.md` - maintained explanation of what this fork changes
  and why, for internal and external readers.
- `docs/comfyui_dynamic_vram_analysis.md` - unique external research comparison;
  not iteration residue without a separate decision to retire that research.

## Runtime and validation keep anchors

The following are not cleanup candidates unless a later caller audit produces
new evidence:

- current Arena API/prepare/runtime, dispatcher, construction, load session,
  residency, transfer plan, and destination-first layout;
- `CanonicalArena`, `ImmutableRuntime`, `ResidencyState`, `pin_manager.py`,
  `vram_budget.py`, `allocator_cap.py`, NVML/DXGI safety, and current FP8 owner;
- plain `MemoryManager.attach()` / detach, ordinary per-linear legacy forward,
  current sampling support required by non-Arena models, and text-encoder
  offload; this surface is retained for upstream compatibility and capabilities
  not yet owned by Arena;
- ordinary Krea2 model-owned forward and generic saved-forward dispatcher;
- current `layer_offloading_prefetch_depth` and live Arena selection/config
  fields;
- current Arena, canonical-storage, direct-quantization, dispatcher, residency,
  transfer, allocator, pin, and WDDM tests;
- current legacy tests after removing only retired bounce-fill/prefetch groups;
- `scripts/bench_pin_assumptions.py`, `scripts/dxgi_memwatch.py`,
  `scripts/digest_perf_log.py`, and
  `scripts/sim_working_reserve_controller.py` as recurring diagnostics;
- current runtime/profile/transformer/quantized-linear/Krea2 train and inference
  smokes after obsolete modes are stripped;
- current Mega-Cache probes, diagnostics, matrix harness, tests, and the active
  untracked `scripts/bench_full_model_megacache_residency.py` user work;
- Krea filter/vector work, Anima integration/examples, and active prompt JSON,
  which have current owners or are outside this cleanup's Arena-iteration
  boundary.

## Resolved policy decisions

1. Keep the plain legacy memory manager for upstream compatibility and current
   toolkit capabilities such as text-encoder offload.
2. Repository-supported toolkit callers and explicit workflows define use. No
   undocumented external consumer is assumed.
3. Do not preserve aliases, flags, or behavior solely for this fork's earlier
   iterations.
4. Keep Krea2-specific direct loading until ticket `a36f201` replaces it with a
   model-generic direct/ranged loading seam.
5. Keep and maintain `CHANGES_FROM_UPSTREAM.md` as the fork's useful public and
   internal orientation document.

## Decisions required before their affected cleanup slice

1. **Legacy/smart slice (P4-P7, P10, P12-P13, C1, T4-T5):** establish the exact
   code extraction seam that preserves plain-manager sampling, FP8 behavior
   used by current integrations, and TE offload while deleting the unattached
   smart controller and trace/pool machinery.
2. **Mixed-test slice (P8-P9, T1-T3):** decide whether any unique current
   assertion from predecessor tests must move into a surviving contract-level
   test.
3. **Documentation slice (D1-D7):** choose the minimal current
   contract/guidance destinations before deleting completed plans; do not
   replace them with another historical compendium.
4. **Dirty candidates (the ledger's Existing user work rows):** wait for the
   current owner work to settle, then re-audit from the dedicated cleanup
   worktree.

R1 and R2 have no dependency on these decisions, but still require explicit
slice approval and the worktree-preservation check before execution. P14 is
already **Deferred by approval** to ticket `a36f201` and does not block other
cleanup slices.

## Path-sorted review ledger

Generated from the bounded discovery rules at the recorded HEAD, with the
dirty-worktree overlay refreshed during this revision. It contains one row for
each tracked or untracked file admitted to the review surface. The final
local-state rows are path patterns rather than tracked-file rows; they document
exclusions without turning ignored outputs into source inventory. The two
cleanup documents are the authorized outputs of this task; other dirty rows are
preserved user work, including concurrent edits detected while the ledger was
being built. This overlay is a snapshot, not a live worktree monitor: any path
modified after the capture inherits **Existing user work** until the required
execution-worktree reconciliation proves otherwise.

Anchor IDs: `KA` current Arena, `KL` plain legacy/upstream/TE support, `KS` shared safety, `KC` active compile/MegaCache, `KV` current validation, `KT` recurring tools, `KD` durable guidance, `KO` established out-of-scope work, `KU` existing user work.

| Path | Track | Subsystem | Provisional disposition | Candidate/anchor ID | Dependency | Review decision |
| --- | --- | --- | --- | --- | --- | --- |
| `.agent-hooks/agent_hooks.py` | Dirty boundary | Unrelated current work | Existing user work | KU | Outside bounded cleanup | Preserve; out of cleanup |
| `.agent/tmp/arena_megacache_*` | Local state | Ignored/generated state pattern | Ignored/generated local state | R4 | Not tracked cleanup input | No repository action |
| `.agent/tmp/mini-pr-compile` | Local state | Ignored/generated state pattern | Ignored/generated local state | R4 | Not tracked cleanup input | No repository action |
| `.agent/tmp/zimage_megacache_*` | Local state | Ignored/generated state pattern | Ignored/generated local state | R4 | Not tracked cleanup input | No repository action |
| `.agents/skills/compile-offload/SKILL.md` | Guidance | Compile ownership | Existing user work | D7/KU | Dirty active guidance | Preserve; re-audit from cleanup worktree |
| `.agents/skills/cuda-testing/SKILL.md` | Guidance | Validation and WDDM | Keep | KD | Intentional mirrored packaging | Keep approved |
| `.agents/skills/model-integration/SKILL.md` | Guidance | Model integration | Consolidate | D7 | Repair stale Anima reference | Resolve documentation slice |
| `.agents/skills/wddm-memory/SKILL.md` | Guidance | Validation and WDDM | Keep | KD | Intentional mirrored packaging | Keep approved |
| `.claude/settings.json` | Dirty boundary | Unrelated current work | Existing user work | KU | Outside bounded cleanup | Preserve; out of cleanup |
| `.claude/skills/compile-offload/SKILL.md` | Guidance | Compile ownership | Existing user work | D7/KU | Dirty active guidance | Preserve; re-audit from cleanup worktree |
| `.claude/skills/cuda-testing/SKILL.md` | Guidance | Validation and WDDM | Keep | KD | Intentional mirrored packaging | Keep approved |
| `.claude/skills/model-integration/SKILL.md` | Guidance | Model integration | Consolidate | D7 | Repair stale Anima reference | Resolve documentation slice |
| `.claude/skills/wddm-memory/SKILL.md` | Guidance | Validation and WDDM | Keep | KD | Intentional mirrored packaging | Keep approved |
| `.codex_cleanup_pin_imports.py` | Repository | One-shot editors | Delete | R1 | Verify landed changes; add precise ignore | Candidate only; not approved |
| `.codex_fix_pin_cap_tests.py` | Repository | One-shot editors | Delete | R1 | Verify landed changes; add precise ignore | Candidate only; not approved |
| `.codex_fix_pin_cap.py` | Repository | One-shot editors | Delete | R1 | Verify landed changes; add precise ignore | Candidate only; not approved |
| `.codex_reserve_bounce_for_explicit_pin.py` | Repository | One-shot editors | Delete | R1 | Verify landed changes; add precise ignore | Candidate only; not approved |
| `.codex_reserve_bounce_test.py` | Repository | One-shot editors | Delete | R1 | Verify landed changes; add precise ignore | Candidate only; not approved |
| `.codex/hooks.json` | Dirty boundary | Unrelated current work | Existing user work | KU | Outside bounded cleanup | Preserve; out of cleanup |
| `.env` | Local state | Ignored/generated state pattern | Ignored/generated local state | R4 | Not tracked cleanup input | No repository action |
| `.gitignore` | Repository | Ignore policy | Consolidate | R1/R2 | Add /.codex_*.py and /.worktrees/ | Resolve repository slice |
| `.gitmodules` | Repository | Upstream structure | Out of scope | R3 | Exists upstream | Excluded |
| `.pytest_cache/**` | Local state | Ignored/generated state pattern | Ignored/generated local state | R4 | Not tracked cleanup input | No repository action |
| `.ruff_cache/**` | Local state | Ignored/generated state pattern | Ignored/generated local state | R4 | Not tracked cleanup input | No repository action |
| `.worktrees/local-current` | Repository | Tracked worktree gitlinks | Delete | R2 | Index only; preserve registration/path/SHA | Candidate only; not approved |
| `.worktrees/upstream-naive` | Repository | Tracked worktree gitlinks | Delete | R2 | Index only; preserve registration/path/SHA | Candidate only; not approved |
| `.worktrees/upstream-stock` | Repository | Tracked worktree gitlinks | Delete | R2 | Index only; preserve registration/path/SHA | Candidate only; not approved |
| `**/__pycache__/**` | Local state | Ignored/generated state pattern | Ignored/generated local state | R4 | Not tracked cleanup input | No repository action |
| `AGENTS.md` | Guidance | Repository contract | Existing user work | KD/KU | Dirty active guidance | Preserve; re-audit from cleanup worktree |
| `aitk_db.db` | Local state | Ignored/generated state pattern | Ignored/generated local state | R4 | Not tracked cleanup input | No repository action |
| `build.log` | Local state | Ignored/generated state pattern | Ignored/generated local state | R4 | Not tracked cleanup input | No repository action |
| `CHANGES_FROM_UPSTREAM.md` | Documentation | Fork orientation | Keep | D7/D8 | Refresh stale Arena claims | Keep approved |
| `CLAUDE.md` | Guidance | Repository contract | Consolidate | D7 | Align with AGENTS/current skills | Resolve documentation slice |
| `config/examples/train_lora_anima_24gb.yaml` | Configuration | Supported Anima example | Out of scope | KO | Current Anima integration | Excluded |
| `config/krea2_sdpa_ab_cudnn.json` | Local state | Ignored local configs | Ignored/generated local state | R4 | Ignored by /config/* | No repository action |
| `config/krea2_sdpa_ab_expanded_efficient.json` | Local state | Ignored local configs | Ignored/generated local state | R4 | Ignored by /config/* | No repository action |
| `debug.log` | Local state | Ignored/generated state pattern | Ignored/generated local state | R4 | Not tracked cleanup input | No repository action |
| `docs/ARENA_OFFLOAD_CONTRACT.md` | Documentation | Current Arena contract | Existing user work | KD/KU | Ownership correction applied alongside concurrent compile edits | Preserve; re-audit from cleanup worktree |
| `docs/comfyui_dynamic_vram_analysis.md` | Documentation | External research | Out of scope | D8 | Unique research anchor | Excluded |
| `docs/decisions/MEGACACHE.md` | Documentation | Current compile-cache contract | Existing user work | KC/KD/KU | Untracked active MegaCache guidance | Preserve; out of cleanup |
| `docs/decisions/UPSTREAM_MERGE_2026-07-10.md` | Documentation | Historical strategy | Needs decision | D4 | Extract unique current decisions first | Decide documentation slice |
| `docs/decisions/UPSTREAM_PR_PLAN.md` | Documentation | Historical strategy | Existing user work | D4/KU | Dirty candidate document | Preserve; re-audit from cleanup worktree |
| `extensions_built_in/advanced_generator/ReferenceGenerator.py` | Dirty boundary | Unrelated current work | Existing user work | KU | Outside bounded cleanup | Preserve; out of cleanup |
| `extensions_built_in/captioner/BaseCaptioner.py` | Dirty boundary | Unrelated current work | Existing user work | KU | Outside bounded cleanup | Preserve; out of cleanup |
| `extensions_built_in/diffusion_models/anima/anima.py` | Integration | Plain legacy callers | Keep | KL | Upstream compatibility/non-Arena support | Keep approved |
| `extensions_built_in/diffusion_models/ernie_image/ernie_image.py` | Integration | Plain legacy callers | Keep | KL | Upstream compatibility/non-Arena support | Keep approved |
| `extensions_built_in/diffusion_models/example_model/example_model.py` | Integration | Plain legacy callers | Keep | KL | Upstream compatibility/non-Arena support | Keep approved |
| `extensions_built_in/diffusion_models/flux2/flux2_klein_model.py` | Integration | Plain legacy callers | Keep | KL | Upstream compatibility/non-Arena support | Keep approved |
| `extensions_built_in/diffusion_models/flux2/flux2_model.py` | Integration | Plain legacy callers | Keep | KL | Upstream compatibility/non-Arena support | Keep approved |
| `extensions_built_in/diffusion_models/hidream/hidream_o1_model.py` | Integration | Plain legacy callers | Keep | KL | Upstream compatibility/non-Arena support | Keep approved |
| `extensions_built_in/diffusion_models/ideogram4/ideogram4.py` | Integration | Plain legacy callers | Keep | KL | Upstream compatibility/non-Arena support | Keep approved |
| `extensions_built_in/diffusion_models/krea2/krea2.py` | Integration | Krea2 direct loading | Existing user work | P14/a36f201 | Dirty; generic replacement deferred | Preserve; re-audit from cleanup worktree |
| `extensions_built_in/diffusion_models/krea2/src/mmdit.py` | Integration | Current Arena/legacy callers | Keep | KA/KL | Current production workflow | Keep approved |
| `extensions_built_in/diffusion_models/krea2/src/pipeline.py` | Integration | Current Arena/legacy callers | Keep | KA/KL | Current production workflow | Keep approved |
| `extensions_built_in/diffusion_models/ltx2/ltx2.py` | Integration | Plain legacy callers | Keep | KL | Upstream compatibility/non-Arena support | Keep approved |
| `extensions_built_in/diffusion_models/nucleus_image/nucleus_image_model.py` | Integration | Plain legacy callers | Keep | KL | Upstream compatibility/non-Arena support | Keep approved |
| `extensions_built_in/diffusion_models/qwen_image/qwen_image.py` | Integration | Plain legacy callers | Keep | KL | Upstream compatibility/non-Arena support | Keep approved |
| `extensions_built_in/diffusion_models/wan22/wan22_14b_model.py` | Integration | Plain legacy callers | Keep | KL | Upstream compatibility/non-Arena support | Keep approved |
| `extensions_built_in/diffusion_models/z_image/z_image_l2p_model.py` | Integration | Plain legacy callers | Keep | KL | Upstream compatibility/non-Arena support | Keep approved |
| `extensions_built_in/diffusion_models/z_image/z_image.py` | Integration | Plain legacy callers | Keep | KL | Upstream compatibility/non-Arena support | Keep approved |
| `extensions_built_in/diffusion_models/zeta_chroma/zeta_chroma_model.py` | Integration | Plain legacy callers | Keep | KL | Upstream compatibility/non-Arena support | Keep approved |
| `extensions_built_in/sd_trainer/SDTrainer.py` | Integration | Current Arena/legacy callers | Keep | KA/KL | Current production workflow | Keep approved |
| `jobs/process/BaseSDTrainProcess.py` | Integration | Trainer lifecycle | Existing user work | P1/P12/P13 | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `jobs/process/GenerateProcess.py` | Dirty boundary | Unrelated current work | Existing user work | KU | Outside bounded cleanup | Preserve; out of cleanup |
| `loras/krea_vector_explore/txtfusion_probe/prompts.json` | Data | Active Krea prompt input | Out of scope | KO | Owned by fe5dcca | Excluded |
| `scripts/analyze_fp8_backward_divergence.py` | Scripts | Completed FP8 campaign | Delete | S5 | Preserve current FP8 policy tests | Candidate only; not approved |
| `scripts/bench_allocator_cap_gc.py` | Scripts | Settled allocator experiments | Delete | S3 | Current allocator tests remain | Candidate only; not approved |
| `scripts/bench_block_stream_forward.py` | Scripts | Retired block stream | Delete | S2 | Remove with P5-P7 | Candidate only; not approved |
| `scripts/bench_bounce_fill_group.py` | Scripts | Retired block stream | Delete | S2 | Remove with P5-P7 | Candidate only; not approved |
| `scripts/bench_checkpoint_save.py` | Scripts | One-off save timing | Delete | S6 | Current async-save implementation/test | Candidate only; not approved |
| `scripts/bench_full_model_megacache_residency.py` | Scripts | Current dirty diagnostics/smokes | Existing user work | KU/KC | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `scripts/bench_gc_threshold_allowance.py` | Scripts | Settled allocator experiments | Delete | S3 | Current allocator tests remain | Candidate only; not approved |
| `scripts/bench_pin_assumptions.py` | Scripts | Recurring current diagnostics | Keep | KT | Documented/current workflow | Keep approved |
| `scripts/bench_quantization_matrix.py` | Scripts | Completed quantization matrix | Consolidate | S6/D6 | Move support result; delete script and plan together | Resolve documentation/tool slice |
| `scripts/capture_krea_txtfusion_forward.py` | Scripts | Active Krea filter project | Out of scope | KO | Owned by fe5dcca | Excluded |
| `scripts/digest_perf_log.py` | Scripts | Recurring current diagnostics | Keep | KT | Documented/current workflow | Keep approved |
| `scripts/dxgi_meminfo_probe.py` | Scripts | Old DXGI comparison | Consolidate | S3 | Compare with dxgi_memwatch | Resolve diagnostics slice |
| `scripts/dxgi_memwatch.py` | Scripts | Recurring current diagnostics | Keep | KT | Documented/current workflow | Keep approved |
| `scripts/dxgi_pin_unpin_probe.py` | Scripts | One-time pin proof | Delete | S3 | Current DXGI/pin tests remain | Candidate only; not approved |
| `scripts/exact_edit.py` | Scripts | Repository workflow tools | Keep | KT | Prescribed recurring workflow | Keep approved |
| `scripts/krea2_almost_training_memory_probe.py` | Scripts | Unattached smart harness | Delete | S4 | Current train/inference smokes remain | Candidate only; not approved |
| `scripts/measure_marginal_resident_block.py` | Scripts | Completed policy measurement | Delete | S4 | Result owned by current policy | Candidate only; not approved |
| `scripts/megacache_diagnostics.py` | Scripts | Current dirty diagnostics/smokes | Existing user work | KU/KC | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `scripts/probe_torch_megacache_cuda.py` | Scripts | Active MegaCache probe | Keep | KC | Current compile plan | Keep approved |
| `scripts/replay_prefetch_trace.py` | Scripts | Disconnected replay pipeline | Delete | S8 | No current capture producer | Candidate only; not approved |
| `scripts/run_full_model_megacache_matrix.py` | Scripts | Current dirty diagnostics/smokes | Existing user work | KU/KC | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `scripts/seam_proof_dispatcher.py` | Scripts | Retired in-graph proof | Delete | S1 | Current dispatcher validation named | Candidate only; not approved |
| `scripts/search_replace.py` | Scripts | Generic editor | Needs decision | S6 | Compare recurring workflow with exact_edit | Needs decision |
| `scripts/sim_working_reserve_controller.py` | Scripts | Recurring current diagnostics | Keep | KT | Documented/current workflow | Keep approved |
| `scripts/smoke_cudnn_gqa.py` | Scripts | Current dirty diagnostics/smokes | Existing user work | KU/KC | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `scripts/smoke_krea2_dispatcher_oracle_cuda.py` | Scripts | Current dirty diagnostics/smokes | Existing user work | KU/KC | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `scripts/smoke_krea2_inference_cuda.py` | Scripts | Current dirty diagnostics/smokes | Existing user work | KU/KC | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `scripts/smoke_krea2_quant_cache.py` | Scripts | Unowned quant-cache smoke | Delete | S7 | Current loading tests overlap | Candidate only; not approved |
| `scripts/smoke_krea2_train_cuda.py` | Scripts | Current dirty diagnostics/smokes | Existing user work | KU/KC | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `scripts/smoke_profiles.py` | Scripts | Current dirty diagnostics/smokes | Existing user work | KU/KC | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `scripts/smoke_quantized_linear_cuda.py` | Scripts | Current dirty diagnostics/smokes | Existing user work | KU/KC | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `scripts/smoke_runtime.py` | Scripts | Recurring current diagnostics | Keep | KT | Documented/current workflow | Keep approved |
| `scripts/smoke_transformer_train_cuda.py` | Scripts | Current dirty diagnostics/smokes | Existing user work | KU/KC | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `scripts/spike_ingraph_krea2_scale.py` | Scripts | Retired in-graph proof | Delete | S1 | Current dispatcher validation named | Candidate only; not approved |
| `scripts/tickets.cmd` | Scripts | Repository workflow tools | Keep | KT | Prescribed recurring workflow | Keep approved |
| `tasks/done/.gitkeep` | Repository | Task directory placeholder | Needs decision | R3 | Decide empty-directory policy | Needs decision |
| `tasks/done/ANIMA_BASE_MODEL_REINTEGRATION_PLAN.md` | Plans | Supported Anima integration | Out of scope | D6 | Current integration; compact only with its owner | Excluded |
| `tasks/done/AUTOTUNE_PLAN.md` | Plans | Completed campaign | Consolidate | D6 | Move current result then delete | Resolve documentation slice |
| `tasks/done/BLOCK_STREAM_PLAN.md` | Plans | Superseded predecessor | Consolidate | D2 | Extract named durable result | Resolve documentation slice |
| `tasks/done/COMPILE_NEUTRAL_KREA2_RUNTIME_REFACTOR_PLAN.md` | Plans | Retired execution plans | Delete | D1 | Repair active plan references | Candidate only; not approved |
| `tasks/done/COMPILE_STREAMED_OFFLOAD_PLAN.md` | Plans | Retired execution plans | Delete | D1 | Repair active plan references | Candidate only; not approved |
| `tasks/done/FP8_BACKWARD_DIVERGENCE_PLAN.md` | Plans | Completed campaign | Consolidate | D6 | Move current result then delete | Resolve documentation slice |
| `tasks/done/GENERIC_ADAPTER_IMMUTABLE_RUNTIME_PLAN.md` | Plans | Superseded predecessor | Consolidate | D2 | Extract named durable result | Resolve documentation slice |
| `tasks/done/GENERIC_BLOCK_DISPATCHER_PLAN.md` | Plans | Completed implementation narrative | Consolidate | D3 | Move current invariants to contract/guidance | Resolve documentation slice |
| `tasks/done/IMMUTABLE_TRANSFER_ARENA_PLAN.md` | Plans | Completed implementation narrative | Consolidate | D3 | Move current invariants to contract/guidance | Resolve documentation slice |
| `tasks/done/INGRAPH_GUARD_FREE_LEAVES_PLAN.md` | Plans | Retired execution plans | Delete | D1 | Repair active plan references | Candidate only; not approved |
| `tasks/done/INGRAPH_PHASE3_SAMPLER_PLAN.md` | Plans | Retired execution plans | Delete | D1 | Repair active plan references | Candidate only; not approved |
| `tasks/done/INGRAPH_PHASE4A_TRAINING_PLAN.md` | Plans | Retired execution plans | Delete | D1 | Repair active plan references | Candidate only; not approved |
| `tasks/done/INGRAPH_STREAM_PLAN.md` | Plans | Retired in-graph design | Consolidate | D1 | Retain only current compiler facts | Resolve documentation slice |
| `tasks/done/MODEL_AGNOSTIC_SUBPLAN_1_MEMORY_COMPILE.md` | Plans | Explicitly superseded | Delete | D1 | Repair live references | Candidate only; not approved |
| `tasks/done/MODEL_AGNOSTIC_SUBPLAN_2_FP8_FORWARD.md` | Plans | Explicitly superseded | Delete | D1 | Repair live references | Candidate only; not approved |
| `tasks/done/MODEL_AGNOSTIC_SUBPLAN_3_FP8_BACKWARD.md` | Plans | Explicitly superseded | Delete | D1 | Repair live references | Candidate only; not approved |
| `tasks/done/MODEL_AGNOSTIC_SUBPLAN_4A_QUANT_SEAM.md` | Plans | Explicitly superseded | Delete | D1 | Repair live references | Candidate only; not approved |
| `tasks/done/OSTRIS_ARENA_QUANTIZATION_PLAN.md` | Plans | Completed implementation narrative | Consolidate | D3 | Move current invariants to contract/guidance | Resolve documentation slice |
| `tasks/done/OSTRIS_NAIVE_PREFETCH_BENCHMARK_PLAN.md` | Plans | Superseded predecessor | Consolidate | D2 | Extract named durable result | Resolve documentation slice |
| `tasks/done/PIN_MANAGER_PLAN.md` | Plans | Completed implementation narrative | Consolidate | D3 | Move current invariants to contract/guidance | Resolve documentation slice |
| `tasks/done/PINNED_ARENA_PHASE3_TRAINING.md` | Plans | Superseded predecessor | Consolidate | D2 | Extract named durable result | Resolve documentation slice |
| `tasks/done/PINNED_ARENA_PLAN.md` | Plans | Superseded predecessor | Consolidate | D2 | Extract named durable result | Resolve documentation slice |
| `tasks/done/PREFETCH_2_PLAN.md` | Plans | Completed campaign | Consolidate | D6 | Move current result then delete | Resolve documentation slice |
| `tasks/done/QUANTIZATION_BENCHMARK_MATRIX_PLAN.md` | Plans | Completed campaign | Consolidate | D6 | Move current result then delete | Resolve documentation slice |
| `tasks/done/RESIDENCY_TWO_TIMESCALE_PLAN.md` | Plans | Current policy in completed plan | Consolidate | D6 | Extract policy and repair stale links | Resolve documentation slice |
| `tasks/done/revised_combined_compile_neutral_krea2_refactor_plan.md` | Plans | Retired execution plans | Delete | D1 | Repair active plan references | Candidate only; not approved |
| `tasks/done/UPSTREAM_ARENA_EXTRACTION_PLAN.md` | Plans | Completed implementation narrative | Consolidate | D3 | Move current invariants to contract/guidance | Resolve documentation slice |
| `tasks/done/UPSTREAM_PR_A_SAMPLING_PLAN.md` | Plans | Explicitly superseded | Delete | D1 | Repair live references | Candidate only; not approved |
| `tasks/done/UPSTREAM_PR_C_STREAMING_CORE_PLAN.md` | Plans | Explicitly superseded | Delete | D1 | Repair live references | Candidate only; not approved |
| `tasks/done/UPSTREAM_PR_D_FP8_TRAINING_PLAN.md` | Plans | Explicitly superseded | Delete | D1 | Repair live references | Candidate only; not approved |
| `tasks/done/UPSTREAM_PR_P_DXGI_PROBE_PLAN.md` | Plans | Superseded DXGI plan | Consolidate | D1 | Retain WDDM conclusion in current skill | Resolve documentation slice |
| `tasks/open/ARENA_FULLGRAPH_COMPILE_PLAN.md` | Plans | Active work | Existing user work | D5/D8/KU | Dirty active plan | Preserve; out of cleanup |
| `tasks/open/ARENA_ITERATION_CHAFF_CLEANUP_PLAN.md` | Plans | Superseded cleanup artifact | Delete | D1 | Already removed in working tree | Preserve existing deletion |
| `tasks/open/COMPILE_MEGA_CACHE_PLAN.md` | Plans | Active work | Existing user work | D5/D8/KU | Dirty active plan | Preserve; out of cleanup |
| `tasks/open/DISK_OFFLOAD_TODO.md` | Plans | Active work | Keep | D5/D8 | Trim only stale references/history | Keep approved |
| `tasks/open/FASTER_DOP_ARENA_CHAFF_CLEANUP_PLAN.md` | Plans | Cleanup control | Keep | KD | Delete/move when cleanup closes | Keep until cleanup completion |
| `tasks/open/FASTER_DOP_ARENA_CLEANUP_CANDIDATES.md` | Plans | Temporary review ledger | Keep | KD | Delete after final diff reconciliation | Keep until cleanup completion |
| `tasks/open/KREA2_FILTER_BYPASS_PLAN.md` | Plans | Active work | Keep | D5/D8 | Trim only stale references/history | Keep approved |
| `tasks/README.md` | Documentation | Task retention policy | Consolidate | D7 | Align with cleanup retention rule | Resolve documentation slice |
| `tests/conftest.py` | Tests | Pinned-arena fixture | Consolidate | T4 | Remove only predecessor fixture | Resolve pinned-arena slice |
| `tests/test_agent_hooks.py` | Dirty boundary | Unrelated current tests | Existing user work | KU | Outside bounded cleanup | Preserve; out of cleanup |
| `tests/test_allocator_cap.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_arena_canonical_transaction.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_arena_cap_calibrator.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_arena_fullgraph_readiness.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_arena_lifecycle_contract.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_arena_load_session.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_arena_offload_api.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_arena_offload_planner.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_arena_offload_policy.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_arena_sampling_cap.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_async_save.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_block_forward_stage.py` | Tests | Retired smart/block stream | Delete | T5 | Remove with P5-P7/C1 | Candidate only; not approved |
| `tests/test_block_stream_only.py` | Tests | Retired smart/block stream | Delete | T5 | Remove with P5-P7/C1 | Candidate only; not approved |
| `tests/test_bounce_pool.py` | Tests | Mixed bounce-pool coverage | Consolidate | P6/T5 | Retain only if live pool seam survives | Resolve legacy/smart slice |
| `tests/test_canonical_arena.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_compile_cache.py` | Tests | Active compile/dispatcher work | Existing user work | KC/KV | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `tests/test_compile_shape_bounds.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_compile_stance_probe.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_compile_utils.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_desired_pin_bytes.py` | Tests | Shared pin sizing | Keep | KL/KS | Current attach sizing helper | Keep approved |
| `tests/test_direct_arena_quantize.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_dxgi_meminfo.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_dxgi_pin_policy.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_dynamo_recompile_limit.py` | Tests | Test-only compile helper | Delete | T2 | No current production caller | Candidate only; not approved |
| `tests/test_eager_promote.py` | Tests | Retired smart/block stream | Delete | T5 | Remove with P5-P7/C1 | Candidate only; not approved |
| `tests/test_fast_fp8_dequant.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_fp8_divergence_metrics.py` | Tests | Completed divergence campaign | Delete | T6 | Retain current FP8 mechanism tests | Candidate only; not approved |
| `tests/test_fp8_grad_input_flag.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_gc_counter_deltas.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_generic_block_dispatcher.py` | Tests | Active compile/dispatcher work | Existing user work | KC/KV | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `tests/test_ingraph_fetch_metrics.py` | Tests | Current transfer under stale name | Consolidate | T3 | Move/rename into current transfer tests | Resolve mixed-test slice |
| `tests/test_ingraph_fetch_ops.py` | Tests | Current transfer under stale name | Consolidate | T3 | Move/rename into current transfer tests | Resolve mixed-test slice |
| `tests/test_ingraph_model_agnostic_seam.py` | Tests | Mixed current storage/retired execution | Consolidate | T1/T3 | Migrate unique current storage assertions | Resolve mixed-test slice |
| `tests/test_ingraph_pack.py` | Tests | Mixed current storage/retired execution | Consolidate | T1/T3 | Migrate unique current storage assertions | Resolve mixed-test slice |
| `tests/test_ingraph_region_audit.py` | Tests | Retired in-graph execution | Delete | T1 | Remove with P2/P3/P8 | Candidate only; not approved |
| `tests/test_ingraph_residency_invariance.py` | Tests | Retired in-graph execution | Delete | T1 | Remove with P2/P3/P8 | Candidate only; not approved |
| `tests/test_ingraph_spike.py` | Tests | Retired in-graph execution | Delete | T1 | Remove with P2/P3/P8 | Candidate only; not approved |
| `tests/test_ingraph_stability.py` | Tests | Duplicate transfer stability | Consolidate | T3 | Compare heterogeneous assertion first | Resolve mixed-test slice |
| `tests/test_ingraph_training_leaves.py` | Tests | Retired in-graph execution | Delete | T1 | Remove with P2/P3/P8 | Candidate only; not approved |
| `tests/test_ingraph_training_ops.py` | Tests | Mixed current free/test-only recompute | Consolidate | T2 | Keep free_on_backward contract | Resolve mixed-test slice |
| `tests/test_krea2_arena_loading.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_lora_compile_scalars.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_manager_detach_lora_chain.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_megacache_harness.py` | Tests | Active compile/dispatcher work | Existing user work | KC/KV | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `tests/test_offload_shape_key.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_pin_budget_caps.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_pin_manager.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_pinned_arena_attach.py` | Tests | Pinned-arena predecessor | Delete | T4 | Remove with P4 | Candidate only; not approved |
| `tests/test_pinned_arena_budget.py` | Tests | Pinned-arena predecessor | Delete | T4 | Remove with P4 | Candidate only; not approved |
| `tests/test_pinned_arena_ingraph_borrow.py` | Tests | Pinned-arena predecessor | Delete | T4 | Remove with P4 | Candidate only; not approved |
| `tests/test_pinned_arena_sampling_boundary.py` | Tests | Pinned-arena predecessor | Delete | T4 | Remove with P4 | Candidate only; not approved |
| `tests/test_pinned_arena_streaming_bypass.py` | Tests | Pinned-arena predecessor | Delete | T4 | Remove with P4 | Candidate only; not approved |
| `tests/test_pinned_arena_teardown.py` | Tests | Pinned-arena predecessor | Delete | T4 | Remove with P4 | Candidate only; not approved |
| `tests/test_pinned_arena.py` | Tests | Pinned-arena predecessor | Delete | T4 | Remove with P4 | Candidate only; not approved |
| `tests/test_pinned_budget_cycle.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_pinned_bytes_ledger.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_pinned_stager_snapshot_alias.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_prefetch_trace_replay.py` | Tests | Disconnected replay pipeline | Delete | S8 | Remove with capture fields/script | Candidate only; not approved |
| `tests/test_residency_ownership.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_residency_two_timescale.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_residency.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_sampling_allocator_budget.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_sampling_cold_start_estimate.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_sampling_fp8_demote_frees.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_sampling_working_reserve.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_sdpa_gqa_patch.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_smart_training_pin_scoping.py` | Tests | Unattached smart manager | Delete | P5 | Remove with attach_smart_training | Candidate only; not approved |
| `tests/test_smoke_contention_guard.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_training_compile_synthetic.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_transfer_plan.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_transfer_runtime.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_wddm_cap_relief.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_wddm_deadband.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_wddm_hard_cap.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `tests/test_working_reserve_sim.py` | Tests | Current contract validation | Keep | KV | Surviving runtime contract | Keep approved |
| `toolkit/compile_cache.py` | Runtime | Active MegaCache/compile | Existing user work | KC | Dirty active work | Preserve; out of cleanup |
| `toolkit/compile_utils.py` | Runtime | Active MegaCache/compile | Existing user work | KC | Dirty active work | Preserve; out of cleanup |
| `toolkit/config_modules.py` | Configuration | Arena/legacy fields | Existing user work | P11/C1/C2 | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `toolkit/memory_management/__init__.py` | Runtime | Memory package exports | Consolidate | P2/P4/P10 | Remove only retired exports | Resolve affected runtime slices |
| `toolkit/memory_management/allocator_cap.py` | Runtime | Shared current infrastructure | Keep | KS | Backend-independent/current Arena dependency | Keep approved |
| `toolkit/memory_management/arena_offload/__init__.py` | Runtime | Current Arena | Keep | KA | Current supported owner | Keep approved |
| `toolkit/memory_management/arena_offload/api.py` | Runtime | Current Arena | Keep | KA | Current supported owner | Keep approved |
| `toolkit/memory_management/arena_offload/cap_calibrator.py` | Runtime | Current Arena | Keep | KA | Current supported owner | Keep approved |
| `toolkit/memory_management/arena_offload/construction.py` | Runtime | Current Arena | Keep | KA | Current supported owner | Keep approved |
| `toolkit/memory_management/arena_offload/discovery.py` | Runtime | Current Arena | Keep | KA | Current supported owner | Keep approved |
| `toolkit/memory_management/arena_offload/dispatcher.py` | Runtime | Current Arena | Keep | KA | Current supported owner | Keep approved |
| `toolkit/memory_management/arena_offload/errors.py` | Runtime | Current Arena | Keep | KA | Current supported owner | Keep approved |
| `toolkit/memory_management/arena_offload/fp8.py` | Runtime | Current Arena | Keep | KA | Current supported owner | Keep approved |
| `toolkit/memory_management/arena_offload/layout.py` | Runtime | Arena layout | Consolidate | P8 | Remove retired half after P2/P4 | Resolve in-graph slice |
| `toolkit/memory_management/arena_offload/load_session.py` | Runtime | Current Arena | Keep | KA | Current supported owner | Keep approved |
| `toolkit/memory_management/arena_offload/ownership.py` | Runtime | Current Arena | Keep | KA | Current supported owner | Keep approved |
| `toolkit/memory_management/arena_offload/planner.py` | Runtime | Current Arena | Keep | KA | Current supported owner | Keep approved |
| `toolkit/memory_management/arena_offload/policy.py` | Runtime | Current Arena | Keep | KA | Current supported owner | Keep approved |
| `toolkit/memory_management/arena_offload/resources.py` | Runtime | Current Arena | Keep | KA | Current supported owner | Keep approved |
| `toolkit/memory_management/arena_offload/runtime.py` | Runtime | Arena lifecycle | Consolidate | P1 | Single finalization owner | Resolve Arena lifecycle slice |
| `toolkit/memory_management/arena_offload/transfer.py` | Runtime | Arena transfer | Consolidate | T2 | Keep free_on_backward; decide test-only helpers | Resolve mixed-test slice |
| `toolkit/memory_management/bounce_pool.py` | Runtime | Legacy smart pool | Consolidate | P6 | Move unique live pin safety first | Resolve legacy/smart slice |
| `toolkit/memory_management/canonical_arena.py` | Runtime | Canonical storage | Needs decision | P9 | Decide direct canonicalize test API | Resolve mixed-test slice |
| `toolkit/memory_management/checkpoint_autotuner.py` | Runtime | Shared current infrastructure | Keep | KS | Backend-independent/current Arena dependency | Keep approved |
| `toolkit/memory_management/dxgi_meminfo.py` | Runtime | Shared current infrastructure | Keep | KS | Backend-independent/current Arena dependency | Keep approved |
| `toolkit/memory_management/fp8_transpose.py` | Runtime | FP8 compatibility shim | Delete | P10 | Active import is toolkit.quantization | Candidate only; not approved |
| `toolkit/memory_management/immutable_runtime.py` | Runtime | Shared current infrastructure | Keep | KS | Backend-independent/current Arena dependency | Keep approved |
| `toolkit/memory_management/ingraph_stream_scheduling.py` | Runtime | Retired scheduler | Delete | P3 | Remove subject-only tests | Candidate only; not approved |
| `toolkit/memory_management/ingraph_stream.py` | Runtime | Retired in-graph facade | Delete | P2 | Repoint surviving imports | Candidate only; not approved |
| `toolkit/memory_management/manager_modules.py` | Runtime | Plain legacy modules plus retired extensions | Consolidate | P7 | Keep _BouncingLinearFn ordinary seam | Resolve legacy/smart slice |
| `toolkit/memory_management/manager.py` | Runtime | Plain manager plus retired smart extensions | Consolidate | P4-P6/P10/P12-P13 | Preserve upstream/TE/plain legacy | Resolve legacy/smart slice |
| `toolkit/memory_management/nvml_meminfo.py` | Runtime | Shared current infrastructure | Keep | KS | Backend-independent/current Arena dependency | Keep approved |
| `toolkit/memory_management/pin_manager.py` | Runtime | Shared current infrastructure | Keep | KS | Backend-independent/current Arena dependency | Keep approved |
| `toolkit/memory_management/pinned_arena.py` | Runtime | Pinned-arena predecessor | Delete | P4 | Preserve canonical Arena pin ownership | Candidate only; not approved |
| `toolkit/memory_management/residency.py` | Runtime | Shared current infrastructure | Keep | KS | Backend-independent/current Arena dependency | Keep approved |
| `toolkit/memory_management/runtime.py` | Runtime | Shared current infrastructure | Keep | KS | Backend-independent/current Arena dependency | Keep approved |
| `toolkit/memory_management/test_memory_manager.py` | Runtime | Package-local manual test | Consolidate | P15 | Move unique assertion or delete | Resolve legacy/smart slice |
| `toolkit/memory_management/transfer_plan.py` | Runtime | Shared current infrastructure | Keep | KS | Backend-independent/current Arena dependency | Keep approved |
| `toolkit/memory_management/vram_budget.py` | Runtime | Shared current infrastructure | Keep | KS | Backend-independent/current Arena dependency | Keep approved |
| `toolkit/models/base_model.py` | Integration | Current Arena/legacy callers | Keep | KA/KL | Current production workflow | Keep approved |
| `toolkit/models/wan21/wan21.py` | Integration | Plain legacy callers | Keep | KL | Upstream compatibility/non-Arena support | Keep approved |
| `toolkit/unloader.py` | Integration | Current Arena/legacy callers | Keep | KA/KL | Current production workflow | Keep approved |
| `toolkit/util/quantize.py` | Integration | Current Arena/legacy callers | Keep | KA/KL | Current production workflow | Keep approved |
| `ui/src/app/jobs/new/jobConfig.ts` | UI | Current dirty job UI | Existing user work | C1/C2 | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `ui/src/app/jobs/new/SimpleJob.tsx` | UI | Current dirty job UI | Existing user work | C1/C2 | Dirty in-scope path | Preserve; re-audit from cleanup worktree |
| `ui/src/app/jobs/new/utils.ts` | UI | Retired offload fields | Consolidate | C1/C2 | Remove field docs/reset with runtime fields | Resolve config slice |
| `ui/src/docs.tsx` | UI | Retired offload fields | Consolidate | C1/C2 | Remove field docs/reset with runtime fields | Resolve config slice |
| `ui/src/types.ts` | UI | Current dirty job UI | Existing user work | C1/C2 | Dirty in-scope path | Preserve; re-audit from cleanup worktree |

Ledger row count: **265** (260 concrete paths plus 5 local-state patterns).
Mechanical reconciliation of the bounded discovery queries found 178 unique
matched repository paths and zero paths missing from the ledger. The refreshed
dirty overlay snapshot contained 39 paths and likewise had zero unlisted paths.

## Candidate coverage and next use

Within the bounded audit set, every discovered item above now has a provisional
delete, consolidate, needs-decision, deferred-by-approval, existing-user-work,
out-of-scope, or keep disposition. Reference-following found no tracked Arena
capture/data payload family requiring source cleanup.

Implementation should proceed by coherent subsystem, not by blindly applying
every provisional label. The first executable slice should correct the legacy
boundary and then remove one closed predecessor family together with its flags,
callers, tests, scripts, and references. Mutable decisions and validation
results belong on the cleanup ticket; this register should remain a compact
candidate map until it can be retired.
