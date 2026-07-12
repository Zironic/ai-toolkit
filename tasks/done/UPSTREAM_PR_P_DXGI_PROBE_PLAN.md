> **SUPERSEDED 2026-07-12** - archived, not shipped. The A/P/C/D upstream stack
> and the model-agnostic subplans were retired when the immutable arena runtime
> became the sole transformer backend (commit `3dd7f38`): PRs C and D were built
> on `_BouncingLinearFn`, which no longer exists on the active path. Kept as the
> record of the reasoning. **Current plan:**
> `tasks/open/UPSTREAM_ARENA_EXTRACTION_PLAN.md` (PR P's crash-guard framing and
> upstream audit survive there as Stage 1; PR A as the deferred Stage 4).

# PR P - DXGI shared-budget probe + pinned-memory crash guard (step-by-step)

> **git-bug:** `db47d2d`. Strategy: `docs/decisions/UPSTREAM_PR_PLAN.md`.
> Status lives in the ticket; this file is the execution plan.

Independent of A. C depends on it (softly: C falls back to the RAM-fraction
ledger proxy when the probe is unavailable). Can run in parallel with A.

## Framing: a bugfix, not a feature

**DECIDED 2026-07-05 (was: standalone-vs-stacked open question).** Standalone
PR, framed as a crash fix. Audit of upstream `main` (`f63221e`) found the
consumer:

- `MemoryManager.attach` (`toolkit/memory_management/manager.py:73-127`)
  selects offload layers with `offload_percent` default **1.0** (selection
  below 1.0 is `random.random()`), with **no size accounting of any kind** -
  not model bytes, not host RAM, not any GPU/driver budget.
- Every selected layer runs `_move_params_to_cpu_and_pin`
  (`manager_modules.py:610`, `:649`) -> `_ensure_cpu_pinned`
  (`manager_modules.py:204`) -> `pin_memory()` / `_pin_inner_tensors`
  (`manager_modules.py:176`, recursive for quantized subclasses). A 24 GB BF16
  model pins ~24 GB of host memory unconditionally.
- The only guard is `except RuntimeError: pass` at the pin call - but the real
  failure is deferred: pins commit against the WDDM shared budget, and
  exhaustion surfaces later as a raw `cudaErrorMemoryAllocation` in unrelated
  allocations (the two-cliff model; see `docs`/memory: shared budget is also
  the dedicated cliff's overflow valve). On Linux the equivalent is pinning
  into RAM exhaustion and thrashing the host.

PR pitch: "unbounded pinning can hard-crash Windows runs (and thrash Linux
hosts); this adds a budget probe and refuses to pin past the real limit,
degrading to pageable offload instead."

## Scope discipline

The consumer stays a **minimal clamp**: running pinned-bytes tally checked at
`_ensure_cpu_pinned` / `_move_params_to_cpu_and_pin`; once headroom is
exhausted, leave remaining tensors pageable and warn **once**. Nothing else:
no transactional pin, no unpin relief, no margin governance - the intelligence
arrives with PR C / pin-for-speed. The clamp survives C as its last-resort
floor guard.

## Steps

1. ~~Audit upstream for a consumer~~ **Done** - see framing above.
2. **Extract `dxgi_meminfo.py`** as a self-contained module: ctypes
   `IDXGIAdapter3::QueryVideoMemoryInfo`, vendor-filtered adapter selection,
   NON_LOCAL budget/usage query, headroom computation. Strip local-only
   telemetry and `AI_TOOLKIT_*` env knobs except at most a single disable
   switch.
3. **Port the guarded accessor** (`get_dxgi_meminfo` returning `None` on
   import/query failure) and a `pinned_bytes_headroom`-style helper with the
   fallback contract: non-Windows / missing DXGI / query failure => RAM-fraction
   proxy (conservative psutil-based cap) so Linux gets the thrash guard too.
4. **Implement the clamp** at the pin choke point (upstream has no planner, so
   this is the only place policy can live): tally + headroom check + pageable
   fallback + single warning naming the budget and what was left unpinned.
5. **Cross-platform proof.** Import + call on Linux/macOS is a clean fallback,
   no warning spam; unit test monkeypatching the ctypes load failure.
6. **Windows validation.** Known-numbers check on this box (NON_LOCAL budget
   ~15-16 GB on 32 GB RAM; CurrentUsage moves 1:1 with pins - memory
   `reference_wddm_dxgi_numbers`). Plus a synthetic repro: attach a model
   larger than the shared budget, confirm pre-fix crash / post-fix clean
   degrade with warning.
7. **Document the two-cliff model** briefly in the module docstring - reviewers
   need it to understand why CUDA-side numbers cannot see this limit.
8. **Open the PR.** Description: crash repro, fix behavior, fallback contract,
   explicit note that a smarter budget planner (PR C) builds on the same probe.

## Done when

PR open upstream with the synthetic crash repro demonstrated (before/after),
clamp behavior validated on Windows, clean fallback on non-Windows.
