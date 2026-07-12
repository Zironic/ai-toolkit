> **SUPERSEDED 2026-07-12** - archived, not shipped. The A/P/C/D upstream stack
> and the model-agnostic subplans were retired when the immutable arena runtime
> became the sole transformer backend (commit `3dd7f38`): PRs C and D were built
> on `_BouncingLinearFn`, which no longer exists on the active path. Kept as the
> record of the reasoning. **Current plan:**
> `tasks/open/UPSTREAM_ARENA_EXTRACTION_PLAN.md` (PR P's crash-guard framing and
> upstream audit survive there as Stage 1; PR A as the deferred Stage 4).

# PR C - Bounded training streaming core (step-by-step)

> **git-bug:** `db47d2d` (stack strategy) plus the validation tickets named
> below. Strategy/scope/acceptance: `docs/decisions/UPSTREAM_PR_PLAN.md` PR C
> section. Status lives in the tickets; this file is the execution plan.

The payload PR. Generalization is largely done (no Krea/`.blocks` coupling,
device-derived margins, guarded DXGI fallback, auto planner built) - what gates
this PR is validation and packaging. Do not start extraction until A is open
and the validation phase below is green.

## Phase 1 - validate locally (blockers, in order)

1. **Close `f1c45ae`** (FP8 gate 0-layers) - resolve the cause. Check the
   Quanto hypothesis first: verify the attach-time weights are actually
   torchao (`type(weight.data).__module__`) on a cache-loaded run; PR #930-style
   de-shimming would produce exactly this symptom.
2. **Close `5fa0e3d`** - GPU-validate the auto working_reserve controller
   (grow branch, measurement-gated promotion) on a live run.
3. **Close `e22ee66`** - pinned auto-budget on a live Krea run (capped pin,
   no startup crash, DXGI ledger sane).
4. **One live run with reduced logging + the attach summary lines kept** to
   confirm plan/FP8/pin state is observable in production logging.

## Phase 2 - portability proof

5. **One foreign-machine run** (cloud spot instance is fine): ideally Linux +
   different VRAM size. Checklist: attach plan sane, no WDDM-specific behavior
   invoked, DXGI probe cleanly absent, ledger fallback engaged, training steps
   complete, flags-off path untouched.
6. **Fix what it surfaces**; expect small issues around margin naming/defaults
   on non-WDDM platforms.

## Phase 3 - packaging

7. **Rebase reality check.** Reconcile with upstream `main` (esp. PR #930:
   qfloat8 = Quanto; our streaming must stream Quanto tensors via the dequant
   path - detection already widened upstream - while FP8-native stays
   torchao-scoped or dual-layout per PR A's decision).
8. **Trim the env-var surface.** Every `AI_TOOLKIT_*` knob in the ported code
   is either promoted to config, kept as a documented dev-only override, or
   deleted. Target: reviewers see config flags, not an env zoo.
9. **Extract to a worktree branch** off upstream `main` (on top of PR P if
   option (b) was chosen): streaming core, bounce pool, safety fixes
   (CPU_FILLING reuse, OOM ring recovery, global-state reset), auto planner,
   stream attach adapter, minimal runtime summary. Exclusions per strategy doc
   (no FP8 training math, no checkpoint autotuner, no heavy profiler).
10. **Port the test suite subset** from `tests/` (bounce pool, block stream,
    working-reserve sim, shape keys) - CPU/sim tests are the upstream CI story
    since GPU CI does not exist there either.
11. **Verify the acceptance criteria** in the strategy doc (bounded pinned
    memory, OOM recovery, no stale global state across jobs, disabled path
    identical) and record evidence.
12. **Open the PR** once Ostris has signaled appetite post-A. Description
    includes the Linux/fallback story and perf evidence from the local perf
    log (PR B's data, not its code).

## Done when

All three validation tickets closed, foreign-machine run recorded on the
ticket, PR open upstream with the strategy doc's acceptance criteria
demonstrated.
