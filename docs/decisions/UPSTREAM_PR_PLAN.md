# Upstream PR strategy for the memory-management work

> **git-bug:** see the ticket "Arena offload: pre-PR refactor and upstream
> extraction". Status lives in the ticket; this file is the strategy.
> Execution plan: `../../tasks/open/UPSTREAM_ARENA_EXTRACTION_PLAN.md`.

> **Terminology note.** The reserve vocabulary was renamed; see the glossary at
> the top of `../../tasks/open/AUTOTUNE_PLAN.md`. In short: **working_reserve** =
> our transient working set (the old "headroom"), **system_reserve** = measured
> uncontrolled VRAM, **wddm_margin** = the cushion above the ~500 MB WDDM churn
> cliff, **usable** = what is left for resident weights + ring.

How to turn the local `faster-dop` memory-management work into upstreamable PRs
for `ostris/ai-toolkit` without dragging in local-only workflow code.

## Decision (2026-07-12): the A/P/C/D stack is retired

The previous stack was **A** (resident sampling) -> **P** (DXGI probe) -> **C**
(bounded training streaming core) -> **D** (native FP8 training). It is
superseded. The four plan docs are archived in `tasks/done/`, along with the
four `MODEL_AGNOSTIC_SUBPLAN_*` docs whose seams the new plan absorbs.

**Why C and D died.** Both were built on `_BouncingLinearFn`, the per-linear
streaming autograd function. PR D's own plan said it outright: the FP8-native
training path "is a branch inside `_BouncingLinearFn` - without C's hooks there
is nowhere for this code to run." Commit `3dd7f38` retired that backend; the
immutable runtime is now the sole transformer backend on the fork's active path.
Upstreaming C would mean upstreaming code the fork no longer runs, with D landing
inside it. **The block-native arena is the payload PR now.**

**Why P was right and is preserved.** PR P (DXGI shared-budget probe + pinned
memory crash guard) was correct and its upstream-consumer audit still holds.
It survives as **Stage 1** of the new plan, with the same bugfix framing.

**Why A was demoted.** PR A was the lead PR because it was independent and easy.
It no longer is: it centers on `inference_resident`, which the arena policy work
rewrites and which the two-timescale residency effort is actively changing. It
becomes a deferred Stage 4, reassessed once that code has one owner.

## The load-bearing correction

An early draft of the arena plan treated the host-memory layer (`pin_manager`,
`vram_budget`, `nvml_meminfo`, `dxgi_meminfo`) as arena support infrastructure
that could be seamed away, degrading to a `mem_get_info`-based planner where the
sensors were unavailable.

**That is wrong, and the degraded path is the crash.** Measured locally:
upstream's offloader cannot run a 12 GB model on a 12 GB card without the
over-pinning crash. `mem_get_info` free is a per-process promise and DXGI
`Budget` is a permission; neither reports true availability. The sensor is not a
policy refinement.

Two consequences:

1. **The host-memory layer is its own PR, and it lands first.** It is not arena
   infrastructure - it is a fix to upstream's *existing* per-linear offloader,
   which pins unboundedly today (`manager_modules.py:204`). It carries its own
   bug report and needs no arena to justify it.
2. **The dependency graph has three tiers, not two.** Both backends depend on the
   host-memory layer; it depends on neither. Putting it inside `arena_offload/`
   would trap the bugfix inside a package upstream has not merged.

```text
host_memory (pin_manager, vram_budget, nvml_meminfo, dxgi_meminfo)
    imports neither backend

arena_offload  -> may import host_memory; must NOT import MemoryManager
MemoryManager  -> may import host_memory; must NOT import arena_offload
```

## The stack

| Stage | Theme | Depends on |
| --- | --- | --- |
| 1 | Pinned-memory budget governance (bugfix framing; was PR P) | nothing |
| 2 | Pre-PR refactor, in-fork only (no PR) | nothing |
| 3 | Arena offload as an additional backend | 1 (hard), 2 |
| 4 | Resident + native-FP8 sampling (was PR A) | deferred |

Stages 1 and 2 run in parallel. Step-by-step execution:
`tasks/open/UPSTREAM_ARENA_EXTRACTION_PLAN.md`.

The arena is an **additional** backend. Upstream's per-linear `MemoryManager`
stays, recognizably unchanged - it is what other upstream models use, and the PR
must not read as a rewrite of it.

## Do the extraction dry run early

The plan's central claim - "the upstream diff is dominated by new arena package
files" - is a *prediction*. It is testable in an afternoon: branch from
`ostris/main`, copy the arena files across, and see what fails to import. That
test is what surfaced the ~2300-line host-memory dependency the first draft had
silently assumed away. Do it before spending eight phases of refactor on an
untested prediction, and let its breakage list re-order the work.

## Constraints

- Every PR must be off by default or preserve current behavior when its flags are off.
- Unsupported GPUs, unsupported models, or missing PyTorch features must fall back cleanly.
  (Exception: an *explicitly selected* arena backend must fail loudly, not fall back
  silently to per-linear - see the plan's Phase 3.)
- Do not include local-only workflow changes.
- Heavy diagnostics are development tools, not upstream product surface. Upstream
  gets only low-noise diagnostics needed to understand which path was selected and why.
- Krea-specific assumptions stay gated or inside model integration code.

## Local-only work to keep out of upstream PRs

- DOP prior disk cache experiments;
- DOP prompt/cache behavior tied to local dataset workflow;
- TE worker changes unless submitted as a separate text-embedding-cache PR;
- `captions.json` workflow helpers;
- local search/replace scripts;
- Hugging Face cache-drive configuration;
- project-specific default paths or job assumptions;
- broad UI restructuring unrelated to the submitted feature;
- heavy per-layer offload profiler unless Ostris asks for it;
- `bounce_pool.py` - a throughput optimization, not part of the Stage 1 crash fix.

## Review hygiene checklist

- Defaults preserve current behavior.
- Feature flags are explicit and independent.
- Unsupported hardware/config falls back cleanly (except explicit arena selection).
- Native FP8 only runs when FP8 tensors are already resident on the correct CUDA device.
- `dxgi_meminfo` no-ops cleanly off Windows; NVML absence degrades without raising.
- No hidden coupling to Krea unless the PR is explicitly a Krea integration.
- No local dataset/workflow code.
- Diagnostics are opt-in or low-noise.
- Python changed files compile; relevant CUDA smoke tests pass.
- If UI changed, TypeScript results reported with pre-existing noise separated.

## Open questions for Ostris

- Does he want the heavy performance/offload diagnostics upstream, or dev-only?
- Should the arena toggles live under existing layer-offload config, or a new
  memory-management section?
- Is a Windows-specific DXGI sensor acceptable in the tree (no-op elsewhere), or
  does he want it behind an optional import / plugin?
- Does he want the arena to land with Krea as first integration, or with a
  smaller model-agnostic example?
- What support matrix does he want for native FP8 paths?
- Post-#930: upstream `qfloat8` becomes Quanto. Native FP8 must either scope to
  torchao-layout tensors with a clean Quanto fallback, or ship dual-layout
  extraction. Decide before the arena PR.

## One-sentence upstream framing

**Stage 1:** "unbounded pinning can hard-crash Windows runs and thrash Linux
hosts; here is the budget probe that prevents it." **Stage 3:** "here is an
optional block-native offload backend that trains a 12 GB model on a 12 GB card,
alongside - not replacing - the existing per-linear one."
