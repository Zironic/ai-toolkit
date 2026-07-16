# Upstream PR strategy for the memory-management work

> **git-bug:** see the ticket "Arena offload: pre-PR refactor and upstream
> extraction". Status lives in the ticket; this file is the strategy.
> Execution plan: `../../tasks/done/UPSTREAM_ARENA_EXTRACTION_PLAN.md`.

> **Terminology note.** The reserve vocabulary was renamed; see the glossary at
> the top of `../../tasks/done/AUTOTUNE_PLAN.md`. In short: **working_reserve** =
> our transient working set (the old "headroom"), **system_reserve** = measured
> uncontrolled VRAM, **wddm_margin** = the cushion above the ~500 MB WDDM churn
> cliff, **usable** = what is left for resident weights + ring.

How to turn the local `faster-dop` memory-management work into an upstreamable
PR for `ostris/ai-toolkit` without dragging in local-only workflow code.

## Decision (2026-07-14): one unified PR

The memory-management work ships as **a single complete PR**, not a stack of
dependent PRs. DXGI/NVML sensing, pin accounting, and the WDDM allocator guard
are included as required safety infrastructure of that PR rather than landing
as a prerequisite PR.

Rationale:

- Stacked dependent PRs impose a real coordination cost (rebases, review
  latency, cross-PR sequencing) that the owner has judged not worth the
  intermediate delivery. This is a deliberate trade-off: the standalone
  crash-fix value of the host-memory layer is given up in exchange for a
  single merge event.
- The safety argument is end-to-end: budget sensing, allocator limits, arena
  construction, residency, and teardown are one argument about whether the
  manager operates safely. Reviewing them together lets the maintainer judge
  the whole, not a sensor patch ahead of its largest consumer.
- The internal boundaries (below) are preserved as **commit structure** inside
  the PR, so the readable dependency sequence survives. Splitting into
  multiple PRs becomes a fallback only if the maintainer explicitly asks.

"Complete unit" means the validated memory-manager feature plus everything its
safety and operation require - and nothing else. The local-only exclusion list
below still applies in full.

## Decision (2026-07-12): the A/P/C/D stack is retired

The previous stack was **A** (resident sampling) -> **P** (DXGI probe) -> **C**
(bounded training streaming core) -> **D** (native FP8 training). It is
superseded; its iteration plans were removed after the current rules moved to
the Arena contract and focused guidance.

**Why C and D died.** Both coupled native FP8 training to the C/D smart-training
orchestration around `_BouncingLinearFn`. That orchestration is retired from the
fork's active transformer-training path, which now uses the immutable Arena.
The plain per-linear legacy backend itself remains supported for upstream
compatibility and capabilities Arena does not own, especially text-encoder
offload. Upstreaming C would still mean upstreaming smart machinery the active
Arena path does not run, with D landing inside it. **The block-native arena is
the payload now.**

**Why P was right and is preserved.** PR P (DXGI shared-budget probe + pinned
memory crash guard) was correct and its upstream-consumer audit still holds.
It survives as the **host and WDDM safety foundation** - the first commit group
of the unified PR (it was briefly planned as a standalone lead PR; the
2026-07-14 decision folded it in).

**Why A was demoted.** PR A was the lead PR because it was independent and easy.
It no longer is: it centers on `inference_resident`, which the arena policy work
rewrites and which the two-timescale residency effort is actively changing. It
stays deferred follow-up work, reassessed once that code has one owner.

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

Two consequences (packaging updated for the unified PR, dependency facts
unchanged):

1. **The host-memory layer is the first commit group, and it stands on its own
   feet.** It is not arena infrastructure - it also fixes upstream's *existing*
   per-linear offloader, which pins unboundedly today
   (`manager_modules.py:204`). The unified PR's body should present it as
   independently load-bearing, so the maintainer can see it is not
   arena-shaped scaffolding.
2. **The dependency graph has three tiers, not two.** Both backends depend on
   the host-memory layer; it depends on neither. This stays an import-boundary
   rule inside the tree even though everything ships in one PR.

```text
host_memory (pin_manager, vram_budget, nvml_meminfo, dxgi_meminfo)
    imports neither backend

arena_offload  -> may import host_memory; must NOT import MemoryManager
MemoryManager  -> may import host_memory; must NOT import arena_offload
```

## PR contents and internal structure

The unified PR is Krea2-scoped: an optional block-arena backend whose generic
dispatcher intercepts selected ordinary block calls and invokes their saved
installed forwards over arena-provided state. Krea2 retains its ordinary
transformer forward and model-owned checkpoint loop. The PR does not claim
arbitrary-model support. A second production architecture is deferred until
after this Krea2-scoped extraction is accepted.

The PR adds, as one unit:

- DXGI non-local budget sensing, NVML physical-free sensing, centralized pin
  accounting, and the WDDM allocator guard;
- transactional canonical arena storage, block discovery, state accounting,
  residency, transfer, and lifecycle ownership;
- the generic saved-forward dispatcher and block compile boundary;
- optional cross-process `torch.compile` MegaCache persistence around that
  boundary, with cache misses falling back to ordinary compilation;
- the quantization storage/substitution support the backend requires;
- the minimal Krea2 loader, checkpointing, trainer, sampling, configuration,
  and teardown integration needed to demonstrate the feature end to end;
- focused host-memory, dispatcher, lifecycle, deterministic numerical, and
  production-shaped acceptance evidence.

Arranged as reviewable commit groups, in dependency order:

| Commit group | Theme | Depends on |
| --- | --- | --- |
| 1 | Host and WDDM safety foundation (was PR P) | nothing |
| 2 | Generic arena and dispatcher core | 1 |
| 3 | Krea2 and trainer integration | 2 |
| 4 | Optional MegaCache session and trainer lifecycle hooks | 2, 3 |
| 5 | Acceptance tests, runnable validation instructions, documentation | 4 |

### MegaCache upstream assessment (2026-07-16)

Recommendation: include the portable MegaCache slice as commit group 4. A
fresh FP8 dynamic compile is large enough to dominate startup, while the cache
lifecycle is narrow and does not own Arena residency or model math. The local
full-model matrix and `Mixed -> Mixed -> Full -> Full -> Mixed` benchmark prove
that the pure block kernel reuses the same compiled entries across residency
changes. The contract and measurements are in `MEGACACHE.md`.

The upstream slice is deliberately smaller than the fork integration:

- `toolkit/compile_cache.py` artifact/session logic and stable model/compiler
  identity;
- one load point before the first lazy compiled invocation and save points
  after new variants/before teardown in the trainer lifecycle;
- stable FP8/custom-pass compile identities required for cache hits;
- focused nonfatal-miss, cache-key, cold/warm, and numerical-parity tests;
- one explicit upstream config flag/directory and low-noise logging.

Keep fork-specific output paths, ancillary smoke plumbing, caption/reference
generator expansion, and the guided-UI checkbox out of the first upstream
diff. This fork defaults persistence on for an already compiled workload, but
the upstream extraction stays opt-in unless the maintainer explicitly accepts
the changed disk-write default. Unsupported Torch builds must no-op cleanly.

Estimated work after the Arena compile seam exists is 3-5 focused engineering
days: about one day for extraction/version gating, one for trainer/Krea2
integration, one for tests/docs/acceptance, and up to two for Torch-version and
Windows Triton-bundle compatibility or maintainer-requested config changes.
This is not blocked by residency policy. It is blocked only by the portable
Arena/FP8 compile identity landing first. If PR size becomes the deciding
constraint, commit group 4 can become an immediate follow-up without weakening
Arena correctness.

The in-fork pre-PR refactor (old Stage 2) still happens first and is not part
of the PR diff. Deferred follow-ups (resident + native-FP8 sampling, second
architecture) are unchanged. Step-by-step execution:
`tasks/done/UPSTREAM_ARENA_EXTRACTION_PLAN.md`.

The arena is an **additional** backend. Upstream's per-linear `MemoryManager`
stays, recognizably unchanged apart from adopting the host-memory safety layer -
it is what other upstream models use, and the PR must not read as a rewrite of
it. The new block-native backend is explicit and off by default.

## Do the extraction dry run early

The plan's central claim - "the upstream diff is dominated by new arena package
files" - is a *prediction*. It is testable in an afternoon: branch from
`ostris/main`, copy the arena files across, and see what fails to import. That
test is what surfaced the ~2300-line host-memory dependency the first draft had
silently assumed away. It matters *more* under the unified plan, because the
unit boundary now defines the whole PR. Do it before spending eight phases of
refactor on an untested prediction, and let its breakage list re-order the work.

## Constraints

- The PR must be off by default: every touched behavior preserves current
  upstream semantics when the new flags are off.
- Unsupported GPUs, unsupported models, or missing PyTorch features must fall back cleanly.
  (Exception: an *explicitly selected* arena backend must fail loudly, not fall back
  silently to per-linear - see the plan's Phase 3.)
- Do not include local-only workflow changes.
- Heavy diagnostics are development tools, not upstream product surface. Upstream
  gets only low-noise diagnostics needed to understand which path was selected and why.
- Krea-specific assumptions stay gated or inside model integration code.

## Local-only work to keep out of the upstream PR

- DOP prior disk cache experiments;
- DOP prompt/cache behavior tied to local dataset workflow;
- TE worker changes unless submitted as a separate text-embedding-cache PR;
- `captions.json` workflow helpers;
- local search/replace scripts;
- Hugging Face cache-drive configuration;
- project-specific default paths or job assumptions;
- fork-wide default-on cache policy, ancillary smoke cache plumbing, and the
  guided-UI MegaCache checkbox;
- broad UI restructuring unrelated to the submitted feature;
- heavy per-layer offload profiler unless Ostris asks for it;
- `bounce_pool.py` - a throughput optimization, not part of the safety
  foundation's crash fix;
- speculative second-model work.

## Review hygiene checklist

- Defaults preserve current behavior.
- Feature flags are explicit and independent.
- Unsupported hardware/config falls back cleanly (except explicit arena selection).
- Native FP8 only runs when FP8 tensors are already resident on the correct CUDA device.
- `dxgi_meminfo` no-ops cleanly off Windows; NVML absence degrades without raising.
- No hidden coupling to Krea outside the Krea2 integration commits.
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
- (If he asks for a split:) fall back to landing the host and WDDM safety
  foundation as its own bugfix PR first - the commit structure already
  supports cutting there.

## Resolved questions

- **Post-#930 Quanto vs torchao FP8 layouts (resolved 2026-07-14):** neither
  "torchao-only with fallback" nor a fork - both backends normalize into one
  semantic `Fp8LinearSpec` via per-backend construction adapters (TorchAO
  `Float8Tensor`, Quanto `QBytesTensor`), and native qualification dispatches
  on the spec, never backend identity. Implemented and focus-tested; see
  `../../tasks/done/OSTRIS_ARENA_QUANTIZATION_PLAN.md` and ticket `c9ee48d`.

## One-sentence upstream framing

"An optional block-native offload backend that trains a 12 GB model on a 12 GB
card, alongside - not replacing - the existing per-linear one; it brings the
pinned-memory budget governance that also stops the existing offloader's
unbounded pinning from hard-crashing Windows runs and thrashing Linux hosts."
