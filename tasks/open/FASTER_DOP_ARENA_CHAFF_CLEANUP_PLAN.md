# Faster-DOP Arena Chaff Cleanup Plan

> Durable cleanup plan. Mutable progress, the current audit checklist,
> uncertain cases, and validation results belong on git-bug ticket `d1bd2c9`.
> The temporary deletion-review ledger is
> `tasks/open/FASTER_DOP_ARENA_CLEANUP_CANDIDATES.md`; reconcile it with the
> final diff and delete it when the cleanup closes.

## Outcome

Remove obsolete residue accumulated across the Arena's implementation and
debugging iterations so `faster-dop` reflects the current system rather than
the full history of how it was built.

Review Arena-related production code, integrations, flags, tests, diagnostics,
scripts, configurations, fixtures, plans, and repository artifacts. Keep items
with a present purpose; delete or consolidate items that support only retired
approaches, completed migrations, superseded experiments, or duplicated
knowledge.

A starting commit and clean worktree are execution safeguards. They are not the
outcome. Diff size is not a success criterion.

## Architectural anchor

The current implementation establishes these boundaries:

- the immutable block-native Arena and generic saved-forward dispatcher are the
  supported Arena backend;
- shared host-memory and WDDM safety remain load-bearing infrastructure;
- the old C/D smart-training, prefetch, trace, block-stream, and pinned-arena
  extensions are retired, while the plain per-linear legacy backend, including
  `_BouncingLinearFn`, remains supported;
- architecture-specific Arena execution adapters, runtime-owned model trunks,
  and handwritten complete-model forwards are retired in favor of ordinary
  model-owned forwards and saved installed block forwards;
- the plain legacy `MemoryManager` remains a separate backend for upstream
  compatibility and toolkit features the Arena does not yet own, including
  text-encoder offload;
- current model, quantization, compile, and trainable-adapter variants inside
  the supported ownership model are legitimate and must not be confused with
  retired architecture-execution adapters.

Open tickets and plans are discovery leads, not proof of present utility.
Callers, supported workflows, and an actual current decision determine whether
their associated material remains useful.

Repository-supported toolkit workflows define the compatibility boundary. An
API, alias, flag, or behavior retained only for an undocumented external caller
or one of this repository's previous iterations has no compatibility claim.
Persisted jobs from those unsupported iterations do not require migration or
tolerant loading; current configuration and current saved-job workflows must
continue to parse normally.

## Decision rule

### Keep

Keep an Arena-related item when it serves at least one current purpose:

1. It is used by the supported Arena backend, the separate legacy backend, or
   required shared safety infrastructure.
2. It validates or diagnoses a current supported behavior.
3. It supports active Arena work with a current caller, workflow, or decision.
4. It preserves a durable decision or conclusion not captured more clearly
   elsewhere.
5. It is a reusable repository tool with a recurring workflow and a current
   input/output contract.

### Delete or consolidate

Delete or consolidate an item when no keep rule applies and it:

1. supports only a retired implementation or abandoned integration path;
2. is a one-time editor, migration, probe, repair, replay, or benchmark whose
   change or conclusion has landed;
3. has a superseded result whose durable conclusion is preserved elsewhere;
4. duplicates a maintained module, test, script, document, config, or fixture;
5. refers to architecture, APIs, terminology, flags, or workflows that no
   longer exist;
6. is generated or reproducible data retained as source material; or
7. is compatibility or defensive machinery for an unsupported iteration.

### Defer

Retain and list genuinely uncertain items with the concrete missing fact needed
to decide them. An uncertain item blocks completion only when it sits on a
claimed active or retired execution path, or prevents validation of the
surviving architecture. Other uncertain items may remain as narrowly scoped
follow-up work.

Do not preserve a whole directory or diagnostic family because one item in it
remains useful.

## Bounded candidate set and stopping rule

The initial candidate set is finite:

1. the active Arena and shared host-memory packages and their direct production
   callers;
2. files referring to retired Arena symbols, flags, terminology, or workflows;
3. Arena-related tests, scripts, configs, fixtures, plans, and diagnostics;
4. known root one-shot editors, tracked worktree entries, generated outputs,
   caches, captures, logs, and temporary files;
5. files reached by following references from the preceding candidates when
   needed to decide retention or repair a live reference.

Do not audit unrelated repository content merely because it might contain old
material. Follow dependencies outward only far enough to classify a candidate
or validate the surviving architecture.

Discovery is complete when every item in this bounded set has a keep,
delete/consolidate, or defer disposition, reference-following yields no new
in-scope candidate, and blocking uncertain items are resolved. Maintain one
compact current checklist on ticket `d1bd2c9`; replace obsolete entries instead
of appending an investigation diary.

## Cleanup sequence

### 1. Establish the intended live surface

Trace the current Arena from user-facing configuration through model and
trainer integration, public runtime APIs, construction, dispatch, transfer,
residency, sampling, teardown, and shared host-memory safety.

For each subsystem, identify:

- production entry points and current callers;
- tests establishing its supported contract;
- scripts used by a recurring diagnostic or acceptance workflow;
- current work that materially depends on it;
- the authoritative durable document or minimal set of documents covering its
  current contract and non-obvious decisions.

Record this as the ticket's current subsystem checklist. It is a decision aid,
not a new permanent inventory document.

Perform cleanup in a clean worktree created from a recorded starting commit.
Do not disturb the existing dirty checkout.

### 2. Remove superseded runtime and integration paths

Audit production code for earlier Arena generations and abandoned seams,
including:

- per-linear bouncing, pinned-arena, and in-graph streaming paths no longer
  owned by the supported block-native backend;
- obsolete complete-forward architecture adapters, runtime-owned model trunks,
  adapter variants, and compatibility shims superseded by the current
  model-owned forward and saved-forward dispatcher;
- Arena-specific branches left inside the legacy manager or trainer after
  ownership moved elsewhere;
- duplicate construction, residency, transfer, lifecycle, quantization, or
  compile helpers retained from predecessor designs;
- obsolete metrics, reports, state fields, environment switches, and fallback
  behavior created for retired experiments;
- stale config and UI flags selecting or tuning unsupported behavior.

Candidate names are search seeds, not automatic deletions. Follow imports and
runtime callers before removal. When a retired path contains a unique current
safety rule, move that rule to the active owner and delete the obsolete path.

The target is one supported Arena backend and ownership model, with only its
explicit current execution variants, plus the separately owned legacy manager
and their narrow shared safety dependencies.

### 3. Remove obsolete tests with their retired subjects

Classify tests by the current contract they protect.

- Keep behavior, lifecycle, numerical, safety, and integration tests for
  surviving behavior.
- Delete tests whose only subject is a removed implementation detail or retired
  backend path.
- Move any unique user-visible regression assertion into a current
  contract-level test before deleting the old test.
- Consolidate overlapping tests produced by successive iterations when one
  focused test covers the surviving contract.
- Remove fixtures, global-state shims, and helpers used only by deleted tests.

Do not retain dead production seams merely to keep their tests green.

### 4. Prune iteration-specific scripts, configs, and data scaffolding

Keep scripts only when they support a current recurring diagnostic, maintained
acceptance smoke, active workflow, or reusable repository operation. Delete or
consolidate:

- one-shot source editors and migration helpers;
- probes, captures, replays, simulations, and benchmarks for retired paths or
  settled questions;
- near-duplicate scripts superseded by a maintained smoke or digest tool;
- hard-coded local job helpers with no current workflow;
- stale examples and flags for removed behavior;
- prompt sets, captures, tables, caches, and output scaffolding without a
  current test or diagnostic consumer.

If one file in an experiment area remains useful, retain that file and its
minimal inputs rather than the whole area. Reusable infrastructure such as
`scripts/exact_edit.py` and the git-bug wrappers remains unless a maintained
replacement exists.

### 5. Consolidate plans and documentation

The working tree should explain the current system rather than every route used
to build it.

- Keep the current Arena contract and stable decisions needed to understand the
  implementation.
- Extract unique durable conclusions from superseded plans into the
  authoritative current document or minimal document set.
- Delete duplicate, contradicted, or fully superseded iteration plans after
  consolidation.
- Delete stale run instructions, flag descriptions, architecture diagrams, and
  references to removed scripts or APIs.
- Keep open plans only for genuinely active work and ensure each points to its
  live ticket.

`tasks/done/` is not an automatic retention boundary for this cleanup. Git
history preserves discarded narratives once their still-useful decisions are
consolidated.

### 6. Remove repository-structure residue

Delete the known root `.codex_*.py` one-shot editors and similar temporary
payloads found within the bounded candidate set.

Before touching tracked `.worktrees/*` gitlinks, record:

- `git worktree list --porcelain`;
- each affected path's existence;
- its registered branch and commit.

Remove only the gitlinks from the cleaned index. Do not delete or unregister
the live worktrees. Add precise shared ignore rules for those paths and other
recurring artifact classes actually removed.

Afterward, verify the same paths, branches, and commits remain registered and
intact while the gitlinks are absent from the cleaned index.

### 7. Resolve blocking uncertainty

Resolve uncertain items that affect a claimed active or retired execution path,
or prevent validation. Retain other uncertain items and record them as narrow
follow-up candidates rather than allowing an obscure diagnostic to make the
cleanup indefinite.

## Focused validation

Validation should establish that supported behavior still works after obsolete
paths and scaffolding are removed.

Always:

- compile surviving changed Python modules;
- verify removed modules, symbols, flags, script names, and terminology have no
  unintended live references within the bounded candidate set;
- run focused tests for each surviving contract touched by a deletion;
- inspect each commit and the final diff for unrelated changes;
- confirm every candidate has a current disposition on the ticket checklist.

Conditionally:

- when shared manager or trainer integration changes, run one focused
  Arena-enabled test and one focused Arena-disabled or legacy-manager test;
- when config or UI flags change, verify the default configuration still
  selects the pre-existing non-Arena behavior;
- when host-memory safety changes, validate its backend-independent contract
  and the relevant integration seam;
- when explicit Arena selection changes, verify unsupported configurations
  still fail or fall back according to the current documented contract;
- when runtime deletion could affect GPU execution, run only the narrow
  maintained CUDA seam required by the repository CUDA-testing policy.

Do not run the full suite as a generic confidence ritual. Do not recreate tests
for deleted internal structures.

Report the resulting simplification descriptively: removed production modules
or branches, consolidated tests, deleted scripts and plans, removed stale
flags, and deliberately retained active surfaces. Counts and line reductions
are evidence, not targets.

## Commit structure

Each cleanup commit must be self-consistent and pass the validation relevant to
its own tree. Organize commits by coherent retired subsystem, not by file type:

1. retire one runtime or configuration path together with its directly
   associated tests, fixtures, callers, and live references;
2. retire the next coherent subsystem in the same manner;
3. consolidate remaining diagnostics, scripts, experiment inputs, and
   documentation around the surviving system;
4. remove repository-structure artifacts and add precise ignore rules.

The actual number and order of commits follow discovered subsystem boundaries.
Do not create a special commit merely to record a baseline SHA.

## Non-goals

- Building a comprehensive provenance ledger or repository-wide archaeology.
- Preserving every completed experiment or implementation narrative.
- Deleting useful material to meet a size or line-count target.
- Refactoring active code beyond what is needed to remove obsolete seams or
  consolidate duplicated ownership.
- Auditing unrelated active projects.
- Running broad validation that cannot change a cleanup decision.

## Acceptance criteria

- Every item in the bounded candidate set has a current keep,
  delete/consolidate, or defer disposition.
- No known retired Arena entry point, backend owner, configuration selector, or
  live reference remains within that set.
- The supported Arena backend, its explicit current variants, the separate
  legacy manager, and shared host-memory safety retain clear ownership.
- Tests validate surviving contracts rather than deleted implementations.
- Iteration-specific scripts, configs, fixtures, captures, and data scaffolding
  without a current workflow are gone.
- Superseded plans and stale references are removed after unique durable
  conclusions are consolidated.
- Default-off and legacy behavior is focused-tested whenever shared integration
  changed.
- All affected live worktrees remain registered at their original paths,
  branches, and commits; only their tracked gitlinks are removed.
- Blocking uncertain items are resolved. Any retained non-blocking uncertainty
  is narrowly listed for follow-up.
- Each cleanup commit is self-consistent, and focused validation establishes
  that retained active workflows still import, run, or test correctly.
- Simplification is reported descriptively; diff size is not an acceptance
  criterion.
