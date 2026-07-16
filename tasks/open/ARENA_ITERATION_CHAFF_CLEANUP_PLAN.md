# Arena Iteration Chaff Cleanup Plan

> Durable two-stage cleanup plan. Mutable progress, uncertain cases,
> validation results, and handoff notes belong on git-bug ticket `d1bd2c9`.

## Outcome

Remove the obsolete code, tests, scripts, diagnostics, configurations, data
scaffolding, plans, and repository artifacts accumulated across the Arena's
many implementation and debugging iterations.

This cleanup has two dependent stages:

1. Clean `faster-dop` until it represents the current Arena system and the
   tooling and knowledge that still support it.
2. Clean the Arena PR using the cleaned `faster-dop` implementation and its
   deletion decisions, followed by an independent audit for PR-only chaff.

The result is not a perfect archival baseline. It is a materially smaller and
clearer repository in which retained Arena-related material has a present
purpose. Git history remains the archaeological record of discarded routes.

## Architectural anchor

The current upstream strategy establishes the first important boundary:

- the immutable block-native Arena and saved-forward dispatcher are the active
  payload;
- the host-memory and WDDM safety layer is shared, load-bearing infrastructure;
- `_BouncingLinearFn` and the old C/D per-linear streaming path are retired;
- the legacy upstream `MemoryManager` remains for non-Arena models, but
  Arena-specific experiments embedded in it are not protected merely because
  the legacy backend still exists.

The published Arena PR is a downstream cleanup target, not the definition of a
minimal implementation. It contains some of the same accumulated chaff and
must not be used to justify retaining material in `faster-dop`.

Current open tickets and plans identify intentionally active local work. A
closed ticket or completed experiment is not an automatic deletion, but it
must have another current purpose to remain.

## Decision rule

### Keep

Keep an Arena-related item when it serves at least one current purpose:

1. It is used by the active Arena implementation or required shared safety
   infrastructure.
2. It validates or diagnoses a current supported behavior.
3. It is required by current Arena or upstream-PR work.
4. It preserves a durable decision or conclusion not captured more clearly
   elsewhere.
5. It is a reusable repository tool with a recurring workflow and a current
   input/output contract.

### Delete or consolidate

Delete or consolidate an item when any of these applies and no keep rule does:

1. It supports only a retired implementation or abandoned integration path.
2. It is a one-time editor, migration, probe, repair, replay, or benchmark whose
   change or conclusion has landed.
3. Its result is superseded and the durable conclusion is preserved elsewhere.
4. It duplicates a maintained module, test, script, document, config, or
   fixture.
5. It refers to architecture, APIs, terminology, flags, or workflows that no
   longer exist.
6. It is generated or reproducible data retained as source material.
7. It is compatibility or defensive machinery for an iteration the project no
   longer supports.

### Defer

Defer only when callers, references, open work, and documentation do not reveal
the current role. Record a short explicit deferred list on the ticket. Do not
default an entire directory or diagnostic family to preservation because one
file remains useful.

Present utility is the primary test. File provenance, a starting SHA, and a
clean worktree are execution aids, not retention criteria or deliverables.

## Scope

Review Arena-related material across:

- production memory-management and integration code;
- configuration and UI flags;
- tests and fixtures;
- smoke, diagnostic, replay, profiling, simulation, and benchmark scripts;
- generated-output and experiment-input scaffolding;
- active and completed plans, decisions, and stale references;
- temporary editors, worktree entries, and local workflow utilities.

Unrelated active projects remain outside this cleanup unless they directly
depend on Arena residue being removed. Fork-local material is neither protected
nor condemned solely because it is local.

## Stage 1 - Clean `faster-dop`

### 1. Establish the intended live surface

Trace the current Arena from user-facing configuration through model/trainer
integration, public runtime APIs, construction, dispatch, transfer, residency,
sampling, teardown, and shared host-memory safety.

For each subsystem, identify:

- the production entry point and current callers;
- the tests that establish its supported contract;
- the scripts used by a current diagnostic or acceptance workflow;
- the active plan or ticket, when work is still in flight;
- the one durable document that explains its non-obvious decisions.

Produce a compact subsystem keep-map on ticket `d1bd2c9`. This is not a
file-by-file provenance ledger. Its purpose is to make the intended current
surface explicit enough that predecessor implementations can be removed.

Use a clean worktree and record its starting commit before editing. Do not
disturb the existing dirty checkout. No additional baseline ceremony is
required.

### 2. Remove superseded runtime and integration paths

Audit production code for earlier Arena generations and abandoned seams,
including:

- per-linear bouncing, pinned-arena, and in-graph streaming paths that are no
  longer part of the active block-native runtime;
- architecture adapters, handwritten execution paths, runtime-owned block
  trunks, and compatibility layers superseded by saved installed forwards;
- Arena-specific branches left inside the legacy manager or trainer after
  ownership moved to the current runtime;
- duplicate construction, residency, transfer, lifecycle, quantization, or
  compile helpers retained from predecessor designs;
- obsolete metrics, reports, state fields, environment switches, and fallback
  behavior created for retired experiments;
- stale config and UI flags that select or tune paths no longer supported.

Candidate names are search seeds, not automatic deletions. Follow imports and
runtime callers before removing them. When a retired path contains a unique
current safety rule, move that rule to the active owner and delete the old
path; do not preserve an obsolete subsystem to retain one useful check.

The target is one recognizable Arena execution path plus the separately owned
legacy manager, with a narrow shared host-memory safety dependency between
them.

### 3. Remove obsolete tests with the code they preserve

Classify tests by the current contract they protect.

- Keep behavior, lifecycle, numerical, safety, and integration tests for the
  surviving Arena.
- Delete tests whose only subject is a removed implementation detail or
  retired backend.
- Move any unique user-visible regression assertion into the current
  contract-level test before deleting the old test.
- Consolidate overlapping tests produced by successive iterations when one
  focused test can cover the surviving contract.
- Remove fixtures, global-state shims, and test helpers used only by deleted
  tests.

Do not keep dead production seams merely to keep their tests green. Tests are
evidence for the supported system, not an archive of every implementation.

### 4. Prune iteration-specific scripts, configs, and data scaffolding

Review Arena-related scripts by workflow rather than filename alone.

Keep only scripts that are used for a current recurring diagnostic, maintained
acceptance smoke, active ticket, or reusable repository operation. Delete:

- one-shot source editors and migration helpers;
- probes, captures, replays, simulations, and benchmarks for retired paths or
  settled questions;
- near-duplicate scripts superseded by a maintained smoke or digest tool;
- hard-coded local job helpers with no current workflow;
- stale example configs and flags for removed behavior;
- prompt sets, captures, tables, caches, and output-directory scaffolding that
  no longer support a current test or diagnostic.

If one file in an experiment area remains useful, retain that file and its
minimal inputs; do not preserve the entire area by association.

Reusable repository infrastructure such as `scripts/exact_edit.py` and the
git-bug wrappers remains unless a maintained replacement exists.

### 5. Consolidate the planning and documentation trail

The working tree should explain the current system, not preserve every
step-by-step route used to build it.

- Keep the current Arena contract and the stable decisions still needed to
  understand the implementation.
- Extract any unique durable conclusion from a superseded plan into the
  current contract or decision document.
- Delete duplicate, contradicted, or fully superseded iteration plans after
  that consolidation.
- Delete stale run instructions, old flag descriptions, dead architecture
  diagrams, and references to removed scripts or APIs.
- Keep open plans only for genuinely active work and ensure each points to its
  live ticket.

`tasks/done/` is not an automatic retention boundary for this cleanup. Git
history preserves discarded implementation narratives once their still-useful
decisions have been consolidated.

### 6. Remove repository-structure residue

Delete the known root `.codex_*.py` one-shot editors and any similar temporary
payloads discovered during the audit.

Remove tracked `.worktrees/*` gitlinks from the index without deleting or
unregistering the live worktrees. Add precise shared ignore rules for those
recurring artifact classes and for any confirmed generated-output class removed
by this cleanup.

Review empty files, caches, logs, captures, and generated outputs by current
utility. Do not build a general inherited-file provenance process around this
step.

### 7. Resolve the small deferred list

After caller and reference analysis, list only genuinely uncertain items on
ticket `d1bd2c9`, with the concrete missing fact needed to decide them. Resolve
those items before declaring Stage 1 complete unless the user explicitly
accepts them as follow-up work.

### 8. Validate the surviving system

Validation should answer whether the supported Arena still works after its
predecessors and scaffolding are removed.

At minimum:

- compile the surviving changed Python modules;
- verify removed modules, symbols, flags, script names, and terminology have no
  unintended live references;
- run focused tests for each surviving contract touched by the deletions;
- run config/UI validation only when flags or schema entries change;
- run the narrow maintained CUDA seam only when runtime deletion could affect
  GPU execution, following the repository CUDA-testing policy;
- inspect the final diff for accidental changes to unrelated active work.

Do not run the full suite as a generic confidence ritual. Do not recreate tests
for deleted internal structures.

Report the simplification achieved: production modules or branches removed,
tests consolidated or deleted, scripts and plans removed, stale flags removed,
and the small set of active surfaces deliberately retained. Line count is
supporting evidence, not the objective.

### 9. Stage 1 deliverable

Commit the cleanup in a few reviewable, behavior-oriented slices, for example:

1. remove retired runtime/config paths;
2. remove or consolidate their tests and diagnostics;
3. prune obsolete plans, scripts, data scaffolding, and repository artifacts.

The exact split follows the dependencies discovered during cleanup. The Stage
1 deliverable is the cleaned `faster-dop` implementation plus a concise ticket
summary of the retained surface, removed subsystems, focused validation, and
any user-accepted deferrals. A special commit whose purpose is only to record a
baseline SHA is unnecessary.

## Stage 2 - Clean the Arena PR from Stage 1

Stage 2 begins only after the `faster-dop` cleanup decisions and surviving
architecture are stable enough to serve as the source of truth.

### 1. Project the Stage 1 cleanup onto the PR

For every subsystem, symbol, flag, test, diagnostic, and document removed or
consolidated in Stage 1:

- find its copied, renamed, split, or inlined counterpart in the PR;
- apply the same current-utility decision unless the upstream context creates a
  concrete additional requirement;
- port the cleaned implementation rather than preserving PR code merely
  because it already passed review or validation;
- update PR integration and tests to match the surviving `faster-dop` design.

The Stage 1 commit range and final tree are evidence for Stage 2, not a blind
cherry-pick recipe. The PR may organize the same behavior differently.

### 2. Audit PR-only chaff independently

The PR can contain residue that no longer has a direct local counterpart.
Review its entire diff against its upstream base and require a current purpose
for every added file and every modified upstream seam.

Remove:

- extraction scaffolding and temporary compatibility code;
- diagnostics or tests retained only to support an earlier PR shape;
- duplicated safety, lifecycle, compile, quantization, or integration logic;
- stale PR documentation and claims;
- local workflow assumptions and heavy diagnostics not required by the
  upstream feature.

Do not stop when the Stage 1 deletions have been mirrored; the independent PR
audit is what catches chaff introduced during extraction itself.

### 3. Re-establish the minimal PR claim

The cleaned PR should contain only what its supported feature requires:

- shared host-memory and WDDM safety;
- the current block-native Arena and saved-forward dispatcher;
- the intended model/trainer/config integration;
- focused tests and runnable validation for the public claim;
- concise documentation of behavior, limitations, and lifecycle.

Every touched upstream file must have a direct integration or safety reason.
Every new file must belong to one of those surfaces. Defaults remain unchanged
when Arena offload is not selected.

### 4. Validate the cleaned PR

Validate the PR as a standalone upstream change rather than assuming Stage 1
coverage transfers automatically:

- inspect the complete diff against the current PR base;
- compile changed Python modules;
- run the focused host-safety, Arena contract, lifecycle, and integration tests
  that remain after consolidation;
- run the maintained deterministic CUDA smoke required for the PR claim;
- verify no retired symbol, stale flag, local path, or deleted diagnostic is
  still referenced;
- verify the PR remains understandable in its intended reviewable commit
  structure.

PR branch rewriting, force-pushing, and external review updates are execution
decisions requiring the user's direction; they are not part of defining the
cleanup.

### 5. Stage 2 deliverable

The final PR is a reduced projection of the cleaned `faster-dop` architecture,
plus only the upstream-specific integration and evidence it needs. Record its
validation and any intentional differences from Stage 1 on the ticket and in
the PR handoff.

## Acceptance criteria

### Stage 1

- `faster-dop` has one clear current Arena path and no production code that
  exists solely for a retired Arena iteration.
- Tests validate surviving contracts rather than removed implementations.
- Iteration-specific scripts, configs, fixtures, captures, and data scaffolding
  without a current workflow are gone.
- Superseded plans and stale references are removed after unique durable
  conclusions are consolidated.
- Temporary editors, tracked worktree gitlinks, and generated repository
  residue are removed and precisely ignored where recurrence is plausible.
- The deferred list is empty or explicitly accepted by the user.
- Focused validation establishes that retained active workflows still import,
  run, or test correctly.
- The resulting diff is materially broader than deletion of the obvious root
  scripts and gitlinks and materially simplifies the repository.

### Stage 2

- Every Stage 1 cleanup decision has been mapped to the PR where applicable.
- PR-only extraction residue has been independently removed.
- Every remaining PR file and upstream modification supports the minimal
  current feature claim, required safety, or focused validation.
- The cleaned PR contains no retired Arena path, stale flag, heavy local-only
  diagnostic, or superseded documentation.
- Focused upstream validation passes and intentional differences from cleaned
  `faster-dop` are documented.
