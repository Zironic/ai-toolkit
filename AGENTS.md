# AGENTS.md

Guidance for Codex agents working in this repository.

## Repository Context

This is a fork of `ostris/ai-toolkit` focused on memory-management and
offload-streaming work for training large diffusion models on 12 GB consumer
GPUs, especially Windows/WDDM systems. Most local work is in
`toolkit/memory_management/`, its trainer wiring, Krea2 integration, diagnostics,
and UI/config flags.

`output/` and `datasets/` are Windows junctions shared with the sister checkout.
Use these canonical paths in commands and docs:

- `C:\GenAI\ai-toolkit\output`
- `C:\GenAI\ai-toolkit\datasets`

They currently resolve to the sister checkout's storage directories, but do not
use the resolved target paths in commands unless the user explicitly asks. Only
those two directories are shared; code paths, configs, git state, and other repo
files are not. For code work, stay anchored to this checkout's paths.

Keep upstream-general changes cleanly separable from local-only workflow code.
See `docs/decisions/UPSTREAM_PR_PLAN.md` before preparing upstream-oriented
changes.

## Workspace Conventions

Put throwaway helper scripts (e.g. edit scripts, one-off probes) in the Windows
temp folder (`$env:TEMP`) or the repo-local, gitignored `.agent/tmp/`, never the
repo root. Use `/tmp` only when actually running inside a Linux sandbox session
where no Windows temp location exists.

### Read-only sandbox commands

Run routine read-only repository and diagnostic commands in the normal sandbox
first, including reads through the canonical `output/` and `datasets/` paths.
Do not request elevated execution preemptively. Escalate only after an actual
filesystem or sandbox denial and only when the read cannot be expressed through
a sandbox-compatible command.

Native Windows sandbox commands run in constrained PowerShell language mode.
For read-only summaries, avoid non-core casts such as `[pscustomobject]`, which
fail in that mode even when every file read is allowed. Use plain strings,
hashtables with `ConvertTo-Json`, or a focused Python diagnostic instead. Keep
independent Git/read operations as simple command segments when practical so
the existing prefix rules can recognize them.

### ASCII only

Use plain ASCII in source files and in anything passed through PowerShell
(commands, arguments, here-strings, commit messages). Windows PowerShell 5.1
mangles non-ASCII on the way through, and mixed encodings corrupt files. Write
`->` not `→`, `"..."` not smart quotes, `~2x` not `≈2×`. Existing non-ASCII in
files you edit may stay; just don't introduce more. (Markdown docs edited via a
proper file tool — not the shell — are the one place non-ASCII is tolerated.)

### Preferred edit method

If your native patch/edit tool fails (anchor mismatch, encoding, escaping), do
not fall back to shell-escaped multiline replacement commands. Use the
repo-local helper `scripts/exact_edit.py` instead — it reads old/new text from
files (so nothing passes through shell quoting), fails if the anchor is not
found, and preserves the file's newline style:

```powershell
venv\Scripts\python.exe scripts\exact_edit.py replace path\to\file.py old.txt new.txt
venv\Scripts\python.exe scripts\exact_edit.py splice path\to\file.py --start "def f(" --end-before "def g(" new.txt
venv\Scripts\python.exe scripts\exact_edit.py insert path\to\file.py --anchor "import os" new.txt --where after
```

Put the `old.txt`/`new.txt` helper files in the temp folder, not the repo.

## Planning And State

The Markdown docs are the primary durable planning/design docs. They describe the
plan, rationale, architecture, and acceptance criteria.

Mutable state and priority live in git-bug tickets:

- what is done
- what is currently blocked
- what needs a run
- validation results
- priority/order
- next-agent handoff notes

Do not turn planning docs into running status logs. If the design changes, update
the relevant Markdown plan. If status changes, update or create the relevant
git-bug ticket.

Important locations:

- `tasks/open/` - active durable plans/TODOs
- `tasks/done/` - completed durable plans
- `docs/decisions/` - stable rationale and strategy docs
- `docs/TICKETS.md` - git-bug workflow and current seed tickets

## git-bug

Use git-bug for ticket state. The binary is repo-local and ignored by git:

```powershell
.\tools\git-bug.exe bug --format plain
.\tools\git-bug.exe bug show <id>
.\tools\git-bug.exe bug new --title "Title" --message "Body" --non-interactive
.\tools\git-bug.exe bug comment new <id> --message "Update" --non-interactive
.\tools\git-bug.exe bug status close <id>
```

Convenience wrapper:

```powershell
.\scripts\tickets.cmd list
.\scripts\tickets.cmd show <id>
.\scripts\tickets.cmd comment <id> "Update"
.\scripts\tickets.cmd close <id>
```

Codex sandbox sessions may need approval for git-bug commands because git-bug
stores state under `.git/git-bug`. Normal terminals should not need approval.

Do not use `git-bug webui` — in git-bug v0.10.1 it holds the `.git/git-bug` store lock (even with `--read-only`) for its whole lifetime, blocking CLI ticket work. Browse tickets instead via the repo-local lock-free viewer: the **Tickets page (`/tickets`)** in the `ui/` app, which reads `refs/bugs/*` directly with read-only git and never takes the store lock. Use the CLI/wrapper only for writes.

## Code Map

- `toolkit/memory_management/arena_offload/` - generic block-native arena
  dispatcher (the primary offload runtime): planner, dispatcher, transfer,
  load session, FP8, cap calibration.
- `toolkit/memory_management/immutable_runtime.py` - compile-neutral immutable
  source/residency runtime shared by training and sampling.
- `toolkit/memory_management/pin_manager.py` - single authority for pinned
  host memory (priority, accounting, eviction).
- `toolkit/memory_management/vram_budget.py` - NVML-backed free-VRAM and
  allocator-budget sensors.
- `toolkit/memory_management/allocator_cap.py` - WDDM hard allocator cap.
- `toolkit/memory_management/manager.py` - legacy planner, smart attach, live
  autotune, promote/demote, sampling restore, memory safety.
- `toolkit/memory_management/manager_modules.py` - per-module streaming forward,
  trace lifecycle, block staging, FP8 helpers.
- `toolkit/memory_management/bounce_pool.py` - pageable-to-pinned worker pool,
  prefetch schedules, trace resync.
- `toolkit/memory_management/checkpoint_autotuner.py` - checkpoint keep-last
  controller.
- `toolkit/compile_cache.py` - default-on, best-effort cross-process
  `torch.compile` MegaCache lifecycle and stable model/compiler identity.
- `extensions_built_in/diffusion_models/krea2/` - Krea2 model integration.
- `jobs/process/BaseSDTrainProcess.py` - training loop and step-boundary wiring.
- `extensions_built_in/sd_trainer/SDTrainer.py` - trainer execution.
- `toolkit/config_modules.py` - config classes and `layer_offloading_*` flags.
- `scripts/` - digest/replay/benchmark tooling.
- `tests/` - focused memory-management tests and simulations.
- `ui/` - Next.js web UI.

## Training Configuration Rule

Environment variables do not exist in actual training runs launched through the
web UI. Do not rely on env vars for runtime behavior, memory policy, offload
policy, model behavior, or user-facing tuning outside of tests and one-off debug
scripts. Any behavior that must affect a real training job must be exposed
through the job config and, when applicable, the UI/config schema. Env vars are
acceptable only as test harness controls, local diagnostics, or temporary debug
overrides that are not required for normal training.

## Validation

Tests are evidence-gathering tools, not a ritual. Run a test only when there is
a concrete question it can answer about the change or diagnosis at hand. Before
running it, be able to state what behavior it exercises and how a pass or failure
would affect the next decision. If the result would not change the assessment,
skip the test.

Use the narrowest useful validation: a targeted test case or file, a syntax
check for edited Python, or a focused synthetic CUDA script. Do not run the
entire `tests/` suite by default, as a generic confidence check, or merely
because code changed. A full-suite run is appropriate only when the change is
genuinely cross-cutting, the user explicitly requests it, or a specific release
gate requires it. Do not expand into unrelated tests after focused validation
passes unless there is evidence of a broader interaction.

Documentation, agent-instruction, comment-only, and similarly non-executable
changes normally require inspection or diff review, not test execution. Prefer
focused tests and synthetic CUDA scripts over full training jobs when runtime
validation is actually warranted.

### No CPU compilation

CPU compilation is not a supported or useful validation path in this
repository. The relevant `torch.compile` work targets CUDA block kernels on the
GPU. Do not intentionally compile CPU functions or tensors, do not add a CPU
compile fallback, and do not install or require MSVC/`cl.exe` to satisfy
PyTorch Inductor's CPU code-generation probes.

On Windows, a CUDA `torch.compile` smoke may still enter an incidental Inductor
CPU capability/vector-ISA probe and fail with errors such as
`Compiler: cl is not found`. Treat that as an irrelevant CPU-probe/toolchain
failure, not evidence that the CUDA kernel needs CPU compilation. Do not pursue
the CPU path or change production code around it. Use a CUDA-only focused smoke
that avoids the probe, or report the full-model compiled smoke as blocked by
the incidental CPU probe while continuing CUDA validation through the focused
CUDA seam.

Examples of focused validation commands (choose only those relevant to the
question being answered):

```powershell
venv\Scripts\python.exe -m pytest tests\test_bounce_pool.py -q
venv\Scripts\python.exe -m py_compile toolkit\memory_management\manager.py toolkit\memory_management\manager_modules.py
venv\Scripts\python.exe scripts\digest_perf_log.py output\...\performance_log.jsonl
venv\Scripts\python.exe scripts\replay_prefetch_trace.py output\...\prefetch_capture.jsonl
```

Full training runs are minutes to hours and should only be started when the user
asks. The user launches real jobs through the web interface. Treat env vars as
nonexistent for real training: required runtime behavior must be wired through
config/UI, not hidden behind environment variables.

### Do not hunt test-order leaks

Some `tests/` files pass alone and fail only in the full suite. The known cause
is process-global state -- the pin ledger, CUDA allocator, and the
`qfloat8`->torchao shim all outlive a test. Chasing these is slow and almost
never finds a product bug.

The rule: **confirm it is order-dependent, note it on ticket `f2aceba`, move
on.** Confirming is two runs -- the file alone (passes) and the suite with the
suspect file ignored (passes). That distinguishes a real regression from a leak,
which is the only thing worth knowing. Do not bisect further, do not instrument
`sys.modules`, do not restructure other people's tests to isolate it.

If a test *you are adding* triggers one, prefer testing the seam directly over
driving the whole machine: a test that only needs to prove a helper collects the
right entries should not build real pinned packs.

## Memory-Management Principles

- Windows/WDDM cliff behavior matters: avoid policies that drive driver-free
  VRAM to zero.
- Prefetch hints may be opportunistic; memory-safety decisions must be
  conservative and validated.
- Residency/layout changes should rebuild transfer plans without destroying
  durable execution traces unless execution order actually changed.
- Arena residency and transfers stay outside the pure compiled block kernel;
  Mixed and Full plans share guarded MegaCache entries. See
  `docs/decisions/MEGACACHE.md` before changing the dispatcher ABI, FP8 compile
  identity, cache key, or custom Inductor passes.
- Unsupported hardware/models must fall back cleanly when flags are off or
  unsupported paths are requested.
- Add focused tests for memory-manager behavior; GPU CI is not available here.
