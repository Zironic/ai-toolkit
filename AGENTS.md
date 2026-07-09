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

In Codex sandbox sessions the project temp folder is `/tmp`. In normal Windows
terminals use the standard Windows temp location instead. Either way, put
throwaway helper scripts (e.g. edit scripts, one-off probes) in the temp folder,
not the repo root.

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

- `toolkit/memory_management/manager.py` - planner, smart attach, live autotune,
  promote/demote, sampling restore, memory safety.
- `toolkit/memory_management/manager_modules.py` - per-module streaming forward,
  trace lifecycle, block staging, FP8 helpers.
- `toolkit/memory_management/bounce_pool.py` - pageable-to-pinned worker pool,
  prefetch schedules, trace resync.
- `toolkit/memory_management/checkpoint_autotuner.py` - checkpoint keep-last
  controller.
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

Prefer focused tests and synthetic CUDA scripts over full training jobs.

Useful commands:

```powershell
venv\Scripts\python.exe -m pytest tests\test_bounce_pool.py -q
venv\Scripts\python.exe -m pytest tests\ -q
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
- Unsupported hardware/models must fall back cleanly when flags are off or
  unsupported paths are requested.
- Add focused tests for memory-manager behavior; GPU CI is not available here.

