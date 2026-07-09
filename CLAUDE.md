# CLAUDE.md

Guidance for Claude when working in this repository.

## What this is

A fork of [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) — a training
suite for diffusion models (LoRA + full fine-tune, image/video). Upstream runs as
a CLI (`run.py` over YAML/JSON configs) and a Next.js web UI (`ui/`, port 8675).

**This fork (`faster-dop`) exists for the memory-management / offload-streaming
work** in `toolkit/memory_management/`: streaming quantized weights CPU↔GPU so
large models (Krea2, Z-Image, Anima, etc.) train on a 12 GB consumer card
(RTX 4070, Windows/WDDM). Most local changes orbit that subsystem, its autotune
controllers, and the Krea2 model integration. Keep upstream-general changes
cleanly separable from local-only workflow code (see the upstream-PR decision doc).

## Where things live

- **`toolkit/memory_management/`** — the offload subsystem. `manager.py`
  (planner + live autotune controllers), `manager_modules.py` (per-Linear
  streaming forward, block staging), `bounce_pool.py` (pageable→pinned worker
  pool), `checkpoint_autotuner.py`.
- **`extensions_built_in/diffusion_models/`** — model-specific code; Krea2 lives
  in `.../krea2/`. Model integrations call into the memory manager via
  `attach_smart_training`.
- **`jobs/process/BaseSDTrainProcess.py`** — training loop, step boundary where
  the live controllers run; **`extensions_built_in/sd_trainer/SDTrainer.py`** —
  the main trainer process.
- **`toolkit/config_modules.py`** — config classes (`ModelConfig`, `TrainConfig`,
  …); offload flags like `layer_offloading_*` live on `ModelConfig`.
- **`tests/`** — the memory-management tests (bounce pool, block stream, working
  reserve sim, shape keys, …). **`testing/`** — the general upstream test suite.
- **`scripts/`** — perf/analysis tooling (`digest_perf_log.py` collapses
  `performance_log.jsonl`; `sim_working_reserve_controller.py`; benches).
- **`tasks/open/`** — durable specs/plans for in-flight efforts; **`tasks/done/`**
  — move a spec here when its work fully ships; **`docs/decisions/`** — "why we
  chose X" rationale (e.g. the upstream-PR strategy). See `tasks/README.md`.
- **Division of labour (important):** the `tasks/` and `docs/` **`.md` files are
  the primary, durable planning/design docs** — edit them when the *design*
  changes, not to log progress. **Mutable state** (status, what's done, what's
  blocked) lives in **git-bug** tickets (`refs/bugs/*`). This keeps the planning
  docs stable so they aren't rewritten every work session: read the ticket for
  current state, the `.md` for the plan.

> A stale but occasionally useful training-lifecycle walkthrough exists in the
> sibling checkout at `C:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\docs\training_lifecycle\`
> (highly out of date — treat as inspiration, verify against current code).

## git-bug

Issues use [git-bug](https://github.com/git-bug/git-bug) (v0.10.1), stored in
`refs/bugs/*` in this repo. The binary lives at `tools/git-bug.exe` — a local,
**gitignored** copy, not on PATH (a fresh clone won't have it; download v0.10.1
from the git-bug releases page if missing). The preferred entry point is the
wrapper `.\scripts\tickets.cmd` (`list`, `list-closed`, `show <id>`,
`new "Title" "Body"`, `comment <id> "..."`, `label <id> ...`, `close <id>` —
see `docs/TICKETS.md`). The raw binary works too:

- `./tools/git-bug.exe bug` — list · `bug show <id>` — read
- `./tools/git-bug.exe bug new` — create (`-t` title, `-m` message). Note: with
  `-F <file>` the file's first line becomes the title and `-t` is ignored (git
  commit convention); fix afterwards with `bug title edit <id> -t "..."`.
- `./tools/git-bug.exe bug comment new <id> -m "..."`
- `./tools/git-bug.exe bug status close <id>` / `... open <id>`

Bug refs are **not** moved by normal `git push`/`pull` — use
`./tools/git-bug.exe push` / `pull` or an explicit refspec.

**Browse tickets via the app UI, not `git-bug webui`.** The Tickets page
(`/tickets`, served by `ui/` via `ui/src/server/gitbug.ts`) reconstructs tickets
by reading `refs/bugs/*` directly with read-only git, so it never touches
git-bug's lock. **Do not run `git-bug webui`** — it holds git-bug's single-access
lock for its whole lifetime and blocks all CLI ticket work. Use the CLI above
only for **writes**; if a write ever hangs, make sure no other git-bug process is
running and remove a stale `.git/git-bug/lock` if present, then retry.

## Common commands

```bash
# Memory-management / CUDA tests — through the project venv, hit the real GPU.
# Fine to run directly; these finish in seconds.
venv/Scripts/python.exe -m pytest tests/ -q
venv/Scripts/python.exe -m pytest tests/test_bounce_pool.py -q
venv/Scripts/python.exe scripts/bench_bounce_fill_group.py   # ad-hoc GPU script

# Full training run — minutes to hours. Only when the user asks.
python run.py config/your_config.yaml -n "run_name"

# General upstream test suite (CPU, stubs for diffusers/optimum)
python -m pytest testing -q -k <substring>

# UI (Next.js, port 8675)
cd ui && npm run dev              # hot reload
cd ui && npm test                 # vitest
```

## Job output & perf logs

Live and finished job folders are always at **`output/<job name>/`** in this
repo (`output/` is a Windows junction to the sibling checkout's
`AI-Toolkit-Easy-Install\AI-Toolkit\output\`) — never `find`/grep the whole
drive for `performance_log.jsonl` or `log.txt`, they are always there. Per job
folder:

- `performance_log.jsonl` — the *live* run's perf windows. `logs/{N}_performance_log.jsonl`
  / `logs/{N}_log.txt` — archived from earlier restarts of the same job (numbered).
- `log.txt` — full stdout of the current run; `config.yaml` / `.job_config.json`
  — the resolved job config.

Read perf logs with `scripts/digest_perf_log.py` (accepts a bare job name, a
job folder path, or no arg for the most-recently-updated run under `output/`)
rather than parsing the raw jsonl by hand — see its `--help` for `--last`,
`--all`, `--archived`. To find which job is *currently* running (and confirm
its exact output folder name), check the live `python.exe` process's command
line rather than guessing:
`powershell -Command "Get-CimInstance Win32_Process -Filter \"ProcessId=<pid>\" | Select-Object CommandLine"`
(`run.py` is invoked with the job's `.job_config.json` and `--log log.txt` paths).

## Training Configuration Rule

Environment variables do not exist in actual training runs launched through the
web UI. Do not rely on env vars for runtime behavior, memory policy, offload
policy, model behavior, or user-facing tuning outside of tests and one-off debug
scripts. Any behavior that must affect a real training job must be exposed
through the job config and, when applicable, the UI/config schema. Env vars are
acceptable only as test harness controls, local diagnostics, or temporary debug
overrides that are not required for normal training.

## Working conventions

- **Exercise CUDA directly with small scripts; don't launch full runs.** Real
  CUDA/FP8/memory-manager behaviour can and should be validated on the actual GPU
  with short synthetic scripts through `venv/Scripts/python.exe` — synthetic
  models, a `_scaled_mm` call counter, `torch.cuda.synchronize()` before timing
  (see the CUDA-testing methodology in memory, and `tests/` for the pattern). A
  focused GPU test that runs in seconds is the normal way to check this code — no
  need to ask first. What *does* need the user's go-ahead is a **full training
  run** (`python run.py …`): minutes-to-hours, real datasets/checkpoints, and not
  something a unit test substitutes for.
- **Windows/WDDM reality shapes the memory work.** ~500 MB WDDM churn cliff near
  full VRAM; `expandable_segments` unsupported; `PYTORCH_CUDA_ALLOC_CONF`
  tuning caused ~30× slowdowns. Don't reintroduce those knobs.
  - **Two distinct memory cliffs, different failure modes.** Crossing the
    *dedicated* VRAM ceiling makes WDDM silently page GPU memory to system RAM —
    catastrophic slowdown, **no error** (governed by `torch.cuda.mem_get_info`).
    Exhausting the *shared* (DXGI NON_LOCAL) budget — a slice of system RAM that
    **pinned host memory commits against** — is a hard `cudaErrorMemoryAllocation`
    crash. CUDA can't see the shared budget; only DXGI can (the ledger in
    `bounce_pool.py` is a `RAM*0.25` proxy). Pinned weights spend the shared
    budget, which is *also* the dedicated cliff's overflow valve — so over-pinning
    converts a would-be slowdown into a crash. Sizing pinned memory must respect
    both cliffs independently.
    - **Hard-cap the allocator so the dedicated cliff becomes a loud OOM.**
      `torch.cuda.set_per_process_memory_fraction(1 - hard_gib/total)` makes the
      caching allocator raise a real OOM at ~(total − 1 GiB) instead of letting
      WDDM silently page past the ceiling (we observed `torch_allocated=12.23 GiB`
      on an 11.99 GiB card, then a crash in an unrelated bystander op). Applied at
      `attach_smart_training` / `inference_resident` via
      `MemoryManager._apply_wddm_hard_allocator_cap` (Windows-only, idempotent,
      keyed on the existing `wddm_hard_gib`). Debugging value: a capped allocator
      OOMs at the *true culprit's* allocation line — it found a stray
      `model.to(cuda)` hauling the whole quantized model onto the card in one run,
      where the uncapped version had crashed somewhere downstream.
    - **Pinned host memory grows/shrinks slowly, and torch never gives it back.**
      Page-locking is per-page kernel work (~0.6–2 GB/s on consumer Windows), so
      large pin/unpin is seconds, not free. Worse, anything pinned through torch's
      caching host allocator (`pin_memory=True`, every `non_blocking=True` D2H
      staging buffer) is retained page-locked for the process lifetime on free —
      it commits against the DXGI budget until `torch._C._host_emptyCache()`.
      `cudaHostRegister` (used for weight pins) is the one variant whose unpin
      actually returns budget; that is why the pin manager's eviction rung empties
      the host cache first, then unpins weights.
- **Fail-fast, deterministic, few silent fallbacks** in training/inference code —
  prefer explicit config flags and informative errors. Add a focused test under
  `tests/` for new memory-manager behaviour; the controllers have CPU/sim
  coverage precisely because GPU CI doesn't exist.
- **Don't hunt test-order leaks.** Some `tests/` files pass alone and fail only
  in the full suite, because process-global state (the pin ledger, the CUDA
  allocator, the `qfloat8`→torchao shim) outlives a test. Chasing these is slow
  and almost never finds a product bug. Confirm it's order-dependent — run the
  file alone (passes) and the suite with the suspect file ignored (passes),
  which is all you need to tell a real regression from a leak — then **note it
  on git-bug ticket `f2aceba` and move on.** No further bisecting, no
  `sys.modules` spelunking, no restructuring other tests. If a test *you're
  adding* triggers one, test the seam directly instead of driving the whole
  machine (e.g. assert a helper collects the right entries rather than building
  real pinned packs).
- **Search before adding** — the toolkit has a lot of helpers; reuse over
  reimplement. Offload changes must stay off-by-default / behaviour-preserving
  when their flags are off (see `docs/decisions/UPSTREAM_PR_PLAN.md`).
- **ASCII only in source files and PowerShell.** Windows PowerShell 5.1 mangles
  non-ASCII in commands/arguments, and mixed encodings corrupt files. In code,
  commit messages, and anything passed through a shell, write `->` not `→`,
  plain quotes not smart quotes. Don't introduce new non-ASCII when editing;
  Markdown docs edited via the file tools (not the shell) are the one exception.
- **If Edit fails repeatedly** (anchor/encoding trouble), fall back to
  `scripts/exact_edit.py` — exact-match replace/splice/insert that reads old/new
  text from helper files (bypassing shell quoting), fails fast on missing
  anchors, and preserves newline style. Run it via `venv/Scripts/python.exe`;
  keep the helper text files in the scratchpad/temp dir, not the repo.
</content>
