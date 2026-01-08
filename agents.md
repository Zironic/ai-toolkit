# AGENTS.md

You are a coding agent working on a fork of the Ostris AI-Toolkit. The upstream repository exists at https://github.com/ostris/ai-toolkit

## Project overview

AI Toolkit is a training/finetuning suite for diffusion models. It provides a CLI runner (`run.py`) for launching jobs described by YAML/JSON configs in `config/` and a web UI (in `ui/`) for starting/stopping/monitoring jobs.

This document focuses on fast, safe commands you can run and conventions that help automated tools reason about this project.

## Repository structure (for you) 📁

Know the top-level layout and important locations:

- `config/` — Job config files (YAML/JSON). See `config/examples/` for copy-paste templates to run jobs.
- `jobs/` — Job types (e.g., `TrainJob.py`, `GenerateJob.py`) and job base classes.
- `jobs/process/` — Individual process classes executed by jobs (training steps, extraction, generation, etc.). You can patch or add processes here for automation tasks.
- `toolkit/` — Core helpers and utilities used by jobs (job loader, accelerators, printing helpers).
- `scripts/` — Utility scripts (conversion helpers, dataset repair, etc.) useful for pre/post processing.
- `ui/` — Web UI source (Node.js). Contains its own build/test commands; add a nested `AGENTS.md` here for UI-specific instructions if needed.
- `testing/` — Pytest tests (fast unit tests suitable for you).
- `datasets/` & `output/` — Example datasets and artifacts; treat `output/` as generated artefacts.
- `run.py` — Primary local runner used by you to run jobs locally. Do not run heavy GPU jobs without human confirmation.
- `run_modal.py` — Modal remote-run helper (example of a cloud agent entry point).
- `run-ui`, `run-ui.ps1`, `run-ui.bat` — Helpers for building/starting the UI across platforms.
- `docker/`, `docker-compose.yml` — Docker bits and helper scripts for containerized workflows.
- `requirements.txt` & `package.json` (in `ui/`) — Dependencies for Python and UI respectively.
- `AGENTS.md` — This file; consult the nearest AGENTS.md (repo root or subproject) for rules and quick commands.
- `LEARNINGS.md` — Learnings file; consult this file before every task to check previous attempts. Document any failures or errors in `LEARNINGS.md` to help future contributors.

> Note: prefer running tests and small checks listed below before attempting changes that touch training or GPU-heavy code. Keep long-running runs gated behind a human confirmation in automation flows.

### Toolkit quick reference (for you) 🔎
A concise index of common helpers and where to find them. Check here first before reimplementing features.

- `toolkit/dataloader_mixins.py` — dataset loading, resizing, cropping, random/POI cropping, buckets logic, and image transforms.
- `toolkit/config_modules.py` — `DatasetConfig`, `ModelConfig`, and related defaults for datasets and models.
- `toolkit/stable_diffusion_model.py` — `StableDiffusion` wrapper: `encode_images`, `decode_latents`, `encode_prompt`, `predict_noise` and other model-serving utilities.
- `toolkit/model_utils.py` — safe model loading utilities (inference-mode loader, apply LoRA helpers).
- `toolkit/util/loss_utils.py` — evaluation helpers: `run_dataset_evaluation`, per-example aggregation, and caption-flagging heuristics.
- `toolkit/prompt_utils.py` — prompt embedding helpers, prompt concatenation, and prompt-related utilities.
- `toolkit/train_tools.py` — encoding helpers, tokenization utilities, and training-time helpers useful for inference compatibility.
- `toolkit/accelerator.py` — device/dtype helpers and accelerator selection utilities.
- `toolkit/paths.py` — canonical repo paths (e.g., `MODELS_PATH`, `DIFFUSERS_CONFIGS_ROOT`).
- `toolkit/metadata.py` — helpers for safetensors and metadata extraction.


### Pre-change checklist (use before adding new helpers) ✅
- **Read** `AGENTS.md` and `LEARNINGS.md` to confirm there isn't an existing helper or prior attempt for this task.
- **Search the toolkit** for existing functions/classes: `rg "<keyword>|def <name>|class <Name>" toolkit/ -n` and inspect likely files (`dataloader_mixins.py`, `model_utils.py`, `util/loss_utils.py`).
- **Run targeted tests** relevant to the area: `python -m pytest testing -q -k <test-name-substring>` and add tests for any new behavior in `testing/`.
- **If adding CLI/UI behavior**, update API routes and worker persistence (e.g., `ui/src/app/api/eval_dataset/route.ts`, `ui/cron/actions/processEvalQueue.ts`) and add smoke checks.
- **Document** the change in `AGENTS.md` (brief note) and `LEARNINGS.md` (why, tests, caveats).
- **Training lifecycle docs:** Read the relevant chapter(s) in `docs/training_lifecycle/` before modifying training or pipeline code and add a one-line summary of the chapter(s) to the PR description.
  - `docs/training_lifecycle/Training_Job_Lifecycle.md` — Master index and verification checklist (start here)
  - `docs/training_lifecycle/01-ui-job-creation.md` — UI job creation: form → API → DB
  - `docs/training_lifecycle/02-prisma-and-db.md` — Prisma schema, DB usage, and transactions
  - `docs/training_lifecycle/03-worker-scheduling.md` — Worker cron, queue handling, spawn helper
  - `docs/training_lifecycle/04-run-py-and-job-loader.md` — `run.py` entrypoint and job loader
  - `docs/training_lifecycle/05-pipeline-and-jobs.md` — Pipeline loaders, model/LoRA application, job types
  - `docs/training_lifecycle/06-pretraining-setup.md` — Data, transforms, bucket logic, schedulers
  - `docs/training_lifecycle/07-training-loop.md` — Training loop, optimizer, mixed precision, EMA
  - `docs/training_lifecycle/08-checkpoints.md` — Checkpoint saving, metadata, push-to-hub
  - `docs/training_lifecycle/09-repro-troubleshoot.md` — Repro, troubleshooting, and best practices
  
  (Agents: read the master first, then the chapter(s) relevant to your task.)
- **Keep changes small & testable**; add a focused unit test and a short eval smoke run before larger refactors.

**Using subagents (Raptor Mini) —**
- **Prefer subagents for discovery and research.** Use Raptor Mini subagents to find existing functions, summarize long files, draft test cases, and collect references so you preserve and reuse context.
- **When to use:** code discovery, API/usage lookups, test-case generation, dependency mapping, and summarizing long logs or diffs.
- **When not to use:** running pipelines or GPU jobs, executing deployments, or handling secrets/credentials. Subagents cannot replace manual testing on GPU hardware.
- **Document results:** include a short summary of the subagent query and findings in the PR description and add a note in `LEARNINGS.md` (what was searched, keywords used, and the reason for adding a new helper).
- **Fallback:** if the subagent is unavailable, run the local ripgrep search and open an issue or flag the change for human review.
- **Safety:** never send secrets, credentials, or private keys to a subagent.

**Key principles (must follow) —**
1. **Pipelines & GPU code cannot be smoke-tested in CI.** Heavy GPU runs are manual-only; never rely on CI to validate pipeline behavior.
2. **Training & inference must be deterministic and fail-fast.** Avoid implicit fallbacks; prefer explicit configuration, immediate error propagation, and informative error messages.
3. **Avoid redundant code.** Search the toolkit first and reuse helpers; document why a new helper is needed in `LEARNINGS.md` and add a focused unit test.

**PR checklist (recommended):**
- Search for existing helpers in `toolkit/` (e.g., `rg "<keyword>|def <name>|class <Name>" toolkit/ -n`) and reuse them if possible.
- If the change requires GPU pipeline verification: include manual test steps, hardware requirements, and assign a reviewer with GPU access; mark the PR `manual-testing-required`.
- Confirm code fails fast and is deterministic where applicable, and add tests or documentation that demonstrate this.
- If you used a subagent (Raptor Mini), include the subagent query and a short summary of the findings in the PR description and `LEARNINGS.md`.

---


### UI start (safe checks you can run)
- Start: `.\run-ui-start.ps1` (Windows) or `./run-ui-start.sh` (POSIX)
- Stop: `.\run-ui-stop.ps1` (Windows) or `./run-ui-stop.sh` (POSIX)
- PID file: `ui/ui_start.pid` (contains the server process id)
- Logs: `ui/ui_start.log` (stdout) and `ui/ui_start.err` (stderr)
- Health check (POSIX / default): `curl -sfS http://localhost:8675 || echo "server not reachable"`
- Health check (PowerShell): `try { (Invoke-WebRequest -Uri http://localhost:8675 -UseBasicParsing -TimeoutSec 5).StatusCode } catch { Write-Output 'not-reachable' }`



---

## What you may run (safe, fast checks) ⚠️

These are the commands you can run by default without causing lengthy GPU training or network-heavy actions:

- Run unit tests (fast):
  - `python -m pytest testing -q`
  - Run a single test file: `python -m pytest testing/test_compute_per_example_loss.py -q`
- Run static checks (if added): e.g., `python -m ruff check .` or `black --check .` (project currently has no enforced linter in the repo root)
- Build the UI (fast): `./run-ui build` or `cd ui && npm run build`
- Run a lint/test subset for a change: `python -m pytest testing -q -k <test-name-substring>`


> **Warning: pipelines & GPU tests are manual-only.** Pipeline and other GPU-dependent code cannot be validated in CI. Run these tests manually on GPU machines; they are **not** part of PR CI. This repository does not maintain a GPU CI runner and will not add one. Do not start GPU jobs; any GPU run must be approved and documented in the PR with test steps and hardware requirements.

Important: Do NOT start any GPU training runs (e.g., `python run.py config/...` which may allocate GPUs and run for hours) unless explicitly requested by a human reviewer.


## Development tips (fast iteration) 💡

- Run targeted tests when changing code to reduce CI runtime: `python -m pytest testing/test_foo.py -q`.


Search quickly with ripgrep (e.g., `rg "def my_symbol" toolkit/`) or use the file names above to find implementation details. Update this list when you add or find commonly reused helpers.

**Use built-in toolkit helpers** over reimplementing functionality — check `toolkit/` for dataset helpers (resize/crop), model utilities, and processing utilities before adding new code; this reduces duplication and avoids subtle incompatibilities.


## Testing instructions ✅

- Run all tests: `python -m pytest testing`
- Run single test file: `python -m pytest testing/test_caption_evaluator.py -q`
- Use `-k` to filter tests by substring or `-m` to filter by pytest marks.
- If a new feature changes behavior, add tests under `testing/`.

Prioritize running and fixing the small, fast tests before attempting larger changes.

---

## Coding best practices
- **Prefer deterministic, testable helpers.** Avoid adding silent fallbacks in training or inference that could change behavior; prefer explicit feature flags, configuration options, or documented error paths.
- **Prefer module-level imports.** Use local (inline) imports only when necessary (for example, to avoid circular imports or optional dependency loading); add a short comment explaining why when you do.
- **Catch exceptions only to handle or add context.** Avoid try/except blocks that silently swallow errors. Good pattern:

```python
try:
    result = expensive_operation()
except IOError as exc:
    raise RuntimeError("expensive_operation failed for input X") from exc
```
- **Plan before you code.** Create a short plan that includes design, tests, and a small smoke check to keep training and inference behavior deterministic.
- **Search before you add.** Before implementing a new function, search the toolkit (e.g., `rg "<keyword>|def <name>|class <Name>" toolkit/ -n`) or use a Raptor Mini subagent when available to avoid redundant code. If the subagent or tooling is unavailable, open an issue or flag the change for human review.
- **Avoid loading entire pipelines directly.** Do not call `load_pipeline` to load a complete model (it is heavy and may cause non-determinism or resource exhaustion); use the safer loader helpers in `toolkit/model_utils.py` instead.
- **Deterministic & fail-fast.** Training and inference code must be deterministic and fail-fast: set random seeds (random, numpy, torch) where applicable, prefer deterministic ops, validate inputs early and raise explicit errors, and document any unavoidable non-determinism. Example:

```python
if not data_is_valid(x):
    raise ValueError("invalid training input — failing fast")
```
- **Pipelines cannot be validated in CI.** Pipeline and other GPU-dependent code cannot be reliably executed or smoke-tested in CI or on headless agents without GPU hardware. Unit tests may mock pipeline outputs or verify CPU-only helper logic, but this does not substitute manual verification of the real pipeline. Perform all real pipeline validation manually on GPU-equipped hardware and document the steps and required hardware in the PR. Do not start GPU jobs without explicit human approval.