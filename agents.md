# AGENTS.md

A short, machine-friendly guide for coding agents and contributors working on this repository.

## Project overview

AI Toolkit is a training/finetuning suite for diffusion models. It provides a CLI runner (`run.py`) for launching jobs described by YAML/JSON configs in `config/` and a web UI (in `ui/`) for starting/stopping/monitoring jobs.

This AGENTS.md focuses on fast, safe commands agents can run and conventions that help automated tools reason about this project.

## Repository structure (for agents) 📁

Top-level layout and important locations an agent should know about:

- `config/` — Job config files (YAML/JSON). See `config/examples/` for copy-paste templates to run jobs.
- `jobs/` — Job types (e.g., `TrainJob.py`, `GenerateJob.py`) and job base classes.
- `jobs/process/` — Individual process classes executed by jobs (training steps, extraction, generation, etc.). Agents can patch or add processes here for automation tasks.
- `toolkit/` — Core helpers and utilities used by jobs (job loader, accelerators, printing helpers).
- `scripts/` — Utility scripts (conversion helpers, dataset repair, etc.) useful for pre/post processing.
- `ui/` — Web UI source (Node.js). Contains its own build/test commands; consider adding a nested `AGENTS.md` here for UI-specific instructions.
- `testing/` — Pytest tests (fast unit tests suitable for agents).
- `datasets/` & `output/` — Example datasets and artifacts; treat `output/` as generated artefacts.
- `run.py` — Primary local runner used by humans/agents to run jobs locally (do not run heavy GPU jobs without human confirmation).
- `run_modal.py` — Modal remote-run helper (example of a cloud agent entry point).
- `run-ui`, `run-ui.ps1`, `run-ui.bat` — Helpers for building/starting the UI across platforms.
- `docker/`, `docker-compose.yml` — Docker bits and helper scripts for containerized workflows.
- `requirements.txt` & `package.json` (in `ui/`) — Dependencies for Python and UI respectively.
- `AGENTS.md` — This file; agents should consult the nearest AGENTS.md (repo root or subproject) for rules and quick commands.
- `LEARNINGS.md` — Learnings file; agents should this file before every task to check previous attempted approaches. Whenever an approach fails or causes errors, it should be documented in LEARNINGS.md to help future agents.

> Note: prefer running tests and small checks listed below before attempting changes that touch training or GPU-heavy code. Keep long-running runs gated behind a human confirmation in automation flows.

---

## Setup commands (quick) ✅

Linux / macOS (recommended):

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python3 -m venv venv
source venv/bin/activate
# install torch appropriate for your GPU first (example for CUDA 12.6):
python -m pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python -m venv venv
.\venv\Scripts\Activate.ps1
# install torch matching your CUDA and Python
python -m pip install -r requirements.txt
```

UI (Node.js > 18):

```bash
# from repo root
# Windows: .\run-ui.ps1 or use `run-ui` helper
./run-ui build_and_start
# or: cd ui && npm run build_and_start
```

> Tip: Use `AI_TOOLKIT_AUTH` env var to protect the UI when exposing it publicly.

---

## What agents may run (safe, fast checks) ⚠️

These are the commands an automated agent can run by default without causing lengthy GPU training or network-heavy actions:

- Run unit tests (fast):
  - `python -m pytest testing -q`
  - Run a single test file: `python -m pytest testing/test_compute_per_example_loss.py -q`
- Run static checks (if added): e.g., `python -m ruff check .` or `black --check .` (project currently has no enforced linter in the repo root)
- Build the UI (fast): `./run-ui build` or `cd ui && npm run build`
- Run a lint/test subset for a change: `python -m pytest testing -q -k <test-name-substring>`

### Dataset evaluator (safe checks)
You can run lightweight dataset evaluations that sample or limit the number of items to avoid heavy CPU work. Prefer running on small datasets or with `--max-samples`/`--sample-fraction` set to limit cost.

- CLI (Python):
  - `python tools/eval_dataset.py --dataset-path <dataset_folder> --model <model_name_or_path> --batch-size 1 --out-dir <dataset_folder> --job-name <job_id> --step 0 --sample-fraction 0.1 --max-samples 100`
  - The CLI writes a JSON report named `{job_name}_{step_zfilled}.json` into the dataset folder when `--out-dir` is provided.
- API (UI):
  - POST `/api/eval_dataset` with JSON `{ "dataset_path": "<relative_dataset_name>", "model": "<name>", "batch_size": 1, "sample_fraction": 1.0, "max_samples": null }` to enqueue an EvalJob
  - GET `/api/eval_dataset` — list recent eval jobs
  - GET `/api/eval_dataset/{id}/status` — status and `info` field for failures
  - GET `/api/eval_dataset/{id}/result` — returns the JSON report contents (if available)
- Worker (Node):
  - The worker action `cron/actions/processEvalQueue.ts` will pick queued EvalJob rows and spawn `python tools/eval_dataset.py` for them. In dev you can run the worker once via:
    - `npx ts-node -P ui/tsconfig.worker.json ui/scripts/run_process_eval.ts`
  - The worker captures `stdout`/`stderr` from the Python process and writes a concise message into `EvalJob.info` (useful for diagnostics) and updates `status` to `running`, `finished`, or `error`.

Caveats & safety:
- Do NOT enqueue large evaluations unboundedly. Use `max_samples` or `sample_fraction` to cap work or ask for human approval.
- If the worker cannot write into the dataset folder (permissions), the eval will fail and `EvalJob.info` will contain the error; agents should surface that to a human reviewer rather than retry blindly.

Important: Do NOT start any GPU training runs (e.g., `python run.py config/...` which may allocate GPUs and run for hours) unless explicitly requested by a human reviewer.

---

## How to run training (human-only / explicit)

Training jobs are heavy and should only be started by humans or scheduled agent workflows that explicitly target GPU instances:

```bash
# Example: run a training job locally (requires a GPU and a prepared config)
python run.py config/my_training_config.yml

# Remote (Modal):
python run_modal.py config/my_training_config.yml

# Modal & other cloud runners store outputs in mounted volumes (see run_modal.py for example)
```

If an agent must schedule training remotely, ensure it uses a proper GPU cluster or cloud agent and confirms the job duration and resources with a human.

---

## Development tips (fast iteration) 💡

- When working on the UI, prefer the `run-ui` helper from the repo root (Windows PowerShell: `.
un-ui.ps1`).
- Use targeted tests when changing code to reduce CI runtime: `python -m pytest testing/test_foo.py -q`.
- Run the minimal reproducible test locally before opening a PR.
- **Prefer built-in toolkit helpers over reimplementing functionality.** Check `toolkit/` for dataset helpers (resize/crop), model utilities, and processing utilities before adding new code; this reduces duplication and avoids subtle incompatibilities.

---

## Testing instructions ✅

- Run all tests: `python -m pytest testing`
- Run single test file: `python -m pytest testing/test_caption_evaluator.py -q`
- Use `-k` to filter tests by substring or `-m` to filter by pytest marks.
- If a new feature changes behavior, add tests under `testing/`.

Agents should prioritize running and fixing the small, fast tests before attempting larger changes.

---

## Code style & formatting

- Follow existing style in the repository. For Python, prefer Black/ruff style if added (there is no enforced format in the repo root as of this writing). For JS/TS in `ui/`, follow the project's package.json scripts (Prettier/ESLint are commonly used in the UI).

If you add a formatter or linter, document the exact commands here.

---

## CI, PR and commit guidance ✍️

- Before opening a PR, run tests and fix any failing tests or type errors.
- PR checklist (recommended):
  - Title: `[<area>] Short description` (e.g., `[ui] Fix job list overflow`)
  - Run `python -m pytest testing` and fix failures
  - Run `./run-ui build` if UI changes are included
  - Add or update tests for changed logic

Agents can propose changes and run the checks automatically, but human approval should merge large changes or changes that add long-running tasks.

---

## Security / secrets

- Do not commit secrets (API keys, HF tokens, or dataset credentials).
- Use environment variables for credentials (e.g., `HF_TOKEN`, `WANDB_API_KEY`, `AI_TOOLKIT_AUTH`).
- Agents that need to use tokens must request them from secure secret stores or a human reviewer.

---

## Subproject AGENTS.md

If a subfolder (like `ui/`) has its own tooling and checks, add a `AGENTS.md` inside that folder with focused commands (e.g., `pnpm install`, `pnpm test`, `npm run build`). Agents automatically use the nearest `AGENTS.md` when editing files in a subdirectory.

### Navigation & test-running rules (new)
- Always prefer running commands from the project context directory for the package you're targeting. For UI work run commands from the `ui/` folder (e.g., `cd ui && npm run dev`), *not* from nested folders like `ui/ui`.
- If you are automating via scripts, prefer `npm --prefix ui <cmd>` or `npm exec --prefix ui <cmd>` to avoid being in a wrong cwd.
- When adding Playwright tests, create `ui/playwright.config.ts` and place tests under `ui/tests/e2e`.
- Use `npx playwright install` from the `ui` folder after adding `@playwright/test` to `ui/package.json`.

Small checklist for running Playwright tests locally:
1. cd ui
2. npm install
3. npm i -D @playwright/test
4. npx playwright install
5. Start the UI dev server in another terminal (e.g., `npm run dev`) before running tests
6. npx playwright test --config=playwright.config.ts tests/e2e/<spec>.ts

Tip: If `npx playwright` resolves the wrong package path (e.g., nested `ui/ui`), inspect `Get-Location` or `pwd` before running and prefer `npm --prefix ui exec playwright test ...` when scripting.

Start server in background to avoid blocking test runs:
- Preferred for E2E: run production server (`build` + `start`):
  - `cd ui && npm run build && npm run start`  # avoids dev overlays and dev-only artifacts
- Windows (PowerShell/dev):
  - Start in background: `Start-Process -NoNewWindow -FilePath npm -ArgumentList '--prefix','ui','run','dev' -PassThru`
  - Or use a job: `Start-Job -ScriptBlock { npm --prefix ui run dev } -Name PlaywrightUI`
- macOS / Linux (POSIX):
  - `nohup npm --prefix ui run dev >/dev/null 2>&1 &`

> NOTE (agents): **Always start the UI server in a separate shell and do not run other commands in that same terminal.** Keeping the server in its own terminal avoids accidentally killing it, prevents interference with logs, and ensures Playwright tests run against a stable process.

Quick verification (before running tests):
- Check server reachable: `curl -sfS http://localhost:3000 || (echo "server not reachable"; exit 1)`
- Or use the helper: `./scripts/check_playwright_setup.sh` (POSIX) or `.\scripts\check_playwright_setup.ps1` (PowerShell)

Dealing with Next.js dev overlay interference:
- The Next dev server shows a dev overlay (`<script data-nextjs-dev-overlay>` / `<nextjs-portal>`) that can intercept pointer events and block Playwright clicks. Options:
  - Run the **production** server (`npm run build && npm run start`) which does not include the dev overlay.
  - As a test-time workaround, disable/remove the overlay before interactions using a small Playwright helper that sets `pointer-events: none` or removes the overlay DOM node.

> Tip: Prefer the production server for reliable E2E runs; use the overlay-removal helper only for local quick checks.

Quick sanity helpers (added):
- `scripts/check_playwright_setup.ps1` — PowerShell script you can run from any folder (Windows) to verify `@playwright/test` and install browsers in the `ui/` package.
- `scripts/check_playwright_setup.sh` — POSIX shell equivalent for Linux/macOS.

Usage examples:
- PowerShell: `.\scripts\check_playwright_setup.ps1` (from repo root)
- Bash: `./scripts/check_playwright_setup.sh` (from repo root)

These helpers avoid accidental `ui/ui` path issues by invoking `npm --prefix <ui>` explicitly.

> NOTE (agents): Make sure the UI dev server is running (e.g., `npm --prefix ui run dev`) and *started in background or another terminal* before running Playwright tests. If the tests fail to reach `http://localhost:3000`, verify the server is up and listening and that you used a background start to avoid killing it when running subsequent commands.

> NOTE (agents): Make sure the UI dev server is running (e.g., `npm --prefix ui run dev`) before running Playwright tests. If the tests fail to reach `http://localhost:3000`, verify the server is up and listening.

---

## FAQ

Q: Will agents run training automatically from this file?

A: No. An agent should not kick off heavy GPU jobs without an explicit instruction. This file is intended to list the checks an agent can safely run by default (tests, builds, linters).

Q: How to add more quick checks?

A: Add them to the `What agents may run` section along with cost and runtime expectations.

---

If you'd like, I can:
- Add a short `AGENTS.md` inside `ui/` with `pnpm`/`npm` commands; or
- Add a `smoke` test target to the test suite for a quick agent-run check; or
- Provide a sample minimal `AGENTS.md` for a CI job that runs fast tests only.

---

## Eval & Caption Debugging (agents) 🔎

Guidance for working with the dataset evaluation and caption-debugging flow:

- Quick, safe checks:
  - Run a short eval with sampling capped: `python tools/eval_dataset.py --dataset-path <dataset> --model <model> --samples-per-image 2 --fixed-noise-std 0.6 --sample-fraction 0.1 --max-samples 50 --out-dir <dataset>`
  - Enqueue via the UI API: POST `/api/eval_dataset` with `debug_captions: true`, `samples_per_image`, and `fixed_noise_std` fields in the request body.

- Debug logs & locations:
  - Caption debug lines are printed with the tag `[EVAL-CAPTION-DEBUG]` and persisted to the dataset folder as `*.eval_caption_debug_<jobid>.log` (NDJSON-like lines recommended when available).
  - The evaluation job summary prints `Total examples: X, mean loss: Y` and `Flagged captions: N` to help triage quickly.

- Useful scripts and helpers:
  - `ui/scripts/run_debug_eval.js` — enqueue a debug eval job programmatically.
  - `tools/run_caption_debug_sim.py` — local compute_fn simulator intended for quick checks (requires `diffusers` in the environment).
  - `ui/scripts/check_eval_jobs.js`, `ui/scripts/print_eval_job.js` — DB inspection helpers.

- Defaults & behavior to know:
  - `samples_per_image` default: **8** (configurable via UI/API/CLI).
  - `fixed_noise_std` default: **0.6** (configurable via UI/API/CLI).
  - Flagging: captions that are short, repeated, suspicious tokens, empty, or extremely long are collected via `toolkit/util/loss_utils.py::flag_bad_captions` and returned in the job `flagged` field (contains reasons & scores).

- Quick triage checklist:
  1. Confirm job params: `debug_captions`, `samples_per_image`, `fixed_noise_std` are present in the job row.
  2. Check worker stdout or `*.eval_caption_debug_*.log` for `[EVAL-CAPTION-DEBUG]` lines.
  3. Inspect `res['flagged']` via `ui/scripts/print_eval_job.js` or by reading the JSON report file.
  4. If pipeline load errors occur, inspect `EvalJob.info` and the worker logs for model / HF auth / missing package messages.

- Known gaps and follow-ups for agents:
  - Paired-eval (A/B with identical noise/timesteps) is not implemented yet — avoid assuming paired noise in comparisons.
  - Local simulations (`run_caption_debug_sim.py`) require `diffusers` and related packages; add to env before running.

---

If you'd like, I can also append a short 'Eval & Caption Debugging' checklist to `LEARNINGS.md` summarizing recent discoveries and suggested follow-ups.
