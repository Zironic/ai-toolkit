# AGENTS.md

A short, machine-friendly guide for coding agents and contributors working on this repository.

## Project overview

AI Toolkit is a training/finetuning suite for diffusion models. It provides a CLI runner (`run.py`) for launching jobs described by YAML/JSON configs in `config/` and a web UI (in `ui/`) for starting/stopping/monitoring jobs.

This AGENTS.md focuses on fast, safe commands agents can run and conventions that help automated tools reason about this project.

---

## Project coordination

- **Current project plan:** `ControlTrain.md` — this is the active plan for ongoing work on ControlNet / OpenPose control integration. Agents MUST load and keep `ControlTrain.md` in context before starting any non-trivial change that affects controlnet, control generation, dataloader, or training flow.
- **Plan maintenance rules:** After completing any task that materially changes behavior, tests, or configuration described in `ControlTrain.md`, update `ControlTrain.md` to reflect the change (short blurb of what changed + date) and add a short memory via Serena (`write_memory`) with the summary so future runs see it. If the task introduces further follow-ups, add them to the plan and note them in the file.

> **Agent coding policy:** Prefer adding small, well-tested helper functions in `toolkit/` over adding more logic into `jobs/process/*` classes. Keep process code thin: orchestration only, not heavy parsing or transformation logic. Also, prefer *fail-fast* behavior on unexpected conditions (raise clear, testable errors) rather than silently swallowing exceptions unless there is a documented, tested fallback path.
- **When to consult the plan:** Always consult `ControlTrain.md` before implementing UI, backend, or training changes that touch precompute-first behavior, manifest schema, or offload strategies. If the active plan changes, update this `Current project plan` field accordingly.

---


## Repository structure (for agents) 📁

Top-level layout and important locations an agent should know about:

---

## Repository index — quick lookup 📌
A short, scannable map of important folders and files so agents can jump straight to what matters.

- `config/` — Job config YAMLs and `config/examples/` templates for running jobs and reproducing runs.
- `datasets/` — Example datasets and local dataset folders used by Eval/Train jobs.
- `jobs/` — Job classes (e.g. `TrainJob.py`, `GenerateJob.py`) and `jobs/process/` for per-step processes.
- `toolkit/` — Core helpers and utilities used across jobs. Notable files: `toolkit/dataloader_mixins.py`, `toolkit/config_modules.py`, `toolkit/stable_diffusion_model.py`, `toolkit/model_utils.py`, `toolkit/train_tools.py`, `toolkit/paths.py`.
- `tools/` — CLI utilities and small scripts (e.g. `tools/eval_dataset.py`, `tools/precompute_control.py`, `tools/apply_canny_manifest.py`). Useful for job automation and dataset ops.
- `scripts/` — Convenience scripts and environment helpers (e.g. `scripts/check_playwright_setup.ps1`, `scripts/check_playwright_setup.sh`).
- `ui/` — Next.js web UI and worker code. Key locations: `ui/src/app/api/` (server routes), `ui/cron/actions/` (workers), `ui/prisma/schema.prisma` (DB models like `EvalJob`).
- `testing/` — Pytest test suite (unit and integration tests such as `testing/test_precompute_control.py`).
- `output/` — Generated outputs and artifacts created by runs.
- `run.py` / `run_modal.py` — Primary local runner and remote runner helpers (human-only for GPU training).
- `docker/` & `docker-compose.yml` — Docker helpers and images for containerized workflows.
- `LEARNINGS.md` / `learnings.md` — Operational notes and lessons learned; check before changing major flows.
- `AGENTS.md` / `agents.md` — This file(s): always consult the nearest `AGENTS.md` for scope-specific agent rules.

---


- `config/` — Job config files (YAML/JSON). See `config/examples/` for copy-paste templates to run jobs.

## Repository index — file map (detailed) 🗂️
A short list of specific files and where to look when you need to change behavior or add features. Whenever an agent does not know where to find a behavior or feature, searches for it and finds it. It should update this file map with where it found it.

- `run.py` — Main CLI runner for local jobs and configs (human-first; do not start GPU training without explicit approval).
- `run_modal.py` — Helper to run jobs remotely (Modal / cloud-run examples).
- `config/examples/` — Example job configs and templates (e.g., `controlnet_openpose_train.yml`).
- `tools/gen_control.py` — Generate ControlNet inputs (OpenPose by default); writes `control_manifest.json` with per-file hash/signature when caching is used. Supports `--batch-size` for chunking and idempotent, atomic updates.
- `tools/apply_control_manifest.py` — CLI helper to write dataset-level `control_config.json` from a manifest (safe `--force` flag and no-op behavior).
- `tools/eval_dataset.py` — Dataset evaluation CLI; produces JSON report files and is invoked by the UI worker for `EvalJob`.
- `tools/quick_eval_sim.py` — Eval/caption debug helpers and simulation utilities.
- `toolkit/config_modules.py` — `DatasetConfig`/`ModelConfig` and validations (e.g., blocks geometric augmentations when precomputed controls exist).
- `toolkit/dataloader_mixins.py` — Data loading, transforms, bucketing, and ControlNet control-file lookup (prefers manifest when present).
- `toolkit/stable_diffusion_model.py` — Model wrapper with encode/decode and prediction helpers used across jobs.
- `toolkit/model_utils.py` — Safe model loading, LoRA application, and load helpers.
- `toolkit/train_tools.py` — Training helpers, encoding utilities, and CLI/test conveniences.
- `toolkit/paths.py` — Canonical repo paths (e.g., `MODELS_PATH`, `DIFFUSERS_CONFIGS_ROOT`).
- `jobs/` — Job classes (e.g., `TrainJob.py`, `GenerateJob.py`, `ExtractJob.py`) and `jobs/process/` for per-step process implementations.
- `ui/src/app/api/` — Server routes for the UI; notable: `api/eval_dataset/route.ts` (enqueues `EvalJob`) and control-generation endpoints (enqueue a control generation job, check job status, fetch results, and apply a control manifest to a dataset).
- `ui/cron/actions/processEvalQueue.ts` — Worker that pulls `EvalJob` rows and runs `python tools/eval_dataset.py`, capturing output and updating DB `status`/`info`.
- `ui/cron/actions/processControlGenQueue.ts` — Worker that pulls control-generation jobs and runs `python tools/gen_control.py` (or `tools/apply_control_manifest.py` when applying manifests), capturing progress and updating DB `status`/`info`. After a successful run the worker may auto-apply the generated manifest to the dataset when `params.auto_apply` is not explicitly false; apply results are recorded in job info (e.g., `applied:written`, `applied:noop`, or `apply exit ...`).
- `ui/prisma/schema.prisma` — DB models used by UI and workers: `Job`, `EvalJob`, and `PrecomputeJob` (key fields and indices).
- `ui/src/components/` — UI components (e.g., `EvalJobsList.tsx`, `PrecomputeJobsList.tsx` and modal `PrecomputeDatasetModal.tsx`) and dataset page components that interact with API routes.
- `ui/src/utils/precompute.ts` — Client helpers to create/list Precompute jobs from the UI (`createPrecomputeJob`, `listPrecomputeJobs`).
- `testing/` — Pytest tests and examples. Key tests for the new flow: `testing/test_precompute_control.py`, `testing/test_apply_canny_manifest.py`, `testing/test_precompute_end_to_end.py`.
- `scripts/` — Developer convenience scripts (e.g., `check_playwright_setup.ps1` / `.sh`) and other utilities.
- `docker/` & `docker-compose.yml` — Containerization and dev environment helpers.
- `LEARNINGS.md` / `learnings.md` — Operational notes; check before making systemic changes or re-trying flaky automation approaches.



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
# Recommended (non-interactive / agent-safe): start the production server (non-interactive)
# Use the helper scripts (they perform install/update/build and then start)
# Windows (agent-safe):  .\run-ui-start-background.ps1  # builds by default, then starts
# POSIX (agent-safe):    ./run-ui-start-background.sh    # builds by default, then starts
# To stop the background server:
# Windows: .\run-ui-stop-background.ps1
# POSIX:   ./run-ui-stop-background.sh

# Agent note: the start script builds by default so local changes go live. To skip the build
# (useful for fast agent restarts), pass the -NoBuild switch or set the env var NO_UI_BUILD=1:
#   .\run-ui-start-background.ps1 -NoBuild
#   powershell: $env:NO_UI_BUILD='1'; .\run-ui-start-background.ps1
# The default behavior (no flag) runs `npm run build` and exits if the build fails.

# Alternative (foreground/interactive) - NOT recommended for unattended agents
# ./run-ui build_and_start  # runs in foreground and may block the terminal
# Dev hot-reload (interactive, not agent-safe):
# Windows: .\run-ui-dev.bat (starts Next dev + worker, interactive)
# POSIX: nohup npm --prefix ui run dev >/dev/null 2>&1 &
```

> Tip: Use `AI_TOOLKIT_AUTH` env var to protect the UI when exposing it publicly. For automated agents prefer the non-interactive background start scripts (`run-ui-start.ps1` / `run-ui-start.sh`) rather than running interactive dev servers.

### Agent-safe UI start (quick verification)
- Start: `.\run-ui-start.ps1` (Windows) or `./run-ui-start.sh` (POSIX)
- Stop: `.\run-ui-stop.ps1` (Windows) or `./run-ui-stop.sh` (POSIX)
- PID file: `ui/ui_start.pid` (contains the server process id)
- Logs: `ui/ui_start.log` (stdout) and `ui/ui_start.err` (stderr)
- Health check (POSIX / default): `curl -sfS http://localhost:8675 || echo "server not reachable"`
- Health check (PowerShell): `try { (Invoke-WebRequest -Uri http://localhost:8675 -UseBasicParsing -TimeoutSec 5).StatusCode } catch { Write-Output 'not-reachable' }`

> Note: The background helpers use the repo's `build_and_start` behavior (so the server will listen on port **8675** by default). If you need to confirm the port or readiness, run the health check after starting the helper.


---

## What agents may run (safe, fast checks) ⚠️

These are the commands an automated agent can run by default without causing lengthy GPU training or network-heavy actions:

- Run unit tests (fast):
  - `python -m pytest testing -q`
  - Run a single test file: `python -m pytest testing/test_compute_per_example_loss.py -q`
- Run static checks (if added): e.g., `python -m ruff check .` or `black --check .` (project currently has no enforced linter in the repo root)
- Build the UI (fast): `./run-ui build` or `cd ui && npm run build`
- Run a lint/test subset for a change: `python -m pytest testing -q -k <test-name-substring>`

> **Shell note:** Windows PowerShell does not support POSIX here-doc syntax such as `python - <<PY ... PY`. Avoid sending shell here-doc style commands from PowerShell; instead use `python -c "..."`, or create a temporary script file (e.g., `temp.py`) and run `python temp.py` when you need to execute multi-line Python. Also ensure you activate the project's virtual environment before running Python commands (Windows PowerShell: `.\venv\Scripts\Activate.ps1`; POSIX: `source venv/bin/activate`).
> **Troubleshooting (Python modules):** If you get Python module errors (e.g., `ModuleNotFoundError` / `No module named ...`), **first check whether the virtual environment is active**. Verify the active Python with:
>
> ```bash
> python -c "import sys; print(sys.executable)"
> ```
>
> If that path does not point into `./venv/`, activate the venv (Windows PowerShell: `.\	env\Scripts\Activate.ps1`, POSIX: `source venv/bin/activate`), then re-run `python -m pip install -r requirements.txt` and retry the command. Running `python -m pip show <package>` helps confirm a package is installed in the active environment.

> **PowerShell examples (safe patterns):**
> - One-liner: `python -c "print('hello')"` ✅
> - Multi-line (here-string -> temp file):
>   ```powershell
>   $script = @'
>   print("line1")
>   print("line2")
>   '@
>   Set-Content -Path temp.py -Value $script -Encoding UTF8
>   python temp.py
>   Remove-Item temp.py -Force
>   ```
> - Short inline alternative: `Set-Content -Path temp.py -Value "print('hello')"; python temp.py; Remove-Item temp.py -Force` ✅
> - Helper script (convenient): `.\scripts\run_python.ps1 -Body "print('hello')"` or `.\scripts\run_python.ps1 -File scripts/some_script.py -Args 'arg1','arg2'` ✅
> **Agent tip:** If your agent is running on Windows, prefer PowerShell-safe patterns (`python -c`, temporary script, or `scripts\run_python.ps1`) and avoid POSIX here-docs to prevent parse errors.  
> If you add automation, detect the OS and choose the appropriate invocation pattern automatically.

### Dataset evaluator (safe checks)
You can run lightweight dataset evaluations that sample or limit the number of items to avoid heavy CPU work. Prefer running on small datasets or with `--max-samples`/`--sample-fraction` set to limit cost.

> **GPU tests note (manual only):** Some tests (e.g., Accelerate offload, swap correctness, memory smoke, DDP-safety) require GPU hardware and an Accelerate-configured environment. These tests are intended to be run manually by maintainers on GPU machines and are not included in standard automated test workflows. This repository does not maintain a dedicated GPU test runner; GPU tests should remain manual and on-demand. To run them locally:
> - Activate venv: `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (POSIX)
> - Ensure `accelerate` and CUDA drivers are available and configured.
> - Run the GPU tests: `python -m pytest testing/test_controlnet_offload_gpu.py -q` (skips automatically on CPU-only environments)
> - Run benchmark: `python tools/benchmark_offload.py --strategy accelerate --size-mb 200 --iters 3` (human-run profiling)
> - Record notable findings in `LEARNINGS.md` or review notes.
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
  - The cron worker now also runs a PID-only liveness checker that scans running `Job` and `EvalJob` rows for recorded PIDs and will **mark** jobs `stopped` if the PID is not present on the host; jobs without a recorded PID are left unchanged. This prevents stuck "running" entries when child processes die unexpectedly.

Caveats & safety:
- Do NOT enqueue large evaluations unboundedly. Use `max_samples` or `sample_fraction` to cap work or ask for human approval.
- If the worker cannot write into the dataset folder (permissions), the eval will fail and `EvalJob.info` will contain the error; agents should surface that to a human reviewer rather than retry blindly.

Important: Do NOT start any GPU training runs (e.g., `python run.py config/...` which may allocate GPUs and run for hours) unless explicitly requested by a human reviewer.


## Development tips (fast iteration) 💡

- Use targeted tests when changing code to reduce automated test runtime: `python -m pytest testing/test_foo.py -q`.
- Run the minimal reproducible test locally before submitting changes for review.

### Toolkit quick reference (agents) 🔎
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

Tip: search quickly with ripgrep (e.g., `rg "def my_symbol" toolkit/`) or use the file names above to find implementation details. Update this list when you add or find commonly reused helpers.

**Prefer built-in toolkit helpers** over reimplementing functionality — check `toolkit/` for dataset helpers (resize/crop), model utilities, and processing utilities before adding new code; this reduces duplication and avoids subtle incompatibilities.

### Pre-change checklist (use before adding new helpers) ✅
- **Read** `AGENTS.md`  to confirm there isn't an existing helper or prior attempt for this task.
- **Search the toolkit** for existing functions/classes: `rg "<keyword>|def <name>|class <Name>" toolkit/ -n` and inspect likely files (`dataloader_mixins.py`, `model_utils.py`, `util/loss_utils.py`).
- **Run targeted tests** relevant to the area: `python -m pytest testing -q -k <test-name-substring>` and add tests for any new behavior in `testing/`.
- **If adding CLI/UI behavior**, update API routes and worker persistence (e.g., `ui/src/app/api/eval_dataset/route.ts`, `ui/cron/actions/processEvalQueue.ts`) and add smoke checks.
- **Document** the change in `AGENTS.md` (brief note) and `LEARNINGS.md` (why, tests, caveats).
- **Keep changes small & testable**; add a focused unit test and a short eval smoke run before larger refactors.

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

---

## Security / secrets

- Do not commit secrets (API keys, HF tokens, or dataset credentials).
- Use environment variables for credentials (e.g., `HF_TOKEN`, `WANDB_API_KEY`, `AI_TOOLKIT_AUTH`).
- Agents that need to use tokens must request them from secure secret stores or a human reviewer.

---

