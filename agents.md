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



### Agent-safe UI start (quick verification)
- Start: `.\run-ui-start.ps1` (Windows) or `./run-ui-start.sh` (POSIX)
- Stop: `.\run-ui-stop.ps1` (Windows) or `./run-ui-stop.sh` (POSIX)
- PID file: `ui/ui_start.pid` (contains the server process id)
- Logs: `ui/ui_start.log` (stdout) and `ui/ui_start.err` (stderr)
- Health check (POSIX / default): `curl -sfS http://localhost:8675 || echo "server not reachable"`
- Health check (PowerShell): `try { (Invoke-WebRequest -Uri http://localhost:8675 -UseBasicParsing -TimeoutSec 5).StatusCode } catch { Write-Output 'not-reachable' }`



---

## What agents may run (safe, fast checks) ⚠️

These are the commands an automated agent can run by default without causing lengthy GPU training or network-heavy actions:

- Run unit tests (fast):
  - `python -m pytest testing -q`
  - Run a single test file: `python -m pytest testing/test_compute_per_example_loss.py -q`
- Run static checks (if added): e.g., `python -m ruff check .` or `black --check .` (project currently has no enforced linter in the repo root)
- Build the UI (fast): `./run-ui build` or `cd ui && npm run build`
- Run a lint/test subset for a change: `python -m pytest testing -q -k <test-name-substring>`

> **Shell note:** Windows PowerShell does not support POSIX here-doc syntax such as `python - <<PY ... PY`. Avoid sending shell here-doc style commands from PowerShell; instead use `python -c "..."`, or create a temporary script file (e.g., `temp.py`) and run `python temp.py` when you need to execute multi-line Python. Also ensure you activate the project's virtual environment before running Python commands (Windows PowerShell: `.#\venv\\Scripts\\Activate.ps1`; POSIX: `source venv/bin/activate`).

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
> - Helper script (convenient): `.\
>   scripts\run_python.ps1 -Body "print('hello')"` or `.\n>   scripts\run_python.ps1 -File scripts/some_script.py -Args 'arg1','arg2'` ✅
>
> **Agent tip:** If your agent is running on Windows, prefer PowerShell-safe patterns (`python -c`, temporary script, or `scripts\run_python.ps1`) and avoid POSIX here-docs to prevent parse errors.  
> If you add automation, detect the OS and choose the appropriate invocation pattern automatically.

### Dataset evaluator (safe checks)
You can run lightweight dataset evaluations that sample or limit the number of items to avoid heavy CPU work. Prefer running on small datasets or with `--max-samples`/`--sample-fraction` set to limit cost.

> **GPU tests note (manual only):** Some tests (e.g., Accelerate offload, swap correctness, memory smoke, DDP-safety) require GPU hardware and an Accelerate-configured environment. These tests are intended to be run manually by maintainers on GPU machines and **are not** included in the standard PR CI workflow. **This repository does not maintain a GPU CI runner and we will not add one; GPU tests should remain manual and on-demand.** To run them locally:
> - Activate venv: `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (POSIX)
> - Ensure `accelerate` and CUDA drivers are available and configured.
> - Run the GPU tests: `python -m pytest testing/test_controlnet_offload_gpu.py -q` (skips automatically on CPU-only environments)
> - Run benchmark: `python tools/benchmark_offload.py --strategy accelerate --size-mb 200 --iters 3` (human-run profiling)
> - Record notable findings in `LEARNINGS.md` or PR comments.
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

---

## Development tips (fast iteration) 💡

- Use targeted tests when changing code to reduce CI runtime: `python -m pytest testing/test_foo.py -q`.
- Run the minimal reproducible test locally before opening a PR.

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
- **Read** `AGENTS.md` and `LEARNINGS.md` to confirm there isn't an existing helper or prior attempt for this task.
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
