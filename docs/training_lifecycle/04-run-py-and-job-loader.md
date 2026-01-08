# 04 — `run.py` & Job Loader ✅

TL;DR
- `run.py` is the canonical Python entrypoint for running one or more jobs from config files. It parses CLI args (config file(s), optional `--name`, `--log`, `--recover`) and for each config calls `toolkit.job.get_job` to instantiate the specific job class (e.g., `TrainJob`, `ExtractJob`, `GenerateJob`). Each job's `run()` method executes process-level steps and the top-level `main()` in `run.py` implements error handling (catching exceptions and `KeyboardInterrupt`, calling `on_error` hooks and optionally stopping based on `--recover`).

Files & symbols (key lines)
- `run.py` (main entry, arg parsing, job loop) — lines 1–120
- `toolkit.job.get_job(config_path, name)` — job loader that selects job class (see `toolkit/job.py`, lines 1–60)
- `toolkit.config.get_config` — loads JSON/YAML with `${ENV}` substitution and `[name]` replacement (see `toolkit/config.py`, lines 1–220)
- Job classes: `jobs/BaseJob` (`__init__`, `run`, `load_processes`) and `jobs/TrainJob` (initializes processes and calls `process.run()`)
- Error handling: `run.py` top-level try/except calls `job.process[0].on_error(e)` on exceptions (see `run.py` lines ~40–100)

Step-by-step runtime flow (process start → job class instantiation)
1. CLI invocation: `python run.py config/foo.yaml --log runs/foo.log --name myname` (see `run.py` arg parsing lines 14–56).
2. Logging setup: if `--log` provided, `setup_log_to_file` redirects `print_acc` output to the given file (lines 56–66).
3. Config loading & preprocessing:
   - `get_config(config_path, name)` resolves config paths (searches `config/`), supports `.json/.yaml` and applies `replace_env_vars_in_string()` to substitute `${VAR}` placeholders (raises an error if an env var is missing), and replaces `[name]` tokens (see `toolkit/config.py` lines 1–160).
4. Job loader:
   - `get_job` inspects `config['job']` and returns an instance of the matching class: `ExtractJob`, `TrainJob`, `GenerateJob`, `ExtensionJob`, etc. (see `toolkit/job.py` lines 1–40).
5. Job lifecycle in `run.py`:
   - For each job: `job.run()` is called (see `BaseJob.run` for standard header printouts). After run completes, `job.cleanup()` is called to release resources.
   - Exceptions: `run.py` wraps `job.run()` in try/except: it logs the error via `print_acc`, increments `jobs_failed`, calls `job.process[0].on_error(e)` if available, and will re-raise unless `--recover` is provided (lines ~72–108).

Config parsing details and important behaviors
- Environment variable substitution: `replace_env_vars_in_string` replaces `${VAR}` tokens and raises `ValueError` if the env var is unset—this makes missing-secret mistakes loud and immediate (see `toolkit/config.py` lines ~20–60).
- `[name]` replacement: `preprocess_config()` substitutes `[name]` occurrences with CLI `--name` or the config's `config.name` field.
- Config file search order: looks in `toolkit/config/` with standard extensions, then absolute/relative paths if not found.

How job classes are chosen & processes wired
- `get_job` selects based on `config['job']` (e.g., `'train'` -> `TrainJob(config)`). If unknown, it raises `ValueError`.
- `TrainJob` extends `BaseJob`, loads `process` entries listed under `config.process` and maps `type` strings to process classes using `process_dict` in `jobs/TrainJob.py`. `BaseJob.load_processes` dynamically imports `jobs.process` and instantiates process objects with their index, parent job, and config (see `jobs/BaseJob.py` and `jobs/TrainJob.py`).

Third-party libs & initialization
- Accelerator: `toolkit/accelerator.get_accelerator()` constructs an `Accelerator()` (Hugging Face Accelerate) at import time and is used by trainers to determine `is_main_process` and to wrap models (see `toolkit/accelerator.py`).
- Model frameworks: job/process/trainer code uses `torch`, `diffusers`, and `safetensors` for model loading and checkpointing (see `extensions_built_in/sd_trainer/SDTrainer.py` imports).
- Environment flags: `run.py` sets `DISABLE_TELEMETRY='YES'` and honors `DEBUG_TOOLKIT` for torch anomaly detection.

How the runtime handles errors, termination & exit codes
- `run.py` catches exceptions per-job: it calls `job.process[0].on_error(e)` as a best-effort to let processes flush state, then either continues (if `--recover`) or re-raises and exits with stack trace (non-zero exit). KeyboardInterrupt is similarly handled.
- Job-level `on_error` and `cleanup` hooks are expected to make a best-effort to save state and free resources.

Observability: logs & DB updates
- `--log` argument: `run.py` can log stdout/stderr to a file (via `setup_log_to_file`) for post-hoc inspection.
- `print_acc(...)` messages appear in logs and indicate job start/stop and results.
- Train/Eval processes (e.g., `UITrainer`, `DiffusionTrainer`) update the UI SQLite DB (via `UPDATE Job SET ...`) for `status`, `step`, `info`, etc., and are intended to be observed via the UI or direct sqlite queries (see `UITrainer._update_status`, `DiffusionTrainer._update_key`).

Open questions / TODOs
- Consider improving the error messages produced by missing `${ENV}` substitution so they point to exact config lines (see `toolkit/config.py: replace_env_vars_in_string`).
- Add tests for unknown `config['job']` values returning a clear error instead of crashing unexpectedly (see `toolkit/job.get_job`).
- Add a small test harness to call `run.py` with a dummy job that asserts `job.process[0].on_error` is invoked during an induced exception.

Files read (manifest)
- `run.py` (lines 1–140)
- `toolkit/job.py` (lines 1–60)
- `toolkit/config.py` (lines 1–220)
- `jobs/BaseJob.py` (lines 1–140)
- `jobs/TrainJob.py` (lines 1–120)
- `toolkit/accelerator.py` (lines 1–80)
- `extensions_built_in/sd_trainer/*` (selected files for model/framework calls)

Notes & uncertainties
- Some job types (e.g., `extension`) may rely on project-specific extension loading behavior; if you are adding an extension job, verify the process mapping in `toolkit/job.get_job`.
- Config env substitution strictly errors if an env var is missing; in some deploy contexts you may prefer a warning default instead of hard fail.