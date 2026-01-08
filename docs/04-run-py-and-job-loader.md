# 04 — run.py and job loader

## TL;DR ✅
- Entry point: `run.py` — it sets environment flags, initializes `Accelerator`, parses CLI options (config file list, `--recover`, `--name`, `--log`), then loops over provided config files and for each calls `toolkit.job.get_job(...)` to construct an appropriate Job subclass and executes it (`job.run()` and `job.cleanup()`). (See `run.py` — `main()` and error handling.)
- Config parsing: `toolkit.config.get_config(...)` accepts a dict or filename (searching `config/` for supported extensions `.json|.jsonc|.yaml|.yml`), performs `${ENV}` substitutions (required to exist), and replaces `[name]` tags via `preprocess_config`. Missing mandatory keys raise immediate errors. (See `toolkit/config.py`.)
- Job selection: `toolkit.job.get_job(...)` looks at `config['job']` and maps strings to classes (e.g., `'train' -> TrainJob`, `'extract' -> ExtractJob`, `'generate' -> GenerateJob`, `'extension' -> ExtensionJob'`). (See `toolkit/job.py`.)
- Runtime flow: `run.py` → `get_job` → `Job.__init__` (validates config & sets `self.config`, `self.name`) → `Job.load_processes(...)` (if applicable) → `Job.run()` → each `Process.run()` (e.g., `BaseTrainProcess`, `BaseSDTrainProcess`, or extension trainers) → training/inference logic, periodic saving, and status updates.
- Progress & status: core printing uses `toolkit.print.print_acc` (prints only from local main process); high-level model state uses `print_and_status_update(...)` defined on model objects which calls registered status hooks. The UI-facing `DiffusionTrainer` (an SD extension) implements DB-backed status updates using a per-job SQLite DB path and `AITK_JOB_ID` environment variable. It performs non-blocking updates via an executor and async tasks to write `status`, `info`, `step`, and `speed_string` into the `Job` SQLite row. (See `extensions_built_in/sd_trainer/DiffusionTrainer.py`.)
- Error & exit handling: `run.py` surrounds each job run in try/except. On Exception or KeyboardInterrupt it attempts to call `job.process[0].on_error(e)` (best-effort), prints an error, increments failure count, and either continues (if `--recover`) or re-raises (causing process to exit non-zero). Specific process classes (e.g., `DiffusionTrainer.on_error`) also update DB status to `error` and flush async writes. 

---

## Exact files & symbols referenced 🔧
- `run.py` — main entrypoint, CLI parsing, loop over config files, try/except handling around `get_job(...)` and `job.run()`.
- `toolkit.job.get_job(config_path, name)` — maps `config['job']` to job classes (`ExtractJob, TrainJob, ModJob, GenerateJob, ExtensionJob`).
- `toolkit.config.get_config(config_path_or_dict, name)` — identifies config file path, supports `.json/.jsonc/.yaml/.yml`, does `${ENV}` substitutions via `replace_env_vars_in_string`, and then calls `preprocess_config` (validates `job` and `config` keys and expands `[name]` tokens).
- `jobs.BaseJob` — base job class; `__init__`, `get_conf(...)`, `load_processes(process_dict)` and `cleanup()`.
- `jobs.process.BaseProcess` — base for processes; `get_conf`, `run`, `on_error`, `timer` hooks and `print()` helper.
- `jobs.process.BaseTrainProcess` — training-specific setup (tensorboard, save folder, seed initialization).
- `jobs.process.BaseSDTrainProcess` — SD-specific trainer setup: loads `TrainConfig`, `ModelConfig`, sets up `accelerate` `Accelerator`, configures third-party logging (`transformers`, `diffusers`) and other training modules; this is where many third-party libs are imported and device state chosen.
- `extensions_built_in/sd_trainer/SDTrainer.py` — concrete SD training logic using `diffusers`, `torch`.
- `extensions_built_in/sd_trainer/DiffusionTrainer.py` — UI-aware trainer that updates SQLite DB status and reacts to remote stop/queue signals (reads `AITK_JOB_ID`, `sqlite_db_path`).
- `toolkit.print.print_acc` & `setup_log_to_file` — print helpers that gate prints to local main process and optionally log to a file.
- `toolkit.accelerator.get_accelerator` — wraps Hugging Face `accelerate.Accelerator` instantiation.
- `toolkit.models.base_model.print_and_status_update` — emits `print_acc(status)` and calls status hooks (`_status_update`), used by models to emit status strings that `DiffusionTrainer` may hook into.

---

## Step-by-step runtime flow (process start → job class instantiation → run) 🧭
1. User runs: `python run.py config_name [more_configs] [--recover] [--name NAME] [--log FILE]`.
   - `run.py` sets environment variables early (e.g., `HF_HUB_ENABLE_HF_TRANSFER=1`, `NO_ALBUMENTATIONS_UPDATE=1`, `DISABLE_TELEMETRY=YES`) and conditionally enables torch anomaly detection when `DEBUG_TOOLKIT=1`.
   - `run.py` initializes a global `accelerator = get_accelerator()` (wrapped HF `Accelerator`).
2. CLI parsing: `argparse` collects `config_file_list` (required), `--recover`, `--name`, `--log`.
   - If `--log` provided: `toolkit.print.setup_log_to_file` is called (overrides `sys.stdout`/`sys.stderr` with a `Logger`) — only the local main process creates output directories.
3. For each item in `config_file_list`:
   a. `get_job(config_file, args.name)` is invoked. (See `toolkit.job.get_job`.)
   b. `toolkit.config.get_config` resolves the path: looks under `TOOLKIT_ROOT/config/` for extensions `.json/.jsonc/.yaml/.yml`, falls back to given path or absolute path. The raw text is processed by `replace_env_vars_in_string` which replaces `${ENV_VAR}` placeholders and throws if missing.
   c. `preprocess_config` validates presence of required keys (`job`, top-level `config`, and `config.name` unless `--name` provided). It replaces the `[name]` tag by `--name` or `config.config.name`.
   d. Back in `get_job`, `config['job']` determines the job type string. The function imports the matching job class (e.g., `from jobs import TrainJob`) and returns an instance (e.g., `TrainJob(config)`). If job type unknown, it raises `ValueError('Unknown job type ...')`.
4. Job construction (`BaseJob.__init__`): sets `self.config = config['config']`, `self.raw_config`, `self.job` and resolves `self.name` via `get_conf('name', required=True)`. Optionally loads `meta` and other job-level fields.
5. For job types that have `process` lists (e.g., `TrainJob`), `BaseJob.load_processes(process_dict)` is called. The method imports the process module (`jobs.process`) and for each `config.process[i]` expects a `type` key; it looks up `process_dict` mapping and instantiates the `ProcessClass(i, self, process_config)`.
6. `run.py` then calls `job.run()`.
   - `BaseJob.run()` prints the run banner (console output) and is overridden by job types (`TrainJob.run()` calls `super().run()` then iterates each `process.run()`).
   - At process level, `Process.run()` implementations execute training, sampling, or other logic. Many training flows use `BaseSDTrainProcess` / `SDTrainer` classes which:
     - Instantiate the `Accelerator` and set device/state presets
     - Create `TrainConfig`, `ModelConfig`, `SaveConfig`, `SampleConfig`, etc.
     - Load models (diffusers, transformers, safetensors), datasets, optimizers, EMA, schedulers
     - Enter training loops and call `save()` hooks on configured checkpoints
     - Use `print_and_status_update()` at logical points (loading, training, saving) to emit user-facing messages and status hook updates
7. UI integration / DB status updates (only in UI-aware trainers):
   - `DiffusionTrainer` recognizes `AITK_JOB_ID` env var and `sqlite_db_path` config — if both exist it writes status updates to the SQLite DB non-blocking via a thread pool and async tasks.
   - Status updates include `update_status(status, info)`, `update_step()`, `update_db_key('speed_string', ...)`, and `maybe_stop()` polling to check `stop` or `return_to_queue` values in the DB.
   - If a remote stop is requested, `DiffusionTrainer` flips internal `is_stopping`, updates status to `stopped`, attempts to save if configured (`save_before_stop`), and finally triggers `os.kill(os.getpid(), signal.SIGINT)` to emit a Keyboard Interrupt and stop.
8. Termination & exception paths:
   - If a process raises an exception, it can implement `on_error` to perform cleanup; `run.py` wraps the job run with a top-level try/except that logs errors and calls `job.process[0].on_error(e)` (best-effort). If `--recover` is not set, the exception is re-raised and the whole `run.py` process exits with a non-zero exit (the exception propagates out of `main()`). 
   - `KeyboardInterrupt` is caught similarly and `job.process[0].on_error()` is executed; if not `--recover`, the KeyboardInterrupt is re-raised.

---

## How config overrides & environment variables are applied ⚙️
- Config file selection: a name like `foo` maps to `config/foo.(json|jsonc|yaml|yml)` under `TOOLKIT_ROOT/config/` if present; absolute or relative paths also work.
- Environment substitution: `replace_env_vars_in_string()` uses regex `\$\{VAR_NAME\}` and replaces it with `os.environ[VAR_NAME]`. If missing, an error is raised (fail-fast). (See `toolkit/config.py`.)
- Name override: `--name` CLI arg replaces `[name]` tag in config by string substitution in `preprocess_config` (JSON dump/replace/load) — this allows sharing a single config template for multiple runs.
- Important env vars seen in code:
  - `HF_HUB_ENABLE_HF_TRANSFER=1` — set by `run.py`
  - `NO_ALBUMENTATIONS_UPDATE=1` — set by `run.py`
  - `DISABLE_TELEMETRY=YES` — set to silence diffusers telemetry
  - `DEBUG_TOOLKIT=1` — when set, `torch.autograd.set_detect_anomaly(True)` is enabled (debuggability)
  - `AITK_JOB_ID` — used by `DiffusionTrainer` to map trainer instance to UI `Job` row in SQLite and enable DB-backed status updates
  - `sqlite_db_path` — trainer config field that defaults to `./aitk_db.db` for the UI path

---

## Where third-party libs are loaded and configured 📦
- `accelerate` — `toolkit/accelerator.py` creates a singleton `Accelerator()` used across jobs/processes. `BaseSDTrainProcess` sets TF/diffusers logging levels based on `accelerator.is_local_main_process`.
- `torch` — imported in trainers and base processes (`BaseTrainProcess`, `BaseSDTrainProcess`, `SDTrainer`), used for device and seed control (e.g., `torch.manual_seed`, `torch.cuda.manual_seed`).
- `diffusers` & `transformers` — used extensively in `BaseSDTrainProcess`/`SDTrainer` for loading models, pipeline creation, and more (e.g., `ControlNetModel.from_pretrained()`).
- `safetensors` — used for loading/saving safetensors files (`load_file`, `save_file` in SD train pipeline).
- `huggingface_hub` — used for HF operations (repo handling, auth via `HfApi`, `Repository`).
- `tensorboard` — optional `SummaryWriter` wired up by `BaseTrainProcess.setup_tensorboard()` when `log_dir` configured.

---

## How the job registers progress and uses toolkit helpers 📈
- Console prints are done with `print_acc(...)` (only from local main process) to avoid noisy output when running in distributed mode.
- Model-level status updates use `print_and_status_update(status)` (defined on model classes) which emits a print and also calls registered status hooks.
- UI-integrated trainers (e.g., `DiffusionTrainer`) register a hook into the model's status-update path (`self.sd.add_status_update_hook(self.status_update_hook_func)`) so model messages propagate to DB (`update_status`).
- Trainers push periodic performance metrics by hooking `Timer` and printing composite summaries; `DiffusionTrainer.handle_timing_print_hook` converts timing into `speed_string` DB updates.
- Steps are posted via `update_step()` (non-blocking async DB write) and trainers call `maybe_stop()` to react to remote stop/queue signals.

---

## Where termination, exceptions and exit codes are handled 🛑
- `run.py` (main loop) wraps each job run in a `try/except Exception` and `except KeyboardInterrupt`. On exception it:
  - Calls `print_acc(f"Error running job: {e}")`
  - Attempts `job.process[0].on_error(e)` (best effort)
  - Increments `jobs_failed` and either continues (if `--recover`) or calls `print_end_message()` then `raise e` to terminate the process (non-zero exit code via uncaught exception)
- Process-level `on_error`: many process classes implement cleanup and DB updates for errors; example: `DiffusionTrainer.on_error` sets DB `status='error'`, updates `step` to last save, waits for async operations, and shuts down the executor.
- Remote stop handling: `DiffusionTrainer`'s stop watcher polls DB; if it detects a stop signal it performs final saving (optionally) and issues `os.kill(os.getpid(), signal.SIGINT)` to ensure process stops cleanly and raises a KeyboardInterrupt path handled by run.py.

---

## How to observe — commands & techniques 👀
- Run a job locally (no UI): `python run.py my_config_name` (or `python run.py ./config/my_config.yaml`). Use `--log` to write a log file: `python run.py my_config --log ./logs/run.log`.
- If using the UI: set `AITK_JOB_ID` to the `Job.id` value in your UI DB and ensure `sqlite_db_path` points to the SQLite DB file (defaults `./aitk_db.db`). The `DiffusionTrainer` will then write `status`, `info`, `step`, and `speed_string` into the `Job` row.
- Observe status in the UI (http://localhost:8675) or inspect SQLite directly: `sqlite3 aitk_db.db "SELECT status,info,step,speed_string FROM Job WHERE id = '<job-id>'"`.
- Use `--recover` to continue running subsequent config jobs even if one fails.
- Use `DEBUG_TOOLKIT=1` to enable torch anomaly detection for easier debugging (slower but more informative stack traces).

---

## Open questions / TODOs ❓
- Where and how should the Python worker surface more structured metrics to the UI (loss / latest checkpoint path / JSON) beyond `speed_string` and `step`? Some trainers write `latest_checkpoint` via `update_db_key` but usage is ad-hoc — consider centralizing.
- Confirm expected behavior with distributed training (Accelerate) and the UI database: only local main process updates DB; ensure this is understood when running Multi-GPU setups.
- Race conditions in DB writes: current SQLite writes use small `BEGIN IMMEDIATE` transactions; validate robust behavior under frequent updates and UI polling.
- Tests: add more unit/integration tests to cover the `--name` tag replacement and env var substitution failure modes (missing env raises ValueError currently).

---

## JSON manifest — files inspected and line ranges 📋
```json
{
  "files": [
    {"path": "run.py", "lines": "1-200"},
    {"path": "toolkit/job.py", "lines": "1-200"},
    {"path": "toolkit/config.py", "lines": "1-220"},
    {"path": "jobs/__init__.py", "lines": "1-40"},
    {"path": "jobs/BaseJob.py", "lines": "1-200"},
    {"path": "jobs/process/BaseProcess.py", "lines": "1-220"},
    {"path": "jobs/process/BaseTrainProcess.py", "lines": "1-240"},
    {"path": "jobs/process/BaseSDTrainProcess.py", "lines": "1-420"},
    {"path": "jobs/TrainJob.py", "lines": "1-120"},
    {"path": "extensions_built_in/sd_trainer/SDTrainer.py", "lines": "1-240"},
    {"path": "extensions_built_in/sd_trainer/DiffusionTrainer.py", "lines": "1-420"},
    {"path": "toolkit/accelerator.py", "lines": "1-200"},
    {"path": "toolkit/print.py", "lines": "1-200"},
    {"path": "toolkit/models/base_model.py", "lines": "320-420"},
    {"path": "toolkit/stable_diffusion_model.py", "lines": "1088-1128"}
  ]
}
```

---

If you want, I can:
1. Add explicit code references (line number quotes) for each of the assertions above, or
2. Extend the chapter with a small sequence diagram or a concise checklist for enabling UI-driven status updates and safe stopping.

Would you like me to add line-precise citations in the markdown (every assertion with file:line)?
