# 03 — Worker Scheduling & Process Spawn ✅

TL;DR
- Starting a job via the UI sets `queue_position` and `status` (e.g., `queued`). The cron worker (`ui/src/server/cron.ts`) periodically scans the DB, picks the highest-priority queued job, and calls a spawn helper (e.g., `spawnJobProcess` / `startJob`) that writes run metadata, prepares the runtime folder, spawns a Python process (`run.py`) with injected environment variables, and records the spawned PID and log files in the run folder.

Files & symbols referenced
- `ui/src/app/api/jobs/[jobID]/start/route.ts` — start handler
- `ui/src/server/cron.ts` — top-level cron loop and invocation points
- `ui/src/lib/jobs.ts` / `ui/src/cron/actions/*.ts` — helpers like `startJob`, `spawnJobProcess`, `processQueue` and `processEvalQueue`
- `run.py` — Python entrypoint invoked by spawned processes
- `extensions_built_in/sd_trainer/SDTrainer.py` — trainer extension paths invoked by `run.py` (job-specific classes)

## Search hints
- `rg "spawnJobProcess|spawn_worker|cron" -n ui/ workers/ scripts/`
- `rg "queue_position|return_to_queue|status='queued'" -n`
- Prefer scanning `ui/src/server/cron.ts`, `scripts/` and `workers/` before a broad repo search.

Runtime flow (detailed)
1. User action or API call triggers `POST /api/jobs/{id}/start`:
   - Handler computes `queue_position = (max existing) + 1000`, sets `status = 'queued'`, and writes DB via Prisma. (See `route.ts` start handler.)
2. Cron loop (`cron.ts`) ticks (configurable interval):
   - Calls `processQueue()` / `processEvalQueue()` which list queued jobs ordered by `queue_position`.
   - Skips jobs already `running` or with active PID files.
3. Claiming job & spawn:
   - Worker calls `spawnJobProcess(job, opts)`:
     - Creates or cleans a run folder (e.g., `runs/{job.id}/run_{timestamp}`), writes `job_config.json` and rotates old logs.
     - Builds spawn command and args: typically `python run.py --config <path>` (Python executable detected from env/venv or `python` on PATH).
     - Injects env vars: `AITK_JOB_ID`, `AITK_RUN_DIR`, `AITK_FROM_UI=1`, `CUDA_VISIBLE_DEVICES` (if GPU pins present), and any settings flags.
     - Performs DB update marking job `status='running'` and records runtime `start_time` and `queue_position`.
     - Calls `child_process.spawn(cmd, args, options)` with detached mode where appropriate and streams stdout/stderr to a log file. Writes `spawn.pid` with child PID in run folder.
4. Failure or spawn error handling:
   - If spawn fails, the helper updates job `status='failed'` and writes `spawn_error` and the exception message to DB and run folder logs.
   - If spawn succeeds but process dies early, the liveness checker (`checkRunningJobs` or equivalent) detects missing PID or unexpected exit code and updates DB to `failed` or `stopped` with explanatory info.
5. Python process & DB updates:
   - `run.py` boots the job, instantiates the appropriate job class (e.g., `TrainJob`), and updates DB during runtime by opening the configured SQLite DB (path from Prisma schema or env). It writes periodic status updates (progress, step, speed, latest checkpoint info).

Run folder artifacts
- `job_config.json` — the JSON config the process received
- `spawn.pid` — PID of spawned child
- `spawn.log` / `stdout.log` / `stderr.log` — logs from the child process
- `artifacts/`, `samples/`, `checkpoints/` — produced by training process

Env vars & config used
- `AITK_JOB_ID` — job id string
- `AITK_RUN_DIR` — path to the run folder
- `CUDA_VISIBLE_DEVICES` / `GPU_IDS` — from job or settings
- `AITK_FROM_UI` — indicates spawned from UI/cron
- Python command selection: local virtualenv detection or `python` from PATH

Race conditions & suggested tests
- Queue position race: concurrent `start` calls may compute identical `queue_position` (fix: DB-side atomic increment or `UPDATE ... WHERE status='queued' LIMIT 1` pattern). Test: fire parallel start requests and assert unique queue positions.
- DB `'running'` set before `spawn.pid` written: if spawn fails after DB update, job may be left in `running` with no PID. Test: stub spawn to fail after DB update and assert the job becomes `failed` or is recovered by liveness check.
- Multiple cron workers: two workers could pick the same queued job if claiming isn't atomic. Test: run two cron workers concurrently and watch for duplicate runs.

How to observe
- DB: `sqlite3 aitk_db.db "SELECT id, status, queue_position, updated_at FROM Job ORDER BY queue_position;"`
- Cron logs: server logs where `cron.ts` prints scheduling decisions
- Run folder: check for `spawn.pid` and `spawn.log` in `runs/{job.id}/` directory
- Process: `ps` / Task Manager to observe child Python processes

Open questions / TODOs
- Make queue allocation atomic (use `prisma.$transaction` or DB-conditional update).
- Add playbook/test to simulate spawn failure between DB update and PID file write.
- Consider writing PID to DB as part of the same atomic operation that flips job to `running`.

Files inspected (high level)
- `ui/src/app/api/jobs/[jobID]/start/route.ts` (start handler)
- `ui/src/server/cron.ts` (cron loop and entry points)
- `ui/src/cron/actions/processQueue.ts` (queue processing)
- `ui/src/cron/actions/startJob.ts` (spawn helper)
- `run.py` (python entrypoint)
- `extensions_built_in/sd_trainer/SDTrainer.py` (trainer implementation)