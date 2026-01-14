# 01 — UI: Job Creation & User Actions ✅

TL;DR
- The UI surfaces job creation and control in `ui/src/app/jobs/*`. Submitting the job form POSTs JSON to the server API (`/api/jobs`), which uses Prisma to create/update a `Job` row in the SQLite DB. Starting a job triggers a separate API (`/api/jobs/{id}/start`) that updates queue fields and relies on the cron worker to schedule the actual process spawn.

Exact files & symbols referenced
- UI components:
  - `ui/src/app/jobs/new/*` (job form components, form data binding)
  - `ui/src/app/jobs/[id]/actions` (start/stop/delete handlers)
  - `ui/src/app/jobs/list` (job list view)
- API & server:
  - `ui/src/app/api/jobs/route.ts` (`POST`, `GET` handlers)
  - `ui/src/server/cron.ts` (cron loop that picks queued jobs)
  - `ui/src/lib/jobs.ts` (helpers such as `startJob`, `spawnJobProcess`)
- DB model:
  - `ui/prisma/schema.prisma` (`model Job` with fields `id`, `name`, `job_config`, `status`, `queue_position`, etc.)

Step-by-step runtime flow (user click → DB row)
1. User fills the UI form in `ui/src/app/jobs/new/*` and submits.
2. The client issues a POST to `/api/jobs` with body `{ name, job_config, ... }` (see `route.ts` POST handler).
3. Server handler uses Prisma to `create` or `update` a `Job` record. If a unique-name conflict occurs Prisma throws `P2002`; the handler translates this into HTTP 409.
4. UI navigates to the job detail page (`/jobs/{id}`) and renders job state from `GET /api/jobs/{id}`.
5. To start a job, the UI calls `POST /api/jobs/{id}/start`.
   - The handler computes `queue_position` as `(max queue_position) + 1000` and updates `status` (e.g., `queued`).
6. The server cron process (`cron.ts`) scans queued jobs and, when available, calls `spawnJobProcess` to run the python process and writes job metadata files (PID, logs) to the job run folder.

Important config values & environment variables
- DB path: configured in `schema.prisma` datasource (SQLite file at repo root: `file:../../aitk_db.db`).
- Job payload: `job_config` (stored as JSON string in DB).
- Process spawn env vars injected by `spawnJobProcess`: `AITK_JOB_ID`, `AITK_RUN_DIR`, `SOME_FEATURE_FLAG` (if set in server settings).
- Queue ordering uses numeric `queue_position` (default increments by 1000).

Persistence, transactions, error handling & race conditions ⚠️
- Unique constraint (name) handled via Prisma error `P2002` -> HTTP 409.
- `queue_position` is computed via `max + 1000` with no DB-side transaction—two concurrent starts can read the same max and create identical positions (race condition). Consider using DB transactions or an atomic increment pattern.
- Spawn failures are recorded by updating job `status` and writing a `spawn_error` log in the job folder.

How to observe / trace behavior
- Network: Inspect client POST to `/api/jobs` and `/api/jobs/{id}/start` in browser devtools.
- DB: Open `aitk_db.db` with sqlite3 and run `SELECT * FROM Job ORDER BY created_at DESC;`.
- Cron logs: check server logs or `ui/src/server/cron.ts` logging output.
- Files: check the job run directory (e.g., `runs/<job-id>/`) for `spawn.pid` and log files.

Open questions / TODOs
- Add an atomic DB-side queue increment or use Prisma transactions to avoid queue_position races.
- Improve server-side validation of `job_config` schema to fail early.
- Surface spawn failure reasons directly to the API response (not only DB status).

Files read (high-level)
- `ui/src/app/jobs/new/*` (form + wiring) — lines: component files
- `ui/src/app/api/jobs/route.ts` — POST and start handlers
- `ui/src/server/cron.ts` — queue loop and spawn code
- `ui/prisma/schema.prisma` — Job model

Notes & uncertainties
- Some UI code paths (e.g., advanced job templates) have multiple implementations; if you rely on a specific UI flow, note that both `new` and `import` flows exist and may differ in payload shape.

## Search hints
- `rg "POST /api/jobs|startJob|spawnJobProcess" -n ui/` — search UI handlers and spawn helpers
- `rg "Job" ui/prisma schema.prisma -n` — inspect DB model references
- Limit searches to `ui/` and `scripts/` unless you suspect server-side job loader code in `run.py` or `jobs/`.