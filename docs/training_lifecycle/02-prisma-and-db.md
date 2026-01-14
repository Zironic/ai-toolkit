# 02 — Prisma & DB ✅

TL;DR
- Prisma is configured to use a SQLite database stored at the repo root (referenced in `ui/prisma/schema.prisma`). Key models are `Job`, `EvalJob`, `Settings`, and `Queue`. API handlers and cron workers use the Prisma Client for CRUD; there are no formal migrations in the repository and concurrency control relies mostly on application logic (find → update) which leaves some race conditions.

Exact files & symbols referenced
- `ui/prisma/schema.prisma` — datasource and model definitions (check `model Job`, `model EvalJob`).
- API routes using Prisma client:
  - `ui/src/app/api/jobs/route.ts` (create/update jobs, handle `P2002` unique constraint)
  - `ui/src/app/api/eval/` (EvalJob handlers)
- Shared clients & worker usage:
  - `ui/src/lib/prismaClient.ts` (singleton client export)
  - `ui/src/server/cron.ts` (uses client to pick queued jobs)
- Scripts and tooling:
  - `scripts/prisma/push.sh` or `package.json` scripts referencing `npx prisma db push` or `prisma generate`.

## Search hints
- `rg "model Job|P2002|queue_position" prisma -n`
- `rg "create\(|update\(|prisma\.job" -n ui server`
- Start with DB schema (`ui/prisma/schema.prisma`) and then search server handlers in `ui/src/server/`.

Schema highlights
- The `Job` model includes:
  - `id` (PK), `name` (unique), `job_config` (JSON), `status` (string), `queue_position` (int), `created_at`, `updated_at`.
- `EvalJob` stores evaluation runs and outputs with fields: `id`, `dataset`, `model`, `params` (JSON), `status`, `out_json`.

How Prisma client is initialized & used
- Pattern: many handlers `import prisma from 'lib/prismaClient'` and call methods like `prisma.job.create`, `prisma.job.update`, `prisma.job.findMany`.
- No widespread `$transaction` usage was found; most operations are chained (read-then-write) on the application side.

Error codes & handling
- Unique-name violation => Prisma `P2002`. Handled with specific 409 responses in job create endpoints.
- Other DB errors bubble up to the server and are typically returned as HTTP 500 unless explicitly caught.

DB location & env config
- Datasource URL in `schema.prisma` points to `file:../../aitk_db.db` (SQLite file at repository root). There is no wrapping env var to change this path in all places.
- Prisma client generation & push commands are referenced in scripts; CI should run `npx prisma generate` after schema changes.

Tests & suggested tests
- There are currently **no** tests covering concurrency around queue position or spawn race behavior.
- Recommended tests:
  1. POST duplicate job names → 409 (P2002)
  2. Concurrent `start` requests → assert unique queue positions (or deterministic resolution)
  3. Worker spawn failure → job marked as `failed` and `spawn_error` populated
  4. EvalJob lifecycle tests (create → run → store `out_json`)

Observability & quick checks
- sqlite CLI: `sqlite3 aitk_db.db "SELECT * FROM Job ORDER BY created_at DESC;"`
- Prisma: `prisma studio --schema ui/prisma/schema.prisma` for a quick GUI.
- Add logging around critical updates (e.g., before/after queue position computes) for debugging

Open questions / TODOs
- Should the DB path be configurable via env to support non-local deployments?
- Add DB-side atomic operations (transactions or conditional updates) for safer queueing.
- Add CI test(s) for DB schema changes to avoid accidental breaking of generated client.

Files read (high-level)
- `ui/prisma/schema.prisma` — full schema
- `ui/src/app/api/jobs/route.ts` — create/update/start handlers
- `ui/src/lib/prismaClient.ts` — client wrapper
- `ui/src/server/cron.ts` — worker scanning

Uncertainties
- There is some non-uniformity in how different scripts reference the DB path; depending on the environment the actual runtime DB file might differ. If you plan to deploy to a server, confirm how the path is set or change to env-override.