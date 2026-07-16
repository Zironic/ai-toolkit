# AI Toolkit UI

The web interface for AI Toolkit: start/stop/monitor training jobs, build job
configs, browse datasets and sample images, and (in this fork) read git-bug
tickets at `/tickets`. It is a Next.js App Router app (`src/`) plus a
background worker process.

## Architecture

Both `npm run dev` and `npm run start` launch **two processes** via
`concurrently`:

- **UI** — the Next.js app. Production (`npm run start`) serves on port
  **8675**.
- **Worker** — `cron/worker.ts`. It polls the database for queued jobs and
  actually launches/stops training runs, invoking the repo's Python through
  the project venv (`cron/pythonPath.ts`). Jobs keep running if the UI page is
  closed, but the worker process must be alive for jobs to start and be
  managed.

State lives in a SQLite database managed by Prisma (`prisma/schema.prisma`;
the database file `aitk_db.db` sits at the repo root). `npm run update_db`
runs `prisma generate` + `prisma db push` and is part of `build_and_start`.

## Commands

```bash
npm run build_and_start   # install deps, sync DB, build worker + UI, serve on :8675
npm run dev               # hot-reload dev server + worker
npm test                  # vitest
npm run lint              # next lint
npm run format            # prettier
```

## Notes

- The worker shells out to this repository's Python environment, so the UI
  must run **on the training machine** (repo checkout + venv + GPU). It is
  not deployable to Vercel or any host detached from the toolkit.
- To require an auth token, set the `AI_TOOLKIT_AUTH` environment variable
  before starting (see the root [README](../README.md)).
- The Tickets page (`/tickets`) reconstructs git-bug tickets by reading
  `refs/bugs/*` with read-only git (`src/server/gitbug.ts`); it never takes
  git-bug's store lock, so it is the safe way to browse tickets while the
  CLI is in use.
