# LEARNINGS.md

Purpose: an agent-first **change log of attempted approaches, experiments, failures, and final decisions**. Agents MUST check this file before making or repeating non-trivial changes and MUST add concise entries for any significant attempts or findings.

---

## How agents should use this file
- Always read recent entries (top → bottom) before running experiments, tests, or making changes that may re-run previous work.
- If an attempt fails or behaves unexpectedly, add an entry immediately (see template below) linking to logs, configs, and any follow-ups.
- Prefer short, actionable entries that include commands, config paths, exact error snippets (or links to logs), outcome, and recommended next steps.

---

## Entry template (copy & paste)

```
- date: 2025-12-25
  author: jdoe (or agent-id)
  tags: [training, checkpoint, fail]
  summary: "Interrupt during checkpointing corrupted the latest checkpoint"
  config: config/examples/train_lora_flux_24gb.yaml
  commands:
    - python run.py config/examples/train_lora_flux_24gb.yaml
  logs: path/to/logs/run123.log (or link)
  outcome: "corrupted checkpoint files; training resumed from previous checkpoint after manual restore"
  notes: "Do not Ctrl+C while saving; add pre-save fence or atomic write logic to checkpointing code"
  follow_up: "Add test for atomic checkpoint write; consider making temporary file + rename"
  pr: https://github.com/ostris/ai-toolkit/pull/123  # optional
```

- Use YAML-like lines for machine parsing; agents should parse this block or fallback to plain-text parsing if needed.

---

## Recent entries (most recent on top)

- date: 2025-12-25
  author: GitHub Copilot
  tags: [dataset-eval, backend, ui, prisma, tests]
  summary: "Implemented dataset evaluator: streaming per-item aggregator, CLI, trainer wiring, API endpoints, and worker integration; smoke tests passed and Prisma client issues handled"
  config: `tools/eval_dataset.py`, `toolkit/util/loss_utils.py`, `extensions_built_in/sd_trainer/SDTrainer.py`, `jobs/process/BaseSDTrainProcess.py`, `ui/src/app/api/eval_dataset/*`, `ui/cron/actions/processEvalQueue.ts`
  commands:
    - python tools/eval_dataset.py --dataset-path datasets/test_smoke_eval --model <model> --batch-size 1 --out-dir datasets/test_smoke_eval --job-name test_eval --step 0
    - npx prisma@6.3.1 generate --schema ui/prisma/schema.prisma
    - npx prisma@6.3.1 db push --schema ui/prisma/schema.prisma --accept-data-loss
    - npx ts-node -P ui/tsconfig.worker.json ui/scripts/run_process_eval.ts
  logs: `datasets/test_smoke_eval/<jobid>_000000000.json`, DB `aitk_db.db` EvalJob rows
  outcome: "Added streaming aggregator and compute_per_example_loss, wired trainer to capture per-example scalar losses, added run_dataset_evaluation wrapper, created CLI to run dataset evaluation, added API endpoints and a worker action that runs evals and writes JSON into dataset folders; smoke test generated JSON successfully; one queued job failed due to missing script path which is now handled and recorded in DB 'info'"
  notes: "Prisma v7 schema format is not compatible with the project's current schema; we used `npx prisma@6.3.1` to generate client and push schema. Worker now captures stdout/stderr and persists concise failure messages to `EvalJob.info`. Per-example logging is gated to single-batch/no-grad-accum runs to avoid misleading partial data. Evaluation writes `{job_name}_{step_zfilled}.json` into the dataset folder. A session checkpoint was saved as `SESSION_CHECKPOINT.md` and `session_checkpoint.json` for resumption."
  follow_up:
    - "Add UI components for dataset evaluator: run UI (model selector), show EvalJob status and `info` for failures, link to JSON report in dataset folder"
    - "Add unit and E2E tests for API endpoints, worker, and UI integration"
    - "Consider adding an explicit requeue/retry policy and a 're-run' API in the UI"
    - "Document the EvalJob table and the `info` field interpretation in AGENTS.md"

- date: 2025-12-25
  author: GitHub Copilot
  tags: [ui, tests, navigation]
  summary: "Nested `ui/ui` confusion happened while running Playwright tests — commands executed from `ui/ui` instead of top-level `ui` causing missing config and module errors"
  commands:
    - Get-Location (showed `.../ui/ui`)
    - npx playwright test --config=playwright.config.ts tests/e2e/eval_ui.spec.ts
  logs: "Error: C:\...\ui\playwright.config.ts does not exist (Playwright CLI looked up nested path); Error: Cannot find module '@playwright/test' when config was resolved from wrong cwd"
  outcome: "Discovered that a nested `ui` package exists (`ui/ui`) which caused Playwright to resolve config and modules from the wrong folder. Tests failed due to wrong CWD and missing `@playwright/test` in top-level `ui` package."
  notes: "Agents must ensure they run UI-related commands from `ui/` root (or use `npm --prefix ui`), and check `Get-Location` / `pwd` if tests fail to locate config. Added guidance to `AGENTS.md` with a small checklist for Playwright setup and a troubleshooting tip (use `npm --prefix ui exec playwright test ...` to avoid incorrect nested paths). Also: prefer running production server (`npm run build && npm run start`) for Playwright to avoid dev overlays; alternatively, use the overlay-removal helper included in `ui/tests/e2e/test_helpers.ts` to remove dev overlay before clicking.
  follow_up:
    - "Add small CI smoke job that runs `npx playwright test --config=playwright.config.ts` in `ui/` to catch path issues before commits"
    - "Add a quick sanity script `ui/scripts/check_playwright_setup.sh` that ensures `@playwright/test` is installed and Playwright browsers are present"
    - "Add a Playwright fixture that runs a `curl` health-check and removes the dev overlay before tests run"

- date: 2025-12-25
  author: GitHub Copilot
  tags: [ui, tests, server]
  summary: "Starting the UI dev server interactively can block Playwright tests and cause connection failures if tests are run in the same terminal session"
  commands:
    - npm --prefix ui run dev  # started interactively
    - npm --prefix ui exec playwright test tests/e2e/eval_ui.spec.ts  # ran in same terminal and observed failure
  logs: "Playwright failure: page.goto: net::ERR_CONNECTION_REFUSED when visiting http://localhost:3000/datasets/test_smoke_eval"
  outcome: "Tests failed because the dev server was not available to Playwright; starting the server interactively in the same terminal can leave it unavailable for the test runner."
  notes: "Avoid starting the dev server in the same interactive terminal you use to run tests. Start it in background or another terminal, or use PowerShell `Start-Process` / `Start-Job` or `nohup` on POSIX. Use the `scripts/check_playwright_setup.*` helpers or a quick `curl` health-check before running tests."
  follow_up:
    - "Add a Playwright fixture or pre-check that verifies `http://localhost:3000` is reachable before executing tests"
    - "Add a CI smoke job that starts the server in background, waits for readiness, then runs Playwright smoke tests"

- date: 2025-12-17
  author: CI-agent
  tags: [license, hf_token, training]
  summary: "Training failing when using `black-forest-labs/FLUX.1-dev` due to missing HF token or license acceptance"
  config: `config/examples/train_lora_flux_24gb.yaml` (model.name_or_path=black-forest-labs/FLUX.1-dev)
  commands:
    - python run.py config/examples/train_lora_flux_24gb.yaml
  logs: `output/flux_lora_20251217/error.log`
  outcome: "run aborted with HF permission error"
  notes: "FLUX.1-dev requires HF license acceptance and `HF_TOKEN` with read access; add pre-check to fail early with clear instructions"
  follow_up: "Add a check in training process to verify HF_TOKEN and model access before starting heavy downloads"

- date: 2025-10-03
  author: manual-note
  tags: [modal, volumes]
  summary: "Outputs not persisted when Modal volume not committed"
  config: run_modal.py
  commands:
    - python run_modal.py config/my_training_config.yml
  logs: `modal/logs/run_modal_commit_issue.txt`
  outcome: "Model files were not saved to persistent volume until `model_volume.commit()` called"
  notes: "Ensure wrapper scripts call commit() after training; document in AGENTS.md"
  follow_up: "Consider adding a safety commit on graceful shutdown"

---

## Known pitfalls & short rules (agents: read before running!)
- Never start a heavy training job without human confirmation or scheduling into an explicit GPU worker.
- When using gated HF models (FLUX.1-dev), ensure `HF_TOKEN` is set and model access accepted.
- If a run fails during saving, avoid deleting partial artifacts; prefer preserving for inspection.

---

## How to add an entry (guide)
1. Add a concise entry (as above) to the top of this file in a new `- date:` block.
2. Include links to logs and PR/issue numbers when available.
3. Open a PR for non-trivial findings if you made code changes or tests to fix the issue.
4. If the entry documents a failed experiment, include the command and minimal config so others (or agents) can reproduce quickly.

---

## Machine-friendly access
- Entries are YAML-like; agents should prefer parsing using simple heuristics (look for `- date:` blocks and parse indented fields).
- Short, consistent keys (`date`, `author`, `tags`, `summary`, `commands`, `logs`, `outcome`, `notes`, `follow_up`, `pr`) make automated processing easier.

---

## Governance
- Treat LEARNINGS.md as a living source-of-truth for attempted approaches and anti-patterns.
- Prefer accurate, short, and factual entries over long narratives.
- If adding a follow-up PR/issue, update the entry to include the PR/issue link.

---

If you want, I can:
- seed this file with more historical entries from existing issues/logs, or
- add a small parser script `scripts/parse_learnings.py` to extract entries for agent consumption.
