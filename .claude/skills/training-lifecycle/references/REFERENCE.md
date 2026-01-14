# Training lifecycle - condensed reference

**Agent guidance: Read the chapter(s) in `docs/training_lifecycle/` before taking action — this file is a condensed summary of those chapters.**

This file summarizes the key points for each chapter in `docs/training_lifecycle/` and provides a short action list.

## 01-ui-job-creation
- Purpose: How to create jobs via the UI and map fields to job configs.
- Action items: Validate form inputs; add short smoke configs for CI.

## 02-prisma-and-db
- Purpose: DB schema, transactions, and job persistence.
- Action items: Ensure DB migrations run in CI; verify transactions for job creation.

## 03-worker-scheduling
- Purpose: Cron, queue handling, worker spawn rules.
- Action items: Check cron schedules, backoff policies, and spawn helper settings.

## 04-run-py-and-job-loader
- Purpose: `run.py` entrypoint and job loader behavior.
- Action items: Validate loader for new job types; add unit tests for loader edge cases.

## 05-pipeline-and-jobs
- Purpose: Pipeline loaders, model application, job types.
- Action items: Verify model config compatibility and LoRA application steps.

## 06-pretraining-setup
- Purpose: Data transforms, buckets, and scheduler wiring.
- Action items: Validate auto-crop and bucket logic; add dataset smoke tests.

## 07-training-loop
- Purpose: Training loop mechanics, optimizers, mixed precision.
- Action items: Add small-step smoke runs and monitor for NaNs or divergence.

## 08-checkpoints
- Purpose: Checkpoint saving, metadata, and hub push.
- Action items: Verify checkpoint integrity, metadata fields, and downstream load.

## 09-repro-troubleshoot
- Purpose: Repro and troubleshooting best practices.
- Action items: Add reproducibility checks (seed logging, deterministic ops) and error capture.

---

Run the included `scripts/generate_checklist.py` to create a task checklist from these notes.
