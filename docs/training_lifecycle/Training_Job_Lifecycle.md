# Training Job Lifecycle — Master Guide ✅

This document links the chaptered guides that explain the full lifespan of a training job in this repo — from UI creation through DB, scheduling, pipeline setup, training, checkpointing and troubleshooting.

Chapters
- 01 — UI: Job Creation & User Actions (`01-ui-job-creation.md`)
- 02 — Prisma & DB (`02-prisma-and-db.md`)
- 03 — Worker Scheduling & Process Spawn (`03-worker-scheduling.md`)
- 04 — `run.py` & Job Loader (`04-run-py-and-job-loader.md`)
- 05 — Pipeline Setup & Job Types (`05-pipeline-and-jobs.md`)
- 06 — Pre-training Setup (Data, Transforms, Schedulers) (`06-pretraining-setup.md`)
- 07 — Training Loop (Optimizer, Mixed Precision, Logging) (`07-training-loop.md`)
- 08 — Checkpoint Saving & Metadata (`08-checkpoints.md`)
- 09 — Repro, Troubleshooting & Best Practices (`09-repro-troubleshoot.md`)

Verification checklist (quick start tests & checks)
1. UI Job Creation
   - Create job via UI: POST to `/api/jobs`; verify `Job` row created in `aitk_db.db` with `name`, `job_config` and `status` fields.
   - Check: `sqlite3 aitk_db.db "SELECT id, name, status, queue_position FROM Job ORDER BY created_at DESC;"`
2. Start & Queue
   - Start job via `/api/jobs/{id}/start`; verify `status='queued'` and `queue_position` assigned.
   - Simulate two concurrent starts to validate queue ordering consistency.
3. Worker Scheduling & Spawn
   - Confirm cron picks job and that `spawn.pid` and `spawn.log` are written in the run folder.
   - Verify DB `status='running'` and that PID corresponds to a live Python process.
4. run.py & Job Loader
   - Inspect `run.py` invocation in spawn command. Run manually: `python run.py config/example.yaml --log tmp.log` and check printed start and end logs.
   - Ensure `toolkit.config` env var substitutions succeed or fail clearly when missing.
5. Pipeline & Model Loading
   - Validate model loads by running a small `GenerateJob` config and ensuring `sd.predict_noise()` path exercises the pipeline loader.
   - Test adapter/LoRA paths and verify `load` vs `apply` semantics.
6. Data & Pre-training
   - Run small dataset config and inspect `Dataset` bucket assignment logs; verify bucket sizes and that `setup_buckets()` prints expected counts.
   - Test augmentation disabled with latents caching and assert code warns and disables caching.
7. Training Loop
   - Run a tiny training job and confirm `Timer` prints and `update_step()` updates `Job.step` in DB, `speed_string` is present, and `logger` wrote expected scalars.
   - Validate gradient accumulation counts & optimizer steps via debug prints.
8. Checkpoints & Metadata
   - Ensure `.safetensors` files appear in `save_root`, contain `metadata` keys (use `safetensors.safe_open()`), and `add_model_hash_to_meta()` added model hashes.
   - Confirm `clean_up_saves()` keeps `max_step_saves_to_keep` newest saves.
9. Repro & Troubleshoot
   - Run a deterministic test with `training_seed` set and assert sample generation seed determinism for a simple model.
   - Use `scripts/diagnose_job.py` (suggested) to centralize triage steps.

How to extend this guide
- Add more chapter sections if you add new job types, new adapter formats, or new storage backends (e.g., S3/HF dataset staging).
- Add integration tests under `testing/` for the concurrency/edge cases noted here.

If you want, I can: generate a combined manifest of all files consulted and a summarized 'uncertainties & TODO' list for maintainers, or scaffold the recommended tests and submit them as a PR. Which do you prefer?