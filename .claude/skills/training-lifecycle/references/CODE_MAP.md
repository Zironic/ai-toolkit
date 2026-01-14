# Code map — training lifecycle

**Derived from `docs/training_lifecycle/*`. Agents should consult the corresponding chapter in `docs/training_lifecycle/` first; use this file as a quick-code lookup only.**

This file maps the training lifecycle docs to key files and symbols in the repository so agents immediately know where to look when performing tasks related to training jobs.

## 01-ui-job-creation
- docs: `docs/training_lifecycle/01-ui-job-creation.md`
- key files: `ui/src/app/api/jobs/*` — UI form handlers and `run.py` job config mapping
- helpful symbols: `JobForm`, `createJob`, `JobConfig`

## 02-prisma-and-db
- docs: `docs/training_lifecycle/02-prisma-and-db.md`
- key files: `prisma/` (schema), `ui/src/app/models/`, `toolkit/db/*`
- helpful symbols: Prisma migrations, `Job` model, DB transaction helpers

## 03-worker-scheduling
- docs: `docs/training_lifecycle/03-worker-scheduling.md`
- key files: `ui/cron/`, `workers/`, `scripts/spawn_worker.py`
- helpful symbols: `spawn_worker`, `schedule_job`, `process_queue`

## 04-run-py-and-job-loader
- docs: `docs/training_lifecycle/04-run-py-and-job-loader.md`
- key files: `run.py`, `toolkit/job_loader.py`, `jobs/`
- helpful symbols: `JobLoader`, `RunJob`, `load_job_from_config`

## 05-pipeline-and-jobs
- docs: `docs/training_lifecycle/05-pipeline-and-jobs.md`
- key files: `toolkit/stable_diffusion_model.py`, `jobs/`, `toolkit/model_utils.py`, `extensions_built_in/sd_trainer/SDTrainer.py`
- helpful symbols: `StableDiffusion`, `apply_lora`, `PipelineJob`, `LoRA`, `assistant_lora_path`

## preservation-loss & DOP
- docs: `docs/training_lifecycle/07-training-loop.md`
- key files: `extensions_built_in/sd_trainer/SDTrainer.py`
- helpful symbols: `preservation_loss`, `_compute_and_apply_preservation_loss`, `_last_preservation_loss`
## 06-pretraining-setup
- docs: `docs/training_lifecycle/06-pretraining-setup.md`
- key files: `toolkit/dataloader_mixins.py`, `toolkit/config_modules.py`, `scripts/auto_crop_to_bucket.py`
- helpful symbols: `DatasetConfig`, `auto_crop_to_bucket`, `bucket_logic`

## 07-training-loop
- docs: `docs/training_lifecycle/07-training-loop.md`
- key files: `jobs/process/*`, `toolkit/train_tools.py`, `toolkit/accelerator.py`
- helpful symbols: `train_step`, `optimizer`, `EMA`, `mixed_precision`

## 08-checkpoints
- docs: `docs/training_lifecycle/08-checkpoints.md`
- key files: `toolkit/metadata.py`, `checkpoints/`, `jobs/save_checkpoint.py`
- helpful symbols: `save_checkpoint`, `load_checkpoint`, checkpoint metadata fields

## 09-repro-troubleshoot
- docs: `docs/training_lifecycle/09-repro-troubleshoot.md`
- key files: `tools/repro/`, `scripts/` (seed logging), `toolkit/util/loss_utils.py`
- helpful symbols: `set_seed`, `deterministic_ops`, `capture_traceback`

---

If you want, I can expand each entry with direct source file links (line ranges) and short descriptions of responsibilities for each symbol. This file is intended to be a compact first-stop for agents when they activate the skill.
