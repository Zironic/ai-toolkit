# 09 — Reproducibility, Troubleshooting & Best Practices ✅

TL;DR
- Reproducibility requires setting and recording seeds (torch, torch.cuda, numpy, random), saving the full run metadata (config, `software` info via `toolkit/metadata.get_meta_for_safetensors`), and minimizing non-deterministic operations (avoid non-deterministic kernels or document them). For troubleshooting, inspect UI DB fields (`status`, `step`, `info`), cron logs, run folder (`spawn.pid`, `spawn.log`), and trainer timers to pinpoint performance or failure points.

Files & symbols referenced
- `jobs/process/BaseTrainProcess.py` / `BaseSDTrainProcess.py` — where `training_seed` is read and applied (`torch.manual_seed`, `torch.cuda.manual_seed`, `random.seed`) (lines ~1-40)
- `toolkit/train_tools.py` — helpers for seed extraction from latents and seed utilities
- `toolkit/metadata.py` — how to record software metadata and save it with checkpoints
- `toolkit/timer.py` and `SDTrainer` — timing hooks for performance diagnostics
- `ui/src/server/cron.ts` and `ui/src/app/api/jobs` — spawn & DB interactions to debug job lifecycle issues

Repro best practices
- Always set `training_seed` (job-level or process-level config): `BaseTrainProcess` uses `self.get_conf('training_seed')` and if present calls `torch.manual_seed`, `torch.cuda.manual_seed` and `random.seed`. Set an explicit seed in config to be reproducible.
- Log and save the effective seed in checkpoint metadata via `meta` fields so recoveries record the same seed and runtime environment.
- Lock the environment:
  - Save Python package versions (e.g., via `pip freeze > requirements.txt`) and include `software` metadata in safetensors via `get_meta_for_safetensors()`.
  - Document hardware (CUDA version, GPU model), acceleration flags (xformers), and `accelerate` runtime configuration.
- Avoid or document non-deterministic ops:
  - Some ops (cuDNN algorithms, certain fused kernels, xformers attention) can be non-deterministic across runs — prefer deterministic flags or document their effects.

Common failure modes & how to triage
- Cron / Spawn failures:
  - Symptom: Job `status='running'` in DB but no PID file or no process.
  - Check: `runs/<job-id>/spawn.pid`, server/cron logs, `ui` API logs, and `stderr` in `spawn.log`.
  - Fix: Ensure spawn helper writes PID only after successful spawn or add a rollback on spawn exception. Test by forcing spawn to raise after DB update and assert recovery.
- DB race conditions (queue_position):
  - Symptom: Two started jobs share identical queue positions or conflicting ordering.
  - Check: Inspect `Job` table for `queue_position` duplicates; run concurrent `start` requests and observe resulting `queue_position` values.
  - Fix: Add atomic DB increment or conditional `UPDATE ... WHERE status='queued'` pattern and add a test that runs concurrent starts.
- OOM & Memory errors:
  - Symptom: Out-of-memory exception in UNet/adapter loading or ControlNet.
  - Check: Trainer logs, `controlnet_streaming` config, `ram` checks in `hook_before_train_loop` that suggest streaming, and whether `xformers` or offloading are enabled.
  - Fix: Enable `controlnet_streaming`, `xformers`, or smaller batch sizes; use `model_config.controlnet_mode` or offload options.
- Checkpoint corruption / save failures:
  - Symptom: Missing or incomplete `.safetensors` files, corrupted metadata.
  - Check: Look for `print_acc` output around `Saved checkpoint` and examine `.safetensors` metadata with `safetensors.safe_open`.
  - Fix: Ensure `save()` is called only from main process; guard with `if accelerator.is_main_process` and add `atomic` temporary file write + move.

Observability checklist (where to look)
- UI DB: `SELECT id,status,step,info FROM Job ORDER BY updated_at DESC;` — check job status and info.
- Run folder: `runs/<job-id>/spawn.log`, `spawn.pid`, `latest checkpoint` files and `aitk_meta.yaml`.
- Server logs: cron task logs (cron.ts), server console for spawn errors.
- Trainer logs: `print_acc` and `Timer.print()` outputs, `logger` sqlite tables for metrics.
- System: `dmesg`/Windows Event logs for OOM/kill events.

Tests & recommended automation
- Add concurrency test for `start` queue position collisions: simulate N concurrent `POST /api/jobs/{id}/start` and assert queue positions are unique or that only one becomes running.
- Add spawn-failure rollback test: mock spawn helper to throw after DB 'running' update and ensure liveness checker transitions job to `failed` with spawn_error.
- Add deterministic re-run test: run a small training job with `training_seed` set and assert that model outputs (samples or some deterministic metrics) are identical across runs (within floating variance); use CPU or single GPU to avoid multi-process nondeterminism.

Notes & uncertainties
- Some deterministic flows (like exact weight bytes) may still differ across PyTorch versions or if FSDP / different offload/device mapping is used — always record full environment metadata.
- ControlNet and certain adapters rely on optional libs (e.g., `controlnet_aux`) which, if missing, produce deterministic errors; ensure preflight checks (some exist in `hook_before_train_loop`) are deterministic and clear.

Quick triage script ideas
- `scripts/diagnose_job.py <job-id>` that prints job DB row, checks PID file, tails latest logs, and lists saves in `save_root` — this would standardize troubleshooting steps for maintainers.

---

References and files inspected: `jobs/process/BaseTrainProcess.py`, `BaseSDTrainProcess.py`, `toolkit/metadata.py`, `toolkit/train_tools.py`, `toolkit/timer.py`, `ui/src/server/cron.ts`, `ui/src/app/api/jobs`.
