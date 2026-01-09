# Chapter 07 — Training loop (optimizer, mixed precision, logging) ✅

TL;DR
- The training loop is implemented across `jobs/process/BaseSDTrainProcess.py` (loop orchestration, accumulation, scheduler stepping, I/O, save/sample hooks), and `extensions_built_in/sd_trainer/SDTrainer.py` (per-batch processing, forward passes, loss computation, backward, optimization step, gradient clipping, EMA). It uses `accelerate.Accelerator` for distributed/mixed-precision support and manages logging via `toolkit.timer.Timer`, `toolkit.logging_aitk.UILogger` (DB), and TensorBoard (`SummaryWriter`).

---

## Files & Symbols Referenced 🔧
- `extensions_built_in/sd_trainer/SDTrainer.py`
  - `SDTrainer::hook_before_train_loop()` — init canvas & embedding caching, control net checks (lines ~450-500)
  - `SDTrainer::train_single_accumulation()` — per-accumulation-step forward, loss, backward (lines ~2420-3040)
  - `SDTrainer::hook_train_loop()` — collects batch(es), handles optimizer step, clipping, EMA update, scheduler step (lines ~4360-4670)
  - `self.accelerator.backward(loss)` usages (backward via Accelerator) — see occurrences around lines ~1524, ~4040, ~4296
  - Timers around critical stages: `with self.timer('...')` (many places; examples: `step_total_python` at ~2444, `predict_unet` ~3951, `after_unet_predict` ~2309)
- `jobs/process/BaseSDTrainProcess.py`
  - Main train loop orchestration: data iteration, gradient-accumulation management, `Accelerator.accumulate(...)` context, .step bookkeeping, save/sample intervals, logger commits, perf prints (lines ~2760-3240)
  - Model init & optimizations toggles (xformers, attention backend, compile) (lines ~2276-2336)
  - Accelerator setup via `toolkit/accelerator.get_accelerator()` (top of file)
- `extensions_built_in/sd_trainer/DiffusionTrainer.py`
  - UI-specific hooks, DB updates: `update_step()`, `handle_timing_print_hook()` updates `speed_string` in DB based on `Timer.print()` (lines ~1-380)
  - `on_error()` and `maybe_stop()` / `should_save_before_stop()` (lines ~1-380)
- `toolkit/timer.py` — `Timer` implementation (`start()`, `stop()`, `add_after_print_hook()`, `print()`) used for timing & performance logging (lines ~1-200)
- `toolkit/ema.py` — EMA implementation; `ema.update()` called after optimizer step (lines ~1-400)
- `toolkit/optimizer.py` / `toolkit/scheduler.py` — `get_optimizer` / `get_lr_scheduler` used to instantiate optimizers/schedulers (support for dadaptation, bitsandbytes, adafactor etc.)
- `toolkit/accelerator.py` — helper for getting `Accelerator()` (simple wrapper) and unwrapping models
- `toolkit/logging_aitk.py` — `UILogger` commits per-step metrics to SQLite DB (schema: `steps`, `metrics`), used from `BaseSDTrainProcess.logger`.

---

## Detailed description of the training loop 🔍

### 1) High-level orchestration (loop, epochs, batches)
- The main loop lives in `BaseSDTrainProcess.run()` (lines ~2760-3240).
  - `for step in range(start_step_num, self.train_config.steps):` iterates logical training steps.
  - For each step it forms a list `batch_list` by pulling `gradient_accumulation` many dataloader batches (this enables per-step multi-batch processing and alternating reg/dataset logic).
  - The Accelerator context `with self.accelerator.accumulate(self.modules_being_trained):` wraps the call to `hook_train_loop(batch_list)` so that the `Accelerator` handles gradient accumulation synchronization and scaling behind the scenes.
  - `self.is_grad_accumulation_step` and `self.grad_accumulation_step` flags are updated in `BaseSDTrainProcess` to determine whether an optimizer step should occur.

### 2) Data iteration & preprocessing
- `SDTrainer.train_single_accumulation()` (lines ~2420-3040) receives a `DataLoaderBatchDTO` and:
  - Runs `preprocess_batch()` and `process_general_training_batch()` to produce latents, noise, timesteps, conditioned prompt embeds etc.
  - Handles optional single-item batching, adapter/controlnet images, clip images and mask multipliers, prior regularization flags.
  - Uses `self.timer('...')` contexts liberally to measure each phase (e.g., `get_adapter_images`, `encode_prompt`, `encode_adapter`, `controlnet_forward`, `predict_unet`).

### 3) Forward pass & prediction
- Unet predictions are computed through `self.predict_noise(...)` which delegates to `self.sd.predict_noise(...)` — predictions happen inside measured timers (e.g., `predict_unet`).
- Adapter/ControlNet residuals are computed on-the-fly when `adapter` is present; offload strategies (e.g., `accelerate`) can be applied during controlnet compute via `controlnet_offload` helper (timers: `controlnet_forward` and `controlnet_offload`).
- For special features (e.g., diff_output_preservation, prior-prediction), `get_prior_prediction()` and DOP code compute preservation losses and may produce additional predictions.

### 4) Loss computation
- Primary loss: MSE between model output and target (v-target or other flow targets), often via `torch.nn.functional.mse_loss(..., reduction='none')` then mean.
- Additional losses: preservation loss, masked reconstruction, guidance-based losses, diffusion feature extractor (DFE) weighted losses, optional per-example losses.
- Preservation loss (DOP) recorded and backwarded if it participates in autograd (lines ~4280-4300). The trainer sets `_last_preservation_loss` for logging.
- The code tracks diagnostic scalars (e.g., `last_noise_norms`, `last_loss_over_noise`) for monitoring.

### 5) Backward pass & mixed precision
- Backward is performed via `self.accelerator.backward(loss)` (uses `Accelerate` to handle gradient scaling and mixed precision when configured).
  - There are multiple `self.accelerator.backward(...)` invocations (e.g., main loss ~1524, preservation loss ~4296 and other spots) so any loss contributing to the optimization graph is backwarded under Accelerator.
- Mixed-precision / gradient scaling behavior:
  - The repository relies on `accelerate.Accelerator()` to manage mixed-precision and scaling (no global custom scaler object used explicitly in most codepaths). The `TrainConfig.dtype` controls `fp16` / `bf16` behavior and model dtypes are set accordingly (see `get_torch_dtype(...)`).
  - There is an older, commented patch to `self.scaler._unscale_grads_` in `SDTrainer.__init__` (lines ~140-152) — left as comments and likely not used.
  - Some operations explicitly use `torch.autocast` in other tools (e.g., dataset encoding). The core flow uses dtypes and Accelerate's autocast/scaler when Accelerator is configured.

### 6) Gradient accumulation & optimizer step
- Accumulation is controlled by two config fields: `train_config.gradient_accumulation` (how many loader batches to fill per step) and `train_config.gradient_accumulation_steps` (how many *gradient steps* to accumulate before `optimizer.step()`).
  - The main loop builds `batch_list` with `gradient_accumulation` size and sets `self.is_grad_accumulation_step`/`self.grad_accumulation_step` appropriately (see lines ~2790-2880).
  - `Accelerator.accumulate(...)` context ensures distributed synchronization of gradients at the right moments.
- After each accumulation group, `SDTrainer.hook_train_loop()` performs optimizer step only when `not self.is_grad_accumulation_step` (i.e., when an optimizer step is due):
  - Gradient clipping is applied if optimizer is not `adafactor`: `self.accelerator.clip_grad_norm_(...)` inside a timer `clip_grad` (lines ~4423-4440).
  - `self.optimizer.step()` is executed inside timer `optimizer_step`. Execution guarded by try/finally with logs to indicate if step executed (lines ~4447-4460).
  - After step, `self.optimizer.zero_grad(set_to_none=True)` is called (timer `zero_grad_set_to_none`).
  - `self.ema.update()` is called (timer `ema_update`) if an EMA exists.
- Scheduler step: `self.lr_scheduler.step()` is called every training step (timer `scheduler_step`), with a comment TODO: "Should we only step scheduler on grad step?" (lines ~4480).

### 7) Grad clipping & diagnostics
- Grad clipping: `self.accelerator.clip_grad_norm_` applied to param groups, with support for `params` being either lists or dict-structured param groups (lines ~4425-4440).
- Diagnostics: `grad_diagnostics` timer computes L2 norm of gradients for reporting (attempts to be robust to missing grads). Debug logs print `is_grad_accumulation_step` and grad norm.

### 8) EMA updates
- EMA object is provided by `toolkit.ema.ExponentialMovingAverage` and updated after optimizer step with `self.ema.update()` inside `hook_train_loop()` (timer: `ema_update`). EMA state is saved in `save()` via `self.ema.eval()` and copying to the checkpoint (see `BaseSDTrainProcess.save()`).

### 9) Distributed / accelerator specifics
- Accelerator provided by `accelerate.Accelerator()` (via `toolkit.accelerator.get_accelerator()`). Major interactions:
  - `accelerator.accumulate(self.modules_being_trained)` wraps accumulation loops so `Accelerator` can handle gradient sync/scaler behavior.
  - `accelerator.backward(loss)` used for backward (handles scaler/amp under the hood).
  - `accelerator.clip_grad_norm_(...)` used for gradient clipping.
  - `accelerator.wait_for_everyone()` used at save/sample boundaries before operations requiring global consistency.
  - At end: `accelerator.end_training()` called to finalize.

### 10) Xformers & Attention backends
- XFormers memory-efficient attention toggled with `train_config.xformers` in `BaseSDTrainProcess` where `unet.enable_xformers_memory_efficient_attention()` and `vae.enable_xformers_memory_efficient_attention()` are called (lines ~2276-2320).
- Attention backend (flash, native, other variants) set with `set_attention_backend()` on VAE/unet/text-encoder when `train_config.attention_backend != 'native'`.

---

## Logging & Observability 📊

### Timers & performance
- `toolkit.timer.Timer` is used across the codebase to measure named intervals (`start()/stop()` and `with self.timer('name'):`).
- `Timer.print()` consolidates statistics and calls any `after_print_hook`s.
  - `DiffusionTrainer.hook_before_train_loop()` registers `self.handle_timing_print_hook` as an after-print hook so the UI trainer receives timing summaries and can update DB fields (e.g., `speed_string`) (lines ~280-320 of `DiffusionTrainer.py`).
  - `handle_timing_print_hook` converts `train_loop` avg seconds into either iter/sec or sec/iter and calls `update_db_key('speed_string', ...)`.

### DB & step updates
- `DiffusionTrainer.update_step()` is a non-blocking DB update of the `step` column (calls `_update_key('step', self.step_num)` in an async threadpool task). `end_step_hook()` in `DiffusionTrainer` calls `update_step()` and `maybe_stop()` each step.
- `toolkit.logging_aitk.UILogger` receives key-value logs via `BaseSDTrainProcess.logger.log(...)` then `logger.commit(step=self.step_num)` to persist metrics in an SQLite DB (`steps`, `metrics` tables).
- TensorBoard: `SummaryWriter` writes per-step scalars in `BaseSDTrainProcess` when `logging_config` and `log_dir` are enabled.

### Which metrics are logged? (examples)
- `loss/loss` (total), `loss/preservation`, `train/noise_mean`, `train/loss_over_noise`, per-example losses (`per_example`), `lr`, and other keys found in `loss_dict` produced by `hook_train_loop()` (SDTrainer) and committed by logger in `BaseSDTrainProcess`. The UI updates DB fields like `speed_string` via `handle_timing_print_hook`.

---

## Checkpointing, save hooks & error handling 💾

- Checkpoint saves: `BaseSDTrainProcess.save(step)` handles checkpoint metadata, filename convention (`{job.name}_{step}.safetensors`), and uses EMA weights (it calls `self.ema.eval()` before saving) (lines ~920-1020).
- Save triggers: periodic saves occur when `step_num % save_every == 0` (checked in `run()`), and `save()` is called while the process is the main Accelerator process; the trainer waits (`accelerator.wait_for_everyone()`) where necessary before saving.
- Error handling & stop/queue behavior:
  - `DiffusionTrainer.maybe_stop()` checks DB for `stop` or `return_to_queue` signals and, if present, may `save()` and raise an Exception to terminate the loop.
  - `run.py` top-level `main()` will call `job.process[0].on_error(e)` in case of unhandled exceptions; `DiffusionTrainer.on_error()` updates DB status to `error`, updates steps, flushes pending async tasks, and shuts down thread pool.
  - `BaseProcess.on_error` is present as a hook for other process types to implement custom logic.
- `done_hook()` is invoked at the end of a normal run (e.g., `BaseSDTrainProcess.done_hook()` and `DiffusionTrainer.done_hook()` which update DB state to `completed` and ensure async tasks flush).

---

## Third-party libs & how toggles are exposed 🧩
- `accelerate` (Hugging Face) — used for distributed training, AMP, gradient accumulation, and `Accelerator()` features. Enabled implicitly by using the `Accelerator()` wrapper and the `accelerate` runtime environment.
- `torch.cuda.amp` — mixed precision support is used implicitly via Accelerator and some local uses of `torch.autocast` in auxiliary modules. The dtype (`fp16`, `bf16`) is controlled by `TrainConfig.dtype` and converted to torch dtype via `get_torch_dtype()`.
- `xformers` — memory-efficient attention: toggled by `TrainConfig.xformers` and enabled via `enable_xformers_memory_efficient_attention()` on models.
- `bitsandbytes`, `dadaptation`, `prodigy` — optional/third-party optimizers supported by `toolkit.optimizer.get_optimizer()` when selected in `TrainConfig.optimizer`.

---

## Observability & saved artifacts 🕵️‍♀️
- Timers: `Timer` aggregates per-named-timer averages. Periodic `Timer.print()` calls (triggered every `performance_log_every` steps) produce a consolidated perf summary and feed hooks (UI DB updates).

- Performance tuning: New optional config under `performance` lets you enable `precise_gpu_timing` (opt-in GPU event timers) and `vram_diagnostics` (a lightweight VRAM snapshot printed alongside PERF SUMMARY). These are off by default; enable them for short diagnostic runs only.
- DB fields updated: `step` (via `update_step()`), `speed_string` (via `handle_timing_print_hook()`), `status` and `info` (via `update_status()`), `return_to_queue`/`stop` are polled.`logger` writes to a SQLite DB (`loss_log.db`) the per-step metrics.
- Checkpoints: saved under `save_root` with `.safetensors` files and metadata YAML files.
- Sample generation: periodic sampling runs at `sample_every` and is coordinated with save steps; the trainer will call `sample()` which uses `sd.sample_prompts_cache` when configured.

---

## Open questions & tests to add ❓🧪
- Tests suggested:
  1. **Gradient accumulation correctness**: unit test that simulates `gradient_accumulation` and `gradient_accumulation_steps` combos and verifies `optimizer.step()` happens the expected number of times and that gradient scaling/averaging matches expected results.
  2. **Multi-GPU / Accelerate integration tests**: run a minimal multi-process run (or mock Accelerator) to ensure `accelerator.accumulate()` and `accelerator.backward()` behave as expected; check `optimizer.step()` only occurs on correct ranks and is synchronized.
  3. **Save-before-stop behavior**: test `DiffusionTrainer.maybe_stop()` with `save_before_stop` True to ensure `save()` is called and that final DB `step` and `status` are updated.
  4. **Gradient clipping & NaN handling**: create a test that asserts gradients are clipped and that `optimizer.step()` is robust to NaNs after clipping.
  5. **EMA correctness**: test that `ema.update()` and `ema.copy_to()` produce expected averaged weights after multiple steps.
  6. **Precision toggle tests**: tests for `dtype=fp16` vs `bf16` ensuring no catastrophic loss in training or numerical instabilities; include small model smoke tests when using `xformers`.

---

## JSON manifest of files read and inspected ✅
```json
{
  "files": [
    { "path": "extensions_built_in/sd_trainer/SDTrainer.py", "ranges": [[1,220],[220,600],[1480,1610],[2420,3040],[4280,4360],[4360,4670]] },
    { "path": "jobs/process/BaseSDTrainProcess.py", "ranges": [[1,260],[2276,2336],[2760,2920],[2920,3240],[920,1020]] },
    { "path": "extensions_built_in/sd_trainer/DiffusionTrainer.py", "ranges": [[1,380]] },
    { "path": "toolkit/timer.py", "ranges": [[1,200]] },
    { "path": "toolkit/ema.py", "ranges": [[1,400]] },
    { "path": "toolkit/accelerator.py", "ranges": [[1,200]] },
    { "path": "toolkit/optimizer.py", "ranges": [[1,240]] },
    { "path": "toolkit/scheduler.py", "ranges": [[1,200]] },
    { "path": "toolkit/logging_aitk.py", "ranges": [[1,400]] },
    { "path": "jobs/process/BaseProcess.py", "ranges": [[1,240]] },
    { "path": "jobs/process/BaseTrainProcess.py", "ranges": [[1,220]] },
    { "path": "run.py", "ranges": [[1,160]] }
  ]
}
```

---

## Final notes / important code pointers 💡
- Backward and scaling are consistently routed through `Accelerator` (use of `self.accelerator.backward()` and `self.accelerator.accumulate()`), so configuration for distributed / AMP lives in `accelerate` runtime and `TrainConfig.dtype` and `TrainConfig.mixed_precision` flags.
- Timers are the central facility for performance observability and feed a UI hook (`handle_timing_print_hook`) to keep `speed_string` and DB state in sync.
- The code already contains a number of unit tests exercising timer recording and backward flows; the proposed tests above will strengthen correctness for accumulation and multi-GPU flows.

---

If you want, I can now (a) open a PR that adds the suggested unit tests, (b) write the standalone tests for gradient accumulation correctness and save-before-stop behavior, or (c) expand the chapter with code snippets and small diagrams. ✨
