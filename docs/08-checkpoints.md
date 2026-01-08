# Chapter 08 — Checkpoint saving & metadata ✅

## TL;DR ⚡
- Checkpoints are written periodically during training (every `SaveConfig.save_every` steps), at the end of training, and whenever `save()` is invoked programmatically.
- Supported formats: **`safetensors`** (single-file) and **`diffusers`** (directory with model files + `aitk_meta.yaml`). Embeddings may also be saved as `.pt` (legacy).
- Metadata is stored inside safetensors file metadata (stringified JSON) or as `aitk_meta.yaml` inside diffusers-style folders. Training meta includes `training_info` with `step` and `epoch` and other `ss_*` keys; model hashes are added (`sshs_model_hash`, `sshs_legacy_hash`).
- EMA weights are applied to the model before saving by calling `ema.eval()` (which copies EMA shadow weights into model params) and restored after saving by calling `ema.train()`.
- The trainer keeps only the newest N step-saves (controlled by `SaveConfig.max_step_saves_to_keep`), cleaned up by `BaseSDTrainProcess.clean_up_saves()`.
- Push-to-hub uses Hugging Face APIs (via `HfApi`); README is created and `upload_folder` is called on `save_root`. Note: upload ignores `*.yaml` and `*.pt` by default.

---

## Files & symbols referenced 📂🔧
- `toolkit/config_modules.py` — type and defaults: `SaveConfig` (fields: `save_every`, `dtype`, `max_step_saves_to_keep`, `save_format`, `push_to_hub`, `hf_repo_id`, `hf_private`) (lines ~1–60).
- `jobs/process/BaseSDTrainProcess.py` — core save & lifecycle logic:
  - `BaseSDTrainProcess.save(step=None)` — checkpoint assembly and write; handles networks, adapters, embeddings, optimizer, JSON dataset reports, EMA calls (lines ~960–1270; save invocation points around lines ~2896–3200).
  - `BaseSDTrainProcess.get_latest_save_path(...)` — finds latest checkpoint (by file/directory creation time) (lines ~1220–1416).
  - `BaseSDTrainProcess.clean_up_saves()` — enforces `max_step_saves_to_keep` and removes older saves (lines ~840–1020).
  - `BaseSDTrainProcess.load_training_state_from_metadata(path)` — parses metadata (folder `aitk_meta.yaml` or safetensor metadata) and restores `step`/`epoch` (lines ~1380–1460).
  - `BaseSDTrainProcess.push_to_hub(repo_id, private)` — create repo + upload folder; uses `HfApi.upload_folder` and writes a `README.md` (lines ~3200–3600).
- `extensions_built_in/sd_trainer/SDTrainer.py` & `extensions_built_in/sd_trainer/DiffusionTrainer.py` — sample & save orchestrations, sample naming patterns used by README (samples folder) (e.g., cache and sample helpers around lines ~292–336 and sample wrapper in `DiffusionTrainer.save`) (lines referenced above).
- `toolkit/saving.py` — helper functions for converting diffusers state dict to LDM keys and saving LDM safetensors; helpers used when saving `sd.save(...)` (lines ~1–300+).
- `toolkit/metadata.py` — metadata helpers:
  - `get_meta_for_safetensors(meta, name)` — converts OrderedDict metadata into safetensors-compatible metadata and injects `software`/`format` and JSON-stringifies non-primitive values (lines ~1–80).
  - `add_model_hash_to_meta(state_dict, meta)` — computes safetensors bytes and adds `sshs_model_hash` and `sshs_legacy_hash` (lines ~20–64).
  - `load_metadata_from_safetensors(file_path)` and `parse_metadata_from_safetensors` (lines ~64–110).
- `toolkit/ema.py` — `ExponentialMovingAverage` class and semantics (copy_to/store/restore/average_parameters/eval()/train()) used to put EMA weights into the model before saving and restore them after (lines ~1–240).
- `toolkit/network_mixins.py` — `Network.save_weights` and `Network.get_state_dict` used for saving LoRA and related networks; it calls `add_model_hash_to_meta` and uses `safetensors`/`torch.save` as appropriate (lines ~500–640).
- Tests: `testing/test_integration_save_json.py` — ensures JSON-per-save reports are emitted and validates content (lines ~1–120).

---

## How and when checkpoints are saved (details) 🕒
- Periodic saves:
  - Controlled by `SaveConfig.save_every` (default 1000). In the training loop, a save step is detected when `step % save_every == 0` and `BaseSDTrainProcess.save(step)` is called (see training loop around where `is_save_step` is computed).
- On end-of-training:
  - After the main training loop exits the trainer calls `self.save()` one final time (unconditional final checkpoint).
- Manual/explicit saves:
  - Any code may call `save(step)` to produce a saved checkpoint at the given step.
- On-error / OOM behavior:
  - OOM during a training step is caught and the step is skipped (up to 3 consecutive OOMs) — this path does not trigger a checkpoint write by default. There is no global "save on unexpected exception" hook inside the main loop; users should wrap high-level job controllers if they require crash-dump checkpoints.

File contents written during save:
- Primary model/LoRA/adapter/embedding weights (format depends on component and `SaveConfig.save_format`).
- `optimizer.pt` (if `self.optimizer` exists).
- If `self.snr_gos` exists, `learnable_snr.json` is saved with learnable SNR params.
- Optional per-save dataset JSON (if `train_config.save_loss_json` True): `{job_name}_{step}.json` containing streamed dataset stats.
- Samples (images) are saved to `save_root/samples/` and the README expects names like `<timestamp>__{zero_padded_steps}_{index}.jpg` (used to display representative samples in the README). See README generation logic.

---

## File formats supported & behaviors 🗃️
- `safetensors` (single-file) — default mode. The checkpoint file is e.g. `job_name_000000100.safetensors`.
  - Metadata is embedded using `safetensors` metadata and prepared by `toolkit.metadata.get_meta_for_safetensors()` (stringifies nested values and inserts `software` info and `format: 'pt'`).
  - `toolkit.network_mixins.save_weights()` adds precomputed model hash metadata (`sshs_model_hash`, `sshs_legacy_hash`) before calling `safetensors.torch.save_file`.
- `diffusers` (directory) — when `SaveConfig.save_format` == `'diffusers'` the trainer saves Diffusers pipeline format:
  - The trainer passes a directory path (`<save_root>/<jobname>_{step}` style) and `sd.save()` creates a `save_pretrained` layout and writes `aitk_meta.yaml` (using `yaml.dump`) into the directory root.
  - For ControlNet adapters, the adapter is saved as a directory and `aitk_meta.yaml` is written next to the adapter files.
- `.pt` legacy — some embeddings or optimizer state saved as `.pt` (e.g., `optimizer.pt` and embedding `.pt` when `embed_config.save_format == 'pt'`).

Naming conventions:
- Model / step-saves: `{job.name}_{step_zero_padded_9}.safetensors` (step omitted for non-step save; final save may be `jobname.safetensors` if step not supplied).
- LoRA network saves: `<job.name>_LoRA_{step}.safetensors` or similar (the code uses `lora_name = self.job.name` and appends `_LoRA` when `named_lora` is True).
- Adapter saves: suffixes appended to `job.name`: `_{_t2i, _cn, _clip, _ip, _adapter}` + step.
- Embeddings: filename is `embed_config.trigger_{step}.safetensors` or `.pt` if configured.
- Optimizer: `optimizer.pt` (no step suffix).
- JSON per-save report: `{job.name}_{step_zero_padded_9}.json`.
- ControlNet/diffusers directory saves include `aitk_meta.yaml` (human-friendly YAML metadata).

---

## EMA weights (how it's used on save) 🧮
- If an `ExponentialMovingAverage` instance (`self.ema`) exists: `save()` calls `self.ema.eval()` prior to writing checkpoint.
  - `ema.eval()` performs `store()` then `copy_to()` (swap current params for EMA shadow params) so the saved file contains EMA-smoothed weights.
  - After the save completes, `save()` calls `self.ema.train()` to restore the original training parameters.
- This behavior ensures checkpoints contain EMA weights for inference while retaining training parameters in memory.

---

## Metadata structure & discovery 🧾
- Safetensors-embedded metadata
  - Generated via `get_meta_for_safetensors()` which: injects `software` version info, adds `format: 'pt'`, and stringifies non-primitive metadata values.
  - `add_model_hash_to_meta()` appends `sshs_model_hash` and `sshs_legacy_hash` (for add-on network indexing compat).
  - `toolkit.metadata.load_metadata_from_safetensors()` and `parse_metadata_from_safetensors()` are used to read and decode saved metadata.
- Diffusers-style metadata
  - When saving as a directory, a YAML file `aitk_meta.yaml` is written to the root of the saved folder (using `yaml.dump(self.meta)`).
  - `get_latest_save_path()` expects either `.safetensors` files, `.pt`, or directories; it will prefer the most recent by creation time.
- Finding the "latest" checkpoint
  - `BaseSDTrainProcess.get_latest_save_path(name=None, post='')` collects paths matching `f"{name}*{post}.safetensors"`, `*.pt`, and directories, filters false positives (e.g., `_LoRA`, `_refiner` based on flags) and returns the newest by `os.path.getctime()`.
- How restore uses metadata
  - `load_training_state_from_metadata(path)` reads the metadata (either `aitk_meta.yaml` in a folder or safetensors metadata) and checks for `training_info.step` and `training_info.epoch` to set `self.step_num`, `self.epoch_num`, and `self.start_step` so training can resume from the correct step.

---

## Push-to-Hub behavior (Hugging Face) 🚀
- At the end of training, if `save_config.push_to_hub` is True, the trainer:
  - Ensures `HF_TOKEN` (or runs `interpreter_login` if missing) and calls `push_to_hub(repo_id, private)`.
  - `push_to_hub()` writes a README (with a small auto-detected gallery section based on images in `samples/`) and calls `HfApi.create_repo()` then `HfApi.upload_folder(repo_id, folder_path=self.save_root, ignore_patterns=["*.yaml","*.pt"], repo_type="model")`.
- Notes & caveats:
  - `upload_folder` ignores `*.yaml` and `*.pt` by default (explicit in the call). The saved diffusers `aitk_meta.yaml` may therefore be excluded by default — the code writes the readme and the model files but excludes some artifact types.
  - If `HF_TOKEN` is absent, the code attempts an interactive login via `interpreter_login(new_session=False, write_permission=True)`.

---

## Tests & observability ✅🔍
- Existing tests:
  - `testing/test_integration_save_json.py` validates that calling `save(step)` when `train_config.save_loss_json` is true writes a per-step JSON with dataset statistics and correct file name (asserts content and aggregate stats).
- Practical checks you can add / run to validate correctness:
  - Verify the safetensors file can be opened and `load_metadata_from_safetensors` returns `training_info.step == saved_step`.
  - Assert existence of keys expected by consumers (e.g., `ss_output_name`, `training_info`, `sshs_model_hash`). Use `toolkit.metadata.parse_metadata_from_safetensors` to get typed values.
  - Confirm EMA semantics by saving with `ema` present and checking that weights in the saved file match `ema.shadow_params` (or by using a short `ema.decay` to make observable differences).
  - Validate push-to-hub flow with a mock or a test repo — ensure `upload_folder` is called and that unusual files are excluded by the `ignore_patterns`.
  - Test `clean_up_saves()` by creating > `max_step_saves_to_keep` checkpoint files and asserting older ones are deleted.

---

## Recommendations & gotchas 💡
- There is no general automatic "on-exception" checkpoint dump — if you want crash-dump behavior, wrap the job runner or add a top-level exception handler that calls `save()`.
- The HF uploader ignores `*.yaml` and `*.pt`; if you rely on `aitk_meta.yaml` being pushed, adjust `ignore_patterns` or copy metadata into pushed files explicitly.
- `SaveConfig.merge_network_on_save` exists in config but no global merge-on-save behavior is present in `save()` (keep an eye on model-specific `save` hooks which may merge scalers or other data before saving).

---

## JSON manifest of files and line ranges (source-of-truth) 🧾
Below is the manifest of files and line ranges I used as the source-of-truth for this chapter (paths relative to repo root):

```json
{
  "toolkit/config_modules.py": [1, 260],
  "toolkit/metadata.py": [1, 120],
  "toolkit/saving.py": [1, 400],
  "toolkit/ema.py": [1, 260],
  "toolkit/network_mixins.py": [480, 640],
  "toolkit/stable_diffusion_model.py": [3540, 3620],
  "jobs/process/BaseSDTrainProcess.py": [800, 1320],
  "jobs/process/BaseSDTrainProcess.py": [1380, 1460],
  "jobs/process/BaseSDTrainProcess.py": [2780, 2875],
  "jobs/process/BaseSDTrainProcess.py": [2896, 3200],
  "jobs/process/BaseSDTrainProcess.py": [3200, 3600],
  "extensions_built_in/sd_trainer/SDTrainer.py": [292, 336],
  "extensions_built_in/sd_trainer/DiffusionTrainer.py": [300, 460],
  "testing/test_integration_save_json.py": [1, 140]
}
```

---

If you'd like, I can also:
- Add a small checklist of tests to add (unit tests for metadata keys, safetensors metadata round-trips, EMA save semantics, cleanup policy), or
- Generate small unit tests to verify `clean_up_saves()` and `load_training_state_from_metadata()` behaviors.

Would you like me to add tests and/or a short example snippet demonstrating how to resume training from the latest checkpoint? 🔧
