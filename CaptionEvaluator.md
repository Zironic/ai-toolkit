# Caption Evaluator — Plan & Spec ✅

## Purpose
Provide a reusable, testable system to compute per-example losses, aggregate per-dataset statistics, and detect problem captions (those that cause high loss or otherwise harm training). This supports both lightweight training-time logging and an offline dataset loss estimator feature in the Dataset view.

---

## Goals 🎯
- Capture per-example losses (GPU → CPU safe) during training without changing current training semantics.
- Aggregate per-dataset statistics (mean/median/worst-N) and make them available to training logs and TensorBoard.
- Detect and report problematic captions via heuristics (empty/short/long/repeated tokens, suspicious tokens, trigger mismatches) correlated with high loss.
- Provide a dataset loss estimator that runs offline (eval mode) to compute dataset loss summaries and export CSVs.
- Reuse the same, well-tested utilities for both training-time logging and the dataset evaluator.

---

## Scope & Deliverables 📦
- Utility module: `toolkit/util/loss_utils.py` with these functions:
  - `compute_per_example_loss(...)`
  - `aggregate_by_dataset(...)`
  - `flag_bad_captions(...)`
  - `evaluate_dataset(...)` (convenience)
- Training integration: minor refactor in `SDTrainer.calculate_loss` and `hook_train_loop` to call utilities and attach `dataset_summary`/`per_example` to `loss_dict` (gated behind config).
- Logger integration: `BaseSDTrainProcess` enhancements to log dataset summaries to progress bar, `self.logger`, and TensorBoard (option to write CSV to `save_root`).
- UI: Dataset view hook to run dataset evaluation (model selector, show mean & worst-N, export CSV) — implementation detail belongs to Task 13.
- Tests & docs: Unit tests, integration tests, performance benchmarks, README and usage docs.

---

## Tasks (finalized)
1. Design per-example loss capture API (spec + pseudocode) — 1.5h — High ✅ **Done**
2. Extract shared loss utilities (`loss_utils.py`) — 2h — High
3. Implement per-example capture in `calculate_loss` (use helper) — 3.5h — High
4. Aggregate in `hook_train_loop` and attach to `loss_dict` — 2h — High
5. Implement `flag_bad_captions` & reporting (top-N offenders) — 3h — Med-High
6. Add config flags & limits to `TrainConfig` (`log_per_example`, `max_examples_print`, `log_per_dataset`, `flag_bad_captions`) — 1.5h — Med
7. Integrate logger & TensorBoard changes in `BaseSDTrainProcess` — 2h — Med
8. Tests & CI checks (unit + integration) — 3h — High
9. Performance & memory review; add sampling fallback if needed — 2.5h — Med
10. Docs & example outputs — 1.5h — Low-Med
11. Manual test run & sign-off — 2.5h — High
12. Dataset loss estimation & Dataset view integration — 6h — Med-High

Total: ~24 hours (sum of estimates)

---

## API Spec — toolkit.util.loss_utils

### compute_per_example_loss(sd, batch, train_config, mode='train', return_components=False, device=None) -> dict
- Purpose: Compute per-sample scalar losses before mean (and optional component losses) in a batch.
- Returns dict:
  - `losses`: Tensor shape (B,) on CPU (float)
  - `raw_loss`: optional unreduced loss tensor (B,...)
  - `components`: optional map of tensors per-sample (e.g., prior_loss, dfe_loss)
- Notes:
  - Always detach and CPU-copy the per-sample outputs.
  - For video (5D), compute mean across appropriate spatial and temporal dims to produce a scalar per-sample.
  - `mode='eval'` disables training-time randomness and sampling.

### aggregate_by_dataset(per_example_list, key='dataset') -> dict
- Purpose: Group per-sample records and compute summary stats.
- Input `per_example_list` items: {path, dataset, caption, loss, components...}
- Returns:
  - `dataset_summary`: {dataset_name: {mean, median, count, top_k_samples}}
  - `global`: {mean, median, count}

### flag_bad_captions(per_example_list, heuristics=None, top_n=20) -> dict
- Purpose: Heuristically detect problematic captions correlated with high loss.
- Default heuristics:
  - Empty or very short (< 3 tokens)
  - Extremely long (> 200 tokens)
  - Repeated token sequences or repeated punctuation
  - Suspicious tokens like '###' or '<unk>'
  - Trigger mismatches (dataset-trigger expected but absent/present as appropriate)
- Returns flagged list with reasons and scores.

### evaluate_dataset(sd, dataloader, train_config, sample_fraction=1.0, max_samples=None, out_csv=None) -> dict
- Purpose: Convenience wrapper to run `compute_per_example_loss` in eval mode across a dataset, aggregate, flag captions, and optionally write a CSV.

---

## Integration pseudocode (examples)

- In `SDTrainer.calculate_loss`:
```py
per = loss_utils.compute_per_example_loss(self.sd, batch, self.train_config, mode='train', return_components=True)
self.last_example_losses = per['losses']  # CPU tensor
loss_scalar = per['losses'].mean().to(device_of_backward)
# Backward using loss_scalar exactly as today
```

- In `hook_train_loop` (collect & attach):
```py
for idx, file_item in enumerate(batch.file_items):
    per_example_list.append({
        'path': file_item.path,
        'dataset': file_item.dataset_config.dataset_path or file_item.dataset_config.folder_path,
        'caption': file_item.raw_caption,
        'loss': float(self.last_example_losses[idx]),
    })
summary = loss_utils.aggregate_by_dataset(per_example_list)
loss_dict['dataset_summary'] = summary['dataset_summary']
loss_dict['per_example'] = per_example_list_sampled  # respect config limits
```

- Dataset evaluator:
```py
results = loss_utils.evaluate_dataset(sd, dataloader, train_config, sample_fraction=0.1, out_csv='report.csv')
# UI reads CSV / results and displays mean and top-k worst samples
```

---

## Config suggestions (TrainConfig)

```yaml
log_per_example: false
max_examples_print: 50
log_per_dataset: true
flag_bad_captions: true
dataset_eval_sample_fraction: 0.1
```

- Defaults are conservative: per-example logging disabled; dataset summaries enabled.

---

## Tests & Acceptance Criteria ✅
- Numerical parity: `compute_per_example_loss(...).losses.mean()` matches the scalar loss used in current training (within floating tolerance).
- Aggregation correctness: `aggregate_by_dataset` calculates means, medians, and top-k correctly.
- Caption detection: unit tests flag synthetic bad captions.
- Logging gating: enabling/disabling config flags shapes output as expected.
- Dataset evaluator: CSV output format is stable and includes path, dataset, caption, loss.

---

## Performance considerations ⚡
- Keep per-sample tensors on CPU (detached) to avoid persistent GPU memory growth.
- For training logging, sample or cap printed/stored entries (e.g., `max_examples_print`) to bound memory and I/O.
- For dataset evaluation, allow thorough runs with sampling and batching.

---

## Next step
- Implement `toolkit/util/loss_utils.py` (extract helpers) and add unit tests for `compute_per_example_loss` and `aggregate_by_dataset` — **Estimate: 2h**.

---

If you want, I can start the implementation now and add the unit tests (I recommend doing this next so we can iterate quickly).