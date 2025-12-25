# Detailed Plan — Extract Shared Loss Utilities

Status: IN-PROGRESS (Task: Extract shared loss computation & aggregation utilities)

Created: 2025-12-25

---

## Objective
Extract a small, well-tested utility module that contains shared functionality for per-example loss aggregation and caption quality detection. This will be used both by training-time logging (lightweight) and the offline dataset estimator (thorough), ensuring parity and reducing code duplication.

---

## Deliverables
- New file: `toolkit/util/loss_utils.py` (already created with initial helpers: aggregate_by_dataset, flag_bad_captions, evaluate_dataset)
- Unit tests: `testing/test_caption_evaluator.py` (already added)
- Integration items: instructions & TODOs for wiring `compute_per_example_loss` (implementation depends on model specifics), hooks to call utilities from trainer and dataset evaluator
- Acceptance: utilities pass unit tests; integration points are documented and safe for training-time use

---

## Scope & Decisions
- The utilities in `loss_utils` will *not* compute model-dependent raw losses by themselves. Instead they accept per-example records (path, dataset, caption, loss) or a `compute_fn` that the model owner provides.
- This avoids coupling model internals into the generic utils and ensures the training code (SDTrainer) controls the exact computation of per-example raw losses.
- `compute_per_example_loss`: design/implementation is done in trainer or evaluator code and will call `loss_utils` for aggregation and flagging.

---

## Implementation Steps (detailed)
1. Verify `loss_utils.py` file exists and contains helpers (aggregate_by_dataset, flag_bad_captions, evaluate_dataset).
   - Status: completed (initial version created).
2. Add robust unit tests that cover:
   - Aggregation correctness (means / medians / top-k ordering)
   - Caption flagging heuristics for empty, short, long, repeated, suspicious tokens
   - CSV export behavior in evaluate_dataset if out_csv provided
   - Tests exist in `testing/test_caption_evaluator.py` and should be executed.
   - Status: initial tests added.
3. Define the exact `compute_per_example_loss` contract in docs and add a convenience wrapper placeholder in `loss_utils` that callers can use if convenient.
   - Example: `compute_per_example_loss(sd_model, batch, train_config, mode='train') -> list[ per_sample_dict ]`
   - This will be implemented in `SDTrainer` (and dataset evaluator) since SD model has specifics (scheduler, flow mode, etc.).
4. Update `SDTrainer.calculate_loss` to use the contract (in a follow-up task): call per-example computation, detach CPU copies, and calls `aggregate_by_dataset` and `flag_bad_captions` as needed. Also ensure an option to write a JSON report to `save_root` that records, for each dataset item, the average, min, and max loss observed during the run or evaluation; for large datasets implement streaming or incremental writes to avoid OOM. When running during training, write one JSON per save step named using the job name and zero-padded step count (e.g. `{job.name}_000000123.json`).
5. Add logging integration and config gating (TrainConfig) in follow-up tasks.

---

## Test Plan
- `pytest testing/test_caption_evaluator.py` must pass.
- Additional tests to add later: parity test ensuring `compute_per_example_loss(...).losses.mean()` matches current scalar loss used in training.

---

## Acceptance Criteria
- Unit tests pass locally and in CI.
- Utilities are stable and documented.
- Minimal/no performance impact when called for sampling or small outputs in training mode.

---

## Timeline / Estimates
- Finish any missing unit tests and run them: 0.5–1h
- Add `compute_per_example_loss` convenience wrapper doc and examples for trainer implementors: 0.5–1h
- Prepare PR with file + tests + docs: 0.5–1h

Total for this extract step: ~2h (as planned)

---

## Risks & Mitigations
- Risk: Trainee code might implement per-example loss inconsistently causing mismatch between training and dataset estimates.
  - Mitigation: Provide a clear contract in the docs and add a numerical parity test in Step 8.
- Risk: Additional CPU copy may cause overhead.
  - Mitigation: Copy only per-sample scalars and optional components; sample outputs during training (config gating).

---

## Next Actions (today)
1. Run `pytest testing/test_caption_evaluator.py` locally and fix any failing tests.
2. Add parity, CSV export and JSON export tests (verify JSON contains highest/lowest/average per dataset item) if missing.
3. Prepare a small PR with `toolkit/util/loss_utils.py`, its tests, and `CaptionEvaluator-ExtractUtilitiesPlan.md`.

---

> Plan file (this document) will be used as the reference checklist while I complete the extraction task and associated tests.
