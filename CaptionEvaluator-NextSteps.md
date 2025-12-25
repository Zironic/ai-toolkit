# Caption Evaluator — Next Steps & Integration Test Plan ✅

## Summary
A short, prioritized plan to finish dataset-loss evaluation and training-time per-example logging. This focuses on the immediate next steps: adding an integration test that verifies the streaming aggregation + per-save-step JSON behavior, documenting the plan, and preparing follow-up tasks.

---

## Goals (short term)
- Verify end-to-end behavior that we implemented so far: the streaming aggregator accumulates per-item stats and the trainer (save path) writes a memory-efficient JSON per save step.
- Provide a concise plan document describing actionable next steps and acceptance criteria.

---

## Tasks (this change)
1. Create an integration test that simulates a training save with a populated `StreamingAggregator` and asserts a per-save-step JSON file is written in `save_root` with the correct filename and per-item stats (average, min, max). ✅ (implemented)
2. Add the Markdown plan file (this file) into the repo. ✅
3. Run the tests and fix any issues surfaced.

---

## Acceptance criteria ✅
- Integration test passes (creates JSON file named `{job.name}_{step_zfilled}.json` and the file contains `datasets -> <dataset> -> item_stats -> <path> -> {average_loss,min_loss,max_loss}`).
- Existing unit tests for `StreamingAggregator` and `evaluate_dataset(..., out_json=...)` still pass.
- Integration test and unit tests have been executed in a dev environment and **passed** (3 tests: streaming aggregator, evaluate_dataset JSON, integration save JSON — all passed).

---

## Recent result (integration test)
- Status: **Passed** ✅ — focused test run executed: `testing/test_streaming_aggregator.py`, `testing/test_evaluate_dataset_json.py`, `testing/test_integration_save_json.py` — **3 passed, 0 failed**.
- Location: The integration test writes JSON to `save_root` and is validated by the test to ensure per-item `average_loss`, `min_loss`, `max_loss` are correct.

---

## Follow-ups (next priority)
- Implement `compute_per_example_loss` call and safe CPU-copy behavior inside `SDTrainer.calculate_loss` (capture unreduced per-sample losses reliably across accumulations). (High) — **Done**
- Add tests for `calculate_loss` per-sample behavior and numeric parity with scalar loss. (High) — **Done**
- Logging note: per-example logging is limited to simple runs (single batch & no accumulation) to avoid misleading partial logs. (Decision) — **Done**
- Integrate `dataset_summary` more fully into `BaseSDTrainProcess` logs and TensorBoard output. (Medium)
- Implement a dataset evaluator UI that runs evaluation and lets users export CSV/JSON. (Medium)

---

## Planned next work (detailed)
1. **Backend API & background job runner** (3–5h) — *in progress*
   - Implement POST `/api/eval_dataset` accepting {model, dataset, sample_fraction, max_samples}
   - Start an async background job that runs `run_dataset_evaluation(...)` and writes the streaming JSON report to the job save directory
   - Add GET `/api/eval_status/{job_id}` and GET `/api/eval_result/{job_id}` returning job state and result path/content
   - Tests: unit tests for API validation and a mocked run that verifies JSON write and endpoint flow

2. **Frontend UI (Dataset view)** (3–5h)
   - Add an `Evaluate dataset` panel: model selector, sample_fraction, max_samples, run button
   - Show job status, progress (stdout/log tail), and links to JSON/CSV. Display top flagged captions inline
   - Tests: basic UI tests and manual verification

3. **Integration & QA** (1–2h)
   - End-to-end test: trigger evaluation via API, wait for completion, fetch results, assert JSON schema and flagged captions
   - Update docs with usage examples and developer notes

These tasks are prioritized: Backend API → Frontend UI → Integration tests & docs.


---

## Testing notes
- Integration tests should avoid heavy dependencies and instantiate `BaseSDTrainProcess` via `__new__` while manually setting the minimal attributes needed to exercise the `save()` → JSON write path.
- Keep the integration test deterministic (use tmpdirs for `save_root` and fixed input values).

---

## Estimated time
- Integration test + plan doc + test run & fix: ~1–2 hours
- Follow-up implementation (compute_per_example_loss): ~3–6 hours

---

If you'd like, I can now proceed with the next follow-ups: implementing `compute_per_example_loss` in `SDTrainer.calculate_loss` and adding corresponding unit tests.