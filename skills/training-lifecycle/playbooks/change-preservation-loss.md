# Playbook: Change preservation_loss

When to use
- User asks to modify preservation loss behavior, weighting, failure handling, or diagnostics.

Priority files to open (in order)
1. `extensions_built_in/sd_trainer/SDTrainer.py` — search for `_compute_and_apply_preservation_loss` and its callers. (primary implementation and backward application)
2. `toolkit/train_tools.py` — training loop helpers and loss aggregation utilities.
3. `toolkit/accelerator.py` — dtype/amp interactions that affect numerical behavior.
4. `docs/training_lifecycle/07-training-loop.md` — reading for recommended practices and logging guidance.
5. `testing/` — search for existing tests that exercise preservation behavior or loss logging.

Symbols & queries
- grep for: `preservation_loss`, `_compute_and_apply_preservation_loss`, `preservation_pred`, `prior_pred`, `multiplier`
- Example ripgrep query: `rg "preservation_loss|_compute_and_apply_preservation_loss" -n`

Test & validation checklist
- Add unit tests that compute preservation loss with small synthetic tensors and assert expected numeric behavior (no NaNs, shape matches).
- Add an integration smoke job: one epoch, check that `_last_preservation_loss` is set and logged.
- If changing gradient application, run a backward/optimization step in a small test to ensure parameter updates occur and no runtime errors.

Troubleshooting notes
- If preservation loss returns None, the code currently falls back to normal loss — inspect exception handling (SDTrainer logs warnings).
- Mixed precision (AMP) may change numeric stability; ensure tests cover float16 paths if relevant.

PR guidance
- Add unit tests and one small smoke job commit. In PR description reference `docs/training_lifecycle/07-training-loop.md` and include benchmark logs for before/after on a short run.

Example agent workflow (short)
1. Open playbook and run the ripgrep queries to find exact lines. 2. Edit implementation in `SDTrainer.py` and add unit tests in `testing/`. 3. Run tests and small smoke job locally (note: GPU runs are manual). 4. Run `scripts/validate_skill.py` and update CODE_MAP if new symbols were added.
