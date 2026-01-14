# Playbook: Add a LoRA variant

When to use
- User requests support for a new LoRA variant (different parametrization, checkpoint format, or training regime).

Priority files to open (in order)
1. `extensions_built_in/sd_trainer/SDTrainer.py` — LoRA application logic, LoRA bypass, and preservation interactions.
2. `ui/src/app/jobs/new/SimpleJob.tsx` — job config UI where network type is selected.
3. `toolkit/model_utils.py` — functions used to apply/load LoRAs into models.
4. `ui/src/docs.tsx` — documentation and UI-visible notes about LoRA behavior.
5. `scripts/` — any LoRA-related preprocessing or export scripts.

Symbols & queries
- grep for: `lora`, `LoRA`, `assistant_lora_path`, `apply_lora`, `load_lora`

Implementation notes
- Add a new network `type` option in the job config and the UI form if necessary.
- Ensure the model loader accepts the new LoRA format and conversion helpers if needed.
- Add a small smoke job that exercises training with the new LoRA variant and verifies that outputs (LoRA file, sample images) are created.

Tests & CI
- Unit tests for model_utils conversion and loader.
- Integration smoke job in testing folder demonstrating LoRA train-save-load cycle.

PR guidance
- Update `ui` docs with new option and examples. Add end-to-end smoke configs and link to `docs/training_lifecycle/05-pipeline-and-jobs.md`.

Agent workflow (short)
1. Read this playbook and run grep queries to find exact code.
2. Add network handling in config parsing and apply code in `SDTrainer`. 3. Add unit tests and a smoke job. 4. Update docs and the code map.
