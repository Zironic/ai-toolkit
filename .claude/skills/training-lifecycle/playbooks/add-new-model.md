# Playbook: Add a new model

When to use
- User asks to add or support a new model architecture or a new pre-trained model release.

Priority files to open (in order)
1. `toolkit/model_utils.py` — model loading helpers and compatibility wrappers (search for `load_pipeline|load_model|StableDiffusion`).
   - Quick checks: `rg "StableDiffusion|load_pipeline|load_model" -n toolkit/`
   - Tests: `python -m pytest testing/test_pipeline_symbols_exist.py -q`
2. `toolkit/config_modules.py` — model defaults and config mapping; add model defaults and config docs here.
3. `jobs/*` and `jobs/process/*` — any job types that will use the model; add smoke job configs and tests under `testing/smoke_jobs/`.
4. `docs/training_lifecycle/05-pipeline-and-jobs.md` — chapter guidance about pipeline loaders; update chapter with notes for special model-handling requirements.
5. `ui/src/docs.tsx` and `ui` job forms — update docs and UI if model choices are surfaced to users; include UI examples in PR.

Symbols & queries
- grep for: `load_pipeline`, `StableDiffusion`, `load_model`, `MODEL_ZOO`

Implementation notes
- Add model config defaults into `toolkit/config_modules.py` and add loader support in `toolkit/model_utils.py`.
- Add a small smoke job that loads the model and runs a single sample to ensure end-to-end loading works.
- Validate memory and dtype compatibility and add notes to `docs/training_lifecycle/05-pipeline-and-jobs.md` when special handling is required.

Tests & CI
- Unit test for loader that mocks model artifacts (or loads a tiny model stub) and verifies API usage.
- Integration smoke job for sample generation to ensure loader + pipeline works (manual/GPU required).

Agent workflow (short)
1. Read this playbook and the `05-pipeline-and-jobs.md` chapter.
2. Add loader & config changes and tests. 3. Update docs and CODE_MAP. 4. Run validator and open PR.
