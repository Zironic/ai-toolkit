PR Notes: ControlNet LoRA training integration (VideoX-Fun mirror)

Summary of changes
- Implemented ControlNet transformer loader with VideoX-Fun pattern: base→control copy, safetensors streaming assignment, unexpected-key checks, dry-run forward.
- Added VAE-tiling helpers and tiled encoding/reassembly for control images.
- Integrated tiled-control encoding into training forward paths (uses mocked flows in tests).
- Implemented offload abstraction integration points and tests (manual_swap/cpu paths).
- Added unit tests:
  - `testing/test_controlnet_unexpected_keys.py`
  - `testing/test_controlnet_streaming_fixture.py` (creates small safetensors at test time)
  - `testing/test_training_forward.py` (mocked training forward using control images)
  - `testing/test_model_control_tiling.py`, `testing/test_custom_adapter_control_tiling.py`
  - `testing/test_trainer_assistant_adapter_loading.py` (mocked assistant adapter loading)
- Added runbook: `docs/runbooks/controlnet_runbook.md` (resource guidance, fail-fast messages, recovery steps)

Reviewer Checklist
- Run `pytest testing -q` locally (most tests are unit/mocked; avoid GPU-heavy). Ensure new tests pass.
- Inspect `extensions_built_in/diffusion_models/z_image/z_image.py::load_controlnet_transformer` for fail-fast behavior around unexpected keys and dry-run forward.
- Confirm test coverage for streaming vs non-streaming paths (`testing/test_controlnet_streaming_fixture.py`).
- Ensure runbook is clear and adequate for operators; suggest edits.

Notes
- All new tests avoid requiring large model downloads or GPU access; safetensors used in tests are created at runtime.
- Offload manager integration points are implemented; final `memory_manager` path requires environment-specific verification prior to merging to main.

Next steps
- Finalize offload manager tests for `memory_manager` when MemoryManager is available in CI.
- Open PR and request reviews from maintainers familiar with model-loading and CI infra.
