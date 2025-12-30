# ControlTrain Implementation Status & Plan

This file captures the step-by-step implementation plan (derived from `ControlTrain-Design.md` and `ControlTrain-Design2.md`). Mark progress as you complete each step.

## Phase 1  High-priority tasks (short-term)
1. Add `toolkit/pose.py::make_openpose_map(...)` and unit tests (`testing/test_make_openpose_map.py`)  deterministic generation and metadata.
2. Add `Input tensor contract` enforcement in `toolkit/dataloader_mixins.py`  assert shapes/ranges, normalize uint8 -> float32.
3. Implement `Multi-control fusion` helpers (`compose_union_controls`) and update pipeline wrapper to accept per-control scales (support scalar or per-control list).
4. Checkpoint & inference rules: document and add detection for `Z-Image-Turbo-Fun-Controlnet-Union-2.1*.safetensors` and add recommended defaults (num_inference_steps=8, control_context_scale0.650.90).

## Phase 2  Medium priority
5. Implement `toolkit/controlnet_compat.py` (safetensors loader, resampler, channel normalizer), record converter metadata on conversion.
6. Add `tools/gen_control.py` CLI + manifest emission and `tools/apply_control_manifest.py` tool for applying manifests atomically.
7. Add example `examples/predict_t2i_control_2.1.py` and unit tests `testing/test_controlnet_union.py` (shapes, forward, 8-step smoke). Completed (minimal example/tests added).

## Phase 3  Operational & Tests
8. Add UI fields + `/api/control_gen` endpoints + worker integration; add preview API to the UI.
9. Add manifest validator `toolkit/control_manifest.py::validate_manifest` and tests for idempotence and augmentation-replay enforcement.
10. Add DB migration to persist `control_manifest_path` and params (see `ui/db/migrations/20251230_add_control_manifest.sql` suggestion).

## Testing & Acceptance
- Add unit & integration tests listed in `ControlTrain-Design.md` under the "Tests & verification" section.
- Prioritize fast test coverage (shape/contract tests) and add manual GPU tests for offload/residual behaviors.

## Notes
- For any conversion or tolerant load, record converter metadata (converter_version, source_filename) to ensure reproducibility.
- Use `strict=False` for tolerant safetensors loads and surface explicit errors when data is incompatible.

