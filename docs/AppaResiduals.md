# APPA Residuals Mode — Implementation Plan

## Purpose
Implement the APPA *residuals* mode: a minimal, low-risk adapter that transforms ControlNet per-block residuals before the UNet consumes them, improving the spatial signal available to subject LoRA training.

This document is a focused, actionable plan covering code changes, tests, diagnostics, acceptance criteria, and a recommended rollout schedule.

---

## Background & Motivation
- Current flow: ControlNet computes `down_block_res_samples` and `mid_block_res_sample` → SDTrainer sets `pred_kwargs['down_intrablock_additional_residuals']` → UNet consumes residuals; LoRA is trained only on the UNet.
- Goal: Insert a small trainable APPA residual transformer that adapts residuals to be more compatible with UNet features and with LoRA learning, without modifying ControlNet or UNet internals.

---

## High-level design
1. Add `AppearancePoseAdapter` class with method `transform_residuals(down_res, mid_res, timesteps=None, batch_images=None)`.
2. Add a config toggle: `train_config.appearance_pose_adapter.enabled` and `train_config.appearance_pose_adapter.mode='residuals'` (default false).
3. Integrate into `SDTrainer.train_single_accumulation` immediately after `down_block_additional_residuals` is computed and before assigning `pred_kwargs['down_intrablock_additional_residuals']`.
4. Ensure offload/bring helpers and `controlnet_appa_forward` timer are used for performance and safety.
5. Add unit tests, smoke integration tests, and a short eval harness.

---

## File & API changes (concrete)

### New file
- `toolkit/appearance_pose_adapt.py`
  - class `AppearancePoseAdapter(nn.Module)`
    - __init__(config)
    - `transform_residuals(self, down_block_res_samples, mid_block_res_sample, timesteps=None, batch_images=None)`
    - `state_dict()/load_state_dict()` (standard)
  - Implementation detail (minimal prototype):
    - For each residual tensor r_i ([B, C_i, H_i, W_i]):
      - Optionally project appearance features `za` to H_i×W_i via small conv/bilinear
      - `x = cat([r_i, za_proj], dim=1) if za else r_i`
      - `out_i = conv_1x1(x)` (trainable 1x1 Conv) or small depthwise-separable conv
      - Preserve shape/dtype exactly: `out_i.shape == r_i.shape` and same dtype/device

### Trainer integration
- File: `extensions_built_in/sd_trainer/SDTrainer.py`
  - After block that computes `down_block_additional_residuals = adapter(adapter_images_dev)` and the subsequent multiplication by `adapter_multiplier`, insert:

```python
if self.train_config.appearance_pose_adapter and getattr(self.train_config.appearance_pose_adapter, 'enabled', False) and self.appearance_pose_adapter is not None:
    with self.timer('controlnet_appa_forward'):
        try:
            down_block_additional_residuals = self.appearance_pose_adapter.transform_residuals(
                down_block_additional_residuals, mid_block_res_sample if 'mid_block_res_sample' in locals() else None, timesteps=timesteps, batch_images=adapter_images_dev
            )
        except Exception as e:
            # write diagnostic and re-raise (mirrors ControlNet diagnostic style)
            self._write_appa_diag(e, down_block_additional_residuals)
            raise
```

- Add a `self.appearance_pose_adapter` initializer in the trainer setup when the config is enabled (instantiate `AppearancePoseAdapter` and include parameters in `self.params`/optimizer); add `bring_appa`/`offload_appa` helpers that mirror `toolkit/controlnet_offload.py` API.

---

## Tests

### Unit tests (`testing/test_appa_unit.py`)
- test_transform_residuals_shape_preserve: give list of tensors with multiple channel counts and ensure outputs preserve shapes/dtypes and device.
- test_grad_flow_transform_residuals:
  - Create minimal fake `adapter` returning tensors for residuals; attach a tiny UNet with LoRA on attention (or a tiny module simulating LoRA-attached parameters).
  - Run forward (SDTrainer predict path) and run backward on a small scalar loss.
  - Assert APPA params & LoRA params receive non-zero gradients; ControlNet adapter params remain zero.
- test_offload_compatibility: simulate strategy == 'accelerate' by mocking `bring_appa`/`offload_appa` and run forward.


## Diagnostics & failure handling
- On exception in `transform_residuals`, write `controlnet_appa_diag_<ts>.json` with shapes, dtypes, adapter id, trainer step, and exception trace (mirror ControlNet diagnostics).
- Emit debug prints: shapes of residuals before/after and brief scalar metrics (e.g., mean absolute diff).

---

## Metrics & evaluation
- Compare baseline vs APPA-residuals on:
  - Identity LPIPS (lower is better)
  - Pose keypoint distance (lower is better)
  - APPA training loss & LoRA loss curves (quick 2–5 step smoke visualization)
- Provide a small `eval/appa_eval.py` that loads checkpoints and computes those metrics on a small held-out set.

---

## Checkpointing & saving
- APPA state is saved to a dedicated file named `{job.name}_appa{step_num}.safetensors` in the save root when `self.appearance_pose_adapter` is present. The saver converts tensors to CPU and the configured save dtype before writing.
- On load, the trainer attempts to locate the newest `{job.name}_appa*.safetensors` file in the checkpoint directory (or `save_root`) and applies it with `self.appearance_pose_adapter.load_state_dict(..., strict=False)` if an APPA instance exists.
- The implementation is resilient: failures to save or load APPA are logged but do not abort training. Diagnostics are emitted on load/apply failures to aid recovery.

---

## Acceptance criteria
- Unit & smoke tests pass on the CI (or locally). No dtype/device errors or OOM in the smoke runs.
- Grad-flow test demonstrates APPA params & LoRA params get gradients, ControlNet parameters remain frozen.
- Small eval (2–10 examples) shows measurable gain (LPIPS or keypoint distance) or improved qualitative outputs.

---

## Timeline & estimates
- Prototype minimal APPA skeleton + unit tests: 1 day
- SDTrainer integration + smoke test: 1 day
- Evaluation and tuning + documentation: 1–2 days
- Total: ~3–4 days for a robust residuals-mode PR

---

## PR checklist
- [ ] `toolkit/appearance_pose_adapt.py` with `AppearancePoseAdapter` and basic `transform_residuals`
- [ ] Unit tests: `testing/test_appa_unit.py`
- [ ] SDTrainer patch with `controlnet_appa_forward` timer and offload safety
- [ ] Diagnostic file writing & clear error messages
- [ ] Example config: `config/examples/train_appa_residuals.example.yaml`
- [ ] `docs/APPA.md` and this `docs/AppaResiduals.md` updated

---

If you confirm, I will (a) add the APPA skeleton and unit test next (mark task in-progress), and (b) create the SDTrainer insertion patch after tests pass. Which do you want first: the skeleton + unit tests, or the SDTrainer patch and smoke integration test?