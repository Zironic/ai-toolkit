APPA: Appearance-Pose Pose Adaptation (design and implementation plan)

Goal
----
Improve subject LoRA training for image-only Z-Image LoRA (pose-conditioned subject appearance). Specifically, we aim to make subject LoRAs preserve identity and appearance consistently when the model is conditioned on a frozen ControlNet (pose/Z-Image). APPA is a lightweight, opt-in adapter and LoRA workflow that reconciles ControlNet's pose/control signals with UNet appearance features so that trained subject LoRAs generalize reliably across poses.

Key intents:
- **Primary goal:** Improve subject LoRA quality under pose control (Z-Image), focusing on image-only training (Stage 2 video/temporal extension is optional).
- **Secondary goals:** Provide a low-risk residuals-mode integration path for quick prototyping and a LoRA-injection mode for best final fidelity.

High-level design
-----------------

Comparison: APPA vs current approach
------------------------------------
- **Current approach (typical patterns in this repo):**
  - LoRA/PEFT is applied at model-load time (see `toolkit/model_utils.py`). Users commonly train a subject LoRA by fine-tuning LoRA parameters on subject images (text prompts), while ControlNet (if used for pose) is typically frozen or trained separately.
  - Implementation detail (repo-specific): ControlNet in this codebase produces per-block residuals that are forwarded into the UNet (see `pred_kwargs['down_intrablock_additional_residuals']`) and the subject LoRA is only ever trained on the UNet — not the ControlNet. This means the minimal, low-risk APPA integration path is **residuals-mode**, which transforms those residuals before UNet consumption while leaving ControlNet frozen.
  - Common variants:
    - **UNet-attention LoRA:** attaches low-rank adapters to UNet attention (best for appearance changes that need to interact with attention).
    - **ControlNet/adapters LoRA:** applies LoRA to adapter weights (affects how control signals are produced, not UNet's attention fusion).
  - **Limitation:** there is no explicit, learned reconciliation between ControlNet's pose signals and the LoRA-learned appearance features; this can cause degraded subject fidelity under strong pose shifts or unusual poses.

- **APPA approach:**
  - Adds an explicit appearance encoder and either (A) **LoRA-style attention injection** (preferred) so appearance features za are concatenated into K/V and low-rank updates reconcile them with pose, or (B) a **residual/context transformer** that adapts ControlNet residuals/context before model consumption.
  - **Advantages for subject LoRA training:**
    - Better appearance preservation under pose control (improves subject identity and clothing detail when the pose changes).
    - More sample-efficient adaptation: APPA focuses capacity on aligning appearance and pose instead of relearning pose-conditioned features from scratch.
    - Compatible with frozen ControlNet workflows: APPA learns to translate ControlNet outputs into the UNet-attention space where the LoRA parameters can effectively leverage them.
  - **Tradeoffs:**
    - Engineering: requires small modifications to `SDTrainer` to call APPA (timers, offload safety) and optionally attach LoRA modules to UNet attention (one-time model patch).
    - Runtime: small extra compute (appearance encoder / residual transformer) and additional params to checkpoint.
    - **Best results require LoRA applied to UNet attention;** if your current LoRA is only attached to ControlNet, APPA's LoRA-injection mode will be less effective (residuals mode is the lower-risk path).

Practical guidance:
- If your current subject LoRA is already applied to UNet attention → APPA's LoRA-injection mode is the most direct path and should yield the largest improvement.
- If your current LoRA is applied to ControlNet/adapters → consider:
  - Switching the subject LoRA to UNet-attention (recommended) or
  - Starting with APPA residuals mode to get gains with minimal model patches.
- Short-term experiment plan:
  1. Confirm which LoRA target you currently train (UNet-attention vs ControlNet). (I can scan configs to confirm.)
  2. Prototype residuals-mode APPA in `SDTrainer` (minimal code change) and run a 1–2k step smoke experiment.
  3. If residuals-mode shows gains, iterate to LoRA-injection mode and tune ranks/layers.

What is APPA?
--------------
APpearance-Pose Adaptation (APPA) is a lightweight adapter designed to align two complementary signals used by generative conditioning:
- the pose / control signal produced by a (frozen) ControlNet (or Z-Image control contexts), and
- the appearance features za extracted from a source image.

APPA has two cooperating parts:
1. A small trainable appearance encoder (if not already present) that produces spatial appearance features za matching the spatial resolution of UNet feature z, and
2. A set of lightweight LoRA-style adapters applied to the UNet attention projections so that appearance can be injected into keys/values (and optionally queries) without full fine-tuning of UNet.

Equations (informal)
--------------------
Standard attention (brief):
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V

MagicAnimate-style concatenation we will follow (notation):
    Q = WQ0 z,
    K = WK0 (z || za),
    V = WV0 (z || za),
where || denotes concatenation (spatial/channel as appropriate) and W0 are the pretrained UNet projection matrices.

APPA adds low-rank perturbations (LoRA) to those projections and injects appearance:
    Q = (WQ0 + ΔWQ) z,
    K = (WK0 + ΔWK) (z || za),
    V = (WV0 + ΔWV) (z || za),
with ΔW{Q,K,V} = B{Q,K,V} A{Q,K,V} and B ∈ R^{d×r}, A ∈ R^{r×c} (rank r ≪ d).

Implementation details
----------------------
- LoRA on UNet attention:
  - Apply LoRA adapters to the weight matrices for the Q,K,V projections in each attention block you wish to adapt.
  - If attention projection is implemented as a Conv2d (common in UNet two-dimensional attention), treat LoRA as a 1×1 Conv low-rank update.
  - Reuse existing LoRA helpers if available in the repo, otherwise add a compact utility `toolkit/lora_utils.py` providing `attach_lora(module, rank, target_pattern)` and `remove_lora`.
  - Expose config options to control which attention blocks get LoRA: `layers=['encoder','mid','decoder']` or a regex of module names.

- Appearance encoder and `za`:
  - Compute `za` as a spatial feature map aligned to `z` (same H×W). Options:
    - Small CNN that encodes the source image to match UNet encoder resolution
    - Pooled / projected features broadcasted spatially if only global appearance is desired
  - Provide `AppearanceEncoder` class in `toolkit/appearance_pose_adapt.py` with `forward(image)` → `za`.

- Two operational modes (both supported):
  1. LoRA/Attention injection mode (preferred):
     - Implement LoRA on UNet attention projections so that K and V accept `(z || za)` concatenation and include ΔW updates. This is most faithful to the paper and minimizes changes to ControlNet.
     - The LoRA modules learn to reconcile appearance and pose in attention space.
  2. Residual/context transform mode:
     - For controlnets that produce per-block residuals (the `down_block_res_samples` list), provide an APPA transformer `transform_residuals` that adjusts each residual element-wise to better match UNet.
     - For Z-Image routing, provide `transform_zimage` to refine `zimage_control_images` / control_context tensor.

- Device / dtype / offload compatibility:
  - APPA should follow the same offload strategy as ControlNet adapters; add `bring_appa` / `offload_appa` helpers similar to `controlnet_offload.py` and add a `controlnet_appa_forward` timer.

Choices & hyperparameters
-------------------------
- LoRA rank r: start with 4–16. Lower ranks are smaller and often sufficient.
- Learning rates: APPA & LoRA use a moderately small LR (1e-4 or 1e-5), possibly larger than LoRA for quick adaptation.
- Layers to adapt: start with encoder + mid-block attention; optionally extend to decoder if needed.

Training & checkpointing
------------------------
- Scope for Z-Image / image-only LoRA: For our Z-Image LoRA workflow (image-only pose transfer / APPA), the TCAN-style *Stage 2* (temporal/video extension) is **not required**. Focus on a single-stage image-level training where ControlNet is frozen and APPA + LoRA are trained to align appearance and pose.
- Train:
  - APPA appearance encoder (if used)
  - APPA LoRA adapters on UNet attention
  - (Optionally) Residual/context transformer parameters
- Stage 2 (video / temporal extension) is optional and out-of-scope for Z-Image LoRA experiments; if a future video extension is desired, add temporal layers and consider freezing APPA during initial temporal warm-up, then fine-tune jointly.
- Include APPA parameters in optimizer and in the usual checkpointing flow. Provide named checkpoint keys (e.g., `appa/state_dict`) for compatibility.

Validation & tests
------------------
- Unit tests:
  - `testing/test_appa_unit.py`: shape preservation, dtype/device movement, and gradient flow check (ensures grads reach LoRA/A params but not frozen ControlNet).
- Integration smoke test:
  - `testing/test_appa_smoke.py`: 1–5 training steps on synthetic dataset with Z-Image/pose maps verifying:
     - APPA forward runs and logs `controlnet_appa_forward`
     - `pred_kwargs` contains transformed residuals or transformed zimage images
     - Training step completes and parameters update (LoRA & APPA encoder).

Diagnostics & metrics
---------------------
- Add `controlnet_appa_diag_<time>.json` in case of forward failure including shapes, dtypes, module names and the original error.
- Add `controlnet_appa_forward` timer and emit debug logs on the shapes of transformed residuals and `za`.

Compatibility & opt-in
----------------------
- APPA is opt-in via `train_config.appearance_pose_adapter.enabled` (default false).
- If configured but the module or weights are missing on resume, fail fast with a clear message telling the user how to add or disable APPA in the config.

Practical integration notes
---------------------------
- SDTrainer modifications (where to call APPA):
  - After computing per-block residuals in `train_single_accumulation` (the code that creates `down_block_additional_residuals` and sets `pred_kwargs['down_intrablock_additional_residuals']`), add:
    - If APPA is configured in `residuals` mode: call `down_block_additional_residuals = appa.transform_residuals(down_block_additional_residuals, mid_block_res_sample)` and then set `pred_kwargs['down_intrablock_additional_residuals']`.
    - If APPA is configured in `context` mode and zimage routing is used: call `pred_kwargs['zimage_control_images'] = appa.transform_zimage(pred_kwargs['zimage_control_images'])`.
  - For LoRA-injection mode, attach LoRA modules into UNet attention modules at model initialization (a one-time patch) using a helper `attach_appa_lora(unet, rank, layers)` that registers A/B parameters and modifies forward.

Deliverables (updated)
----------------------
- `toolkit/appearance_pose_adapt.py` : `AppearanceEncoder`, `AppearancePoseAdapter` (residual/context transform), `attach_appa_lora` helper
- Unit tests and integration smoke tests (see test names above)
- `config/examples/train_appa_lora_zimage.example.yaml` with suggested hyperparameters
- SDTrainer changes to call APPA at the adapter forward sites and to include APPA parameters in optimizer/save

Open questions (reiterated)
-------------------------
- Residual vs context vs direct LoRA injection: start with LoRA + optional residual transform fallback; residual mode is lower risk for minimal change. 
- Do we want a small APPA-specific loss (e.g., alignment between transformed residuals and appearance features)? Start without it and add later if necessary.

Integration points
------------------
- File: `extensions_built_in/sd_trainer/SDTrainer.py`
  - After adapter forward (where `down_block_additional_residuals = adapter(adapter_images_dev)` or where `pred_kwargs['zimage_control_images']` is set), call APPA:
    - If adapter returned per-block residuals: call `appa.transform_residuals(down_block_res_samples, mid_block_res_sample)` -> update `pred_kwargs['down_intrablock_additional_residuals']`.
    - If zimage routing is used: call `appa.transform_zimage(zimage_ctrl)` -> set `pred_kwargs['zimage_control_images']`.
- Add config toggles in training config (`train_config.appearance_pose_adapter: {enabled: bool, type: 'residuals'|'context', module: '<path>', params: {...}}`).
- Ensure APPA parameters are included in optimizer when enabled and saved as model checkpoint via `network.save_weights` or adapter-specific save.

Detailed specification & API
----------------------------

Module API (precise)
---------------------
- class toolkit.appearance_pose_adapt.AppearancePoseAdapter(nn.Module):
  - __init__(self, config: dict)
    - config fields (examples below):
      - mode: 'lora' | 'residuals' | 'context'  # behavior
      - lora_rank: int
      - lora_layers: list or regex
      - appearance_encoder: {type: 'cnn'|'resnet', out_channels: int, pretrained: bool}
      - optimizer: {lr: float, weight_decay: float}
      - offload_strategy: 'none'|'accelerate'|'manual_swap'|'memory_manager'
  - forward_residuals(self, down_block_res_samples: List[Tensor], mid_block_res_sample: Tensor, timesteps: Optional[Tensor]=None, **kwargs) -> Tuple[List[Tensor], Tensor]
    - Input shapes: each element in `down_block_res_samples` is [B, C_i, H_i, W_i]; mid_block_res_sample is [B, C_mid, H_mid, W_mid]
    - Output: same list length and shapes (semantic-preserving). Implementation may internally apply small conv/MLP or per-channel scaling.
  - forward_context(self, zimage_ctrl: Tensor, timesteps: Optional[Tensor]=None, **kwargs) -> Tensor
    - Input shapes: `zimage_ctrl` can be either [B, C, F, H, W] (Z-Image with frames) or [B, C, H, W]; output should match shape expected by model-side routing.
  - attach_lora_to_unet(unet: nn.Module, rank: int, layers: list|regex) -> None
    - Attaches LoRA modules (A/B pairs) to matched attention projection modules and registers the LoRA parameters.
  - state_dict()/load_state_dict() behave like standard nn.Module for checkpointing.

Example config snippet (YAML)
----------------------------
```yaml
train_config:
  appearance_pose_adapter:
    enabled: true
    mode: lora            # 'lora'|'residuals'|'context'
    lora_rank: 8
    lora_layers: ['encoder.*attn.*','mid.*attn.*']
    appearance_encoder:
      type: 'cnn'
      out_channels: 64
      pretrained: false
    offload_strategy: 'accelerate'
    optimizer:
      lr: 3e-5
      weight_decay: 0.0
    aux_loss:
      enabled: false
      type: 'cosine'   # optional: 'cosine'|'l2'
      weight: 0.1
```

Pseudocode & integration (concise)
----------------------------------
1) Model initialization (one-time):
```python
if cfg.appearance_pose_adapter.enabled and cfg.appearance_pose_adapter.mode == 'lora':
    attach_lora_to_unet(unet, rank=cfg.appearance_pose_adapter.lora_rank, layers=cfg.appearance_pose_adapter.lora_layers)
    # LoRA params become part of optimizer by normal registration
```

2) During `train_single_accumulation` after adapter forward (residuals branch):
```python
with self.timer('controlnet_appa_forward'):
    if cfg.appearance_pose_adapter.enabled and cfg.appearance_pose_adapter.mode == 'residuals':
        down_block_additional_residuals = self.appa.forward_residuals(down_block_additional_residuals, mid_block_res_sample, timesteps=timesteps)
        pred_kwargs['down_intrablock_additional_residuals'] = down_block_additional_residuals
```

3) During `train_single_accumulation` after zimage routing (context branch):
```python
if cfg.appearance_pose_adapter.enabled and cfg.appearance_pose_adapter.mode == 'context' and 'zimage_control_images' in pred_kwargs:
    pred_kwargs['zimage_control_images'] = self.appa.forward_context(pred_kwargs['zimage_control_images'], timesteps=timesteps)
```

Shape & dtype contracts
-----------------------
- `forward_residuals` MUST preserve list length and output tensors with same shape and dtype as inputs (device may change to `noisy_latents` device). If it cannot preserve shape, it must raise a descriptive error.
- `forward_context` must output `zimage_control_images` shaped as expected by the model (either keep [B,C,F,H,W] or [B,C,H,W]).
- Both methods must be autograd-friendly (return tensors that require_grad when training) and support `torch.no_grad()` when called under inference.

Unit tests (detailed)
---------------------
- `testing/test_appa_unit.py` should include:
  1. test_forward_residuals_shape_preserved: pass a list of fake tensors with varied channels and ensure output shapes match and dtype/device are correct.
  2. test_forward_context_supports_both_shapes: pass both [B,C,H,W] and [B,C,F,H,W] and verify output shape matches expectation.
  3. test_grad_flow: run forward + small loss backward and assert that APPA parameters receive gradients while a frozen ControlNet remains unmodified.
  4. test_offload_compatibility: simulate `offload_strategy` by calling `bring_appa`/`offload_appa` helpers and ensure forward still runs.
  5. test_state_dict_roundtrip: save/load state and verify parameter equality.

Integration smoke test (detailed)
---------------------------------
- `testing/test_appa_smoke.py`:
  - Create synthetic mini-dataset: 2–4 samples with random images and corresponding pose maps or zimage tensors.
  - Configure trainer with `appearance_pose_adapter.enabled=true` and `mode=residuals` (and a small LoRA rank).
  - Run 3–5 training steps and assert:
    - `controlnet_appa_forward` timer was executed
    - `pred_kwargs` contains the APPA-transformed outputs
    - optimizer step updated APPA/LoRA params (numeric check of parameter norms before/after)

Evaluation & metrics
--------------------
- Recommended metrics for APPA evaluation (image animation / pose transfer):
  - Perceptual similarity (LPIPS) between generated frame and target pose-conditioned reference
  - FID / IS on generated images (small eval set)
  - Pose consistency: run OpenPose/keypoint extractor on generated images and compute keypoint distance to driving pose (lower is better)
  - Appearance preservation: compare feature similarity to source image (LPIPS or cosine in a pretrained feature space)
  - Background consistency: simple masked PSNR/SSIM of background regions
- Provide an `eval/` script that given a checkpoint produces metric tables and a visual grid (source, driving frame, generated) for quick inspection.

Ablation study (recommended experiments)
---------------------------------------
1. Modes: Residuals vs Context vs LoRA-injection
2. LoRA rank sweep: r ∈ {1, 2, 4, 8, 16}
3. With/without appearance encoder (za learned vs za fixed/global)
4. With/without APPA aux loss (if implemented)
5. Different layer sets: encoder-only vs encoder+mid vs all-attn

For each experiment, log: training curves (loss, aux loss), evaluation metrics above, and a small gallery of qualitative examples.

Hyperparameters & training schedule (recommended defaults)
---------------------------------------------------------
- Optimizer: AdamW
  - APPA encoder: lr=3e-4 (small encoder) or 1e-4 for larger nets
  - LoRA params: lr=3e-4 -> 3e-5 (tune)
  - weight_decay for LoRA: 0.0
- Batch size: keep small for initial experiments (8–32) depending on memory
- Steps: 2k–10k for a quick run; longer runs for final results
- Mixed precision: use BF16 if available (consistent with zimage adapter usage)
- Checkpoint frequency: every 500–1000 steps with a short eval run

Performance & instrumentation
-----------------------------
- Add `controlnet_appa_forward` timer (already planned). Record:
  - per-step wall time and memory delta (if accessible)
- Expected overhead: LoRA injection is negligible (<5% compute for small ranks); appearance encoder/residual transformer adds memory/time proportional to its size (design it small).
- If offload_strategy is active, ensure `bring_appa`/`offload_appa` follow the same code patterns used for ControlNet to avoid memory spikes.

Reproducibility & checkpoints
-----------------------------
- Save APPA state under `appa/` prefix in checkpoints (e.g., `appa/state_dict.pt`). Store the training `cfg` alongside the checkpoint.
- Set deterministic seeds (torch, numpy, random) in the smoke/eval scripts and record them in logs.

Evaluation dataset & minimal experiment (practical)
--------------------------------------------------
- Use a small subset of a human-pose dataset (if available in `datasets/`); otherwise, synthesize using existing Jinx images + OpenPose-generated pose maps.
- Minimal experiment: train LoRA+APPA residual mode for 2k steps on 1–2 source images, evaluate LPIPS and pose-keypoint error on held-out poses.

References & citations
----------------------
- TCAN (arXiv:2407.09012) — Appearance-Pose Adaptation concept and training stages
- MagicAnimate — concatenation-in-attention injection idea
- LoRA (H. Hu et al.) — low-rank adaptation used for efficient fine-tuning

Training checklist & PR tasks (expanded)
----------------------------------------
- [ ] Add `toolkit/appearance_pose_adapt.py` with `AppearanceEncoder`, `AppearancePoseAdapter`, and `attach_lora_to_unet`
- [ ] Add `toolkit/lora_utils.py` if repo lacks robust helpers
- [ ] Add unit tests `testing/test_appa_unit.py`
- [ ] Add integration smoke `testing/test_appa_smoke.py` and example config
- [ ] Add SDTrainer patch to call APPA with timers and offload safety
- [ ] Add eval script & baseline comparisons and a short LEARNINGS.md note of best hyperparameters

If you'd like, I can now incorporate these additional sections into the file (examples/pseudocode, YAML, tests, metrics and ablation plan) and push the updated `docs/APPA.md`. Would you like me to proceed with those edits, or change anything in the direction above first?
Tests & smoke checks
--------------------
- Unit tests for APPA module: shape preservation, dtype/device transitions, simple forward pass.
- Integration smoke test: small synthetic dataset (1–3 images) with Z-Image control; run 1–5 steps ensuring:
  - APPA gets called (new timer / debug log)
  - `pred_kwargs` contains transformed residuals or transformed zimage images
  - Training step runs and loss decreases on tiny synthetic task
- Add performance instrumentation: `controlnet_appa_forward` timer and ensure offload/bring_adapter interplay is safe.

Diagnostics & debug
-------------------
- Extend existing ControlNet debug messages: indicate APPA presence and shapes.
- If APPA fails, write a diagnostic `controlnet_appa_diag_<time>.json` containing shapes, dtypes, and adapter name.

Backward compatibility
----------------------
- APPA is opt-in via config; default is disabled.
- If an APPA module is configured but absent on load, fail fast with actionable error.

Deliverables (minimal PR)
-------------------------
1. `docs/APPA.md` (this file) — design + plan
2. Add `toolkit/appearance_pose_adapt.py` skeleton with `AppearancePoseAdapter` + unit tests in `testing/`
3. Modify `SDTrainer.train_single_accumulation` to call APPA when configured (with timers and safe offload handling)
4. Add example config `config/examples/train_appa_lora_zimage.example.yaml` and a smoke `testing/test_appa_smoke.py`
5. Add PR checklist and small note to `cleanup.md`/`LEARNINGS.md` about APPA choice.

Project timeline & estimates
----------------------------
- Draft & PR (APPA doc + skeleton + tests): 1–2 days
- Integration & smoke tests: 2–3 days
- Iteration and tuning: 1–2 days

Open questions
--------------
- Residual vs context transform: which is higher priority? (Residual mode is lowest friction; context mode may be more powerful for Z-Image.)
- Should we add a small APPA-specific aux loss, or rely on LoRA+controlnet_aux_loss? (Prototype both; start without additional loss.)

Next steps (concrete)
---------------------
- Implement minimal APPA skeleton and unit tests (task 2 in TODOs).
- Integrate call in `SDTrainer` and add timers/diagnostics (task 3).

Contact
-------
If you'd like, I can now scaffold the APPA module, tests, and the PR changes. Which part should I do first: prototype module or SDTrainer integration?