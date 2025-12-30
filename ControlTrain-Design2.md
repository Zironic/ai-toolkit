# ControlTrain Design (Tensor Contracts & Inference Quickstart)

This companion doc records the **exact tensor shapes, ranges, and calling conventions** used by the controlnet/union training & inference flows. It is referenced by `ControlTrain-Design.md` and by runtime checks in `toolkit/dataloader_mixins.py` and `tools/gen_control.py`.

## Canonical tensor contracts ✅

- **Images (model input)**: `torch.FloatTensor` shaped **[B, C, H, W]**, where
  - B: batch size
  - C: channels (typically 3 for RGB)
  - H, W: height, width
  - Values: **in [0.0, 1.0]** (normalized float32)

- **Videos (model input)**: `torch.FloatTensor` shaped **[B, C, T, H, W]**, where
  - T: time dimension (frames)
  - Values: **in [0.0, 1.0]** (normalized float32)

- **Controls**: `torch.FloatTensor` shaped **[B, Cc, H, W]** for per-frame/per-image controls
  - `Cc` is control-channel count (heatmaps, skeletons, RGBA references, etc.)
  - Controls must be normalized to **[0.0, 1.0]** (even if written to disk as 0-255 uint8)
  - When produced from `make_openpose_map()` the dataloader must convert/normalize to float and ensure `Cc` is explicit.

- **Latents & helper outputs**: document canonical shapes and slicing in helper docs (examples follow):
  - `get_image_latent(image)` → expected to return either `[B, C_lat, H_lat, W_lat]` (image latent) or `[B, C_lat, T, H_lat, W_lat]` for video-context helpers. When an image->video latent is used the VideoX‑Fun examples index into the time axis using `[..., 0]` to extract a single-frame latent (i.e. `get_image_latent(...)[..., 0]`), and the codebase should use explicit slicing rather than implicit reshapes.
  - `get_video_to_video_latent(video)` → returns `[B, C_lat, T, H_lat, W_lat]`.

> Note: When a helper returns either 4D or 5D latents, code that consumes them MUST use explicit axis-aware indexing to avoid ambiguous behavior (do not assume singleton time dim implicitly).

## Range invariants

- All floating input tensors passed to model pipelines must be in **[0.0, 1.0]**; VAE encoders and internal pipelines will assume this range for correct scaling and numeric stability.
- When reading on-disk control images (uint8), the dataloader must map `uint8 -> float32` with division by 255.0.

## How helpers are used (VideoX‑Fun examples)

- Example usage patterns are present in VideoX‑Fun repository examples (local `examples/`): search for `predict` files (e.g., `examples/predict_*`) to see typical usages:
  - Extracting an image latent for a single frame: `img_latent = get_image_latent(img); frame_latent = img_latent[..., 0]` (explicit indexing).
  - Converting image control -> video control contexts by broadcasting or tiled repeat over `T` when necessary.

## Cross-references / enforcement points

- **`toolkit/dataloader_mixins.py`**: add explicit assertions for shapes/ranges and log helpful messages when mismatches occur. Recommended check sequence:
  - Assert `tensor.ndim` is expected (4 for image, 5 for video)
  - Cast to `torch.float32` and clamp to `[0.0, 1.0]` (or raise when values fall outside the expected range in debug mode)
  - If `control` is provided as an image (uint8 on disk), perform `control = control.astype(np.float32) / 255.0`

- **`tools/gen_control.py`**: the manifest and written files should include the declared `format` (`uint8` or `float32`) and `control_channels`. The CLI should emit machine-readable JSON describing the contract for downstream loaders.

## Quick reference snippet (inference)

```py
# Quickstart (recommended defaults):
pipe = load_union_pipeline('Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors')
pipe.num_inference_steps = 8  # prefer distilled 8-step checkpoint
pipe.control_context_scale = 0.75
out = pipe.predict(prompt, image_tensor, controls=[c1, c2])
```

See `examples/predict_t2i_control_2.1.py` for a full minimal example and learn how controls and per-control scales are passed.

---

(End of tensor contract notes)