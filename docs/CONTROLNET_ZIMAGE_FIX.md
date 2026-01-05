# Z-Image (VideoX) ControlNet channel mismatch fix

This project includes a targeted fix to handle channel mismatches that previously caused runtime Conv2d errors when using VideoX/Z-Image-style ControlNets (e.g., `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`).

What was happening
- The trainer attempted to infer the adapter's expected input channels by frequency over all conv modules; for some adapters this returned `1280` and caused noisy latents to be padded to 1280 channels, which later failed when the primary `conv_in` actually expected 4 channels.

What we changed
- Added `infer_expected_in_ch(adapter)` which prefers:
  1. `adapter.control_in_dim` if present
  2. `adapter.conv_in.weight.shape[1]` if present
  3. fallback to the frequency-based conv-weight heuristic, while resolving 3/4 conflicts by preferring 3 when both are present.
- The zimage routing now uses this helper and performs grouped-mean reductions (e.g., 1280->4) when `channels % expected == 0` rather than padding to a larger count.
- `VideoXControlnetWrapper` now dynamically detects Conv channel-mismatch runtime errors, adapts `control_context` channels (trim/pad) and retries once before surfacing a clear error.
- Added unit tests `testing/test_control_util.py`, `testing/test_control_channels.py`, and `testing/test_controlnet_compat.py` to validate these behaviors.

Quick checks & smoke runs
- Run unit tests:
  - `python -m pytest testing/test_zimage_channel_inference.py -q`
- Run a minimal CPU smoke job (modify your job config):
  - Set `device: "cpu"`, `steps: 1`, `batch_size: 1`, and `controlnet_model` to the local folder path for the ControlNet.
  - Ensure `controlnet_mode='zimage'` or let the trainer auto-detect.

Fail-fast & diagnostics
- If a ControlNet is misconfigured or incompatible we'll raise a descriptive `RuntimeError` with guidance (e.g., mismatch in expected channels or missing tokenizer). The trainer will not silently continue when control residuals cannot be computed.

If you still see a Conv2d channel mismatch for `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`, please run the unit test above and open an issue with a small reproduction (checkpoint path and a minimal job config).