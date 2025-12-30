"""Minimal example: predict t2i with Union ControlNet 2.1 style

Shows: loading a union checkpoint, preparing per-control tensors, passing them to pipeline
with per-control context scales and using the recommended 8-step distilled checkpoint.
"""
import torch

# Pseudocode / example only - adapt to your runtime helpers
from toolkit.model_utils import load_model_for_inference

# Load pipeline (assume wrapper exposes .predict(prompt, image, controls, control_context_scale))
pipe = load_model_for_inference('Z-Image-Turbo-Fun-Controlnet-Union-2.1', device='cpu')
pipe.num_inference_steps = 8

# Prepare inputs (synthetic example)
B, C, H, W = 1, 3, 512, 512
image = torch.rand(B, C, H, W, dtype=torch.float32)  # values in [0,1]

# Two synthetic control tensors (per-control), normalized
control_a = torch.rand(B, 1, H, W, dtype=torch.float32)
control_b = torch.rand(B, 3, H, W, dtype=torch.float32)
controls = [control_a, control_b]

# Per-control context scales
control_context_scale = [0.8, 0.7]

# Run predictable 8-step inference
out = pipe.predict(prompt="A fantasy portrait", image=image, controls=controls, control_context_scale=control_context_scale)
print('Output (placeholder):', type(out))
