"""Diagnostic script: load a ControlNet adapter and run a single forward
with float32 noisy latents / timesteps to surface dtype mismatches.
"""
import torch
from diffusers import ControlNetModel
import os

CONTROLNET_PATH = r"C:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\models\Personalized_Model\ZIT-Controlnet-Union-2.1-8steps"

print(f"Loading ControlNet from {CONTROLNET_PATH}")
cn = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=None)

# Inspect parameter dtypes
params = list(cn.parameters())
if len(params) == 0:
    print("No parameters found on ControlNet")
else:
    print("First param dtype:", params[0].dtype, "device:", params[0].device)

# Look for a time-related param dtype
time_dtype = None
for name, p in cn.named_parameters():
    lname = name.lower()
    if 'time' in lname or 'timestep' in lname or 'time_proj' in lname or 'time_embed' in lname or 'time_embedding' in lname:
        time_dtype = p.dtype
        print(f"Found time-related param: {name}, dtype={p.dtype}")
        break

# Make synthetic inputs
noisy_latents = torch.randn(1, 4, 16, 16, dtype=torch.float32)
# timesteps may be scalar or tensor; use tensor
timesteps = torch.tensor([10], dtype=torch.float32)

print("Input dtypes: noisy_latents", noisy_latents.dtype, "timesteps", timesteps.dtype)

# Attempt a forward with float32 inputs; wrap to catch error and print stack
try:
    with torch.no_grad():
        out = cn(noisy_latents, timesteps, encoder_hidden_states=None, controlnet_cond=None)
    print("Forward succeeded. Output types:")
    if isinstance(out, tuple) or isinstance(out, list):
        for i, o in enumerate(out):
            if hasattr(o, 'dtype'):
                print(f" out[{i}] dtype={o.dtype} shape={getattr(o, 'shape', None)}")
            else:
                print(f" out[{i}] type={type(o)}")
    else:
        print("Out type", type(out))
except Exception as e:
    print("Forward failed:")
    import traceback
    traceback.print_exc()
    
    # Try casting inputs to controlnet param dtype if available
    if params:
        target = params[0].dtype
        print(f"Retrying with inputs cast to param dtype {target}")
        try:
            noisy2 = noisy_latents.to(dtype=target)
            timesteps2 = timesteps.to(dtype=target)
            with torch.no_grad():
                out = cn(noisy2, timesteps2, encoder_hidden_states=None, controlnet_cond=None)
            print("Forward succeeded after casting to param dtype")
        except Exception:
            print("Still failed after casting to param dtype")
            traceback.print_exc()

print("Done.")
