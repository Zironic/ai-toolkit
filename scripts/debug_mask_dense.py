from PIL import Image
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from toolkit.masked_recon import generate_control_mask

# Generate gradient image 64x64
im = Image.linear_gradient('L').resize((64,64)).convert('RGB')
# Save to tmp for visual if needed
im.save('tmp_ctrl_dense.png')
mask = generate_control_mask(im, target_size=(64,64), threshold=0.01, base_dilate=6, auto_scale_factor=2.0, blur=3, ref_max_dim=128)
print('mask shape', mask.shape)
print('area', (mask>0.01).float().sum().item())

# Also call with path
mask2 = generate_control_mask('tmp_ctrl_dense.png', target_size=(64,64), threshold=0.01, base_dilate=6, auto_scale_factor=2.0, blur=3, ref_max_dim=128)
print('mask2 shape', mask2.shape)
print('area2', (mask2>0.01).float().sum().item())
