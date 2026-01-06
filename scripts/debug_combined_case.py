import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image, ImageDraw
import numpy as np
import cv2
from scripts.preview_mask import make_pose_coverage_mask
from toolkit.masked_recon import generate_control_mask

W,H=64,64
base = 'tmp_base_rect.jpg'
im = Image.new('RGB', (W, H), (255, 255, 255))
draw = ImageDraw.Draw(im)
rect = (12, 12, 52, 50)
draw.rectangle(rect, fill=(0, 0, 0))
im.save(base)

controls_dir = '_controls'
os.makedirs(controls_dir, exist_ok=True)
ctrl = os.path.join(controls_dir, 'img.pose.png')
c = Image.new('RGB', (W, H), (0,0,0))
cd = ImageDraw.Draw(c)
cd.ellipse((30, 26, 34, 30), fill=(255,255,255))
c.save(ctrl)

mask = make_pose_coverage_mask(ctrl, target_size=(H,W), threshold=0.01, base_dilate=6, auto_scale_factor=1.0, blur=1, ref_max_dim=128, base_img_path=base, use_edges=True)
print('mask area', (mask>0.1).sum().item())
mask_t = generate_control_mask(ctrl, base_img_path=base, target_size=(H,W), threshold=0.01, base_dilate=6, auto_scale_factor=1.0, blur=1, ref_max_dim=128, use_edges=True)
print('mask_t area', (mask_t>0.1).sum().item())
# compute edges
g = cv2.imread(base, cv2.IMREAD_GRAYSCALE)
med = int(np.median(g))
lower = max(1, int(0.66 * med))
upper = min(255, int(1.33 * med))
edges = cv2.Canny(g, lower, upper)
print('edges nonzero', int(np.count_nonzero(edges)), 'size', edges.shape)
# compute connected components on inverted
ref_H, ref_W = 128,128
img_res = Image.open(base).convert('L').resize((ref_W, ref_H))
g = np.array(img_res)
med = int(np.median(g))
lower = max(1, int(0.66 * med))
upper = min(255, int(1.33 * med))
edges = cv2.Canny(g, lower, upper)
# dilate edges
r_px = int(round(6 * (1.0 + max((rect[3]-rect[1]+1),(rect[2]-rect[0]+1))/float(max(ref_H,ref_W)) * 1.0)))
k_close = max(3, int(round(r_px / 6)) * 2 + 1)
edges_dil = cv2.dilate((edges>0).astype(np.uint8)*255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k_close,k_close)), iterations=1)
inverted = (edges_dil == 0).astype(np.uint8) * 255
print('inverted nonzero', int(np.count_nonzero(inverted)), 'shape', inverted.shape)
num, labels = cv2.connectedComponents(inverted, connectivity=8)
print('num components', num)
# list component areas
areas = {}
for lab in range(1,num):
    areas[lab] = int((labels==lab).sum())
print('areas', areas)
# cleanup
os.remove(base)
os.remove(ctrl)
os.rmdir(controls_dir)
