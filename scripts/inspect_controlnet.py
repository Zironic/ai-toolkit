import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolkit.control_util import infer_expected_in_ch
from toolkit.print import print_acc

from diffusers import ControlNetModel

p = r"C:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\models\Personalized_Model\ZIT-Controlnet-Union-2.1-8steps"
print(f"Loading ControlNet from: {p}")
cn = ControlNetModel.from_pretrained(p, use_auth_token=None, local_files_only=True)
print('Loaded ControlNet, class:', cn.__class__)

# show conv_in if present
conv_in = getattr(cn, 'conv_in', None)
print('conv_in present:', conv_in is not None)
if conv_in is not None:
    w = conv_in.weight
    print('conv_in weight shape:', tuple(w.shape))

# count conv in_channels across modules
conv_counts = {}
for m in cn.modules():
    import torch.nn as nn
    if isinstance(m, nn.Conv2d):
        try:
            in_ch = int(m.weight.shape[1])
        except Exception:
            in_ch = None
        conv_counts[in_ch] = conv_counts.get(in_ch, 0) + 1
print('Conv in_channel counts:', conv_counts)

inferred = infer_expected_in_ch(cn)
print('infer_expected_in_ch =>', inferred)

# Also test if wrapper unwrapping would change anything
class Wrapper:
    def __init__(self, inner):
        self.inner = inner

wrapped = Wrapper(cn)
print('Wrapped infer =>', infer_expected_in_ch(wrapped))
