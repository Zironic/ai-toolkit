import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from diffusers import ControlNetModel
from toolkit.control_channels import adapt_control_images

p = r"C:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\models\Personalized_Model\ZIT-Controlnet-Union-2.1-8steps"
print('Loading ControlNet from:', p)
cn = ControlNetModel.from_pretrained(p, local_files_only=True)
ctrl = torch.randn(1, 4, 512, 512)
adapted, expected = adapt_control_images(ctrl, cn)
print('expected:', expected, 'adapted.shape=', tuple(adapted.shape))
# try passing through conv_in and first conv to ensure no conv mismatch
try:
    out = cn.conv_in(adapted)
    print('cn.conv_in ok ->', tuple(out.shape))
except Exception as e:
    print('cn.conv_in failed:', e)

# Try searching for any conv module that would raise on adapted input
import torch.nn as nn
try:
    for m in cn.modules():
        if isinstance(m, nn.Conv2d):
            try:
                # build a small fake tensor with adapted channel count
                fake = torch.randn(1, adapted.shape[1], 16, 16)
                m(fake)
            except Exception as e:
                print('Module conv failed for in_ch', adapted.shape[1], 'err:', e)
                break
    else:
        print('All conv modules accepted in_ch', adapted.shape[1])
except Exception as e:
    print('Error while sweeping modules:', e)
