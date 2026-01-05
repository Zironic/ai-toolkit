import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from diffusers import ControlNetModel
from toolkit.controlnet_compat import VideoXControlnetWrapper

p = r"C:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\models\Personalized_Model\ZIT-Controlnet-Union-2.1-8steps"
print('Loading ControlNet from:', p)
cn = ControlNetModel.from_pretrained(p, local_files_only=True)
wrapper = VideoXControlnetWrapper(cn)
latents = torch.randn(1, 4, 512, 512)
ctrl4 = torch.randn(1, 4, 512, 512)
try:
    out = wrapper(latents, 0, ctrl4, conditioning_scale=1.0)
    print('Wrapper call succeeded, output type:', type(out))
except Exception as e:
    print('Wrapper call failed:', e)
