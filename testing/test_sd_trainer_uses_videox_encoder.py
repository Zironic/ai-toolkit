import types
import torch
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class DummySDV:
    def __init__(self):
        # transformer must advertise control_in_dim=33 for VideoX
        self.transformer = types.SimpleNamespace(control_in_dim=33)
        self.called = False

    def encode_control_images_videox(self, imgs, height=None, width=None, tile=False):
        # Record that we were called and return simple latents as [B,C,H,W]
        self.called = True
        outs = []
        for img in imgs:
            outs.append(torch.randn(16, 32, 32))
        return torch.stack(outs, dim=0)


class DummyTrainer(SDTrainer):
    def __init__(self):
        # minimal init: attach sd and model_config
        self.sd = DummySDV()
        self.model_config = types.SimpleNamespace(control_use_tiling=False, control_tiling_size=256, control_tiling_overlap=32)


def test_sd_trainer_prefers_videox_encoder():
    t = DummyTrainer()
    imgs = [torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8) for _ in range(2)]
    out = t._encode_and_assemble_zimage_controls(imgs)
    assert t.sd.called
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 5
    assert out.shape[1] == 33
