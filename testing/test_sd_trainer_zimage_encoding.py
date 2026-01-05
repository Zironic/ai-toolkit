import types
import torch
import pytest
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class DummySD:
    def __init__(self):
        # transformer must advertise control_in_dim=33 for VideoX
        self.transformer = types.SimpleNamespace(control_in_dim=33)
        self.encoded_called = False

    def encode_control_images(self, imgs, tile=False, tile_size=256, overlap=32):
        # Return simple latents: for each img, return a tensor [C=16,H=32,W=32]
        outs = []
        for img in imgs:
            # create dummy latent shaped [C,H,W]
            outs.append(torch.randn(16, 32, 32))
        # return tensor stacked as [B,C,H,W]
        return torch.stack(outs, dim=0)


class DummyTrainer(SDTrainer):
    def __init__(self):
        # minimal init: attach sd and model_config
        self.sd = DummySD()
        self.model_config = types.SimpleNamespace(control_use_tiling=False, control_tiling_size=256, control_tiling_overlap=32)


def test_encode_and_assemble_returns_5d_control_context():
    t = DummyTrainer()
    # create pixel images [B,3,H,W]
    imgs = torch.randn((2, 3, 512, 512))
    out = t._encode_and_assemble_zimage_controls(imgs)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 5
    # channel count should be 33 per assemble
    assert out.shape[1] == 33

