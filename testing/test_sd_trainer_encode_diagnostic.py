import types
import torch
import pytest
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class DummySDCrash:
    def __init__(self):
        self.transformer = types.SimpleNamespace(control_in_dim=33)

    def encode_control_images_videox(self, imgs, height=None, width=None, tile=False):
        raise ValueError("simulated encoder crash")


class DummyTrainer(SDTrainer):
    def __init__(self):
        self.sd = DummySDCrash()
        self.model_config = types.SimpleNamespace(control_use_tiling=False, control_tiling_size=256, control_tiling_overlap=32)


def test_encode_failure_diagnostic_includes_device_info():
    t = DummyTrainer()
    # Provide a 5D control_context shaped (1,3,1,512,512) like the real job
    control_context = torch.randint(0, 255, (1, 3, 1, 512, 512), dtype=torch.uint8)
    with pytest.raises(RuntimeError) as ei:
        t._encode_and_assemble_zimage_controls(control_context)
    msg = str(ei.value)
    assert 'Failed while encoding Z-Image control images' in msg
    assert 'Inputs' in msg
    # device/dtype info should be present even if None
    assert 'device' in msg and 'dtype' in msg
