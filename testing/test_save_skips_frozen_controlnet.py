import os
import types
import tempfile
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


class DummyAdapter:
    def __init__(self):
        self.saved = False
        self.device = 'cpu'
        self.dtype = None

    def state_dict(self):
        return {}

    def to(self, device, dtype=None):
        return self

    def save_pretrained(self, path, dtype=None, safe_serialization=False):
        # mark if called
        self.saved = True


def make_minimal_proc(tmp_path):
    # Bypass __init__ complexity
    p = object.__new__(BaseSDTrainProcess)
    p.accelerator = types.SimpleNamespace(is_main_process=True)
    p.ema = None
    p.save_root = str(tmp_path)
    p.job = types.SimpleNamespace(name='JOB')
    p.meta = {}
    p.save_config = types.SimpleNamespace(save_format='safetensors', dtype='float32')
    # train_config must exist with attributes used by save()
    p.train_config = types.SimpleNamespace(train_unet=False, train_text_encoder=False, train_refiner=False, save_loss_json=False)
    p.network = None
    p.embedding = None
    p.decorator = None
    p.adapter = DummyAdapter()
    p.adapter_config = types.SimpleNamespace(type='control_net', train=False, train_only_image_encoder=False)
    p.sd = types.SimpleNamespace(refiner_unet=None, unet=None, text_encoder=None, is_multistage=False)
    p.optimizer = None
    p.lr_scheduler = None
    # ensure save root exists
    os.makedirs(p.save_root, exist_ok=True)
    return p


def test_save_does_not_write_frozen_controlnet(tmp_path):
    p = make_minimal_proc(tmp_path)

    # Run save; should not raise and should not call adapter.save_pretrained
    p.save(step=123)

    assert p.adapter.saved is False, "Frozen ControlNet adapter was unexpectedly saved to disk"
