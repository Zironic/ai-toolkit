import torch
from types import SimpleNamespace

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.config_modules import AdapterConfig


def test_assignment_replacement_fails_strict(monkeypatch):
    proc = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
    proc.name = 'test_job'
    proc.network_config = None
    proc.get_latest_save_path = lambda name: None
    proc.device_torch = torch.device('cpu')
    proc.sd = SimpleNamespace()

    tcfg = SimpleNamespace()
    tcfg.dtype = 'float32'
    proc.train_config = tcfg

    # Simulate model-provided controlnet and adapter_config
    class LegacyCN:
        def __init__(self):
            self.name_or_path = 'my_zimage_model_v1'
        def forward(self, *a, **k):
            return None

    proc.sd.controlnet = LegacyCN()
    proc.adapter = None

    acfg = AdapterConfig(type='control_net')
    acfg.name_or_path = 'my_zimage_model_v1'
    acfg.controlnet_mode = 'zimage'
    acfg.train = False
    proc.adapter_config = acfg

    # make loader raise
    import extensions_built_in.diffusion_models.z_image_adapter as zmod
    def boom(*a, **k):
        raise RuntimeError('boom-loader')
    monkeypatch.setattr(zmod, 'load_videox_control_adapter', boom)

    try:
        proc.setup_adapter()
        raise AssertionError('Expected setup_adapter to raise when replacement fails')
    except RuntimeError:
        # Strict failure is enforced; exact error text may come from loader or our guard.
        pass
