import torch
from types import SimpleNamespace

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.config_modules import AdapterConfig


def test_setup_adapter_prefers_videox_loader(monkeypatch):
    # Create a lightweight instance without running full __init__
    proc = BaseSDTrainProcess.__new__(BaseSDTrainProcess)

    # minimal attributes required by setup_adapter
    proc.name = 'test_job'
    proc.network_config = None
    proc.get_latest_save_path = lambda name: None
    proc.device_torch = torch.device('cpu')
    proc.sd = SimpleNamespace()  # minimal object used by setup_adapter for assignment

    tcfg = SimpleNamespace()
    tcfg.dtype = 'float32'
    proc.train_config = tcfg

    # Adapter config that hints at zimage via name
    acfg = AdapterConfig(type='control_net')
    acfg.name_or_path = 'my_zimage_model_v1'
    acfg.train = False
    proc.adapter_config = acfg

    # Monkeypatch the z_image adapter module to provide a simple `from_pretrained`-capable class
    import extensions_built_in.diffusion_models.z_image_adapter as z_adapter_mod

    class FakeZImageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.control_in_dim = 33
        def forward(self, x, t, control_context=None, control_context_scale: float = 1.0, *args, **kwargs):
            # simple pass-through return shaped like a control residual
            if isinstance(x, (list, tuple)):
                batch = len(x)
            elif isinstance(x, torch.Tensor):
                batch = x.shape[0]
            else:
                batch = 1
            return torch.zeros(batch, 16, 8, 8)
        @classmethod
        def from_pretrained(cls, name_or_path, **kwargs):
            return cls()

    z_adapter_mod.ZImageControlTransformer2DModel = FakeZImageModel

    # run setup
    proc.setup_adapter()

    # Should have loaded via VideoX loader -> adapter should be a VideoXControlnetWrapper instance
    from toolkit.controlnet_compat import VideoXControlnetWrapper
    assert isinstance(proc.adapter, VideoXControlnetWrapper), f"Expected VideoXControlnetWrapper, got {type(proc.adapter)}"

    # Ensure inner forward accepts control_context
    inner = getattr(proc.adapter, 'inner', None)
    assert inner is not None
    # verify forward signature existence (should accept control_context when inspectable)
    import inspect
    target_fn = getattr(inner, 'forward', None)
    if target_fn is not None:
        sig = inspect.signature(target_fn)
        assert 'control_context' in sig.parameters
