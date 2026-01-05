import torch
from types import SimpleNamespace

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
from toolkit.config_modules import AdapterConfig


def test_assignment_replaces_model_provided_controlnet(monkeypatch):
    # Create a lightweight instance without running full __init__
    proc = BaseSDTrainProcess.__new__(BaseSDTrainProcess)

    proc.name = 'test_job'
    proc.network_config = None
    proc.get_latest_save_path = lambda name: None
    proc.device_torch = torch.device('cpu')
    proc.sd = SimpleNamespace()

    tcfg = SimpleNamespace()
    tcfg.dtype = 'float32'
    proc.train_config = tcfg

    # Simulate model-provided legacy ControlNet (expects controlnet_cond positional arg)
    class LegacyCN:
        def __init__(self):
            self.name_or_path = 'my_zimage_model_v1'
        def forward(self, latents, timestep, controlnet_cond=None, *args, **kwargs):
            assert controlnet_cond is not None
            return torch.zeros_like(latents[0])

    proc.sd.controlnet = LegacyCN()

    # ensure adapter is unset so assignment branch runs
    proc.adapter = None

    # Monkeypatch the z_image loader to return a proper VideoX-style adapter
    from types import SimpleNamespace as SN

    class FakeVideoXAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.control_in_dim = 33
        def forward(self, x, t, control_context=None, control_context_scale=1.0, *a, **k):
            assert control_context is not None
            if isinstance(x, (list, tuple)):
                batch = len(x)
            elif isinstance(x, torch.Tensor):
                batch = x.shape[0]
            else:
                batch = 1
            return torch.zeros(batch, 16, 8, 8)

    def fake_loader(name_or_path=None, device=None, torch_dtype=None, **kwargs):
        return FakeVideoXAdapter()

    import extensions_built_in.diffusion_models.z_image_adapter as zmod
    monkeypatch.setattr(zmod, 'load_videox_control_adapter', fake_loader)

    # Run the assignment logic inside setup_adapter
    acfg = AdapterConfig(type='control_net')
    acfg.name_or_path = 'my_zimage_model_v1'
    acfg.train = False
    proc.adapter_config = acfg

    proc.setup_adapter()

    # After setup, proc.adapter should be replaced with the VideoX loader's adapter
    from toolkit.controlnet_compat import VideoXControlnetWrapper
    # loader returns raw adapter (not wrapped) in our fake loader, assignment path sets proc.adapter directly
    assert isinstance(proc.adapter, FakeVideoXAdapter) or isinstance(proc.adapter, VideoXControlnetWrapper)
    # sd.controlnet also replaced
    assert isinstance(proc.sd.controlnet, (FakeVideoXAdapter,))
