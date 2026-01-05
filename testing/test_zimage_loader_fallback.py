import types
import torch
from toolkit.control_util import ensure_control_in_dim, infer_expected_in_ch
from toolkit.controlnet_compat import VideoXControlnetWrapper


class DummyAdapterNoControlDim:
    def __init__(self):
        self.name_or_path = 'alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps'


class DummyAdapterWithConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # a conv that might have misled heuristic code in the past
        self.conv_in = torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.name_or_path = 'alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps'


class DummyInnerAsserts33(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name_or_path = 'alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps'

    def forward(self, latents, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
        # Accept either tensor-form latents or VideoX-style list-of-sample latents
        sample_latent = None
        if isinstance(latents, torch.Tensor):
            sample_latent = latents
        elif isinstance(latents, (list, tuple)) and len(latents) > 0 and isinstance(latents[0], torch.Tensor):
            sample_latent = latents[0]

        if isinstance(control_context, torch.Tensor):
            assert control_context.shape[1] == 33, f"expected 33 channels, got {control_context.shape[1]}"
        return torch.zeros_like(sample_latent)


def test_ensure_control_in_dim_sets_fallback_33():
    a = DummyAdapterNoControlDim()
    assert not hasattr(a, 'control_in_dim')
    ok = ensure_control_in_dim(a, strict=False, fallback=33)
    assert ok is True
    assert getattr(a, 'control_in_dim', None) == 33


def test_conflicting_conv_does_not_override_control_dim():
    a = DummyAdapterWithConv()
    # ensure_control_in_dim should set fallback=33 and not try to "infer" from conv
    ok = ensure_control_in_dim(a, strict=False, fallback=33)
    assert ok is True
    assert getattr(a, 'control_in_dim') == 33
    # infer_expected_in_ch must return the explicit attribute and not inspect conv modules
    assert infer_expected_in_ch(a) == 33


def test_video_x_wrapper_accepts_33_channel_control_context_after_loader_fallback():
    inner = DummyInnerAsserts33()
    # simulate loader behavior: set fallback
    ensure_control_in_dim(inner, strict=False, fallback=33)
    assert inner.control_in_dim == 33
    wrapper = VideoXControlnetWrapper(inner)
    lat = torch.randn(1, 16, 8, 8)
    ctrl33 = torch.randn(1, 33, 64, 64)
    # should not raise
    out = wrapper(lat, 0.0, ctrl33, conditioning_scale=1.0)
    assert isinstance(out, torch.Tensor)


def test_enforce_forces_33_on_conflicting_adapter():
    from toolkit.control_util import enforce_zimage_control_in_dim
    a = DummyAdapterWithConv()
    # no control_in_dim initially
    assert getattr(a, 'control_in_dim', None) is None
    modified = enforce_zimage_control_in_dim(a, expected=33, force=True)
    assert modified is True
    assert getattr(a, 'control_in_dim') == 33
    assert getattr(a, '_control_in_dim_forced', False) is True


def test_loader_sets_adapter_name_on_from_pretrained(monkeypatch, tmp_path):
    # Simulate ControlNet.from_pretrained returning an object without name_or_path
    class DummyControlNet:
        def __init__(self):
            # intentionally omit name_or_path
            self.control_in_dim = None
        def parameters(self):
            return []
        def buffers(self):
            return []

    def fake_from_pretrained(cpath, torch_dtype=None, subfolder=None):
        return DummyControlNet()

    monkeypatch.setattr('diffusers.ControlNetModel.from_pretrained', fake_from_pretrained)

    # Monkeypatch transformer loader to avoid heavy external loads
    class DummyTransformer:
        def __init__(self):
            pass
        @staticmethod
        def from_pretrained(path, subfolder=None, torch_dtype=None):
            return DummyTransformer()

    monkeypatch.setattr('diffusers.models.transformers.ZImageTransformer2DModel.from_pretrained', DummyTransformer.from_pretrained)
    # Stub tokenizer load to avoid HF network calls
    class DummyTokenizer:
        __module__ = 'transformers'
        def __init__(self):
            pass
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda *args, **kwargs: DummyTokenizer())
    # Stub VAE load to raise (non-fatal path) to avoid heavy loads
    monkeypatch.setattr('diffusers.AutoencoderKL.from_pretrained', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('no vae')))

    # Create ModelConfig instance
    from toolkit.config_modules import ModelConfig
    cfg = ModelConfig(name_or_path='dummy/base-model', controlnet_enabled=True, controlnet_name_or_path='repo/controlnet-id', controlnet_file=None, controlnet_streaming=False, controlnet_offload_strategy='none')

    from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel
    sd = ZImageModel(device='cpu', model_config=cfg, dtype='float32')

    # Call only the controlnet-loading portion by simulating relevant attributes
    # We call the code path by setting model_config and invoking load_model, which will use our monkeypatches
    sd.load_model()

    assert getattr(sd.controlnet, 'name_or_path', None) == cfg.controlnet_name_or_path

