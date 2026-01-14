import pytest
pytest.importorskip("torch")
from types import SimpleNamespace
from toolkit.stable_diffusion_model import StableDiffusion


def test_predict_noise_zimage_uses_cached_context_and_skips_encode():
    # Construct minimal StableDiffusion instance without heavy init
    sd = StableDiffusion.__new__(StableDiffusion)

    # Simple timer context manager used by the method; provide a no-op
    class DummyTimer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    sd.timer = DummyTimer()

    # Create a dummy unet that records kwargs passed
    recorded = {}

    class DummyUNet:
        def __call__(self, *args, **kwargs):
            recorded['args'] = args
            recorded['kwargs'] = kwargs
            # return a tensor shaped like latents
            lat = args[0]
            return torch.zeros_like(lat)

    sd.unet = DummyUNet()

    # Make encode_control_images raise if called (we should not call it)
    def bogus_encode(imgs, tile=False, tile_size=None, overlap=None):
        raise AssertionError("encode_control_images was called unexpectedly")

    sd.encode_control_images = bogus_encode

    # Prepare inputs: latents, text_embeddings (simple tensor), timestep
    B, C, H, W = 1, 4, 32, 32
    latents = torch.zeros((B, C, H, W))
    text_embeds = torch.zeros((B, 77, 768))
    timestep = torch.tensor([10.0])

    # Provide a precomputed control_context (batched latents-like tensor)
    precomputed_ctx = torch.randn((B, 33, H, W))  # must be 33 channels for strict mode

    # Create a dummy controlnet that records whether it was called
    ctrl_called = {}

    class DummyControlNet:
        def __call__(self, *args, **kwargs):
            ctrl_called['args'] = args
            ctrl_called['kwargs'] = kwargs
            return None

    zcn = DummyControlNet()

    # Call the helper with explicit precomputed context kwarg
    with pytest.raises(RuntimeError):
        sd._predict_noise_zimage(latents=latents, text_embeddings=text_embeds, timestep=timestep,
                                 zimage_controlnet=zcn, zimage_control_images=None,
                                 zimage_conditioning_scale=1.0, zimage_control_context=precomputed_ctx)

    # Ensure encode_control_images was NOT called (bogus_encode would raise if invoked)
    # Ensure the controlnet was called with our precomputed context
    assert 'args' in ctrl_called and len(ctrl_called['args']) >= 3
    assert torch.allclose(ctrl_called['args'][2], precomputed_ctx)

    # Check diagnostic flag on sd for hints presence remains False
    assert getattr(sd, '_last_zimage_control_hints_present', False) is False
