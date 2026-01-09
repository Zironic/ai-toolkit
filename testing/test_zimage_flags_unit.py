import torch
from types import SimpleNamespace
from toolkit.stable_diffusion_model import StableDiffusion


def test_predict_noise_zimage_sets_flags_unit():
    # Fake ControlNet that returns a single-tensor control_hints
    def fake_cn(sample, timestep, control_context, conditioning_scale=1.0, **kwargs):
        # ensure adapter kwarg was passed
        assert kwargs.get('controlnet') is fake_cn
        # return a simple tensor matching latent shape
        return torch.zeros((sample.shape[0], sample.shape[1], sample.shape[2], sample.shape[3]))

    # minimal sd_like with a unet that accepts control_context
    sd_like = SimpleNamespace()
    def fake_unet(latents, timestep, cap_feats, return_dict=False, control_context=None, **kwargs):
        # record that control_context was received
        sd_like._unet_received_control_context = control_context
        return torch.zeros((latents.shape[0], 4, latents.shape[2], latents.shape[3]))
    sd_like.unet = fake_unet

    # Prepare inputs
    latents = torch.zeros((1,4,16,16))
    text_embeddings = SimpleNamespace(); text_embeddings.text_embeds = torch.zeros((1,1,128))
    timestep = torch.tensor([10])
    zimage_ctrl = torch.randn(1,4,1,16,16)

    # Call helper
    StableDiffusion._predict_noise_zimage(sd_like, latents, text_embeddings, timestep, zimage_controlnet=fake_cn, zimage_control_images=zimage_ctrl, zimage_conditioning_scale=1.0)

    # Validate flags set on the helper call
    assert getattr(sd_like, '_last_zimage_control_hints_present', False) is True
    assert getattr(sd_like, '_last_zimage_control_hints_shapes', None) is not None
    assert getattr(sd_like, '_last_zimage_control_context_passed', False) is True
    assert getattr(sd_like, '_last_zimage_control_context_shape', None) is not None


def test_predict_noise_routes_and_sets_flags():
    # Test predict_noise routing with zimage kwargs
    sd = StableDiffusion.__new__(StableDiffusion)  # minimal instance

    # Provide minimal attributes used by predict_noise and helper
    sd.unet = lambda *a, **k: torch.zeros((a[0].shape[0], 4, a[0].shape[2], a[0].shape[3]))
    sd.is_xl = False
    sd.device_torch = torch.device('cpu')
    sd.torch_dtype = torch.float32

    # create a fake controlnet function
    def fake_cn(sample, timestep, control_context, conditioning_scale=1.0):
        return torch.zeros((sample.shape[0], sample.shape[1], sample.shape[2], sample.shape[3]))

    latents = torch.zeros((1,4,16,16))
    text_embeddings = SimpleNamespace(); text_embeddings.text_embeds = torch.zeros((1,1,128))
    timestep = torch.tensor([10])
    zimage_ctrl = torch.randn(1,4,1,16,16)

    # Call predict_noise with zimage kwargs
    sd.predict_noise(latents, text_embeddings, timestep, zimage_controlnet=fake_cn, zimage_control_images=zimage_ctrl, zimage_conditioning_scale=1.0)

    # After call, the model flags should be set by _predict_noise_zimage
    assert getattr(sd, '_last_predict_detected_zimage', False) is True
    assert getattr(sd, '_last_zimage_control_hints_present', False) is True
    assert getattr(sd, '_last_zimage_control_context_passed', False) is True
