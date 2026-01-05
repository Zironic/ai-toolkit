import torch
from types import SimpleNamespace
from toolkit.stable_diffusion_model import StableDiffusion


def test_predict_noise_zimage_resizes_control_images_to_latents():
    # Construct fake SD instance
    sd = SimpleNamespace()

    # Fake unet: accept calls and return object with .sample = zeros same shape as input
    class FakeUnet:
        def __call__(self, latents, timestep, cap_feats, return_dict=False, control_context=None, **kwargs):
            # Just return an object with .sample and ignore control_context
            class R: pass
            R.sample = torch.zeros_like(latents)
            return R

    sd.unet = FakeUnet()

    # Prepare latents with small spatial dims
    latents = torch.randn((1, 16, 48, 48))

    # Prepare zimage control images with larger spatial dims -> shape B,C,F,H,W
    zimage_control_images = torch.randn((1, 3, 1, 512, 512))

    # Fake controlnet that asserts the control_context spatial dims equal latents'
    class FakeControlNet:
        def __call__(self, sample, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
            # control_context may be tensor [B,C,H,W] or list of such
            if isinstance(control_context, torch.Tensor):
                h, w = int(control_context.shape[-2]), int(control_context.shape[-1])
                assert h == 48 and w == 48, f"control_context dims {h}x{w} != 48x48"
            elif isinstance(control_context, (list, tuple)):
                for t in control_context:
                    h, w = int(t.shape[-2]), int(t.shape[-1])
                    assert h == 48 and w == 48
            # return dummy residuals
            return torch.zeros((1, 4, 48, 48))

    fakecn = FakeControlNet()

    # Minimal text embeddings placeholder
    text_embeddings = torch.zeros((1, 1, 16))

    # Provide a fake encoder so the function can auto-encode raw control images
    def fake_encode(imgs, tile=False, tile_size=None, overlap=None):
        # return batched latents tensor [B, C, H, W] matching expected 48x48
        encoded = []
        for img in imgs:
            encoded.append(torch.zeros((1, 4, 48, 48)))
        return torch.cat(encoded, dim=0)

    sd.encode_control_images = fake_encode
    # ensure model_config attrs used by encoding path exist
    sd.model_config = SimpleNamespace(control_use_tiling=False, control_tiling_size=256, control_tiling_overlap=32)

    # Call the static method (bind to our fake sd instance)
    func = StableDiffusion._predict_noise_zimage
    # If the function returns without assertion, resizing happened correctly
    out = func(sd, latents, text_embeddings, torch.tensor([1.0]), zimage_controlnet=fakecn, zimage_control_images=zimage_control_images, zimage_conditioning_scale=1.0)

    assert out is not None


def test_predict_noise_zimage_resizes_control_latents_to_latents():
    """When control images are already encoded as latents, ensure control_context is resized
    to match the noisy latents spatial dims before calling the ControlNet."""
    sd = SimpleNamespace()

    class FakeUnet:
        def __call__(self, latents, timestep, cap_feats, return_dict=False, control_context=None, **kwargs):
            class R: pass
            R.sample = torch.zeros_like(latents)
            return R

    sd.unet = FakeUnet()

    # latents (noisy) have 64x64 spatial dims
    latents = torch.randn((1, 16, 64, 64))

    # control latents encoded at 60x60 (mismatch) - simulating VAE encoding/reassembly
    zimage_control_latents = torch.randn((1, 4, 60, 60))

    class FakeControlNet:
        def __call__(self, sample, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
            # Ensure we received control_context resized to 64x64
            if isinstance(control_context, torch.Tensor):
                h, w = int(control_context.shape[-2]), int(control_context.shape[-1])
                assert h == 64 and w == 64, f"control_context dims {h}x{w} != 64x64"
            elif isinstance(control_context, (list, tuple)):
                for t in control_context:
                    h, w = int(t.shape[-2]), int(t.shape[-1])
                    assert h == 64 and w == 64
            return torch.zeros((1, 4, 64, 64))

    fakecn = FakeControlNet()

    text_embeddings = torch.zeros((1, 1, 16))

    # Call with control latents directly (no auto-encode step)
    func = StableDiffusion._predict_noise_zimage
    out = func(sd, latents, text_embeddings, torch.tensor([1.0]), zimage_controlnet=fakecn, zimage_control_images=zimage_control_latents, zimage_conditioning_scale=1.0)

    assert out is not None
