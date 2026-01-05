import torch
from types import SimpleNamespace
from toolkit.stable_diffusion_model import StableDiffusion


def test_reassemble_uses_per_image_sizes():
    sd = SimpleNamespace()

    # fake unet
    class FakeUnet:
        def __call__(self, latents, timestep, cap_feats, return_dict=False, control_context=None, **kwargs):
            class R: pass
            R.sample = torch.zeros_like(latents)
            return R
    sd.unet = FakeUnet()

    # fake controlnet
    class FakeControlNet:
        def __call__(self, sample, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
            return torch.zeros((1, 4, sample.shape[2], sample.shape[3]))
    fakecn = FakeControlNet()

    # Prepare two control images of different sizes
    img1 = torch.randn((3, 512, 512))
    img2 = torch.randn((3, 768, 600))
    # Use a list-style control_images to simulate varied sizes (VideoX accepts lists)
    zimage_control_images = [img1, img2]

    # Track reassemble calls
    called = {'sizes': []}
    def fake_reassemble(tile_latents, full_size, latent_downsample=1):
        called['sizes'].append(full_size)
        # return a dummy latents tensor
        return torch.zeros((1, 4, 48, 48))

    sd.reassemble_tile_latents = fake_reassemble

    # fake encoder returns per-image tile lists (single tile using original pixel dims)
    def fake_encode(imgs, tile=False, tile_size=None, overlap=None):
        res = []
        for img in imgs:
            h, w = int(img.shape[-2]), int(img.shape[-1])
            # fake one tile: (latent, pos, (w,h))
            res.append([(torch.zeros((1,4,8,8)), (0,0), (w,h))])
        return res

    sd.encode_control_images = fake_encode
    sd.model_config = SimpleNamespace(control_use_tiling=True, control_tiling_size=256, control_tiling_overlap=32)

    text_embeddings = torch.zeros((1,1,16))
    latents = torch.randn((1, 16, 48, 48))

    func = StableDiffusion._predict_noise_zimage
    out = func(sd, latents, text_embeddings, torch.tensor([1.0]), zimage_controlnet=fakecn, zimage_control_images=zimage_control_images, zimage_conditioning_scale=1.0)

    assert (600,512) in called['sizes'] or (512,512) in called['sizes'] or (768,600) in called['sizes']
    # precisely check matches: our fake_encode produced (w,h) pairs above
    assert (512,512) in called['sizes']
    assert (600,768) in [(s[0], s[1]) for s in called['sizes']] or (768,600) in called['sizes']
    assert out is not None
