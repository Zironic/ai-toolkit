import torch
from types import SimpleNamespace
from toolkit.masked_recon import apply_masked_recon_loss


def make_fake_sd():
    class FakeVAE:
        def decode(self, latents):
            # Return a dummy decoded image (not mutating latents)
            b, c, h, w = latents.shape
            rand_img = torch.rand((b, 3, h, w), dtype=latents.dtype)
            return SimpleNamespace(sample=rand_img)

    class FakeSD:
        def __init__(self):
            self.vae = FakeVAE()
            self.trainer = None

    return FakeSD()


def test_masked_recon_does_not_mutate_inputs():
    # prepare inputs
    b = 2
    noisy_latents = torch.randn((b, 4, 16, 16), requires_grad=True)
    imgs = torch.rand((b, 3, 16, 16))
    batch = SimpleNamespace()
    batch.unaugmented_tensor = imgs.clone()
    # simple config enabling masked recon
    train_config = SimpleNamespace(masked_recon_weight=1.0, masked_recon_type='illum', masked_recon_control_threshold=0.05,
                                   masked_recon_control_dilate=3, masked_recon_control_dilate_auto=True,
                                   masked_recon_control_dilate_scale_factor=2.0, masked_recon_control_blur=3,
                                   masked_recon_control_ref_max_size=1024, mask_preview_enabled=False)

    sd = make_fake_sd()

    noisy_before = noisy_latents.clone()
    imgs_before = imgs.clone()
    batch_before = batch.unaugmented_tensor.clone()

    loss_before = torch.tensor(0.0)
    loss_after, mloss = apply_masked_recon_loss(loss_before, train_config, sd, noisy_latents, imgs, batch, torch.float32, torch.device('cpu'))

    # assert inputs unchanged
    assert torch.allclose(noisy_latents, noisy_before), "noisy_latents was mutated"
    assert torch.allclose(imgs, imgs_before), "imgs was mutated"
    assert torch.allclose(batch.unaugmented_tensor, batch_before), "batch.unaugmented_tensor was mutated"
    # mloss may be None depending on mask detection; just ensure function ran
    assert loss_after is not None
