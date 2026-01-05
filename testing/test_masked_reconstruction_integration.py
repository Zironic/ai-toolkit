import torch
import types
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class DummyVAE:
    def decode(self, z):
        # z shape (B, C, h, w) -> upsample by 8
        return types.SimpleNamespace(sample=torch.nn.functional.interpolate(z, scale_factor=8, mode='bilinear', align_corners=False))


class DummySD:
    def __init__(self):
        self.vae = DummyVAE()


class DummyTrainer(SDTrainer):
    def __init__(self):
        # don't call super constructor heavy work; just set minimal attrs
        self.sd = DummySD()
        self.device_torch = torch.device('cpu')
        # minimal train_config
        self.train_config = types.SimpleNamespace()
        self.train_config.masked_recon_weight = 1.0
        self.train_config.masked_recon_type = 'illum'
        self.train_config.masked_recon_mask_key = None


def test_compute_and_apply_masked_recon_loss_runs():
    t = DummyTrainer()
    b = 2
    # noisy latents shape (B,4,4,4) -> decoded to (B,4,32,32)
    noisy_latents = torch.randn((b, 4, 4, 4))
    # imgs: simulated input images
    imgs = torch.zeros((b, 3, 32, 32))
    # make target with small bright region
    target = imgs.clone()
    target[:, :, 12:20, 12:20] = 1.0
    batch = types.SimpleNamespace()
    batch.unaugmented_tensor = target
    batch.file_items = [types.SimpleNamespace() for _ in range(b)]

    loss_before = torch.tensor(1.0)
    loss_after, mloss = t._compute_and_apply_masked_recon_loss(loss_before, noisy_latents, imgs, batch, torch.float32)
    assert isinstance(loss_after, torch.Tensor)
    assert (mloss is None) or (isinstance(mloss, torch.Tensor))


def test_masked_recon_produces_gradients_on_noisy_latents():
    t = DummyTrainer()
    b = 2
    noisy_latents = torch.randn((b, 4, 4, 4), requires_grad=True)
    imgs = torch.zeros((b, 3, 32, 32))
    target = imgs.clone()
    target[:, :, 12:20, 12:20] = 1.0
    batch = types.SimpleNamespace()
    batch.unaugmented_tensor = target
    batch.file_items = [types.SimpleNamespace() for _ in range(b)]

    loss_before = torch.tensor(1.0, requires_grad=True)
    loss_after, mloss = t._compute_and_apply_masked_recon_loss(loss_before, noisy_latents, imgs, batch, torch.float32)
    assert isinstance(loss_after, torch.Tensor)
    assert isinstance(mloss, torch.Tensor)

    # Backprop and assert grad on noisy_latents
    loss_after.backward()
    assert noisy_latents.grad is not None and noisy_latents.grad.abs().sum() > 0

