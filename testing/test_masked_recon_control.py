import torch
import types
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class DummyVAE:
    def decode(self, z):
        return types.SimpleNamespace(sample=torch.nn.functional.interpolate(z, scale_factor=8, mode='bilinear', align_corners=False))


class DummySD:
    def __init__(self):
        self.vae = DummyVAE()


class DummyTrainer(SDTrainer):
    def __init__(self):
        # set minimal attrs
        self.sd = DummySD()
        self.device_torch = torch.device('cpu')
        self.train_config = types.SimpleNamespace()
        self.train_config.masked_recon_weight = 1.0
        self.train_config.masked_recon_type = 'control'
        self.train_config.masked_recon_control_threshold = 0.01
        self.train_config.masked_recon_control_dilate = 5
        self.train_config.masked_recon_control_blur = 3


def make_stick_figure_control(b=1, c=3, h=32, w=32):
    t = torch.zeros((b, c, h, w))
    # draw a simple T-shaped stick (vertical line + horizontal top)
    x_center = w // 2
    t[:, :, 4:28, x_center - 1:x_center + 1] = 1.0  # vertical thicker line
    t[:, :, 4:8, x_center - 6:x_center + 6] = 1.0  # top bar
    return t


def test_control_mask_and_masked_loss():
    t = DummyTrainer()
    b = 1
    noisy_latents = torch.randn((b, 4, 4, 4))
    imgs = torch.zeros((b, 3, 32, 32))
    target = imgs.clone()
    batch = types.SimpleNamespace()
    batch.unaugmented_tensor = target
    # attach control tensor mimicking openpose stick
    batch.control_tensor = make_stick_figure_control(b=b)
    batch.file_items = [types.SimpleNamespace() for _ in range(b)]

    before = torch.tensor(1.0)
    loss_after, mloss = t._compute_and_apply_masked_recon_loss(before, noisy_latents, imgs, batch, torch.float32)
    assert isinstance(loss_after, torch.Tensor)
    assert mloss is None or isinstance(mloss, torch.Tensor)
    # Ensure the masked branch didn't crash and returns a modified loss when possible
    # If mask computed, mask weight applied -> loss_after != before
    assert (loss_after != before) or (mloss is None)
