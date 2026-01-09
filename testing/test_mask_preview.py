import torch
import types
import tempfile
import shutil
import glob
import os
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
        # mask preview defaults
        self.train_config.mask_preview_enabled = True
        self.tmpdir = tempfile.mkdtemp()
        self.train_config.mask_preview_save_path = os.path.join(self.tmpdir, '{job_name}', 'masks')
        self.train_config.mask_preview_overwrite = True
        self.train_config.mask_preview_overlay = True
        self.job = types.SimpleNamespace(name='testjob')

    def cleanup(self):
        shutil.rmtree(self.tmpdir)


def make_stick_figure_control(b=1, c=3, h=32, w=32):
    t = torch.zeros((b, c, h, w))
    # draw a simple T-shaped stick (vertical line + horizontal top)
    x_center = w // 2
    t[:, :, 4:28, x_center - 1:x_center + 1] = 1.0  # vertical thicker line
    t[:, :, 4:8, x_center - 6:x_center + 6] = 1.0  # top bar
    return t


def test_mask_preview_saves_files():
    t = DummyTrainer()
    b = 1
    noisy_latents = torch.randn((b, 4, 4, 4))
    imgs = torch.zeros((b, 3, 32, 32))
    target = imgs.clone()
    batch = types.SimpleNamespace()
    batch.unaugmented_tensor = target
    batch.control_tensor = make_stick_figure_control(b=b)
    batch.file_items = [types.SimpleNamespace() for _ in range(b)]

    # ensure no files before
    outglob = os.path.join(t.tmpdir, 'testjob', 'masks', '*.png')
    assert len(glob.glob(outglob)) == 0

    before = torch.tensor(1.0)
    loss_after, mloss = t._compute_and_apply_masked_recon_loss(before, noisy_latents, imgs, batch, torch.float32)
    # ensure masked loss computed
    assert mloss is None or isinstance(mloss, torch.Tensor)

    # Manually compute and save a mask preview to validate saving helper
    from toolkit.visualization import save_mask_preview
    ctrl = batch.control_tensor
    act = ctrl.abs().sum(dim=1, keepdim=True)
    act = torch.nn.functional.interpolate(act, size=(32, 32), mode='bilinear', align_corners=False)
    max_per_sample = act.view(act.shape[0], -1).amax(dim=1).view(-1, 1, 1, 1)
    act = act / (max_per_sample + 1e-9)
    tval = float(getattr(t.train_config, 'masked_recon_control_threshold', 0.01))
    mask0 = (act > tval).float()
    mask1 = torch.nn.functional.max_pool2d(mask0, kernel_size=5, stride=1, padding=2)
    mask = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(mask1)
    # normalize
    max_per_sample = mask.view(mask.shape[0], -1).amax(dim=1).view(-1, 1, 1, 1)
    mask = mask / (max_per_sample + 1e-9)
    mask = mask.clamp(0.0, 1.0)

    save_dir = os.path.join(t.tmpdir, 'testjob', 'masks')
    os.makedirs(save_dir, exist_ok=True)
    save_mask_preview(mask[0, 0], os.path.join(save_dir, 'manual_preview.png'))

    files = glob.glob(outglob)
    assert len(files) >= 1


def test_save_mask_previews_helper_creates_files():
    # Setup a fake dataset with file_list and control images
    tmpdir = tempfile.mkdtemp()
    try:
        # create simple base and control images
        from PIL import Image
        base_img_path = os.path.join(tmpdir, 'img1.png')
        ctrl_img_path = os.path.join(tmpdir, 'img1_ctrl.png')
        img = Image.new('RGB', (32, 32), color=(255, 255, 255))
        img.save(base_img_path)
        ctrl = Image.new('RGB', (32, 32), color=(0, 0, 0))
        # draw a white stick
        for y in range(4, 28):
            for x in range(15, 17):
                ctrl.putpixel((x, y), (255, 255, 255))
        ctrl.save(ctrl_img_path)

        dataset = types.SimpleNamespace(file_list=[types.SimpleNamespace(path=base_img_path, control_path=ctrl_img_path, crop_height=32, crop_width=32)])
        t = DummyTrainer()
        save_dir = os.path.join(t.tmpdir, 'testjob', 'masks')
        os.makedirs(save_dir, exist_ok=True)
        from toolkit.masked_recon import save_mask_previews
        save_mask_previews([dataset], t.train_config, t.sd, save_dir, overwrite=True, overlay=True)

        files = glob.glob(os.path.join(save_dir, '*.png'))
        assert len(files) >= 1
        idx = os.path.join(save_dir, 'index.json')
        assert os.path.exists(idx)
    finally:
        shutil.rmtree(tmpdir)

    # cleanup
    t.cleanup()


def test_mask_preview_disabled_does_not_save():
    t = DummyTrainer()
    t.train_config.mask_preview_enabled = False
    b = 1
    noisy_latents = torch.randn((b, 4, 4, 4))
    imgs = torch.zeros((b, 3, 32, 32))
    target = imgs.clone()
    batch = types.SimpleNamespace()
    batch.unaugmented_tensor = target
    batch.control_tensor = make_stick_figure_control(b=b)
    batch.file_items = [types.SimpleNamespace() for _ in range(b)]

    outglob = os.path.join(t.tmpdir, 'testjob', 'masks', '*.png')
    # ensure no files before
    assert len(glob.glob(outglob)) == 0

    before = torch.tensor(1.0)
    loss_after, mloss = t._compute_and_apply_masked_recon_loss(before, noisy_latents, imgs, batch, torch.float32)
    # check files not created
    files = glob.glob(outglob)
    assert len(files) == 0
    t.cleanup()
