import torch
from types import SimpleNamespace
from toolkit.splitflux import build_rca_combined_block_dims, freeze_unet_blocks
from toolkit.config_modules import TrainConfig


class DummyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down_blocks_1_resnets_0 = torch.nn.Conv2d(3, 3, 3)
        self.down_blocks_2_resnets_0 = torch.nn.Conv2d(3, 3, 3)
        self.down_blocks_20_resnets_0 = torch.nn.Conv2d(3, 3, 3)
        self.up_blocks_32_resnets_0 = torch.nn.Conv2d(3, 3, 3)


def test_build_rca_combined_block_dims_defaults():
    cfg = SimpleNamespace()
    # set defaults as in TrainConfig defaults
    cfg.splitflux_content_blocks = list(range(20, 30))
    cfg.splitflux_style_blocks = list(range(30, 58))
    cfg.splitflux_content_primary_rank = 64
    cfg.splitflux_spatial_rank = 32
    cfg.splitflux_style_primary_rank = 64
    cfg.splitflux_secondary_rank = 16

    combined = build_rca_combined_block_dims(cfg)
    # sample checks
    assert combined[20] == 64
    assert combined[30] == 32
    assert combined[32] == 64
    # early block has secondary rank
    assert combined[1] == cfg.splitflux_secondary_rank


def test_freeze_unet_blocks_dummy():
    unet = DummyUNet()
    frozen = freeze_unet_blocks(unet, [1, 2])
    assert len(frozen) > 0
    # verify params in those modules are frozen
    for name, p in unet.named_parameters():
        if 'down_blocks_1_resnets_0' in name or 'down_blocks_2_resnets_0' in name:
            assert not p.requires_grad
    # others remain trainable
    for name, p in unet.named_parameters():
        if 'down_blocks_20_resnets_0' in name or 'up_blocks_32_resnets_0' in name:
            assert p.requires_grad