from toolkit.config_modules import TrainConfig


def test_masked_recon_default_enabled():
    cfg = TrainConfig()
    assert cfg.masked_recon_weight is not None
    assert cfg.masked_recon_weight > 0, "masked_recon_weight should be >0 by default"