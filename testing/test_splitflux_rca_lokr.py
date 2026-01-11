import pytest
from toolkit.splitflux import build_rca_combined_block_dims, build_splitflux_block_dims


class SimpleNetworkConfig:
    def __init__(self, **kwargs):
        self.type = kwargs.get('type', 'lokr')
        self.linear = kwargs.get('linear', 64)
        self.linear_alpha = kwargs.get('linear_alpha', None)


class DummyTrainConfig:
    pass


def test_rca_combined_block_dims_for_lokr():
    train_config = DummyTrainConfig()
    # use defaults for splitflux blocks
    # Case 1: normal LoKr settings
    net_cfg = SimpleNetworkConfig(**{"type": "lokr", "linear": 64, "linear_alpha": 48})

    combined_dims, combined_alphas = build_rca_combined_block_dims(train_config, network_config=net_cfg)

    # first 19 blocks (0..18) should be zero dims and zero alphas
    for i in range(0, 19):
        assert combined_dims[i] == 0, f"block {i} expected dim 0, got {combined_dims[i]}"
        assert combined_alphas[i] == 0, f"block {i} expected alpha 0, got {combined_alphas[i]}"

    # content blocks default are 20-29 (1-based); internal indices should be 19..28 and set to job_size (64)
    for i in range(19, 29):
        assert combined_dims[i] == 64, f"content block {i} expected dim 64, got {combined_dims[i]}"
        assert combined_alphas[i] == 48, f"content block {i} expected alpha 48, got {combined_alphas[i]}"

    # spatial blocks should be indices 29 and 30 (0-based)
    assert combined_dims[29] == 32
    assert combined_dims[30] == 32
    assert combined_alphas[29] == 24
    assert combined_alphas[30] == 24

    # style blocks default are 30-57 (1-based) -> indices 29..56; style_primary alpha should be job_alpha (48)
    for i in range(31, 57):
        assert combined_dims[i] >= 32, f"style block {i} expected >=32, got {combined_dims[i]}"
        assert combined_alphas[i] >= 24, f"style block {i} expected alpha >=24, got {combined_alphas[i]}"

    # Case 2: lokr_full_rank True - linear may be set to huge sentinel, but rank/linear_alpha should be used
    net_cfg2 = SimpleNetworkConfig(**{"type": "lokr", "linear": 9999999999, "linear_alpha": 9999999999})
    net_cfg2.lokr_full_rank = True
    # original intended values are available in 'rank' and 'linear_alpha'
    net_cfg2.rank = 32
    net_cfg2.linear_alpha = 32

    combined_dims2, combined_alphas2 = build_rca_combined_block_dims(train_config, network_config=net_cfg2)

    # content blocks should pick up the original rank/alpha values
    for i in range(19, 29):
        assert combined_dims2[i] == 32, f"content block {i} expected dim 32, got {combined_dims2[i]}"
        assert combined_alphas2[i] == 32, f"content block {i} expected alpha 32, got {combined_alphas2[i]}"

    # spatial block alphas should be half of 32 (indices 29 and 30)
    assert combined_alphas2[29] == 16, f"spatial block 29 expected alpha 16, got {combined_alphas2[29]}"
    assert combined_alphas2[30] == 16, f"spatial block 30 expected alpha 16, got {combined_alphas2[30]}"