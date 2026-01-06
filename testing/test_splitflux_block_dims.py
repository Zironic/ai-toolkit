from toolkit.splitflux import build_splitflux_block_dims
from toolkit.kohya_lora import LoRANetwork


def test_default_block_dims_lengths():
    class DummyCfg:
        pass

    cfg = DummyCfg()
    content_bd, style_bd = build_splitflux_block_dims(cfg)
    num_total_blocks = LoRANetwork.NUM_OF_BLOCKS * 2 + 1
    assert len(content_bd) == num_total_blocks
    assert len(style_bd) == num_total_blocks


def test_default_values_at_known_indices():
    class DummyCfg:
        # Use indices inside the allowable LoRA block range for the test helper
        # num_total_blocks = LoRANetwork.NUM_OF_BLOCKS * 2 + 1
        splitflux_content_blocks = list(range(2, 12))
        splitflux_style_blocks = list(range(12, 24))
        splitflux_content_primary_rank = 64
        splitflux_spatial_rank = 32
        splitflux_style_primary_rank = 64
        splitflux_secondary_rank = 16

    cfg = DummyCfg()
    content_bd, style_bd = build_splitflux_block_dims(cfg)

    # content blocks should be set as configured
    for i in cfg.splitflux_content_blocks:
        assert content_bd[i] == cfg.splitflux_content_primary_rank

    # service check: spatial blocks should be within range and set to spatial rank if present
    for b in (30, 31):
        if b < len(content_bd):
            assert content_bd[b] == cfg.splitflux_spatial_rank
            assert style_bd[b] == cfg.splitflux_spatial_rank

    # style blocks should be set as configured
    for i in cfg.splitflux_style_blocks:
        assert style_bd[i] == cfg.splitflux_style_primary_rank

    # pick a sample early index (not in content_blocks) and ensure it isn't primary
    sample_idx = cfg.splitflux_content_blocks[0] - 1 if cfg.splitflux_content_blocks[0] > 0 else 0
    assert content_bd[sample_idx] != cfg.splitflux_content_primary_rank
    assert style_bd[sample_idx] != cfg.splitflux_style_primary_rank