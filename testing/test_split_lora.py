import torch
from toolkit.split_lora import split_lora_state_dict


def make_dummy_weight():
    return torch.ones((2,2))


def test_split_lora_simple():
    sd = {}
    # keys mapping to blocks via get_block_index naming pattern
    sd['lora_unet.down_blocks_1_resnets_0_.lora_down.weight'] = make_dummy_weight()
    sd['lora_unet.down_blocks_20_resnets_0_.lora_down.weight'] = make_dummy_weight()
    sd['lora_unet.down_blocks_30_resnets_0_.lora_down.weight'] = make_dummy_weight()
    sd['lora_unet.up_blocks_32_resnets_0_.lora_down.weight'] = make_dummy_weight()
    sd['some_global.alpha'] = make_dummy_weight()

    # compute block indices using the same get_block_index logic for keys used above
    from toolkit.kohya_lora import get_block_index
    idx20 = get_block_index('down_blocks_20_resnets_0_')
    idx30 = get_block_index('down_blocks_30_resnets_0_')
    idx32 = get_block_index('up_blocks_32_resnets_0_')

    content_blocks = [idx20]
    style_blocks = [idx30, idx32]

    # sanity checks for mapping
    assert idx20 is not None
    assert idx30 is not None
    assert idx32 is not None
    assert idx20 != idx30
    assert idx32 != idx20

    c_sd, s_sd = split_lora_state_dict(sd, content_blocks, style_blocks)

    # key for block 1 should be in both because it's not in either set (early block)
    assert any(k.startswith('lora_unet.down_blocks_1_resnets_0') for k in c_sd.keys())
    assert any(k.startswith('lora_unet.down_blocks_1_resnets_0') for k in s_sd.keys())

    # block 20 -> content only (check by prefix to accommodate naming variants)
    assert any(k.startswith('lora_unet.down_blocks_20_resnets_0') for k in c_sd.keys())
    assert (any(k.startswith('lora_unet.down_blocks_20_resnets_0') for k in s_sd.keys())) == (idx20 in style_blocks)

    # block 30 -> style only (present in style_blocks)
    assert (any(k.startswith('lora_unet.down_blocks_30_resnets_0') for k in s_sd.keys())) == (idx30 in style_blocks)

    # block 32 -> style
    assert any(k.startswith('lora_unet.up_blocks_32_resnets_0') for k in s_sd.keys())

    # global key -> both
    assert 'some_global.alpha' in c_sd and 'some_global.alpha' in s_sd