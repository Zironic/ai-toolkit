import torch
import pytest
from toolkit.prompt_utils import PromptEmbeds, expand_prompt_embeds_to_batch, build_per_block_prompt_map


def make_pe(batch, seq, dim=16):
    t = torch.randn(batch, seq, dim)
    return PromptEmbeds(t)


def test_expand_prompt_embeds_to_batch_expands():
    pe = make_pe(1, 8, dim=32)
    out = expand_prompt_embeds_to_batch(pe, 3)
    assert out.text_embeds.shape[0] == 3


def test_build_per_block_prompt_map_success():
    batch_size = 2
    cond = make_pe(batch_size, 8, dim=32)
    split = make_pe(1, 8, dim=32)
    blank = make_pe(1, 8, dim=32)

    per_block = build_per_block_prompt_map(cond, split, blank, batch_size)

    # check content block mapping
    assert 20 in per_block
    assert per_block[20].text_embeds.shape[0] == batch_size

    # blank blocks
    assert per_block[30].text_embeds.shape[0] == batch_size

    # style blocks (e.g., 32) use split prompt expanded
    assert per_block[32].text_embeds.shape[0] == batch_size


def test_build_per_block_prompt_map_missing_split_raises():
    batch_size = 1
    cond = make_pe(batch_size, 8, dim=32)
    split = None
    blank = make_pe(1, 8, dim=32)

    with pytest.raises(RuntimeError):
        build_per_block_prompt_map(cond, split, blank, batch_size)


def test_build_per_block_prompt_map_seq_len_mismatch_raises():
    batch_size = 2
    cond = make_pe(batch_size, 8, dim=32)
    split = make_pe(1, 10, dim=32)  # mismatch seq len
    blank = make_pe(1, 8, dim=32)

    with pytest.raises(RuntimeError):
        build_per_block_prompt_map(cond, split, blank, batch_size)
