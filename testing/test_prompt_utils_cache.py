import os
from pathlib import Path
import torch
from toolkit.prompt_utils import PromptEmbeds


def test_prompt_embeds_save_load(tmp_path):
    # create a small prompt embeds object
    pe = PromptEmbeds(torch.randn(1, 768))
    p = tmp_path / 'pe.safetensors'
    pe.save(str(p))
    assert p.exists()
    # load back
    loaded = PromptEmbeds.load(str(p))
    assert hasattr(loaded, 'text_embeds')
    # cleanup
    p.unlink()
