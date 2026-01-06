import os
import json
import tempfile
import torch

from toolkit.split_lora import save_split_lora_from_file, split_lora_state_dict

try:
    from safetensors.torch import save_file, load_file
    from safetensors import safe_open
    HAS_SAFETENSORS = True
except Exception:
    HAS_SAFETENSORS = False


def make_dummy_sd(path):
    sd = {}
    sd['lora_unet.down_blocks_20_resnets_0_.lora_down.weight'] = torch.ones((2,2))
    sd['lora_unet.down_blocks_30_resnets_0_.lora_down.weight'] = torch.ones((2,2))
    sd['global.alpha'] = torch.ones((1,))
    save_file(sd, path, {})


def test_save_split_lora_metadata_safetensors(tmp_path):
    if not HAS_SAFETENSORS:
        return
    src = tmp_path / "combined.safetensors"
    make_dummy_sd(str(src))

    content_out = tmp_path / "content.safetensors"
    style_out = tmp_path / "style.safetensors"

    # content blocks = [20], style blocks = [30]
    save_split_lora_from_file(str(src), str(content_out), str(style_out), [20], [30], metadata={'job':'test'}, dtype=None)

    # validate metadata via safe_open
    with safe_open(str(content_out), framework="pt") as f:
        meta = f.metadata()
        assert meta.get('split_type') == 'content'
        assert 'blocks' in meta
        assert meta.get('source_file') == os.path.basename(str(src))

    with safe_open(str(style_out), framework="pt") as f:
        meta = f.metadata()
        assert meta.get('split_type') == 'style'
        assert 'blocks' in meta


def test_save_split_lora_metadata_torch(tmp_path):
    # Write a torch file as source
    src = tmp_path / "combined.pt"
    sd = {}
    sd['lora_unet.down_blocks_20_resnets_0_.lora_down.weight'] = torch.ones((2,2))
    sd['lora_unet.down_blocks_30_resnets_0_.lora_down.weight'] = torch.ones((2,2))
    sd['global.alpha'] = torch.ones((1,))
    torch.save(sd, str(src))

    content_out = tmp_path / "content.pt"
    style_out = tmp_path / "style.pt"

    save_split_lora_from_file(str(src), str(content_out), str(style_out), [20], [30], metadata={'job':'test_torch'}, dtype=None)

    # check sidecar metadata files exist
    with open(str(content_out) + '.meta.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
        assert meta['split_type'] == 'content'
        assert meta['source_file'] == os.path.basename(str(src))

    with open(str(style_out) + '.meta.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
        assert meta['split_type'] == 'style'