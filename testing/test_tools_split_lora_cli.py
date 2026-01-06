import os
import tempfile
import subprocess
import sys
from pathlib import Path

import torch
from toolkit.split_lora import split_lora_state_dict

try:
    from safetensors.torch import save_file
    HAS_SAFETENSORS = True
except Exception:
    HAS_SAFETENSORS = False


def make_dummy_safetensors(path):
    sd = {}
    sd['lora_unet.down_blocks_20_resnets_0_.lora_down.weight'] = torch.ones((2,2))
    sd['lora_unet.down_blocks_30_resnets_0_.lora_down.weight'] = torch.ones((2,2))
    sd['global.alpha'] = torch.ones((1,))
    try:
        from safetensors.torch import save_file
        save_file(sd, path, {})
    except Exception:
        torch.save(sd, path)


def test_cli_split(tmp_path, monkeypatch):
    src = tmp_path / 'combined.safetensors'
    make_dummy_safetensors(str(src))
    content = tmp_path / 'content.safetensors'
    style = tmp_path / 'style.safetensors'

    # Run script
    cmd = [sys.executable, '-m', 'tools.split_lora', str(src), '--content-out', str(content), '--style-out', str(style), '--content-blocks', '20', '--style-blocks', '30']
    res = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent), capture_output=True)
    assert res.returncode == 0
    assert content.exists()
    assert style.exists()