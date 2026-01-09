import os
import tempfile
import torch
from safetensors.torch import save_file, load_file
from toolkit.appearance_pose_adapt import AppearancePoseAdapter
from toolkit.train_tools import get_torch_dtype


def test_appa_save_and_load_roundtrip(tmp_path):
    appa = AppearancePoseAdapter()
    # create dummy inputs to initialize lazy convs
    down = [torch.randn(2, 8, 8, 8), torch.randn(2, 16, 4, 4)]
    mid = torch.randn(2, 32, 2, 2)
    appa.transform_residuals(down, mid)

    state = appa.state_dict()
    # prepare safe tensors
    safe_state = {k: v.detach().to('cpu', dtype=get_torch_dtype('fp16')) for k, v in state.items() if isinstance(v, torch.Tensor)}

    out_file = os.path.join(tmp_path, 'test_appa.safetensors')
    save_file(safe_state, out_file, metadata={})

    loaded = load_file(out_file, device='cpu')
    new_appa = AppearancePoseAdapter()
    # initialize to create convs
    new_appa.transform_residuals(down, mid)
    new_appa.load_state_dict(loaded, strict=False)

    # compare a few tensors
    for k in state:
        if isinstance(state[k], torch.Tensor) and k in loaded:
            assert torch.allclose(state[k].to(dtype=loaded[k].dtype), loaded[k].to(dtype=state[k].dtype), atol=1e-3, rtol=1e-3)
