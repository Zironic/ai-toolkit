import torch
from types import SimpleNamespace
from toolkit.control_util import infer_expected_in_ch


def test_zimage_prefer_explicit_control_in_dim_only():
    # With strict policy we only honor an explicit `control_in_dim` attribute.
    adapter = SimpleNamespace()
    adapter.name_or_path = 'alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps'
    # conv_in field indicates 3
    adapter.conv_in = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

    # Add a module with 4-channel conv (simulate inner modules)
    class FakeModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1)

    adapter._fake = FakeModule()

    # Without explicit control_in_dim we return None
    expected = infer_expected_in_ch(adapter)
    assert expected is None, f"Expected None when no explicit control_in_dim provided, got {expected}"

    # When explicit attribute is set, it should be returned
    adapter.control_in_dim = 4
    expected2 = infer_expected_in_ch(adapter)
    assert expected2 == 4, f"Expected explicit control_in_dim to be returned, got {expected2}"
