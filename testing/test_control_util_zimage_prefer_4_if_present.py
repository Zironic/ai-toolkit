import torch
from types import SimpleNamespace
from toolkit.control_util import infer_expected_in_ch


def test_zimage_prefer_4_when_both_3_and_4_present():
    # adapter exposes conv_in with in_channels==3 and also has another module with in_channels==4
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

    expected = infer_expected_in_ch(adapter)
    assert expected == 4, f"Expected prefer 4 when both 3 and 4 present, got {expected}"
