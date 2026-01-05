import torch
from toolkit.control_diagnostics import inspect_adapter, diagnose_adapter


class Dummy:
    def __init__(self):
        self.conv = torch.nn.Parameter(torch.zeros(1, 4, 3, 3))
        self.name_or_path = 'alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1'

    def modules(self):
        yield self


def test_inspect_adapter_and_diagnose(capfd):
    dummy = Dummy()
    inferred = inspect_adapter(dummy)
    # Diagnostics are disabled for automated training and inspection; helper should be inert
    assert inferred is None

    lat = torch.zeros(1, 16, 112, 84)
    ctrl = torch.zeros(1, 3, 512, 512)
    diagnose_adapter(dummy, lat, ctrl)
    captured = capfd.readouterr()
    assert captured.out == '' or '[DIAG]' not in captured.out