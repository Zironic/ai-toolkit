import torch
from types import SimpleNamespace
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess

class FakeListSensitiveControlNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.control_in_dim = 4
        self.in_channels = 4
        self.p = torch.nn.Parameter(torch.tensor([0.0]))
    def forward(self, sample, timestep=None, encoder_hidden_states=None, control_context=None, controlnet_cond=None, **kwargs):
        # accept either VideoX name `control_context` or legacy `controlnet_cond` and prefer the former
        control = control_context if control_context is not None else controlnet_cond
        if isinstance(control, torch.Tensor):
            c = control.shape[1] if control.ndim >= 2 else None
            if c is not None and c != 3:
                raise RuntimeError(f"Given groups=1, weight of size [16, 3, 3, 3], expected input[1, {c}, 32, 32] to have 3 channels, but got {c} channels instead")
            return torch.zeros(1)
        return self.conv_in(control)

dummy = SimpleNamespace()
dummy.print_and_status_update = lambda msg: print('LOG:',msg)
dummy.sd = SimpleNamespace()
dummy.sd.is_controlnet_enabled = True
fake = FakeListSensitiveControlNet()
for p in fake.parameters():
    p.requires_grad = False
dummy.sd.controlnet = fake
dummy.dataset_configs = [SimpleNamespace(control_type='pose')]

try:
    BaseSDTrainProcess.setup_controlnet_training(dummy)
    print('No exception')
except Exception as e:
    import traceback
    traceback.print_exc()
