import torch
import pytest
from toolkit.controlnet_offload import offload_adapter, bring_adapter, compute_control_residuals


class DummyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # small parameter so device moves are visible
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, control, noisy_latents, timesteps):
        # simple deterministic operation for tests
        x = control.mean(dim=[1, 2, 3]) if control.dim() == 4 else control.mean()
        # broadcast to batch x channels x h x w shape like a residual
        batch = noisy_latents.shape[0]
        residual = torch.ones((batch, 4), device=noisy_latents.device) * x.unsqueeze(-1)
        return residual


def test_manual_swap_moves_to_cpu():
    adapter = DummyAdapter().to(torch.device('cpu'))
    # ensure on CPU initially
    for p in adapter.parameters():
        assert p.device.type == 'cpu'

    # simulate bringing to a device then offloading
    bring_adapter(adapter, torch.device('cpu'), strategy='none')
    offload_adapter(adapter, strategy='manual_swap')

    for p in adapter.parameters():
        assert p.device.type == 'cpu'


def test_bring_adapter_to_device():
    adapter = DummyAdapter().to(torch.device('cpu'))
    bring_adapter(adapter, torch.device('cpu'), strategy='manual_swap')
    for p in adapter.parameters():
        assert p.device.type == 'cpu'


def test_accelerate_offload_moves_to_accelerator_device():
    from toolkit.accelerator import get_accelerator
    acc = get_accelerator()
    adapter = DummyAdapter().to(torch.device('cpu'))
    # offload to accelerator (on CPU env this will keep it on CPU)
    offload_adapter(adapter, strategy='accelerate')
    for p in adapter.parameters():
        # accelerator.device may omit an index (e.g. 'cuda' vs 'cuda:0'); compare by device type
        assert p.device.type == acc.device.type


def test_bring_adapter_with_accelerate():
    from toolkit.accelerator import get_accelerator
    acc = get_accelerator()
    adapter = DummyAdapter().to(torch.device('cpu'))
    bring_adapter(adapter, acc.device, strategy='accelerate')
    for p in adapter.parameters():
        # accelerator.device may omit an index (e.g. 'cuda' vs 'cuda:0'); compare by device type
        assert p.device.type == acc.device.type

def test_compute_control_residuals_cpu():
    adapter = DummyAdapter().to(torch.device('cpu'))
    batch = 2
    control = torch.randn((batch, 3, 16, 16))
    noisy = torch.randn((batch, 4))
    timesteps = torch.tensor([10, 20])

    residuals = compute_control_residuals(adapter, control, noisy, timesteps, device=torch.device('cpu'), residual_storage='gpu')
    assert isinstance(residuals, tuple)
    assert len(residuals) == 1
    r = residuals[0]
    assert r.device.type in ('cpu', 'meta') or r.device.type == 'cpu'
    assert r.shape[0] == batch


@pytest.mark.parametrize('storage', ['gpu', 'cpu_pinned'])
def test_compute_control_residuals_storage(storage):
    adapter = DummyAdapter().to(torch.device('cpu'))
    batch = 1
    control = torch.randn((batch, 3, 8, 8))
    noisy = torch.randn((batch, 4))
    timesteps = torch.tensor([5])

    residuals = compute_control_residuals(adapter, control, noisy, timesteps, device=torch.device('cpu'), residual_storage=storage)
    assert len(residuals) == 1
    r = residuals[0]
    assert r.requires_grad is False
    if storage == 'cpu_pinned':
        # cpu pinned yields pinned memory tensor
        assert r.is_pinned()
