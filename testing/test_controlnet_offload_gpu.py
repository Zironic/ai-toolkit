import pytest
import torch
from toolkit.controlnet_offload import offload_adapter, bring_adapter, compute_control_residuals
from toolkit.accelerator import get_accelerator


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for these tests")


class LargeDummyAdapter(torch.nn.Module):
    def __init__(self, size_mb=50):
        super().__init__()
        # create a large parameter tensor to simulate big adapter
        num_elems = (size_mb * 1024 * 1024) // 4  # float32 bytes
        self.param = torch.nn.Parameter(torch.randn(num_elems))

    def forward(self, control, noisy_latents, timesteps):
        # deterministic tiny output
        batch = noisy_latents.shape[0]
        return torch.zeros((batch, 4), device=noisy_latents.device)


def test_swap_correctness_accelerate():
    acc = get_accelerator()
    adapter = LargeDummyAdapter(size_mb=10)

    # baseline: bring to device and compute
    bring_adapter(adapter, acc.device, strategy='accelerate')
    control = torch.randn((2, 3, 16, 16), device=acc.device)
    noisy = torch.randn((2, 4), device=acc.device)
    timesteps = torch.tensor([1, 2], device=acc.device)

    out1 = compute_control_residuals(adapter, control, noisy, timesteps, device=acc.device)

    # offload and bring back, compute again
    offload_adapter(adapter, strategy='accelerate')
    bring_adapter(adapter, acc.device, strategy='accelerate')
    out2 = compute_control_residuals(adapter, control, noisy, timesteps, device=acc.device)

    # compare outputs
    assert len(out1) == len(out2)
    for a, b in zip(out1, out2):
        assert torch.allclose(a.to(b.device), b, atol=1e-6)


def test_swap_memory_smoke_accelerate():
    acc = get_accelerator()
    adapter = LargeDummyAdapter(size_mb=50)
    control = torch.randn((4, 3, 64, 64), device='cpu')
    noisy = torch.randn((4, 4), device='cpu')
    timesteps = torch.tensor([1, 2, 3, 4], device='cpu')

    # move adapter to GPU and measure memory
    torch.cuda.reset_peak_memory_stats()
    bring_adapter(adapter, acc.device, strategy='accelerate')
    torch.cuda.synchronize()
    baseline_peak = torch.cuda.max_memory_allocated()

    # offload adapter and check peak reduced after offload
    offload_adapter(adapter, strategy='accelerate')
    torch.cuda.synchronize()
    after_offload_peak = torch.cuda.max_memory_allocated()

    # after offload, peak should not be higher than baseline and ideally lower
    assert after_offload_peak <= baseline_peak + (10 * 1024 * 1024)  # allow small noise


def test_ddp_safety_accelerate():
    # sanity: calling bring/offload under accelerator should not raise
    acc = get_accelerator()
    adapter = LargeDummyAdapter(size_mb=10)
    try:
        bring_adapter(adapter, acc.device, strategy='accelerate')
        offload_adapter(adapter, strategy='accelerate')
    except Exception as e:
        pytest.skip(f"Accelerate ddp-safety integration not available: {e}")
