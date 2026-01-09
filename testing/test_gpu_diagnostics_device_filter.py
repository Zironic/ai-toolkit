import types
import torch
from toolkit.gpu_diagnostics import dump_vram_map


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(10))


def test_no_cuda_emits_global_module_list_only():
    root = {'adapter': DummyModule()}
    s = dump_vram_map(root, deep_scan=False, include_nvidia_smi=False)
    # CPU-only: should report the global list and not per-device headers
    assert "No CUDA devices available" in s or "All modules by size" in s
    assert "Top parameters on this device" not in s


def test_monkeypatched_cuda_device_loop(monkeypatch):
    # Enable device loop via monkeypatching CUDA helpers; no real CUDA tensors.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda dev: f"gpu{dev}")
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda dev: 123)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda dev: 456)

    root = {'adapter': DummyModule()}
    s = dump_vram_map(root, deep_scan=False, include_nvidia_smi=False)
    # Should include Device 0 header and must not crash
    assert 'Device 0:' in s
    # We have CPU-only parameters, so per-device param groups should not be emitted
    assert 'Top parameter groups on this device' not in s
    assert 'Role breakdown' not in s


def test_third_tier_grouping(monkeypatch):
    # Ensure third-tier groups are aggregated in the global listing
    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.noise_refiner = torch.nn.ModuleList([torch.nn.Linear(16, 16), torch.nn.Linear(16, 16)])

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()

    root = {'adapter': Adapter()}
    s = dump_vram_map(root, deep_scan=False, include_nvidia_smi=False)
    # Should include aggregated group adapter.inner.noise_refiner
    assert 'adapter.inner.noise_refiner' in s
    # And should not list each raw Linear weight separately at top group level
    assert 'adapter.inner.noise_refiner.0.weight' not in s

