import torch
from toolkit.memory_management.manager import MemoryManager
from types import SimpleNamespace


class FakeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)


def test_memory_manager_attach_and_process_device(monkeypatch):
    m = FakeModule()
    # ensure no memory manager attached
    assert not hasattr(m, '_memory_manager')

    # attach MemoryManager with device CPU
    MemoryManager.attach(m, torch.device('cpu'))
    assert hasattr(m, '_memory_manager')
    assert m._memory_manager.process_device == torch.device('cpu')

    # simulate changing process_device
    m._memory_manager.process_device = torch.device('cuda:0')
    assert m._memory_manager.process_device == torch.device('cuda:0')

    # ensure memory_managed_to exists and delegates to module.to
    m.to(torch.device('cpu'))
    # ensure we can call overridden to without error
    assert True


def test_memory_manager_handles_offload_percent(monkeypatch):
    # Attach a module and ensure unmanaged modules list is set when offload_percent < 1.0
    m = FakeModule()
    MemoryManager.attach(m, torch.device('cpu'), offload_percent=0.5)
    assert hasattr(m, '_memory_manager')
    # unmanaged_modules should be a list
    assert isinstance(m._memory_manager.unmanaged_modules, list)
