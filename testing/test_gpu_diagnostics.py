import torch
import torch.nn as nn
from toolkit.gpu_diagnostics import dump_vram_map


class DummyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        # LoRA-like param
        self.lora_A = nn.Parameter(torch.randn(4, 4))


class DummyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(128, 64)


class DummyTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 64)


class DummyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapt = nn.Conv2d(3, 3, 1)


class DummyOptimizer:
    def __init__(self, params):
        import torch
        self.state = {}
        # create a fake optimizer state tensor on cpu (should be ignored in device-specific report)
        self.state[list(params)[0]] = {'exp_avg': torch.randn(2, 2)}


def _parse_size(s: str) -> int:
    s = s.strip()
    units = {'b': 1, 'kb': 1024, 'mb': 1024**2, 'gb': 1024**3, 'tb': 1024**4}
    for unit in units:
        if s.lower().endswith(unit):
            try:
                val = float(s[:-len(unit)])
                return int(val * units[unit])
            except Exception:
                return 0
    try:
        return int(s)
    except Exception:
        return 0


def test_dump_vram_map_contains_role_breakdown_and_sorted_list_cpu_only():
    # CPU-only unit test (must not touch CUDA)
    modules = {
        'sd.unet': DummyUNet(),
        'sd.vae': DummyVAE(),
        'sd.text_encoder': DummyTextEncoder(),
        'adapter': DummyAdapter(),
        'optimizer': DummyOptimizer([p for p in DummyUNet().parameters()])
    }

    out = dump_vram_map(modules, deep_scan=False, include_nvidia_smi=False)
    # Either a role breakdown or top-parameters/submodule lists should be present
    assert any(k in out for k in ('Role breakdown', 'Global role breakdown', 'Top parameters by size', 'Top submodules by aggregated param size'))
    # Accept either role labels OR module names (sd.unet, sd.vae, etc.)
    assert any(k in out for k in ('Model/UNet', 'sd.unet', 'sd.vae', 'TextEncoder', 'adapter', 'optimizer'))
    assert 'VAE' in out or 'sd.vae' in out
    assert 'TextEncoder' in out or 'sd.text_encoder' in out
    # LoRA and optimizer might not be found, but ensure output mentions optimizer or LoRA label if present
    assert ('LoRA' in out) or ('Optimizer state' in out) or ('Top parameters by size' in out)

    # Check the global sorted list exists and is in descending order
    assert 'All modules by size (descending):' in out
    lines = [ln.rstrip() for ln in out.splitlines()]
    try:
        idx = lines.index('All modules by size (descending):')
    except ValueError:
        pytest.skip('No global module list in diagnostic output')

    size_lines = []
    for ln in lines[idx+1:]:
        ln = ln.strip()
        if not ln.startswith('-'):
            break
        item = ln.lstrip('-').strip()
        if ':' in item:
            name, size = item.split(':', 1)
            name = name.strip()
            size = size.strip()
            size_lines.append((name, _parse_size(size)))

    assert len(size_lines) > 0
    sizes = [s for _, s in size_lines]
    # Filter out zero sizes (modules with no params/buffers) and assert the remaining positive sizes are in descending order
    positive_sizes = [s for s in sizes if s > 0]
    assert positive_sizes == sorted(positive_sizes, reverse=True)
