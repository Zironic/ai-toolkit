import torch
import torch.nn as nn
from toolkit.gpu_diagnostics import dump_vram_map


class BigModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1024, 1024))  # ~8MB (float32)


class MidModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(256, 256))


class SmallModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(16, 16))


def test_sorted_items_present_and_ordered():
    modules = {
        'sd.unet': BigModule(),
        'sd.vae': MidModule(),
        'adapter': SmallModule(),
    }

    if torch.cuda.is_available():
        for k, m in list(modules.items()):
            try:
                modules[k] = m.to('cuda')
            except Exception:
                pass

    out = dump_vram_map(modules, deep_scan=False, include_nvidia_smi=False)
    assert 'Sorted items (desc):' in out

    # extract sizes (bytes) from the Sorted items block and ensure they are non-increasing
    started = False
    sizes = []
    for line in out.splitlines():
        if 'Sorted items (desc):' in line:
            started = True
            continue
        if started:
            if not line.strip().startswith('-'):
                break
            # line like '    - sd.unet: 10.00MB'
            parts = line.split(':')
            if len(parts) >= 2:
                size_str = parts[-1].strip()
                # convert to bytes roughly
                # supports B, KB, MB, GB
                num = float(''.join([c for c in size_str if (c.isdigit() or c == '.' )]))
                if 'KB' in size_str:
                    num_bytes = num * 1024
                elif 'MB' in size_str:
                    num_bytes = num * 1024 * 1024
                elif 'GB' in size_str:
                    num_bytes = num * 1024 * 1024 * 1024
                else:
                    num_bytes = num
                sizes.append(num_bytes)
    # ensure sizes are non-increasing
    assert sizes == sorted(sizes, reverse=True)
