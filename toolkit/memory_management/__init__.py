# SPDX-License-Identifier: GPL-3.0-or-later
#
# Portions of this file are derived from ComfyUI
# (https://github.com/comfyanonymous/ComfyUI), copyright comfyanonymous and
# contributors, licensed under the GNU General Public License v3.
# The GPL terms apply to this file.  The remainder of the ai-toolkit project
# is licensed under the MIT License.

from .manager import MemoryManager

import torch

try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except AttributeError:
    OOM_EXCEPTION = Exception

try:
    _ACCELERATOR_ERROR = torch.AcceleratorError
except AttributeError:
    _ACCELERATOR_ERROR = RuntimeError


def is_oom(e: Exception) -> bool:
    """Return True if *e* is a CUDA out-of-memory error."""
    if isinstance(e, OOM_EXCEPTION):
        return True
    if isinstance(e, _ACCELERATOR_ERROR) and (
        getattr(e, "error_code", None) == 2 or "out of memory" in str(e).lower()
    ):
        return True
    return False


def soft_empty_cache() -> None:
    """
    Flush the CUDA allocator cache more aggressively than empty_cache alone.

    Calls synchronize + empty_cache + ipc_collect.  The ipc_collect step
    releases cross-process GPU memory handles that empty_cache alone leaves
    behind, recovering an extra 500 MB-1 GB on fragmented allocators.
    Derived from ComfyUI's soft_empty_cache().
    """
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def get_free_vram(device=None) -> int:
    """Return free VRAM in bytes for *device* (defaults to current CUDA device)."""
    if not torch.cuda.is_available():
        return 0
    if device is None:
        device = torch.cuda.current_device()
    free, _ = torch.cuda.mem_get_info(device)
    return free


def get_total_vram(device=None) -> int:
    """Return total VRAM in bytes for *device*."""
    if not torch.cuda.is_available():
        return 0
    if device is None:
        device = torch.cuda.current_device()
    _, total = torch.cuda.mem_get_info(device)
    return total


def model_size_bytes(model: torch.nn.Module) -> int:
    """Return the sum of parameter storage sizes in bytes (does not count buffers)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def log_vram(label: str) -> None:
    """Print current VRAM allocated and reserved (allocator cache) with a label.

    'alloc' is memory actually held by live tensors.
    'res' is memory held by the PyTorch CUDA allocator (includes freed tensors
    not yet returned to the driver).  The gap between the two is pages the
    allocator is sitting on but no tensor currently needs — a large gap here
    usually means empty_cache() would help.
    """
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    res = torch.cuda.memory_reserved() / 1e9
    print(f"[VRAM] {label}: {alloc:.2f} GB alloc / {res:.2f} GB reserved")


def safe_ram_flush() -> None:
    """
    Deep RAM flush intended for use after model quantization.

    Safetensors loads model weights via mmap. After quantization the new
    quantized tensors are fresh allocations, but the original mmap'd file
    handles stay open as long as any Python object holds a reference —
    including objects kept alive by GC reference cycles. On Windows these
    mapped pages remain in the process working set until the mmap closes,
    causing RAM to spike when subsequent models (e.g. perceptual encoders)
    are loaded on top.

    Strategy: GC first (drops Python refs and lets mmap file handles close),
    then CUDA sync + cache clear (releases CUDA allocator reserve pages),
    then one final GC pass (cleans up any Python wrappers freed by CUDA).
    Three GC passes are used because a cycle broken in pass N may release
    objects whose finalizers free further refs caught only in pass N+1.
    """
    import gc
    gc.collect()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()