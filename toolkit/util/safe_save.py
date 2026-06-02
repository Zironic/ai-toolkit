"""Windows-safe atomic safetensors write.

safetensors.save_file uses memory-mapped I/O on Windows. When the same
cache file was previously read via load_file in the same process, the mmap
section can stay live and block an in-place rewrite, producing
OSError 1224 ("user-mapped section open"). Writing to a sibling temp file
and using os.replace sidesteps that: rename succeeds even when the
destination is currently mmap-mapped.
"""
import os
import tempfile

from safetensors.torch import save_file


def atomic_save_file(save_data: dict, cache_path: str, metadata: dict = None) -> None:
    cache_dir = os.path.dirname(cache_path) or "."
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".safetensors", dir=cache_dir)
    os.close(fd)
    try:
        save_file(save_data, tmp_path, metadata=metadata)
        os.replace(tmp_path, cache_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
