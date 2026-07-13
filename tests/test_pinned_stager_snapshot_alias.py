"""PinnedStager.snapshot must return tensors that own their memory.

The step-100 save failure this pins down: with ``save.dtype: bf16`` and bf16
LoRA weights, ``view.to(out_dtype)`` was a dtype no-op and returned the view
itself -- an alias into the shared pinned staging buffer. safetensors then
refused the write (shared storage), and had it not, the next drain() would
have overwritten the aliased chunks with other tensors' bytes.
"""

import unittest

import torch

from toolkit.async_save import PinnedStager


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class PinnedStagerAliasTests(unittest.TestCase):
    def test_same_dtype_snapshot_owns_memory_across_chunk_reuse(self):
        torch.manual_seed(7)
        # Cap small enough that the item set needs several drain() rounds,
        # so any alias into the buffer gets visibly clobbered.
        stager = PinnedStager(cap_bytes=4096, register=False)
        try:
            items = [
                (f"w{i}", torch.randn(16, 16, device="cuda", dtype=torch.bfloat16))
                for i in range(8)
            ]
            expected = {k: v.detach().float().cpu() for k, v in items}

            out = stager.snapshot(items, out_dtype=torch.bfloat16)

            buf_ptr = stager._buf.untyped_storage().data_ptr()
            ptrs = set()
            for key, snap in out.items():
                ptr = snap.untyped_storage().data_ptr()
                self.assertNotEqual(ptr, buf_ptr, f"{key} aliases the pinned buffer")
                self.assertNotIn(ptr, ptrs, f"{key} shares storage with another snapshot")
                ptrs.add(ptr)
                torch.testing.assert_close(snap.float(), expected[key], rtol=0, atol=0)
        finally:
            stager.close()

    def test_cpu_only_path_does_not_alias_live_tensor(self):
        # Simulate the no-CUDA fallback by feeding CPU tensors of out_dtype
        # through the same conversion the fallback uses: mutate the source
        # afterwards and the snapshot must not follow.
        stager = PinnedStager(cap_bytes=4096, register=False)
        try:
            live = torch.randn(8, 8, dtype=torch.bfloat16)
            snap = live.detach().to("cpu", torch.bfloat16, copy=True)
            self.assertNotEqual(
                snap.untyped_storage().data_ptr(),
                live.untyped_storage().data_ptr(),
            )
            before = snap.clone()
            live.add_(1.0)
            torch.testing.assert_close(snap, before, rtol=0, atol=0)
        finally:
            stager.close()


if __name__ == "__main__":
    unittest.main()

import pytest

pytestmark = pytest.mark.process_isolated
