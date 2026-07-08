"""Demoting a resident FP8-sampling block must actually free GPU memory.

Regression for the 2000px WDDM-fallback failure: the resident FP8 sampling
forwards capture the GPU qdata/scale as closure constants, so demoting the
layer (param.data -> CPU) freed NOTHING while the demote loop reported the
block's weight bytes as progress -- it walked the whole model without
relieving the OOM. The fix releases the demoted layers' fp8 closures first
(_release_fp8_sampling_for) and the OOM path now governs on measured bytes.
"""

import unittest

import torch

from toolkit.memory_management import MemoryManager
from toolkit.util.quantize import get_qtype, quantize

GIB = 1024 ** 3
MIB = 1024 ** 2


def _fp8_block_model(blocks=3, width=1024):
    torch.manual_seed(11)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(width, width, bias=True)
            self.fc2 = torch.nn.Linear(width, width, bias=True)

        def forward(self, x):
            return self.fc2(self.fc1(x))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(Block() for _ in range(blocks))

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    model = Model().to(device="cuda", dtype=torch.bfloat16)
    quantize(model, weights=get_qtype("float8"))
    return model


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SamplingFp8DemoteFreesTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        torch.cuda.empty_cache()

    def _attach_resident(self, model):
        MemoryManager.attach(
            model,
            self.device,
            offload_percent=1.0,
            ignore_modules=[],
            _offload_module_ids=set(),
            pinned_weight_gib=0.0,
        )
        return {"model_bytes": MemoryManager._module_bytes(model)}

    def test_demote_with_fp8_restores_frees_device_memory(self):
        model = _fp8_block_model()
        plan = self._attach_resident(model)
        restores, resident_layers, _ = MemoryManager._enable_fp8_sampling(model)
        self.assertGreater(resident_layers, 0)
        entries_before = len(restores)

        before = MemoryManager._torch_allocatable_bytes(self.device)
        freed_plan = MemoryManager._sampling_demote_largest_block(
            model, plan, self.device, ignore_modules=[], fp8_restores=restores
        )
        measured = MemoryManager._torch_allocatable_bytes(self.device) - before

        self.assertGreater(freed_plan, 0)
        # The demoted block's fp8 closures must be gone from the restore list.
        self.assertLess(len(restores), entries_before)
        # The device must actually gain most of the demoted weight bytes.
        self.assertGreater(measured, int(freed_plan * 0.5))
        # Demoted layers stream from now on and keep fp8 execution.
        streamed = [
            child for child in model.modules()
            if hasattr(child, "_layer_memory_manager")
        ]
        self.assertTrue(streamed)
        self.assertTrue(
            all(
                getattr(child, "_memory_management_fp8_sampling", False)
                for child in streamed
            )
        )
        MemoryManager._disable_fp8_sampling(model, restores)
        MemoryManager.detach(model)

    def test_demote_without_release_leaks_by_closure(self):
        """Negative control: proves the closure pin is what the fix removes."""
        model = _fp8_block_model()
        plan = self._attach_resident(model)
        restores, resident_layers, _ = MemoryManager._enable_fp8_sampling(model)
        self.assertGreater(resident_layers, 0)

        before = MemoryManager._torch_allocatable_bytes(self.device)
        freed_plan = MemoryManager._sampling_demote_largest_block(
            model, plan, self.device, ignore_modules=[], fp8_restores=None
        )
        measured = MemoryManager._torch_allocatable_bytes(self.device) - before

        self.assertGreater(freed_plan, 0)
        # The closures still hold the GPU storage: almost nothing comes back.
        self.assertLess(measured, int(freed_plan * 0.5))
        MemoryManager._disable_fp8_sampling(model, restores)
        MemoryManager.detach(model)


if __name__ == "__main__":
    unittest.main()
