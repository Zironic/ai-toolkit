"""Residency-ownership invariants (CPU, fake modules).

The memory subsystem now has four ownership classes for a module's weights:
managed (streaming layer manager), unmanaged (moved freely by model.to),
resident-pinned, and ingraph pack-source (CPU residency IS the design; the
trunk streams from the pinned pack). The rules for who may move what are
policy, and both recent Krea2-scale failures were logic bugs in this layer
(not implementation typos) -- e.g. _move_unmanaged_parameters hauling 11 GiB
of pack sources onto the card because the pack-source class postdated the
rule. These tests pin the ownership contract down abstractly.
"""

import unittest

import torch

from toolkit.memory_management.manager import MemoryManager


class _FakeLMM:
    pass


def _linear(n=8):
    return torch.nn.Linear(n, n, bias=True)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA as the move target")
class ResidencyOwnershipTests(unittest.TestCase):
    def _module(self):
        m = torch.nn.Module()
        m.managed = _linear()
        m.unmanaged = _linear()
        m.pack_source = _linear()
        m.managed._layer_memory_manager = _FakeLMM()
        m.pack_source._mm_ingraph_pack_source = True
        return m

    def test_unmanaged_move_respects_ownership_classes(self):
        m = self._module()
        MemoryManager._move_unmanaged_parameters(m, torch.device("cuda"))

        # Unmanaged: moved.
        self.assertEqual(m.unmanaged.weight.device.type, "cuda")
        self.assertEqual(m.unmanaged.bias.device.type, "cuda")
        # Managed: untouched (owned by the streaming manager).
        self.assertEqual(m.managed.weight.device.type, "cpu")
        # Pack source: untouched (CPU residency is the design).
        self.assertEqual(m.pack_source.weight.device.type, "cpu")
        self.assertEqual(m.pack_source.bias.device.type, "cpu")

    def test_pack_source_mark_cleared_restores_unmanaged_semantics(self):
        m = self._module()
        del m.pack_source._mm_ingraph_pack_source
        MemoryManager._move_unmanaged_parameters(m, torch.device("cuda"))
        self.assertEqual(m.pack_source.weight.device.type, "cuda")

    def test_buffers_follow_the_same_ownership(self):
        m = self._module()
        m.unmanaged.register_buffer("stat", torch.zeros(2))
        m.pack_source.register_buffer("stat", torch.zeros(2))
        MemoryManager._move_unmanaged_parameters(m, torch.device("cuda"))
        self.assertEqual(m.unmanaged.stat.device.type, "cuda")
        self.assertEqual(m.pack_source.stat.device.type, "cpu")


class MmditPackSourceLifecycleTests(unittest.TestCase):
    """enable/disable_ingraph_training must set and clear the pack-source
    mark on exactly the packed linears (tiny CPU SingleStreamDiT)."""

    def _model(self):
        from extensions_built_in.diffusion_models.krea2.src.mmdit import (
            SingleMMDiTConfig,
            SingleStreamDiT,
        )

        return SingleStreamDiT(
            SingleMMDiTConfig(
                features=32,
                tdim=16,
                txtdim=32,
                heads=4,
                multiplier=1,
                layers=2,
                patch=1,
                channels=4,
                txtheads=4,
                txtkvheads=4,
            )
        )

    def test_marks_set_on_enable_and_cleared_on_disable(self):
        model = self._model()
        for param in model.parameters():
            param.requires_grad_(False)
        # Float (non-quantized) blocks pack fine on CPU; compile off.
        count = model.enable_ingraph_training(depth=2, compile=False)
        self.assertEqual(count, 2)
        marked = [
            name
            for name, child in model.named_modules()
            if getattr(child, "_mm_ingraph_pack_source", False)
        ]
        self.assertEqual(len(marked), 16)  # 2 blocks x 8 linears
        self.assertTrue(all(name.startswith("blocks.") for name in marked))

        model.disable_ingraph_training()
        marked_after = [
            name
            for name, child in model.named_modules()
            if getattr(child, "_mm_ingraph_pack_source", False)
        ]
        self.assertEqual(marked_after, [])

    def test_enable_disable_round_trip_is_repeatable(self):
        model = self._model()
        for param in model.parameters():
            param.requires_grad_(False)
        for _ in range(2):
            model.enable_ingraph_training(depth=2, compile=False)
            model.disable_ingraph_training()
        self.assertEqual(
            [
                name
                for name, child in model.named_modules()
                if getattr(child, "_mm_ingraph_pack_source", False)
            ],
            [],
        )


if __name__ == "__main__":
    unittest.main()
