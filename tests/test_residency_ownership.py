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


class _FakeLayerManager:
    """What MemoryManager.attach leaves on an offloaded Linear: the marker
    is_streamed_module reads, plus the base-forward slot the strip restores."""

    def __init__(self, child):
        self._original_forward = child.forward
        self._forward_container = child
        self._forward_attribute = "forward"


class MmditPackSourceLifecycleTests(unittest.TestCase):
    """enable/disable_ingraph_training must set and clear the pack-source mark
    on exactly the STREAMED linears (tiny CPU SingleStreamDiT).

    Residency is chosen per-Linear, so a block is routinely part streamed / part
    resident. Marking a resident leaf a pack source pins it to the host, and a
    sampler block promoted to the device then runs F.linear against a CPU weight
    ("mat2 is on cpu"). Marking only what the manager actually streams is the
    invariant this file guards.
    """

    @staticmethod
    def _mark_streamed(model, names):
        streamed = []
        for name, child in model.named_modules():
            if name in names:
                child._layer_memory_manager = _FakeLayerManager(child)
                streamed.append(name)
        return streamed

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

    def _marked(self, model):
        return [
            name
            for name, child in model.named_modules()
            if getattr(child, "_mm_ingraph_pack_source", False)
        ]

    def test_marks_set_on_enable_and_cleared_on_disable(self):
        model = self._model()
        for param in model.parameters():
            param.requires_grad_(False)
        streamed = self._mark_streamed(
            model,
            {
                "blocks.0.attn.wq",
                "blocks.0.mlp.down",
                "blocks.1.attn.wo",
            },
        )
        self.assertEqual(len(streamed), 3)

        # Float (non-quantized) blocks pack fine on CPU; compile off.
        count = model.enable_ingraph_training(depth=2, compile=False)
        self.assertEqual(count, 2)  # every block is in the trunk...
        self.assertEqual(model._ingraph_training_streamed_leaves, 3)
        self.assertEqual(model._ingraph_training_resident_leaves, 13)
        self.assertEqual(model._ingraph_training_resident_blocks, 0)

        # ...but only the streamed leaves are pack sources. Marking the other 13
        # would strand them on the host when sampling promotes their block.
        self.assertEqual(sorted(self._marked(model)), sorted(streamed))

        model.disable_ingraph_training()
        self.assertEqual(self._marked(model), [])

    def test_fully_resident_block_packs_nothing_and_marks_nothing(self):
        """No manager attached anywhere: every leaf is already on the device, so
        the trunk builds no flat, pins nothing, and emits no fetch."""
        model = self._model()
        for param in model.parameters():
            param.requires_grad_(False)

        count = model.enable_ingraph_training(depth=2, compile=False)
        self.assertEqual(count, 2)
        self.assertEqual(model._ingraph_training_resident_blocks, 2)
        self.assertEqual(model._ingraph_training_streamed_leaves, 0)
        self.assertEqual(model._ingraph_training_resident_leaves, 16)
        self.assertEqual(model._ingraph_training_borrowed_count, 0)
        self.assertEqual(model._ingraph_training_owned_count, 0)
        self.assertEqual(self._marked(model), [])
        for plan in model._ingraph_training_plans.values():
            self.assertIsNone(plan.pack)
            self.assertFalse(plan.streams)

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

import pytest

pytestmark = pytest.mark.leaky  # order-dependent under full suite; see ticket f2aceba
