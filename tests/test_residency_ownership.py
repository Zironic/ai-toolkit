"""Residency-ownership invariants (CPU, fake modules).

The memory subsystem now has four ownership classes for a module's weights:
managed (streaming layer manager), unmanaged (moved freely by model.to),
resident-pinned, and pack-source (CPU residency IS the design; the caller
streams from the pinned pack). The rules for who may move what are
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


if __name__ == "__main__":
    unittest.main()
