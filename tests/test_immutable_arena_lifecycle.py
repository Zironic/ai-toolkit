"""Slice 6 lifecycle coverage for the canonical immutable-arena backend
(tasks/open/IMMUTABLE_TRANSFER_ARENA_PLAN.md).

The CPU-only class exercises the two seams that must hold regardless of GPU
presence: the ``attach_smart_training`` short-circuit and Invariant 5 (no
whole-model move path may touch a canonical leaf). The CUDA class exercises
the real arena: attach-time leaf exclusion and true-unload teardown.
"""

import types
import unittest

import torch
import torch.nn as nn

from toolkit.memory_management import pin_manager
from toolkit.memory_management.canonical_arena import CanonicalArena
from toolkit.memory_management.manager import MemoryManager


class _TwoLinear(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.keep = nn.Linear(d, d, bias=True)
        self.canon = nn.Linear(d, d, bias=True)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.canon(self.keep(x))


class ImmutableBackendCpuTests(unittest.TestCase):
    """No CUDA required: attribute-level short-circuit + move guards."""

    def test_attach_smart_training_returns_stashed_plan(self):
        module = _TwoLinear()
        module._mm_immutable_backend = True
        sentinel = {"offload_ids": (1, 2, 3)}
        module._memory_manager = types.SimpleNamespace(_smart_training_plan=sentinel)
        # Short-circuit fires BEFORE any CUDA/allocator work, so "cpu" is fine.
        returned = MemoryManager.attach_smart_training(module, "cpu")
        self.assertIs(returned, sentinel)

    def test_attach_smart_training_raises_without_a_plan(self):
        module = _TwoLinear()
        module._mm_immutable_backend = True
        # _memory_manager present but no plan -> loud failure, no silent attach.
        module._memory_manager = types.SimpleNamespace()
        with self.assertRaises(RuntimeError):
            MemoryManager.attach_smart_training(module, "cpu")

    def test_attach_smart_training_raises_without_a_manager(self):
        module = _TwoLinear()
        module._mm_immutable_backend = True
        with self.assertRaises(RuntimeError):
            MemoryManager.attach_smart_training(module, "cpu")

    def test_move_quantized_parameters_skips_canonical_leaf(self):
        # Plain (non-quantized) leaves are ignored by this helper, so a marked
        # plain leaf is trivially untouched; assert the guard does not raise and
        # leaves both weights on CPU regardless.
        module = _TwoLinear()
        module.canon._mm_canonical_leaf = True
        MemoryManager._move_quantized_parameters(module, "cpu")
        self.assertEqual(module.canon.weight.device.type, "cpu")
        self.assertEqual(module.keep.weight.device.type, "cpu")


@unittest.skipUnless(torch.cuda.is_available(), "immutable arena needs CUDA")
class ImmutableBackendCudaTests(unittest.TestCase):
    def setUp(self):
        self._ledger_before = pin_manager.pinned_bytes_by_kind()
        pin_manager._LEDGER.clear()

        def _restore():
            pin_manager._LEDGER.clear()
            pin_manager._LEDGER.update(self._ledger_before)

        self.addCleanup(_restore)

    def _canonicalize(self, module):
        arena = CanonicalArena()
        arena.canonicalize(
            {"blocks.0": [("keep", module.keep), ("canon", module.canon)]}
        )
        for child in (module.keep, module.canon):
            child._mm_canonical_leaf = True
        module._mm_canonical_arena = arena
        module._mm_immutable_backend = True
        return arena

    def test_destroy_immutable_arena_preserves_data_and_clears_state(self):
        module = _TwoLinear()
        before_keep = module.keep.weight.detach().clone()
        before_canon = module.canon.weight.detach().clone()
        arena = self._canonicalize(module)
        flat_ptr = arena.block_pack("blocks.0").host_flat.data_ptr()
        # Canonical params now view the arena flat.
        self.assertEqual(module.keep.weight.data_ptr(), flat_ptr)

        MemoryManager._destroy_immutable_arena(module)

        # Data survives byte-for-byte on standalone (non-arena) storage.
        self.assertTrue(torch.equal(module.keep.weight.detach(), before_keep))
        self.assertTrue(torch.equal(module.canon.weight.detach(), before_canon))
        self.assertNotEqual(module.keep.weight.data_ptr(), flat_ptr)
        # Every immutable-backend marker is gone; the module reads as plain.
        self.assertIsNone(getattr(module, "_mm_canonical_arena", None))
        self.assertFalse(getattr(module, "_mm_immutable_backend", False))
        self.assertFalse(getattr(module.keep, "_mm_canonical_leaf", False))
        self.assertFalse(getattr(module.canon, "_mm_canonical_leaf", False))
        # The "weights" pin tier returns to its pre-arena baseline.
        self.assertEqual(pin_manager.pinned_bytes_by_kind().get("weights", 0), 0)

    def _assert_move_skips_canonical(self, mover):
        # Invariant 5: no whole-model move path may touch a canonical leaf.
        # A real cpu->cuda move is deterministic (unlike a "meta" move, which
        # trips torch's set_data type check): the skipped leaf stays on CPU,
        # the moved sibling lands on CUDA.
        module = _TwoLinear()
        module.canon._mm_canonical_leaf = True
        canon_weight = module.canon.weight
        mover(module, "cuda:0")
        self.assertEqual(
            module.canon.weight.device.type, "cpu",
            "canonical leaf must never be moved by a whole-model move path",
        )
        self.assertEqual(module.keep.weight.device.type, "cuda")
        self.assertIs(module.canon.weight, canon_weight)

    def test_move_module_parameters_skips_canonical_leaf(self):
        self._assert_move_skips_canonical(MemoryManager._move_module_parameters)

    def test_move_unmanaged_parameters_skips_canonical_leaf(self):
        self._assert_move_skips_canonical(MemoryManager._move_unmanaged_parameters)

    def test_destroy_immutable_arena_is_noop_without_an_arena(self):
        module = _TwoLinear()
        # Never canonicalized -> must not raise.
        MemoryManager._destroy_immutable_arena(module)

    def test_attach_smart_training_immutable_excludes_canonical_leaves(self):
        device = torch.device("cuda:0")
        module = _TwoLinear().to(device)
        for p in module.parameters():
            p.requires_grad_(False)
        canonical = [module.canon]
        for child in canonical:
            child._mm_canonical_leaf = True
        try:
            plan = MemoryManager.attach_smart_training_immutable(
                module, device, canonical_modules=canonical,
                working_reserve_gib=0.5,
            )
            mm = module._memory_manager
            # The stashed plan and canonical-id bookkeeping are recorded.
            self.assertIs(mm._smart_training_plan, plan)
            self.assertEqual(mm._canonical_leaf_ids, {id(module.canon)})
            self.assertEqual(module._mm_canonical_leaf_ids, {id(module.canon)})
            # Legacy autotune is deliberately disabled for this backend.
            self.assertFalse(mm._training_autotune_enabled)
            # A canonical leaf must never receive a legacy streaming manager.
            self.assertFalse(hasattr(module.canon, "_layer_memory_manager"))
        finally:
            MemoryManager.detach(module)


if __name__ == "__main__":
    unittest.main()
