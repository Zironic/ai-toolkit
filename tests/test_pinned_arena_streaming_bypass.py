"""Slice A: the eager/pre-compile streaming pinned-bypass must recognize
register-pinned arena flat VIEWS as pinned.

A streamed leaf is a view into a per-block flat at an offset. The flat is
pinned with cudaHostRegister, so torch's ``is_pinned()`` is False for it, and
the leaf's own ``data_ptr`` (flat base + offset) misses the exact-ptr
``_REGISTERED_HOST_PINS`` table. The fix records each pinned flat's storage
base ptr in ``pin_manager`` so ``is_arena_backed`` -- consulted by
``manager_modules._profile_is_pinned`` and ``bounce_pool._is_pinned`` -- treats
every view into it as pinned. Without this, training streams arena weights
through bounce staging anyway and the pin buys nothing off the ingraph path.

These are UNIT tests: they never issue a real ``cudaHostRegister`` (that is
page-recycling-collision-prone in a shared test process -- see 763bb75 -- and
is exercised end to end by scripts/smoke_krea2_ingraph_cuda.py instead). The
pin is stubbed to "succeed" so the arena's registration bookkeeping and the
recognition logic run deterministically on CPU.
"""

import unittest
from unittest import mock

import torch
import torch.nn as nn
from optimum.quanto import freeze

from toolkit.util.quantize import get_qtype, quantize
from toolkit.memory_management import pin_manager
from toolkit.memory_management.manager_modules import _profile_is_pinned
from toolkit.memory_management.bounce_pool import _is_pinned as _bounce_is_pinned
from toolkit.memory_management.pinned_arena import PinnedWeightArena


def _linear(in_f=8, out_f=4, bias=True):
    layer = nn.Linear(in_f, out_f, bias=bias)
    layer.weight.requires_grad_(False)
    if bias:
        layer.bias.requires_grad_(False)
    return layer


class _StubbedPinMixin:
    """Make ``pin_register`` report success WITHOUT a real cudaHostRegister, so
    ``build()`` runs its ``pack.pinned -> register_arena_storage`` path and the
    recognition logic is exercised on plain pageable host memory. The exact-ptr
    ``_REGISTERED_HOST_PINS`` table stays empty (the real pin populates it), but
    Slice A recognition keys on the arena STORAGE set, which build() fills."""

    def setUp(self):
        self._pin_patch = mock.patch.object(
            pin_manager, "pin_tensor_in_place", return_value=True
        )
        self._pin_patch.start()
        self.addCleanup(self._pin_patch.stop)


class ArenaStorageRegistryTests(unittest.TestCase):
    """The pin_manager storage-base registry, exercisable without CUDA."""

    def test_view_into_registered_flat_is_arena_backed(self):
        flat = torch.empty(4096, dtype=torch.uint8)
        pin_manager.register_arena_storage(flat)
        try:
            view = flat[128:256]
            self.assertTrue(pin_manager.is_arena_backed(flat))
            self.assertTrue(pin_manager.is_arena_backed(view))
        finally:
            pin_manager.unregister_arena_storage(flat)
        self.assertFalse(pin_manager.is_arena_backed(flat))

    def test_unrelated_storage_is_not_arena_backed(self):
        flat = torch.empty(4096, dtype=torch.uint8)
        stranger = torch.empty(4096, dtype=torch.uint8)
        pin_manager.register_arena_storage(flat)
        try:
            self.assertFalse(pin_manager.is_arena_backed(stranger))
        finally:
            pin_manager.unregister_arena_storage(flat)

    def test_refcount_survives_transient_recycled_ptr(self):
        flat = torch.empty(4096, dtype=torch.uint8)
        pin_manager.register_arena_storage(flat)
        pin_manager.register_arena_storage(flat)  # e.g. rebuild reused same base
        try:
            pin_manager.unregister_arena_storage(flat)
            self.assertTrue(pin_manager.is_arena_backed(flat))  # still one ref
        finally:
            pin_manager.unregister_arena_storage(flat)
        self.assertFalse(pin_manager.is_arena_backed(flat))

    def test_gpu_tensor_never_arena_backed(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        t = torch.empty(16, device="cuda")
        self.assertFalse(pin_manager.is_arena_backed(t))


class ProfileIsPinnedRecognizesArenaViewsTests(_StubbedPinMixin, unittest.TestCase):
    """The two streaming-bypass leaf checks must recognize arena views."""

    def test_plain_view_reports_pinned_to_both_bypasses(self):
        layer = _linear()
        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin", layer)]})
        self.assertTrue(arena.block_pack("blocks.0").pinned)
        try:
            # is_arena_backed is the deterministic signal (the storage set the
            # arena fills); the two bypasses must derive pinned-ness from it,
            # since a register-pinned flat view reports is_pinned()==False.
            self.assertTrue(pin_manager.is_arena_backed(layer.weight))
            self.assertTrue(_profile_is_pinned(layer.weight))
            self.assertTrue(_profile_is_pinned(layer.bias))
            self.assertTrue(_bounce_is_pinned(layer.weight))
        finally:
            arena.release()
        # Flag-off / unchanged path: release drops the registration. (Assert on
        # is_arena_backed, not _profile_is_pinned: a fresh tensor can land on a
        # page torch recycled from another test's real pinned host cache, so
        # is_pinned() -- and thus _profile_is_pinned -- is not deterministically
        # False here. The arena storage set is what this code controls.)
        self.assertFalse(pin_manager.is_arena_backed(layer.weight))

    def test_fp8_wrapper_leaves_report_pinned(self):
        model = nn.Sequential(nn.Linear(8, 4, bias=False).to(torch.bfloat16))
        quantize(model, weights=get_qtype("qfloat8"))
        freeze(model)
        layer = model[0]
        layer.weight.requires_grad_(False)
        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin", layer)]})
        try:
            # The wrapper's qdata/scale leaves are both views into the flat, so
            # the recursion in _profile_is_pinned must see the whole wrapper as
            # pinned.
            self.assertTrue(_profile_is_pinned(layer.weight))
            self.assertTrue(_bounce_is_pinned(layer.weight))
        finally:
            arena.release()

    def test_unpinned_non_arena_tensor_is_not_pinned(self):
        t = torch.empty(16, dtype=torch.float32)
        self.assertFalse(_profile_is_pinned(t))
        self.assertFalse(_bounce_is_pinned(t))


class ArenaAutoRegistersPinnedFlatsTests(_StubbedPinMixin, unittest.TestCase):
    """build()/release()/rebuild manage the registry automatically."""

    def test_build_registers_and_release_unregisters(self):
        layer = _linear()
        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin", layer)]})
        self.assertTrue(arena.block_pack("blocks.0").pinned)
        self.assertTrue(pin_manager.is_arena_backed(layer.weight))
        self.assertTrue(_profile_is_pinned(layer.weight))
        arena.release()
        # See test_plain_view: assert on the deterministic arena storage set,
        # not _profile_is_pinned (a recycled torch-pinned page can make
        # is_pinned() True independent of the arena).
        self.assertFalse(pin_manager.is_arena_backed(layer.weight))

    def test_rebuild_unregisters_old_flat(self):
        a = _linear()
        b = _linear()
        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin_a", a)]})
        old_flat = arena.block_pack("blocks.0").host_flat
        old_ptr = old_flat.untyped_storage().data_ptr()
        # Rebuild the same block key with a different membership -> old flat
        # released + unregistered, new flat registered.
        arena.build({"blocks.0": [("lin_a", a), ("lin_b", b)]})
        new_flat = arena.block_pack("blocks.0").host_flat
        try:
            self.assertTrue(_profile_is_pinned(a.weight))
            self.assertTrue(_profile_is_pinned(b.weight))
            if new_flat.untyped_storage().data_ptr() != old_ptr:
                self.assertFalse(
                    pin_manager.is_arena_backed(old_flat),
                    "old flat storage should be unregistered after rebuild",
                )
        finally:
            arena.release()


if __name__ == "__main__":
    unittest.main()
