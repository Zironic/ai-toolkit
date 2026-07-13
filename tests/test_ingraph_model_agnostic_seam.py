"""Phase 3 Slice C: the arena + in-graph pack machinery has no Krea2 dependency.

A synthetic multi-block module drives MemoryManager._build_pinned_arena and the
shared ingraph_stream.build_or_borrow_block_packs helper end to end. If any
Krea2 assumption (block container shape, `blocks.{i}` naming, entry ordering,
SingleStreamDiT types) leaks into the shared layer, these fail.

The helper's policy is also pinned down here, since it is the one place every
model's enable_ingraph_* glue depends on: borrow when the arena has the block,
own only when allowed, fail closed on a pageable pack, and never release a
borrowed arena flat.
"""

import unittest
from unittest import mock

import torch
import torch.nn as nn

from toolkit.memory_management import pin_manager
from toolkit.memory_management.ingraph_stream import (
    IngraphPackError,
    assemble_leaf_args,
    block_tensor_views,
    build_block_leaf_plans,
    build_or_borrow_block_packs,
    release_pack,
    resident_linear_tensors,
    streamed_linear_tensors,
)
from toolkit.memory_management.manager import MemoryManager
from toolkit.memory_management.pinned_arena import PinnedWeightArena


class _Stage(nn.Module):
    """Deliberately NOT Krea2's block shape: different child names, a nested
    sub-module, and a container attribute that is not called `blocks`."""

    def __init__(self, d):
        super().__init__()
        self.proj_in = nn.Linear(d, d, bias=True)
        self.inner = nn.Module()
        self.inner.proj_out = nn.Linear(d, d, bias=False)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.inner.proj_out(self.proj_in(x))


class _SynthModel(nn.Module):
    def __init__(self, d=32, n=3):
        super().__init__()
        self.stages = nn.ModuleList([_Stage(d) for _ in range(n)])

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x

    def block_entries(self):
        """The model-side glue from the protocol: stable block_key -> entries."""
        return {
            f"stages.{i}": [
                ("proj_in", stage.proj_in),
                ("inner.proj_out", stage.inner.proj_out),
            ]
            for i, stage in enumerate(self.stages)
        }


class _StubbedPinMixin:
    """Report pin success without a real cudaHostRegister -- see
    test_pinned_arena_streaming_bypass for why the unit tests never really pin.

    Two stubs are needed. ``pin_tensor_in_place`` makes ``pin_register`` hand
    back a pinned PinHandle (that is what an OWNED pack's ``pinned`` reads), but
    it never populates the exact-ptr registration table, so a BORROWED pack --
    whose ``pinned`` comes from ``is_host_pinned(flat)`` -- would still report
    pageable and sink the set with non_pinned_pack. Stub both."""

    def setUp(self):
        for name, value in (
            ("pin_tensor_in_place", True),
            ("is_host_pinned", True),
        ):
            patch = mock.patch.object(pin_manager, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)


class SharedHelperPolicyTests(_StubbedPinMixin, unittest.TestCase):
    def test_borrows_every_block_the_arena_covers(self):
        model = _SynthModel()
        arena = PinnedWeightArena()
        self.addCleanup(arena.release)
        arena.build(model.block_entries())

        result = build_or_borrow_block_packs(arena, model.block_entries())
        self.assertEqual(result.borrowed, 3)
        self.assertEqual(result.owned, 0)
        self.assertEqual(sorted(result.packs), ["stages.0", "stages.1", "stages.2"])
        for pack in result.packs.values():
            self.assertTrue(pack.borrowed_from_arena)
            self.assertTrue(pack.pinned)
            self.assertFalse(pack.owns_flat)

    def test_owned_fallback_when_no_arena(self):
        model = _SynthModel()
        result = build_or_borrow_block_packs(None, model.block_entries())
        try:
            self.assertEqual(result.borrowed, 0)
            self.assertEqual(result.owned, 3)
            for pack in result.packs.values():
                self.assertFalse(pack.borrowed_from_arena)
                self.assertTrue(pack.owns_flat)
        finally:
            for pack in result.packs.values():
                release_pack(pack)

    def test_owned_fallback_can_be_forbidden(self):
        """Strict pinned-arena validation: a borrow miss must fail, not silently
        build an owned pack and let the run 'pass' without proving a borrow."""
        model = _SynthModel()
        with self.assertRaises(IngraphPackError) as ctx:
            build_or_borrow_block_packs(
                None, model.block_entries(), allow_owned_fallback=False
            )
        self.assertEqual(ctx.exception.reasons, ("arena_borrow_required",))

    def test_partial_arena_coverage_borrows_some_and_owns_rest(self):
        model = _SynthModel()
        arena = PinnedWeightArena()
        self.addCleanup(arena.release)
        entries = model.block_entries()
        arena.build({"stages.1": entries["stages.1"]})

        result = build_or_borrow_block_packs(arena, entries)
        try:
            self.assertEqual(result.borrowed, 1)
            self.assertEqual(result.owned, 2)
            self.assertTrue(result.packs["stages.1"].borrowed_from_arena)
            self.assertFalse(result.packs["stages.0"].borrowed_from_arena)
        finally:
            for pack in result.packs.values():
                release_pack(pack)

    def test_pageable_pack_fails_the_whole_set_closed(self):
        """Strict in-graph is all-or-nothing: one pageable pack sinks the set."""
        model = _SynthModel()
        # Refuse the pin so pack_block_host yields pageable flats.
        with mock.patch.object(pin_manager, "pin_tensor_in_place", return_value=False):
            with self.assertRaises(IngraphPackError) as ctx:
                build_or_borrow_block_packs(None, model.block_entries())
        self.assertEqual(ctx.exception.reasons, ("non_pinned_pack",))

    def test_failure_releases_owned_packs_but_never_borrowed_flats(self):
        """stages.0 borrows; the rest build owned pageable packs -> the set fails
        closed. Cleanup must release the owned packs and leave the arena's flat
        pinned and current -- release_pack no-ops on a borrowed pack."""
        model = _SynthModel()
        arena = PinnedWeightArena()
        self.addCleanup(arena.release)
        entries = model.block_entries()
        arena.build({"stages.0": entries["stages.0"]})

        with mock.patch.object(pin_manager, "pin_tensor_in_place", return_value=False):
            with mock.patch(
                "toolkit.memory_management.arena_offload.layout.release_pack",
                wraps=release_pack,
            ) as released:
                with self.assertRaises(IngraphPackError) as ctx:
                    build_or_borrow_block_packs(arena, entries)

        self.assertEqual(ctx.exception.reasons, ("non_pinned_pack",))
        released_packs = [call.args[0] for call in released.call_args_list]
        # Every pack built in the attempt was handed to release_pack, including
        # the borrowed one -- which release_pack must ignore.
        self.assertEqual(len(released_packs), 3)
        borrowed = [p for p in released_packs if p.borrowed_from_arena]
        self.assertEqual(len(borrowed), 1)
        self.assertFalse(borrowed[0].owns_flat)
        # The arena's flat survived the cleanup untouched.
        self.assertTrue(arena.block_pack("stages.0").pinned)
        self.assertTrue(arena.is_current(model.stages[0].proj_in))
        self.assertTrue(arena.is_current(model.stages[0].inner.proj_out))


def _mark_streamed(*modules):
    """What MemoryManager.attach leaves on an offloaded Linear. is_streamed_module
    reads exactly this, so the default predicate is under test too."""
    for module in modules:
        module._layer_memory_manager = object()


class PartialBlockResidencyTests(_StubbedPinMixin, unittest.TestCase):
    """A block is routinely part streamed / part resident: the planner splits
    residency per-Linear. The pack must cover only the streamed leaves, and the
    resident ones must come straight off their Parameters."""

    def test_fully_streamed_block_packs_every_leaf(self):
        model = _SynthModel(n=1)
        _mark_streamed(model.stages[0].proj_in, model.stages[0].inner.proj_out)

        result = build_block_leaf_plans(None, model.block_entries())
        try:
            plan = result.plans["stages.0"]
            self.assertTrue(plan.streams)
            self.assertEqual(plan.sources, ((True, 0), (True, 1)))
            self.assertEqual(result.streamed_leaves, 2)
            self.assertEqual(result.resident_leaves, 0)
            self.assertEqual(result.fully_resident, 0)
        finally:
            for plan in result.plans.values():
                release_pack(plan.pack)

    def test_partially_resident_block_packs_only_the_streamed_leaf(self):
        model = _SynthModel(n=1)
        stage = model.stages[0]
        _mark_streamed(stage.proj_in)  # inner.proj_out stays resident

        result = build_block_leaf_plans(None, model.block_entries())
        try:
            plan = result.plans["stages.0"]
            self.assertTrue(plan.streams)
            # Canonical order preserved: proj_in from the flat, proj_out resident.
            self.assertEqual(plan.sources, ((True, 0), (False, 0)))
            # The flat is sized for ONE linear -- this is the smaller ring.
            self.assertEqual(len(plan.pack.linears), 1)
            self.assertEqual(plan.pack.linears[0].name, "proj_in")
            self.assertEqual(result.streamed_leaves, 1)
            self.assertEqual(result.resident_leaves, 1)
            # The resident leaf aliases the live Parameter, it is not a copy.
            resident_weight = plan.resident_args[0][0]
            self.assertEqual(
                resident_weight.data_ptr(), stage.inner.proj_out.weight.data_ptr()
            )
        finally:
            for plan in result.plans.values():
                release_pack(plan.pack)

    def test_fully_resident_block_builds_no_pack_at_all(self):
        model = _SynthModel(n=1)  # nothing marked streamed

        with mock.patch(
            "toolkit.memory_management.arena_offload.layout.pack_block_host"
        ) as packed:
            result = build_block_leaf_plans(None, model.block_entries())

        packed.assert_not_called()  # no flat, no pin, no fetch node
        plan = result.plans["stages.0"]
        self.assertIsNone(plan.pack)
        self.assertFalse(plan.streams)
        self.assertEqual(plan.sources, ((False, 0), (False, 1)))
        self.assertEqual(result.fully_resident, 1)
        self.assertEqual(result.streamed_leaves, 0)

    def test_assemble_restores_canonical_order_across_both_sources(self):
        model = _SynthModel(n=1)
        stage = model.stages[0]
        _mark_streamed(stage.inner.proj_out)  # the SECOND leaf streams

        result = build_block_leaf_plans(None, model.block_entries())
        try:
            plan = result.plans["stages.0"]
            self.assertEqual(plan.sources, ((False, 0), (True, 0)))
            streamed_views = block_tensor_views(plan.pack.host_flat, plan.pack)
            leaf_args = assemble_leaf_args(plan, streamed_views)

            self.assertEqual(len(leaf_args), 2)
            # leaf 0 is resident proj_in, leaf 1 is the streamed proj_out view.
            self.assertEqual(leaf_args[0][0].data_ptr(), stage.proj_in.weight.data_ptr())
            self.assertTrue(
                torch.equal(leaf_args[1][0], stage.inner.proj_out.weight.detach())
            )
        finally:
            for plan in result.plans.values():
                release_pack(plan.pack)

    def test_resident_leaf_computes_what_the_module_computes(self):
        """The whole premise: a resident triple and a streamed triple are
        interchangeable inputs to streamed_linear_tensors."""
        linear = nn.Linear(8, 5, bias=True).eval()
        x = torch.randn(3, 8)

        (weight, bias, scale), qualifies = resident_linear_tensors(linear)
        self.assertIsNone(scale)
        self.assertFalse(qualifies)
        out = streamed_linear_tensors(x, weight, bias, scale, fp8_qualifies=False)

        torch.testing.assert_close(out, linear(x))

    def test_arena_borrows_the_streamed_subset_of_a_split_block(self):
        """The stale_modules=3/8 regression: the arena only ever holds the
        offloaded leaves, so the trunk must ask for exactly those."""
        model = _SynthModel(n=1)
        stage = model.stages[0]
        _mark_streamed(stage.proj_in)

        arena = PinnedWeightArena()
        self.addCleanup(arena.release)
        arena.build({"stages.0": [("proj_in", stage.proj_in)]})

        result = build_block_leaf_plans(
            arena, model.block_entries(), allow_owned_fallback=False
        )
        plan = result.plans["stages.0"]
        self.assertEqual(result.borrowed, 1)
        self.assertEqual(result.owned, 0)
        self.assertTrue(plan.borrowed_from_arena)
        self.assertEqual(plan.sources, ((True, 0), (False, 0)))

    def test_unsupported_resident_wrapper_fails_closed(self):
        model = _SynthModel(n=1)
        with mock.patch(
            "toolkit.memory_management.arena_offload.layout._flatten_leaves",
            return_value=[torch.zeros(2), torch.zeros(2), torch.zeros(2)],
        ):
            with self.assertRaises(IngraphPackError) as ctx:
                build_block_leaf_plans(None, model.block_entries())
        self.assertEqual(ctx.exception.reasons, ("unsupported_quant_wrapper",))


@unittest.skipUnless(torch.cuda.is_available(), "attach needs CUDA")
class SyntheticModelDrivesAttachArenaTests(unittest.TestCase):
    """_build_pinned_arena must group and cover a non-Krea2 module's blocks."""

    def test_attach_builds_arena_over_synthetic_blocks_and_helper_borrows(self):
        device = torch.device("cuda:0")
        model = _SynthModel(d=64, n=3)
        offload_ids = {id(m) for m in model.modules() if isinstance(m, nn.Linear)}
        MemoryManager.attach(
            model, device, _offload_module_ids=offload_ids, use_pinned_arena=True
        )
        try:
            arena = model._mm_weight_arena
            self.assertIsNotNone(arena)
            # Grouping is derived from module names, not any Krea2 convention.
            for i, stage in enumerate(model.stages):
                self.assertEqual(arena.arena_block_of(stage.proj_in), f"stages.{i}")
                self.assertTrue(arena.is_current(stage.inner.proj_out))

            result = build_or_borrow_block_packs(
                arena, model.block_entries(), allow_owned_fallback=False
            )
            self.assertEqual(result.borrowed, 3)
            self.assertEqual(result.owned, 0)
        finally:
            MemoryManager.detach(model)
            # This is the one test here that really cudaHostRegisters. Return the
            # pins (register unpins return the DXGI budget immediately) and empty
            # torch's retained host-pin cache, so a later CUDA test in the suite
            # does not inherit the pressure. detach() deliberately leaves the
            # arena alive, so release it explicitly.
            arena.release()
            pin_manager.reconcile(0, allow_shrink=False)


if __name__ == "__main__":
    unittest.main()

import pytest

pytestmark = pytest.mark.process_isolated
