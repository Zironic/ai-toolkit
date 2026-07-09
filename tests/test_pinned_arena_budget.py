"""Ticket 534ea49 Phase 2 Slice B: the arena must consume the SAME resolved
pin budget the per-tensor path would have, and must be the sole pinner when
active -- no per-tensor cudaHostRegister during arena-enabled attach, and a
re-attach must only ever request the still-uncommitted delta."""

import unittest
from unittest import mock

import torch
import torch.nn as nn

from toolkit.memory_management import pin_manager
from toolkit.memory_management.manager import MemoryManager
from toolkit.memory_management.pinned_arena import PinnedWeightArena


def _linear(in_f=64, out_f=64, bias=True):
    layer = nn.Linear(in_f, out_f, bias=bias)
    layer.weight.requires_grad_(False)
    if bias:
        layer.bias.requires_grad_(False)
    return layer


class ArenaBuildBudgetCapTests(unittest.TestCase):
    """CPU-level: PinnedWeightArena.build's own budget_bytes gate."""

    def _block_bytes(self, layer):
        total = layer.weight.numel() * layer.weight.element_size()
        if layer.bias is not None:
            total += layer.bias.numel() * layer.bias.element_size()
        return total

    def test_blocks_beyond_budget_are_pageable_in_iteration_order(self):
        layers = [_linear() for _ in range(4)]
        one_block_bytes = self._block_bytes(layers[0])
        # Budget for ~2 blocks (with headroom so the 3rd/4th definitely miss).
        budget = int(one_block_bytes * 2.2)

        arena = PinnedWeightArena()
        entries_by_block = {
            f"blocks.{i}": [(f"blocks.{i}.lin", layer)]
            for i, layer in enumerate(layers)
        }
        stats = arena.build(entries_by_block, budget_bytes=budget)

        pinned_flags = [arena.block_pack(f"blocks.{i}").pinned for i in range(4)]
        # First blocks (in iteration order) win the budget; later ones fall
        # back to pageable -- never the reverse, and never a partial mix that
        # ignores insertion order.
        self.assertEqual(pinned_flags, sorted(pinned_flags, reverse=True))
        self.assertEqual(sum(pinned_flags), 2)
        self.assertEqual(stats.pageable_blocks, 2)
        # Still repointed even though pageable.
        for i, layer in enumerate(layers):
            self.assertEqual(
                layer.weight.untyped_storage().data_ptr(),
                arena.block_pack(f"blocks.{i}").host_flat.untyped_storage().data_ptr(),
            )

    def test_none_budget_is_unlimited(self):
        layers = [_linear() for _ in range(3)]
        arena = PinnedWeightArena()
        entries_by_block = {
            f"blocks.{i}": [(f"blocks.{i}.lin", layer)]
            for i, layer in enumerate(layers)
        }
        stats = arena.build(entries_by_block, budget_bytes=None)
        self.assertEqual(stats.pageable_blocks, 0)
        self.assertTrue(all(arena.block_pack(f"blocks.{i}").pinned for i in range(3)))

    def test_zero_budget_makes_every_block_pageable_but_still_repointed(self):
        layer = _linear()
        arena = PinnedWeightArena()
        stats = arena.build({"blocks.0": [("lin", layer)]}, budget_bytes=0)
        self.assertEqual(stats.pageable_blocks, 1)
        self.assertFalse(arena.block_pack("blocks.0").pinned)
        self.assertEqual(
            layer.weight.untyped_storage().data_ptr(),
            arena.block_pack("blocks.0").host_flat.untyped_storage().data_ptr(),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "pinning requires CUDA")
    def test_rebuild_credits_released_previous_flat_against_the_budget(self):
        """Regression (whole-group rebuild): growing a pinned block by a few
        linears releases its old flat, so a near-zero remaining budget must
        not force the grown block pageable -- the released bytes are credited
        back inside build()."""
        a = _linear()
        b = _linear()
        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin_a", a)]}, budget_bytes=None)
        self.assertTrue(arena.block_pack("blocks.0").pinned)
        old_bytes = arena.block_pack("blocks.0").required_pin_bytes

        # Remaining budget only covers the DELTA (b's bytes + alignment),
        # nowhere near the full new flat -- the credit for the released old
        # flat must make up the difference.
        delta_budget = self._block_bytes(b) + 512
        arena.build(
            {"blocks.0": [("lin_a", a), ("lin_b", b)]}, budget_bytes=delta_budget
        )
        new_pack = arena.block_pack("blocks.0")
        self.assertTrue(new_pack.pinned)
        self.assertGreater(new_pack.required_pin_bytes, old_bytes)
        self.assertTrue(arena.is_current(a))
        self.assertTrue(arena.is_current(b))
        arena.release()


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Linear(d, d, bias=True)
        self.b = nn.Linear(d, d, bias=True)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.b(self.a(x))


class _Model(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(d) for _ in range(n)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


@unittest.skipUnless(torch.cuda.is_available(), "attach needs CUDA")
class ArenaIsSolePinnerDuringAttachTests(unittest.TestCase):
    def setUp(self):
        self._ledger_before = pin_manager.pinned_bytes_by_kind()
        pin_manager._LEDGER.clear()

        def _restore():
            pin_manager._LEDGER.clear()
            pin_manager._LEDGER.update(self._ledger_before)

        self.addCleanup(_restore)

    def _offload_ids(self, model):
        return {id(m) for m in model.modules() if isinstance(m, nn.Linear)}

    def test_no_per_tensor_pin_calls_when_arena_enabled(self):
        device = torch.device("cuda:0")
        model = _Model(64, 3)
        param_ptrs = {p.data_ptr() for p in model.parameters()}
        with mock.patch.object(
            pin_manager, "pin_tensor_in_place", wraps=pin_manager.pin_tensor_in_place
        ) as spy:
            MemoryManager.attach(
                model, device, _offload_module_ids=self._offload_ids(model),
                use_pinned_arena=True,
            )
        try:
            # The arena's flats legitimately register via pin_tensor_in_place
            # (pin_register carves uint8 flat buffers); what must never happen
            # is a PER-TENSOR pin of a module parameter (Slice B2: the
            # pin/unpin/repin churn the arena exists to remove).
            for call in spy.call_args_list:
                pinned_tensor = call.args[0]
                self.assertEqual(pinned_tensor.dtype, torch.uint8)
                self.assertNotIn(pinned_tensor.data_ptr(), param_ptrs)
            arena = model._mm_weight_arena
            self.assertIsNotNone(arena)
            self.assertTrue(arena.has_all_blocks_pinned())
        finally:
            MemoryManager.detach(model)

    def test_reattach_requests_only_the_uncommitted_delta(self):
        device = torch.device("cuda:0")
        model = _Model(64, 2)
        MemoryManager.attach(
            model, device, _offload_module_ids=self._offload_ids(model),
            use_pinned_arena=True,
        )
        arena = model._mm_weight_arena
        MemoryManager.detach(model)

        with mock.patch.object(
            PinnedWeightArena, "build", wraps=arena.build
        ) as build_spy:
            MemoryManager.attach(
                model, device, _offload_module_ids=self._offload_ids(model),
                use_pinned_arena=True,
            )
        try:
            # Nothing new to build -- every child is already arena-current,
            # so build() must not even be called (zero churn on re-attach).
            build_spy.assert_not_called()
        finally:
            MemoryManager.detach(model)

    def test_plan_budgets_request_is_full_desired_not_a_committed_delta(self):
        """On a re-attach that GROWS the streamed set, attach must request the
        full desired from plan_budgets, NOT `desired - committed`. The plan's
        headroom is already net of the arena's committed bytes (they sit in
        DXGI usage), so min(desired, usable) is exactly the room for new pins;
        the arena's build() only rebuilds stale/pageable groups, so a grant
        larger than the rebuild need can't over-pin. The delta form
        UNDERSHOOTS -- when the set grows a couple of blocks, `desired -
        committed` caps the grant below the bytes those blocks actually need
        and forces them pageable despite real free headroom (observed live,
        768px fp8). (The double-count to avoid is subtracting committed on
        BOTH sides -- request AND grant; requesting full desired with no
        second subtraction in _build_pinned_arena is correct.)"""
        device = torch.device("cuda:0")
        model = _Model(64, 2)
        MemoryManager.attach(
            model, device, _offload_module_ids={id(model.blocks[0].a)},
            use_pinned_arena=True,
        )
        arena = model._mm_weight_arena
        committed_before = arena.committed_pinned_bytes()
        self.assertGreater(committed_before, 0)
        MemoryManager.detach(model)

        grown_ids = {id(model.blocks[0].a), id(model.blocks[1].a)}
        desired_total = MemoryManager._desired_pin_bytes_for_offload_ids(
            model, grown_ids, None
        )
        captured = {}
        real_plan = pin_manager.plan_budgets

        def _spy_plan(**kwargs):
            captured["offloaded_weight_bytes"] = kwargs.get("offloaded_weight_bytes")
            return real_plan(**kwargs)

        with mock.patch.object(pin_manager, "plan_budgets", side_effect=_spy_plan):
            MemoryManager.attach(
                model, device, _offload_module_ids=grown_ids,
                use_pinned_arena=True,
            )
        try:
            self.assertIn("offloaded_weight_bytes", captured)
            self.assertEqual(captured["offloaded_weight_bytes"], desired_total)
            # And the growth actually pinned: block 1 is not pageable.
            block1_key = arena.arena_block_of(model.blocks[1].a)
            self.assertTrue(arena.block_pack(block1_key).pinned)
        finally:
            MemoryManager.detach(model)

    def test_partial_block_growth_rebuilds_the_whole_group(self):
        """Regression (live run: `borrow refused: stale_modules=5/8`): the
        smart-training plan splits blocks per-LAYER, so training's arena
        build can cover a subset of a block's linears. The sampling attach
        used to rebuild the block with only the missing linears, bumping the
        generation and stranding the previously-built siblings as stale --
        every subsequent borrow refused. The rebuild must include the WHOLE
        group so all members land in one flat at one generation."""
        device = torch.device("cuda:0")
        model = _Model(64, 1)
        # "Training": only blocks.0.a streamed (per-layer split).
        MemoryManager.attach(
            model, device, _offload_module_ids={id(model.blocks[0].a)},
            use_pinned_arena=True,
        )
        arena = model._mm_weight_arena
        self.assertTrue(arena.is_current(model.blocks[0].a))
        MemoryManager.detach(model)

        # "Sampling": the whole block streams. The rebuild must produce ONE
        # flat holding BOTH linears, both current.
        MemoryManager.attach(
            model, device,
            _offload_module_ids={id(model.blocks[0].a), id(model.blocks[0].b)},
            use_pinned_arena=True,
        )
        try:
            self.assertTrue(arena.is_current(model.blocks[0].a))
            self.assertTrue(arena.is_current(model.blocks[0].b))
            block_key = arena.arena_block_of(model.blocks[0].a)
            self.assertEqual(block_key, arena.arena_block_of(model.blocks[0].b))
            borrowed = arena.try_borrow_pack(
                block_key,
                [("a", model.blocks[0].a), ("b", model.blocks[0].b)],
            )
            self.assertIsNotNone(borrowed)
            self.assertTrue(borrowed.pinned)
        finally:
            MemoryManager.detach(model)

    def test_arena_flats_use_register_mechanism_not_pin_alloc(self):
        """Regression (live run): pin_alloc goes through torch's caching
        host allocator, which rounds every request up to a power-of-two
        bucket -- 8.86 GiB of arena flats committed 12.70 GiB of DXGI
        usage, and the invisible ~40% overhead starved the last blocks of
        the model. Arena flats must use cudaHostRegister (exact DXGI cost,
        budget returned immediately on release) via pin_register."""
        layer = _linear()
        arena = PinnedWeightArena()
        with mock.patch.object(
            pin_manager, "pin_register", wraps=pin_manager.pin_register
        ) as register_spy, mock.patch.object(
            pin_manager, "pin_alloc", wraps=pin_manager.pin_alloc
        ) as alloc_spy:
            arena.build({"blocks.0": [("lin", layer)]}, budget_bytes=None)
        try:
            register_spy.assert_called()
            alloc_spy.assert_not_called()
        finally:
            arena.release()

    @unittest.skipUnless(torch.cuda.is_available(), "pinning requires CUDA")
    def test_pin_register_release_returns_budget_immediately(self):
        """cudaHostRegister'd bytes must leave the ledger on release without
        needing _empty_host_pin_cache (the pin_alloc asymmetry)."""
        kind = "weights"
        before = pin_manager.pinned_bytes_by_kind().get(kind, 0)
        handle = pin_manager.pin_register(1 << 20, kind, required=True)
        self.assertTrue(handle.pinned)
        # cudaHostRegister'd memory reports is_pinned()==False (torch only
        # tracks its own caching-allocator pins); the registration table is
        # the source of truth.
        self.assertTrue(pin_manager.is_host_pinned(handle.tensor))
        self.assertEqual(
            pin_manager.pinned_bytes_by_kind().get(kind, 0), before + (1 << 20)
        )
        pin_manager.release(handle)
        self.assertFalse(handle.pinned)
        self.assertEqual(pin_manager.pinned_bytes_by_kind().get(kind, 0), before)

    @unittest.skipUnless(torch.cuda.is_available(), "attach needs CUDA")
    def test_pageable_block_is_retried_when_budget_recovers(self):
        """Self-healing: a block that fell back to a pageable flat under a
        tight budget stays arena-current, so the stale check alone never
        revisits it -- strict ingraph would fail on it forever. A later
        attach with real budget must rebuild it pinned."""
        device = torch.device("cuda:0")
        model = _Model(64, 1)
        real_plan = pin_manager.plan_budgets

        def _zero_grant_plan(**kwargs):
            # Force zero pinnable headroom through EVERY path attach reads:
            # the arena now derives its build budget from the weight-tier
            # available_for_pin (the sole-pinner "use all usable weight-tier
            # headroom" rule), so zeroing weight_budget_bytes alone no longer
            # forces pageable -- the direct available_for_pin below does.
            plan = dict(real_plan(**kwargs))
            plan["weight_budget_bytes"] = 0
            plan["bounce_budget_bytes"] = 0
            plan["reserve_bytes"] = int(plan.get("headroom_bytes") or 0)
            return plan

        with mock.patch.object(pin_manager, "plan_budgets", side_effect=_zero_grant_plan), \
                mock.patch.object(pin_manager, "available_for_pin", return_value=0):
            MemoryManager.attach(
                model, device, _offload_module_ids=self._offload_ids(model),
                use_pinned_arena=True,
            )
        arena = model._mm_weight_arena
        block_key = arena.arena_block_of(model.blocks[0].a)
        self.assertFalse(arena.block_pack(block_key).pinned)
        MemoryManager.detach(model)

        MemoryManager.attach(
            model, device, _offload_module_ids=self._offload_ids(model),
            use_pinned_arena=True,
        )
        try:
            self.assertTrue(arena.block_pack(block_key).pinned)
            self.assertTrue(arena.is_current(model.blocks[0].a))
        finally:
            MemoryManager.detach(model)

    def test_attach_reconciles_host_pin_cache_before_planning_arena_budget(self):
        """Regression (live run: free=0.00 with 12.70 GiB DXGI usage against
        an 8.86 GiB ledger): torch's caching host allocator retains every
        staging buffer page-locked forever, and that retained cache sits
        inside the headroom plan_budgets measures. pin_alloc reconciles on
        refusal, but the arena's per-block pin decision is made from the
        PLAN's grant -- so attach must reconcile (empty the host cache,
        allow_shrink=False: weights never evict bounce) BEFORE plan_budgets,
        or blocks flip pageable without ever attempting a pin."""
        device = torch.device("cuda:0")
        model = _Model(64, 2)
        calls = []
        real_reconcile = pin_manager.reconcile
        real_plan = pin_manager.plan_budgets

        def _spy_reconcile(*args, **kwargs):
            calls.append(("reconcile", kwargs.get("allow_shrink")))
            return real_reconcile(*args, **kwargs)

        def _spy_plan(**kwargs):
            calls.append(("plan_budgets", None))
            return real_plan(**kwargs)

        with mock.patch.object(pin_manager, "reconcile", side_effect=_spy_reconcile), \
                mock.patch.object(pin_manager, "plan_budgets", side_effect=_spy_plan):
            MemoryManager.attach(
                model, device, _offload_module_ids=self._offload_ids(model),
                use_pinned_arena=True,
            )
        try:
            reconcile_idx = calls.index(("reconcile", False))
            plan_idx = calls.index(("plan_budgets", None))
            self.assertLess(
                reconcile_idx, plan_idx,
                "host-pin cache must be reclaimed before plan_budgets "
                "measures headroom",
            )
        finally:
            MemoryManager.detach(model)

    def test_attach_without_arena_does_not_reconcile_before_planning(self):
        """Flag-off behavior preservation: the pre-plan reconcile is arena-
        only (emptying the host cache changes allocator behavior for the
        legacy per-tensor path)."""
        device = torch.device("cuda:0")
        model = _Model(64, 1)
        with mock.patch.object(
            pin_manager, "reconcile", wraps=pin_manager.reconcile
        ) as spy:
            MemoryManager.attach(
                model, device, _offload_module_ids=self._offload_ids(model),
                use_pinned_arena=False,
            )
        try:
            for call in spy.call_args_list:
                self.assertNotEqual(
                    call.kwargs.get("allow_shrink"), False,
                    "non-arena attach must not run the arena's pre-plan "
                    "cache reclaim",
                )
        finally:
            MemoryManager.detach(model)

    def test_reserve_pin_for_ingraph_pinned_weight_gib_formula_does_not_starve_arena_growth(self):
        """Regression: inference_resident's two attach() call sites used to
        force pinned_weight_gib=0.0 whenever reserve_pin_for_ingraph=True,
        reasoning it was irrelevant with the arena active (per-tensor pinning
        is bypassed either way). But that budget also sizes the ARENA itself
        (Slice B) -- zeroing it starved any block newly added to the arena at
        sampling time (e.g. resident during training, streamed for sampling),
        forcing it pageable. Ingraph's borrow then failed closed to an owned
        pack with zero headroom left (the arena already held the real
        budget), surfacing as 'pin refused (ingraph_pack)... returning
        pageable' followed by 'non_pinned_pack' in strict mode. Fixed formula:
        only zero pinned_weight_gib when the arena is NOT active."""
        device = torch.device("cuda:0")
        model = _Model(64, 2)
        # "Training": only block 0 is streamed (arena covers just that).
        MemoryManager.attach(
            model, device, _offload_module_ids={id(model.blocks[0].a)},
            use_pinned_arena=True,
        )
        arena = model._mm_weight_arena
        self.assertTrue(
            arena.block_pack(arena.arena_block_of(model.blocks[0].a)).pinned
        )
        MemoryManager.detach(model)

        # "Sampling" with reserve_pin_for_ingraph=True: block 1 is NEWLY
        # streamed too. Apply the exact fixed formula from
        # MemoryManager.inference_resident's attach() call sites.
        args = {"pinned_weight_gib": -1.0, "use_pinned_arena": True}
        reserve_pin_for_ingraph = True
        resolved_pinned_weight_gib = (
            0.0
            if reserve_pin_for_ingraph and not args.get("use_pinned_arena", False)
            else args.get("pinned_weight_gib")
        )
        self.assertNotEqual(resolved_pinned_weight_gib, 0.0)

        MemoryManager.attach(
            model, device,
            _offload_module_ids={id(model.blocks[0].a), id(model.blocks[1].a)},
            use_pinned_arena=True,
            pinned_weight_gib=resolved_pinned_weight_gib,
        )
        try:
            block1_key = arena.arena_block_of(model.blocks[1].a)
            self.assertIsNotNone(block1_key)
            self.assertTrue(
                arena.block_pack(block1_key).pinned,
                "newly-streamed block during reserve_pin_for_ingraph must be "
                "pinned, not pageable",
            )
        finally:
            MemoryManager.detach(model)


if __name__ == "__main__":
    unittest.main()
