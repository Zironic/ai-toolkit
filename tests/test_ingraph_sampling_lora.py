"""In-graph sampling must render the LoRA, not the frozen base.

`enable_ingraph_training` folds every attached LoRA into the traced trunk.
`enable_ingraph_sampling` strips the same hijacks -- and then never folds them
back, so the compiled sampling trunk is pure base math. Previews would silently
show the base model while training reported a falling loss.

This is reachable in exactly the configuration in-graph sampling exists for.
`BaseSDTrainProcess` sets `can_merge_in = False` whenever `quantize` or
`layer_offloading` is on, so `generate_images` cannot take its merge-in
shortcut: the LoRA stays a live forward hijack for the whole sample.

The multiplier has to be a live graph input rather than a trace-time constant.
`generate_images` assigns `network.multiplier = gen_config.network_multiplier`
per image inside `with network:`, and krea2's `reuse_ingraph` path keeps one
trunk across all of them -- a folded scalar would pin every preview to the first
image's multiplier. `network._update_torch_multiplier` rebinds `torch_multiplier`
to a fresh tensor, so the trunk cannot hold that tensor either; it holds its own
and refreshes the value in place.
"""

import unittest
import weakref

import torch
import torch.nn as nn

from extensions_built_in.diffusion_models.krea2.src.mmdit import (
    CompileRegionError,
    SingleMMDiTConfig,
    SingleStreamDiT,
)
from toolkit.memory_management.ingraph_stream import block_tensor_views, pack_block_host


class _FakeNetwork:
    """The Network surface _collect_lora_entry and the refresh actually read."""

    def __init__(self, multiplier: float = 1.0):
        self.is_active = True
        self.is_merged_in = False
        self.is_lorm = False
        self.vector_gates = None
        self._multiplier = multiplier
        self.torch_multiplier = torch.tensor([multiplier])

    @property
    def multiplier(self):
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value):
        # Mirrors _update_torch_multiplier: a NEW tensor every time, so anything
        # that captured the old one silently goes stale.
        self._multiplier = value
        self.torch_multiplier = torch.tensor([value])


class _FakeLoraModule:
    """A LoRA hijack: child.forward is bound here, base hangs off org_forward."""

    def __init__(self, child: nn.Linear, network: _FakeNetwork, rank: int = 2):
        self.lora_down = nn.Linear(child.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, child.out_features, bias=False)
        # Non-zero up-weights, or the LoRA is invisible and every test passes.
        nn.init.normal_(self.lora_up.weight, std=0.5)
        self.scale = 0.25
        self.dropout = None
        self.module_dropout = None
        self.rank_dropout = None
        self.network_ref = weakref.ref(network)
        self.org_forward = child.forward
        child.forward = self.forward

    def forward(self, x):
        network = self.network_ref()
        base = self.org_forward(x)
        if not network.is_active or network.is_merged_in or network._multiplier == 0:
            return base
        multiplier = float(network.torch_multiplier.reshape(()))
        lora = self.lora_up(self.lora_down(x)) * self.scale * multiplier
        return base + lora


def _model(layers: int = 2) -> SingleStreamDiT:
    torch.manual_seed(123)
    model = SingleStreamDiT(
        SingleMMDiTConfig(
            features=32, tdim=16, txtdim=32, heads=4, multiplier=1,
            layers=layers, patch=1, channels=4, txtheads=4, txtkvheads=4,
        )
    ).eval()
    model.requires_grad_(False)
    return model


def _attach_loras(model, network, block_indices=(0,)):
    loras = {}
    for index in block_indices:
        for name, child in model._block_linear_entries(model.blocks[index]):
            loras[(index, name)] = _FakeLoraModule(child, network)
    return loras


def _block_inputs(model):
    return (
        torch.randn(1, 5, 32),
        torch.randn(1, 192),
        model.posemb(torch.zeros(1, 5, 3)),
        torch.ones(1, 1, 5, 5, dtype=torch.bool),
    )


class SamplingTrunkFoldsLoraTests(unittest.TestCase):
    def setUp(self):
        self.model = _model(layers=1)
        self.network = _FakeNetwork(multiplier=1.0)
        self.loras = _attach_loras(self.model, self.network)
        self.block = self.model.blocks[0]
        self.inputs = _block_inputs(self.model)

    def _pack(self):
        return pack_block_host(
            "blocks.0",
            self.model._block_linear_entries(self.block),
            repoint=False,
            pin=False,
        )

    def test_the_lora_actually_moves_the_eager_output(self):
        """Guard the guard: a zero-effect LoRA would make every test below pass."""
        x, vec, freqs, mask = self.inputs
        with torch.no_grad():
            with_lora = self.block(x, vec, freqs, mask)
            self.network.is_active = False
            without_lora = self.block(x, vec, freqs, mask)
            self.network.is_active = True
        self.assertFalse(torch.allclose(with_lora, without_lora))

    def test_streamed_block_without_loras_renders_the_base(self):
        """The defect, stated positively: pack views carry base weights only."""
        pack = self._pack()
        x, vec, freqs, mask = self.inputs
        with torch.no_grad():
            eager = self.block(x, vec, freqs, mask)
            self.network.is_active = False
            base = self.block(x, vec, freqs, mask)
            self.network.is_active = True
            streamed = self.block.forward_streamed(
                x, vec, freqs, mask,
                block_tensor_views(pack.host_flat, pack), pack.fp8_flags,
            )
        torch.testing.assert_close(streamed, base, rtol=0, atol=0)
        self.assertFalse(torch.allclose(streamed, eager))

    def test_streamed_block_with_loras_matches_eager(self):
        pack = self._pack()
        x, vec, freqs, mask = self.inputs
        entries = {
            name: self.model._collect_lora_entry(child)
            for name, child in self.model._block_linear_entries(self.block)
        }
        self.assertEqual(len([e for e in entries.values() if e is not None]), 8)

        multiplier = torch.tensor(1.0)
        lora_args = self.model._block_lora_tuple(entries, multiplier)
        with torch.no_grad():
            eager = self.block(x, vec, freqs, mask)
            streamed = self.block.forward_streamed(
                x, vec, freqs, mask,
                block_tensor_views(pack.host_flat, pack), pack.fp8_flags,
                loras=lora_args,
            )
        torch.testing.assert_close(streamed, eager, rtol=1e-5, atol=1e-6)

    def test_streamed_block_tracks_a_multiplier_change_without_rebuilding(self):
        """generate_images sets network.multiplier per image; krea2 reuses the
        trunk across images. The folded-float trunk would freeze image 0."""
        pack = self._pack()
        x, vec, freqs, mask = self.inputs
        entries = {
            name: self.model._collect_lora_entry(child)
            for name, child in self.model._block_linear_entries(self.block)
        }
        multiplier = torch.tensor(1.0)

        def streamed():
            # Built INSIDE the call, exactly as the block fns do it: `scale *
            # multiplier` is a traced op there, so it re-reads the live tensor.
            # Hoisting it out would bake in whatever value was current.
            return self.block.forward_streamed(
                x, vec, freqs, mask,
                block_tensor_views(pack.host_flat, pack), pack.fp8_flags,
                loras=self.model._block_lora_tuple(entries, multiplier),
            )

        with torch.no_grad():
            # Nothing is rebuilt; only the tensor's value changes.
            multiplier.fill_(0.35)
            self.network.multiplier = 0.35
            torch.testing.assert_close(
                streamed(), self.block(x, vec, freqs, mask), rtol=1e-5, atol=1e-6
            )
            multiplier.fill_(0.0)
            self.network.multiplier = 0.0
            torch.testing.assert_close(
                streamed(), self.block(x, vec, freqs, mask), rtol=1e-5, atol=1e-6
            )


class SamplingEnableCollectsLorasTests(unittest.TestCase):
    """Drive the collection seam directly. Going through enable_ingraph_sampling
    would build real pinned packs against the process-global pin ledger, which
    other tests assert on (see ticket f2aceba); the pack build itself is covered
    by tests/test_ingraph_sampling_borrow_counts.py."""

    def test_every_lora_leaf_of_every_block_is_collected(self):
        model = _model(layers=2)
        network = _FakeNetwork()
        _attach_loras(model, network, block_indices=(0, 1))

        loras, found = model._collect_block_loras((0, 1))
        self.assertIs(found, network)
        self.assertEqual(sorted(loras), [0, 1])
        self.assertEqual(sum(len(v) for v in loras.values()), 16)

        multiplier = model._ensure_ingraph_lora_multiplier(network, loras)
        self.assertIsNotNone(multiplier)
        self.assertEqual(multiplier.shape, ())

    def test_no_lora_means_no_entries_and_no_multiplier(self):
        model = _model(layers=1)
        loras, network = model._collect_block_loras((0,))
        self.assertEqual(loras, {})
        self.assertIsNone(network)
        self.assertIsNone(model._ensure_ingraph_lora_multiplier(network, loras))
        self.assertIsNone(model._ingraph_lora_multiplier)


class MultiplierRefreshTests(unittest.TestCase):
    """The live scalar must mirror what the eager LoRA forward would do, and keep
    its identity so torch.compile never sees a new tensor."""

    def setUp(self):
        self.model = _model(layers=1)
        self.network = _FakeNetwork(multiplier=1.0)
        _attach_loras(self.model, self.network)
        loras, network = self.model._collect_block_loras((0,))
        self.multiplier = self.model._ensure_ingraph_lora_multiplier(network, loras)

    def _refresh(self):
        self.model._refresh_ingraph_lora_multiplier()
        return float(self.multiplier)

    def test_refresh_tracks_the_networks_multiplier_in_place(self):
        self.network.multiplier = 0.4
        self.assertAlmostEqual(self._refresh(), 0.4)
        self.assertIs(self.model._ingraph_lora_multiplier, self.multiplier)

    def test_inactive_merged_or_zero_network_renders_the_base(self):
        for mutate in (
            lambda: setattr(self.network, "is_active", False),
            lambda: setattr(self.network, "is_merged_in", True),
            lambda: setattr(self.network, "multiplier", 0.0),
        ):
            self.setUp()
            mutate()
            self.assertEqual(self._refresh(), 0.0)

    def test_a_multiplier_that_stops_being_scalar_fails_closed(self):
        self.network.torch_multiplier = torch.tensor([1.0, 0.5])
        self.network._multiplier = [1.0, 0.5]
        with self.assertRaises(RuntimeError):
            self.model._refresh_ingraph_lora_multiplier()


class FoldedLoraCompilesTests(unittest.TestCase):
    """The fold has to survive fullgraph tracing, and a per-image multiplier
    change must not trigger a recompile -- the trunk is 28 blocks of Krea2."""

    def test_fullgraph_no_breaks_and_multiplier_change_does_not_recompile(self):
        model = _model(layers=1)
        network = _FakeNetwork(multiplier=1.0)
        _attach_loras(model, network)
        block = model.blocks[0]
        entries, found = model._collect_block_loras((0,))
        multiplier = model._ensure_ingraph_lora_multiplier(found, entries)
        pack = pack_block_host(
            "blocks.0", model._block_linear_entries(block), repoint=False, pin=False
        )
        leaf_args = block_tensor_views(pack.host_flat, pack)
        x, vec, freqs, mask = _block_inputs(model)

        def fn(x, vec, freqs, mask):
            return block.forward_streamed(
                x, vec, freqs, mask, leaf_args, pack.fp8_flags,
                loras=model._block_lora_tuple(entries[0], multiplier),
            )

        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="eager")

        with torch.no_grad():
            torch.testing.assert_close(compiled(x, vec, freqs, mask), fn(x, vec, freqs, mask))
            frames = torch._dynamo.utils.counters["frames"]["ok"]

            # A new multiplier for the next preview image: value changes, tensor
            # identity does not, so the compiled graph is reused as-is.
            network.multiplier = 0.3
            model._refresh_ingraph_lora_multiplier()
            torch.testing.assert_close(compiled(x, vec, freqs, mask), fn(x, vec, freqs, mask))

        self.assertEqual(sum(torch._dynamo.utils.counters["graph_break"].values()), 0)
        self.assertEqual(torch._dynamo.utils.counters["frames"]["ok"], frames)
        # ...and it actually tracked, rather than reusing a stale folded scalar.
        self.assertAlmostEqual(float(multiplier), 0.3, places=6)


class SamplingLoraFailsClosedTests(unittest.TestCase):
    def test_chained_lora_hijacks_fail_closed(self):
        """assistant_lora applies a SECOND network to the same Linears. Only the
        outer hijack is reachable from child.forward; folding it alone would
        silently drop the inner one."""
        model = _model(layers=1)
        network_a = _FakeNetwork()
        network_b = _FakeNetwork()
        child = model.blocks[0].attn.wq
        _FakeLoraModule(child, network_a)
        _FakeLoraModule(child, network_b)  # chains onto the first
        with self.assertRaises(CompileRegionError) as caught:
            model._collect_lora_entry(child)
        self.assertIn("lora_chained", caught.exception.reasons)

    def test_two_networks_across_leaves_fail_closed(self):
        model = _model(layers=1)
        network_a = _FakeNetwork()
        network_b = _FakeNetwork()
        _FakeLoraModule(model.blocks[0].attn.wq, network_a)
        _FakeLoraModule(model.blocks[0].attn.wk, network_b)
        with self.assertRaises(RuntimeError) as caught:
            model.enable_ingraph_sampling()
        self.assertIn("lora_multiple_networks", str(caught.exception))
        model.disable_ingraph_sampling()

    def test_unrecognized_forward_hijack_on_a_packed_linear_fails_closed(self):
        model = _model(layers=1)
        child = model.blocks[0].mlp.up

        class _Mystery:
            def __init__(self, child):
                self.org_forward = child.forward
                child.forward = self.forward

            def forward(self, x):
                return self.org_forward(x) * 2.0

        _Mystery(child)
        with self.assertRaises(RuntimeError) as caught:
            model.enable_ingraph_sampling()
        self.assertIn("unknown_forward_hijack", str(caught.exception))
        model.disable_ingraph_sampling()


if __name__ == "__main__":
    unittest.main()
