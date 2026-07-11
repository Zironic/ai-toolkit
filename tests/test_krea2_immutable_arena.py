import pytest
import torch

from extensions_built_in.diffusion_models.krea2.src.immutable_arena import (
    KreaImmutableArenaAdapter,
    KreaImmutableArenaError,
)
from extensions_built_in.diffusion_models.krea2.src.mmdit import (
    SingleMMDiTConfig,
    SingleStreamDiT,
)
from toolkit.memory_management import ingraph_stream
from toolkit.memory_management.canonical_arena import CanonicalArena
from toolkit.memory_management.ingraph_stream import LoraEntry
from toolkit.memory_management.residency import ResidencyPlan, ResidencyState

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"), pytest.mark.leaky]


def _model():
    torch.manual_seed(123)
    model = SingleStreamDiT(
        SingleMMDiTConfig(
            features=32,
            tdim=16,
            txtdim=32,
            heads=4,
            multiplier=1,
            layers=1,
            patch=1,
            channels=4,
            txtheads=4,
            txtkvheads=4,
        )
    ).cuda().eval()
    model.requires_grad_(False)
    return model


def _inputs(model, *, requires_grad=False):
    x = torch.randn(1, 5, 32, device="cuda", requires_grad=requires_grad)
    vec = torch.randn(1, 192, device="cuda")
    freqs = model.posemb(torch.zeros(1, 5, 3, device="cuda"))
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool, device="cuda")
    return x, vec, freqs, mask


def _canonical_fixture(resident_names=("attn.wq", "mlp.gate")):
    model = _model()
    entries = model._block_linear_entries(model.blocks[0])
    reference_args = tuple(
        (
            child.weight.detach().clone(),
            None if child.bias is None else child.bias.detach().clone(),
            None,
        )
        for _name, child in entries
    )
    identities = tuple(id(child.weight) for _name, child in entries)
    arena = CanonicalArena()
    arena.canonicalize({"blocks.0": entries})
    canonical_identities = tuple(id(child.weight) for _name, child in entries)
    state = ResidencyState(arena, "cuda")
    state.reconcile(
        ResidencyPlan.build(
            "train", (("blocks.0", name) for name in resident_names)
        )
    )
    return model, arena, state, reference_args, identities, canonical_identities


def test_partial_residency_eager_parity_and_parameter_stability():
    model, arena, state, reference_args, _old_ids, canonical_ids = _canonical_fixture()
    adapter = KreaImmutableArenaAdapter(model, state)
    inputs = _inputs(model)
    try:
        with torch.no_grad():
            expected = model.blocks[0].forward_streamed(
                *inputs,
                reference_args,
                (False,) * 8,
            )
            actual = adapter.forward_block(0, *inputs)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        transfer = adapter.block_plans[0].transfer
        assert transfer is not None
        assert set(transfer.streamed_leaf_names) == {
            "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.up", "mlp.down"
        }
        assert tuple(id(child.weight) for _name, child in model._block_linear_entries(
            model.blocks[0]
        )) == canonical_ids
    finally:
        ingraph_stream.drain_fetch_runtime()
        arena.release()


def test_checkpoint_recompute_reuses_source_map_and_all_loras_get_gradients():
    model, arena, state, _reference_args, _old_ids, _canonical_ids = _canonical_fixture()
    rank = 4
    loras = {}
    trainable = []
    for name, child in model._block_linear_entries(model.blocks[0]):
        a = torch.randn(rank, child.in_features, device="cuda", requires_grad=True)
        b = torch.randn(child.out_features, rank, device="cuda", requires_grad=True)
        loras[name] = LoraEntry(a=a, b=b, scale=0.125)
        trainable.extend((a, b))
    adapter = KreaImmutableArenaAdapter(
        model,
        state,
        loras_by_block={0: loras},
        lora_multiplier=torch.tensor(0.75, device="cuda"),
    )
    model.enable_gradient_checkpointing()
    source_maps = []
    original_leaf_args = adapter._leaf_args

    def record_sources(block_plan, compact_flat=None):
        source_maps.append(
            (
                block_plan.transfer.fingerprint,
                tuple(sorted(state.plan.resident_leaf_keys)),
            )
        )
        return original_leaf_args(block_plan, compact_flat)

    adapter._leaf_args = record_sources
    inputs = _inputs(model, requires_grad=True)
    ingraph_stream.fetch_stats(reset=True)
    try:
        out = adapter.forward_blocks(*inputs)
        out.square().mean().backward()
        torch.cuda.synchronize()
        assert len(source_maps) == 2
        assert source_maps[0] == source_maps[1]
        assert ingraph_stream.fetch_stats()["fetches"] == 2
        assert all(tensor.grad is not None for tensor in trainable)
        assert all(torch.isfinite(tensor.grad).all() for tensor in trainable)
    finally:
        ingraph_stream.drain_fetch_runtime()
        arena.release()


def test_model_enable_routes_training_trunk_and_plan_change_fails_closed():
    model, arena, state, _reference_args, _old_ids, _canonical_ids = _canonical_fixture()
    inputs = _inputs(model, requires_grad=True)
    try:
        adapter = model.enable_immutable_arena_eager(arena, state)
        out = model._blocks_trunk(*inputs)
        out.sum().backward()
        torch.cuda.synchronize()
        assert out.device.type == "cuda"

        state.reconcile(
            ResidencyPlan.build("sample", (("blocks.0", "attn.wk"),))
        )
        with pytest.raises(KreaImmutableArenaError, match="residency_plan_changed"):
            adapter.forward_block(0, *_inputs(model))
    finally:
        model.disable_immutable_arena_eager()
        ingraph_stream.drain_fetch_runtime()
        arena.release()
