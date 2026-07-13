"""Slice 5 (IMMUTABLE_TRANSFER_ARENA_PLAN.md): compiled train/sample plans
over one canonical arena + boundary phase switching.

Covers the slice's test bullets: zero graph breaks in the compiled trunks,
bounded unique graphs across repeated phase cycles, same-plan cycles reusing
compiled artifacts, and arena pointers/registrations/ledger fixed across
every boundary. The all-streamed sampling fallback and fail-closed staleness
checks run with compile_blocks=False (same mechanism, no compile cost).
"""

import pytest
import torch

from toolkit.memory_management.immutable_runtime import (
    ImmutableRuntimeError,
    ImmutableTransformerRuntime,
)
from extensions_built_in.diffusion_models.krea2.src.mmdit import (
    SingleMMDiTConfig,
    SingleStreamDiT,
)
from toolkit.memory_management import ingraph_stream, pin_manager
from toolkit.memory_management.adapters import SingleStreamMMDiTAdapter
from toolkit.memory_management.canonical_arena import CanonicalArena
from toolkit.memory_management.ingraph_stream import LoraEntry
from toolkit.memory_management.residency import ResidencyPlan, ResidencyState
from toolkit.models.lokr import LokrModule

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.process_isolated,
]

LAYERS = 2


def _runtime(model, state, *, compile_blocks=True, **finalize_kwargs):
    """Build a runtime through the two-phase lifecycle the trainer uses.

    Phase A (construct) sees no LoRA; phase B (finalize_execution) installs the
    LoRA metadata and builds the permanent programs.
    """
    runtime = ImmutableTransformerRuntime(
        model,
        state,
        architecture_adapter=SingleStreamMMDiTAdapter(),
        compile_blocks=compile_blocks,
    )
    runtime.finalize_execution(**finalize_kwargs)
    return runtime


def _run(executor, *args):
    mode = executor.TRAIN if torch.is_grad_enabled() else executor.SAMPLE
    with executor.execution(mode):
        return executor.run(args[0], args[1:])

def _model():
    torch.manual_seed(123)
    model = SingleStreamDiT(
        SingleMMDiTConfig(
            features=32,
            tdim=16,
            txtdim=32,
            heads=4,
            multiplier=1,
            layers=LAYERS,
            patch=1,
            channels=4,
            txtheads=4,
            txtkvheads=4,
        )
    ).cuda().eval()
    model.requires_grad_(False)
    return model


_ADAPTER = SingleStreamMMDiTAdapter()


def _inputs(model, *, requires_grad=False):
    x = torch.randn(1, 5, 32, device="cuda", requires_grad=requires_grad)
    vec = torch.randn(1, 192, device="cuda")
    freqs = model.posemb(torch.zeros(1, 5, 3, device="cuda"))
    mask = torch.ones(1, 1, 5, 5, dtype=torch.bool, device="cuda")
    return x, vec, freqs, mask


def _fixture():
    model = _model()
    entries_by_block = {
        f"blocks.{index}": list(_ADAPTER.leaf_entries(model.blocks[index]))
        for index in range(LAYERS)
    }
    reference_args_by_block = {
        index: tuple(
            (
                child.weight.detach().clone().cuda(),
                None if child.bias is None else child.bias.detach().clone().cuda(),
            )
            for _name, child in entries_by_block[f"blocks.{index}"]
        )
        for index in range(LAYERS)
    }
    arena = CanonicalArena()
    arena.canonicalize(entries_by_block)
    state = ResidencyState(arena, "cuda")
    return model, arena, state, reference_args_by_block


def _loras(model, rank=4):
    loras = {}
    trainable = []
    for index in range(LAYERS):
        block_loras = {}
        for name, child in _ADAPTER.leaf_entries(model.blocks[index]):
            a = torch.randn(rank, child.in_features, device="cuda", requires_grad=True)
            b = torch.randn(child.out_features, rank, device="cuda", requires_grad=True)
            block_loras[name] = LoraEntry(a=a, b=b, scale=0.125)
            trainable.extend((a, b))
        loras[index] = block_loras
    return loras, trainable


def _train_plan():
    return ResidencyPlan.build(
        "train",
        (("blocks.0", "attn.wq"), ("blocks.0", "mlp.gate"), ("blocks.1", "attn.wk")),
    )


def _sample_plan():
    return ResidencyPlan.build(
        "sample",
        (("blocks.0", "attn.wq"), ("blocks.1", "mlp.up"), ("blocks.1", "mlp.down")),
    )


def _arena_signature(arena):
    return (
        tuple(
            (key, arena.block_record(key).host_flat.data_ptr())
            for key in arena.block_keys()
        ),
        pin_manager.total_pinned_bytes(),
        tuple(sorted(pin_manager.pinned_bytes_by_kind().items())),
    )


def _parameter_ids(model):
    return tuple(
        id(child.weight)
        for index in range(LAYERS)
        for _name, child in _ADAPTER.leaf_entries(model.blocks[index])
    )


def _dynamo_counts():
    counters = torch._dynamo.utils.counters
    return (
        counters["stats"].get("unique_graphs", 0),
        sum(counters["graph_break"].values()),
    )


def test_compiled_train_parity_lora_grads_and_zero_graph_breaks():
    torch._dynamo.reset()
    model, arena, state, reference_args = _fixture()
    loras, trainable = _loras(model)
    executor = _runtime(
        model,
        state,
        adapters_by_block=loras,
        adapter_multiplier=torch.tensor(0.75, device="cuda"),
    )
    try:
        executor.activate(executor.TRAIN, _train_plan())
        inputs = _inputs(model, requires_grad=True)
        out = _run(executor, *inputs)
        out.square().mean().backward()
        torch.cuda.synchronize()

        assert all(tensor.grad is not None for tensor in trainable)
        assert all(torch.isfinite(tensor.grad).all() for tensor in trainable)

        # Numerical parity of the forward against a pure eager reference over
        # the pre-canonicalization weights + the same folded LoRAs.
        with torch.no_grad():
            expected = inputs[0].detach()
            for index in range(LAYERS):
                lora_args = _ADAPTER.build_adapter_args(
                    index, loras, torch.tensor(0.75, device="cuda")
                )
                expected = model.blocks[index].forward_streamed(
                    expected,
                    *inputs[1:],
                    reference_args[index],
                    _ADAPTER.bind_block_operations(
                        model.blocks[index],
                        "cuda",
                    ),
                    loras=lora_args,
                )
        torch.testing.assert_close(
            out.detach(), expected, rtol=1e-4, atol=1e-5
        )

        graphs, breaks = _dynamo_counts()
        assert breaks == 0
        assert graphs >= 1
    finally:
        ingraph_stream.drain_fetch_runtime()
        arena.release()




def test_compiled_train_accepts_functional_lokr_owner():
    class Network:
        network_type = "lora"
        is_active = True
        is_merged_in = False
        is_lorm = False
        _multiplier = 1.0
        torch_multiplier = torch.tensor([0.75], device="cuda")
        vector_gates = None
        is_assistant_adapter = False
        base_model_ref = None

    torch._dynamo.reset()
    model, arena, state, _reference_args = _fixture()
    child = model.blocks[0].attn.wq
    network = Network()
    lokr = LokrModule(
        "test",
        child,
        network=network,
        lora_dim=4,
        alpha=4,
    ).cuda()
    lokr.apply_to()
    adapters = {0: {"attn.wq": lokr}}
    executor = _runtime(model, state, adapters_by_block=adapters)
    try:
        executor.activate(executor.TRAIN, _train_plan())
        inputs = _inputs(model, requires_grad=True)
        out = _run(executor, *inputs)
        out.square().mean().backward()
        torch.cuda.synchronize()

        trainable = tuple(parameter for parameter in lokr.parameters() if parameter.requires_grad)
        assert all(parameter.grad is not None for parameter in trainable)
        assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)
    finally:
        ingraph_stream.drain_fetch_runtime()
        arena.release()
def test_phase_cycles_reuse_graphs_and_arena_fixed_across_boundaries():
    torch._dynamo.reset()
    model, arena, state, _reference_args = _fixture()
    loras, _trainable = _loras(model)
    executor = _runtime(
        model,
        state,
        adapters_by_block=loras,
        adapter_multiplier=torch.tensor(0.5, device="cuda"),
    )
    train_plan, sample_plan = _train_plan(), _sample_plan()
    signature = _arena_signature(arena)
    parameter_ids = _parameter_ids(model)

    def cycle():
        executor.activate(executor.TRAIN, train_plan)
        inputs = _inputs(model, requires_grad=True)
        _run(executor, *inputs).square().mean().backward()
        executor.activate(executor.SAMPLE, sample_plan)
        with torch.no_grad():
            _run(executor, *_inputs(model))
        torch.cuda.synchronize()

    train_program = executor.program(executor.TRAIN)
    sample_program = executor.program(executor.SAMPLE)
    trunks = (train_program.trunk, sample_program.trunk)
    fingerprints = (train_program.fingerprint, sample_program.fingerprint)

    try:
        cycle()
        graphs_after_first, breaks = _dynamo_counts()
        assert breaks == 0
        generation_after_first = executor.source_generation

        cycle()
        graphs_after_second, breaks = _dynamo_counts()
        assert breaks == 0
        # Same-plan cycles add no unique graphs: residency publication swaps
        # source snapshots only.
        assert graphs_after_second == graphs_after_first

        # The programs are permanent: republishing residency never rebuilds a
        # trunk or changes a program fingerprint, only the source generation.
        assert executor.program(executor.TRAIN) is train_program
        assert executor.program(executor.SAMPLE) is sample_program
        assert (
            executor.program(executor.TRAIN).trunk,
            executor.program(executor.SAMPLE).trunk,
        ) == trunks
        assert (
            executor.program(executor.TRAIN).fingerprint,
            executor.program(executor.SAMPLE).fingerprint,
        ) == fingerprints
        assert executor.source_generation > generation_after_first

        # Arena pointers, registrations, pin ledger, and Parameter identity
        # are byte-for-byte fixed across every boundary.
        assert _arena_signature(arena) == signature
        assert _parameter_ids(model) == parameter_ids
    finally:
        ingraph_stream.drain_fetch_runtime()
        arena.release()


def test_bounce_pin_churn_does_not_invalidate_phase_transitions(monkeypatch):
    model, arena, state, _reference_args = _fixture()
    executor = _runtime(model, state, compile_blocks=False)
    during_handle = None
    original_build_sidecar = state._build_sidecar

    try:
        executor.activate(executor.TRAIN, _train_plan())

        # Unrelated bounce ownership may change between phase transitions.
        between_handle = pin_manager.pin_alloc(4096, "bounce", required=True)
        assert between_handle.pinned
        pin_manager.release(between_handle)

        # It may also change while reconcile is creating GPU sidecars. Keep
        # this allocation live through both residency and executor post-checks.
        def build_sidecar_with_bounce(key):
            nonlocal during_handle
            if during_handle is None:
                during_handle = pin_manager.pin_alloc(
                    4096, "bounce", required=True
                )
                assert during_handle.pinned
            return original_build_sidecar(key)

        monkeypatch.setattr(state, "_build_sidecar", build_sidecar_with_bounce)
        executor.activate(executor.SAMPLE, _sample_plan())
        with torch.no_grad():
            assert _run(executor,*_inputs(model)).shape == (1, 5, 32)
        pin_manager.release(during_handle)
        during_handle = None
        monkeypatch.setattr(state, "_build_sidecar", original_build_sidecar)

        executor.activate(executor.TRAIN, _train_plan())
        assert _run(executor,
            *_inputs(model, requires_grad=True)
        ).shape == (1, 5, 32)
    finally:
        if during_handle is not None:
            pin_manager.release(during_handle)
        ingraph_stream.drain_fetch_runtime()
        arena.release()


def test_actual_arena_flat_unregistration_still_fails_closed():
    model, arena, state, _reference_args = _fixture()
    executor = _runtime(model, state, compile_blocks=False)
    flat = arena.block_record("blocks.0").host_flat
    try:
        assert pin_manager.unpin_tensor_in_place(flat, "weights")
        with pytest.raises(
            ImmutableRuntimeError, match="arena_mutated_at_boundary"
        ):
            executor.activate(executor.SAMPLE, _sample_plan())
    finally:
        ingraph_stream.drain_fetch_runtime()
        arena.release()


def test_all_streamed_sampling_fallback_needs_no_host_rebuild():
    model, arena, state, _reference_args = _fixture()
    executor = _runtime(model, state, compile_blocks=False)
    signature = _arena_signature(arena)
    try:
        # Simulate mid-run state: a partially resident sampling plan.
        executor.activate(executor.SAMPLE, _sample_plan())
        with torch.no_grad():
            _run(executor, *_inputs(model))

        # Emergency demotion: prebuilt all-streamed plan, no host-side work.
        executor.activate_sampling_fallback()
        assert state.resident_bytes() == 0
        for index in range(LAYERS):
            snapshot = executor.source(index)
            assert snapshot.transfer is not None
            assert snapshot.transfer.fully_streamed
            assert snapshot.transfer.num_ranges == 1
        assert _arena_signature(arena) == signature

        ingraph_stream.fetch_stats(reset=True)
        with torch.no_grad():
            out = _run(executor, *_inputs(model))
        torch.cuda.synchronize()
        assert out.shape == (1, 5, 32)
        stats = ingraph_stream.fetch_stats()
        assert stats["fetches"] == LAYERS
        assert stats["copies"] == LAYERS  # single-copy fast path per block
        assert _arena_signature(arena) == signature
    finally:
        ingraph_stream.drain_fetch_runtime()
        arena.release()


def test_run_fails_closed_without_activation_and_on_stale_plan():
    model, arena, state, _reference_args = _fixture()
    executor = _runtime(model, state, compile_blocks=False)
    try:
        # No residency published yet: there is no source table to run against.
        with torch.no_grad(), pytest.raises(
            ImmutableRuntimeError, match="no_residency_source_table"
        ):
            _run(executor, *_inputs(model))

        executor.activate(executor.SAMPLE, _sample_plan())
        # Republishing residency under a live execution must fail closed: the
        # in-flight trunk is reading the current source snapshots.
        with torch.no_grad(), executor.execution(executor.SAMPLE):
            with pytest.raises(
                ImmutableRuntimeError, match="residency_transition_during_execution"
            ):
                executor.set_residency_plan(_train_plan())
            inputs = _inputs(model)
            assert executor.run(inputs[0], inputs[1:]).shape == (1, 5, 32)
    finally:
        ingraph_stream.drain_fetch_runtime()
        arena.release()
