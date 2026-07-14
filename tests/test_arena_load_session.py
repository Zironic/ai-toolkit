from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from toolkit.memory_management import pin_manager
from toolkit.memory_management.arena_offload import (
    model_load_arena_session,
    prepare_arena_offload,
    prepare_canonical_storage_from_state_dict,
)
from toolkit.memory_management.arena_offload.load_session import (
    PENDING_CANONICAL_BUILD_ATTR,
    claim_pending_canonical_build,
    try_prepare_canonical_from_state_dict,
)
from toolkit.memory_management.arena_offload.construction import (
    CanonicalStateInferenceError,
)
from toolkit.memory_management.canonical_arena import CanonicalArena
from toolkit.util.quantize import assign_quantized_state_dict, get_qtype, quantize


pytestmark = pytest.mark.process_isolated


class ExtraStateLinear(torch.nn.Linear):
    def __init__(self):
        super().__init__(4, 4)
        self.register_buffer("input_scale", torch.tensor(1.0))
        self.register_buffer("output_scale", torch.tensor(1.0))


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = ExtraStateLinear()


class ClonedStateLinear(ExtraStateLinear):
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        key = f'{kwargs.get("prefix", "")}weight'
        state[key] = state[key].clone()
        return state


class ClonedStateBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = ClonedStateLinear()


class Transformer(torch.nn.Module):
    def __init__(self, count=2):
        super().__init__()
        self.blocks = torch.nn.ModuleList(Block() for _ in range(count))
        self.head = torch.nn.Linear(4, 4)


class QuantBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)


class QuantTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([QuantBlock(), QuantBlock()])


def base_model():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            layer_offloading=True,
            layer_offloading_smart=True,
        ),
        device_torch=torch.device("cpu"),
        te_only=False,
        get_transformer_block_names=lambda: ["blocks"],
    )


def frozen_transformer(count=2):
    model = Transformer(count=count)
    model.requires_grad_(False)
    return model


def test_generic_schema_consumes_only_declared_execution_storage():
    source = frozen_transformer()
    state = {key: value.clone() for key, value in source.state_dict().items()}
    expected = source.blocks[0].linear.weight.clone()
    target = frozen_transformer()

    build = prepare_canonical_storage_from_state_dict(
        target, state, block_names=("blocks",)
    )

    assert "blocks.0.linear.weight" not in state
    assert "blocks.0.linear.bias" not in state
    assert "blocks.0.linear.input_scale" in state
    assert "blocks.0.linear.output_scale" in state
    assert "head.weight" in state
    build.commit()
    try:
        torch.testing.assert_close(target.blocks[0].linear.weight, expected)
    finally:
        CanonicalArena.unguard_whole_model_to(target)
        build.arena.release()


def test_load_session_publishes_and_claims_inferred_build():
    source = frozen_transformer()
    state = {key: value.clone() for key, value in source.state_dict().items()}
    target = frozen_transformer()

    with model_load_arena_session(base_model()):
        build = try_prepare_canonical_from_state_dict(target, state)
        assert build is not None
        assert getattr(target, PENDING_CANONICAL_BUILD_ATTR) is build
        assert claim_pending_canonical_build(target) is build
        assert not hasattr(target, PENDING_CANONICAL_BUILD_ATTR)

    build.rollback()


def test_normal_arena_attach_claims_generic_loader_build():
    source = frozen_transformer()
    state = {key: value.clone() for key, value in source.state_dict().items()}
    target = frozen_transformer()
    target.gradient_checkpointing = True
    target._checkpoint_keep_last = 0
    config = SimpleNamespace(
        enabled=True,
        _policy=SimpleNamespace(checkpoint_keep_last=0),
    )
    runtime = object()

    with model_load_arena_session(base_model()):
        build = try_prepare_canonical_from_state_dict(target, state)
        with mock.patch(
            "toolkit.memory_management.arena_offload.runtime."
            "ArenaOffloadRuntime._prepare",
            return_value=runtime,
        ) as prepare:
            assert prepare_arena_offload(
                target,
                device="cpu",
                config=config,
                block_names=("blocks",),
            ) is runtime

    assert prepare.call_args.kwargs["canonical_build"] is build
    assert not hasattr(target, PENDING_CANONICAL_BUILD_ATTR)
    build.rollback()


def test_common_quantized_assignment_uses_active_generic_session():
    from optimum.quanto import freeze

    source = QuantTransformer()
    quantize(source, weights=get_qtype("qfloat8"))
    freeze(source)
    source.requires_grad_(False)
    state = source.state_dict()
    expected = source.blocks[0].linear.weight.dequantize().clone()
    with torch.device("meta"):
        target = QuantTransformer()

    with model_load_arena_session(base_model()):
        assign_quantized_state_dict(target, state, "qfloat8")
        build = claim_pending_canonical_build(target)
        assert build is not None

    assert "blocks.0.linear.weight._data" not in state
    assert "blocks.0.linear.weight._scale" not in state
    resources = build._arena_resources
    build.commit()
    resources.mark_canonical_committed()
    try:
        torch.testing.assert_close(
            target.blocks[0].linear.weight.dequantize(), expected
        )
    finally:
        resources.release()


def test_noninferable_model_falls_back_without_consuming_state():
    source = frozen_transformer(count=1)
    state = {key: value.clone() for key, value in source.state_dict().items()}
    before = tuple(state)
    target = frozen_transformer(count=1)

    with model_load_arena_session(base_model()) as session:
        assert try_prepare_canonical_from_state_dict(target, state) is None
        assert session.unsupported_reason is not None

    assert tuple(state) == before
    assert not hasattr(target, PENDING_CANONICAL_BUILD_ATTR)


def test_custom_cloned_serialization_is_rejected_instead_of_guessed():
    target = torch.nn.Module()
    target.blocks = torch.nn.ModuleList([ClonedStateBlock(), ClonedStateBlock()])
    target.requires_grad_(False)
    state = {key: value.clone() for key, value in target.state_dict().items()}
    before = tuple(state)

    with pytest.raises(
        CanonicalStateInferenceError,
        match="storage_not_serialized",
    ):
        prepare_canonical_storage_from_state_dict(
            target, state, block_names=("blocks",)
        )

    assert tuple(state) == before


def test_unclaimed_destructive_build_is_rolled_back_and_rejected():
    source = frozen_transformer()
    state = {key: value.clone() for key, value in source.state_dict().items()}
    target = frozen_transformer()
    before = pin_manager.pinned_bytes_by_kind().get("weights", 0)

    with pytest.raises(RuntimeError, match="build_not_claimed"):
        with model_load_arena_session(base_model()):
            assert try_prepare_canonical_from_state_dict(target, state) is not None

    assert not hasattr(target, PENDING_CANONICAL_BUILD_ATTR)
    assert pin_manager.pinned_bytes_by_kind().get("weights", 0) == before
