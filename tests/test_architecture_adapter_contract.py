"""Maintainer-runnable proof that the arena block ABI is architecture-neutral."""

from dataclasses import fields
import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from toolkit.quantization.fp8_linear import bind_linear_operation

from toolkit.memory_management.adapters import validate_architecture_adapter
from toolkit.memory_management.immutable_runtime import (
    ImmutableBlockABI,
    build_sample_trunk,
    build_train_trunk,
)
from toolkit.memory_management.arena_offload import layout
from toolkit.memory_management import immutable_runtime, pinned_arena, residency
from toolkit.memory_management import transfer_plan


class _Stage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4, bias=False)


class _StageAdapter:
    """Deliberately differs from Krea in container, leaves, and block args."""

    architecture_key = "synthetic_stages"

    def validate_transformer(self, transformer):
        if not getattr(transformer, "stages", None):
            raise TypeError("synthetic adapter requires stages")

    def execution_blocks(self, transformer):
        return tuple(transformer.stages)

    def block_key(self, _transformer, index):
        return f"stages.{index}"

    def leaf_entries(self, block):
        return (("projection", block.projection),)

    def collect_execution_adapters(self, _transformer, network):
        return {} if network is None else network.entries

    def build_adapter_args(self, index, adapters_by_block, multiplier=None):
        del multiplier
        return (adapters_by_block or {}).get(index)

    def can_run_current_call(self, block_args, **_kwargs):
        return set(block_args) == {"offset", "gain"}

    def bind_block_operations(self, block, device):
        return (
            bind_linear_operation(block.projection.weight, device=device),
        )

    def forward_block(
        self,
        block,
        hidden,
        block_args,
        leaf_args,
        linear_operations,
        adapter_args,
        *,
        training,
    ):
        del block, linear_operations, adapter_args, training
        weight = leaf_args[0][0]
        return F.linear(hidden, weight) * block_args["gain"] + block_args["offset"]


def _block_functions(model, adapter):
    functions = []
    for block in adapter.execution_blocks(model):
        leaf_args = tuple(
            ((module.weight,))
            for _name, module in adapter.leaf_entries(block)
        )
        operations = adapter.bind_block_operations(block, "cpu")

        def block_fn(hidden, block_args, block=block, leaf_args=leaf_args):
            return adapter.forward_block(
                block,
                hidden,
                block_args,
                leaf_args,
                operations,
                None,
                training=torch.is_grad_enabled(),
            )

        functions.append(block_fn)
    return tuple(functions)


def test_second_adapter_uses_non_krea_structure_and_opaque_pytree_args():
    model = SimpleNamespace(stages=(_Stage(), _Stage()))
    adapter = _StageAdapter()
    validate_architecture_adapter(adapter)
    adapter.validate_transformer(model)

    assert adapter.block_key(model, 1) == "stages.1"
    assert tuple(name for name, _ in adapter.leaf_entries(model.stages[0])) == (
        "projection",
    )

    hidden = torch.randn(2, 4, requires_grad=True)
    block_args = {
        "offset": torch.randn(2, 4),
        "gain": torch.tensor(0.5),
    }
    block_fns = _block_functions(model, adapter)
    sample = build_sample_trunk(block_fns)(hidden, block_args)
    train = build_train_trunk(block_fns)(hidden, block_args)

    torch.testing.assert_close(train, sample)
    train.sum().backward()
    assert hidden.grad is not None


def test_incomplete_adapter_is_rejected_before_arena_preparation():
    with pytest.raises(TypeError, match="missing validate_transformer"):
        validate_architecture_adapter(SimpleNamespace(architecture_key="incomplete"))


def test_arena_movement_contains_no_fp8_execution_contract():
    stale_fields = {
        "kind",
        "weight_scale",
        "fp8_qualifies",
        "fp8_flags",
        "native_fp8_eligible",
    }
    for record_type in (
        layout.LinearSpec,
        layout.BlockPack,
        layout.LayerStorageView,
        ImmutableBlockABI,
    ):
        assert stale_fields.isdisjoint(field.name for field in fields(record_type))

    assert "operation" not in {
        field.name for field in fields(layout.LayerStorageView)
    }

    movement_source = "\n".join(
        inspect.getsource(module)
        for module in (
            layout,
            immutable_runtime,
            pinned_arena,
            residency,
            transfer_plan,
        )
    )
    for execution_detail in (
        "torch._scaled_mm",
        "qdata",
        "fp8_qualifies",
        "fp8_flags",
        "weight_scale",
        "bind_storage_operation",
        "bind_parameter_operation",
    ):
        assert execution_detail not in movement_source
