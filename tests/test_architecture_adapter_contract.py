"""Behavioral proof that the arena block ABI is architecture-neutral."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from toolkit.memory_management.adapters import validate_architecture_adapter
from toolkit.memory_management.canonical_arena import CanonicalArena
from toolkit.memory_management.immutable_runtime import ImmutableTransformerRuntime
from toolkit.memory_management.residency import ResidencyPlan, ResidencyState


class _Stage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 4, bias=False)
        self.requires_grad_(False)


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

    def bind_block_operations(self, _storage_views, _device):
        return (None,)

    def forward_block(
        self,
        _block,
        hidden,
        block_args,
        leaf_args,
        _linear_operations,
        _adapter_args,
        *,
        training,
    ):
        del training
        weight = leaf_args[0][0]
        return (
            F.linear(hidden, weight) * block_args["gain"]
            + block_args["offset"]
        )


def test_second_adapter_runs_non_krea_structure_through_generic_runtime():
    torch.manual_seed(7)
    model = SimpleNamespace(stages=(_Stage(), _Stage()))
    adapter = _StageAdapter()
    validate_architecture_adapter(adapter)
    adapter.validate_transformer(model)
    reference_weights = tuple(
        stage.projection.weight.detach().clone() for stage in model.stages
    )

    arena = CanonicalArena()
    arena.canonicalize(
        {
            adapter.block_key(model, index): list(adapter.leaf_entries(stage))
            for index, stage in enumerate(model.stages)
        }
    )
    residency = ResidencyState(arena, "cpu")
    resident_keys = tuple(
        (adapter.block_key(model, index), "projection")
        for index in range(len(model.stages))
    )
    plan = ResidencyPlan.build("train", resident_keys)
    runtime = ImmutableTransformerRuntime(
        model,
        residency,
        architecture_adapter=adapter,
        block_operations=((None,), (None,)),
        compile_blocks=False,
    )

    try:
        runtime.finalize_execution()
        runtime.activate(runtime.TRAIN, plan)
        hidden = torch.randn(2, 4, requires_grad=True)
        block_args = {
            "offset": torch.randn(2, 4),
            "gain": torch.tensor(0.5),
        }
        expected = hidden
        for weight in reference_weights:
            expected = F.linear(expected, weight) * block_args["gain"] + block_args[
                "offset"
            ]

        assert runtime.can_run_current_call(block_args)
        with runtime.execution(runtime.TRAIN):
            actual = runtime.run(hidden, block_args)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        assert hidden.grad is not None
    finally:
        runtime.close()
        residency.clear()
        arena.release()
