from types import MethodType
from unittest import mock
from dataclasses import replace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from toolkit.memory_management.arena_offload import (
    ArenaOffloadConfig,
    close_arena_offload,
    discover_blocks,
    prepare_arena_offload,
)
from toolkit.memory_management.arena_offload.discovery import BlockDiscoveryError
from toolkit.memory_management.arena_offload.ownership import active_process_owner
from toolkit.memory_management.residency import ResidencyPlan
from toolkit.memory_management.runtime import get_memory_runtime


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, value):
        return torch.nn.functional.silu(self.proj(value))


class _Transformer(torch.nn.Module):
    def __init__(self, count=3):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block() for _ in range(count)])
        self.gradient_checkpointing = False
        self._checkpoint_keep_last = 0

    def enable_gradient_checkpointing(self, keep_last=0):
        self.gradient_checkpointing = True
        self._checkpoint_keep_last = int(keep_last)

    def forward(self, value):
        cutoff = len(self.blocks) - self._checkpoint_keep_last
        for index, block in enumerate(self.blocks):
            if self.gradient_checkpointing and torch.is_grad_enabled() and index < cutoff:
                value = checkpoint(block, value, use_reentrant=False)
            else:
                value = block(value)
        return value


def _frozen_transformer():
    model = _Transformer()
    model.requires_grad_(False)
    return model


def test_declared_container_discovery_accounts_all_block_state():
    model = _frozen_transformer()
    selection = discover_blocks(model, container_paths=("blocks",))
    assert selection.container_paths == ("blocks",)
    assert selection.block_keys == ("blocks.0", "blocks.1", "blocks.2")
    assert all(len(entries) == 1 for entries in selection.entries_by_block.values())
    assert selection.accounting.managed_entries == 6
    assert selection.accounting.managed_bytes > 0


def test_shared_managed_state_is_rejected_before_construction():
    model = _frozen_transformer()
    shared = model.blocks[0].proj.weight
    model.blocks[1].proj.weight = shared
    with pytest.raises(BlockDiscoveryError, match="shared_managed"):
        discover_blocks(model, container_paths=("blocks",))


def test_checkpointing_rejection_precedes_canonical_commit():
    model = _frozen_transformer()
    original = model.blocks[0].proj.weight
    with pytest.raises(ValueError, match="requires model gradient checkpointing"):
        prepare_arena_offload(
            model,
            device="cpu",
            block_names=("blocks",),
            config=ArenaOffloadConfig(enabled=True),
        )
    assert model.blocks[0].proj.weight is original
    assert active_process_owner() is None
    assert not hasattr(model, "_arena_offload_runtime")


def test_saved_installed_forward_checkpoint_backward_and_teardown():
    torch.manual_seed(17)
    model = _frozen_transformer()
    model.enable_gradient_checkpointing(keep_last=1)
    reference_input = torch.randn(2, 4)
    reference = model(reference_input).detach()
    with mock.patch(
        "toolkit.memory_management.arena_offload.planner.vram_budget.device_mem_info",
        return_value=(8 * 1024**3, 12 * 1024**3),
    ), mock.patch(
        "toolkit.memory_management.arena_offload.planner.vram_budget.auto_margin_gib",
        return_value=1.0,
    ):
        config = ArenaOffloadConfig(enabled=True, compile_blocks=False)
        config = replace(
            config,
            _policy=replace(config._policy, checkpoint_keep_last=1),
        )
        runtime = prepare_arena_offload(
            model,
            device="cpu",
            block_names=("blocks",),
            config=config,
        )
    installed = []
    adapters = []
    for block in model.blocks:
        saved = block.forward
        block.adapter_gain = torch.nn.Parameter(torch.zeros(()))

        def installed_forward(self, value, _saved=saved):
            return _saved(value) + self.adapter_gain * value

        bound = MethodType(installed_forward, block)
        block.forward = bound
        installed.append(bound)
        adapters.append(block.adapter_gain)

    runtime.finalize()
    diagnostics = runtime.diagnostics()
    accounting = diagnostics["accounting"]
    assert diagnostics["checkpoint_owner"] == "model"
    assert diagnostics["state_audit"]["managed_entries"] == 6
    assert accounting["payload_reconciled"]
    assert accounting["canonical_payload_bytes"] == (
        accounting["canonical_resident_payload_bytes"]
        + accounting["streamed_payload_bytes"]
    )
    assert accounting["protected_training_blocks"] == ("blocks.2",)
    assert accounting["protected_training_blocks_resident"]
    with pytest.raises(RuntimeError, match="outside_transformer_execution"):
        model.blocks[0](torch.randn(2, 4))

    value = reference_input.detach().clone().requires_grad_(True)
    with runtime.training_step(shape_key=(2, 4), step_num=1):
        output = model(value)
        output.sum().backward()
    torch.testing.assert_close(output.detach(), reference)
    assert value.grad is not None
    assert all(parameter.grad is not None for parameter in adapters)

    protected = runtime._executor.protected_training_leaf_keys
    assert any(block == "blocks.2" for block, _leaf in protected)
    with pytest.raises(RuntimeError, match="protected_training_block"):
        runtime.transition_training_block("blocks.2", resident=False)

    close_arena_offload(model)
    assert all(
        block.forward is saved
        for block, saved in zip(model.blocks, installed, strict=True)
    )
    assert get_memory_runtime(model) is None
    assert model._arena_offload_disposed
    assert active_process_owner() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_streamed_compiled_train_sample_train():
    from toolkit.memory_management.arena_offload import transfer

    torch.manual_seed(23)
    device = torch.device("cuda")
    model = _frozen_transformer().to(device)
    model.enable_gradient_checkpointing(keep_last=1)
    config = ArenaOffloadConfig(
        enabled=True,
        compile_blocks=True,
        _compile_dynamic=False,
    )
    config = replace(
        config,
        _policy=replace(
            config._policy,
            working_reserve_gib=0.0,
            wddm_margin_gib=0.0,
            wddm_hard_gib=1.0,
            checkpoint_keep_last=1,
        ),
    )
    with mock.patch(
        "toolkit.memory_management.arena_offload.planner.vram_budget.device_mem_info",
        return_value=(1 * 1024**3, 12 * 1024**3),
    ):
        runtime = prepare_arena_offload(
            model,
            device=device,
            block_names=("blocks",),
            config=config,
        )
    adapters = []
    for block in model.blocks:
        saved = block.forward
        block.adapter_gain = torch.nn.Parameter(torch.zeros((), device=device))

        def installed_forward(self, value, _saved=saved):
            return _saved(value) + self.adapter_gain * value

        block.forward = MethodType(installed_forward, block)
        adapters.append(block.adapter_gain)
    runtime.finalize()
    accounting = runtime.diagnostics()["accounting"]
    assert accounting["payload_reconciled"]
    assert accounting["mixed_residency"]
    assert accounting["resident_blocks"] >= 1
    assert accounting["streamed_blocks"] >= 1
    assert accounting["planned_training_h2d_bytes"] == (
        2 * accounting["planned_forward_h2d_bytes"]
    )
    assert accounting["protected_training_blocks_resident"]
    streamed = [
        runtime._executor.source(index).transfer is not None
        for index in range(runtime.block_count)
    ]
    assert any(streamed)
    assert not streamed[-1]

    def train_once(step):
        transfer_before = transfer.lifetime_fetch_stats()["bytes"]
        planned = runtime.diagnostics()["accounting"][
            "planned_training_h2d_bytes"
        ]
        value = torch.randn(2, 4, device=device, requires_grad=True)
        with runtime.training_step(shape_key=(2, 4), step_num=step):
            output = model(value)
            output.square().mean().backward()
        assert value.grad is not None
        assert all(parameter.grad is not None for parameter in adapters)
        for parameter in adapters:
            parameter.grad = None
        assert transfer.lifetime_fetch_stats()["bytes"] - transfer_before == planned
        return output.detach()

    first = train_once(1)
    sample_plan = ResidencyPlan.build("sample", ())
    runtime._executor.activate(runtime._executor.SAMPLE, sample_plan)
    sample_accounting = runtime.diagnostics()["accounting"]
    sample_transfer_before = transfer.lifetime_fetch_stats()["bytes"]
    with torch.no_grad(), runtime._executor.execution(runtime._executor.SAMPLE):
        sampled = model(torch.randn(2, 4, device=device))
    assert transfer.lifetime_fetch_stats()["bytes"] - sample_transfer_before == (
        sample_accounting["planned_forward_h2d_bytes"]
    )
    assert torch.isfinite(sampled).all()
    runtime._executor.activate(runtime._executor.TRAIN, runtime._training_plan)
    second = train_once(2)
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    close_arena_offload(model)
    assert active_process_owner() is None
