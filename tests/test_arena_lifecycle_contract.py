import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from toolkit.memory_management.arena_offload import (
    ArenaOffloadConfig,
    ArenaSetupFatalError,
    prepare_canonical_storage,
    prepare_arena_offload,
)
from toolkit.memory_management.adapters import SingleStreamMMDiTAdapter
from toolkit.memory_management.arena_offload.errors import (
    ArenaCleanupError,
    is_fatal_arena_setup,
    recover_allows_next_job,
)
from toolkit.memory_management.arena_offload.ownership import (
    acquire_process_owner,
    active_process_owner,
    release_process_owner,
)
from toolkit.memory_management.arena_offload.resources import ArenaRuntimeResources
from toolkit.memory_management.arena_offload.runtime import ArenaOffloadRuntime
from toolkit.quantization.fp8_linear import bind_storage_operation


class _Adapter:
    architecture_key = "test_linear"

    def validate_transformer(self, model):
        if not isinstance(model, torch.nn.Linear):
            raise TypeError("expected linear")

    def execution_blocks(self, model):
        return (model,)

    def block_key(self, _model, index):
        return f"blocks.{index}"

    def leaf_entries(self, block):
        return (("linear", block),)

    def collect_execution_adapters(self, _model, _network):
        return {}

    def build_adapter_args(self, _index, _adapters, multiplier=None):
        del multiplier
        return None

    def can_run_current_call(self, _block_args, **_kwargs):
        return True

    def bind_block_operations(self, storage_views, device):
        del storage_views, device
        return (None,)

    def forward_block(
        self,
        _block,
        hidden,
        _block_args,
        _leaf_args,
        _linear_operations,
        _adapter_args,
        *,
        training,
    ):
        del training
        return hidden


class _UnsupportedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, payload):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            payload.shape,
            strides=payload.stride(),
            storage_offset=payload.storage_offset(),
            dtype=payload.dtype,
            layout=payload.layout,
            device=payload.device,
            requires_grad=False,
        )

    def __init__(self, payload):
        self.payload = payload

    def __tensor_flatten__(self):
        return ["payload"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, _context, _size, _stride):
        return _UnsupportedTensor(inner_tensors["payload"])

    @classmethod
    def __torch_dispatch__(cls, func, _types, args=(), kwargs=None):
        if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
            return cls(args[0].payload)
        raise NotImplementedError(func)


class _UnsupportedLeaf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._parameters["weight"] = _UnsupportedTensor(torch.randn(4, 4))
        self._parameters["bias"] = None


class _UnsupportedStorageAdapter(_Adapter):
    def validate_transformer(self, model):
        if not hasattr(model, "leaf"):
            raise TypeError("expected leaf")

    def execution_blocks(self, model):
        return (model.leaf,)

    def leaf_entries(self, block):
        return (("linear", block),)

    def bind_block_operations(self, storage_views, device):
        return tuple(
            bind_storage_operation(
                view.tensors,
                execution_key=view.spec.execution_key,
                weight_leaf_count=view.spec.weight_leaf_count,
                device=device,
            )
            for view in storage_views
        )


def _frozen_linear():
    model = torch.nn.Linear(4, 4)
    model.requires_grad_(False)
    return model


def test_process_owner_is_exclusive_sequential_and_stale_safe():
    first = acquire_process_owner("cpu")
    try:
        with pytest.raises(RuntimeError, match="arena_runtime_already_active"):
            acquire_process_owner("cpu")
    finally:
        release_process_owner(first)

    second = acquire_process_owner("cpu")
    try:
        with pytest.raises(RuntimeError, match="owner_mismatch"):
            release_process_owner(first)
        assert active_process_owner() is second
    finally:
        release_process_owner(second)


def test_precommit_failure_preserves_original_classification_and_model():
    model = _frozen_linear()
    original = model.weight
    adapter = _Adapter()
    adapter.execution_blocks = mock.Mock(side_effect=ValueError("bad architecture"))

    with pytest.raises(ValueError, match="bad architecture"):
        prepare_arena_offload(
            model,
            device="cpu",
            adapter=adapter,
            config=ArenaOffloadConfig(enabled=True),
        )

    assert model.weight is original
    assert active_process_owner() is None
    assert not hasattr(model, "_arena_offload_runtime")
    assert not hasattr(model, "_arena_offload_disposed")


def test_unsupported_storage_operation_fails_before_canonical_commit():
    model = torch.nn.Module()
    model.leaf = _UnsupportedLeaf()
    original = model.leaf.weight

    with pytest.raises(ValueError, match="unsupported_linear_storage_operation"):
        prepare_arena_offload(
            model,
            device="cpu",
            adapter=_UnsupportedStorageAdapter(),
            config=ArenaOffloadConfig(enabled=True),
        )

    assert model.leaf.weight is original
    assert active_process_owner() is None
    assert not hasattr(model, "_arena_offload_runtime")
    assert not hasattr(model, "_arena_offload_disposed")


def test_disabled_config_and_unsupported_architecture_fail_before_mutation():
    model = _frozen_linear()
    original = model.weight
    with pytest.raises(ValueError, match="arena_offload_not_enabled"):
        prepare_arena_offload(
            model,
            device="cpu",
            adapter=_Adapter(),
            config=ArenaOffloadConfig(enabled=False),
        )
    with pytest.raises(TypeError, match="transformer.blocks"):
        prepare_arena_offload(
            model,
            device="cpu",
            adapter=SingleStreamMMDiTAdapter(),
            config=ArenaOffloadConfig(enabled=True),
        )
    assert model.weight is original
    assert active_process_owner() is None
    assert not hasattr(model, "_arena_offload_runtime")
    assert not hasattr(model, "_arena_offload_disposed")


def test_direct_loader_rollback_releases_preparation_owner():
    model = _frozen_linear()
    build = prepare_canonical_storage(model, _Adapter(), device="cpu")
    assert active_process_owner() is not None
    build.rollback()
    assert active_process_owner() is None
    assert not hasattr(model, "_arena_offload_disposed")


def test_postcommit_failure_is_fatal_disposes_and_releases_owner():
    model = _frozen_linear()
    original_error = RuntimeError("residency construction failed")
    with mock.patch(
        "toolkit.memory_management.arena_offload.runtime.build_training_plan",
        side_effect=original_error,
    ):
        with pytest.raises(ArenaSetupFatalError) as caught:
            prepare_arena_offload(
                model,
                device="cpu",
                adapter=_Adapter(),
                config=ArenaOffloadConfig(enabled=True),
            )

    wrapper = RuntimeError("wrapper")
    wrapper.__cause__ = caught.value
    assert caught.value.__cause__ is original_error
    assert is_fatal_arena_setup(wrapper)
    assert not recover_allows_next_job(wrapper, True)
    assert recover_allows_next_job(RuntimeError("ordinary"), True)
    assert active_process_owner() is None
    assert model._arena_offload_disposed
    with pytest.raises(RuntimeError, match="transformer_disposed"):
        model(torch.randn(1, 4))
    with pytest.raises(RuntimeError, match="transformer_disposed"):
        model.to("cpu")


@pytest.mark.parametrize(
    "target",
    (
        "toolkit.memory_management.arena_offload.runtime.ResidencyState",
        "toolkit.memory_management.arena_offload.runtime.prepare_immutable_runtime",
    ),
)
def test_postcommit_fault_boundaries_are_fatal_and_never_fall_back(target):
    model = _frozen_linear()
    failure = RuntimeError(f"injected:{target.rsplit('.', 1)[-1]}")
    plan = {
        "offload_ids": set(),
        "protected_training_leaf_keys": frozenset(),
    }
    patches = [
        mock.patch(
            "toolkit.memory_management.arena_offload.runtime.build_training_plan",
            return_value=plan,
        ),
        mock.patch(target, side_effect=failure),
    ]
    with patches[0], patches[1]:
        with pytest.raises(ArenaSetupFatalError) as caught:
            prepare_arena_offload(
                model,
                device="cpu",
                adapter=_Adapter(),
                config=ArenaOffloadConfig(enabled=True),
            )
    assert caught.value.__cause__ is failure
    assert active_process_owner() is None
    assert model._arena_offload_disposed
    assert not hasattr(model, "_memory_manager")


def test_resource_release_continues_after_cleanup_error_and_is_idempotent():
    calls = []
    model = _frozen_linear()
    resources = ArenaRuntimeResources(model, "cpu")
    resources.acquire_process_owner()
    resources.canonical_committed = True
    resources.arena = SimpleNamespace(
        unguard_whole_model_to=lambda _model: calls.append("unguard"),
        release=lambda: calls.append("arena"),
    )
    resources.residency = SimpleNamespace(clear=lambda: calls.append("residency"))
    resources.executor = SimpleNamespace(
        active_executions=0,
        close=lambda: (_ for _ in ()).throw(RuntimeError("executor boom")),
    )

    with pytest.raises(ArenaCleanupError, match="executor boom"):
        resources.release()
    resources.release()

    assert calls == ["residency", "unguard", "arena"]
    assert resources.released
    assert resources.disposed
    assert active_process_owner() is None


def test_transfer_cleanup_failure_retains_process_owner_until_retry():
    model = _frozen_linear()
    resources = ArenaRuntimeResources(model, "cpu")
    resources.acquire_process_owner()
    token = resources.owner_token

    with mock.patch(
        "toolkit.memory_management.arena_offload.transfer.release_fetch_runtime",
        side_effect=RuntimeError("transfer cleanup failed"),
    ):
        with pytest.raises(ArenaCleanupError, match="transfer cleanup failed"):
            resources.release()

    assert active_process_owner() is token
    assert not resources.released
    resources.release()
    assert active_process_owner() is None
    assert resources.released


def test_finalize_cap_failure_is_fatal_after_runtime_publication():
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._closed = False
    runtime._disposed = False
    runtime._resources = SimpleNamespace(release=mock.Mock())
    failure = RuntimeError("allocator cap failed")
    runtime._bind_training_cap = mock.Mock(side_effect=failure)

    with pytest.raises(ArenaSetupFatalError) as caught:
        runtime.finalize()

    assert caught.value.__cause__ is failure
    runtime._resources.release.assert_called_once_with()


def test_phase7_import_and_private_state_boundaries():
    root = Path(__file__).parents[1]
    arena_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (root / "toolkit" / "memory_management" / "arena_offload").glob("*.py")
    )
    assert "from ..manager import" not in arena_sources
    assert "manager_modules" not in arena_sources

    for relative in (
        "jobs/process/BaseSDTrainProcess.py",
        "extensions_built_in/sd_trainer/SDTrainer.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert "memory_management.arena_offload" not in source
        assert not any(
            isinstance(node, ast.Attribute) and node.attr.startswith("_mm_")
            for node in ast.walk(tree)
        )

    krea_root = root / "extensions_built_in" / "diffusion_models" / "krea2"
    for path in krea_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "_mm_" not in source, path
        assert "_immutable_runtime" not in source, path


def test_phase8_legacy_manager_has_no_arena_execution_bridge():
    root = Path(__file__).parents[1]
    manager = (
        root / "toolkit" / "memory_management" / "manager.py"
    ).read_text(encoding="utf-8")
    for obsolete in (
        "attach_smart_training_immutable",
        "smart_immutable",
        "_mm_immutable_",
        "_immutable_runtime",
        "_mm_canonical_leaf",
        "canonical_relief",
    ):
        assert obsolete not in manager
