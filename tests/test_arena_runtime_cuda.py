import pytest
import torch

from toolkit.memory_management.arena_offload import (
    ArenaOffloadConfig,
    prepare_arena_offload,
)
from toolkit.memory_management.arena_offload.ownership import active_process_owner


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.process_isolated,
]


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Linear(16, 16)
        self.requires_grad_(False)

    def forward(self, value):
        return self.block(value)


class _Adapter:
    architecture_key = "test_linear"

    def validate_transformer(self, model):
        assert hasattr(model, "block")

    def execution_blocks(self, model):
        return (model.block,)

    def block_key(self, _model, index):
        return f"blocks.{index}"

    def leaf_entries(self, block):
        return (("linear", block),)

    def collect_execution_adapters(self, _model, _network):
        return {}

    def can_run_current_call(self, _block_args, **_kwargs):
        return True

    def build_adapter_args(self, _index, _adapters, _multiplier=None):
        return None

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
        _lora_args,
        *,
        training,
    ):
        del training
        return hidden


def test_outer_runtime_finalize_close_and_sequential_reacquire():
    config = ArenaOffloadConfig(enabled=True, compile_blocks=False)
    first_model = _Model()
    first = prepare_arena_offload(
        first_model,
        device="cuda:0",
        adapter=_Adapter(),
        config=config,
    )
    first.finalize()
    assert active_process_owner() is not None
    assert first.finalized
    with first.training_step(shape_key=(16,), step_num=1):
        pass
    with first.sampling_session():
        with first.sampling_image(shape_key=(16,), cold_working_bytes=0):
            pass
    with first.training_step(shape_key=(16,), step_num=2):
        pass
    first.close()
    first.close()
    assert active_process_owner() is None
    assert first.disposed
    with pytest.raises(RuntimeError, match="transformer_disposed"):
        first_model(torch.randn(1, 16))

    second_model = _Model()
    second = prepare_arena_offload(
        second_model,
        device="cuda:0",
        adapter=_Adapter(),
        config=config,
    )
    second.close()
    assert active_process_owner() is None


def test_compile_enabled_runtime_reuses_same_cuda_specialization():
    config = ArenaOffloadConfig(
        enabled=True,
        compile_blocks=True,
        _compile_dynamic=False,
    )
    model = _Model()
    runtime = prepare_arena_offload(
        model,
        device="cuda:0",
        adapter=_Adapter(),
        config=config,
    )
    runtime.finalize()
    value = torch.randn(2, 16, device="cuda:0")
    tvec = torch.randn(2, 16, device="cuda:0")
    try:
        with runtime.training_step(shape_key=(2, 16), step_num=1):
            first = runtime.run_blocks(value, {"conditioning": tvec})
        torch.cuda.synchronize()
        frames_after_first = int(
            torch._dynamo.utils.counters["frames"].get("total", 0)
        )
        with runtime.training_step(shape_key=(2, 16), step_num=2):
            second = runtime.run_blocks(value, {"conditioning": tvec})
        torch.cuda.synchronize()
        frames_after_second = int(
            torch._dynamo.utils.counters["frames"].get("total", 0)
        )
        torch.testing.assert_close(first, value)
        torch.testing.assert_close(second, value)
        assert frames_after_second == frames_after_first
    finally:
        runtime.close()
