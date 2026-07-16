from types import SimpleNamespace

import pytest
import torch

from toolkit.memory_management.arena_offload import prepare_canonical_storage
from toolkit.util.quantize import quantize_model


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16, bias=True)


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(), _Block()])
        self.head = torch.nn.Linear(16, 16, bias=False)


class _BaseModel:
    def __init__(self, device="cpu", dtype=torch.float32):
        self.device_torch = torch.device(device)
        self.torch_dtype = dtype
        self.model_config = SimpleNamespace(
            qtype="qfloat8",
            accuracy_recovery_adapter=None,
        )

    @staticmethod
    def get_transformer_block_names():
        return ["blocks"]

    @staticmethod
    def get_quantization_exclude_modules():
        return []

    @staticmethod
    def print_and_status_update(_message):
        return None


def test_quantize_model_can_populate_canonical_blocks_without_second_copy():
    transformer = _Transformer()
    transformer.requires_grad_(False)
    build = prepare_canonical_storage(
        transformer,
        block_names=("blocks",),
        defer_blocks=True,
    )

    try:
        returned = quantize_model(
            _BaseModel(),
            transformer,
            canonical_build=build,
        )

        assert returned is build
        assert transformer.blocks[0].linear.weight.device.type == "meta"
        assert transformer.blocks[1].linear.weight.device.type == "meta"
        assert transformer.head.weight.device.type == "cpu"
        assert build.storage_views("blocks.0")
        assert build.storage_views("blocks.1")
    finally:
        build.rollback()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_quantize_model_populates_host_canonical_storage():
    transformer = _Transformer()
    transformer.requires_grad_(False)
    build = prepare_canonical_storage(
        transformer,
        block_names=("blocks",),
        device="cuda",
        defer_blocks=True,
    )

    try:
        quantize_model(
            _BaseModel("cuda", torch.bfloat16),
            transformer,
            canonical_build=build,
        )
        torch.cuda.synchronize()

        assert transformer.blocks[0].linear.weight.device.type == "meta"
        assert transformer.blocks[1].linear.weight.device.type == "meta"
        for block_key in ("blocks.0", "blocks.1"):
            for storage in build.storage_views(block_key):
                assert all(tensor.device.type == "cpu" for tensor in storage.tensors)
    finally:
        build.rollback()
