from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file

from extensions_built_in.diffusion_models.krea2.krea2 import (
    _arena_destination_key,
    _stream_and_quantize_checkpoint,
    _stream_checkpoint,
    _try_load_quantized_transformer_cache,
)
from toolkit.memory_management.arena_offload import prepare_canonical_storage
from toolkit.memory_management.canonical_arena import CanonicalArena
from toolkit.memory_management import MemoryManager


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.wq = torch.nn.Linear(4, 4)


class _Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block()])
        self.head = torch.nn.Linear(4, 4)


class _TinyAdapter:
    architecture_key = "test_quant_blocks"

    def validate_transformer(self, transformer):
        if not getattr(transformer, "blocks", None):
            raise TypeError("expected blocks")

    def execution_blocks(self, transformer):
        return tuple(transformer.blocks)

    def block_key(self, transformer, index):
        return f"blocks.{index}"

    def leaf_entries(self, block):
        return (("linear", block.linear),)

    def collect_execution_adapters(self, _transformer, _network):
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


class _QuantBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)


class _QuantTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_QuantBlock()])


class _QuantBaseModel:
    device_torch = torch.device("cpu")

    class model_config:
        qtype = "qfloat8"

    def print_and_status_update(self, _message):
        pass


def test_krea_memory_integration_stays_on_public_arena_surface():
    source = (
        Path(__file__).parents[1]
        / "extensions_built_in"
        / "diffusion_models"
        / "krea2"
        / "krea2.py"
    ).read_text(encoding="utf-8")
    for private_name in (
        "arena_offload.layout",
        "arena_offload.construction",
        "CanonicalArena",
        "ResidencyState",
        "ResidencyPlan",
        "prepare_immutable_runtime",
        "_mm_",
    ):
        assert private_name not in source


def test_arena_destination_key_maps_float_and_quantized_cache_leaves():
    assert _arena_destination_key("blocks.3.attn.wq.weight") == (
        "blocks.3", "attn.wq", "weight"
    )
    assert _arena_destination_key("blocks.3.attn.wq.weight._data") == (
        "blocks.3", "attn.wq", "qdata"
    )
    assert _arena_destination_key("blocks.3.attn.wq.weight._scale") == (
        "blocks.3", "attn.wq", "scale"
    )
    assert _arena_destination_key("head.weight") is None


def test_legacy_sampling_context_runs_installed_guard():
    model = torch.nn.Linear(2, 2)
    calls = []
    model._mm_sampling_guard = lambda: calls.append("guard")

    with MemoryManager.sampling_image(model):
        calls.append("body")

    assert calls == ["guard", "body"]


def test_ranged_loader_populates_final_arena_storage_directly(tmp_path):
    source = _Transformer()
    expected = {
        key: torch.full_like(value, index + 1)
        for index, (key, value) in enumerate(source.state_dict().items())
    }
    checkpoint = tmp_path / "tiny.safetensors"
    save_file(expected, str(checkpoint))

    with torch.device("meta"):
        transformer = _Transformer()
    transformer.requires_grad_(False)
    canonical = transformer.blocks[0].attn.wq
    original_weight = canonical.weight
    arena = CanonicalArena()
    build = arena.prepare(
        {"blocks.0": [("attn.wq", canonical)]}, model=transformer
    )
    destination = build.destinations[("blocks.0", "attn.wq", "weight")]

    _stream_checkpoint(
        transformer,
        str(checkpoint),
        torch.float32,
        canonical_build=build,
    )

    assert canonical.weight is original_weight
    assert canonical.weight.device.type == "meta"
    torch.testing.assert_close(destination, expected["blocks.0.attn.wq.weight"])
    torch.testing.assert_close(transformer.head.weight, expected["head.weight"])

    build.commit()
    try:
        assert canonical.weight.data_ptr() == destination.data_ptr()
        torch.testing.assert_close(
            canonical.weight, expected["blocks.0.attn.wq.weight"]
        )
    finally:
        CanonicalArena.unguard_whole_model_to(transformer)
        arena.release()


def test_quantized_ranged_loader_releases_each_source_before_commit(tmp_path):
    transformer = _QuantTransformer()
    expected_weight = transformer.blocks[0].linear.weight.detach().clone()
    expected_bias = transformer.blocks[0].linear.bias.detach().clone()
    checkpoint = tmp_path / "tiny_quant.safetensors"
    save_file(transformer.state_dict(), str(checkpoint))
    transformer.requires_grad_(False)
    adapter = _TinyAdapter()
    build = prepare_canonical_storage(transformer, adapter, defer_blocks=True)

    _stream_and_quantize_checkpoint(
        _QuantBaseModel(),
        transformer,
        str(checkpoint),
        torch.float32,
        canonical_build=build,
        adapter=adapter,
    )

    linear = transformer.blocks[0].linear
    assert linear.weight.device.type == "meta"
    build.commit()
    try:
        assert linear.weight.device.type == "cpu"
        torch.testing.assert_close(
            linear.weight.dequantize(), expected_weight, rtol=0.15, atol=0.02
        )
        torch.testing.assert_close(linear.bias, expected_bias)
    finally:
        CanonicalArena.unguard_whole_model_to(transformer)
        build.arena.release()

import pytest

pytestmark = pytest.mark.process_isolated


def test_quantized_cache_load_reconstructs_quanto_wrappers_on_meta_model(tmp_path):
    from optimum.quanto import freeze
    from toolkit.util.quantize import get_qtype, quantize

    source = _QuantTransformer()
    quantize(source, weights=get_qtype("qfloat8"))
    freeze(source)
    source.requires_grad_(False)
    expected = source.blocks[0].linear.weight.dequantize().clone()
    metadata = {"schema": "test"}
    cache = tmp_path / "tiny_quantized.pt"
    torch.save(
        {"metadata": metadata, "state_dict": source.state_dict()}, str(cache)
    )
    with torch.device("meta"):
        transformer = _QuantTransformer()

    loaded, _build = _try_load_quantized_transformer_cache(
        _QuantBaseModel(), transformer, cache, metadata
    )

    assert loaded
    assert transformer.blocks[0].linear.weight.device.type == "cpu"
    torch.testing.assert_close(
        transformer.blocks[0].linear.weight.dequantize(), expected
    )


def test_quantized_cache_populates_final_arena_without_model_sized_assignment(tmp_path):
    from optimum.quanto import freeze
    from toolkit.util.quantize import get_qtype, quantize

    source = _QuantTransformer()
    quantize(source, weights=get_qtype("qfloat8"))
    freeze(source)
    source.requires_grad_(False)
    expected = source.blocks[0].linear.weight.dequantize().clone()
    metadata = {"schema": "test-direct-arena"}
    cache = tmp_path / "tiny_quantized_arena.pt"
    torch.save(
        {"metadata": metadata, "state_dict": source.state_dict()}, str(cache)
    )
    with torch.device("meta"):
        transformer = _QuantTransformer()

    with mock.patch(
        "extensions_built_in.diffusion_models.krea2.krea2."
        "assign_quantized_state_dict",
        side_effect=AssertionError("full-model cache assignment is forbidden"),
    ):
        loaded, build = _try_load_quantized_transformer_cache(
            _QuantBaseModel(),
            transformer,
            cache,
            metadata,
            canonical_adapter=_TinyAdapter(),
            canonical_device="cpu",
        )

    assert loaded
    assert build is not None
    assert transformer.blocks[0].linear.weight.device.type == "meta"
    build.commit()
    try:
        assert transformer.blocks[0].linear.weight.device.type == "cpu"
        torch.testing.assert_close(
            transformer.blocks[0].linear.weight.dequantize(), expected
        )
    finally:
        CanonicalArena.unguard_whole_model_to(transformer)
        build.arena.release()


def test_quantized_cache_rejects_incomplete_canonical_payload(tmp_path):
    from optimum.quanto import freeze
    from toolkit.util.quantize import get_qtype, quantize

    source = _QuantTransformer()
    quantize(source, weights=get_qtype("qfloat8"))
    freeze(source)
    state_dict = source.state_dict()
    del state_dict["blocks.0.linear.bias"]
    metadata = {"schema": "test-incomplete-arena"}
    cache = tmp_path / "incomplete_quantized_arena.pt"
    torch.save({"metadata": metadata, "state_dict": state_dict}, str(cache))
    with torch.device("meta"):
        transformer = _QuantTransformer()

    loaded, build = _try_load_quantized_transformer_cache(
        _QuantBaseModel(),
        transformer,
        cache,
        metadata,
        canonical_adapter=_TinyAdapter(),
        canonical_device="cpu",
    )

    assert not loaded
    assert build is None
    assert transformer.blocks[0].linear.weight.device.type == "meta"
