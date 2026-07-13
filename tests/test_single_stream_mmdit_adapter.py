from types import SimpleNamespace

import pytest
import torch

from extensions_built_in.diffusion_models.krea2.src.mmdit import SingleStreamDiT
from toolkit.memory_management.adapters import SingleStreamMMDiTAdapter


class _Block:
    def __init__(self):
        self.attn = SimpleNamespace(
            wq=object(),
            wk=object(),
            wv=object(),
            gate=object(),
            wo=object(),
        )
        self.mlp = SimpleNamespace(
            gate=object(),
            up=object(),
            down=object(),
        )
        self.calls = []

    def forward_streamed(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return args[0]


def test_adapter_exposes_canonical_single_stream_structure():
    adapter = SingleStreamMMDiTAdapter()
    block = _Block()
    transformer = SimpleNamespace(blocks=[block])

    assert adapter.execution_blocks(transformer) == (block,)
    assert adapter.block_key(transformer, 0) == "blocks.0"
    assert tuple(name for name, _module in adapter.leaf_entries(block)) == (
        "attn.wq",
        "attn.wk",
        "attn.wv",
        "attn.gate",
        "attn.wo",
        "mlp.gate",
        "mlp.up",
        "mlp.down",
    )


def test_adapter_rejects_reference_calls():
    adapter = SingleStreamMMDiTAdapter()
    ordinary = (torch.empty(1), torch.empty(1), None)
    reference = ((torch.empty(1), torch.empty(1), 1), torch.empty(1), None)

    assert adapter.can_run_current_call(ordinary)
    assert not adapter.can_run_current_call(reference)
    assert not adapter.can_run_current_call(
        ordinary,
        ref_kv_capture=[],
    )
    assert not adapter.can_run_current_call(
        ordinary,
        blockcaches=[],
    )


class _Runtime:
    def __init__(self):
        self.calls = []

    def can_run_blocks(
        self,
        block_args,
        *,
        ref_kv_capture=None,
        blockcaches=None,
    ):
        tvec, _freqs, _mask = block_args
        return not isinstance(tvec, tuple) and ref_kv_capture is None and blockcaches is None

    def run_blocks(self, combined, block_args):
        self.calls.append((combined, block_args))
        return combined + 1


class _EagerBlock:
    def __init__(self):
        self.calls = []

    def __call__(self, combined, tvec, freqs, mask, **kwargs):
        self.calls.append((combined, tvec, freqs, mask, kwargs))
        return combined + 2


def _dispatch_fixture():
    runtime = _Runtime()
    block = _EagerBlock()
    transformer = SimpleNamespace(
        _arena_offload_runtime=runtime,
        _checkpoint_keep_last=0,
        gradient_checkpointing=False,
        blocks=[block],
        _refresh_runtime_lora_multiplier=lambda: None,
    )
    return transformer, runtime, block


def test_normal_block_dispatch_prefers_generic_runtime():
    transformer, runtime, block = _dispatch_fixture()
    hidden = torch.zeros(1)

    result = SingleStreamDiT._blocks_trunk(
        transformer,
        hidden,
        torch.zeros(1),
        torch.zeros(1),
        None,
    )

    assert torch.equal(result, torch.ones(1))
    assert len(runtime.calls) == 1
    assert block.calls == []


def test_reference_block_dispatch_uses_pure_eager_math():
    transformer, runtime, block = _dispatch_fixture()
    hidden = torch.zeros(1)
    tvec = (torch.zeros(1), torch.zeros(1), 1)

    result = SingleStreamDiT._blocks_trunk(
        transformer,
        hidden,
        tvec,
        torch.zeros(1),
        None,
        blockcaches=[None],
    )

    assert torch.equal(result, torch.full((1,), 2.0))
    assert runtime.calls == []
    assert len(block.calls) == 1


def test_adapter_owns_lora_order_and_functional_block_call():
    adapter = SingleStreamMMDiTAdapter()
    block = _Block()
    entry = SimpleNamespace(
        a=torch.tensor(1.0),
        b=torch.tensor(2.0),
        scale=torch.tensor(3.0),
    )
    lora_args = adapter.build_adapter_args(
        0,
        {0: {"attn.wq": entry}},
        torch.tensor(2.0),
    )

    assert len(lora_args) == 8
    assert lora_args[0][0] is entry.a
    assert lora_args[0][1] is entry.b
    assert torch.equal(lora_args[0][2], torch.tensor(6.0))
    assert lora_args[1:] == (None,) * 7

    hidden = torch.randn(1, 2, 3)
    result = adapter.forward_block(
        block,
        hidden,
        (torch.empty(1), torch.empty(1), None),
        ((torch.empty(1), None, None),) * 8,
        (False,) * 8,
        lora_args,
        training=True,
    )
    assert result is hidden
    assert block.calls[0][1]["training"] is True
    assert block.calls[0][1]["loras"] is lora_args


def test_adapter_finalization_uses_explicit_network_entries():
    adapter = SingleStreamMMDiTAdapter()
    block = _Block()
    transformer = SimpleNamespace(blocks=[block])
    network = SimpleNamespace(is_lorm=False)
    owner_type = type("LoRAModule", (), {})
    owner = owner_type()
    owner.functional_forward = lambda *args: None
    owner.orig_module_ref = lambda: block.attn.wq
    owner.network_ref = lambda: network
    network.unet_loras = [owner]

    entries = adapter.collect_execution_adapters(transformer, network)

    assert entries == {0: {"attn.wq": owner}}


def test_adapter_rejects_multiple_network_entries_for_one_linear():
    adapter = SingleStreamMMDiTAdapter()
    block = _Block()
    transformer = SimpleNamespace(blocks=[block])
    network = SimpleNamespace(is_lorm=False)
    owner_type = type("LoRAModule", (), {})
    owners = []
    for _ in range(2):
        owner = owner_type()
        owner.functional_forward = lambda *args: None
        owner.orig_module_ref = lambda: block.attn.wq
        owner.network_ref = lambda: network
        owners.append(owner)
    network.unet_loras = owners

    with pytest.raises(RuntimeError, match="one adapter per canonical Linear"):
        adapter.collect_execution_adapters(transformer, network)
