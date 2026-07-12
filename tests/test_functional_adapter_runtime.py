from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from extensions_built_in.diffusion_models.krea2.src.mmdit import (
    SingleStreamDiT,
    _streamed_arg_linear_train,
)
from toolkit.functional_adapter import FunctionalLinear
from toolkit.lora_special import FullModule, LoRAModule
from toolkit.models.DoRA import DoRAModule
from toolkit.models.lokr import LokrModule


class _Network:
    network_type = "lora"
    is_active = True
    is_merged_in = False
    is_lorm = False
    _multiplier = 1.0
    torch_multiplier = torch.tensor([0.75])
    vector_gates = None
    is_assistant_adapter = False
    base_model_ref = None


def _functional_linear(linear):
    def call_fn(x, weight, bias, scale):
        assert scale is None
        return F.linear(x, weight, bias)

    return FunctionalLinear(
        weight=linear.weight,
        bias=linear.bias,
        scale=None,
        call_fn=call_fn,
        materialize_fn=lambda weight, scale: weight,
    )


def _adapter_parameters(adapter):
    return tuple(parameter for parameter in adapter.parameters() if parameter.requires_grad)


@pytest.mark.parametrize(
    "adapter_type,kwargs",
    [
        (LoRAModule, {"lora_dim": 2, "alpha": 2}),
        (LokrModule, {"lora_dim": 2, "alpha": 2}),
        (DoRAModule, {"lora_dim": 2, "alpha": 2}),
        (FullModule, {}),
    ],
)
def test_functional_adapter_matches_ordinary_eager_output_and_grads(adapter_type, kwargs):
    torch.manual_seed(17)
    network = _Network()
    linear = torch.nn.Linear(4, 4, bias=True)
    linear.weight.requires_grad_(False)
    linear.bias.requires_grad_(False)
    adapter = adapter_type("test", linear, network=network, **kwargs)
    for parameter in _adapter_parameters(adapter):
        if parameter.numel():
            torch.nn.init.normal_(parameter, std=0.2)
    adapter.apply_to()

    x_eager = torch.randn(2, 3, 4, requires_grad=True)
    eager = adapter(x_eager)
    eager.sum().backward()
    eager_input_grad = x_eager.grad.detach().clone()
    eager_grads = tuple(parameter.grad.detach().clone() for parameter in _adapter_parameters(adapter))

    for parameter in _adapter_parameters(adapter):
        parameter.grad = None
    x_functional = x_eager.detach().clone().requires_grad_(True)
    functional = adapter.functional_forward(_functional_linear(linear), x_functional)
    functional.sum().backward()

    torch.testing.assert_close(functional, eager)
    torch.testing.assert_close(x_functional.grad, eager_input_grad)
    for parameter, expected in zip(_adapter_parameters(adapter), eager_grads):
        torch.testing.assert_close(parameter.grad, expected)
    assert linear.weight.grad is None
    assert linear.bias.grad is None


def test_unsupported_forward_owner_reports_class_and_full_target():
    class UnsupportedAdapter:
        def org_forward(self, x):
            return x

        def forward(self, x):
            return x

    child = torch.nn.Linear(2, 2)
    owner = UnsupportedAdapter()
    child.forward = owner.forward

    with pytest.raises(RuntimeError) as exc:
        SingleStreamDiT._collect_adapter_entry(child, "blocks.3.attn.wq")

    message = str(exc.value)
    assert "arena offload supports" in message
    assert "UnsupportedAdapter" in message
    assert "blocks.3.attn.wq" in message


def test_supported_lokr_owner_is_collected_at_setup():
    network = _Network()
    child = torch.nn.Linear(4, 4)
    adapter = LokrModule("test", child, network=network, lora_dim=2, alpha=2)
    adapter.apply_to()

    assert SingleStreamDiT._collect_adapter_entry(child, "blocks.0.attn.wq") is adapter


def test_lokr_functional_leaf_runs_under_block_compile():
    network = _Network()
    child = torch.nn.Linear(4, 4)
    adapter = LokrModule("test", child, network=network, lora_dim=2, alpha=2)
    adapter.apply_to()
    arg = (child.weight.detach(), child.bias.detach(), None)

    def leaf(x):
        return _streamed_arg_linear_train(x, arg, False, adapter)

    x = torch.randn(2, 3, 4, requires_grad=True)
    expected = leaf(x)
    compiled = torch.compile(leaf, backend="eager", fullgraph=True)
    actual = compiled(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert x.grad is not None
    assert all(parameter.grad is not None for parameter in _adapter_parameters(adapter))
